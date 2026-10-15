//! Immutable continuations retain failed graders and reuse original generations.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ImportedAttempt {
    pub source_policy: Artefact,
    pub student_plan: Artefact,
    pub student_raw: Artefact,
    pub student_grade: Artefact,
    pub generation_engine_sha256: String,
}

pub(super) fn verify_import(
    root: &Path,
    plan: &RolloutPlan,
    import: &ImportedAttempt,
) -> Result<RolloutPlan> {
    let policy: serde_json::Value =
        serde_json::from_slice(&fs::read(import.source_policy.verify(root)?)?)?;
    let old: RolloutPlan = serde_json::from_slice(&fs::read(import.student_plan.verify(root)?)?)?;
    ensure!(
        old.imported_student_attempt.is_none(),
        "import the original student batch, not a chain of reserialised continuations"
    );
    ensure!(
        old.schema_version == 1
            && json_equal(&old.task, &plan.task)?
            && json_equal(&old.student_request, &plan.student_request)?
            && old.student_base == plan.student_base
            && old.student_kind == plan.student_kind
            && old.student_identity == plan.student_identity
            && json_equal(&old.student_assets, &plan.student_assets)?
            && json_equal(&old.student_checkpoint, &plan.student_checkpoint)?
            && old.student_runtime == plan.student_runtime
            && old.input_bundle.sha256 == plan.input_bundle.sha256
            && old.engine_sha256 == import.generation_engine_sha256
            && policy["engine_sha256"] == old.engine_sha256
            && policy["student_assets"] == serde_json::to_value(&old.student_assets)?
            && policy["input"] == serde_json::to_value(&old.input_bundle)?
            && policy["student_checkpoint"] == serde_json::to_value(&old.student_checkpoint)?
            && policy["student_runtime"] == serde_json::to_value(&old.student_runtime)?
            && policy["max_tokens"] == old.student_request.max_new_tokens
            && policy["seed"] == old.student_request.seed,
        "imported student checkpoint/request/task/seed/runtime differs from the new continuation"
    );
    let raw: Raw = serde_json::from_slice(&fs::read(import.student_raw.verify(root)?)?)?;
    ensure!(
        json_equal(&raw.plan, &import.student_plan)?,
        "imported raw belongs to a different generation engine/plan"
    );
    verify::verify_raw(
        root,
        &import.student_raw,
        &import.student_plan,
        &old.student_identity,
        &old.student_runtime,
        old.student_request.max_new_tokens,
    )?;
    import.student_grade.verify(root)?;
    let preflight: sandbox::PreflightReport =
        serde_json::from_slice(&fs::read(old.preflight.verify(root)?)?)?;
    ensure!(
        preflight.probes_passed
            && preflight.image_id == old.sandbox.expected_image_id
            && preflight.policy_sha256 == sandbox::policy_sha256(&old.sandbox)?,
        "imported student used different or unenforced container controls"
    );
    Ok(old)
}

pub(super) fn import(root: &Path, source: &Path, plan: &RolloutPlan) -> Result<ImportedAttempt> {
    relative(source)?;
    let directory = source.join(&plan.task.id);
    let old_plan = capture(root, &directory.join("student-plan.json"))?;
    let old: RolloutPlan = serde_json::from_slice(&fs::read(old_plan.verify(root)?)?)?;
    let import = ImportedAttempt {
        source_policy: capture(root, &source.join("rollout-policy.json"))?,
        student_plan: old_plan,
        student_raw: capture(root, &directory.join("student-raw.json"))?,
        student_grade: capture(root, &directory.join("student-grade.json"))?,
        generation_engine_sha256: old.engine_sha256,
    };
    verify_import(root, plan, &import)?;
    Ok(import)
}

pub(super) fn classify_grade(
    mut observed: Grade,
    task: &Task,
    raw: &Raw,
    config: &SandboxConfig,
    root: &Path,
) -> Result<Grade> {
    ensure!(
        observed.report.is_none() || observed.unverified_report.is_none(),
        "a grade cannot be both verified and unverified"
    );
    if let Some(report) = observed.report.take() {
        let r: SandboxReport = serde_json::from_slice(&fs::read(report.verify(root)?)?)?;
        let checked = raw
            .generation
            .as_ref()
            .context("grader report has no actual generation")
            .and_then(|g| {
                sandbox::verify_report(
                    &r,
                    config,
                    task,
                    &crate::evaluate::extract_code(&g.completion),
                )
            });
        match checked {
            Ok(()) => observed.report = Some(report),
            Err(error) => {
                observed.error = Some(format!(
                    "Returned grading report is unverifiable and remains a failed observation, never trusted feedback: {error:#}"
                ));
                observed.unverified_report = Some(report);
            }
        }
    }
    if let Some(report) = &observed.unverified_report {
        let r: SandboxReport = serde_json::from_slice(&fs::read(report.verify(root)?)?)?;
        let g = raw
            .generation
            .as_ref()
            .context("unverified grader report has no original generation")?;
        ensure!(
            sandbox::verify_report(
                &r,
                config,
                task,
                &crate::evaluate::extract_code(&g.completion)
            )
            .is_err()
                && observed.error.as_ref().is_some_and(|e| !e.is_empty()),
            "unverified grader classification differs from its actual report/error"
        );
    }
    Ok(observed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_audit_stays_a_traceable_failure_and_never_trusted_feedback() {
        let root = std::env::current_dir().unwrap();
        fs::create_dir_all("state").unwrap();
        let directory = tempfile::tempdir_in(root.join("state")).unwrap();
        let relative = directory.path().strip_prefix(&root).unwrap();
        let task = train_tasks().remove(0);
        let config = SandboxConfig {
            expected_image_id: "synthetic-unit-image".into(),
            report_dir: relative.to_owned(),
            ..Default::default()
        };
        let text =
            "pub fn solve(_: serde_json::Value) -> serde_json::Value { serde_json::Value::Null }";
        let report = SandboxReport {
            run_id: "synthetic-unit-failed-deadline".into(),
            task_id: task.id.clone(),
            passed: false,
            quarantined: true,
            quarantine_reasons: vec!["synthetic unit deadline failure".into()],
            cases_total: task.public_examples.len() + tasks::private_cases(&task).unwrap().len(),
            cases_passed: 0,
            trace_complete: false,
            policy_sha256: sandbox::policy_sha256(&config).unwrap(),
            image_id: config.expected_image_id.clone(),
            oracle_sha256: task.oracle_sha256.clone(),
            code_sha256: tasks::hash(text),
            stdout_sha256: tasks::hash(b""),
            sandbox_config: config.clone(),
            audit_events: vec![],
            audit_path: relative.join("absent.audit.json"),
            audit_sha256: String::new(),
            failure: Some("synthetic unit fixture; no candidate executed".into()),
            duration_ms: 1,
            timeout_evidence: None,
        };
        let report_path = relative.join("failed-report.json");
        artefact::write_json(&report_path, &report).unwrap();
        let report_artifact = capture(&root, &report_path).unwrap();
        let raw = Raw {
            schema_version: 1,
            evidence_kind: "synthetic_unit_fixture_not_training_evidence".into(),
            plan: report_artifact.clone(),
            generation: Some(Generation {
                completion: text.into(),
                checkpoint_id: "synthetic-unit-only".into(),
                finish_reason: "length".into(),
                input_tokens: 1,
                output_tokens: 1,
                elapsed_ms: 1,
                complete: false,
            }),
            generation_error: None,
            execution: None,
        };
        let result = classify_grade(
            Grade {
                report: Some(report_artifact.clone()),
                unverified_report: None,
                error: None,
            },
            &task,
            &raw,
            &config,
            &root,
        )
        .unwrap();
        assert!(result.report.is_none());
        assert_eq!(
            result.unverified_report.as_ref().unwrap().sha256,
            report_artifact.sha256
        );
        assert!(
            result
                .error
                .as_deref()
                .unwrap()
                .contains("failed observation")
        );
        report_artifact.verify(&root).unwrap();
        let again = classify_grade(result.clone(), &task, &raw, &config, &root).unwrap();
        assert!(json_equal(&again, &result).unwrap());
        assert!(
            classify_grade(
                Grade {
                    report: None,
                    unverified_report: Some(report_artifact),
                    error: None
                },
                &task,
                &raw,
                &config,
                &root
            )
            .is_err()
        );
    }

    #[test]
    fn continuation_rejects_changed_student_seed_or_identity_before_reuse() {
        let root = std::env::current_dir().unwrap();
        fs::create_dir_all("state").unwrap();
        let directory = tempfile::tempdir_in(root.join("state")).unwrap();
        let relative = directory.path().strip_prefix(&root).unwrap();
        let placeholder = relative.join("synthetic-unit-only.json");
        artefact::write_json(&placeholder, &serde_json::json!({"fixture":true})).unwrap();
        let placeholder = capture(&root, &placeholder).unwrap();
        let task = train_tasks().remove(0);
        let mut plan = RolloutPlan {
            schema_version: 1,
            mode: RolloutMode::OnPolicy,
            task: task.clone(),
            input_bundle: placeholder.clone(),
            student_base: "synthetic-unit-base-never-loaded".into(),
            student_kind: DistillationStudent::ConventionalQwenLora,
            student_assets: placeholder.clone(),
            student_checkpoint: None,
            student_identity: "a".repeat(64),
            student_runtime: RuntimeOptions::default(),
            student_request: GenerationRequest {
                prompt: task.prompt(),
                max_new_tokens: 16,
                seed: 7,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: 120,
            },
            sandbox: SandboxConfig::default(),
            preflight: placeholder.clone(),
            engine_sha256: "b".repeat(64),
            imported_student_attempt: None,
            content_role_policy: None,
        };
        let plan_path = relative.join("student-plan.json");
        artefact::write_json(&plan_path, &plan).unwrap();
        let policy = serde_json::json!({"engine_sha256":plan.engine_sha256,"student_assets":plan.student_assets,"input":plan.input_bundle,"student_checkpoint":plan.student_checkpoint,"student_runtime":plan.student_runtime,"max_tokens":16,"seed":7});
        let policy_path = relative.join("policy.json");
        artefact::write_json(&policy_path, &policy).unwrap();
        let import = ImportedAttempt {
            source_policy: capture(&root, &policy_path).unwrap(),
            student_plan: capture(&root, &plan_path).unwrap(),
            student_raw: placeholder.clone(),
            student_grade: placeholder,
            generation_engine_sha256: plan.engine_sha256.clone(),
        };
        plan.student_request.seed = 8;
        assert!(
            verify_import(&root, &plan, &import)
                .unwrap_err()
                .to_string()
                .contains("checkpoint/request/task/seed")
        );
        plan.student_request.seed = 7;
        plan.student_identity = "c".repeat(64);
        assert!(
            verify_import(&root, &plan, &import)
                .unwrap_err()
                .to_string()
                .contains("checkpoint/request/task/seed")
        );
    }
}
