//! Independently reconstruct derived rows from original bytes and owned grades.
use super::*;

pub(super) fn identity(protocol: &GradingProtocol) -> GradingIdentity {
    GradingIdentity {
        kind: PLAN_KIND.into(),
        engine_sha256: protocol.engine.sha256.clone(),
        controller_version: protocol.sandbox.controller_version,
        policy_sha256: protocol.policy_sha256.clone(),
        image_id: protocol.image_id.clone(),
        extractor_sha256: Some(protocol.extractor_sha256.clone()),
    }
}

fn grading(root: &Path, protocol: &GradingProtocol) -> Result<()> {
    protocol.engine.verify(root)?;
    ensure!(
        protocol.sandbox.controller_version == 3
            && protocol.sandbox.expected_image_id == protocol.image_id
            && sandbox::policy_sha256(&protocol.sandbox)? == protocol.policy_sha256
            && protocol.extractor_sha256 == extractor_hash(),
        "new grading engine/policy/image/extractor binding differs"
    );
    let probe: serde_json::Value = read(&protocol.preflight.verify(root)?)?;
    ensure!(
        probe["probes_passed"] == true
            && probe["image_id"] == protocol.image_id
            && probe["policy_sha256"] == protocol.policy_sha256,
        "derived grading lacks its actual sandbox preflight"
    );
    Ok(())
}

pub(super) fn prepared(
    root: &Path,
    artifact: &Artefact,
    plan: &RegradePlan,
    plan_artifact: &Artefact,
    task: &Task,
    code: &str,
    raw: &Artefact,
) -> Result<sandbox::PreparedGrade> {
    let prepared: sandbox::PreparedGrade = read(&artifact.verify(root)?)?;
    sandbox::verify_prepared(&prepared)?;
    let mut expected = plan.grading.sandbox.clone();
    expected.report_dir = plan_artifact
        .path
        .parent()
        .context("derived output directory absent")?
        .join("sandbox");
    let attempt_key =
        artefact::hash(format!("{}:{}:{}", plan_artifact.sha256, task.id, raw.sha256).as_bytes());
    ensure!(
        same(&prepared.task, task)?
            && prepared.code == code
            && same(&prepared.config, &expected)?
            && prepared.attempt_key == attempt_key
            && root_relative(root, &prepared.receipt_path)? == artifact.path,
        "prepared grade task/code/policy/attempt binding differs"
    );
    Ok(prepared)
}

pub(super) fn passed(task: &Task, report: &SandboxReport) -> Result<bool> {
    let required = task.public_examples.len() + tasks::private_cases(task)?.len();
    Ok(report.passed
        && !report.quarantined
        && report.trace_complete
        && report.cases_total == required
        && report.cases_passed == required)
}

pub(super) fn observation(
    root: &Path,
    plan: &RegradePlan,
    plan_artifact: &Artefact,
    task: &Task,
    imported: &ImportedGeneration,
    row: &RegradedObservation,
) -> Result<RawGeneration> {
    ensure!(
        row.schema_version == 1
            && row.regrade_plan_sha256 == plan_artifact.sha256
            && row.task_id == task.id
            && row.source_id == task.source_id
            && same(&row.original_plan, &plan.original.plan)?
            && same(&row.original_raw, &imported.original_raw)?,
        "derived row changed original pointers or task identity"
    );
    let raw = original_raw(root, &plan.original, imported, task)?;
    match &row.outcome {
        Outcome::OriginalGenerationFailure => ensure!(
            !usable(&raw) && !row.success,
            "an original missing/empty/truncated/errored generation became a grade"
        ),
        Outcome::InfrastructureFailure {
            prepared: receipt,
            error,
        } => {
            ensure!(
                usable(&raw) && !row.success && !error.trim().is_empty(),
                "invalid infrastructure failure row"
            );
            if let Some(receipt) = receipt {
                let code = crate::evaluate::extract_code(
                    &raw.generation
                        .as_ref()
                        .context("checked generation missing")?
                        .completion,
                );
                prepared(
                    root,
                    receipt,
                    plan,
                    plan_artifact,
                    task,
                    &code,
                    &row.original_raw,
                )?;
            }
        }
        Outcome::Graded {
            prepared: receipt,
            report,
            audit,
        } => {
            ensure!(usable(&raw), "partial original output was graded");
            let code = crate::evaluate::extract_code(
                &raw.generation
                    .as_ref()
                    .context("checked generation missing")?
                    .completion,
            );
            let prepared = prepared(
                root,
                receipt,
                plan,
                plan_artifact,
                task,
                &code,
                &row.original_raw,
            )?;
            let observed: SandboxReport = read(&report.verify(root)?)?;
            ensure!(
                root_relative(root, &prepared.report_path)? == report.path
                    && root_relative(root, &observed.audit_path)? == audit.path
                    && audit.sha256 == observed.audit_sha256,
                "derived grade/audit points to another owned run"
            );
            audit.verify(root)?;
            // The strict V3 verifier reopens checksum-bound lifecycle, drains,
            // exact ownership/terminal evidence and the candidate's real audit.
            sandbox::verify_prepared_report(&prepared, &observed)?;
            ensure!(
                observed.policy_sha256 == plan.grading.policy_sha256
                    && observed.image_id == plan.grading.image_id
                    && row.success == passed(task, &observed)?,
                "derived verdict differs from complete trusted evidence"
            );
        }
    }
    Ok(raw)
}

pub(super) fn one(output: &Path) -> Result<VerifiedEvaluation> {
    let root = std::env::current_dir()?.canonicalize()?;
    let directory = root_relative(&root, output)?;
    let plan_artifact = capture(&root, &directory.join("plan.json"))?;
    let plan: RegradePlan = read(&plan_artifact.verify(&root)?)?;
    ensure!(
        plan.schema_version == 1
            && plan.evidence_kind == PLAN_KIND
            && !plan.production_eligible
            && (1..=256).contains(&plan.maximum_concurrent_graders),
        "unsupported derived grading plan"
    );
    let batch: BatchPlan = read(&plan.batch_plan.verify(&root)?)?;
    validate_cohort(&batch.originals, batch.expected_generations)?;
    ensure!(
        batch.schema_version == 1
            && batch.evidence_kind == BATCH_KIND
            && !batch.production_eligible
            && batch.expected_generations
                == batch.originals.iter().map(|o| o.rows.len()).sum::<usize>()
            && same(&plan.grading, &batch.grading)?
            && plan.maximum_concurrent_graders == batch.maximum_concurrent_graders,
        "derived plan differs from its full frozen batch"
    );
    let index = batch
        .originals
        .iter()
        .position(|o| same(o, &plan.original).unwrap_or(false))
        .context("derived original is absent from frozen batch")?;
    let expected = plan
        .batch_plan
        .path
        .parent()
        .context("batch directory absent")?
        .join(child_name(index, &plan.original));
    ensure!(
        directory == expected,
        "derived plan is outside its frozen child directory"
    );
    grading(&root, &plan.grading)?;
    import::verify(&root, &plan.original)?;
    let mut rows = Vec::new();
    let mut raws = Vec::new();
    for (task, imported) in plan
        .original
        .generation_plan
        .tasks
        .iter()
        .zip(&plan.original.rows)
    {
        let row: RegradedObservation = read(
            &directory
                .join("observations")
                .join(format!("{}.json", task.id)),
        )?;
        let raw = observation(&root, &plan, &plan_artifact, task, imported, &row)?;
        rows.push(Observation {
            schema_version: 1,
            plan_sha256: plan.original.plan.sha256.clone(),
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            success: row.success,
            raw_generation: row.original_raw,
            sandbox_report: match &row.outcome {
                Outcome::Graded { report, .. } => Some(report.clone()),
                _ => None,
            },
            grading_error: match row.outcome {
                Outcome::InfrastructureFailure { error, .. } => Some(error),
                _ => None,
            },
        });
        raws.push(raw);
    }
    let mut report = evidence::summary(
        &plan.original.generation_plan,
        &rows,
        &raws,
        &plan_artifact.sha256,
    )?;
    report["evidence_kind"] = "derived_local_study_generation_regrade".into();
    report["original_plan"] = serde_json::to_value(&plan.original.plan)?;
    report["grading_identity"] = serde_json::to_value(identity(&plan.grading))?;
    report["generation_reused"] = true.into();
    report["new_model_calls"] = 0.into();
    report["new_inference_tokens"] = 0.into();
    report["batch_plan"] = serde_json::to_value(&plan.batch_plan)?;
    report["limitations"].as_array_mut().context("summary limitations absent")?.push("All inference timings/tokens are reused original measurements. New host grading has a separately bound controller/policy/image and never regenerates an output. Original and replacement verdicts remain distinct; no SHIP decision.".into());
    let execution = if let Some(artifact) = plan
        .original
        .execution_receipts
        .iter()
        .find(|a| a.path == plan.original.directory.join("execution.json"))
    {
        read(&artifact.verify(&root)?)?
    } else {
        serde_json::Value::Null
    };
    let numerical = parity::verify_numerical_equivalence_metadata(&plan.original.directory)?;
    Ok(VerifiedEvaluation {
        generation_plan: plan.original.generation_plan,
        execution,
        grading: identity(&plan.grading),
        report,
        numerical,
    })
}

pub(super) fn compute_batch(
    output: &Path,
) -> Result<(serde_json::Value, Vec<(PathBuf, VerifiedEvaluation)>)> {
    let root = std::env::current_dir()?.canonicalize()?;
    let output = root_relative(&root, output)?;
    let batch_path = output.join("batch-plan.json");
    let batch: BatchPlan = read(&batch_path)?;
    validate_cohort(&batch.originals, batch.expected_generations)?;
    ensure!(
        batch.schema_version == 1
            && batch.evidence_kind == BATCH_KIND
            && !batch.production_eligible
            && !batch.originals.is_empty(),
        "unsupported regrade batch"
    );
    let mut views = Vec::new();
    let mut children = Vec::new();
    let mut planned = 0usize;
    let mut passed = 0usize;
    let mut errors = 0usize;
    for (index, original) in batch.originals.iter().enumerate() {
        let directory = output.join(child_name(index, original));
        let view = one(&directory)?;
        let recorded: serde_json::Value = read(&directory.join("report.json"))?;
        ensure!(
            view.report == recorded,
            "derived report differs from its independently reconstructed evidence"
        );
        planned += original.rows.len();
        passed += view.report["passed"]
            .as_u64()
            .context("derived pass count absent")? as usize;
        errors += view.report["infrastructure_or_generation_errors"]
            .as_u64()
            .context("derived error count absent")? as usize;
        children.push(serde_json::json!({"directory":directory,"report":capture(&root,&directory.join("report.json"))?,"checkpoint_id":view.generation_plan.checkpoint.checkpoint_id,"split":view.generation_plan.split,"planned":view.report["planned"],"passed":view.report["passed"],"healthy_complete":view.report["healthy_complete"]}));
        views.push((directory, view));
    }
    ensure!(
        planned == batch.expected_generations,
        "regrade batch denominator incomplete"
    );
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"completed_immutable_generation_regrade_batch","batch_plan":capture(&root,&batch_path)?,"planned":planned,"observed":planned,"passed":passed,"coverage_complete":true,"healthy_complete":errors==0,"infrastructure_or_generation_errors":errors,"evaluations":children,"new_model_calls":0,"new_inference_tokens":0,"held_out_used":false,"production_eligible":false,"limitation":"Original generation failures remain failures. New complete trusted candidate failures score zero; missing/partial/invalid ownership/audit remains infrastructure failure. This is descriptive regrading under an explicit new host protocol, never SHIP."});
    Ok((report, views))
}

pub fn verify_batch(output: &Path) -> Result<serde_json::Value> {
    let (computed, _) = verified_batch(output)?;
    Ok(computed)
}

pub(in crate::study_evaluate) fn verified_batch(
    output: &Path,
) -> Result<(serde_json::Value, Vec<(PathBuf, VerifiedEvaluation)>)> {
    let (computed, views) = compute_batch(output)?;
    let recorded: serde_json::Value = read(&output.join("batch-report.json"))?;
    ensure!(
        computed == recorded,
        "regrade batch summary differs from all independently verified original/graded rows"
    );
    Ok((computed, views))
}

pub(in crate::study_evaluate) fn verify_view(output: &Path) -> Result<VerifiedEvaluation> {
    let root = std::env::current_dir()?.canonicalize()?;
    let output = root_relative(&root, output)?;
    let plan: RegradePlan = read(&output.join("plan.json"))?;
    let directory = plan
        .batch_plan
        .path
        .parent()
        .context("batch directory absent")?;
    verified_batch(directory)?
        .1
        .into_iter()
        .find(|(path, _)| *path == output)
        .map(|(_, view)| view)
        .context("requested regrade view is not in its completed batch")
}
