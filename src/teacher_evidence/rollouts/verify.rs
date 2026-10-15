//! Reconstruct rollout availability from original processes and trusted grades.
use super::*;

pub(super) fn verify_raw(
    root: &Path,
    artifact: &Artefact,
    plan: &Artefact,
    identity: &str,
    runtime: &RuntimeOptions,
    max_tokens: usize,
) -> Result<Raw> {
    let raw: Raw = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
    ensure!(
        raw.schema_version == 1
            && raw.evidence_kind == "actual_checkpoint_bound_rollout_generation"
            && json_equal(&raw.plan, plan)?,
        "unsupported or unbound student/feedback raw evidence"
    );
    if let Some(g) = &raw.generation {
        ensure!(
            raw.generation_error.is_none()
                && g.checkpoint_id == identity
                && g.input_tokens > 0
                && g.output_tokens <= max_tokens
                && g.complete == (g.finish_reason == "stop"),
            "rollout generation identity, termination or token evidence changed"
        );
        let e = raw
            .execution
            .as_ref()
            .context("rollout generation lacks actual runtime metadata")?;
        ensure!(
            e["checkpoint_id"] == identity
                && e["engine"] == "native_rust_candle"
                && e["cpu_fallback"] == false,
            "rollout runtime identity/fallback mismatch"
        );
        if e.get("requested").is_some() {
            ensure!(
                e["requested"] == serde_json::to_value(runtime)?,
                "rollout executed with another requested runtime"
            );
        } else {
            ensure!(
                runtime.device == model::ExecutionDevice::Cpu
                    && runtime.dtype == model::InferenceDtype::F32
                    && e["actual_device"] == "Cpu"
                    && e["actual_dtype"] == "F32"
                    && e["architecture"] == "bitnet",
                "runtime without a request is not the explicitly supported native BitNet CPU/F32 path"
            );
        }
    } else {
        ensure!(
            raw.generation_error.as_ref().is_some_and(|e| !e.is_empty()),
            "missing rollout without an explicit error"
        );
    }
    Ok(raw)
}

fn verified_grade(
    root: &Path,
    a: &Artefact,
    plan: &RolloutPlan,
    raw: &Raw,
    config: &SandboxConfig,
) -> Result<SandboxReport> {
    let generation = raw
        .generation
        .as_ref()
        .context("graded rollout has no output")?;
    ensure!(
        !generation.completion.trim().is_empty(),
        "empty rollout cannot have a grade"
    );
    let report: SandboxReport = serde_json::from_slice(&fs::read(a.verify(root)?)?)?;
    sandbox::verify_report(
        &report,
        config,
        &plan.task,
        &crate::evaluate::extract_code(&generation.completion),
    )?;
    Ok(report)
}

pub(crate) fn append_verified(
    root: &Path,
    bundle: &mut EvidenceBundle,
    artifact: &Artefact,
) -> Result<()> {
    ensure!(
        !bundle
            .rollout_records
            .iter()
            .any(|a| a.path == artifact.path),
        "duplicate rollout record"
    );
    let record: Record = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
    let plan: RolloutPlan = serde_json::from_slice(&fs::read(record.plan.verify(root)?)?)?;
    ensure!(
        record.schema_version == 1
            && record.evidence_kind == "actual_checkpoint_bound_rollout_observation"
            && plan.schema_version == 1,
        "unsupported rollout observation"
    );
    let expected = train_tasks()
        .into_iter()
        .find(|t| t.id == plan.task.id)
        .context("rollout task is not an original train task")?;
    ensure!(
        json_equal(&expected, &plan.task)? && plan.student_request.prompt == plan.task.prompt(),
        "student rollout task or original prompt changed"
    );
    crate::curate::validate_train_origin(&plan.task)?;
    plan.input_bundle.verify(root)?;
    plan.student_assets.verify(root)?;
    if let Some(cp) = &plan.student_checkpoint {
        cp.verify(root)?;
    }
    let (_, _, identity) = validated_student(
        &plan.student_base,
        plan.student_checkpoint
            .as_ref()
            .map(|a| a.path.parent().context("rollout checkpoint parent absent"))
            .transpose()?,
        plan.student_kind,
    )?;
    ensure!(
        identity == plan.student_identity,
        "recorded rollout student identity differs from its actual checkpoint"
    );
    let preflight: sandbox::PreflightReport =
        serde_json::from_slice(&fs::read(plan.preflight.verify(root)?)?)?;
    ensure!(
        preflight.probes_passed
            && preflight.image_id == plan.sandbox.expected_image_id
            && preflight.policy_sha256 == sandbox::policy_sha256(&plan.sandbox)?,
        "rollout sandbox differs from its observed enforcement"
    );
    let original_plan = plan
        .imported_student_attempt
        .as_ref()
        .map(|import| recovery::verify_import(root, &plan, import))
        .transpose()?;
    let initial_sandbox = original_plan
        .as_ref()
        .map(|p| &p.sandbox)
        .unwrap_or(&plan.sandbox);
    let student_plan = if let Some(import) = &plan.imported_student_attempt {
        ensure!(
            json_equal(&record.student_raw, &import.student_raw)?,
            "continued observation substituted its original student generation"
        );
        &import.student_plan
    } else {
        &record.plan
    };
    let student = verify_raw(
        root,
        &record.student_raw,
        student_plan,
        &identity,
        &plan.student_runtime,
        plan.student_request.max_new_tokens,
    )?;
    let classified = recovery::classify_grade(
        Grade {
            report: record.student_grade.clone(),
            unverified_report: record.unverified_student_grade.clone(),
            error: (!record.errors.is_empty()).then(|| record.errors.join("; ")),
        },
        &plan.task,
        &student,
        initial_sandbox,
        root,
    )?;
    ensure!(
        json_equal(&classified.report, &record.student_grade)?
            && json_equal(
                &classified.unverified_report,
                &record.unverified_student_grade
            )?,
        "student report was not classified against its actual audit before feedback"
    );
    if let Some(import) = &plan.imported_student_attempt {
        let original: Grade =
            serde_json::from_slice(&fs::read(import.student_grade.verify(root)?)?)?;
        let original =
            recovery::classify_grade(original, &plan.task, &student, initial_sandbox, root)?;
        ensure!(
            json_equal(&original.report, &record.student_grade)?
                && json_equal(
                    &original.unverified_report,
                    &record.unverified_student_grade
                )?,
            "continued observation changed its original student grading result"
        );
    }
    let student_grade = record
        .student_grade
        .as_ref()
        .map(|a| verified_grade(root, a, &plan, &student, initial_sandbox))
        .transpose()?;
    let content_role = if let Some(policy) = &plan.content_role_policy {
        ensure!(
            policy == CONTENT_ROLE_POLICY,
            "unknown student content-role policy"
        );
        record
            .student_grade
            .as_ref()
            .zip(student.generation.as_ref())
            .map(|(report, g)| {
                classify_content(
                    root,
                    report,
                    initial_sandbox,
                    &plan.task,
                    &g.completion,
                    g.complete,
                )
            })
            .transpose()?
    } else {
        None
    };
    ensure!(
        json_equal(&content_role, &record.student_content_eligibility)?,
        "student content eligibility differs from its unchanged original grade and actual audit"
    );
    let teacher = if plan.mode == RolloutMode::SelfDistillation {
        let teacher = self_teacher(root, &plan)?;
        ensure!(
            record.teacher_id == teacher.id,
            "self-distillation labelled another teacher"
        );
        if let Some(existing) = bundle.teachers.iter().find(|t| t.id == teacher.id) {
            ensure!(
                json_equal(&existing.model, &teacher.model)?
                    && json_equal(&existing.adapter_checkpoint, &teacher.adapter_checkpoint)?
                    && json_equal(&existing.assets, &teacher.assets)?,
                "self teacher changed checkpoint/base"
            );
        } else {
            bundle
                .capabilities
                .insert(teacher.id.clone(), teacher.capabilities.clone());
            bundle.teachers.push(teacher.clone());
        }
        teacher
    } else {
        bundle
            .teachers
            .iter()
            .find(|t| t.id == record.teacher_id)
            .cloned()
            .context("on-policy correction teacher absent from original evidence")?
    };
    ensure!(
        teacher.rights.eligible,
        "rollout teacher has unresolved training rights"
    );
    let mut success = false;
    if let Some(fp_a) = &record.feedback_plan {
        let fp: FeedbackPlan = serde_json::from_slice(&fs::read(fp_a.verify(root)?)?)?;
        let initial = student_grade
            .as_ref()
            .context("feedback lacks an actual student grade")?;
        ensure!(
            content_role
                .as_ref()
                .map(ContentEligibility::feedback_allowed)
                .unwrap_or(!initial.quarantined)
                && fp.schema_version == 1
                && json_equal(&fp.initial_plan, &record.plan)?
                && json_equal(&fp.student_raw, &record.student_raw)?
                && Some(&fp.student_grade.sha256)
                    == record.student_grade.as_ref().map(|a| &a.sha256)
                && fp.teacher_id == teacher.id
                && json_equal(&fp.teacher_assets, &teacher.assets)?
                && json_equal(&fp.teacher_checkpoint, &teacher.adapter_checkpoint)?,
            "feedback initial attempt, trusted grade or real teacher changed"
        );
        ensure!(
            fp.request.prompt
                == feedback_prompt(
                    bundle,
                    &plan,
                    &student,
                    initial,
                    &fp.supporting_sample_ids,
                    content_role.as_ref()
                )?,
            "teacher feedback prompt/context changed"
        );
        let checkpoint = if let Some(a) = &teacher.adapter_checkpoint {
            let m: ExperimentalCheckpointManifest =
                serde_json::from_slice(&fs::read(a.verify(root)?)?)?;
            m.adapter_sha256
        } else {
            let a: model::AssetManifest =
                serde_json::from_slice(&fs::read(teacher.assets.verify(root)?)?)?;
            comparative::comparative_checkpoint_id(&a)?
        };
        let feedback_a = record
            .feedback_raw
            .as_ref()
            .context("feedback plan has no raw observation")?;
        let feedback = verify_raw(
            root,
            feedback_a,
            fp_a,
            &checkpoint,
            &fp.runtime,
            fp.request.max_new_tokens,
        )?;
        let classified = recovery::classify_grade(
            Grade {
                report: record.feedback_grade.clone(),
                unverified_report: record.unverified_feedback_grade.clone(),
                error: (!record.errors.is_empty()).then(|| record.errors.join("; ")),
            },
            &plan.task,
            &feedback,
            &plan.sandbox,
            root,
        )?;
        ensure!(
            json_equal(&classified.report, &record.feedback_grade)?
                && json_equal(
                    &classified.unverified_report,
                    &record.unverified_feedback_grade
                )?,
            "corrective report was not classified against its actual audit before curation"
        );
        if let Some(grade_a) = &record.feedback_grade {
            let grade = verified_grade(root, grade_a, &plan, &feedback, &plan.sandbox)?;
            let generation = feedback
                .generation
                .as_ref()
                .context("graded feedback output absent")?;
            success = grade.passed
                && !grade.quarantined
                && grade.trace_complete
                && generation.complete
                && !generation.completion.trim().is_empty();
            if success {
                let (preference_prompt, prompt_transform) = preference_conditioning(
                    &plan.task,
                    &fp.request.prompt,
                    &plan.student_request.prompt,
                );
                let id = artefact::hash(
                    format!("{}:{}:{:?}", artifact.sha256, plan.task.id, plan.mode).as_bytes(),
                );
                let content = if plan.mode == RolloutMode::SelfDistillation {
                    ContentKind::Context
                } else {
                    ContentKind::ToolTrajectory
                };
                bundle.samples.push(EvidenceSample{id:id.clone(),task:plan.task.clone(),teacher_id:teacher.id.clone(),prompt:fp.request.prompt,completion:generation.completion.clone(),content,observation:artifact.clone(),raw_generation:feedback_a.clone(),sandbox_report:grade_a.clone(),audit:capture(root,&grade.audit_path)?,sequence_sha256:artefact::hash(generation.completion.as_bytes()),provenance_kind:"actual_checkpoint_bound_student_rollout_teacher_feedback_and_independent_regrade".into(),student_rollout:Some(StudentRolloutProvenance{student_checkpoint_sha256:identity,student_raw_rollout:record.student_raw.clone(),teacher_feedback:feedback_a.clone(),regraded_sandbox_report:grade_a.clone()})});
                if let Some(initial_a) = &record.student_grade
                    && content_role
                        .as_ref()
                        .is_some_and(ContentEligibility::negative_allowed)
                    && student
                        .generation
                        .as_ref()
                        .is_some_and(|g| !g.completion.trim().is_empty())
                {
                    let rejected = EvidenceCandidate {
                        task_id: plan.task.id.clone(),
                        teacher_id: format!("frozen-student-{}", &plan.student_identity[..16]),
                        completion: student
                            .generation
                            .as_ref()
                            .context("student negative absent")?
                            .completion
                            .clone(),
                        observation: artifact.clone(),
                        raw_generation: record.student_raw.clone(),
                        sandbox_report: initial_a.clone(),
                        audit: capture(root, &initial.audit_path)?,
                        passed: false,
                        quarantined: initial.quarantined,
                    };
                    bundle.preferences.push(PreferencePair{id:artefact::hash(format!("{}:{}",id,rejected.raw_generation.sha256).as_bytes()),task_id:plan.task.id.clone(),chosen_sample_id:id,rejected,reason:"Actual executed oracle-only student failure and independently verified teacher correction; original quarantine retained and preference conditioning declared separately".into(),rejected_eligibility:content_role.clone(),preference_prompt:Some(preference_prompt),prompt_transform:Some(prompt_transform)});
                }
                let current = bundle
                    .teachers
                    .iter_mut()
                    .find(|t| t.id == teacher.id)
                    .context("rollout teacher vanished")?;
                let mark = |c: &mut SignalAvailability| {
                    c.available = true;
                    c.artefacts.push(artifact.clone());
                    c.reason="Actual checkpoint-bound frozen-current-student attempt, trusted aggregate tool feedback, real corrective generation and independent regrade; subsequent optimiser updates are offline".into();
                };
                mark(&mut current.capabilities.complete_sequences);
                mark(&mut current.capabilities.on_policy);
                if plan.mode == RolloutMode::SelfDistillation {
                    mark(&mut current.capabilities.self_distillation);
                    mark(&mut current.capabilities.context);
                } else {
                    mark(&mut current.capabilities.tool_trajectories);
                }
                if bundle.preferences.iter().any(|p| {
                    p.chosen_sample_id == bundle.samples.last().map(|s| s.id.as_str()).unwrap_or("")
                }) {
                    mark(&mut current.capabilities.preferences);
                }
                bundle
                    .capabilities
                    .insert(current.id.clone(), current.capabilities.clone());
            }
        } else {
            ensure!(
                record.errors.iter().any(|e| !e.is_empty())
                    || feedback.generation.as_ref().is_some_and(|g| !g.complete),
                "complete feedback has neither grade nor explicit error"
            );
        }
    } else {
        ensure!(
            record.feedback_raw.is_none()
                && record.feedback_grade.is_none()
                && record.unverified_feedback_grade.is_none()
                && !record.errors.is_empty(),
            "missing feedback without explicit failed observation"
        );
    }
    ensure!(
        success == record.success,
        "rollout success differs from actual generation/grade evidence"
    );
    bundle.rollout_records.push(artifact.clone());
    Ok(())
}
