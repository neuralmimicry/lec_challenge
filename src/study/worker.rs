//! Supervise genuine updates and a separate checkpoint evaluation process.
use super::*;
pub fn worker_spec(request: &experiments::WorkerRequest) -> Result<DistillationWorkerSpec> {
    let student = request
        .inputs
        .students
        .iter()
        .find(|s| s.architecture == request.arm.student)
        .context("worker student identity absent")?;
    let base_assets = student
        .base_assets
        .as_ref()
        .context("student base assets missing")?;
    base_assets.verify(&std::env::current_dir()?)?;
    let base = base_assets
        .path
        .parent()
        .context("student base parent missing")?
        .to_owned();
    let mut losses = request
        .arm
        .signals
        .iter()
        .map(|signal| WeightedLoss {
            kind: match signal {
                SignalMethod::SequenceSft => LossKind::SequenceSft,
                SignalMethod::SoftTokenKl => LossKind::SoftTokenKl,
                SignalMethod::HiddenFeature => LossKind::HiddenFeature,
                SignalMethod::AttentionRelation => LossKind::AttentionRelation,
            },
            weight: 1.,
        })
        .collect::<Vec<_>>();
    let mut contents = if request.arm.contents.is_empty() {
        vec![ContentKind::CompleteSequence]
    } else {
        request
            .arm
            .contents
            .iter()
            .map(|c| match c {
                ContentMethod::Process => ContentKind::Process,
                ContentMethod::ToolTrajectory => ContentKind::ToolTrajectory,
                ContentMethod::Context => ContentKind::Context,
                ContentMethod::Preference => ContentKind::Preference,
            })
            .collect()
    };
    if request.arm.contents.contains(&ContentMethod::Preference) {
        losses.push(WeightedLoss {
            kind: LossKind::PreferenceDpo,
            weight: 1.,
        });
        // A preference row identifies its original positive sequence through
        // the chosen sample ID; the worker independently verifies that link.
        if contents.iter().all(|c| *c == ContentKind::Preference) {
            contents.push(ContentKind::CompleteSequence);
        }
    }
    let initial_checkpoint = if request.arm.data_policy == experiments::DataPolicy::OnPolicy
        || request.arm.self_distillation
    {
        let bundle = teacher_evidence::load_verified(&request.inputs.evidence_bundle.path)?;
        teacher_evidence::resolve_rollout_initial_checkpoint(&bundle, &request.arm.teacher_ids)?
    } else {
        None
    };
    Ok(DistillationWorkerSpec {
        schema_version: 1,
        arm_id: request.arm.id.clone(),
        student_kind: match request.arm.student {
            StudentArchitecture::ConventionalQwenLora => {
                distillation::DistillationStudent::ConventionalQwenLora
            }
            StudentArchitecture::TernaryBitnet => distillation::DistillationStudent::TernaryBitnet,
        },
        student_base: base,
        evidence_bundle: request.inputs.evidence_bundle.path.clone(),
        output: request.output.join("checkpoint"),
        seed: request.arm.seed,
        steps: request.training.steps,
        max_sequence_tokens: request.training.max_sequence_tokens,
        sequence_policy: match request.training.sequence_policy {
            experiments::SequencePolicy::RejectOverlength => {
                distillation::SequencePolicy::RejectOverlength
            }
            experiments::SequencePolicy::AssistantWindow => {
                distillation::SequencePolicy::AssistantWindow
            }
        },
        rank: request.training.rank,
        alpha: request.training.alpha,
        learning_rate: request.training.learning_rate,
        temperature: request.training.temperature,
        dpo_beta: request.training.dpo_beta,
        losses,
        teacher_ids: request.arm.teacher_ids.clone(),
        contents,
        teacher_runtime: request.inputs.teacher_runtime.clone(),
        student_runtime: request.inputs.student_runtime.clone(),
        teacher_target_policy: if request.teacher_targets_cache_only {
            distillation::TeacherTargetPolicy::CacheOnly
        } else {
            distillation::TeacherTargetPolicy::Acquire
        },
        initial_checkpoint,
        data_policy: match request.arm.data_policy {
            experiments::DataPolicy::Offline => distillation::DataPolicy::Offline,
            experiments::DataPolicy::OnPolicy => distillation::DataPolicy::OnPolicy,
        },
        progressive: request.arm.progressive,
        self_distillation: request.arm.self_distillation,
    })
}

/// Numerical work runs in the scheduler's dedicated process. Publish the
/// completion envelope only after real steps, export and verified reload.
pub async fn run_worker(
    request_path: PathBuf,
    output: PathBuf,
) -> Result<experiments::WorkerOutcome> {
    let verify_path = request_path.clone();
    let request =
        tokio::task::spawn_blocking(move || experiments::validate_worker_request(&verify_path))
            .await??;
    ensure!(
        request.output == output,
        "worker output differs from its frozen request"
    );
    let requested_spec = request.clone();
    let spec = tokio::task::spawn_blocking(move || worker_spec(&requested_spec)).await??;
    let started = std::time::Instant::now();
    let trained = tokio::task::spawn_blocking(move || distillation::run_worker(&spec)).await??;
    let path = output.join("training-artifact.json");
    artefact::write_json(&path, &trained)?;
    let mut command = tokio::process::Command::new(std::env::current_exe()?);
    command
        .args(["study-evaluate-worker", "--request"])
        .arg(&request_path)
        .arg("--training-artifact")
        .arg(&path)
        .arg("--output")
        .arg(output.join("development"))
        .kill_on_drop(true);
    #[cfg(target_os = "linux")]
    {
        let expected_parent = std::process::id() as libc::pid_t;
        // SAFETY: the child hook performs only direct, allocation-free syscalls.
        // The fixed Tokio executor thread survives for the worker's lifetime.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != expected_parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
    }
    let mut child = command.spawn()?;
    let evaluation_pid = child.id().context("fresh evaluation process has no PID")?;
    artefact::write_json(
        &output.join("evaluation-process.json"),
        &serde_json::json!({
            "schema_version":1,"evidence_kind":"actual_fresh_process_checkpoint_reload",
            "training_process_id":std::process::id(),"evaluation_process_id":evaluation_pid,
            "engine":capture(&std::env::current_exe()?)?,"training_artifact":capture(&path)?,
        }),
    )?;
    eprintln!("study: fresh-process checkpoint evaluation started, PID={evaluation_pid}");
    ensure!(
        child.wait().await?.success(),
        "fresh-process development evaluation failed; training and partial evaluation retained"
    );
    let evaluation_output = output.join("development");
    let development = tokio::task::spawn_blocking(move || {
        crate::study_evaluate::verify_completed(&evaluation_output)
    })
    .await??;
    ensure!(
        development["coverage_complete"] == true,
        "study arm lacks complete matched development observations"
    );
    let outcome = experiments::WorkerOutcome {
        schema_version: 1,
        arm_id: request.arm.id.clone(),
        request_sha256: artefact::file_hash(&request_path)?,
        training_artifact: capture(&path)?,
        checkpoint_artifacts: vec![
            trained.manifest.clone(),
            capture(&trained.checkpoint.join("adapter.safetensors"))?,
        ],
        checkpoint_changed: trained.changed_adapter,
        checkpoint_reloaded: trained.reloaded,
        completed_steps: trained.steps,
        finite_losses: trained.losses.clone(),
        finite_gradient_norms: trained.gradient_norms.clone(),
        train_evaluation: None,
        development_evaluation: Some(capture(&output.join("development/report.json"))?),
        measured_costs: serde_json::json!({"new_remote_api_spend_usd":"0", "local_worker_wall_seconds":started.elapsed().as_secs_f64(),
            "electricity_usd":null,"hardware_amortisation_usd":null,"cost_per_point_usd":null}),
        limitations: trained.limitations,
    };
    artefact::write_json(&output.join("outcome.json"), &outcome)?;
    Ok(outcome)
}

/// The optimiser process delegates reload and evaluation to this separate
/// process, which independently validates the request and checkpoint identity.
pub async fn evaluate_worker(
    request_path: PathBuf,
    training_path: PathBuf,
    output: PathBuf,
) -> Result<serde_json::Value> {
    let checked = request_path.clone();
    let request =
        tokio::task::spawn_blocking(move || experiments::validate_worker_request(&checked))
            .await??;
    ensure!(
        output == request.output.join("development")
            && training_path == request.output.join("training-artifact.json"),
        "evaluation paths differ from the frozen worker"
    );
    let trained: distillation::DistillationArtifact =
        serde_json::from_slice(&fs::read(&training_path)?)?;
    let requested_spec = request.clone();
    let spec = tokio::task::spawn_blocking(move || worker_spec(&requested_spec)).await??;
    ensure!(
        trained.arm_id == request.arm.id
            && trained.steps == request.training.steps
            && trained.checkpoint == spec.output
            && trained.changed_adapter
            && trained.reloaded
            && !trained.production_eligible
            && trained.adapter_sha256
                == artefact::file_hash(&spec.output.join("adapter.safetensors"))?,
        "evaluation training record does not match its real exported checkpoint"
    );
    let base_directory = spec.student_base.clone();
    let validate_spec = spec.clone();
    tokio::task::spawn_blocking(move || -> Result<()> {
        let assets = match validate_spec.student_kind {
            distillation::DistillationStudent::ConventionalQwenLora => {
                model::validate_assets(&validate_spec.student_base)?
            }
            distillation::DistillationStudent::TernaryBitnet => {
                crate::hub::validate_bitnet_student_assets(&validate_spec.student_base)?
            }
        };
        let manifest =
            distillation::checkpoint::validate_checkpoint(&validate_spec.output, &assets)?;
        ensure!(
            serde_json::to_value(&manifest.spec)? == serde_json::to_value(&validate_spec)?,
            "fresh-process checkpoint configuration differs from the frozen arm"
        );
        Ok(())
    })
    .await??;
    let bitnet_backend = if request.arm.student == StudentArchitecture::TernaryBitnet {
        let original = request.study_plan.clone();
        tokio::task::spawn_blocking(move || -> Result<_> {
            let root = std::env::current_dir()?;
            let plan: experiments::StudyPlan =
                serde_json::from_slice(&fs::read(original.verify(&root)?)?)?;
            Ok(experiments::StudyConfig::load(&plan.config.verify(&root)?)?.bitnet_backend)
        })
        .await??
    } else {
        None
    };
    super::evaluate_checkpoint(super::CheckpointEvalOptions {
        base: base_directory,
        checkpoint: Some(trained.checkpoint),
        architecture: request.arm.student,
        runtime: request.inputs.student_runtime,
        output,
        split: crate::study_evaluate::StudySplit::Development,
        cpu_threads: request.resources.cpu_threads as usize,
        offline: request.offline,
        bitnet_backend,
    })
    .await
}
