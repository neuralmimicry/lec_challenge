//! Matched local evaluation of real baseline weights and experimental adapters.
//! This entry point exposes train/development only and cannot authorise shipping.
use super::*;

enum BitNetEvaluationRuntime {
    Rust(Box<crate::bitnet::BitNetRuntime>),
    Cpp(Box<crate::bitnet::BitNetCppRuntime>),
}
impl crate::study_evaluate::StudyGenerator for BitNetEvaluationRuntime {
    fn generate(
        &mut self,
        request: crate::model::GenerationRequest,
    ) -> Result<crate::model::Generation> {
        match self {
            Self::Rust(runtime) => runtime.generate(request),
            Self::Cpp(runtime) => runtime.generate(request),
        }
    }
    fn execution_metadata(&self) -> serde_json::Value {
        match self {
            Self::Rust(runtime) => {
                crate::study_evaluate::StudyGenerator::execution_metadata(runtime.as_ref())
            }
            Self::Cpp(runtime) => {
                crate::study_evaluate::StudyGenerator::execution_metadata(runtime.as_ref())
            }
        }
    }
}

pub struct CheckpointEvalOptions {
    pub base: PathBuf,
    /// Absence means the unadapted original base, with its actual tensor hash.
    pub checkpoint: Option<PathBuf>,
    pub architecture: StudentArchitecture,
    pub runtime: RuntimeOptions,
    pub output: PathBuf,
    pub split: crate::study_evaluate::StudySplit,
    pub cpu_threads: usize,
    pub offline: bool,
    /// Explicit choices are checked against the saved backend on every resume.
    pub bitnet_backend: Option<crate::bitnet::BitNetBackendOptions>,
}

pub async fn evaluate_checkpoint(options: CheckpointEvalOptions) -> Result<serde_json::Value> {
    ensure!(
        options.cpu_threads > 0,
        "evaluation requires assigned CPU capacity"
    );
    let preparation_output = options.output.clone();
    let _preparation_lease = tokio::task::spawn_blocking(move || -> Result<_> {
        fs::create_dir_all(&preparation_output)?;
        let lease = crate::engine::RunLease::acquire(
            &preparation_output.join(".study-evaluation-preparation.lock"),
        )?;
        let receipt = preparation_output.join("launch-engine.json");
        let executable = std::env::current_exe()?;
        if receipt.exists() {
            let pinned: crate::engine::PinnedEngine = serde_json::from_slice(&fs::read(&receipt)?)?;
            pinned.verify()?;
            ensure!(
                pinned.sha256 == artefact::file_hash(&executable)?,
                "evaluation preparation belongs to its original executable; resume that pinned engine"
            );
        } else {
            let pinned = crate::engine::pin_executable(&executable, Path::new("state/engines"))?;
            artefact::write_json(&receipt, &pinned)?;
        }
        Ok(lease)
    }).await??;
    if options.architecture == StudentArchitecture::TernaryBitnet {
        ensure!(
            options.runtime == RuntimeOptions::default(),
            "the pinned BitNet C++ backend requires explicit CPU/F32 execution"
        );
    }
    let base = options.base.clone();
    let checkpoint = options.checkpoint.clone();
    let architecture = options.architecture;
    let identity = tokio::task::spawn_blocking(move || -> Result<_> {
        let assets = match architecture {
            StudentArchitecture::ConventionalQwenLora => model::validate_assets(&base)?,
            StudentArchitecture::TernaryBitnet => {
                crate::hub::validate_bitnet_student_assets(&base)?
            }
        };
        let base_assets = capture(&base.join("assets.json"))?;
        let (checkpoint_id, tensor, manifest) = if let Some(path) = checkpoint {
            let manifest = distillation::checkpoint::validate_checkpoint(&path, &assets)?;
            let expected = match architecture {
                StudentArchitecture::ConventionalQwenLora => {
                    distillation::DistillationStudent::ConventionalQwenLora
                }
                StudentArchitecture::TernaryBitnet => {
                    distillation::DistillationStudent::TernaryBitnet
                }
            };
            ensure!(
                manifest.spec.student_kind == expected && manifest.spec.student_base == base,
                "local evaluation checkpoint architecture/base differs from the requested model"
            );
            (
                manifest.adapter_sha256,
                capture(&path.join("adapter.safetensors"))?,
                capture(&path.join("manifest.json"))?,
            )
        } else {
            (
                assets.files["model.safetensors"].clone(),
                capture(&base.join("model.safetensors"))?,
                base_assets.clone(),
            )
        };
        Ok(crate::study_evaluate::Checkpoint {
            checkpoint_id,
            tensor,
            manifest,
            base_assets,
            architecture,
            tokenizer: capture(&base.join("tokenizer.json"))?,
        })
    })
    .await??;
    let mut backend_artefacts = Vec::new();
    let mut prepared_bitnet = None;
    let selection = options.bitnet_backend.clone().unwrap_or_default();
    if options.architecture == StudentArchitecture::TernaryBitnet {
        let effective_offline =
            options.offline || std::env::var("LEC_STUDY_OFFLINE").is_ok_and(|value| value == "1");
        let saved = options.output.join("plan.json");
        prepared_bitnet = if saved.exists() {
            let frozen: serde_json::Value = serde_json::from_slice(&fs::read(saved)?)?;
            backend_artefacts = serde_json::from_value(
                frozen
                    .get("backend_artefacts")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])),
            )?;
            let frozen_artefacts = backend_artefacts.clone();
            let requested = selection.clone();
            tokio::task::spawn_blocking(move || {
                crate::bitnet::resolve_frozen_backend(&frozen_artefacts, &requested)
            })
            .await??
        } else {
            let prepared = crate::bitnet::prepare_backend(
                &options.base,
                options.checkpoint.as_deref(),
                &options.output,
                &selection,
                effective_offline,
                options.cpu_threads,
            )
            .await?;
            backend_artefacts = prepared.backend_artefacts.clone();
            Some(prepared)
        };
    } else {
        ensure!(
            selection.engine_manifest.is_none()
                && selection.gguf.is_none()
                && selection.parity_report.is_none(),
            "BitNet backend overrides cannot apply to a conventional checkpoint"
        );
    }
    let evaluation = crate::study_evaluate::StudyEvalOptions {
        checkpoint: identity,
        runtime: serde_json::to_value(&options.runtime)?,
        engine: capture(&std::env::current_exe()?)?,
        backend_artefacts,
        output: options.output.clone(),
        split: options.split,
        maximum_concurrent_graders: options.cpu_threads.clamp(1, 256),
        sandbox: crate::sandbox::SandboxConfig {
            report_dir: options.output.join("sandbox"),
            ..Default::default()
        },
    };
    let output = options.output.clone();
    let parity_report = selection.parity_report.clone();
    let mut report = match options.architecture {
        StudentArchitecture::ConventionalQwenLora => {
            crate::study_evaluate::run(evaluation, move || {
                if let Some(checkpoint) = options.checkpoint {
                    distillation::load_runtime(&options.base, &checkpoint, options.runtime)
                } else {
                    model::ModelRuntime::load_with_options(&options.base, None, options.runtime)
                }
            })
            .await
        }
        StudentArchitecture::TernaryBitnet => {
            crate::study_evaluate::run(evaluation, move || {
                let Some(prepared) = prepared_bitnet else {
                    // Historical Rust-only plans remain Rust-only, including
                    // their original lack of native backend artefacts.
                    let runtime = if let Some(checkpoint) = &options.checkpoint {
                        crate::bitnet::BitNetRuntime::load(&options.base, checkpoint)
                    } else {
                        crate::bitnet::BitNetRuntime::load_base(&options.base)
                    }?;
                    return Ok(BitNetEvaluationRuntime::Rust(Box::new(runtime)));
                };
                let runtime = crate::bitnet::CppRuntimeOptions {
                    engine_manifest: prepared.engine_manifest,
                    gguf: prepared.gguf,
                    threads: options.cpu_threads.saturating_sub(1).max(1),
                    ..Default::default()
                };
                crate::bitnet::BitNetCppRuntime::load_with_options(
                    &options.base,
                    options.checkpoint.as_deref(),
                    runtime,
                )
                .map(|runtime| BitNetEvaluationRuntime::Cpp(Box::new(runtime)))
            })
            .await
        }
    }?;
    if options.architecture == StudentArchitecture::TernaryBitnet {
        if let Some(parity) = parity_report {
            let target = output.clone();
            tokio::task::spawn_blocking(move || {
                crate::study_evaluate::parity::attach_numerical_equivalence(&target, &parity)
            })
            .await??;
        }
        let numerical = tokio::task::spawn_blocking(move || {
            crate::study_evaluate::verify_numerical_equivalence(&output)
        })
        .await??;
        report["numerical_equivalence"] = serde_json::to_value(numerical)?;
    }
    Ok(report)
}
