//! Derive an immutable experiment plan from independently verified evidence.
use super::*;
pub struct PlanOptions {
    pub config: PathBuf,
    pub evidence_bundle: PathBuf,
    pub base: PathBuf,
    pub bitnet_base: PathBuf,
    pub output: PathBuf,
    pub maximum_parallel_jobs: Option<usize>,
    pub teacher_runtime: RuntimeOptions,
}

pub(super) fn capture(path: &Path) -> Result<Artefact> {
    let root = std::env::current_dir()?.canonicalize()?;
    let relative = path.canonicalize()?.strip_prefix(&root)?.to_owned();
    Artefact::capture(&root, relative)
}

fn content_cap(signal: &SignalAvailability) -> Capability {
    Capability {
        supported: signal.supported,
        available: signal.available,
        acquirable_offline: false,
        reason: signal.reason.clone(),
        artefacts: signal.artefacts.clone(),
    }
}

fn tensor_cap(signal: &SignalAvailability, assets: &Artefact) -> Capability {
    let mut cap = content_cap(signal);
    if cap.supported && !cap.available {
        cap.acquirable_offline = true;
        cap.reason = format!(
            "Native teacher-forced acquisition from verified cached weights is supported; actual tensors must be published before optimisation. {}",
            cap.reason
        );
        cap.artefacts.push(assets.clone());
    }
    cap
}

fn tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path).map_err(|error| anyhow::anyhow!(error.to_string()))
}

fn student_capability(qa: &Path, bitnet: bool) -> Result<Capability> {
    if !qa.is_file() {
        return Ok(Capability {
            supported: true,
            available: false,
            acquirable_offline: false,
            reason: format!(
                "Native backend implemented; required numerical update/export/reload QA has not been observed at {}",
                qa.display()
            ),
            artefacts: vec![],
        });
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(qa)?)?;
    let valid = if bitnet {
        value["genuine_ternary_training"] == true
            && value["changed_effective_weights"] == true
            && value["reloaded"] == true
    } else {
        value["base_sha256"] == model::MODEL_SHA256
            && value["gradient_norm"]
                .as_f64()
                .is_some_and(|x| x.is_finite() && x > 0.)
            && value["changed_parameter_l2"]
                .as_f64()
                .is_some_and(|x| x.is_finite() && x > 0.)
            && value["frozen_base_sentinel_unchanged"] == true
            && value["adapter_safetensors_roundtrip"] == true
    };
    ensure!(valid, "student capability QA is invalid: {}", qa.display());
    Ok(Capability { supported: true, available: true, acquirable_offline: false,
        reason: if bitnet { "Observed differentiable ternary quantisation, changed effective weights and reload; this establishes operator capability, not task quality." }
        else { "Observed actual pinned-base forward/backward/optimiser update, frozen-base invariant and adapter reload; synthetic QA is not teacher-derived learning." }.into(),
        artefacts: vec![capture(qa)?] })
}

/// Existing plans retain their measured capacity and original semantic inputs.
/// Current resource availability is rechecked by the scheduler before dispatch.
pub fn create_plan(options: PlanOptions) -> Result<experiments::StudyPlan> {
    let bundle_path = if options.evidence_bundle.is_dir() {
        options.evidence_bundle.join("bundle.json")
    } else {
        options.evidence_bundle.clone()
    };
    let existing = options.output.join("plan.json");
    if existing.is_file() {
        let plan = experiments::load_plan(&existing)?;
        ensure!(
            plan.config.sha256 == artefact::file_hash(&options.config)?
                && plan.inputs.evidence_bundle.sha256 == artefact::file_hash(&bundle_path)?
                && plan.inputs.student_base.sha256
                    == artefact::file_hash(&options.base.join("assets.json"))?
                && serde_json::to_value(&plan.inputs.teacher_runtime)?
                    == serde_json::to_value(&options.teacher_runtime)?,
            "resume inputs differ from the immutable study; use --fresh or a new explicit output"
        );
        if let Some(jobs) = options.maximum_parallel_jobs {
            ensure!(
                jobs == plan.inputs.resources.maximum_parallel_jobs,
                "resume concurrency differs from the frozen study"
            );
        }
        if let Some(bitnet) = plan
            .inputs
            .students
            .iter()
            .find(|s| s.architecture == StudentArchitecture::TernaryBitnet)
            && let Some(assets) = &bitnet.base_assets
        {
            ensure!(
                assets.sha256 == artefact::file_hash(&options.bitnet_base.join("assets.json"))?
                    && assets.path == capture(&options.bitnet_base.join("assets.json"))?.path,
                "resume BitNet base differs from the frozen student"
            );
        }
        eprintln!(
            "study: reusing immutable plan and validated progress at {}",
            existing.display()
        );
        return Ok(plan);
    }
    let bundle = teacher_evidence::load_verified(&bundle_path)?;
    let base = model::validate_assets(&options.base)?;
    let student_tokenizer = tokenizer(&options.base.join("tokenizer.json"))?;
    let student_vocab = student_tokenizer.get_vocab(true);
    let mut teachers = Vec::new();
    let mut teacher_vocabs = BTreeMap::new();
    for teacher in &bundle.teachers {
        let source = if teacher.model.directory.join("gguf-assets.json").is_file() {
            teacher.model.directory.join("source")
        } else {
            teacher.model.directory.clone()
        };
        let vocabulary = tokenizer(&source.join("tokenizer.json"))?.get_vocab(true);
        let aligned = vocabulary == student_vocab;
        teacher_vocabs.insert(teacher.id.clone(), vocabulary);
        let caps = &teacher.capabilities;
        let signals = BTreeMap::from([
            (
                SignalMethod::SequenceSft,
                content_cap(&caps.complete_sequences),
            ),
            (
                SignalMethod::SoftTokenKl,
                tensor_cap(&caps.logits, &teacher.assets),
            ),
            (
                SignalMethod::HiddenFeature,
                tensor_cap(&caps.hidden_states, &teacher.assets),
            ),
            (
                SignalMethod::AttentionRelation,
                tensor_cap(&caps.attention, &teacher.assets),
            ),
        ]);
        let source_model_id = if teacher
            .model
            .directory
            .join("gguf-provenance.json")
            .is_file()
        {
            let proof: crate::hub::HubGgufProvenance = serde_json::from_slice(&fs::read(
                teacher.model.directory.join("gguf-provenance.json"),
            )?)?;
            proof.source_assets.manifest.model_id
        } else {
            teacher.model.model_id.clone()
        };
        teachers.push(experiments::TeacherProfile {
            id: teacher.id.clone(),
            source_model_id,
            revision: teacher.model.revision.clone(),
            tokenizer_sha256: caps.tokeniser_sha256.clone(),
            token_alignment_verified: aligned,
            rights_approved: teacher.rights.eligible,
            signals,
            contents: BTreeMap::from([
                (ContentMethod::Process, content_cap(&caps.process)),
                (
                    ContentMethod::ToolTrajectory,
                    content_cap(&caps.tool_trajectories),
                ),
                (ContentMethod::Context, content_cap(&caps.context)),
                (ContentMethod::Preference, content_cap(&caps.preferences)),
            ]),
            on_policy: content_cap(&caps.on_policy),
            self_distillation: content_cap(&caps.self_distillation),
            rollout_data: experiments::rollout_profile(&bundle, &teacher.id)?,
        });
    }
    let conventional = student_capability(
        Path::new("reports/comparative/native-gradient-attention-fix.json"),
        false,
    )?;
    let mut students = vec![experiments::StudentProfile {
        architecture: StudentArchitecture::ConventionalQwenLora,
        base_assets: Some(capture(&options.base.join("assets.json"))?),
        tokenizer_sha256: Some(base.files["tokenizer.json"].clone()),
        teacher_token_alignment: teachers
            .iter()
            .map(|t| (t.id.clone(), t.token_alignment_verified))
            .collect(),
        training: conventional.clone(),
        exported_checkpoint_reload: conventional,
    }];
    let mut bitnet_qa = student_capability(
        Path::new("reports/comparative/bitnet/operator-qa.json"),
        true,
    )?;
    let bitnet_assets = if options.bitnet_base.join("assets.json").is_file() {
        let verified = crate::hub::validate_bitnet_student_assets(&options.bitnet_base)?;
        let vocabulary = tokenizer(&options.bitnet_base.join("tokenizer.json"))?.get_vocab(true);
        Some((
            capture(&options.bitnet_base.join("assets.json"))?,
            verified.files["tokenizer.json"].clone(),
            teacher_vocabs
                .iter()
                .map(|(id, v)| (id.clone(), *v == vocabulary))
                .collect(),
        ))
    } else {
        None
    };
    if bitnet_assets.is_none() {
        bitnet_qa.available = false;
        bitnet_qa.reason = "Optional BitNet student weights are not cached; the conventional student remains available".into();
    }
    students.push(experiments::StudentProfile {
        architecture: StudentArchitecture::TernaryBitnet,
        base_assets: bitnet_assets.as_ref().map(|x| x.0.clone()),
        tokenizer_sha256: bitnet_assets.as_ref().map(|x| x.1.clone()),
        teacher_token_alignment: bitnet_assets.map(|x| x.2).unwrap_or_default(),
        training: bitnet_qa.clone(),
        exported_checkpoint_reload: bitnet_qa,
    });
    let hardware = platform::detect()?;
    // CPU optimisation and GPU target acquisition are distinct resource phases.
    // Reservations remain conservative when acquisition has not been prewarmed.
    let ram_per_worker = 16u64 << 30;
    ensure!(
        hardware.usable_ram_bytes >= ram_per_worker,
        "study needs {ram_per_worker} usable RAM bytes for its configured full worker; reduce its declared fidelity explicitly"
    );
    let jobs = options
        .maximum_parallel_jobs
        .unwrap_or(4)
        .min(hardware.available_cpu_threads)
        .min((hardware.usable_ram_bytes / (8u64 << 30)) as usize)
        .max(1);
    let gpu = match options.teacher_runtime.device {
        model::ExecutionDevice::Cpu => None,
        model::ExecutionDevice::Cuda { ordinal } => Some(ordinal),
    };
    let vram = gpu
        .and_then(|ordinal| hardware.gpus.iter().find(|g| g.ordinal == ordinal))
        .map_or(0, |g| {
            g.total_memory_bytes
                .saturating_sub(g.total_memory_bytes / 10)
                .saturating_sub(256 << 20)
        });
    let mut capture_vram = 0;
    if gpu.is_some() {
        for teacher in &bundle.teachers {
            let quantised = teacher.model.directory.join("gguf-assets.json").is_file();
            let config_path = teacher.model.directory.join(if quantised {
                "source/config.json"
            } else {
                "config.json"
            });
            let cfg: model::QwenConfig = serde_json::from_slice(&fs::read(config_path)?)?;
            let manifest: model::AssetManifest = serde_json::from_slice(&fs::read(
                teacher.assets.verify(&std::env::current_dir()?)?,
            )?)?;
            let weight_bytes = manifest
                .files
                .keys()
                .filter(|name| name.ends_with(".gguf") || name.ends_with(".safetensors"))
                .try_fold(0u64, |total, name| -> Result<u64> {
                    total
                        .checked_add(fs::metadata(teacher.model.directory.join(name))?.len())
                        .context("weight size overflow")
                })?;
            let estimate = if quantised {
                model::estimate_quantised_memory(&cfg, weight_bytes, 1024, true)?
                    .conservative_total_bytes
            } else {
                weight_bytes
                    .saturating_mul(
                        if options.teacher_runtime.dtype == model::InferenceDtype::F32 {
                            2
                        } else {
                            1
                        },
                    )
                    .saturating_add(768 << 20)
            };
            capture_vram = capture_vram.max(estimate);
        }
    }
    let quick_threads = u32::try_from(hardware.available_cpu_threads / jobs)?;
    let full_jobs = jobs
        .min((hardware.usable_ram_bytes / ram_per_worker) as usize)
        .max(1);
    let full_threads = u32::try_from(hardware.available_cpu_threads / full_jobs)?;
    let mut student_resources = vec![experiments::StudentResources {
        student: StudentArchitecture::ConventionalQwenLora,
        quick: experiments::ResourceRequest {
            cpu_threads: quick_threads,
            ram_bytes: 8u64 << 30,
            vram_bytes: 0,
        },
        full: experiments::ResourceRequest {
            cpu_threads: full_threads,
            ram_bytes: ram_per_worker,
            vram_bytes: 0,
        },
        tensor_capture_vram_bytes: capture_vram,
        teacher_targets_cache_only: false,
    }];
    if hardware.usable_ram_bytes >= 24u64 << 30 {
        let bitnet_jobs = jobs
            .min((hardware.usable_ram_bytes / (24u64 << 30)) as usize)
            .max(1);
        let bitnet_threads = u32::try_from(hardware.available_cpu_threads / bitnet_jobs)?;
        student_resources.push(experiments::StudentResources {
            student: StudentArchitecture::TernaryBitnet,
            quick: experiments::ResourceRequest {
                cpu_threads: bitnet_threads,
                ram_bytes: 24u64 << 30,
                vram_bytes: 0,
            },
            full: experiments::ResourceRequest {
                cpu_threads: bitnet_threads,
                ram_bytes: 24u64 << 30,
                vram_bytes: 0,
            },
            tensor_capture_vram_bytes: capture_vram,
            teacher_targets_cache_only: false,
        });
    } else if let Some(bitnet) = students
        .iter_mut()
        .find(|s| s.architecture == StudentArchitecture::TernaryBitnet)
    {
        bitnet.training.available = false;
        bitnet.training.reason = "Native backend exists, but this plan's available CPU RAM is below its conservative 24 GiB worker requirement".into();
    }
    let inputs = experiments::StudyInputs {
        schema_version: 1,
        evidence_bundle: capture(&bundle_path)?,
        student_base: capture(&options.base.join("assets.json"))?,
        student_tokenizer_sha256: base.files["tokenizer.json"].clone(),
        teachers,
        students,
        resources: experiments::ResourceCapacity {
            cpu_threads: u32::try_from(hardware.available_cpu_threads)?,
            ram_bytes: hardware.usable_ram_bytes,
            gpu_ordinal: gpu,
            vram_bytes: vram,
            maximum_parallel_jobs: jobs,
        },
        worker_resources: experiments::ResourceRequest {
            cpu_threads: quick_threads,
            ram_bytes: ram_per_worker,
            vram_bytes: 0,
        },
        student_resources,
        teacher_runtime: options.teacher_runtime,
        student_runtime: RuntimeOptions::default(),
    };
    experiments::create_plan(&options.config, &options.output, inputs)
}
