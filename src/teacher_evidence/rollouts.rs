//! Checkpoint-bound student rollouts and independently regraded teacher revisions.
use super::*;
use crate::{
    distillation::{DistillationStudent, checkpoint::ExperimentalCheckpointManifest},
    model::{Generation, GenerationRequest, ModelRuntime},
    sandbox::{self, SandboxConfig},
};

mod health;
mod recovery;
mod runtime;
mod verify;
pub(crate) use runtime::RecordedModel;
pub(super) use runtime::{validate_context_observation, validate_teacher_request};
pub(super) use verify::append_verified;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RolloutMode {
    OnPolicy,
    SelfDistillation,
}

pub struct RolloutCollectOptions {
    pub bundle: PathBuf,
    pub output: PathBuf,
    pub student_base: PathBuf,
    pub student_checkpoint: Option<PathBuf>,
    pub student_kind: DistillationStudent,
    pub mode: RolloutMode,
    pub teacher_ids: Vec<String>,
    pub teacher_runtime: RuntimeOptions,
    /// An explicit context applies only to the original GGUF correction teacher.
    pub teacher_context_tokens: Option<usize>,
    pub student_runtime: RuntimeOptions,
    pub max_tokens: usize,
    pub seed: u64,
    pub maximum_tasks: usize,
    /// Reuse immutable student attempts; feedback has its own new protocol.
    pub reuse_student_attempts_from: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutInitialMetadata {
    pub student_base: PathBuf,
    pub student_kind: DistillationStudent,
    pub base_sha256: String,
    pub checkpoint_sha256: String,
    pub checkpoint: Option<Artefact>,
    pub rank: Option<usize>,
    pub alpha: Option<f64>,
}

/// Resolve the exact frozen student underlying all selected rollout teachers.
/// `None` means actual validated base-weight rollouts, never an unknown model.
pub fn resolve_rollout_initial_checkpoint(
    bundle: &EvidenceBundle,
    teacher_ids: &[String],
) -> Result<Option<PathBuf>> {
    resolve_rollout_initial_metadata(bundle, teacher_ids)?
        .checkpoint
        .map(|a| {
            a.path
                .parent()
                .context("rollout checkpoint parent absent")
                .map(Path::to_owned)
        })
        .transpose()
}

pub fn resolve_rollout_initial_metadata(
    bundle: &EvidenceBundle,
    teacher_ids: &[String],
) -> Result<RolloutInitialMetadata> {
    let root = std::env::current_dir()?;
    let mut metadata = None;
    let mut seen = BTreeSet::new();
    for sample in &bundle.samples {
        if !teacher_ids.contains(&sample.teacher_id) {
            continue;
        }
        let Some(provenance) = &sample.student_rollout else {
            continue;
        };
        let raw: Raw =
            serde_json::from_slice(&fs::read(provenance.student_raw_rollout.verify(&root)?)?)?;
        let plan: RolloutPlan = serde_json::from_slice(&fs::read(raw.plan.verify(&root)?)?)?;
        ensure!(
            plan.student_identity == provenance.student_checkpoint_sha256,
            "rollout provenance differs from its original student plan"
        );
        let assets: model::AssetManifest =
            serde_json::from_slice(&fs::read(plan.student_assets.verify(&root)?)?)?;
        let adapter = plan
            .student_checkpoint
            .as_ref()
            .map(|a| -> Result<ExperimentalCheckpointManifest> {
                Ok(serde_json::from_slice(&fs::read(a.verify(&root)?)?)?)
            })
            .transpose()?;
        if let Some(m) = &adapter {
            ensure!(
                m.adapter_sha256 == plan.student_identity
                    && m.base_sha256 == assets.files["model.safetensors"]
                    && m.spec.student_kind == plan.student_kind
                    && m.spec.student_base == plan.student_base,
                "rollout metadata differs from its immutable initial adapter"
            );
        } else {
            ensure!(
                plan.student_identity == assets.files["model.safetensors"],
                "base rollout has another student identity"
            );
        }
        let item = RolloutInitialMetadata {
            student_base: plan.student_base,
            student_kind: plan.student_kind,
            base_sha256: assets.files["model.safetensors"].clone(),
            checkpoint_sha256: plan.student_identity,
            checkpoint: plan.student_checkpoint,
            rank: adapter.as_ref().map(|m| m.adapter.rank),
            alpha: adapter.as_ref().map(|m| m.adapter.alpha),
        };
        if let Some(existing) = &metadata {
            ensure!(
                json_equal(existing, &item)?,
                "selected rollout teachers have different frozen initial student metadata"
            );
        } else {
            metadata = Some(item);
        }
        seen.insert(sample.teacher_id.clone());
    }
    ensure!(
        metadata.is_some() && teacher_ids.iter().all(|t| seen.contains(t)),
        "selected rollout teachers lack a homogeneous frozen initial student checkpoint"
    );
    metadata.context("rollout initial metadata absent")
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RolloutPlan {
    schema_version: u32,
    mode: RolloutMode,
    task: Task,
    input_bundle: Artefact,
    student_base: PathBuf,
    student_kind: DistillationStudent,
    student_assets: Artefact,
    student_checkpoint: Option<Artefact>,
    student_identity: String,
    student_runtime: RuntimeOptions,
    student_request: GenerationRequest,
    sandbox: SandboxConfig,
    preflight: Artefact,
    engine_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imported_student_attempt: Option<recovery::ImportedAttempt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content_role_policy: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Raw {
    schema_version: u32,
    evidence_kind: String,
    plan: Artefact,
    generation: Option<Generation>,
    generation_error: Option<String>,
    execution: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FeedbackPlan {
    schema_version: u32,
    initial_plan: Artefact,
    student_raw: Artefact,
    student_grade: Artefact,
    teacher_id: String,
    teacher_assets: Artefact,
    teacher_checkpoint: Option<Artefact>,
    runtime: RuntimeOptions,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context_tokens: Option<usize>,
    request: GenerationRequest,
    supporting_sample_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    schema_version: u32,
    evidence_kind: String,
    teacher_id: String,
    plan: Artefact,
    student_raw: Artefact,
    student_grade: Option<Artefact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unverified_student_grade: Option<Artefact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    student_content_eligibility: Option<ContentEligibility>,
    feedback_plan: Option<Artefact>,
    feedback_raw: Option<Artefact>,
    feedback_grade: Option<Artefact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unverified_feedback_grade: Option<Artefact>,
    errors: Vec<String>,
    success: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Grade {
    report: Option<Artefact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    unverified_report: Option<Artefact>,
    error: Option<String>,
}

fn validated_student(
    base: &Path,
    checkpoint: Option<&Path>,
    kind: DistillationStudent,
) -> Result<(
    model::AssetManifest,
    Option<ExperimentalCheckpointManifest>,
    String,
)> {
    let assets = match kind {
        DistillationStudent::ConventionalQwenLora => model::validate_assets(base)?,
        DistillationStudent::TernaryBitnet => hub::validate_bitnet_student_assets(base)?,
    };
    let manifest = checkpoint
        .map(|p| crate::distillation::checkpoint::validate_checkpoint(p, &assets))
        .transpose()?;
    if let Some(m) = &manifest {
        ensure!(
            m.spec.student_kind == kind && m.spec.student_base == base,
            "rollout student kind/base differs from checkpoint"
        );
    }
    let identity = manifest
        .as_ref()
        .map(|m| m.adapter_sha256.clone())
        .unwrap_or_else(|| assets.files["model.safetensors"].clone());
    Ok((assets, manifest, identity))
}

fn self_teacher(root: &Path, plan: &RolloutPlan) -> Result<TeacherEvidence> {
    let checkpoint = plan
        .student_checkpoint
        .as_ref()
        .map(|a| a.path.parent().context("self checkpoint parent absent"))
        .transpose()?;
    let (assets, initial, identity) =
        validated_student(&plan.student_base, checkpoint, plan.student_kind)?;
    ensure!(
        identity == plan.student_identity,
        "self teacher identity differs from recorded student"
    );
    let mut rights=LocalRightsDecision {eligible:true,scope:"verified local self-distillation; inherited original student and teacher licences".into(),licence:assets.licence.clone(),licence_artefacts:vec![capture(root,&plan.student_base.join("LICENSE"))?],inherited_restrictions:vec!["Retain original model licence, attribution, notices and all inherited teacher-output obligations.".into()],reasoning:"The actual frozen student is the teacher; output ownership alone does not remove derivative obligations.".into()};
    if let Some(m) = initial {
        let ancestors: Vec<TeacherEvidence> = serde_json::from_value(m.teacher_rights)?;
        ensure!(
            ancestors.iter().all(|t| t.rights.eligible),
            "self teacher has unresolved inherited rights"
        );
        for ancestor in ancestors {
            rights
                .licence_artefacts
                .extend(ancestor.rights.licence_artefacts);
            rights
                .inherited_restrictions
                .extend(ancestor.rights.inherited_restrictions);
        }
    }
    let baseline = model::validate_assets(Path::new("offline_bundle/base"))?;
    let tokeniser = assets.files["tokenizer.json"].clone();
    let absent =
        || SignalAvailability::absent(true, "No verified positive frozen-student generation yet");
    let capabilities = EvidenceCapabilities {
        complete_sequences: absent(),
        logits: SignalAvailability::absent(
            true,
            "Native checkpoint-aware target capture is supported; no target trace is inferred from text",
        ),
        hidden_states: absent(),
        attention: absent(),
        process: absent(),
        tool_trajectories: absent(),
        context: absent(),
        preferences: absent(),
        on_policy: absent(),
        self_distillation: absent(),
        tokeniser_sha256: tokeniser.clone(),
        student_tokeniser_sha256: baseline.files["tokenizer.json"].clone(),
        tokeniser_aligned: tokeniser == baseline.files["tokenizer.json"],
    };
    let id = format!("student-self-{}", &identity[..16]);
    Ok(TeacherEvidence {
        id: id.clone(),
        model: RegistryModel {
            id,
            model_id: assets.model_id.clone(),
            revision: assets.revision.clone(),
            directory: plan.student_base.clone(),
            architecture: match plan.student_kind {
                DistillationStudent::ConventionalQwenLora => "qwen2",
                DistillationStudent::TernaryBitnet => "bitnet",
            }
            .into(),
            role: "frozen_student_self_teacher".into(),
            licence: assets.licence,
            developer_id: "local-adapted-student".into(),
            weight_format: WeightFormat::Safetensors,
        },
        assets: plan.student_assets.clone(),
        adapter_checkpoint: plan.student_checkpoint.clone(),
        capabilities,
        rights,
    })
}

fn feedback_prompt(
    bundle: &EvidenceBundle,
    plan: &RolloutPlan,
    raw: &Raw,
    grade: &SandboxReport,
    supporting: &[String],
    role: Option<&ContentEligibility>,
) -> Result<String> {
    let generation = raw
        .generation
        .as_ref()
        .context("feedback lacks actual student text")?;
    let mut context = String::new();
    for id in supporting {
        let sample = bundle
            .samples
            .iter()
            .find(|s| &s.id == id)
            .context("self context sample absent")?;
        crate::curate::validate_train_origin(&sample.task)?;
        ensure!(
            sample.task.id != plan.task.id,
            "self context must be a different training task"
        );
        context.push_str(&format!(
            "Verified different training contract: {}\nVerified solution:\n{}\n",
            sample.task.specification, sample.completion
        ));
    }
    ensure!(
        plan.mode == RolloutMode::SelfDistillation || supporting.is_empty(),
        "ordinary on-policy feedback cannot silently add richer self context"
    );
    let mut feedback = serde_json::json!({"tool":"trusted_container_grader","run_id":grade.run_id,"passed":grade.passed,"cases_passed":grade.cases_passed,"cases_total":grade.cases_total,"trace_complete":grade.trace_complete,"quarantined":grade.quarantined,"student_generation_complete":generation.complete,"private_inputs_or_expected_outputs_exposed":false});
    if let Some(role) = role {
        feedback["content_role"] = serde_json::to_value(role.role)?;
        feedback["original_failure_is_not_a_success"] = serde_json::json!(true);
        feedback["bounded_diagnostic_untrusted_data"] = serde_json::json!(
            grade
                .failure
                .as_deref()
                .unwrap_or("")
                .chars()
                .take(4096)
                .collect::<String>()
        );
    }
    Ok(format!(
        "{context}New task:\n{}\nActual frozen student candidate:\n{}\nActual trusted tool feedback:\n{}\nCritique and revise the candidate. Put any short explanation in Rust // comments and return complete compilable Rust source only. Do not invent tool results.",
        plan.task.prompt(),
        generation.completion,
        feedback
    ))
}

async fn generate(
    saved: &Path,
    plan: &Artefact,
    request: GenerationRequest,
    mut loaded: Option<RecordedModel>,
    teacher_context: Option<(TeacherEvidence, usize)>,
    load: impl FnOnce() -> Result<RecordedModel> + Send + 'static,
) -> Result<(Raw, Option<RecordedModel>)> {
    if saved.exists() {
        let raw: Raw = serde_json::from_slice(&fs::read(saved)?)?;
        ensure!(
            json_equal(&raw.plan, plan)?,
            "saved rollout raw belongs to another immutable plan"
        );
        return Ok((raw, loaded));
    }
    let result = tokio::task::spawn_blocking(move || {
        if let Some((teacher, context_tokens)) = &teacher_context {
            validate_teacher_request(teacher, Some(*context_tokens), &request)?;
        }
        let mut model = match loaded.take() {
            Some(m) => m,
            None => load()?,
        };
        let execution = model.execution_metadata();
        let result = model.generate(request);
        Ok::<_, anyhow::Error>((model, execution, result))
    })
    .await?;
    let (generation, generation_error, execution, model) = match result {
        Ok((m, e, Ok(g))) => (Some(g), None, Some(e), Some(m)),
        Ok((m, e, Err(err))) => (None, Some(format!("{err:#}")), Some(e), Some(m)),
        Err(err) => (None, Some(format!("{err:#}")), None, None),
    };
    let raw = Raw {
        schema_version: 1,
        evidence_kind: "actual_checkpoint_bound_rollout_generation".into(),
        plan: plan.clone(),
        generation,
        generation_error,
        execution,
    };
    artefact::write_json(saved, &raw)?;
    Ok((raw, model))
}

async fn grade(
    path: &Path,
    task: &Task,
    raw: &Raw,
    config: &SandboxConfig,
    root: &Path,
) -> Result<Grade> {
    if path.exists() {
        return recovery::classify_grade(
            serde_json::from_slice(&fs::read(path)?)?,
            task,
            raw,
            config,
            root,
        );
    }
    let result = if let Some(g) = raw
        .generation
        .as_ref()
        .filter(|g| !g.completion.trim().is_empty())
    {
        match sandbox::grade(task, &crate::evaluate::extract_code(&g.completion), config).await {
            Ok(r) => Grade {
                report: Some(capture(
                    root,
                    &config.report_dir.join(format!("{}.json", r.run_id)),
                )?),
                unverified_report: None,
                error: None,
            },
            Err(e) => Grade {
                report: None,
                unverified_report: None,
                error: Some(format!("{e:#}")),
            },
        }
    } else {
        Grade {
            report: None,
            unverified_report: None,
            error: Some("No student/corrective output; this observation is a failure".into()),
        }
    };
    let result = recovery::classify_grade(result, task, raw, config, root)?;
    artefact::write_json(path, &result)?;
    Ok(result)
}

/// Collect a frozen current-student batch, then actual teacher corrections.
/// This is an on-policy collection round followed by offline optimiser updates;
/// it never claims that a teacher is queried at every optimisation step.
pub async fn collect_rollouts(options: RolloutCollectOptions) -> Result<EvidenceBundle> {
    ensure!(
        options.teacher_context_tokens.is_none() || options.mode == RolloutMode::OnPolicy,
        "explicit teacher context is supported for original GGUF on-policy correction teachers only"
    );
    for path in [&options.bundle, &options.output, &options.student_base] {
        relative(path)?;
    }
    if let Some(path) = &options.student_checkpoint {
        relative(path)?;
    }
    if let Some(source) = &options.reuse_student_attempts_from {
        relative(source)?;
        ensure!(
            source != &options.output && source.join("sandbox-preflight.json").exists(),
            "student continuation needs a different output and original successful preflight"
        );
    }
    ensure!(
        (options.output.starts_with("reports") || options.output.starts_with("data"))
            && (1..=48).contains(&options.maximum_tasks)
            && (1..=1024).contains(&options.max_tokens)
            && options.teacher_context_tokens.is_none_or(|context| (1..=4096).contains(&context) && options.max_tokens < context),
        "invalid bounded rollout collection policy"
    );
    let input = options.bundle.clone();
    let mut bundle = tokio::task::spawn_blocking(move || load_verified(&input)).await??;
    let inherited_records = bundle.rollout_records.len();
    let inherited_samples = bundle.samples.len();
    let root = std::env::current_dir()?.canonicalize()?;
    fs::create_dir_all(&options.output)?;
    let _lease =
        crate::engine::RunLease::acquire(&options.output.join(".rollout-collection.lock"))?;
    let base = options.student_base.clone();
    let cp = options.student_checkpoint.clone();
    let kind = options.student_kind;
    let (assets, _, identity) =
        tokio::task::spawn_blocking(move || validated_student(&base, cp.as_deref(), kind))
            .await??;
    let checkpoint = options
        .student_checkpoint
        .as_ref()
        .map(|p| capture(&root, &p.join("manifest.json")))
        .transpose()?;
    let student_assets = capture(&root, &options.student_base.join("assets.json"))?;
    ensure!(
        json_equal(
            &assets,
            &serde_json::from_slice::<model::AssetManifest>(&fs::read(
                student_assets.verify(&root)?
            )?)?
        )?,
        "rollout asset manifest differs from validated student"
    );
    let input_file = if options.bundle.is_dir() {
        options.bundle.join("bundle.json")
    } else {
        options.bundle.clone()
    };
    let input = capture(&root, &input_file)?;
    let mut policy = serde_json::json!({"schema_version":1,"input":input,"student_assets":student_assets,"student_checkpoint":checkpoint,"student_kind":kind,"mode":options.mode,"teacher_ids":options.teacher_ids,"student_runtime":options.student_runtime,"teacher_runtime":options.teacher_runtime,"maximum_tasks":options.maximum_tasks,"max_tokens":options.max_tokens,"seed":options.seed,"engine_sha256":artefact::file_hash(&std::env::current_exe()?)?,"reused_student_policy":options.reuse_student_attempts_from.as_ref().map(|p|capture(&root,&p.join("rollout-policy.json"))).transpose()?});
    if let Some(context_tokens) = options.teacher_context_tokens {
        policy["teacher_context_tokens"] = context_tokens.into();
    }
    let policy_path = options.output.join("rollout-policy.json");
    if policy_path.exists() {
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path)?)? == policy,
            "rollout policy changed; use a new output directory"
        );
    } else {
        artefact::write_json(&policy_path, &policy)?;
    }
    let mut sandbox = SandboxConfig {
        report_dir: options.output.join("sandbox"),
        ..Default::default()
    };
    let preflight_path = options.output.join("sandbox-preflight.json");
    let preflight = if preflight_path.exists() {
        serde_json::from_slice::<sandbox::PreflightReport>(&fs::read(&preflight_path)?)?
    } else {
        let p = sandbox::preflight(&sandbox).await?;
        artefact::write_json(&preflight_path, &p)?;
        p
    };
    ensure!(preflight.probes_passed, "rollout sandbox preflight failed");
    sandbox.expected_image_id = preflight.image_id;
    let preflight = capture(&root, &preflight_path)?;
    let tasks = train_tasks()
        .into_iter()
        .take(options.maximum_tasks)
        .collect::<Vec<_>>();
    let mut student = None;
    for task in &tasks {
        crate::curate::validate_train_origin(task)?;
        let directory = options.output.join(&task.id);
        fs::create_dir_all(&directory)?;
        let mut plan = RolloutPlan {
            schema_version: 1,
            mode: options.mode,
            task: task.clone(),
            input_bundle: input.clone(),
            student_base: options.student_base.clone(),
            student_kind: kind,
            student_assets: student_assets.clone(),
            student_checkpoint: checkpoint.clone(),
            student_identity: identity.clone(),
            student_runtime: options.student_runtime.clone(),
            student_request: GenerationRequest {
                prompt: task.prompt(),
                max_new_tokens: options.max_tokens,
                seed: options.seed,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: 120,
            },
            sandbox: sandbox.clone(),
            preflight: preflight.clone(),
            engine_sha256: policy["engine_sha256"]
                .as_str()
                .context("engine identity absent")?
                .into(),
            imported_student_attempt: None,
            content_role_policy: Some(CONTENT_ROLE_POLICY.into()),
        };
        if let Some(source) = &options.reuse_student_attempts_from {
            plan.imported_student_attempt = Some(recovery::import(&root, source, &plan)?);
        }
        let path = directory.join("student-plan.json");
        if path.exists() {
            ensure!(
                json_equal(
                    &plan,
                    &serde_json::from_slice::<RolloutPlan>(&fs::read(&path)?)?
                )?,
                "student rollout plan changed"
            );
        } else {
            artefact::write_json(&path, &plan)?;
        }
        let plan_artifact = capture(&root, &path)?;
        let base = options.student_base.clone();
        let cp = options.student_checkpoint.clone();
        let runtime = options.student_runtime.clone();
        let (raw, loaded) = if let Some(import) = &plan.imported_student_attempt {
            let original_plan = recovery::verify_import(&root, &plan, import)?;
            let raw: Raw = serde_json::from_slice(&fs::read(import.student_raw.verify(&root)?)?)?;
            let old: Grade =
                serde_json::from_slice(&fs::read(import.student_grade.verify(&root)?)?)?;
            let classified =
                recovery::classify_grade(old, task, &raw, &original_plan.sandbox, &root)?;
            let grade_path = directory.join("student-grade.json");
            if grade_path.exists() {
                ensure!(
                    json_equal(
                        &classified,
                        &serde_json::from_slice::<Grade>(&fs::read(&grade_path)?)?
                    )?,
                    "continued student grade classification changed"
                );
            } else {
                artefact::write_json(&grade_path, &classified)?;
            }
            (raw, student)
        } else {
            generate(
                &directory.join("student-raw.json"),
                &plan_artifact,
                plan.student_request.clone(),
                student,
                None,
                move || RecordedModel::student_optional(&base, cp.as_deref(), kind, runtime),
            )
            .await?
        };
        student = loaded;
        if plan.imported_student_attempt.is_none() {
            grade(
                &directory.join("student-grade.json"),
                task,
                &raw,
                &sandbox,
                &root,
            )
            .await?;
        }
        eprintln!(
            "student rollout task={} complete={}",
            task.id,
            raw.generation.as_ref().is_some_and(|g| g.complete)
        );
    }
    if let Some(model) = &student {
        model.device().synchronize()?;
    }
    drop(student);
    let template: RolloutPlan = serde_json::from_slice(&fs::read(
        options.output.join(&tasks[0].id).join("student-plan.json"),
    )?)?;
    let teachers = if options.mode == RolloutMode::SelfDistillation {
        ensure!(
            options.teacher_ids.is_empty(),
            "self-distillation selects the actual frozen student, not an external teacher ID"
        );
        vec![self_teacher(&root, &template)?]
    } else {
        ensure!(
            !options.teacher_ids.is_empty(),
            "on-policy collection needs named real teachers"
        );
        options
            .teacher_ids
            .iter()
            .map(|id| {
                bundle
                    .teachers
                    .iter()
                    .find(|t| &t.id == id)
                    .cloned()
                    .context("on-policy teacher absent from input evidence")
            })
            .collect::<Result<Vec<_>>>()?
    };
    let expected_records = teachers.len() * tasks.len();
    for teacher in teachers {
        let mut loaded = None;
        for task in &tasks {
            let directory = options.output.join(&task.id);
            let record_dir = directory.join(&teacher.id);
            fs::create_dir_all(&record_dir)?;
            let record_path = record_dir.join("record.json");
            if record_path.exists() {
                append_verified(&root, &mut bundle, &capture(&root, &record_path)?)?;
                continue;
            }
            let plan_artifact = capture(&root, &directory.join("student-plan.json"))?;
            let plan: RolloutPlan =
                serde_json::from_slice(&fs::read(plan_artifact.verify(&root)?)?)?;
            let student_raw = if let Some(import) = &plan.imported_student_attempt {
                import.student_raw.clone()
            } else {
                capture(&root, &directory.join("student-raw.json"))?
            };
            let raw: Raw = serde_json::from_slice(&fs::read(student_raw.verify(&root)?)?)?;
            let original_plan = plan
                .imported_student_attempt
                .as_ref()
                .map(|a| recovery::verify_import(&root, &plan, a))
                .transpose()?;
            let initial_sandbox = original_plan
                .as_ref()
                .map(|p| &p.sandbox)
                .unwrap_or(&sandbox);
            let observed = recovery::classify_grade(
                serde_json::from_slice(&fs::read(directory.join("student-grade.json"))?)?,
                task,
                &raw,
                initial_sandbox,
                &root,
            )?;
            let content_role = observed
                .report
                .as_ref()
                .zip(raw.generation.as_ref())
                .map(|(report, g)| {
                    classify_content(
                        &root,
                        report,
                        initial_sandbox,
                        task,
                        &g.completion,
                        g.complete,
                    )
                })
                .transpose()?;
            let mut record = Record {
                schema_version: 1,
                evidence_kind: "actual_checkpoint_bound_rollout_observation".into(),
                teacher_id: teacher.id.clone(),
                plan: plan_artifact.clone(),
                student_raw: student_raw.clone(),
                student_grade: observed.report.clone(),
                unverified_student_grade: observed.unverified_report.clone(),
                student_content_eligibility: content_role.clone(),
                feedback_plan: None,
                feedback_raw: None,
                feedback_grade: None,
                unverified_feedback_grade: None,
                errors: observed.error.into_iter().collect(),
                success: false,
            };
            if let Some(initial_grade) = &record.student_grade {
                let trusted: SandboxReport =
                    serde_json::from_slice(&fs::read(initial_grade.verify(&root)?)?)?;
                if content_role
                    .as_ref()
                    .is_some_and(ContentEligibility::feedback_allowed)
                    && raw
                        .generation
                        .as_ref()
                        .is_some_and(|g| !g.completion.trim().is_empty())
                {
                    let supporting = if plan.mode == RolloutMode::SelfDistillation {
                        bundle
                            .samples
                            .iter()
                            .filter(|s| {
                                s.task.id != task.id
                                    && s.task.source_id == task.source_id
                                    && s.student_rollout.is_none()
                            })
                            .take(1)
                            .map(|s| s.id.clone())
                            .collect()
                    } else {
                        vec![]
                    };
                    let text = feedback_prompt(
                        &bundle,
                        &plan,
                        &raw,
                        &trusted,
                        &supporting,
                        content_role.as_ref(),
                    )?;
                    let fp = FeedbackPlan {
                        schema_version: 1,
                        initial_plan: plan_artifact,
                        student_raw,
                        student_grade: initial_grade.clone(),
                        teacher_id: teacher.id.clone(),
                        teacher_assets: teacher.assets.clone(),
                        teacher_checkpoint: teacher.adapter_checkpoint.clone(),
                        runtime: options.teacher_runtime.clone(),
                        context_tokens: options.teacher_context_tokens,
                        request: GenerationRequest {
                            prompt: text,
                            max_new_tokens: options.max_tokens,
                            seed: options.seed,
                            temperature: 0.2,
                            top_p: 0.95,
                            timeout_seconds: 120,
                        },
                        supporting_sample_ids: supporting,
                    };
                    let fp_path = record_dir.join("feedback-plan.json");
                    if fp_path.exists() {
                        ensure!(
                            json_equal(
                                &fp,
                                &serde_json::from_slice::<FeedbackPlan>(&fs::read(&fp_path)?)?
                            )?,
                            "feedback request changed after interruption"
                        );
                    } else {
                        artefact::write_json(&fp_path, &fp)?;
                    }
                    let fp_artifact = capture(&root, &fp_path)?;
                    let model = teacher.clone();
                    let runtime = options.teacher_runtime.clone();
                    let context_tokens = options.teacher_context_tokens;
                    let (feedback, next) = generate(
                        &record_dir.join("feedback-raw.json"),
                        &fp_artifact,
                        fp.request.clone(),
                        loaded,
                        context_tokens.map(|context| (teacher.clone(), context)),
                        move || {
                            RecordedModel::teacher_with_context(&model, runtime, context_tokens)
                        },
                    )
                    .await?;
                    loaded = next;
                    let final_grade = grade(
                        &record_dir.join("feedback-grade.json"),
                        task,
                        &feedback,
                        &sandbox,
                        &root,
                    )
                    .await?;
                    record.success = if let Some(a) = &final_grade.report {
                        let r: SandboxReport =
                            serde_json::from_slice(&fs::read(a.verify(&root)?)?)?;
                        r.passed
                            && !r.quarantined
                            && r.trace_complete
                            && feedback
                                .generation
                                .as_ref()
                                .is_some_and(|g| g.complete && !g.completion.trim().is_empty())
                    } else {
                        false
                    };
                    record.feedback_plan = Some(fp_artifact);
                    record.feedback_raw =
                        Some(capture(&root, &record_dir.join("feedback-raw.json"))?);
                    record.feedback_grade = final_grade.report;
                    record.unverified_feedback_grade = final_grade.unverified_report;
                    record.errors.extend(final_grade.error);
                } else {
                    record.errors.push("Unsafe/quarantined or empty student attempt cannot become a teacher-feedback prompt".into());
                }
            }
            artefact::write_json(&record_path, &record)?;
            append_verified(&root, &mut bundle, &capture(&root, &record_path)?)?;
            eprintln!(
                "rollout revision mode={:?} teacher={} task={} accepted={}",
                options.mode, teacher.id, task.id, record.success
            );
        }
        if let Some(model) = &loaded {
            model.device().synchronize()?;
        }
    }
    artefact::write_json(&options.output.join("bundle.json"), &bundle)?;
    artefact::write_json(
        &options.output.join("manifest.json"),
        &serde_json::json!({"schema_version":1,"bundle":capture(&root,&options.output.join("bundle.json"))?,"train_corpus_sha256":bundle.train_corpus_sha256,"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"promotable":false}),
    )?;
    let mut report = health::batch_report(
        &root,
        &bundle.rollout_records[inherited_records..],
        expected_records,
        bundle.samples[inherited_samples..]
            .iter()
            .filter(|sample| sample.student_rollout.is_some())
            .count(),
    )?;
    report["mode"] = serde_json::to_value(options.mode)?;
    report["tasks"] = serde_json::json!(tasks.len());
    report["inherited_rollout_records"] = serde_json::json!(inherited_records);
    report["inherited_samples"] = serde_json::json!(inherited_samples);
    report["student_checkpoint_sha256"] = serde_json::json!(identity);
    artefact::write_json(&options.output.join("rollout-report.json"), &report)?;
    Ok(bundle)
}

#[cfg(test)]
mod context_tests {
    use super::*;

    #[test]
    fn correction_context_omission_preserves_legacy_feedback_plan_bytes() -> Result<()> {
        let file = Artefact {
            path: "labelled/unused".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        let mut plan = FeedbackPlan {
            schema_version: 1,
            initial_plan: file.clone(),
            student_raw: file.clone(),
            student_grade: file.clone(),
            teacher_id: "synthetic".into(),
            teacher_assets: file,
            teacher_checkpoint: None,
            runtime: RuntimeOptions::default(),
            context_tokens: None,
            request: GenerationRequest {
                prompt: "synthetic feedback".into(),
                max_new_tokens: 768,
                seed: 1729,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: 120,
            },
            supporting_sample_ids: vec![],
        };
        let legacy = serde_json::to_vec(&plan)?;
        ensure!(serde_json::to_value(&plan)?.get("context_tokens").is_none());
        let restored: FeedbackPlan = serde_json::from_slice(&legacy)?;
        ensure!(serde_json::to_vec(&restored)? == legacy);
        plan.context_tokens = Some(2048);
        ensure!(
            !json_equal(&plan, &restored)?,
            "correction cap must change the frozen request"
        );
        Ok(())
    }
}
