//! Additional genuine teacher generations with train-only context and feedback.
//! Saved raw output always precedes grading and is reused after interruption.
use super::*;
use crate::{
    model::{Generation, GenerationRequest},
    sandbox::{self, SandboxConfig},
};
use std::os::fd::AsRawFd;

pub struct ContentCollectOptions {
    pub bundle: PathBuf,
    pub output: PathBuf,
    pub content: Vec<ContentKind>,
    pub runtime: RuntimeOptions,
    pub max_tokens: usize,
    pub seed: u64,
    pub maximum_tasks: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ContentPlan {
    schema_version: u32,
    task_id: String,
    teacher_id: String,
    content: ContentKind,
    request: GenerationRequest,
    runtime: RuntimeOptions,
    teacher_assets: Artefact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    teacher_checkpoint: Option<Artefact>,
    supporting_sample_ids: Vec<String>,
    feedback_candidate: Option<Artefact>,
    feedback_summary: Option<String>,
    sandbox: SandboxConfig,
    preflight: Artefact,
    engine_binary_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ContentRaw {
    schema_version: u32,
    evidence_kind: String,
    plan: Artefact,
    generation: Option<Generation>,
    generation_error: Option<String>,
    execution: Option<serde_json::Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ContentRecord {
    schema_version: u32,
    evidence_kind: String,
    plan: Artefact,
    raw_generation: Artefact,
    sandbox_report: Option<Artefact>,
    grading_error: Option<String>,
    success: bool,
}
struct Lease(fs::File);
impl Lease {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        // SAFETY: flock uses a retained live descriptor and no Rust pointers.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another content collector owns this directory"
        );
        Ok(Self(file))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        // SAFETY: the file remains live until this lease has been unlocked.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}
type PromptParts = (String, Vec<String>, Option<Artefact>, Option<String>);
fn prompt(
    root: &Path,
    bundle: &EvidenceBundle,
    teacher: &TeacherEvidence,
    task: &Task,
    content: ContentKind,
) -> Result<PromptParts> {
    match content {
        ContentKind::Process => Ok((
            format!(
                "{}\nStart with a short implementation plan written as Rust // comments. Then provide the implementation. Return compilable Rust source only. Do not claim tests were run.",
                task.prompt()
            ),
            vec![],
            None,
            None,
        )),
        ContentKind::Context => {
            let context = bundle
                .samples
                .iter()
                .find(|s| {
                    s.content == ContentKind::CompleteSequence
                        && s.teacher_id == teacher.id
                        && s.task.source_id == task.source_id
                        && s.task.id != task.id
                })
                .context("no verified same-source training example for contextual generation")?;
            Ok((
                format!(
                    "A verified example from the same training source follows. It is a different task; implement only the new contract.\nExample contract: {}\nExample solution:\n{}\nNew task:\n{}",
                    context.task.specification,
                    context.completion,
                    task.prompt()
                ),
                vec![context.id.clone()],
                None,
                None,
            ))
        }
        ContentKind::ToolTrajectory => {
            let mut permitted = Vec::new();
            for candidate in bundle
                .candidates
                .iter()
                .filter(|c| c.task_id == task.id && c.teacher_id == teacher.id)
            {
                if collect::candidate_eligibility(root, candidate, task)?.feedback_allowed() {
                    permitted.push(candidate);
                }
            }
            let previous = permitted
                .into_iter()
                .min_by_key(|c| c.passed)
                .context("no safe original teacher candidate for a real feedback trajectory")?;
            let grade: SandboxReport =
                serde_json::from_slice(&fs::read(&previous.sandbox_report.path)?)?;
            let feedback = serde_json::to_string(
                &serde_json::json!({"tool":"trusted_container_grader","run_id":grade.run_id,"passed":grade.passed,"cases_passed":grade.cases_passed,"cases_total":grade.cases_total,"quarantined":grade.quarantined,"trace_complete":grade.trace_complete,"private_inputs_or_expected_outputs_exposed":false}),
            )?;
            Ok((
                format!(
                    "{}\nYour previous candidate:\n{}\nActual trusted tool feedback:\n{}\nReview and revise the implementation. Return only complete Rust source. Do not invent tool calls or test results.",
                    task.prompt(),
                    previous.completion,
                    feedback
                ),
                vec![],
                Some(previous.observation.clone()),
                Some(feedback),
            ))
        }
        _ => anyhow::bail!("generated variants accept process, context and tool_trajectory only"),
    }
}

/// The policy and each request are frozen before inference. Completed raw
/// generations and grades survive restarts; a new policy needs a new directory.
pub async fn collect_content(options: ContentCollectOptions) -> Result<EvidenceBundle> {
    let output = options.output.clone();
    let result = collect_content_inner(options).await;
    if let Err(error) = &result
        && output.is_dir()
        && !output.join("content-report.json").exists()
    {
        artefact::write_json(
            &output.join("content-report.json"),
            &serde_json::json!({
                "schema_version":1,"evidence_kind":"current_batch_teacher_content_collection",
                "healthy_complete":false,"collection_completed":false,"production_eligible":false,
                "failure":format!("{error:#}"),
                "limitation":"Earlier inherited examples cannot establish a successful current collection."
            }),
        )?;
    }
    result
}

async fn collect_content_inner(options: ContentCollectOptions) -> Result<EvidenceBundle> {
    relative(&options.output)?;
    ensure!(
        options.output.starts_with("reports") || options.output.starts_with("data"),
        "content output must remain under reports or data"
    );
    ensure!(
        (1..=48).contains(&options.maximum_tasks) && (1..=1024).contains(&options.max_tokens),
        "invalid bounded content collection"
    );
    ensure!(
        !options.content.is_empty()
            && options.content.iter().all(|c| matches!(
                c,
                ContentKind::Process | ContentKind::Context | ContentKind::ToolTrajectory
            )),
        "unsupported generated content type"
    );
    let input = options.bundle.clone();
    let mut bundle = tokio::task::spawn_blocking(move || load_verified(&input)).await??;
    ensure!(
        bundle.rollout_records.is_empty(),
        "content variants after student/self rollouts require ordered dependency replay, which is not implemented; no base-only fallback is permitted"
    );
    fs::create_dir_all(&options.output)?;
    let _lease = Lease::acquire(&options.output.join(".content-collection.lock"))?;
    let root = std::env::current_dir()?.canonicalize()?;
    let input_path = if options.bundle.is_dir() {
        options.bundle.join("bundle.json")
    } else {
        options.bundle.clone()
    };
    ensure!(
        input_path.canonicalize()?.parent() != Some(options.output.canonicalize()?.as_path()),
        "content output must not overwrite its verified input bundle"
    );
    let policy_path = options.output.join("content-policy.json");
    let mut policy = serde_json::json!({"schema_version":1,"input":capture(&root,&input_path)?,"content":options.content,"runtime":options.runtime,"max_tokens":options.max_tokens,"seed":options.seed,"maximum_tasks":options.maximum_tasks,"teachers":bundle.teachers.iter().map(|t|&t.id).collect::<Vec<_>>()});
    // New protocols bind their executable before any variant is generated.
    // Historical policy bytes retain their original serialised shape; their
    // individual generation plans already bind the original engine identity.
    if !policy_path.exists()
        || serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path)?)?
            .get("engine_binary_sha256")
            .is_some()
    {
        policy["engine_binary_sha256"] = artefact::file_hash(&std::env::current_exe()?)?.into();
    }
    if policy_path.exists() {
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path)?)? == policy,
            "saved policy changed; use a new output directory for a fresh study"
        );
    } else {
        artefact::write_json(&policy_path, &policy)?;
    }
    let mut sandbox = SandboxConfig {
        report_dir: options.output.join("sandbox"),
        ..Default::default()
    };
    let preflight_path = options.output.join("sandbox-preflight.json");
    if preflight_path.exists() {
        let saved: sandbox::PreflightReport = serde_json::from_slice(&fs::read(&preflight_path)?)?;
        ensure!(saved.probes_passed, "saved sandbox preflight failed");
        sandbox.expected_image_id = saved.image_id;
    } else {
        let observed = sandbox::preflight(&sandbox).await?;
        ensure!(
            observed.probes_passed,
            "content sandbox controls unavailable"
        );
        sandbox.expected_image_id = observed.image_id.clone();
        artefact::write_json(&preflight_path, &observed)?;
    }
    let preflight = capture(&root, &preflight_path)?;
    let engine = artefact::file_hash(&std::env::current_exe()?)?;
    let mut unavailable = Vec::new();
    for teacher in bundle.teachers.clone() {
        let mut loaded: Option<RecordedModel> = None;
        for task in train_tasks().into_iter().take(options.maximum_tasks) {
            for content in &options.content {
                let tag = serde_json::to_value(content)?
                    .as_str()
                    .context("content tag absent")?
                    .to_owned();
                let directory = options.output.join(&teacher.id).join(&task.id).join(tag);
                fs::create_dir_all(&directory)?;
                let record_path = directory.join("record.json");
                if record_path.exists() {
                    append_verified(&root, &mut bundle, &capture(&root, &record_path)?)?;
                    continue;
                }
                let (text, supporting_sample_ids, feedback_candidate, feedback_summary) =
                    match prompt(&root, &bundle, &teacher, &task, *content) {
                        Ok(value) => value,
                        Err(error) => {
                            unavailable.push(serde_json::json!({"teacher":teacher.id,"task":task.id,"content":content,"reason":error.to_string()}));
                            continue;
                        }
                    };
                let plan = ContentPlan {
                    schema_version: 1,
                    task_id: task.id.clone(),
                    teacher_id: teacher.id.clone(),
                    content: *content,
                    request: GenerationRequest {
                        prompt: text,
                        max_new_tokens: options.max_tokens,
                        seed: options.seed,
                        temperature: 0.2,
                        top_p: 0.95,
                        timeout_seconds: 120,
                    },
                    runtime: options.runtime.clone(),
                    teacher_assets: teacher.assets.clone(),
                    teacher_checkpoint: teacher.adapter_checkpoint.clone(),
                    supporting_sample_ids,
                    feedback_candidate,
                    feedback_summary,
                    sandbox: sandbox.clone(),
                    preflight: preflight.clone(),
                    engine_binary_sha256: engine.clone(),
                };
                let plan_path = directory.join("plan.json");
                let raw_path = directory.join("raw-generation.json");
                let plan = if plan_path.exists() {
                    let old: ContentPlan = serde_json::from_slice(&fs::read(&plan_path)?)?;
                    let mut expected = plan.clone();
                    if raw_path.exists() {
                        // An already-recorded generation retains the binary
                        // identity that actually produced it. Regrading after
                        // an unrelated rebuild must not regenerate that work.
                        expected.engine_binary_sha256 = old.engine_binary_sha256.clone();
                    }
                    ensure!(
                        json_equal(&old, &expected)?,
                        "saved content request changed; refusing implicit resampling"
                    );
                    old
                } else {
                    artefact::write_json(&plan_path, &plan)?;
                    plan
                };
                let plan_artifact = capture(&root, &plan_path)?;
                let raw: ContentRaw = if raw_path.exists() {
                    let saved: ContentRaw = serde_json::from_slice(&fs::read(&raw_path)?)?;
                    ensure!(
                        json_equal(&saved.plan, &plan_artifact)?,
                        "saved raw content belongs to another request"
                    );
                    saved
                } else {
                    let teacher_spec = teacher.clone();
                    let runtime = options.runtime.clone();
                    let request = plan.request.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        let mut model = match loaded {
                            Some(model) => model,
                            None => RecordedModel::teacher(&teacher_spec, runtime)?,
                        };
                        let execution = model.execution_metadata();
                        let result = model.generate(request);
                        Ok::<_, anyhow::Error>((model, execution, result))
                    })
                    .await?;
                    let (generation, generation_error, execution) = match result {
                        Ok((model, execution, result)) => {
                            loaded = Some(model);
                            match result {
                                Ok(g) => (Some(g), None, Some(execution)),
                                Err(e) => (None, Some(format!("{e:#}")), Some(execution)),
                            }
                        }
                        Err(e) => {
                            loaded = None;
                            (None, Some(format!("{e:#}")), None)
                        }
                    };
                    let raw = ContentRaw {
                        schema_version: 1,
                        evidence_kind: "actual_local_teacher_content_generation".into(),
                        plan: plan_artifact.clone(),
                        generation,
                        generation_error,
                        execution,
                    };
                    artefact::write_json(&raw_path, &raw)?;
                    raw
                };
                let (grade, grading_error) = if let Some(g) = raw
                    .generation
                    .as_ref()
                    .filter(|g| g.complete && !g.completion.trim().is_empty())
                {
                    let link = directory.join("grade-link.json");
                    if link.exists() {
                        (
                            Some(serde_json::from_slice::<Artefact>(&fs::read(&link)?)?),
                            None,
                        )
                    } else {
                        match sandbox::grade(
                            &task,
                            &crate::evaluate::extract_code(&g.completion),
                            &sandbox,
                        )
                        .await
                        {
                            Ok(report) => {
                                let a = capture(
                                    &root,
                                    &sandbox.report_dir.join(format!("{}.json", report.run_id)),
                                )?;
                                artefact::write_json(&link, &a)?;
                                (Some(a), None)
                            }
                            Err(e) => (None, Some(format!("{e:#}"))),
                        }
                    }
                } else {
                    (None, None)
                };
                let success = if let Some(a) = &grade {
                    let r: SandboxReport = serde_json::from_slice(&fs::read(a.verify(&root)?)?)?;
                    r.passed
                        && !r.quarantined
                        && r.trace_complete
                        && (*content != ContentKind::Process
                            || raw.generation.as_ref().is_some_and(|g| {
                                g.completion
                                    .lines()
                                    .any(|l| l.trim_start().starts_with("//"))
                            }))
                } else {
                    false
                };
                let record = ContentRecord {
                    schema_version: 1,
                    evidence_kind: "actual_local_teacher_content_observation".into(),
                    plan: plan_artifact,
                    raw_generation: capture(&root, &raw_path)?,
                    sandbox_report: grade,
                    grading_error,
                    success,
                };
                artefact::write_json(&record_path, &record)?;
                append_verified(&root, &mut bundle, &capture(&root, &record_path)?)?;
                eprintln!(
                    "teacher content progress teacher={} task={} content={:?} accepted={} total_samples={}",
                    teacher.id,
                    task.id,
                    content,
                    success,
                    bundle.samples.len()
                );
            }
        }
    }
    artefact::write_json(&options.output.join("unavailable.json"), &unavailable)?;
    artefact::write_json(&options.output.join("bundle.json"), &bundle)?;
    artefact::write_json(
        &options.output.join("manifest.json"),
        &serde_json::json!({"schema_version":1,"bundle":capture(&root,&options.output.join("bundle.json"))?,"train_corpus_sha256":bundle.train_corpus_sha256,"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"promotable":false}),
    )?;
    write_content_report(&root, &options, &bundle, &unavailable)?;
    Ok(bundle)
}

/// Count only observations produced by this policy/output, including validated
/// resumed records. Inherited positives cannot conceal a failed new batch.
fn write_content_report(
    root: &Path,
    options: &ContentCollectOptions,
    bundle: &EvidenceBundle,
    unavailable: &[serde_json::Value],
) -> Result<()> {
    let current = bundle
        .content_records
        .iter()
        .filter(|file| file.path.starts_with(&options.output))
        .collect::<Vec<_>>();
    let mut generation_errors = 0;
    let mut grading_errors = 0;
    let mut complete_nonempty_generations = 0;
    let mut incomplete_or_empty_generations = 0;
    let mut infrastructure_failures = 0;
    let mut passing_records = 0;
    for artifact in &current {
        let record: ContentRecord = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
        let raw: ContentRaw =
            serde_json::from_slice(&fs::read(record.raw_generation.verify(root)?)?)?;
        generation_errors += usize::from(raw.generation_error.is_some());
        grading_errors += usize::from(record.grading_error.is_some());
        if let Some(generation) = raw.generation {
            if generation.complete && !generation.completion.trim().is_empty() {
                complete_nonempty_generations += 1;
            } else {
                incomplete_or_empty_generations += 1;
            }
        }
        if let Some(grade) = record.sandbox_report {
            let report: SandboxReport = serde_json::from_slice(&fs::read(grade.verify(root)?)?)?;
            // Compiler diagnostics are legitimate negative candidate outcomes;
            // a missing trace for another reason cannot establish healthy QA.
            let compilation_failed = report.failure.as_deref().is_some_and(|text| {
                text.starts_with("compilation failed")
                    || text.starts_with("rustc compilation failed")
            });
            infrastructure_failures += usize::from(!report.trace_complete && !compilation_failed);
        }
        passing_records += usize::from(record.success);
    }
    let current_positive_samples = bundle
        .samples
        .iter()
        .filter(|sample| sample.observation.path.starts_with(&options.output))
        .count();
    let current_positive_ids = bundle
        .samples
        .iter()
        .filter(|sample| sample.observation.path.starts_with(&options.output))
        .map(|sample| sample.id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let current_preference_pairs = bundle
        .preferences
        .iter()
        .filter(|pair| current_positive_ids.contains(pair.chosen_sample_id.as_str()))
        .count();
    let requested = bundle.teachers.len() * options.maximum_tasks * options.content.len();
    let healthy_complete = current.len() == requested
        && !current.is_empty()
        && unavailable.is_empty()
        && generation_errors == 0
        && grading_errors == 0
        && infrastructure_failures == 0
        && incomplete_or_empty_generations == 0
        && current_positive_samples > 0;
    artefact::write_json(
        &options.output.join("content-report.json"),
        &serde_json::json!({
            "schema_version":1,"evidence_kind":"current_batch_teacher_content_collection",
            "policy":capture(root,&options.output.join("content-policy.json"))?,
            "bundle":capture(root,&options.output.join("bundle.json"))?,
            "collection_completed":true,"healthy_complete":healthy_complete,
            "requested_content_requests":requested,"completed_current_content_records":current.len(),
            "unavailable_requests":unavailable.len(),"generation_errors":generation_errors,
            "grading_errors":grading_errors,"infrastructure_failures":infrastructure_failures,
            "complete_nonempty_generations":complete_nonempty_generations,
            "incomplete_or_empty_generations":incomplete_or_empty_generations,
            "passing_current_records":passing_records,"current_positive_samples":current_positive_samples,
            "current_preference_pairs":current_preference_pairs,
            "inherited_positive_samples":bundle.samples.len()-current_positive_samples,
            "total_positive_samples":bundle.samples.len(),"production_eligible":false,
            "semantics":"Current batch includes validated resumed work from this output only. Inherited positives never establish successful new content collection; passing samples do not establish model learning."
        }),
    )
}

pub(super) fn append_verified(
    root: &Path,
    bundle: &mut EvidenceBundle,
    artifact: &Artefact,
) -> Result<()> {
    ensure!(
        !bundle
            .content_records
            .iter()
            .any(|a| a.path == artifact.path),
        "duplicate content record"
    );
    let record: ContentRecord = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
    let plan: ContentPlan = serde_json::from_slice(&fs::read(record.plan.verify(root)?)?)?;
    let raw: ContentRaw = serde_json::from_slice(&fs::read(record.raw_generation.verify(root)?)?)?;
    ensure!(
        record.schema_version == 1
            && record.evidence_kind == "actual_local_teacher_content_observation"
            && plan.schema_version == 1
            && raw.schema_version == 1
            && raw.evidence_kind == "actual_local_teacher_content_generation"
            && json_equal(&raw.plan, &record.plan)?,
        "unsupported or unbound content evidence"
    );
    let teacher = bundle
        .teachers
        .iter()
        .find(|t| t.id == plan.teacher_id)
        .context("unknown content teacher")?
        .clone();
    let task = train_tasks()
        .into_iter()
        .find(|t| t.id == plan.task_id)
        .context("content task is not an original train task")?;
    let (text, supporting, feedback, summary) =
        prompt(root, bundle, &teacher, &task, plan.content)?;
    ensure!(
        plan.request.prompt == text
            && plan.supporting_sample_ids == supporting
            && json_equal(&plan.feedback_candidate, &feedback)?
            && plan.feedback_summary == summary
            && json_equal(&plan.teacher_assets, &teacher.assets)?
            && json_equal(&plan.teacher_checkpoint, &teacher.adapter_checkpoint)?,
        "content context/feedback lineage changed"
    );
    let preflight: sandbox::PreflightReport =
        serde_json::from_slice(&fs::read(plan.preflight.verify(root)?)?)?;
    ensure!(
        preflight.probes_passed
            && preflight.image_id == plan.sandbox.expected_image_id
            && preflight.policy_sha256 == sandbox::policy_sha256(&plan.sandbox)?,
        "content sandbox differs from preflight"
    );
    let mut success = false;
    if let Some(generation) = &raw.generation {
        ensure!(
            raw.generation_error.is_none(),
            "content generation also claims an error"
        );
        let execution = raw
            .execution
            .as_ref()
            .context("content lacks actual execution metadata")?;
        let assets: model::AssetManifest =
            serde_json::from_slice(&fs::read(teacher.assets.verify(root)?)?)?;
        let checkpoint = if let Some(adapter) = &teacher.adapter_checkpoint {
            let manifest: crate::distillation::checkpoint::ExperimentalCheckpointManifest =
                serde_json::from_slice(&fs::read(adapter.verify(root)?)?)?;
            manifest.adapter_sha256
        } else if teacher.model.architecture == "bitnet" {
            assets
                .files
                .get("model.safetensors")
                .context("BitNet teacher tensor identity absent")?
                .clone()
        } else {
            comparative::comparative_checkpoint_id(&assets)?
        };
        let requested_matches = execution["requested"] == serde_json::to_value(&plan.runtime)?
            || execution["requested"].is_null()
                && teacher.model.architecture == "bitnet"
                && execution["architecture"] == "bitnet"
                && execution["actual_device"] == "Cpu"
                && execution["actual_dtype"] == "F32"
                && plan.runtime.device == model::ExecutionDevice::Cpu
                && plan.runtime.dtype == model::InferenceDtype::F32;
        ensure!(
            generation.checkpoint_id == checkpoint
                && execution["checkpoint_id"] == checkpoint
                && execution["engine"] == "native_rust_candle"
                && requested_matches
                && execution["cpu_fallback"] == false
                && execution["model_id"] == teacher.model.model_id
                && execution["revision"] == teacher.model.revision,
            "content model/runtime binding failed"
        );
        ensure!(
            generation.complete == (generation.finish_reason == "stop")
                && generation.input_tokens > 0
                && generation.output_tokens <= plan.request.max_new_tokens,
            "invalid generation token/termination evidence"
        );
        if let Some(grade) = &record.sandbox_report {
            ensure!(
                generation.complete && !generation.completion.trim().is_empty(),
                "incomplete content was graded"
            );
            let report: SandboxReport = serde_json::from_slice(&fs::read(grade.verify(root)?)?)?;
            sandbox::verify_report(
                &report,
                &plan.sandbox,
                &task,
                &crate::evaluate::extract_code(&generation.completion),
            )?;
            success = report.passed
                && !report.quarantined
                && report.trace_complete
                && (plan.content != ContentKind::Process
                    || generation
                        .completion
                        .lines()
                        .any(|line| line.trim_start().starts_with("//")));
            if success {
                let id = artefact::hash(
                    format!("{}:{}:{:?}", artifact.sha256, task.id, plan.content).as_bytes(),
                );
                bundle.samples.push(EvidenceSample {
                    id,
                    task,
                    teacher_id: teacher.id.clone(),
                    prompt: plan.request.prompt,
                    completion: generation.completion.clone(),
                    content: plan.content,
                    observation: artifact.clone(),
                    raw_generation: record.raw_generation.clone(),
                    sandbox_report: grade.clone(),
                    audit: capture(root, &report.audit_path)?,
                    sequence_sha256: artefact::hash(generation.completion.as_bytes()),
                    provenance_kind: "actual_local_teacher_content_with_verified_container_grade"
                        .into(),
                    student_rollout: None,
                });
                let chosen = bundle
                    .samples
                    .last()
                    .context("accepted content sample absent")?
                    .clone();
                for rejected in bundle.candidates.iter().filter(|c| {
                    c.task_id == chosen.task.id && !c.passed && !c.completion.trim().is_empty()
                }) {
                    let role = collect::candidate_eligibility(root, rejected, &chosen.task)?;
                    if !role.negative_allowed() {
                        continue;
                    }
                    let id = artefact::hash(
                        format!("{}:{}", chosen.id, rejected.observation.sha256).as_bytes(),
                    );
                    if bundle.preferences.iter().any(|pair| pair.id == id) {
                        continue;
                    }
                    let (preference_prompt, prompt_transform) = preference_conditioning(
                        &chosen.task,
                        &chosen.prompt,
                        &chosen.task.prompt(),
                    );
                    bundle.preferences.push(PreferencePair{id,task_id:chosen.task.id.clone(),chosen_sample_id:chosen.id.clone(),rejected:rejected.clone(),rejected_eligibility:Some(role),preference_prompt:Some(preference_prompt),prompt_transform:Some(prompt_transform),reason:"Additional process/context/tool-feedback answer passes the same original train task; rejected has a verified oracle-negative role and retains its original task prompt/quarantine; canonical task conditioning is explicitly bound to both actual source prompts".into()});
                }
                let teacher = bundle
                    .teachers
                    .iter_mut()
                    .find(|t| t.id == teacher.id)
                    .context("content teacher vanished")?;
                let capability = match plan.content {
                    ContentKind::Process => &mut teacher.capabilities.process,
                    ContentKind::Context => &mut teacher.capabilities.context,
                    ContentKind::ToolTrajectory => &mut teacher.capabilities.tool_trajectories,
                    _ => anyhow::bail!("unsupported content"),
                };
                capability.available = true;
                capability.artefacts.push(artifact.clone());
                capability.reason="Actual additional generation, frozen train-only context/feedback and verified final solution; generated comments do not prove faithful internal reasoning".into();
                if bundle
                    .preferences
                    .iter()
                    .any(|p| p.chosen_sample_id == chosen.id)
                {
                    teacher.capabilities.preferences.available = true;
                    teacher.capabilities.preferences.reason="Genuine same-task passing/benign-failing pairs include additional generated content; rejected origin prompts remain explicit".into();
                    teacher
                        .capabilities
                        .preferences
                        .artefacts
                        .push(artifact.clone());
                }
                bundle
                    .capabilities
                    .insert(teacher.id.clone(), teacher.capabilities.clone());
            }
        }
        ensure!(
            !generation.complete
                || generation.completion.trim().is_empty()
                || record.sandbox_report.is_some()
                || record.grading_error.is_some(),
            "complete content lacks grading or explicit error"
        );
    } else {
        ensure!(
            raw.generation_error.as_ref().is_some_and(|e| !e.is_empty())
                && record.sandbox_report.is_none(),
            "missing generation without explicit failure"
        );
    }
    ensure!(
        success == record.success && (!success || record.grading_error.is_none()),
        "content verdict differs from raw evidence"
    );
    bundle.content_records.push(artifact.clone());
    Ok(())
}

#[cfg(test)]
mod report_tests {
    use super::*;

    #[test]
    fn inherited_positive_cannot_hide_an_empty_or_generation_failed_batch() -> Result<()> {
        fs::create_dir_all("reports")?;
        let temporary = tempfile::tempdir_in("reports")?;
        let root = std::env::current_dir()?.canonicalize()?;
        let output = temporary
            .path()
            .canonicalize()?
            .strip_prefix(&root)?
            .to_owned();
        artefact::write_json(&output.join("content-policy.json"), &serde_json::json!({}))?;
        artefact::write_json(&output.join("bundle.json"), &serde_json::json!({}))?;
        let inherited = Artefact {
            path: "inherited/example.json".into(),
            sha256: "0".repeat(64),
            bytes: 0,
        };
        let task = train_tasks().remove(0);
        let sample = EvidenceSample {
            id: "inherited".into(),
            task,
            teacher_id: "fixture".into(),
            prompt: String::new(),
            completion: "fixture".into(),
            content: ContentKind::CompleteSequence,
            observation: inherited.clone(),
            raw_generation: inherited.clone(),
            sandbox_report: inherited.clone(),
            audit: inherited.clone(),
            sequence_sha256: "0".repeat(64),
            provenance_kind: "synthetic_unit_fixture".into(),
            student_rollout: None,
        };
        let mut bundle = EvidenceBundle {
            schema_version: 1,
            evidence_kind: "synthetic_unit_fixture".into(),
            created_unix_ms: 0,
            teachers: vec![],
            samples: vec![sample],
            candidates: vec![],
            preferences: vec![],
            input_comparisons: vec![],
            content_records: vec![],
            rollout_records: vec![],
            capabilities: BTreeMap::new(),
            train_corpus_sha256: "0".repeat(64),
            limitations: vec![],
        };
        let options = ContentCollectOptions {
            bundle: "unused".into(),
            output: output.clone(),
            content: vec![ContentKind::Process],
            runtime: RuntimeOptions::default(),
            max_tokens: 16,
            seed: 1729,
            maximum_tasks: 1,
        };
        write_content_report(&root, &options, &bundle, &[])?;
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("content-report.json"))?)?;
        assert_eq!(report["healthy_complete"], false);
        assert_eq!(report["inherited_positive_samples"], 1);
        assert_eq!(report["current_positive_samples"], 0);
        let raw = ContentRaw {
            schema_version: 1,
            evidence_kind: "synthetic_unit_fixture".into(),
            plan: inherited.clone(),
            generation: None,
            generation_error: Some("controlled generation failure".into()),
            execution: None,
        };
        artefact::write_json(&output.join("raw.json"), &raw)?;
        let record = ContentRecord {
            schema_version: 1,
            evidence_kind: "synthetic_unit_fixture".into(),
            plan: inherited,
            raw_generation: capture(&root, &output.join("raw.json"))?,
            sandbox_report: None,
            grading_error: None,
            success: false,
        };
        artefact::write_json(&output.join("record.json"), &record)?;
        bundle
            .content_records
            .push(capture(&root, &output.join("record.json"))?);
        write_content_report(&root, &options, &bundle, &[])?;
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("content-report.json"))?)?;
        assert_eq!(report["healthy_complete"], false);
        assert_eq!(report["generation_errors"], 1);
        assert_eq!(report["completed_current_content_records"], 1);
        assert_eq!(report["inherited_positive_samples"], 1);
        Ok(())
    }
}
