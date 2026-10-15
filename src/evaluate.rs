//! Complete checkpoint evaluations bound to preregistered requests and trusted traces.
//!
//! Reusing an output under another seed/task, changing a decoding cap, or accepting
//! an unverified summary must never manufacture evidence of learning.
use crate::{
    artefact::{self, Artefact},
    gate::{self, GateConfig, GateReport, Pair},
    model::{self, Generation, GenerationRequest, ModelRuntime},
    sandbox::{self, SandboxConfig, SandboxReport},
    tasks::{self, Split},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
mod resume;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEvidence {
    pub checkpoint_id: String,
    pub tensor: Artefact,
    pub metadata: Artefact,
    /// The complete pinned asset manifest, not just an asserted base identifier.
    pub base_assets: Artefact,
    pub base_tensor_sha256: String,
    pub tokenizer_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferencePlan {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub timeout_seconds: u64,
}
impl InferencePlan {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=1024).contains(&self.max_new_tokens),
            "invalid evaluation output cap"
        );
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "invalid evaluation temperature"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "invalid evaluation top_p"
        );
        ensure!(
            (1..=1800).contains(&self.timeout_seconds),
            "invalid evaluation inference timeout"
        );
        Ok(())
    }
    fn request(&self, prompt: String, seed: u64) -> GenerationRequest {
        GenerationRequest {
            prompt,
            max_new_tokens: self.max_new_tokens,
            seed,
            temperature: self.temperature,
            top_p: self.top_p,
            timeout_seconds: self.timeout_seconds,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preregistration {
    pub schema_version: u32,
    pub frozen_unix_ms: i64,
    pub checkpoint: CheckpointEvidence,
    pub gate_config_sha256: String,
    pub corpus_sha256: String,
    pub split: Split,
    pub evaluation_seeds: Vec<u64>,
    pub inference: InferencePlan,
    pub sandbox_config: SandboxConfig,
    pub sandbox_policy_sha256: String,
    pub sandbox_image_id: String,
    pub sandbox_preflight: Artefact,
}

/// A generation error is a failed observation, never a missing denominator entry.
/// Loading an invalid model, losing the worker or lacking sandbox controls instead
/// invalidates the evaluation and leaves only its durable partial report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationEvidence {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub request: GenerationRequest,
    pub request_sha256: String,
    pub generation: Option<Generation>,
    pub model_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub task_id: String,
    pub source_id: String,
    pub split: Split,
    pub seed: u64,
    pub specification_sha256: String,
    pub oracle_sha256: String,
    pub checkpoint_id: String,
    pub success: bool,
    pub generation_complete: bool,
    pub generation: Artefact,
    pub sandbox_report: Option<Artefact>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub checkpoint: CheckpointEvidence,
    pub preregistration: Artefact,
    pub gate_config_sha256: String,
    pub corpus_sha256: String,
    pub split: Split,
    pub outcomes: Vec<Outcome>,
    pub frozen_before_heldout_unix_ms: i64,
    pub completed_unix_ms: i64,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct VerifiedGate {
    #[serde(flatten)]
    pub report: GateReport,
    pub incumbent_checkpoint_id: String,
    pub candidate_checkpoint_id: String,
    pub incumbent_evaluation_sha256: String,
    pub candidate_evaluation_sha256: String,
    pub gate_config_sha256: String,
}

fn read_preregistration(eval: &Evaluation, root: &Path) -> Result<Preregistration> {
    Ok(serde_json::from_slice(&std::fs::read(
        eval.preregistration.verify(root)?,
    )?)?)
}
fn same_serialised(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_vec(a)? == serde_json::to_vec(b)?)
}

fn bind_audit_to_root(report: &mut SandboxReport, root: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let path = if report.audit_path.is_absolute() {
        report.audit_path.clone()
    } else {
        root.join(&report.audit_path)
    };
    let path = path
        .canonicalize()
        .context("trusted audit artifact missing")?;
    ensure!(
        path.starts_with(&root),
        "trusted audit artifact escapes evidence root"
    );
    ensure!(
        artefact::file_hash(&path)? == report.audit_sha256,
        "trusted audit artifact checksum mismatch"
    );
    report.audit_path = path;
    Ok(())
}

pub fn verify(eval: &Evaluation, root: &Path, gate_path: &Path, cfg: &GateConfig) -> Result<()> {
    cfg.validate()?;
    ensure!(
        eval.schema_version == 2 && eval.evidence_kind == "actual_checkpoint_generations",
        "unverified, obsolete or fixture evaluation"
    );
    let config_hash = artefact::file_hash(gate_path)?;
    ensure!(
        eval.gate_config_sha256 == config_hash,
        "evaluation gate configuration changed"
    );
    let prereg = read_preregistration(eval, root)?;
    ensure!(
        prereg.schema_version == 1 && prereg.gate_config_sha256 == config_hash,
        "invalid preregistered gate configuration"
    );
    ensure!(
        prereg.frozen_unix_ms > 0
            && prereg.frozen_unix_ms == eval.frozen_before_heldout_unix_ms
            && eval.completed_unix_ms >= prereg.frozen_unix_ms,
        "invalid preregistration timestamps"
    );
    ensure!(
        same_serialised(&prereg.checkpoint, &eval.checkpoint)?,
        "checkpoint changed after preregistration"
    );
    ensure!(
        prereg.evaluation_seeds == cfg.evaluation_seeds,
        "decoding seeds changed after preregistration"
    );
    ensure!(
        prereg.split == eval.split,
        "evaluation split changed after preregistration"
    );
    prereg.inference.validate()?;
    let corpus = tasks::corpus();
    let corpus_hash = artefact::hash(&serde_json::to_vec(&corpus)?);
    ensure!(
        eval.corpus_sha256 == corpus_hash && prereg.corpus_sha256 == corpus_hash,
        "task corpus changed"
    );
    ensure!(
        eval.checkpoint.checkpoint_id == eval.checkpoint.tensor.sha256
            && eval.checkpoint.base_tensor_sha256 == model::MODEL_SHA256
            && eval.checkpoint.tokenizer_sha256 == model::TOKENIZER_SHA256,
        "checkpoint/base/tokenizer identity mismatch"
    );
    let base_manifest = eval.checkpoint.base_assets.verify(root)?;
    let assets = model::validate_assets(
        base_manifest
            .parent()
            .context("base asset manifest has no parent")?,
    )?;
    ensure!(
        assets.files.get("tokenizer.json").map(String::as_str) == Some(model::TOKENIZER_SHA256),
        "unpinned tokenizer"
    );
    let tensor = eval.checkpoint.tensor.verify(root)?;
    let metadata_path = eval.checkpoint.metadata.verify(root)?;
    // Parse tensors too: a renamed configuration is not a checkpoint.
    let tensors = candle_core::safetensors::load(&tensor, &candle_core::Device::Cpu)?;
    ensure!(!tensors.is_empty(), "checkpoint contains no tensors");
    if eval.checkpoint.checkpoint_id != model::MODEL_SHA256 {
        let metadata: crate::train::CheckpointManifest =
            serde_json::from_slice(&std::fs::read(&metadata_path)?)?;
        ensure!(
            metadata.adapter_sha256 == eval.checkpoint.tensor.sha256,
            "checkpoint manifest refers to different tensors"
        );
        crate::train::validate_checkpoint(
            metadata_path
                .parent()
                .context("checkpoint manifest has no parent")?,
            &metadata,
            &assets,
        )?;
    } else {
        let metadata: model::AssetManifest =
            serde_json::from_slice(&std::fs::read(metadata_path)?)?;
        ensure!(
            metadata.revision == model::MODEL_REVISION && metadata.files == assets.files,
            "unpinned base metadata"
        );
    }
    ensure!(
        !prereg.sandbox_image_id.is_empty()
            && prereg.sandbox_config.expected_image_id == prereg.sandbox_image_id,
        "sandbox image was not frozen before held-out access"
    );
    ensure!(
        sandbox::policy_sha256(&prereg.sandbox_config)? == prereg.sandbox_policy_sha256,
        "sandbox policy changed after preregistration"
    );
    let preflight: serde_json::Value =
        serde_json::from_slice(&std::fs::read(prereg.sandbox_preflight.verify(root)?)?)?;
    ensure!(
        preflight["probes_passed"] == true
            && preflight["image_id"] == prereg.sandbox_image_id
            && preflight["policy_sha256"] == prereg.sandbox_policy_sha256,
        "effective sandbox controls were not verified before evaluation"
    );
    let expected = corpus
        .iter()
        .filter(|task| task.split == eval.split)
        .flat_map(|task| {
            cfg.evaluation_seeds
                .iter()
                .map(move |seed| (task.id.clone(), *seed))
        })
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    // Reject extra non-held-out observations too: a caller cannot hide mixed output
    // behind a filter and then reuse the same file as a clean sealed evaluation.
    for outcome in &eval.outcomes {
        ensure!(
            observed.insert((outcome.task_id.clone(), outcome.seed)),
            "duplicate evaluation task/seed"
        );
        verify_cached_outcome(outcome, &eval.checkpoint, &prereg, &corpus, root)?;
    }
    ensure!(
        observed == expected,
        "evaluation lacks expected held-out task/seed, or includes extras"
    );
    Ok(())
}

fn verify_cached_outcome(
    outcome: &Outcome,
    checkpoint: &CheckpointEvidence,
    prereg: &Preregistration,
    corpus: &[tasks::Task],
    root: &Path,
) -> Result<()> {
    ensure!(
        outcome.split == prereg.split,
        "evaluation contains mixed split outcomes"
    );
    let task = corpus
        .iter()
        .find(|task| task.id == outcome.task_id)
        .context("unknown evaluation task")?;
    ensure!(
        task.split == prereg.split
            && task.source_id == outcome.source_id
            && task.specification_sha256 == outcome.specification_sha256
            && task.oracle_sha256 == outcome.oracle_sha256,
        "task/source/grader mismatch"
    );
    ensure!(
        outcome.checkpoint_id == checkpoint.checkpoint_id,
        "mixed checkpoint outcomes"
    );
    let generated: GenerationEvidence =
        serde_json::from_slice(&std::fs::read(outcome.generation.verify(root)?)?)?;
    verify_generation(
        &generated,
        &prereg.inference,
        task.prompt(),
        outcome.seed,
        &outcome.checkpoint_id,
    )?;
    match (&generated.generation, &generated.model_error) {
        (None, Some(_)) => ensure!(
            !outcome.success && !outcome.generation_complete && outcome.sandbox_report.is_none(),
            "generation error was counted as success or supplied invented execution"
        ),
        (Some(generation), None) => {
            ensure!(
                generation.complete == outcome.generation_complete,
                "generation completion state mismatch"
            );
            if !generation.complete || generation.completion.trim().is_empty() {
                ensure!(
                    !outcome.success && outcome.sandbox_report.is_none(),
                    "incomplete/empty output did not fail before execution"
                );
                return Ok(());
            }
            let mut report: SandboxReport = serde_json::from_slice(&std::fs::read(
                outcome
                    .sandbox_report
                    .as_ref()
                    .context("completed generation lacks trusted grading report")?
                    .verify(root)?,
            )?)?;
            bind_audit_to_root(&mut report, root)?;
            let code = extract_code(&generation.completion);
            sandbox::verify_report(&report, &prereg.sandbox_config, task, &code)?;
            ensure!(
                report.policy_sha256 == prereg.sandbox_policy_sha256
                    && report.image_id == prereg.sandbox_image_id,
                "execution used a different sandbox image/policy"
            );
            let required = task.public_examples.len() + tasks::private_cases(task)?.len();
            let passed = report.passed
                && !report.quarantined
                && report.trace_complete
                && report.cases_total == required
                && report.cases_passed == required;
            ensure!(
                outcome.success == passed,
                "success disagrees with complete trusted verdict"
            );
        }
        _ => anyhow::bail!("generation must contain exactly one real result or model error"),
    }
    Ok(())
}

pub fn verify_split(
    eval: &Evaluation,
    root: &Path,
    gate_path: &Path,
    cfg: &GateConfig,
    split: Split,
) -> Result<()> {
    ensure!(
        eval.split == split,
        "cached evaluation split differs from required split"
    );
    verify(eval, root, gate_path, cfg)
}

fn verify_generation(
    evidence: &GenerationEvidence,
    plan: &InferencePlan,
    prompt: String,
    seed: u64,
    checkpoint_id: &str,
) -> Result<()> {
    ensure!(
        evidence.schema_version == 1 && evidence.checkpoint_id == checkpoint_id,
        "invalid generation identity"
    );
    let expected = plan.request(prompt, seed);
    ensure!(
        same_serialised(&evidence.request, &expected)?
            && evidence.request_sha256 == artefact::hash(&serde_json::to_vec(&expected)?),
        "generation task, seed or decoding parameters differ from preregistration"
    );
    if let Some(generation) = &evidence.generation {
        ensure!(
            evidence.model_error.is_none() && generation.checkpoint_id == checkpoint_id,
            "mixed generation/checkpoint evidence"
        );
        ensure!(
            generation.output_tokens <= plan.max_new_tokens && generation.input_tokens > 0,
            "generation token counts violate inference plan"
        );
        ensure!(
            matches!(
                generation.finish_reason.as_str(),
                "stop" | "length" | "timeout"
            ) && generation.complete == (generation.finish_reason == "stop"),
            "generation finish reason/completion mismatch"
        );
    } else {
        ensure!(
            evidence
                .model_error
                .as_ref()
                .is_some_and(|error| !error.trim().is_empty()),
            "missing model result/error"
        );
    }
    Ok(())
}

pub fn gate_files(
    incumbent_path: &Path,
    candidate_path: &Path,
    root: &Path,
    gate_path: &Path,
) -> Result<VerifiedGate> {
    let cfg = GateConfig::load(gate_path)?;
    let incumbent: Evaluation = serde_json::from_slice(&std::fs::read(incumbent_path)?)?;
    let candidate: Evaluation = serde_json::from_slice(&std::fs::read(candidate_path)?)?;
    ensure!(
        incumbent.split == Split::HeldOut && candidate.split == Split::HeldOut,
        "only sealed held-out evaluations can authorise promotion"
    );
    verify(&incumbent, root, gate_path, &cfg)?;
    verify(&candidate, root, gate_path, &cfg)?;
    let incumbent_plan = read_preregistration(&incumbent, root)?;
    let candidate_plan = read_preregistration(&candidate, root)?;
    ensure!(
        incumbent_plan.inference == candidate_plan.inference,
        "checkpoint comparison changed decoding caps/parameters"
    );
    ensure!(
        incumbent_plan.sandbox_policy_sha256 == candidate_plan.sandbox_policy_sha256
            && incumbent_plan.sandbox_image_id == candidate_plan.sandbox_image_id,
        "checkpoint comparison changed sandbox policy/image"
    );
    let base = incumbent
        .outcomes
        .iter()
        .map(|outcome| ((outcome.task_id.as_str(), outcome.seed), outcome))
        .collect::<BTreeMap<_, _>>();
    let mut pairs = Vec::new();
    for outcome in &candidate.outcomes {
        let previous = base
            .get(&(outcome.task_id.as_str(), outcome.seed))
            .context("unpaired result")?;
        pairs.push(Pair {
            source_id: outcome.source_id.clone(),
            task_id: outcome.task_id.clone(),
            seed: outcome.seed,
            incumbent_pass: previous.success,
            candidate_pass: outcome.success,
        });
    }
    let mut operational_ok = true;
    for evaluation in [&incumbent, &candidate] {
        for outcome in &evaluation.outcomes {
            if let Some(artifact) = &outcome.sandbox_report {
                let mut report: SandboxReport =
                    serde_json::from_slice(&std::fs::read(artifact.verify(root)?)?)?;
                bind_audit_to_root(&mut report, root)?;
                let raw: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&report.audit_path)?)?;
                // An incorrect answer is a quality failure in the denominator.
                // Prohibited syscalls and incomplete traces of executed candidates
                // additionally veto promotion. Compilation failure runs no candidate.
                operational_ok &= report.audit_events.is_empty()
                    && (report.trace_complete || raw["compile_ok"] == false);
            }
        }
    }
    let report = gate::decide(
        &pairs,
        &cfg,
        operational_ok,
        "verified_checkpoint_evaluation",
    );
    Ok(VerifiedGate {
        report,
        incumbent_checkpoint_id: incumbent.checkpoint.checkpoint_id,
        candidate_checkpoint_id: candidate.checkpoint.checkpoint_id,
        incumbent_evaluation_sha256: artefact::file_hash(incumbent_path)?,
        candidate_evaluation_sha256: artefact::file_hash(candidate_path)?,
        gate_config_sha256: artefact::file_hash(gate_path)?,
    })
}

/// Only strip a complete Rust Markdown fence. Partial output stays a failure.
pub fn extract_code(text: &str) -> String {
    let trimmed = text.trim();
    if (trimmed.starts_with("```rust\n") || trimmed.starts_with("```\n"))
        && trimmed.ends_with("```")
    {
        return trimmed
            .split_once('\n')
            .map(|(_, rest)| rest[..rest.len() - 3].trim().to_owned())
            .unwrap_or_else(|| trimmed.into());
    }
    trimmed.to_owned()
}

pub async fn run(
    base_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    root: PathBuf,
    gate_path: PathBuf,
    output: PathBuf,
    sandbox_config: SandboxConfig,
    max_tokens: usize,
) -> Result<Evaluation> {
    run_split(
        base_dir,
        checkpoint,
        root,
        gate_path,
        output,
        sandbox_config,
        max_tokens,
        Split::HeldOut,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_split(
    base_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    root: PathBuf,
    gate_path: PathBuf,
    output: PathBuf,
    mut sandbox_config: SandboxConfig,
    max_tokens: usize,
    split: Split,
) -> Result<Evaluation> {
    let cfg = GateConfig::load(&gate_path)?;
    let inference = InferencePlan {
        max_new_tokens: max_tokens,
        temperature: 0.2,
        top_p: 0.95,
        timeout_seconds: 120,
    };
    inference.validate()?;
    let root = root.canonicalize()?;
    std::fs::create_dir_all(&output)?;
    let _lease = crate::engine::RunLease::acquire(&output.join(".evaluation.lock"))?;
    let relative = |path: &Path| -> Result<PathBuf> {
        Ok(path.canonicalize()?.strip_prefix(&root)?.to_owned())
    };
    let assets_dir = base_dir.clone();
    let assets = tokio::task::spawn_blocking(move || model::validate_assets(&assets_dir)).await??;
    let tensor_path = checkpoint
        .as_ref()
        .map(|p| p.join("adapter.safetensors"))
        .unwrap_or_else(|| base_dir.join("model.safetensors"));
    let metadata_path = checkpoint
        .as_ref()
        .map(|p| p.join("manifest.json"))
        .unwrap_or_else(|| base_dir.join("assets.json"));
    let tensor = Artefact::capture(&root, relative(&tensor_path)?)?;
    let evidence = CheckpointEvidence {
        checkpoint_id: tensor.sha256.clone(),
        tensor,
        metadata: Artefact::capture(&root, relative(&metadata_path)?)?,
        base_assets: Artefact::capture(&root, relative(&base_dir.join("assets.json"))?)?,
        base_tensor_sha256: model::MODEL_SHA256.into(),
        tokenizer_sha256: assets.files["tokenizer.json"].clone(),
    };
    let corpus = tasks::corpus();
    let corpus_sha256 = artefact::hash(&serde_json::to_vec(&corpus)?);
    let config_sha256 = artefact::file_hash(&gate_path)?;
    let prereg_path = output.join("preregistration.json");
    let existing = if prereg_path.exists() {
        Some(serde_json::from_slice::<Preregistration>(&std::fs::read(
            &prereg_path,
        )?)?)
    } else {
        ensure!(
            !output.join("partial.json").exists()
                && !output.join("evaluation.json").exists()
                && !std::fs::read_dir(&output)?.any(|entry| entry.is_ok_and(|e| e
                    .file_name()
                    .to_string_lossy()
                    .ends_with("-generation.json")
                    || e.file_name().to_string_lossy().ends_with("-outcome.json"))),
            "orphan evaluation evidence lacks its original preregistration; retain it and use an explicit fresh output"
        );
        None
    };
    if let Some(prereg) = &existing {
        if sandbox_config.expected_image_id.is_empty() {
            sandbox_config.expected_image_id = prereg.sandbox_image_id.clone();
        }
        resume::validate(
            prereg,
            &resume::Intent {
                checkpoint: &evidence,
                config_sha256: &config_sha256,
                corpus_sha256: &corpus_sha256,
                split,
                seeds: &cfg.evaluation_seeds,
                inference: &inference,
                sandbox: &sandbox_config,
            },
        )?;
        prereg.sandbox_preflight.verify(&root)?;
        let completed_path = output.join("evaluation.json");
        if completed_path.exists() {
            let completed: Evaluation = serde_json::from_slice(&std::fs::read(&completed_path)?)?;
            ensure!(
                same_serialised(
                    &completed.preregistration,
                    &Artefact::capture(&root, relative(&prereg_path)?)?
                )?,
                "finished evaluation refers to another preregistration"
            );
            let checked = completed.clone();
            let checked_root = root.clone();
            let checked_gate = gate_path.clone();
            let checked_cfg = cfg.clone();
            tokio::task::spawn_blocking(move || {
                verify_split(&checked, &checked_root, &checked_gate, &checked_cfg, split)
            })
            .await??;
            eprintln!(
                "evaluate: reusing complete verified evaluation; no model or containers started"
            );
            return Ok(completed);
        }
    }
    // A resumed incomplete run must verify the current effective sandbox too.
    // Its original successful preflight artefact and timestamps remain immutable.
    let preflight = sandbox::preflight(&sandbox_config).await?;
    ensure!(
        preflight.probes_passed,
        "sandbox probes failed before evaluation access"
    );
    if sandbox_config.expected_image_id.is_empty() {
        sandbox_config.expected_image_id = preflight.image_id.clone();
    }
    ensure!(
        sandbox_config.expected_image_id == preflight.image_id,
        "configured sandbox image does not match verified image"
    );
    let prereg = if let Some(prereg) = existing {
        prereg
    } else {
        let preflight_path = output.join("sandbox-preflight.json");
        artefact::write_json(&preflight_path, &preflight)?;
        let prereg = Preregistration {
            schema_version: 1,
            frozen_unix_ms: crate::ledger::unix_millis(),
            checkpoint: evidence.clone(),
            gate_config_sha256: config_sha256,
            corpus_sha256,
            split,
            evaluation_seeds: cfg.evaluation_seeds.clone(),
            inference: inference.clone(),
            sandbox_policy_sha256: sandbox::policy_sha256(&sandbox_config)?,
            sandbox_image_id: preflight.image_id,
            sandbox_config: sandbox_config.clone(),
            sandbox_preflight: Artefact::capture(&root, relative(&preflight_path)?)?,
        };
        artefact::write_json(&prereg_path, &prereg)?;
        prereg
    };
    let preregistration = Artefact::capture(&root, relative(&prereg_path)?)?;
    let copied_output = output.clone();
    let copied_prereg = prereg.clone();
    let copied_corpus = corpus.clone();
    let copied_root = root.clone();
    let mut completed = tokio::task::spawn_blocking(move || {
        resume::outcomes(&copied_output, &copied_prereg, &copied_corpus, &copied_root)
    })
    .await??;
    let mut runtime: Option<ModelRuntime> = None;
    for task in corpus.iter().filter(|task| task.split == split) {
        for &seed in &cfg.evaluation_seeds {
            let key = (task.id.clone(), seed);
            if completed.contains_key(&key) {
                eprintln!(
                    "evaluate: reusing trusted graded outcome {} seed={seed}",
                    task.id
                );
                continue;
            }
            let generation_path = output.join(format!("{}-{seed}-generation.json", task.id));
            let generated =
                if let Some(record) = resume::generation(&generation_path, &prereg, task, seed)? {
                    eprintln!(
                        "evaluate: reusing durable raw generation {} seed={seed}",
                        task.id
                    );
                    record
                } else {
                    if runtime.is_none() {
                        let copied_base = base_dir.clone();
                        let copied_checkpoint = checkpoint.clone();
                        runtime = Some(
                            tokio::task::spawn_blocking(move || {
                                ModelRuntime::load(&copied_base, copied_checkpoint.as_deref())
                            })
                            .await??,
                        );
                    }
                    eprintln!(
                        "evaluate: generating {} seed={seed} checkpoint={}",
                        task.id, evidence.checkpoint_id
                    );
                    let request = inference.request(task.prompt(), seed);
                    let generation_request = request.clone();
                    let mut worker = runtime.take().context("evaluation model was not loaded")?;
                    let (returned, result) = tokio::task::spawn_blocking(move || {
                        let generated = worker.generate(generation_request);
                        (worker, generated)
                    })
                    .await?;
                    runtime = Some(returned);
                    let (generation, model_error) = match result {
                        Ok(generation) => (Some(generation), None),
                        Err(error) => (None, Some(format!("{error:#}"))),
                    };
                    let generated = GenerationEvidence {
                        schema_version: 1,
                        checkpoint_id: evidence.checkpoint_id.clone(),
                        request_sha256: artefact::hash(&serde_json::to_vec(&request)?),
                        request,
                        generation,
                        model_error,
                    };
                    verify_generation(
                        &generated,
                        &inference,
                        task.prompt(),
                        seed,
                        &evidence.checkpoint_id,
                    )?;
                    // The raw result is durable before any grading, including errors.
                    artefact::write_json(&generation_path, &generated)?;
                    generated
                };
            let (success, generation_complete, sandbox_report) = if let Some(generation) =
                &generated.generation
            {
                if generation.complete && !generation.completion.trim().is_empty() {
                    let code = extract_code(&generation.completion);
                    let report = sandbox::grade(task, &code, &sandbox_config).await?;
                    sandbox::verify_report(&report, &sandbox_config, task, &code)?;
                    let report_path = sandbox_config
                        .report_dir
                        .join(format!("{}.json", report.run_id));
                    let required = task.public_examples.len() + tasks::private_cases(task)?.len();
                    (
                        report.passed
                            && !report.quarantined
                            && report.trace_complete
                            && report.cases_total == required
                            && report.cases_passed == required,
                        true,
                        Some(Artefact::capture(&root, relative(&report_path)?)?),
                    )
                } else {
                    (false, generation.complete, None)
                }
            } else {
                (false, false, None)
            };
            let outcome = Outcome {
                task_id: task.id.clone(),
                source_id: task.source_id.clone(),
                split: task.split,
                seed,
                specification_sha256: task.specification_sha256.clone(),
                oracle_sha256: task.oracle_sha256.clone(),
                checkpoint_id: evidence.checkpoint_id.clone(),
                success,
                generation_complete,
                generation: Artefact::capture(&root, relative(&generation_path)?)?,
                sandbox_report,
            };
            verify_cached_outcome(&outcome, &evidence, &prereg, &corpus, &root)?;
            artefact::write_json(
                &output.join(format!("{}-{seed}-outcome.json", task.id)),
                &outcome,
            )?;
            completed.insert(key, outcome);
            artefact::write_json(
                &output.join("partial.json"),
                &completed.values().collect::<Vec<_>>(),
            )?;
        }
    }
    let eval = Evaluation {
        schema_version: 2,
        evidence_kind: "actual_checkpoint_generations".into(),
        checkpoint: evidence,
        preregistration,
        gate_config_sha256: prereg.gate_config_sha256,
        corpus_sha256: prereg.corpus_sha256,
        split,
        outcomes: completed.into_values().collect(),
        frozen_before_heldout_unix_ms: prereg.frozen_unix_ms,
        completed_unix_ms: crate::ledger::unix_millis(),
    };
    let checked = eval.clone();
    let checked_root = root.clone();
    let checked_gate = gate_path.clone();
    let checked_cfg = cfg.clone();
    tokio::task::spawn_blocking(move || {
        verify(&checked, &checked_root, &checked_gate, &checked_cfg)
    })
    .await??;
    artefact::write_json(&output.join("evaluation.json"), &eval)?;
    Ok(eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_complete_fences_are_removed() {
        assert_eq!(extract_code("```rust\nfn solve() {}\n```"), "fn solve() {}");
        assert_eq!(extract_code("```rust\nfn solve()"), "```rust\nfn solve()");
    }
    fn evidence(plan: &InferencePlan) -> GenerationEvidence {
        let request = plan.request("fixed task".into(), 17);
        GenerationEvidence {
            schema_version: 1,
            checkpoint_id: "checkpoint".into(),
            request_sha256: artefact::hash(&serde_json::to_vec(&request).unwrap()),
            request,
            generation: Some(Generation {
                completion: "fn solve() {}".into(),
                checkpoint_id: "checkpoint".into(),
                finish_reason: "stop".into(),
                input_tokens: 10,
                output_tokens: 5,
                elapsed_ms: 1,
                complete: true,
            }),
            model_error: None,
        }
    }
    #[test]
    fn reused_task_seed_and_decoding_caps_are_rejected() {
        let plan = InferencePlan {
            max_new_tokens: 32,
            temperature: 0.2,
            top_p: 0.95,
            timeout_seconds: 120,
        };
        let record = evidence(&plan);
        assert!(verify_generation(&record, &plan, "fixed task".into(), 17, "checkpoint").is_ok());
        assert!(
            verify_generation(&record, &plan, "another task".into(), 17, "checkpoint").is_err()
        );
        assert!(verify_generation(&record, &plan, "fixed task".into(), 18, "checkpoint").is_err());
        let mut changed = plan.clone();
        changed.max_new_tokens += 1;
        assert!(
            verify_generation(&record, &changed, "fixed task".into(), 17, "checkpoint").is_err()
        );
    }
    #[test]
    fn errors_and_incomplete_generations_cannot_assert_completion() {
        let plan = InferencePlan {
            max_new_tokens: 32,
            temperature: 0.2,
            top_p: 0.95,
            timeout_seconds: 120,
        };
        let mut record = evidence(&plan);
        record.generation.as_mut().unwrap().finish_reason = "length".into();
        assert!(verify_generation(&record, &plan, "fixed task".into(), 17, "checkpoint").is_err());
        record.generation = None;
        record.model_error = Some("non-finite model logits".into());
        assert!(verify_generation(&record, &plan, "fixed task".into(), 17, "checkpoint").is_ok());
        record.model_error = None;
        assert!(verify_generation(&record, &plan, "fixed task".into(), 17, "checkpoint").is_err());
    }
}
