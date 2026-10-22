//! Bounded native optimisation with immutable order and durable step journals.
use super::{
    alignment::{self, EncodedSequence},
    checkpoint::{self, ExperimentalCheckpointManifest, Progress},
    targets, *,
};
use crate::{
    artefact,
    model::{self, AdapterConfig, ModelRuntime, StableRng},
    teacher_evidence::{self, EvidenceBundle, EvidenceSample},
    train::{ResumableAdam, TrainConfig},
};
use anyhow::{Context, Result, ensure};
use candle_core::{D, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::fd::AsRawFd,
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;

mod eligibility;
pub use eligibility::preflight_eligibility;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Step {
    pub(super) sample_id: String,
    pub(super) teacher_id: String,
    pub(super) stage: usize,
}

pub(super) struct Selection {
    pub(super) order: Vec<Step>,
    pub(super) samples: BTreeMap<String, EvidenceSample>,
    pub(super) encoded: BTreeMap<String, EncodedSequence>,
    pub(super) preferences: BTreeMap<String, preference::FrozenPreference>,
    context_transforms: BTreeMap<String, conditioning::PromptTransform>,
    pub(super) rejected_overlength: BTreeMap<String, String>,
    pub(super) eligible_sequences: BTreeMap<String, serde_json::Value>,
}

fn validate_origins(bundle: &EvidenceBundle) -> Result<()> {
    let mut checked = BTreeSet::new();
    for sample in &bundle.samples {
        if checked.insert((&sample.task.id, &sample.task.specification_sha256)) {
            crate::curate::validate_train_origin(&sample.task)?;
        }
    }
    Ok(())
}

fn validate(spec: &DistillationWorkerSpec) -> Result<()> {
    spec.student_runtime
        .validate_build_support()
        .context("unsupported student runtime before selection or optimisation")?;
    if targets::required(spec) {
        spec.teacher_runtime
            .validate_build_support()
            .context("unsupported teacher capture runtime before selection or optimisation")?;
    }
    if let Some(policy) = &spec.health_check {
        ensure!(
            spec.schema_version >= 2
                && spec.student_kind == DistillationStudent::ConventionalQwenLora
                && !policy.checkpoints.is_empty()
                && policy.checkpoints.len() <= 4
                && policy.checkpoints.windows(2).all(|p| p[0] < p[1])
                && policy
                    .checkpoints
                    .iter()
                    .all(|s| *s > 0 && *s <= spec.steps)
                && (1..=512).contains(&policy.max_new_tokens)
                && (1..=120).contains(&policy.timeout_seconds),
            "unsupported train-only generation health policy"
        );
    }
    ensure!(
        matches!(spec.schema_version, 1 | 2)
            && !spec.arm_id.is_empty()
            && spec.arm_id.len() <= 128
            && spec
                .arm_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
        "invalid distillation arm identity"
    );
    validate_training_bounds(
        spec.steps,
        spec.max_sequence_tokens,
        spec.rank,
        spec.alpha,
        spec.learning_rate,
    )?;
    ensure!(
        spec.temperature.is_finite()
            && (0.1..=10.).contains(&spec.temperature)
            && spec.dpo_beta.is_finite()
            && (0.001..=10.).contains(&spec.dpo_beta),
        "unsupported frozen temperature or DPO beta"
    );
    ensure!(
        !spec.teacher_ids.is_empty()
            && spec.teacher_ids.len() <= 8
            && spec.teacher_ids.iter().collect::<BTreeSet<_>>().len() == spec.teacher_ids.len(),
        "teacher order is empty or duplicated"
    );
    ensure!(
        !spec.contents.is_empty()
            && spec.contents.iter().collect::<BTreeSet<_>>().len() == spec.contents.len(),
        "content choices are empty or duplicated"
    );
    ensure!(
        !spec.losses.is_empty()
            && spec
                .losses
                .iter()
                .all(|l| l.weight.is_finite() && l.weight > 0. && l.weight <= 100.)
            && spec
                .losses
                .iter()
                .map(|l| l.kind)
                .collect::<BTreeSet<_>>()
                .len()
                == spec.losses.len(),
        "invalid or duplicated objective weights"
    );
    ensure!(
        spec.steps >= spec.teacher_ids.len(),
        "total update budget must cover every selected teacher"
    );
    ensure!(
        spec.student_runtime.dtype == model::InferenceDtype::F32,
        "experimental optimisation currently requires explicit float32 student tensors"
    );
    for path in [&spec.student_base, &spec.evidence_bundle, &spec.output] {
        ensure!(
            !path.is_absolute()
                && !path.as_os_str().is_empty()
                && path
                    .components()
                    .all(|p| matches!(p, std::path::Component::Normal(_))),
            "worker paths must remain relative without traversal"
        );
    }
    Ok(())
}

fn encode_sample(
    spec: &DistillationWorkerSpec,
    sample: &EvidenceSample,
    tokenizer: &Tokenizer,
) -> Result<EncodedSequence> {
    alignment::encode_with_policy(
        tokenizer,
        &student::prefix(spec, &conditioning::student_prompt(sample)),
        &sample.completion,
        student::eos(spec),
        spec.max_sequence_tokens,
        spec.sequence_policy,
    )
}

pub(super) fn choose(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    tokenizer: &Tokenizer,
) -> Result<Selection> {
    let dpo = spec
        .losses
        .iter()
        .any(|l| l.kind == LossKind::PreferenceDpo);
    ensure!(
        dpo == spec.contents.contains(&ContentKind::Preference),
        "preference content and DPO objective must be declared together"
    );
    for id in &spec.teacher_ids {
        let teacher = bundle
            .teachers
            .iter()
            .find(|t| &t.id == id)
            .context("teacher absent from verified evidence")?;
        ensure!(teacher.rights.eligible, "teacher rights are unresolved");
        if spec.data_policy == DataPolicy::OnPolicy {
            ensure!(
                teacher.capabilities.on_policy.available,
                "on-policy arm lacks actual checkpoint-bound student rollout and teacher feedback evidence"
            );
        }
        if spec.self_distillation {
            ensure!(
                teacher.capabilities.self_distillation.available,
                "self-distillation arm lacks actual frozen student-teacher evidence"
            );
        }
    }
    let mut samples = BTreeMap::new();
    let mut encoded = BTreeMap::new();
    let mut preferences = BTreeMap::new();
    let mut context_transforms = BTreeMap::new();
    let mut rejected_overlength = BTreeMap::new();
    let mut groups = BTreeMap::<String, Vec<String>>::new();
    let rollout_identity = if spec.data_policy == DataPolicy::OnPolicy || spec.self_distillation {
        Some(if let Some(path) = &spec.initial_checkpoint {
            let initial: ExperimentalCheckpointManifest =
                serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
            initial.adapter_sha256
        } else {
            student::assets(spec)?.files["model.safetensors"].clone()
        })
    } else {
        None
    };
    for sample in &bundle.samples {
        if !spec.teacher_ids.contains(&sample.teacher_id) {
            continue;
        }
        let content = spec.contents.contains(&sample.content)
            || (dpo
                && bundle
                    .preferences
                    .iter()
                    .any(|p| p.chosen_sample_id == sample.id));
        if !content
            || (dpo
                && !bundle.preferences.iter().any(|p| {
                    p.chosen_sample_id == sample.id
                        && teacher_evidence::negative_pair_is_eligible(p)
                }))
        {
            continue;
        }
        if spec.data_policy == DataPolicy::OnPolicy || spec.self_distillation {
            let Some(rollout) = sample.student_rollout.as_ref() else {
                continue;
            };
            ensure!(
                Some(&rollout.student_checkpoint_sha256) == rollout_identity.as_ref(),
                "student rollout belongs to another initial checkpoint"
            );
        }
        let pair = if dpo {
            match preference::prepare(spec, bundle, sample, tokenizer) {
                Ok(pair) => Some(pair),
                Err(error)
                    if spec.sequence_policy == SequencePolicy::RejectOverlength
                        && error
                            .downcast_ref::<alignment::SequenceCapacityError>()
                            .is_some() =>
                {
                    rejected_overlength.insert(
                        sample.id.clone(),
                        if spec.schema_version == 1 {
                            "distillation sequence exceeds its frozen token cap".into()
                        } else {
                            format!(
                                "preference student sequence; sample={} content={:?}: {error:#}",
                                sample.id, sample.content
                            )
                        },
                    );
                    continue;
                }
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let sequence = if spec
            .losses
            .iter()
            .all(|loss| loss.kind == LossKind::PreferenceDpo)
        {
            pair.as_ref()
                .context("pure DPO encoding absent")?
                .chosen
                .clone()
        } else {
            match encode_sample(spec, sample, tokenizer) {
                Ok(s) => s,
                Err(error)
                    if spec.sequence_policy == SequencePolicy::RejectOverlength
                        && error
                            .downcast_ref::<alignment::SequenceCapacityError>()
                            .is_some() =>
                {
                    rejected_overlength.insert(
                        sample.id.clone(),
                        if spec.schema_version == 1 {
                            "distillation sequence exceeds its frozen token cap".into()
                        } else {
                            format!(
                                "student sequence; sample={} content={:?}: {error:#}",
                                sample.id, sample.content
                            )
                        },
                    );
                    continue;
                }
                Err(e) => return Err(e),
            }
        };
        if spec.schema_version >= 2 && targets::required(spec) {
            match targets::encode_teacher(spec, bundle, sample) {
                Ok(teacher) => {
                    targets::validate_capacity(spec, bundle, sample, &sequence, &teacher)?;
                }
                Err(error)
                    if error
                        .downcast_ref::<alignment::SequenceCapacityError>()
                        .is_some() =>
                {
                    rejected_overlength.insert(
                        sample.id.clone(),
                        format!(
                            "teacher sequence; sample={} content={:?}: {error:#}",
                            sample.id, sample.content
                        ),
                    );
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        if let Some(transform) = conditioning::transform(sample)? {
            context_transforms.insert(sample.id.clone(), transform);
        }
        if let Some(pair) = pair {
            preferences.insert(sample.id.clone(), pair);
        }
        groups
            .entry(sample.teacher_id.clone())
            .or_default()
            .push(sample.id.clone());
        samples.insert(sample.id.clone(), sample.clone());
        encoded.insert(sample.id.clone(), sequence);
    }
    for content in &spec.contents {
        if *content != ContentKind::Preference {
            ensure!(
                samples.values().any(|s| &s.content == content),
                "requested content has no verified in-capacity samples: {content:?}; exclusions={rejected_overlength:?}"
            );
        }
    }
    let mut rng = StableRng::new(spec.seed);
    for id in &spec.teacher_ids {
        let rows = groups
            .get_mut(id)
            .context("selected teacher has no eligible bounded training rows")?;
        rows.sort();
        rng.shuffle(rows);
    }
    let mut required_contents = spec
        .contents
        .iter()
        .copied()
        .filter(|c| *c != ContentKind::Preference)
        .collect::<Vec<_>>();
    if required_contents.is_empty() {
        // Preference-only arms may use corrections originally collected as
        // process/context/tool content; the verified pair is the declared axis.
        required_contents.push(ContentKind::Preference);
    }
    for (teacher_index, id) in spec.teacher_ids.iter().enumerate() {
        let updates = spec.steps / spec.teacher_ids.len()
            + usize::from(teacher_index < spec.steps % spec.teacher_ids.len());
        ensure!(
            updates >= required_contents.len(),
            "declared update budget cannot cover every content kind for every selected teacher"
        );
        for content in &required_contents {
            ensure!(
                groups[id]
                    .iter()
                    .any(|row| *content == ContentKind::Preference
                        || samples[row].content == *content),
                "teacher {id} lacks verified bounded rows for requested content {content:?}"
            );
        }
    }
    let mut order = Vec::new();
    let mut counters = BTreeMap::<String, usize>::new();
    for step in 0..spec.steps {
        let stage = if spec.progressive {
            let base = spec.steps / spec.teacher_ids.len();
            let extra = spec.steps % spec.teacher_ids.len();
            let mut remaining = step;
            let mut stage = 0;
            for i in 0..spec.teacher_ids.len() {
                let count = base + usize::from(i < extra);
                if remaining < count {
                    stage = i;
                    break;
                }
                remaining -= count;
            }
            stage
        } else {
            step % spec.teacher_ids.len()
        };
        let teacher = &spec.teacher_ids[stage];
        let rows = &groups[teacher];
        let cursor = counters.entry(teacher.clone()).or_default();
        let desired = required_contents[*cursor % required_contents.len()];
        let eligible = rows
            .iter()
            .filter(|row| desired == ContentKind::Preference || samples[*row].content == desired)
            .collect::<Vec<_>>();
        let sample_id = eligible[(*cursor / required_contents.len()) % eligible.len()].to_string();
        *cursor += 1;
        order.push(Step {
            sample_id,
            teacher_id: teacher.clone(),
            stage,
        });
    }
    let selected = order
        .iter()
        .map(|s| s.sample_id.clone())
        .collect::<BTreeSet<_>>();
    let eligible_sequences = encoded
        .iter()
        .map(|(id, sequence)| (id.clone(), eligibility::sequence_summary(sequence)))
        .collect();
    samples.retain(|id, _| selected.contains(id));
    encoded.retain(|id, _| selected.contains(id));
    preferences.retain(|id, _| selected.contains(id));
    context_transforms.retain(|id, _| selected.contains(id));
    Ok(Selection {
        order,
        samples,
        encoded,
        preferences,
        context_transforms,
        rejected_overlength,
        eligible_sequences,
    })
}

pub(super) fn excluded_overlength(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    tokenizer: &Tokenizer,
) -> Result<BTreeMap<String, String>> {
    Ok(choose(spec, bundle, tokenizer)?.rejected_overlength)
}

fn load_student(spec: &DistillationWorkerSpec, adapter: Option<&Path>) -> Result<student::Student> {
    student::load(spec, adapter)
}

fn selected_logits(model: &mut student::Student, sequence: &EncodedSequence) -> Result<Tensor> {
    model.clear_cache();
    let input = Tensor::from_slice(
        &sequence.input_ids,
        (1, sequence.input_ids.len()),
        model.device(),
    )?;
    let hidden = model.hidden(&input)?.squeeze(0)?.index_select(
        &Tensor::from_slice(
            &sequence.prediction_positions,
            sequence.prediction_positions.len(),
            model.device(),
        )?,
        0,
    )?;
    model.logits(&hidden)
}

fn summed_log_probability(
    model: &mut student::Student,
    sequence: &EncodedSequence,
) -> Result<Tensor> {
    let logits = selected_logits(model, sequence)?;
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let targets = Tensor::from_slice(
        &sequence.targets,
        (sequence.targets.len(), 1),
        model.device(),
    )?;
    Ok(logp.gather(&targets, 1)?.sum_all()?)
}

fn adapter_tensors(model: &student::Student) -> BTreeMap<String, Tensor> {
    model
        .variables()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect()
}

fn artifact(
    spec: &DistillationWorkerSpec,
    m: &ExperimentalCheckpointManifest,
) -> Result<DistillationArtifact> {
    Ok(DistillationArtifact {
        schema_version: 1,
        arm_id: spec.arm_id.clone(),
        checkpoint: spec.output.clone(),
        manifest: Artefact::capture(&std::env::current_dir()?, spec.output.join("manifest.json"))?,
        adapter_sha256: m.adapter_sha256.clone(),
        base_sha256: m.base_sha256.clone(),
        evidence_bundle_sha256: m.evidence_bundle.sha256.clone(),
        steps: m.steps,
        losses: m.losses.clone(),
        gradient_norms: m.gradient_norms.clone(),
        changed_adapter: m.adapter_sha256 != m.initial_adapter_sha256,
        reloaded: m.reloaded,
        production_eligible: false,
        limitations: m.limitations.clone(),
    })
}

/// Run this blocking numerical worker in its dedicated subprocess or blocking
/// pool. Teacher capture finishes and drops every teacher before student loading.
pub fn run_worker(spec: &DistillationWorkerSpec) -> Result<DistillationArtifact> {
    validate(spec)?;
    let started = Instant::now();
    let root = std::env::current_dir()?;
    ensure!(
        !spec.output.join("train-health-stop.json").exists(),
        "predeclared train health policy already stopped this run; preserve the failed observation and use a separately declared experiment"
    );
    let assets = student::assets(spec)?;
    let base_hash = assets
        .files
        .get("model.safetensors")
        .context("student base weight digest absent")?
        .clone();
    fs::create_dir_all(&spec.output)?;
    let lease = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(spec.output.join(".training.lock"))?;
    // SAFETY: File owns the descriptor; a crash automatically releases flock.
    ensure!(
        unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another native worker owns this checkpoint"
    );
    if spec.output.join("manifest.json").exists() {
        let complete = checkpoint::validate_checkpoint(&spec.output, &assets)?;
        ensure!(
            serde_json::to_value(&complete.spec)? == serde_json::to_value(spec)?,
            "completed arm differs from requested immutable configuration"
        );
        return artifact(spec, &complete);
    }
    // A completed checkpoint's validator already reconstructs its evidence.
    // New work verifies the bundle here before selection or optimisation.
    let bundle = teacher_evidence::load_verified(&spec.evidence_bundle)?;
    validate_origins(&bundle)?;
    let tokenizer = Tokenizer::from_file(spec.student_base.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let Selection {
        order,
        samples,
        encoded,
        preferences,
        context_transforms,
        rejected_overlength,
        eligible_sequences: _,
    } = choose(spec, &bundle, &tokenizer)?;
    let initial = spec
        .initial_checkpoint
        .as_ref()
        .map(|path| {
            let m = checkpoint::validate_checkpoint(path, &assets)?;
            ensure!(
                m.adapter.rank == spec.rank
                    && m.adapter.alpha == spec.alpha
                    && m.spec.student_kind == spec.student_kind,
                "progressive initial adapter configuration differs"
            );
            Artefact::capture(&root, path.join("manifest.json"))
        })
        .transpose()?;
    let evidence = Artefact::capture(&root, spec.evidence_bundle.clone())?;
    let mut freeze = serde_json::json!({"schema_version":1,"spec":spec,"evidence":evidence,"base":assets,"engine_binary_sha256":artefact::file_hash(&std::env::current_exe()?)?,"order":order,"initial_checkpoint":initial,"encoded_sequences":encoded});
    if spec.schema_version >= 2 {
        let prepared = preflight_eligibility(spec)?;
        let scheduler_receipt = spec
            .output
            .parent()
            .context("checkpoint parent absent")?
            .join("prepared-selection.json");
        if scheduler_receipt.exists() {
            let planned: serde_json::Value =
                serde_json::from_slice(&fs::read(&scheduler_receipt)?)?;
            ensure!(
                planned == prepared,
                "planner/prewarm/worker prepared selection differs before optimisation"
            );
        }
        freeze["schema_version"] = 2.into();
        freeze["prepared_selection"] = prepared;
    }
    // Omit empty extensions so historical non-context/non-DPO freezes remain
    // byte-semantically valid and interrupted work can still be reverified.
    if !preferences.is_empty() {
        freeze["preference_sequences"] = serde_json::to_value(&preferences)?;
    }
    if !context_transforms.is_empty() {
        freeze["context_prompt_transforms"] = serde_json::to_value(&context_transforms)?;
    }
    if !rejected_overlength.is_empty() {
        freeze["excluded_overlength_samples"] = serde_json::to_value(&rejected_overlength)?;
    }
    let freeze_path = spec.output.join("freeze.json");
    if freeze_path.exists() {
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&freeze_path)?)? == freeze,
            "interrupted worker data/order/objective/seed/runtime/engine changed; use an explicit fresh output"
        );
    } else {
        artefact::write_json(&freeze_path, &freeze)?;
    }
    let freeze_hash = artefact::file_hash(&freeze_path)?;
    let traces = targets::prepare(spec, &bundle, &samples, &encoded, &tokenizer)?;
    let initial_adapter = spec
        .initial_checkpoint
        .as_ref()
        .map(|p| p.join("adapter.safetensors"));
    let mut student = load_student(spec, initial_adapter.as_deref())?;
    let initial_trits = student.effective_trits()?;
    let initial_path = spec.output.join("initial-adapter.safetensors");
    if !initial_path.exists() {
        checkpoint::save_tensors(&initial_path, &adapter_tensors(&student))?;
    }
    let saved_initial = candle_core::safetensors::load(&initial_path, student.device())?;
    ensure!(
        saved_initial.len() == student.variables().len(),
        "initial adapter tensor inventory changed"
    );
    for (name, var) in student.variables() {
        let saved = saved_initial
            .get(name)
            .context("initial adapter tensor absent")?;
        ensure!(
            saved.dims() == var.dims()
                && saved
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?
                    == var
                        .flatten_all()?
                        .to_device(&Device::Cpu)?
                        .to_vec1::<f32>()?,
            "cached initial adapter does not match the frozen seed/preceding checkpoint"
        );
    }
    drop(saved_initial);
    let initial_hash = artefact::file_hash(&initial_path)?;
    let cfg = TrainConfig {
        steps: spec.steps,
        seed: spec.seed,
        max_sequence_tokens: spec.max_sequence_tokens,
        rank: spec.rank,
        alpha: spec.alpha,
        learning_rate: spec.learning_rate,
        ..TrainConfig::default()
    };
    let mut references = BTreeMap::<String, (f64, f64)>::new();
    let preference = spec
        .losses
        .iter()
        .any(|l| l.kind == LossKind::PreferenceDpo);
    if preference {
        let path = spec.output.join("dpo-reference.json");
        let preference_sha256 = artefact::hash(&serde_json::to_vec(&preferences)?);
        if path.exists() {
            let saved: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
            ensure!(
                saved["initial_adapter_sha256"] == initial_hash
                    && saved["freeze_sha256"] == freeze_hash
                    && saved["preference_sequences_sha256"] == preference_sha256,
                "DPO reference checkpoint binding changed"
            );
            references = serde_json::from_value(saved["log_probabilities"].clone())?;
        } else {
            for (id, pair) in &preferences {
                let chosen = summed_log_probability(&mut student, &pair.chosen)?
                    .detach()
                    .to_scalar::<f32>()? as f64;
                let no = summed_log_probability(&mut student, &pair.rejected)?
                    .detach()
                    .to_scalar::<f32>()? as f64;
                references.insert(id.clone(), (chosen, no));
            }
            artefact::write_json(
                &path,
                &serde_json::json!({"initial_adapter_sha256":initial_hash,"freeze_sha256":freeze_hash,"preference_sequences_sha256":preference_sha256,"log_probabilities":references}),
            )?;
        }
        ensure!(
            references.len() == samples.len()
                && samples.keys().all(|id| references
                    .get(id)
                    .is_some_and(|(a, b)| a.is_finite() && b.is_finite())),
            "DPO reference coverage is incomplete or non-finite"
        );
    }
    let progress_path = spec.output.join("progress.json");
    let mut progress = if progress_path.exists() {
        let p: Progress = serde_json::from_slice(&fs::read(&progress_path)?)?;
        ensure!(
            p.freeze_sha256 == freeze_hash
                && p.initial_adapter_sha256 == initial_hash
                && p.steps <= spec.steps
                && p.losses.len() == p.steps
                && p.gradient_norms.len() == p.steps
                && p.component_losses.len() == p.steps
                && p.sample_ids.len() == p.steps,
            "saved optimiser progress differs from frozen work"
        );
        p.adapter.verify(&root)?;
        p.optimizer.verify(&root)?;
        ensure!(
            p.sample_ids
                == order
                    .iter()
                    .take(p.steps)
                    .map(|s| s.sample_id.clone())
                    .collect::<Vec<_>>()
                && p.losses.iter().all(|x| x.is_finite())
                && p.gradient_norms.iter().all(|x| x.is_finite() && *x > 0.),
            "saved updates differ from the frozen order or numerical evidence"
        );
        p
    } else {
        Progress {
            freeze_sha256: freeze_hash.clone(),
            steps: 0,
            adapter: Artefact::capture(&root, initial_path)?,
            optimizer: Artefact {
                path: spec.output.join("unused-initial-optimizer"),
                sha256: String::new(),
                bytes: 0,
            },
            initial_adapter_sha256: initial_hash.clone(),
            losses: vec![],
            gradient_norms: vec![],
            component_losses: vec![],
            alignment_diagnostics: vec![],
            sample_ids: vec![],
        }
    };
    if let Some(policy) = &spec.health_check {
        for step in policy
            .checkpoints
            .iter()
            .filter(|step| **step <= progress.steps)
        {
            let path = spec
                .output
                .join(format!("step-{step:06}/train-generation-health.json"));
            let saved:serde_json::Value=serde_json::from_slice(&fs::read(&path).with_context(||format!("committed update lacks its predeclared health observation at {}; interrupted evidence is retained without silently bypassing it",path.display()))?)?;
            ensure!(
                saved["policy"] == serde_json::to_value(policy)?
                    && saved["step"] == *step
                    && (!policy.stop_on_unhealthy || saved["generation_healthy"] == true),
                "saved train-only health policy stopped or changed; use a new declared experiment"
            );
        }
    }
    let mut optimiser = if progress.steps > 0 {
        drop(student);
        student = load_student(spec, Some(&progress.adapter.verify(&root)?))?;
        ResumableAdam::load(
            student.variables(),
            &cfg,
            &progress.optimizer.verify(&root)?,
            progress.steps,
        )?
    } else {
        ResumableAdam::new(student.variables(), &cfg)?
    };
    let mut semantic_ids = tokenizer.get_vocab(true).into_values().collect::<Vec<_>>();
    semantic_ids.sort_unstable();
    semantic_ids.dedup();
    let need_features = spec.losses.iter().any(|l| {
        matches!(
            l.kind,
            LossKind::HiddenFeature | LossKind::AttentionRelation
        )
    });
    let need_logits = spec
        .losses
        .iter()
        .any(|l| matches!(l.kind, LossKind::SequenceSft | LossKind::SoftTokenKl));
    for (step, plan) in order.iter().enumerate().skip(progress.steps) {
        let sequence = &encoded[&plan.sample_id];
        let device = student.device().clone();
        let input =
            Tensor::from_slice(&sequence.input_ids, (1, sequence.input_ids.len()), &device)?;
        let trace = if need_features {
            Some(student.forward_trace(&input, (student.layer_count() - 1) / 2)?)
        } else {
            None
        };
        let logits = if need_logits {
            let final_hidden = match &trace {
                Some(t) => t.final_hidden.clone(),
                None => student.hidden(&input)?,
            };
            let hidden = final_hidden.squeeze(0)?.index_select(
                &Tensor::from_slice(
                    &sequence.prediction_positions,
                    sequence.prediction_positions.len(),
                    &device,
                )?,
                0,
            )?;
            Some(student.logits(&hidden)?)
        } else {
            None
        };
        let targets_tensor =
            Tensor::from_slice(&sequence.targets, sequence.targets.len(), &device)?;
        let teacher = traces.get(&plan.sample_id);
        let teacher_tensors = teacher
            .map(|t| -> Result<_> {
                Ok(candle_core::safetensors::load(
                    t.record.tensors.verify(&root)?,
                    &device,
                )?)
            })
            .transpose()?;
        let mut total = None;
        let mut components = BTreeMap::new();
        let mut alignment_diagnostic = None;
        let normaliser = spec.losses.iter().map(|x| x.weight).sum::<f64>();
        for objective in &spec.losses {
            let loss = match objective.kind {
                LossKind::SequenceSft => losses::sequence_sft(
                    logits.as_ref().context("student SFT logits absent")?,
                    &targets_tensor,
                )?,
                LossKind::SoftTokenKl => {
                    let (loss, diagnostic) = losses::aligned_softened_kl(
                        logits.as_ref().context("student KL logits absent")?,
                        &teacher_tensors
                            .as_ref()
                            .context("native logit targets absent")?["logits"],
                        &semantic_ids,
                        spec.temperature,
                    )?;
                    alignment_diagnostic = Some(
                        serde_json::json!({"step":step+1,"teacher_id":plan.teacher_id,"sample_id":plan.sample_id,"padding_mass":diagnostic,"vocabulary_alignment":teacher.context("teacher alignment record absent")?.record.vocabulary_alignment}),
                    );
                    loss
                }
                LossKind::HiddenFeature | LossKind::AttentionRelation => {
                    let trace = trace
                        .as_ref()
                        .context("student feature trace was not captured")?;
                    let t = teacher.context("native feature targets absent")?;
                    let a = t.record.anchors.as_ref().context("byte alignment absent")?;
                    let index = Tensor::from_slice(&a.student_rows, a.student_rows.len(), &device)?;
                    let features = trace.layer_hidden.index_select(&index, 0)?;
                    let frozen = teacher_tensors
                        .as_ref()
                        .context("native feature tensors absent")?;
                    if objective.kind == LossKind::HiddenFeature {
                        losses::hidden_feature(&features, &frozen["hidden"])?
                    } else {
                        let attention = trace
                            .mean_attention
                            .index_select(&index, 0)?
                            .index_select(&index, 1)?;
                        losses::attention_relation(
                            &attention,
                            &frozen["attention"],
                            &features,
                            &frozen["hidden"],
                        )?
                    }
                }
                LossKind::PreferenceDpo => {
                    let pair = &preferences[&plan.sample_id];
                    let chosen = summed_log_probability(&mut student, &pair.chosen)?;
                    let no = summed_log_probability(&mut student, &pair.rejected)?;
                    let (c, r) = references[&plan.sample_id];
                    losses::preference_dpo(
                        &chosen,
                        &no,
                        &Tensor::new(c as f32, &device)?,
                        &Tensor::new(r as f32, &device)?,
                        spec.dpo_beta,
                    )?
                }
            };
            let scalar = loss.to_scalar::<f32>()? as f64;
            ensure!(
                scalar.is_finite(),
                "non-finite native distillation component"
            );
            components.insert(objective.kind, scalar);
            let weighted = (loss * (objective.weight / normaliser))?;
            total = Some(match total {
                None => weighted,
                Some(old) => (old + weighted)?,
            });
        }
        let loss = total.context("no distillation objective constructed")?;
        let scalar = loss.to_scalar::<f32>()? as f64;
        let before = adapter_tensors(&student);
        let norm = optimiser.update_selected(student.variables(), &loss)?;
        let directory = spec.output.join(format!("step-{:06}", step + 1));
        let update_norms = student
            .variables()
            .iter()
            .map(|(name, var)| -> Result<_> {
                let norm = (var.as_tensor() - &before[name])?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    .sqrt();
                ensure!(norm.is_finite(), "non-finite selected adapter update");
                Ok((name.clone(), norm))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        if spec.schema_version >= 2 {
            artefact::write_json(
                &directory.join("update-diagnostics.json"),
                &serde_json::json!({"schema_version":1,"sample_id":plan.sample_id,"step":step+1,"component_losses":components,"normalised_weights":spec.losses.iter().map(|l|serde_json::json!({"kind":l.kind,"weight":l.weight/normaliser})).collect::<Vec<_>>(),"gradient_norm":norm,"adapter_update_l2_by_tensor":update_norms,"interpretation":"Numerical objective/update evidence only; no coding improvement claim."}),
            )?;
        }
        checkpoint::save_tensors(
            &directory.join("adapter.safetensors"),
            &adapter_tensors(&student),
        )?;
        checkpoint::save_tensors(
            &directory.join("optimizer.safetensors"),
            &optimiser.tensors(),
        )?;
        progress.steps = step + 1;
        progress.losses.push(scalar);
        progress.gradient_norms.push(norm);
        progress.component_losses.push(components);
        if let Some(diagnostic) = alignment_diagnostic {
            progress.alignment_diagnostics.push(diagnostic);
        }
        progress.sample_ids.push(plan.sample_id.clone());
        progress.adapter = Artefact::capture(&root, directory.join("adapter.safetensors"))?;
        progress.optimizer = Artefact::capture(&root, directory.join("optimizer.safetensors"))?;
        artefact::write_json(&directory.join("progress.json"), &progress)?;
        artefact::write_json(&progress_path, &progress)?;
        if let Some(policy) = &spec.health_check
            && policy.checkpoints.contains(&(step + 1))
        {
            let model = match student {
                student::Student::Qwen(model) => model,
                student::Student::BitNet(_) => anyhow::bail!("health policy requires Qwen"),
            };
            let mut runtime = ModelRuntime::from_pinned_qwen_model(
                model,
                tokenizer.clone(),
                assets.clone(),
                progress.adapter.sha256.clone(),
                spec.student_runtime.clone(),
            );
            let first = &order[0].sample_id;
            let prompt = if spec
                .losses
                .iter()
                .all(|l| l.kind == LossKind::PreferenceDpo)
            {
                preferences[first].common_prompt.clone()
            } else {
                conditioning::student_prompt(&samples[first])
            };
            let request = model::GenerationRequest {
                prompt,
                max_new_tokens: policy.max_new_tokens,
                seed: spec.seed,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: policy.timeout_seconds,
            };
            let execution = runtime.execution_metadata();
            let generation = runtime
                .generate(request.clone())
                .map_err(|error| format!("{error:#}"));
            let healthy = generation
                .as_ref()
                .is_ok_and(|g| g.complete && !g.completion.trim().is_empty());
            runtime.model.clear_cache();
            student = student::Student::Qwen(runtime.model);
            artefact::write_json(
                &directory.join("train-generation-health.json"),
                &serde_json::json!({"schema_version":1,"policy":policy,"step":step+1,"sample_id":first,"task_id":samples[first].task.id,"source_id":samples[first].task.source_id,"adapter":progress.adapter,"request":request,"execution":execution,"generation":generation,"generation_healthy":healthy,"sandbox_correctness_measured":false,"held_out_used":false,"interpretation":"Predeclared train-only generation health; normal termination does not establish correct code."}),
            )?;
            if !healthy && policy.stop_on_unhealthy {
                artefact::write_json(
                    &spec.output.join("train-health-stop.json"),
                    &serde_json::json!({"step":step+1,"policy":policy,"observation":Artefact::capture(&root,directory.join("train-generation-health.json"))?}),
                )?;
            }
            ensure!(
                healthy || !policy.stop_on_unhealthy,
                "predeclared train-only generation health stop at step {}; failed observation and numerical checkpoint retained",
                step + 1
            );
        }
        eprintln!(
            "distillation: durable update {}/{} loss={scalar:.6} gradient_norm={norm:.6}",
            step + 1,
            spec.steps
        );
    }
    let final_adapter = spec.output.join("adapter.safetensors");
    let final_optimizer = spec.output.join("optimizer.safetensors");
    fs::copy(progress.adapter.verify(&root)?, &final_adapter)?;
    fs::File::open(&final_adapter)?.sync_all()?;
    fs::copy(progress.optimizer.verify(&root)?, &final_optimizer)?;
    fs::File::open(&final_optimizer)?.sync_all()?;
    let expected = student
        .variables()
        .iter()
        .map(|(k, v)| {
            Ok((
                k.clone(),
                v.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let reloaded = candle_core::safetensors::load(&final_adapter, &Device::Cpu)?;
    ensure!(
        reloaded.len() == expected.len()
            && expected.iter().all(|(k, v)| reloaded
                .get(k)
                .and_then(|t| t.flatten_all().ok())
                .and_then(|t| t.to_vec1::<f32>().ok())
                .as_ref()
                == Some(v)),
        "exported experimental adapter failed exact reload"
    );
    let final_trits = student.effective_trits()?;
    let student_execution = student.execution_metadata();
    let qat_effective_trits = match (&initial_trits, &final_trits) {
        (Some(initial), Some(final_state)) => {
            let evidence = serde_json::json!({"initial":initial,"final":final_state,"semantics":"Actual ternary q/v forward lattice after STE QAT, including names and scales; distinct from latent adapter identity."});
            if initial.sha256 == final_state.sha256 || final_state.changed_from_frozen_base == 0 {
                artefact::write_json(
                    &spec.output.join("blocked-no-discrete-update.json"),
                    &serde_json::json!({"status":"blocked_no_discrete_ternary_update","steps":progress.steps,"adapter":progress.adapter,"effective_trits":evidence,"reason":"Real latent optimiser updates crossed no additional ternary thresholds within the preregistered budget; this is not a successful adapted ternary checkpoint."}),
                )?;
                anyhow::bail!(
                    "BitNet QAT changed latent adapters but no effective ternary weights; declared update budget exhausted, progress preserved"
                );
            }
            Some(evidence)
        }
        (None, None) => None,
        _ => anyhow::bail!("student backend changed during training"),
    };
    student.device().synchronize()?;
    drop(student);
    let mut fresh = student::load_for_inference(spec, &final_adapter)?;
    if let Some(expected) = &final_trits {
        ensure!(
            fresh
                .effective_trits()?
                .is_some_and(|actual| actual.sha256 == expected.sha256),
            "fresh BitNet reload changed its effective ternary lattice"
        );
    }
    let probe = &encoded[&order[0].sample_id];
    ensure!(
        selected_logits(&mut fresh, probe)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()),
        "fresh model reload produced non-finite logits"
    );
    drop(fresh);
    let after = artefact::file_hash(&spec.student_base.join("model.safetensors"))?;
    ensure!(
        after == base_hash,
        "frozen student base changed during experimental optimisation"
    );
    let mut limitations = bundle.limitations.clone();
    limitations.push("Exploratory local-teacher adaptation; no independent API-provider or held-out promotion claim. Reload is a fresh model instance inside the worker; external evaluation must reload in its own process.".into());
    if spec.sequence_policy == SequencePolicy::AssistantWindow {
        limitations.push("Declared assistant-token windows exercise actual operators and optimiser updates; omitted context/targets are recorded in freeze.json and these quick arms do not establish task learning.".into());
    }
    let manifest = ExperimentalCheckpointManifest {
        schema_version: spec.schema_version,
        evidence_kind: "verified_local_teacher_experimental_distillation".into(),
        spec: spec.clone(),
        freeze_sha256: freeze_hash,
        base_sha256: base_hash,
        base_sha256_after: after,
        tokenizer_sha256: assets.files["tokenizer.json"].clone(),
        evidence_bundle: evidence,
        adapter: AdapterConfig {
            rank: spec.rank,
            alpha: spec.alpha,
        },
        adapter_sha256: artefact::file_hash(&final_adapter)?,
        optimizer_sha256: artefact::file_hash(&final_optimizer)?,
        initial_adapter_sha256: initial_hash,
        initial_adapter: Artefact::capture(&root, spec.output.join("initial-adapter.safetensors"))?,
        initial_checkpoint: initial,
        dpo_reference: if preference {
            Some(Artefact::capture(
                &root,
                spec.output.join("dpo-reference.json"),
            )?)
        } else {
            None
        },
        steps: progress.steps,
        losses: progress.losses,
        gradient_norms: progress.gradient_norms,
        component_losses: progress.component_losses,
        alignment_diagnostics: progress.alignment_diagnostics,
        qat_effective_trits,
        sample_ids: progress.sample_ids,
        realised_update_counts: {
            let mut counts = BTreeMap::<String, usize>::new();
            for plan in &order {
                *counts
                    .entry(format!("teacher/{}", plan.teacher_id))
                    .or_default() += 1;
                let content = serde_json::to_value(samples[&plan.sample_id].content)?
                    .as_str()
                    .context("content label absent")?
                    .to_owned();
                *counts.entry(format!("content/{content}")).or_default() += 1;
                if preference {
                    *counts.entry("content/preference".into()).or_default() += 1;
                }
            }
            counts
        },
        trace_records: traces.values().map(|t| t.artefact.clone()).collect(),
        teacher_rights: serde_json::to_value(
            bundle
                .teachers
                .iter()
                .filter(|t| spec.teacher_ids.contains(&t.id))
                .collect::<Vec<_>>(),
        )?,
        student_execution,
        reloaded: true,
        production_eligible: false,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        limitations,
        training_health: spec
            .health_check
            .as_ref()
            .map(|p| {
                p.checkpoints
                    .iter()
                    .map(|step| {
                        Artefact::capture(
                            &root,
                            spec.output
                                .join(format!("step-{step:06}/train-generation-health.json")),
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default(),
    };
    ensure!(
        manifest.adapter_sha256 != manifest.initial_adapter_sha256,
        "native training left adapter unchanged"
    );
    artefact::write_json(&spec.output.join("manifest.json"), &manifest)?;
    checkpoint::validate_checkpoint(&spec.output, &assets)?;
    artifact(spec, &manifest)
}

/// This loader is explicitly experimental; the assessment loader continues to
/// reject this manifest kind. Callers must preserve evaluation-only serving.
pub fn load_runtime(
    base: &Path,
    checkpoint: &Path,
    options: model::RuntimeOptions,
) -> Result<ModelRuntime> {
    let assets = model::validate_assets(base)?;
    let m = checkpoint::validate_checkpoint(checkpoint, &assets)?;
    // validate_checkpoint has already independently reconstructed the exact
    // hash-bound evidence bundle in this process. No file-stat cache is used.
    let mut spec = m.spec.clone();
    spec.student_base = base.to_owned();
    spec.student_runtime = options.clone();
    let model = match student::load_for_inference(&spec, &checkpoint.join("adapter.safetensors"))? {
        student::Student::Qwen(model) => model,
        student::Student::BitNet(_) => {
            anyhow::bail!("load native BitNet checkpoints through BitNetRuntime")
        }
    };
    let tokenizer = Tokenizer::from_file(base.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(ModelRuntime::from_pinned_qwen_model(
        model,
        tokenizer,
        assets,
        m.adapter_sha256,
        options,
    ))
}

/// Prewarm native targets once on the reserved teacher device. Subsequent CPU
/// arms use CacheOnly and therefore cannot unexpectedly allocate GPU memory.
pub fn prepare_teacher_targets(spec: &DistillationWorkerSpec) -> Result<Vec<Artefact>> {
    validate(spec)?;
    let bundle = teacher_evidence::load_verified(&spec.evidence_bundle)?;
    validate_origins(&bundle)?;
    student::assets(spec)?;
    let tokenizer = Tokenizer::from_file(spec.student_base.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let Selection {
        samples, encoded, ..
    } = choose(spec, &bundle, &tokenizer)?;
    Ok(
        targets::prepare(spec, &bundle, &samples, &encoded, &tokenizer)?
            .values()
            .map(|t| t.artefact.clone())
            .collect(),
    )
}
