//! Assistant-only supervised LoRA training, durable optimiser state and provenance.
//!
//! The curator's complete validation runs before model construction or an optimiser
//! step. Reference solutions and unit-test tensors are never assessment data.

use crate::model::{
    self, AdapterConfig, AdapterVariables, AssetManifest, QwenConfig, QwenModel, StableRng,
    atomic_write, sha256_file,
};
use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    #[serde(default)]
    pub purpose: TrainingPurpose,
    pub seed: u64,
    pub steps: usize,
    pub max_sequence_tokens: usize,
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub max_gradient_norm: f64,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingPurpose {
    #[default]
    Candidate,
    SeedControl,
    DegradedControl,
    OverfitControl,
}
impl TrainingPurpose {
    pub fn target_transform(self) -> &'static str {
        match self {
            Self::DegradedControl => "replace_with_declared_incorrect_constant_rust_v1",
            _ => "identity_provider_winner",
        }
    }
    fn completion(self, original: &str) -> String {
        match self{Self::DegradedControl=>"pub fn solve(_input: serde_json::Value) -> serde_json::Value { serde_json::Value::Null }".into(),_=>original.into()}
    }
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            purpose: TrainingPurpose::Candidate,
            seed: 1701,
            steps: 48,
            max_sequence_tokens: 512,
            rank: 8,
            alpha: 16.,
            learning_rate: 0.0002,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            max_gradient_norm: 1.,
        }
    }
}
impl TrainConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let cfg: Self = toml::from_str(&fs::read_to_string(path)?)?;
        cfg.validate()?;
        Ok(cfg)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.steps > 0 && self.steps <= 100000,
            "training steps must be 1–100000"
        );
        ensure!(
            (16..=4096).contains(&self.max_sequence_tokens),
            "sequence cap must be 16–4096"
        );
        ensure!(
            self.rank > 0 && self.rank <= 128 && self.alpha.is_finite() && self.alpha > 0.,
            "invalid adapter configuration"
        );
        ensure!(
            self.learning_rate.is_finite() && self.learning_rate > 0. && self.learning_rate <= 0.1,
            "invalid learning rate"
        );
        ensure!(
            self.beta1.is_finite()
                && (0.0..1.0).contains(&self.beta1)
                && self.beta2.is_finite()
                && (0.0..1.0).contains(&self.beta2),
            "invalid Adam coefficients"
        );
        ensure!(
            self.epsilon.is_finite()
                && self.epsilon > 0.
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.
                && self.max_gradient_norm.is_finite()
                && self.max_gradient_norm > 0.,
            "invalid optimiser settings"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub base_revision: String,
    pub base_sha256: String,
    pub tokenizer_sha256: String,
    pub adapter: AdapterConfig,
    pub adapter_sha256: String,
    pub optimizer_sha256: String,
    pub dataset_sha256: String,
    pub dataset_manifest_sha256: String,
    pub heldout_index_sha256: String,
    pub config_sha256: String,
    pub seed: u64,
    pub steps: usize,
    pub effective_examples: usize,
    pub rejected_overlength: Vec<String>,
    pub initial_adapter_sha256: String,
    pub losses: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub elapsed_seconds: f64,
    pub evidence_kind: String,
    pub training_purpose: TrainingPurpose,
    pub target_transform: String,
}

pub fn validate_checkpoint(
    path: &Path,
    m: &CheckpointManifest,
    assets: &AssetManifest,
) -> Result<()> {
    ensure!(
        m.schema_version == 1
            && m.model_id == model::MODEL_ID
            && m.base_revision == assets.revision
            && m.base_sha256 == model::MODEL_SHA256
            && m.tokenizer_sha256 == assets.files["tokenizer.json"],
        "checkpoint base/tokenizer mismatch"
    );
    ensure!(
        m.evidence_kind == "curated_provider_training",
        "checkpoint is not curated provider training evidence"
    );
    ensure!(
        m.target_transform == m.training_purpose.target_transform(),
        "checkpoint training purpose/target transform mismatch"
    );
    ensure!(
        m.steps > 0
            && m.effective_examples > 0
            && m.losses.len() == m.steps
            && m.gradient_norms.len() == m.steps,
        "incomplete checkpoint training evidence"
    );
    ensure!(
        m.losses.iter().all(|v| v.is_finite())
            && m.gradient_norms.iter().all(|v| v.is_finite() && *v > 0.),
        "invalid loss/gradient evidence"
    );
    ensure!(
        m.adapter_sha256 != m.initial_adapter_sha256,
        "unchanged adapter is not a trained checkpoint"
    );
    ensure!(
        sha256_file(&path.join("adapter.safetensors"))? == m.adapter_sha256,
        "adapter tensor checksum mismatch"
    );
    ensure!(
        sha256_file(&path.join("optimizer.safetensors"))? == m.optimizer_sha256,
        "optimiser checksum mismatch"
    );
    Ok(())
}

/// Only target positions whose entire text range belongs to the assistant are learnt.
/// A BPE token straddling the prompt boundary is deliberately excluded.
#[derive(Debug)]
pub struct EncodedSample {
    pub id: String,
    pub input_ids: Vec<u32>,
    pub positions: Vec<u32>,
    pub targets: Vec<u32>,
}
pub fn encode_sample(
    tokenizer: &Tokenizer,
    id: &str,
    prompt: &str,
    completion: &str,
    max_tokens: usize,
) -> Result<EncodedSample> {
    ensure!(!completion.trim().is_empty(), "empty assistant target");
    let prefix = model::chat_prefix(prompt);
    let boundary = prefix.len();
    let text = format!("{prefix}{completion}<|im_end|>");
    let enc = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let ids = enc.get_ids();
    ensure!(
        ids.len() >= 2 && ids.len() <= max_tokens,
        "sample {id} exceeds sequence cap ({}/{max_tokens})",
        ids.len()
    );
    let mut positions = Vec::new();
    let mut targets = Vec::new();
    for (i, &target) in ids.iter().enumerate().skip(1) {
        let (start, end) = enc.get_offsets()[i];
        if start >= boundary && end > start {
            positions.push((i - 1) as u32);
            targets.push(target);
        }
    }
    ensure!(
        !targets.is_empty(),
        "sample {id} has no assistant-only learnable tokens"
    );
    Ok(EncodedSample {
        id: id.into(),
        input_ids: ids[..ids.len() - 1].to_vec(),
        positions,
        targets,
    })
}

pub fn assistant_loss(model: &mut QwenModel, sample: &EncodedSample) -> Result<Tensor> {
    model.clear_cache();
    let input = Tensor::from_slice(
        &sample.input_ids,
        (1, sample.input_ids.len()),
        model.device(),
    )?;
    let hidden = model.hidden(&input, 0, false)?.squeeze(0)?;
    // Project only supervised positions: the large vocabulary head need not allocate
    // prompt logits, and prompt targets never enter the objective.
    let selected = hidden.index_select(
        &Tensor::from_slice(&sample.positions, sample.positions.len(), model.device())?,
        0,
    )?;
    let logits = model.logits(&selected)?;
    let log_prob = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let targets = Tensor::from_slice(&sample.targets, (sample.targets.len(), 1), model.device())?;
    Ok(log_prob.gather(&targets, 1)?.mean_all()?.neg()?)
}

struct Moment {
    m: Tensor,
    v: Tensor,
}
pub(crate) struct ResumableAdam {
    moments: BTreeMap<String, Moment>,
    pub step: usize,
    config: TrainConfig,
}
impl ResumableAdam {
    pub(crate) fn new(vars: &AdapterVariables, cfg: &TrainConfig) -> Result<Self> {
        let moments = vars
            .iter()
            .map(|(k, var)| {
                Ok((
                    k.clone(),
                    Moment {
                        m: var.zeros_like()?,
                        v: var.zeros_like()?,
                    },
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            moments,
            step: 0,
            config: cfg.clone(),
        })
    }
    fn load(vars: &AdapterVariables, cfg: &TrainConfig, path: &Path, step: usize) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
        ensure!(
            tensors.len() == vars.len() * 2,
            "optimiser state tensor count mismatch"
        );
        let moments = vars
            .iter()
            .map(|(key, var)| {
                let m = tensors
                    .get(&format!("{key}.m"))
                    .context("missing first moment")?
                    .clone();
                let v = tensors
                    .get(&format!("{key}.v"))
                    .context("missing second moment")?
                    .clone();
                ensure!(
                    m.shape() == var.shape()
                        && v.shape() == var.shape()
                        && m.dtype() == DType::F32
                        && v.dtype() == DType::F32,
                    "optimiser tensor shape/dtype mismatch"
                );
                Ok((key.clone(), Moment { m, v }))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            moments,
            step,
            config: cfg.clone(),
        })
    }
    pub(crate) fn update(&mut self, vars: &AdapterVariables, loss: &Tensor) -> Result<f64> {
        let grads = loss
            .backward()
            .context("differentiable training backward failed")?;
        let mut norm2 = 0.;
        let mut seen = 0;
        for var in vars.values() {
            let g = grads.get(var).context("missing adapter gradient")?;
            let n = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            ensure!(n.is_finite(), "non-finite adapter gradient");
            norm2 += n;
            seen += 1;
        }
        let norm = norm2.sqrt();
        ensure!(
            seen > 0 && norm.is_finite() && norm > 0.,
            "zero or invalid adapter gradient norm"
        );
        let clip = (self.config.max_gradient_norm / norm).min(1.);
        let step = self.step + 1;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        // Construct every update before mutating any variable. Numeric failures cannot
        // leave a half-updated in-memory optimiser, and durable state is published last.
        let mut updates = Vec::new();
        for (key, var) in vars {
            let moment = &self.moments[key];
            let g = (grads.get(var).context("gradient disappeared")? * clip)?;
            let m = ((&moment.m * b1)? + (&g * (1. - b1))?)?;
            let v = ((&moment.v * b2)? + (g.sqr()? * (1. - b2))?)?;
            let mh = (&m / (1. - b1.powf(step as f64)))?;
            let vh = (&v / (1. - b2.powf(step as f64)))?;
            let delta = (mh / (vh.sqrt()? + self.config.epsilon)?)?;
            let next = ((var.as_tensor()
                * (1. - self.config.learning_rate * self.config.weight_decay))?
                - (delta * self.config.learning_rate)?)?;
            ensure!(
                next.sqr()?.sum_all()?.to_scalar::<f32>()?.is_finite(),
                "non-finite adapter update"
            );
            updates.push((key.clone(), next.detach(), m.detach(), v.detach()));
        }
        for (key, next, m, v) in updates {
            vars[&key].set(&next)?;
            self.moments.insert(key, Moment { m, v });
        }
        self.step = step;
        Ok(norm)
    }
    fn tensors(&self) -> BTreeMap<String, Tensor> {
        self.moments
            .iter()
            .flat_map(|(k, m)| {
                [
                    (format!("{k}.m"), m.m.clone()),
                    (format!("{k}.v"), m.v.clone()),
                ]
            })
            .collect()
    }
}

fn save_tensors(path: &Path, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let tmp = path.with_extension("safetensors.tmp");
    let named = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    candle_core::safetensors::save(&named, &tmp)?;
    fs::File::open(&tmp)?.sync_all()?;
    fs::rename(tmp, path)?;
    fs::File::open(path.parent().context("tensor parent")?)?.sync_all()?;
    Ok(())
}
fn adapter_tensors(vars: &AdapterVariables) -> BTreeMap<String, Tensor> {
    vars.iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect()
}
fn json_hash<T: Serialize>(value: &T) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

#[derive(Serialize, Deserialize)]
struct Progress {
    manifest: CheckpointManifest,
    completed_directory: String,
}

/// Execute genuine CPU float32 SFT. A missing/unapproved dataset is a hard failure.
pub fn run(
    dataset_dir: &Path,
    heldout_index: &Path,
    base_dir: &Path,
    cfg: TrainConfig,
    output_dir: &Path,
    resume: bool,
) -> Result<CheckpointManifest> {
    let started = Instant::now();
    cfg.validate()?;
    // This call is intentionally before loading weights, preparing variables or
    // opening an optimiser: planted held-out rows cannot consume a training step.
    let validated = crate::curate::validate_dataset(dataset_dir, heldout_index)?;
    ensure!(
        !validated.rows.is_empty(),
        "curated dataset contains no training samples"
    );
    let assets = model::validate_assets(base_dir)?;
    ensure!(
        !output_dir.join("manifest.json").exists(),
        "checkpoint already completed; immutable output requires a new directory"
    );
    fs::create_dir_all(output_dir)?;
    let lock_path = output_dir.join(".training.lock");
    let lock = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)?;
    // An advisory lock is released by the kernel even after SIGKILL.
    use std::os::fd::AsRawFd;
    // SAFETY: flock acts on a live file descriptor and does not access Rust memory.
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another trainer owns this checkpoint directory"
    );
    let tokenizer = Tokenizer::from_file(base_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut samples = Vec::new();
    let mut rejected = Vec::new();
    for row in &validated.rows {
        let completion = cfg.purpose.completion(&row.completion);
        let prefix = model::chat_prefix(&row.prompt);
        let count = tokenizer
            .encode(format!("{prefix}{completion}<|im_end|>"), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .len();
        if count > cfg.max_sequence_tokens {
            rejected.push(row.id.clone());
            continue;
        }
        samples.push(encode_sample(
            &tokenizer,
            &row.id,
            &row.prompt,
            &completion,
            cfg.max_sequence_tokens,
        )?);
    }
    ensure!(
        !samples.is_empty(),
        "all curated rows exceeded the sequence cap; no optimiser was opened"
    );
    let config_sha256 = json_hash(&cfg)?;
    let progress_path = output_dir.join("progress.json");
    let progress = if progress_path.exists() {
        ensure!(resume, "partial training exists; --resume is required");
        Some(serde_json::from_slice::<Progress>(&fs::read(
            &progress_path,
        )?)?)
    } else {
        None
    };
    if let Some(p) = &progress {
        ensure!(
            p.manifest.config_sha256 == config_sha256
                && p.manifest.dataset_sha256 == validated.manifest.data_sha256
                && p.manifest.heldout_index_sha256 == validated.manifest.heldout_index_sha256,
            "resume data/config differs from saved training"
        );
        ensure!(
            p.completed_directory == format!("step-{:06}", p.manifest.steps),
            "invalid resume directory"
        );
        validate_checkpoint(
            &output_dir.join(&p.completed_directory),
            &p.manifest,
            &assets,
        )?;
    }
    let weights = candle_core::safetensors::load(base_dir.join("model.safetensors"), &Device::Cpu)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    let base_cfg: QwenConfig = serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
    let adapter = AdapterConfig {
        rank: cfg.rank,
        alpha: cfg.alpha,
    };
    let loaded = progress
        .as_ref()
        .map(|p| {
            candle_core::safetensors::load(
                output_dir
                    .join(&p.completed_directory)
                    .join("adapter.safetensors"),
                &Device::Cpu,
            )
        })
        .transpose()?;
    let mut model = QwenModel::from_weights(
        base_cfg,
        weights,
        Some(adapter.clone()),
        cfg.seed,
        loaded,
        true,
    )?;
    let mut optimizer = if let Some(p) = &progress {
        ResumableAdam::load(
            &model.adapter_vars,
            &cfg,
            &output_dir
                .join(&p.completed_directory)
                .join("optimizer.safetensors"),
            p.manifest.steps,
        )?
    } else {
        ResumableAdam::new(&model.adapter_vars, &cfg)?
    };
    let mut manifest = if let Some(p) = progress {
        p.manifest
    } else {
        let initial = output_dir.join("initial-adapter.safetensors");
        save_tensors(&initial, &adapter_tensors(&model.adapter_vars))?;
        CheckpointManifest {
            schema_version: 1,
            model_id: model::MODEL_ID.into(),
            base_revision: assets.revision.clone(),
            base_sha256: model::MODEL_SHA256.into(),
            tokenizer_sha256: assets.files["tokenizer.json"].clone(),
            adapter,
            adapter_sha256: String::new(),
            optimizer_sha256: String::new(),
            dataset_sha256: validated.manifest.data_sha256.clone(),
            dataset_manifest_sha256: validated.manifest.manifest_sha256.clone(),
            heldout_index_sha256: validated.manifest.heldout_index_sha256.clone(),
            config_sha256,
            seed: cfg.seed,
            steps: 0,
            effective_examples: samples.len(),
            rejected_overlength: rejected,
            initial_adapter_sha256: sha256_file(&initial)?,
            losses: Vec::new(),
            gradient_norms: Vec::new(),
            elapsed_seconds: 0.,
            evidence_kind: "curated_provider_training".into(),
            training_purpose: cfg.purpose,
            target_transform: cfg.purpose.target_transform().into(),
        }
    };
    ensure!(
        manifest.effective_examples == samples.len(),
        "effective sample count changed on resume"
    );
    let previous_elapsed = manifest.elapsed_seconds;
    atomic_write(
        &output_dir.join("training-config.json"),
        &serde_json::to_vec_pretty(&cfg)?,
    )?;
    while optimizer.step < cfg.steps {
        let epoch = optimizer.step / samples.len();
        let offset = optimizer.step % samples.len();
        let mut order = (0..samples.len()).collect::<Vec<_>>();
        StableRng::new(cfg.seed ^ (epoch as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .shuffle(&mut order);
        let sample = &samples[order[offset]];
        let step_started = Instant::now();
        let loss = assistant_loss(&mut model, sample)?;
        let scalar = loss.to_scalar::<f32>()? as f64;
        ensure!(scalar.is_finite(), "non-finite SFT loss");
        let gradient_norm = optimizer.update(&model.adapter_vars, &loss)?;
        manifest.steps = optimizer.step;
        manifest.losses.push(scalar);
        manifest.gradient_norms.push(gradient_norm);
        manifest.elapsed_seconds = previous_elapsed + started.elapsed().as_secs_f64();
        let name = format!("step-{:06}", optimizer.step);
        let dir = output_dir.join(&name);
        fs::create_dir_all(&dir)?;
        // Crash before publication leaves an unreferenced directory, never a completed
        // step. Restart loads only the state named by the durable progress pointer.
        save_tensors(
            &dir.join("adapter.safetensors"),
            &adapter_tensors(&model.adapter_vars),
        )?;
        save_tensors(&dir.join("optimizer.safetensors"), &optimizer.tensors())?;
        manifest.adapter_sha256 = sha256_file(&dir.join("adapter.safetensors"))?;
        manifest.optimizer_sha256 = sha256_file(&dir.join("optimizer.safetensors"))?;
        ensure!(
            manifest.adapter_sha256 != manifest.initial_adapter_sha256,
            "optimiser did not change adapter tensors"
        );
        atomic_write(
            &dir.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest)?,
        )?;
        atomic_write(
            &progress_path,
            &serde_json::to_vec_pretty(&Progress {
                manifest: manifest.clone(),
                completed_directory: name,
            })?,
        )?;
        eprintln!(
            "training: step {}/{} sample={} assistant_tokens={} loss={scalar:.6} gradient_norm={gradient_norm:.6} elapsed={:.2}s",
            optimizer.step,
            cfg.steps,
            sample.id,
            sample.targets.len(),
            step_started.elapsed().as_secs_f64()
        );
    }
    let last = output_dir.join(format!("step-{:06}", manifest.steps));
    for name in ["adapter.safetensors", "optimizer.safetensors"] {
        fs::copy(last.join(name), output_dir.join(name))?;
        fs::File::open(output_dir.join(name))?.sync_all()?;
    }
    validate_checkpoint(output_dir, &manifest, &assets)?;
    atomic_write(
        &output_dir.join("manifest.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
    // JSONL contains only measured state. It can be regenerated from the durable
    // manifest after an interrupted publication without repeating training.
    let log = manifest
        .losses
        .iter()
        .zip(&manifest.gradient_norms)
        .enumerate()
        .map(|(i, (loss, norm))| {
            serde_json::to_string(
                &serde_json::json!({"step":i+1,"loss":loss,"gradient_norm":norm,"seed":cfg.seed}),
            )
        })
        .collect::<std::result::Result<Vec<_>, _>>()?
        .join("\n")
        + "\n";
    atomic_write(&output_dir.join("training-log.jsonl"), log.as_bytes())?;
    drop(lock);
    Ok(manifest)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NativeGradientSmoke {
    pub evidence_kind: String,
    pub base_sha256: String,
    pub sequence_tokens: usize,
    pub supervised_tokens: usize,
    pub trainable_parameters: usize,
    pub adapter_variables: usize,
    pub loss: f64,
    pub gradient_norm: f64,
    pub changed_parameter_l2: f64,
    pub frozen_base_sentinel_unchanged: bool,
    pub adapter_safetensors_roundtrip: bool,
    pub elapsed_seconds: f64,
}

/// Profile one complete forward/backward/update using synthetic token tensors.
/// No assessment checkpoint is published and no provider-data claim is made.
pub fn native_gradient_smoke(
    base_dir: &Path,
    sequence_tokens: usize,
) -> Result<NativeGradientSmoke> {
    ensure!(
        (8..=512).contains(&sequence_tokens),
        "gradient smoke sequence must be 8–512 tokens"
    );
    model::validate_assets(base_dir)?;
    let started = Instant::now();
    let weights = candle_core::safetensors::load(base_dir.join("model.safetensors"), &Device::Cpu)?
        .into_iter()
        .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
        .collect::<Result<HashMap<_, _>>>()?;
    use candle_core::IndexOp;
    let sentinel = weights["model.embed_tokens.weight"]
        .i(0)?
        .narrow(0, 0, 16)?;
    let sentinel_values = sentinel.to_vec1::<f32>()?;
    let cfg: QwenConfig = serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
    let mut qwen = QwenModel::from_weights(
        cfg,
        weights,
        Some(AdapterConfig::default()),
        1701,
        None,
        true,
    )?;
    let mut rng = StableRng::new(42);
    let ids = (0..=sequence_tokens)
        .map(|_| (rng.next_u64() % 1000 + 1) as u32)
        .collect::<Vec<_>>();
    let first = sequence_tokens / 2;
    let sample = EncodedSample {
        id: "synthetic token tensor: technical QA only".into(),
        input_ids: ids[..sequence_tokens].to_vec(),
        positions: (first..sequence_tokens).map(|i| i as u32).collect(),
        targets: ids[first + 1..].to_vec(),
    };
    let before = qwen
        .adapter_vars
        .iter()
        .map(|(k, v)| Ok((k.clone(), v.copy()?.detach())))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let loss = assistant_loss(&mut qwen, &sample)?;
    let scalar = loss.to_scalar::<f32>()? as f64;
    ensure!(scalar.is_finite(), "non-finite smoke objective");
    let mut opt = ResumableAdam::new(&qwen.adapter_vars, &TrainConfig::default())?;
    let gradient_norm = opt.update(&qwen.adapter_vars, &loss)?;
    let mut change = 0.;
    for (k, var) in &qwen.adapter_vars {
        change += (var.as_tensor() - &before[k])?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()? as f64;
    }
    ensure!(
        change.is_finite() && change > 0.,
        "full-base gradient smoke did not update adapter"
    );
    let temp = tempfile::tempdir()?;
    let saved = temp.path().join("synthetic-invariant-only.safetensors");
    save_tensors(&saved, &adapter_tensors(&qwen.adapter_vars))?;
    let reloaded = candle_core::safetensors::load(&saved, &Device::Cpu)?;
    let mut roundtrip = true;
    for (k, var) in &qwen.adapter_vars {
        roundtrip &=
            var.flatten_all()?.to_vec1::<f32>()? == reloaded[k].flatten_all()?.to_vec1::<f32>()?;
    }
    ensure!(roundtrip, "adapter safetensors roundtrip failed");
    let frozen = sentinel.to_vec1::<f32>()? == sentinel_values;
    ensure!(frozen, "base sentinel changed");
    Ok(NativeGradientSmoke {
        evidence_kind: "synthetic_tensor_invariant_only_not_assessment_training".into(),
        base_sha256: model::MODEL_SHA256.into(),
        sequence_tokens,
        supervised_tokens: sample.targets.len(),
        trainable_parameters: qwen.adapter_vars.values().map(|v| v.elem_count()).sum(),
        adapter_variables: qwen.adapter_vars.len(),
        loss: scalar,
        gradient_norm,
        changed_parameter_l2: change.sqrt(),
        frozen_base_sentinel_unchanged: frozen,
        adapter_safetensors_roundtrip: roundtrip,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;
    #[test]
    fn adam_resume_is_numerically_equivalent() {
        let cfg = TrainConfig::default();
        let mut vars = BTreeMap::new();
        vars.insert(
            "adapter".into(),
            Var::new(&[1_f32, 2.], &Device::Cpu).unwrap(),
        );
        let mut opt = ResumableAdam::new(&vars, &cfg).unwrap();
        let loss = vars["adapter"].sqr().unwrap().sum_all().unwrap();
        opt.update(&vars, &loss).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("optim.safetensors");
        save_tensors(&path, &opt.tensors()).unwrap();
        let mut restored_vars = BTreeMap::new();
        restored_vars.insert(
            "adapter".into(),
            Var::from_tensor(vars["adapter"].as_tensor()).unwrap(),
        );
        let mut restored = ResumableAdam::load(&restored_vars, &cfg, &path, 1).unwrap();
        opt.update(&vars, &vars["adapter"].sqr().unwrap().sum_all().unwrap())
            .unwrap();
        restored
            .update(
                &restored_vars,
                &restored_vars["adapter"].sqr().unwrap().sum_all().unwrap(),
            )
            .unwrap();
        assert_eq!(
            vars["adapter"].to_vec1::<f32>().unwrap(),
            restored_vars["adapter"].to_vec1::<f32>().unwrap()
        );
        assert_eq!(restored.step, 2);
    }
    #[test]
    fn non_finite_loss_cannot_update_weights() {
        let cfg = TrainConfig::default();
        let mut vars = BTreeMap::new();
        vars.insert("adapter".into(), Var::new(&[1_f32], &Device::Cpu).unwrap());
        let mut opt = ResumableAdam::new(&vars, &cfg).unwrap();
        let loss = (vars["adapter"].sum_all().unwrap() * f64::NAN).unwrap();
        assert!(opt.update(&vars, &loss).is_err());
        assert_eq!(vars["adapter"].to_vec1::<f32>().unwrap(), vec![1.]);
        assert_eq!(opt.step, 0);
    }

    #[test]
    fn only_declared_degraded_control_changes_targets() {
        let original = "fn add(a:i64,b:i64)->i64 { a+b }";
        for purpose in [
            TrainingPurpose::Candidate,
            TrainingPurpose::SeedControl,
            TrainingPurpose::OverfitControl,
        ] {
            assert_eq!(purpose.completion(original), original);
            assert_eq!(purpose.target_transform(), "identity_provider_winner");
        }
        let bad = TrainingPurpose::DegradedControl;
        assert_ne!(bad.completion(original), original);
        assert!(bad.target_transform().contains("incorrect"));
    }

    #[test]
    #[ignore = "requires explicitly bootstrapped original Qwen tokenizer assets"]
    fn original_tokenizer_masks_prompt_and_preserves_assistant_eos() {
        let tok = Tokenizer::from_file("offline_bundle/base/tokenizer.json").unwrap();
        let completion = "fn add(a: i64, b: i64) -> i64 { a + b }";
        let sample = encode_sample(
            &tok,
            "mask-test",
            "Write a Rust function. Include no explanation.",
            completion,
            512,
        )
        .unwrap();
        let decoded = tok.decode(&sample.targets, false).unwrap();
        assert_eq!(decoded, format!("{completion}<|im_end|>"));
        assert!(sample.positions.iter().all(|p| *p > 0));
        assert_eq!(sample.positions.len(), sample.targets.len());
        for (position, target) in sample.positions.iter().zip(&sample.targets) {
            let next = *position as usize + 1;
            if next < sample.input_ids.len() {
                assert_eq!(sample.input_ids[next], *target);
            }
        }
    }
}
