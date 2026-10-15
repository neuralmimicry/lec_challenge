//! Bounded development-only teacher selection with durable evidence and owned pruning.
//! Qualification is an exploratory selection rule, never a held-out promotion gate.
use crate::{
    artefact::{self, Artefact},
    comparative::{self, CompareOptions, ComparisonSplit, Registry, RegistryModel, WeightFormat},
    hub, model, platform,
    tasks::{self, Split},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
    time::Instant,
};

mod evidence;
mod prune;
pub mod qualification;
mod selection;
mod workflow;

fn engine_sha256() -> Result<String> {
    static HASH: std::sync::OnceLock<std::result::Result<String, String>> =
        std::sync::OnceLock::new();
    HASH.get_or_init(|| {
        artefact::file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Candidate {
    id: String,
    model_id: String,
    revision: String,
    source_model_id: String,
    source_revision: String,
    filename: String,
    sha256: String,
    bytes: u64,
    directory: PathBuf,
    architecture: String,
    licence: String,
    developer_id: String,
    dense_embedding_bytes: u64,
    kv_bytes_per_token: u64,
}

impl Candidate {
    fn registry_model(&self) -> RegistryModel {
        RegistryModel {
            id: self.id.clone(),
            model_id: self.model_id.clone(),
            revision: self.revision.clone(),
            directory: self.directory.clone(),
            architecture: self.architecture.clone(),
            role: "coding_teacher".into(),
            licence: self.licence.clone(),
            developer_id: self.developer_id.clone(),
            weight_format: WeightFormat::Gguf,
        }
    }
    fn residency(&self, context: usize) -> Result<(u64, u64)> {
        let kv = self
            .kv_bytes_per_token
            .checked_mul(u64::try_from(context)?)
            .context("KV residency overflow")?;
        let base = self
            .dense_embedding_bytes
            .checked_add(kv)
            .context("embedding/cache residency overflow")?;
        let gpu = self
            .bytes
            .checked_add(base)
            .and_then(|n| n.checked_add(256 << 20))
            .context("GPU residency overflow")?;
        let cpu = self
            .bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(base))
            .and_then(|n| n.checked_add(512 << 20))
            .context("CPU residency overflow")?;
        Ok((gpu, cpu))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SearchConfig {
    schema_version: u32,
    target_models: usize,
    minimum_pass_rate: f64,
    maximum_iterations: usize,
    maximum_download_bytes: u64,
    minimum_free_disk_bytes: u64,
    device: String,
    gpu_ordinal: usize,
    context_tokens: usize,
    max_new_tokens: usize,
    seed: u64,
    managed_roots: Vec<PathBuf>,
    keep_models: Vec<String>,
    initial_registry: PathBuf,
    initial_comparisons: Vec<PathBuf>,
    candidates: Vec<Candidate>,
}

fn relative(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "search paths must be relative without traversal"
    );
    Ok(())
}
fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}
fn model_id(value: &str) -> bool {
    let pieces = value.split('/').collect::<Vec<_>>();
    pieces.len() == 2
        && pieces.iter().all(|p| {
            !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        })
}
fn pin(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

impl SearchConfig {
    fn load(path: &Path) -> Result<Self> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        let registry = Registry::load(&config.initial_registry)?;
        ensure!(
            config.candidates.iter().all(|c| !registry
                .models
                .iter()
                .any(|m| m.id == c.id || m.directory == c.directory)),
            "search candidates must not replace original model identities/directories"
        );
        ensure!(
            config
                .keep_models
                .iter()
                .all(|id| config.candidates.iter().any(|c| &c.id == id)
                    || registry.models.iter().any(|m| &m.id == id)),
            "keep-model identity is undeclared"
        );
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && (1..=12).contains(&self.target_models)
                && (1..=24).contains(&self.maximum_iterations)
                && self.maximum_iterations <= self.candidates.len()
                && self.minimum_pass_rate.is_finite()
                && (0.0..=1.0).contains(&self.minimum_pass_rate)
                && self.maximum_download_bytes > 0
                && self.maximum_download_bytes <= 256 << 30
                && matches!(self.device.as_str(), "auto" | "cpu" | "cuda")
                && self.context_tokens == 1024
                && self.max_new_tokens == 512
                && self.seed == 1729,
            "invalid bounded development search policy"
        );
        ensure!(
            !self.managed_roots.is_empty(),
            "search needs explicit owned model roots"
        );
        for root in &self.managed_roots {
            relative(root)?;
            ensure!(
                root.starts_with("offline_bundle")
                    && root != Path::new("offline_bundle")
                    && !root.starts_with("offline_bundle/base"),
                "managed model roots must exclude the assessment base"
            );
        }
        relative(&self.initial_registry)?;
        for path in &self.initial_comparisons {
            relative(path)?;
        }
        let mut ids = BTreeSet::new();
        let mut dirs = BTreeSet::new();
        for candidate in &self.candidates {
            relative(&candidate.directory)?;
            ensure!(
                safe_id(&candidate.id)
                    && model_id(&candidate.model_id)
                    && model_id(&candidate.source_model_id)
                    && ids.insert(&candidate.id)
                    && dirs.insert(&candidate.directory)
                    && self
                        .managed_roots
                        .iter()
                        .any(|r| candidate.directory.starts_with(r) && candidate.directory != *r)
                    && !candidate.directory.starts_with("offline_bundle/base")
                    && !candidate.directory.starts_with("checkpoints")
                    && pin(&candidate.revision, 40)
                    && pin(&candidate.source_revision, 40)
                    && pin(&candidate.sha256, 64)
                    && candidate.bytes > 0
                    && candidate.bytes <= 32 << 30
                    && matches!(candidate.architecture.as_str(), "qwen2" | "llama")
                    && candidate.filename.ends_with(".gguf")
                    && !candidate.filename.contains('/')
                    && !candidate.filename.contains("..")
                    && candidate.dense_embedding_bytes > 0
                    && candidate.kv_bytes_per_token > 0,
                "unsafe or duplicate search candidate"
            );
            candidate.residency(self.context_tokens)?;
        }
        ensure!(
            self.keep_models.iter().all(|id| safe_id(id)),
            "unsafe keep-model identifier"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Trial {
    id: String,
    model_id: String,
    source_model_id: String,
    revision: String,
    directory: PathBuf,
    role: String,
    status: String,
    passed: Option<u64>,
    planned: u64,
    errors: u64,
    qualified: bool,
    #[serde(default)]
    qualification_comparison: Option<PathBuf>,
    #[serde(default)]
    qualification_report: Option<Artefact>,
    #[serde(default)]
    qualification_passed: Option<u64>,
    #[serde(default)]
    qualification_errors: Option<u64>,
    comparison: Option<PathBuf>,
    report: Option<Artefact>,
    reason: Option<String>,
    elapsed_seconds: Option<f64>,
    imported: bool,
    pruned: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct State {
    schema_version: u32,
    freeze_sha256: String,
    initialised: bool,
    trials: Vec<Trial>,
    download_bytes_reserved: u64,
    cached_only: bool,
    connectivity_reason: Option<String>,
    events: Vec<serde_json::Value>,
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
        // SAFETY: flock operates on a live owned descriptor and touches no Rust memory.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another teacher search owns this output or model pool"
        );
        Ok(Self(file))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

fn safe_directory(path: &Path) -> Result<()> {
    relative(path)?;
    let mut prefix = PathBuf::new();
    for component in path.components() {
        prefix.push(component);
        match fs::symlink_metadata(&prefix) {
            Ok(m) => ensure!(
                !m.file_type().is_symlink(),
                "symlink in teacher-search path"
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    fs::create_dir_all(path)?;
    Ok(())
}

async fn save(output: &Path, state: &State) -> Result<()> {
    let path = output.join("state.json");
    let state = state.clone();
    tokio::task::spawn_blocking(move || artefact::write_json(&path, &state)).await?
}

fn selection_order(a: &Trial, b: &Trial) -> std::cmp::Ordering {
    a.qualified
        .cmp(&b.qualified)
        .then_with(|| {
            if a.qualified && b.qualified {
                let a_primary = a.passed.unwrap_or_default();
                let b_primary = b.passed.unwrap_or_default();
                let a_fresh = a.qualification_passed.unwrap_or_default();
                let b_fresh = b.qualification_passed.unwrap_or_default();
                a_primary
                    .min(a_fresh)
                    .cmp(&b_primary.min(b_fresh))
                    .then_with(|| (a_primary + a_fresh).cmp(&(b_primary + b_fresh)))
            } else {
                a.passed.cmp(&b.passed)
            }
        })
        .then_with(|| b.id.cmp(&a.id))
}

fn retained(state: &State, config: &SearchConfig) -> BTreeSet<String> {
    let mut selected = config.keep_models.iter().cloned().collect::<BTreeSet<_>>();
    let mut ranked = state
        .trials
        .iter()
        .filter(|t| {
            t.passed.is_some() && t.errors == 0 && t.role != "student_baseline" && !t.pruned
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| selection_order(b, a));
    let mut sources = BTreeSet::new();
    for trial in ranked.iter().copied().filter(|t| t.qualified) {
        if sources.insert(&trial.source_model_id) {
            selected.insert(trial.id.clone());
        }
    }
    for trial in ranked.iter().copied().filter(|t| {
        t.qualification_passed.is_none()
            && t.passed
                .is_some_and(|passed| passed as f64 / 16.0 >= config.minimum_pass_rate)
    }) {
        selected.insert(trial.id.clone());
    }
    if let Some(best) = ranked.first().filter(|t| !t.qualified) {
        selected.insert(best.id.clone());
    }
    for trial in &state.trials {
        if trial.role == "student_baseline"
            || trial.errors > 0
            || trial.qualification_errors.unwrap_or(0) > 0
        {
            selected.insert(trial.id.clone());
        }
    }
    selected
}

fn report(state: &State, config: &SearchConfig, status: &str) -> serde_json::Value {
    let active = retained(state, config);
    let mut unique = BTreeSet::new();
    let mut qualified = state
        .trials
        .iter()
        .filter(|t| t.qualified && !t.pruned && active.contains(&t.id))
        .collect::<Vec<_>>();
    qualified.sort_by(|a, b| selection_order(b, a));
    qualified.retain(|t| unique.insert(t.source_model_id.clone()));
    serde_json::json!({"schema_version":1,"evidence_kind":"bounded_exploratory_development_teacher_selection","status":status,
        "target_models":config.target_models,"minimum_pass_rate":config.minimum_pass_rate,"qualification_rule":"Each distinct original model must meet the threshold separately on both the original 16-task development pack and the independently frozen fresh 16-task development qualification pack, with complete coverage and zero infrastructure/generation errors in each.","qualified_target_met":qualified.len()>=config.target_models,
        "qualified_models":qualified,"retained_model_ids":active,"selection_scope":if state.trials.iter().any(|t| t.qualified && !t.pruned) { "Qualified models are ranked by the lower pack score, then combined passes, then model identifier" } else { "Below-target fallback is ranked only by original development selection score; fresh status is disclosed per trial" },"trials":state.trials,"cached_only":state.cached_only,
        "connectivity_reason":state.connectivity_reason,"download_bytes_reserved":state.download_bytes_reserved,"download_budget_kind":"Logical declared weight bytes plus 64 MiB auxiliary allocation per candidate; not measured wire transfer or cash expenditure","events":state.events,
        "held_out_access":false,"promotable":false,"cost":"Local compute, bandwidth, storage and energy cash costs remain unmeasured; no paid provider calls are made.",
        "limitations":["Repeated development selection is exploratory and can overfit locally authored source groups; the independent fresh pack improves selection discipline but does not establish clean held-out quality.","Every 16-task development denominator remains complete. Seed 1729 and 512 output tokens are fixed before inference.","Quantisation variants share original model identity and cannot inflate the teacher count. Different fine-tunes may still share pretraining and correlated errors.","Automatic hardware selection may use different device/precision; every actual choice and reason are retained in per-trial evidence.","Locally hosted models do not satisfy the original three independent real-provider businesses; no assessment learning gain or SHIP is claimed."]})
}

async fn finish(
    output: &Path,
    state: &State,
    config: &SearchConfig,
    status: &str,
) -> Result<serde_json::Value> {
    save(output, state).await?;
    let value = report(state, config, status);
    let path = output.join("report.json");
    let copy = value.clone();
    tokio::task::spawn_blocking(move || artefact::write_json(&path, &copy)).await??;
    Ok(value)
}

/// Online mode uses only the audited Hub download module; it never implements credential handling.
pub async fn run(config: &Path, output: &Path) -> Result<serde_json::Value> {
    run_with_options(config, output, false).await
}

/// Offline mode performs no metadata requests, downloads, health probes or HTTP fallbacks.
pub async fn run_with_options(
    config: &Path,
    output: &Path,
    offline: bool,
) -> Result<serde_json::Value> {
    workflow::run(config, output, offline).await
}

/// Revalidate linked trial evidence and return the best retained non-baseline
/// model for local serving. Development selection never implies a ship decision.
pub async fn load_best_model(config: &Path, output: &Path) -> Result<Option<RegistryModel>> {
    evidence::load_best_model(config, output).await
}

/// All qualified, retained distinct original sources in selection order; if
/// none qualifies, retain only the strongest measured research fallback.
pub async fn load_selected_models(config: &Path, output: &Path) -> Result<Vec<RegistryModel>> {
    evidence::load_selected_models(config, output).await
}

/// Validate owned deletion authority before a frozen study rehydrates a model.
/// A present modified weight is rejected rather than treated as a cache miss.
pub fn verify_owned_pruned_model(model: &RegistryModel) -> Result<bool> {
    prune::verify_pruned_directory(
        &model.directory,
        &model.id,
        &model.model_id,
        &model.revision,
    )
}

#[cfg(test)]
mod tests;
