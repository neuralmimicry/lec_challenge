//! One resumable local research workflow. This does not run the sealed gate or
//! turn local model developers into independent API provider businesses.
use crate::{
    artefact::{self, Artefact},
    comparative::{self, Registry, RegistryModel, WeightFormat},
    engine::{self, PinnedEngine, RunLease},
    experiments, hub,
    model::{self, RuntimeOptions},
    study, teacher_evidence, teacher_search,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
    time::{Duration, Instant},
};

mod baselines;
mod cohorts;
mod process;
mod selection;
mod workflow;
pub use workflow::run;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExploreOptions {
    pub search_config: PathBuf,
    pub search_output: PathBuf,
    pub study_config: PathBuf,
    pub output: PathBuf,
    pub base: PathBuf,
    pub bitnet_base: PathBuf,
    pub maximum_parallel_jobs: Option<usize>,
    pub maximum_run_seconds: u64,
    pub content_maximum_tasks: usize,
    pub content_max_tokens: usize,
    pub collection_max_tokens: usize,
    pub top_groups: usize,
    pub offline: bool,
    /// None adopts the selected teacher's original measured runtime.
    pub teacher_runtime: Option<RuntimeOptions>,
    pub reuse_train_comparisons: Vec<PathBuf>,
}
impl Default for ExploreOptions {
    fn default() -> Self {
        Self {
            search_config: "config/teacher-search.toml".into(),
            search_output: "reports/comparative/search/round-001".into(),
            study_config: "config/distillation-study.toml".into(),
            output: "reports/local-study/explore/round-001".into(),
            base: "offline_bundle/base".into(),
            bitnet_base: "offline_bundle/students/bitnet-2b-4t".into(),
            maximum_parallel_jobs: None,
            maximum_run_seconds: 86_400,
            content_maximum_tasks: 48,
            content_max_tokens: 768,
            collection_max_tokens: 512,
            top_groups: 2,
            offline: false,
            teacher_runtime: None,
            reuse_train_comparisons: vec!["reports/local-study/qwen7b-train".into()],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Protocol {
    schema_version: u32,
    evidence_kind: String,
    options: Value,
    search_config: Artefact,
    study_config: Artefact,
    stage_engine: PinnedEngine,
    held_out_used: bool,
    production_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Stage {
    status: String,
    attempts: Vec<Artefact>,
    artefacts: Vec<Artefact>,
    details: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct State {
    schema_version: u32,
    protocol: Artefact,
    stages: BTreeMap<String, Stage>,
    selection: Option<Artefact>,
    events: Vec<Value>,
}

struct ContextState {
    options: ExploreOptions,
    root: PathBuf,
    protocol: Protocol,
    state: State,
    started: Instant,
    deadline: Instant,
}
impl ContextState {
    fn remaining(&self) -> Result<Duration> {
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        ensure!(
            !remaining.is_zero(),
            "exploration wall-time budget exhausted; validated progress is retained"
        );
        Ok(remaining)
    }
    async fn save(&self) -> Result<()> {
        let path = self.options.output.join("state.json");
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || artefact::write_json(&path, &state)).await?
    }
    async fn publish(
        &mut self,
        name: &str,
        status: &str,
        files: Vec<Artefact>,
        details: Value,
    ) -> Result<()> {
        let stage = self.state.stages.entry(name.into()).or_insert(Stage {
            status: "pending".into(),
            attempts: vec![],
            artefacts: vec![],
            details: Value::Null,
        });
        stage.status = status.into();
        stage.artefacts = files;
        stage.details = details;
        self.save().await
    }
    fn cached(&self, name: &str) -> Result<bool> {
        let Some(stage) = self.state.stages.get(name) else {
            return Ok(false);
        };
        if stage.status != "verified_complete" {
            return Ok(false);
        }
        for file in &stage.artefacts {
            file.verify(&self.root)?;
        }
        Ok(true)
    }
}

fn relative(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|p| matches!(p, Component::Normal(_))),
        "exploration paths must be workspace-relative without traversal"
    );
    Ok(())
}
fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    let path = path.canonicalize()?.strip_prefix(root)?.to_owned();
    Artefact::capture(root, path)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn same(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}
fn runtime_arguments(runtime: &RuntimeOptions) -> Vec<String> {
    let (device, gpu) = match runtime.device {
        model::ExecutionDevice::Cpu => ("cpu", 0),
        model::ExecutionDevice::Cuda { ordinal } => ("cuda", ordinal),
    };
    vec![
        "--device".into(),
        device.into(),
        "--gpu".into(),
        gpu.to_string(),
        "--dtype".into(),
        if runtime.dtype == model::InferenceDtype::F32 {
            "f32"
        } else {
            "f16"
        }
        .into(),
    ]
}

#[cfg(test)]
mod tests;
