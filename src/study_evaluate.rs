//! Matched train/development evaluation for experimental local checkpoints.
//! This module cannot open the sealed promotion split and never authorises ship.
use crate::{
    artefact::{self, Artefact},
    experiments::StudentArchitecture,
    model::{Generation, GenerationRequest, ModelRuntime},
    sandbox::{self, SandboxConfig, SandboxReport},
    tasks::{self, Split, Task},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

mod backend;
mod compare;
mod evidence;
pub mod parity;
pub mod regrade;
mod run;
pub use compare::{compare_completed, compare_many};
pub use parity::{NumericalEquivalence, verify_numerical_equivalence};
pub use run::run;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct GradingIdentity {
    kind: String,
    engine_sha256: String,
    controller_version: u32,
    policy_sha256: String,
    image_id: String,
    extractor_sha256: Option<String>,
}

#[derive(Clone)]
struct VerifiedEvaluation {
    generation_plan: Plan,
    execution: serde_json::Value,
    grading: GradingIdentity,
    report: serde_json::Value,
    numerical: NumericalEquivalence,
}

/// Verify original observations or an explicitly derived regrading wrapper.
pub fn verify_completed(output: &Path) -> Result<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(output.join("plan.json"))?)?;
    if value["evidence_kind"] == regrade::PLAN_KIND {
        Ok(regrade::verify_view(output)?.report)
    } else {
        evidence::verify_original(output)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StudySplit {
    Train,
    Development,
}
impl StudySplit {
    fn corpus_split(self) -> Split {
        match self {
            Self::Train => Split::Train,
            Self::Development => Split::Development,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub checkpoint_id: String,
    pub tensor: Artefact,
    pub manifest: Artefact,
    pub base_assets: Artefact,
    pub tokenizer: Artefact,
    pub architecture: StudentArchitecture,
}

#[derive(Clone, Debug)]
pub struct StudyEvalOptions {
    pub checkpoint: Checkpoint,
    pub runtime: serde_json::Value,
    pub engine: Artefact,
    /// Additional native executable/export receipts, frozen before inference.
    pub backend_artefacts: Vec<Artefact>,
    pub output: PathBuf,
    pub split: StudySplit,
    pub maximum_concurrent_graders: usize,
    pub sandbox: SandboxConfig,
}

/// Backends must expose the actual requested runtime and checkpoint identity in
/// execution_metadata; serving wrappers must not silently switch devices.
pub trait StudyGenerator: Send {
    fn generate(&mut self, request: GenerationRequest) -> Result<Generation>;
    fn execution_metadata(&self) -> serde_json::Value;
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
impl StudyGenerator for ModelRuntime {
    fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        ModelRuntime::generate(self, request)
    }
    fn execution_metadata(&self) -> serde_json::Value {
        ModelRuntime::execution_metadata(self)
    }
    fn synchronize(&self) -> Result<()> {
        self.model.device().synchronize().map_err(Into::into)
    }
}
impl StudyGenerator for crate::bitnet::BitNetRuntime {
    fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        crate::bitnet::BitNetRuntime::generate(self, request)
    }
    fn execution_metadata(&self) -> serde_json::Value {
        let mut metadata = crate::bitnet::BitNetRuntime::execution_metadata(self);
        metadata["requested"] = serde_json::to_value(crate::model::RuntimeOptions::default())
            .expect("runtime options serialise");
        metadata
    }
}
impl StudyGenerator for crate::bitnet::BitNetCppRuntime {
    fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        crate::bitnet::BitNetCppRuntime::generate(self, request)
    }
    fn execution_metadata(&self) -> serde_json::Value {
        crate::bitnet::BitNetCppRuntime::execution_metadata(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Plan {
    schema_version: u32,
    evidence_kind: String,
    frozen_unix_ms: i64,
    checkpoint: Checkpoint,
    runtime: serde_json::Value,
    engine: Artefact,
    #[serde(default)]
    backend_artefacts: Vec<Artefact>,
    split: StudySplit,
    tasks: Vec<Task>,
    seed: u64,
    max_new_tokens: usize,
    temperature: f64,
    top_p: f64,
    timeout_seconds: u64,
    maximum_concurrent_graders: usize,
    sandbox: SandboxConfig,
    sandbox_image_id: String,
    sandbox_policy_sha256: String,
    sandbox_preflight: Artefact,
    production_eligible: bool,
}
impl Plan {
    fn request(&self, task: &Task) -> GenerationRequest {
        GenerationRequest {
            prompt: task.prompt(),
            max_new_tokens: self.max_new_tokens,
            seed: self.seed,
            temperature: self.temperature,
            top_p: self.top_p,
            timeout_seconds: self.timeout_seconds,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RawGeneration {
    schema_version: u32,
    plan_sha256: String,
    task_id: String,
    source_id: String,
    request: GenerationRequest,
    generation: Option<Generation>,
    model_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Observation {
    schema_version: u32,
    plan_sha256: String,
    task_id: String,
    source_id: String,
    success: bool,
    raw_generation: Artefact,
    sandbox_report: Option<Artefact>,
    grading_error: Option<String>,
}

fn same<T: Serialize>(a: &T, b: &T) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}

#[cfg(test)]
mod tests;
