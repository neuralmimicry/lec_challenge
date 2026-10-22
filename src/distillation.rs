//! Experimental local-teacher adaptation is separate from assessment training.
//! Each arm retains real evidence, objective identity and changed/reloaded tensors.
use crate::{artefact::Artefact, model::RuntimeOptions, teacher_evidence::ContentKind};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod alignment;
pub mod checkpoint;
pub mod conditioning;
pub mod diagnose;
pub mod losses;
mod preference;
mod student;
mod targets;
mod worker;
pub use student::DistillationStudent;
pub use worker::{load_runtime, preflight_eligibility, prepare_teacher_targets, run_worker};

/// The executable worker bounds also constrain configuration and planning.
pub fn validate_training_bounds(
    steps: usize,
    tokens: usize,
    rank: usize,
    alpha: f64,
    rate: f64,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        (1..=64).contains(&steps)
            && (4..=1024).contains(&tokens)
            && (1..=32).contains(&rank)
            && alpha.is_finite()
            && alpha > 0.
            && rate.is_finite()
            && (0.0000001..=0.01).contains(&rate),
        "unsupported native training bounds: steps={steps} (1–64), sequence including EOS={tokens} (4–1024), rank={rank} (1–32), alpha={alpha}, learning_rate={rate} (1e-7–0.01)"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    SequenceSft,
    SoftTokenKl,
    HiddenFeature,
    AttentionRelation,
    PreferenceDpo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedLoss {
    pub kind: LossKind,
    pub weight: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataPolicy {
    #[default]
    Offline,
    OnPolicy,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequencePolicy {
    #[default]
    RejectOverlength,
    AssistantWindow,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TeacherTargetPolicy {
    #[default]
    Acquire,
    CacheOnly,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillationWorkerSpec {
    pub schema_version: u32,
    pub arm_id: String,
    pub student_base: PathBuf,
    pub student_kind: DistillationStudent,
    pub evidence_bundle: PathBuf,
    pub output: PathBuf,
    pub seed: u64,
    pub steps: usize,
    pub max_sequence_tokens: usize,
    pub sequence_policy: SequencePolicy,
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub temperature: f64,
    pub dpo_beta: f64,
    pub losses: Vec<WeightedLoss>,
    pub teacher_ids: Vec<String>,
    pub contents: Vec<ContentKind>,
    pub teacher_runtime: RuntimeOptions,
    pub teacher_target_policy: TeacherTargetPolicy,
    pub student_runtime: RuntimeOptions,
    pub initial_checkpoint: Option<PathBuf>,
    pub data_policy: DataPolicy,
    pub progressive: bool,
    pub self_distillation: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health_check: Option<TrainingHealthPolicy>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingHealthPolicy {
    pub checkpoints: Vec<usize>,
    pub max_new_tokens: usize,
    pub timeout_seconds: u64,
    pub stop_on_unhealthy: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistillationArtifact {
    pub schema_version: u32,
    pub arm_id: String,
    pub checkpoint: PathBuf,
    pub manifest: Artefact,
    pub adapter_sha256: String,
    pub base_sha256: String,
    pub evidence_bundle_sha256: String,
    pub steps: usize,
    pub losses: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub changed_adapter: bool,
    pub reloaded: bool,
    pub production_eligible: bool,
    pub limitations: Vec<String>,
}
