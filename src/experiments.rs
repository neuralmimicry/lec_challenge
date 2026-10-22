//! Finite, preregistered distillation studies with explicit applicability and budgets.
//!
//! A scheduled arm is not a successful experiment. Only independently validated
//! training artefacts can complete an arm; queued, unsupported and failed work
//! remain visible in the complete configured study catalogue.
use crate::{
    artefact::{self, Artefact},
    model::RuntimeOptions,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

mod catalogue;
mod config;
pub mod evaluation_failure;
mod rollout;
mod scheduler;
pub use rollout::{RolloutProfile, rollout_profile};

pub use config::{FidelityConfig, StudentTraining, StudyConfig};
pub use scheduler::{CompletedResult, RunOptions, completed_results, run};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SignalMethod {
    SequenceSft,
    SoftTokenKl,
    HiddenFeature,
    AttentionRelation,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ContentMethod {
    Process,
    ToolTrajectory,
    Context,
    Preference,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum DataPolicy {
    Offline,
    OnPolicy,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum StudentArchitecture {
    ConventionalQwenLora,
    TernaryBitnet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capability {
    pub supported: bool,
    pub available: bool,
    /// Only native tensor signals may be acquired from verified cached weights
    /// inside the worker before the first optimiser update.
    #[serde(default)]
    pub acquirable_offline: bool,
    pub reason: String,
    pub artefacts: Vec<Artefact>,
}
impl Capability {
    pub fn usable(&self) -> bool {
        self.supported && (self.available || self.acquirable_offline)
    }
}

/// These claims must be translated from the independently verified collector
/// bundle, never inferred from a model name or from a successful download.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherProfile {
    pub id: String,
    pub source_model_id: String,
    pub revision: String,
    pub tokenizer_sha256: String,
    pub token_alignment_verified: bool,
    pub rights_approved: bool,
    pub signals: BTreeMap<SignalMethod, Capability>,
    pub contents: BTreeMap<ContentMethod, Capability>,
    pub on_policy: Capability,
    pub self_distillation: Capability,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_data: Option<RolloutProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudentProfile {
    pub architecture: StudentArchitecture,
    #[serde(default)]
    pub base_assets: Option<Artefact>,
    #[serde(default)]
    pub tokenizer_sha256: Option<String>,
    /// Full-vocabulary equivalence for this specific student, by teacher ID.
    #[serde(default)]
    pub teacher_token_alignment: BTreeMap<String, bool>,
    pub training: Capability,
    pub exported_checkpoint_reload: Capability,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyInputs {
    pub schema_version: u32,
    pub evidence_bundle: Artefact,
    pub student_base: Artefact,
    pub student_tokenizer_sha256: String,
    pub teachers: Vec<TeacherProfile>,
    pub students: Vec<StudentProfile>,
    pub resources: ResourceCapacity,
    pub worker_resources: ResourceRequest,
    #[serde(default)]
    pub student_resources: Vec<StudentResources>,
    pub teacher_runtime: RuntimeOptions,
    pub student_runtime: RuntimeOptions,
    /// Historical schema-1 plans used the optimisation runtime for evaluation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation: Option<EvaluationProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationProfile {
    pub runtime: RuntimeOptions,
    pub cpu_threads: u32,
    pub max_new_tokens: usize,
    pub timeout_seconds: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudentResources {
    pub student: StudentArchitecture,
    pub quick: ResourceRequest,
    pub full: ResourceRequest,
    pub tensor_capture_vram_bytes: u64,
    /// The worker must reject a missing/corrupt prewarmed trace in this mode.
    pub teacher_targets_cache_only: bool,
}

pub fn resources_for(inputs: &StudyInputs, arm: &ArmSpec, fidelity: Fidelity) -> ResourceRequest {
    if let Some(profile) = inputs
        .student_resources
        .iter()
        .find(|p| p.student == arm.student)
    {
        let mut request = match fidelity {
            Fidelity::Quick => profile.quick.clone(),
            Fidelity::Full => profile.full.clone(),
        };
        if arm.signals.iter().any(|s| *s != SignalMethod::SequenceSft)
            && !profile.teacher_targets_cache_only
        {
            request.vram_bytes = request.vram_bytes.max(profile.tensor_capture_vram_bytes);
        }
        request
    } else {
        inputs.worker_resources.clone()
    }
}

/// Memory is accounted in bytes; CPU threads use the process affinity limit.
/// GPU ordinal is explicit: this version never sums disjoint GPUs as one card.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceCapacity {
    pub cpu_threads: u32,
    pub ram_bytes: u64,
    pub gpu_ordinal: Option<usize>,
    pub vram_bytes: u64,
    pub maximum_parallel_jobs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceRequest {
    pub cpu_threads: u32,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArmSpec {
    pub id: String,
    pub student: StudentArchitecture,
    pub teacher_ids: Vec<String>,
    pub signals: Vec<SignalMethod>,
    pub contents: Vec<ContentMethod>,
    pub data_policy: DataPolicy,
    pub progressive: bool,
    pub self_distillation: bool,
    pub seed: u64,
    pub applicable: bool,
    pub inapplicable_reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_initial: Option<crate::teacher_evidence::RolloutInitialMetadata>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    Quick,
    Full,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequencePolicy {
    RejectOverlength,
    AssistantWindow,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudyPlan {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub config: Artefact,
    pub inputs: StudyInputs,
    pub arms: Vec<ArmSpec>,
    pub quick_arms: Vec<String>,
    pub quick_coverage_gaps: Vec<String>,
    pub quick: FidelityConfig,
    pub full: FidelityConfig,
    pub student_training: Vec<StudentTraining>,
    pub held_out_used: bool,
    pub production_eligible: bool,
}

pub fn training_for(plan: &StudyPlan, arm: &ArmSpec, fidelity: Fidelity) -> FidelityConfig {
    let mut training = match fidelity {
        Fidelity::Quick => plan.quick.clone(),
        Fidelity::Full => plan.full.clone(),
    };
    if let Some(override_) = plan
        .student_training
        .iter()
        .find(|o| o.student == arm.student)
    {
        training.learning_rate = match fidelity {
            Fidelity::Quick => override_.quick_learning_rate,
            Fidelity::Full => override_.full_learning_rate,
        };
    }
    training
}

pub fn fidelity_inapplicable_reasons(arm: &ArmSpec, training: &FidelityConfig) -> Vec<String> {
    let mut reasons = Vec::new();
    let contents = arm
        .contents
        .iter()
        .filter(|c| **c != ContentMethod::Preference)
        .count()
        .max(1);
    let minimum = minimum_updates_for(arm);
    if training.steps < minimum {
        reasons.push(format!(
            "{} total updates cannot cover {} teachers × {} non-preference content strata; at least {minimum} are required",
            training.steps,
            arm.teacher_ids.len(),
            contents
        ));
    }
    if arm.data_policy == DataPolicy::OnPolicy || arm.self_distillation {
        match &arm.rollout_initial {
            None => reasons.push(
                "rollout-based arm lacks one verified homogeneous initial student identity".into(),
            ),
            Some(initial) => {
                if let Some(rank) = initial.rank
                    && rank != training.rank
                {
                    reasons.push(format!("rollout initial adapter rank {rank} is incompatible with this fidelity's rank {}; no silent rank expansion is implemented",training.rank));
                }
                if let Some(alpha) = initial.alpha
                    && alpha != training.alpha
                {
                    reasons.push(format!("rollout initial adapter alpha {alpha} is incompatible with this fidelity's alpha {}",training.alpha));
                }
            }
        }
    }
    reasons
}
pub fn minimum_updates_for(arm: &ArmSpec) -> usize {
    arm.teacher_ids.len().saturating_mul(
        arm.contents
            .iter()
            .filter(|c| **c != ContentMethod::Preference)
            .count()
            .max(1),
    )
}

/// Frozen, self-contained input passed to the separately pinned Rust worker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerRequest {
    pub schema_version: u32,
    pub study_plan: Artefact,
    pub arm: ArmSpec,
    pub fidelity: Fidelity,
    pub training: FidelityConfig,
    pub inputs: StudyInputs,
    pub resources: ResourceRequest,
    pub teacher_targets_cache_only: bool,
    pub output: PathBuf,
    pub offline: bool,
}

/// The worker publishes this envelope only after real adapter export, numerical
/// updates, reload and the declared matched evaluation have been verified.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerOutcome {
    pub schema_version: u32,
    pub arm_id: String,
    pub request_sha256: String,
    pub training_artifact: Artefact,
    pub checkpoint_artifacts: Vec<Artefact>,
    pub checkpoint_changed: bool,
    pub checkpoint_reloaded: bool,
    pub completed_steps: usize,
    pub finite_losses: Vec<f64>,
    pub finite_gradient_norms: Vec<f64>,
    pub train_evaluation: Option<Artefact>,
    pub development_evaluation: Option<Artefact>,
    pub measured_costs: serde_json::Value,
    pub limitations: Vec<String>,
}

pub fn create_plan(config_path: &Path, output: &Path, inputs: StudyInputs) -> Result<StudyPlan> {
    let config = StudyConfig::load(config_path)?;
    config::validate_inputs(&inputs)?;
    let root = std::env::current_dir()?;
    inputs.evidence_bundle.verify(&root)?;
    inputs.student_base.verify(&root)?;
    for teacher in &inputs.teachers {
        for cap in teacher
            .signals
            .values()
            .chain(teacher.contents.values())
            .chain([&teacher.on_policy, &teacher.self_distillation])
        {
            for artifact in &cap.artefacts {
                artifact.verify(&root)?;
            }
        }
    }
    for student in &inputs.students {
        if let Some(base) = &student.base_assets {
            base.verify(&root)?;
        }
        for cap in [&student.training, &student.exported_checkpoint_reload] {
            for artifact in &cap.artefacts {
                artifact.verify(&root)?;
            }
        }
    }
    let arms = catalogue::enumerate(&config, &inputs)?;
    let (quick_arms, quick_coverage_gaps) =
        catalogue::quick_cover(&arms, config.quick.maximum_trials, &config.quick);
    let plan = StudyPlan {
        schema_version: 2,
        evidence_kind: "preregistered_finite_local_distillation_study".into(),
        config: Artefact::capture(&root, config_path.to_owned())?,
        inputs,
        arms,
        quick_arms,
        quick_coverage_gaps,
        quick: config.quick,
        full: config.full,
        student_training: config.student_training,
        held_out_used: false,
        production_eligible: false,
    };
    fs::create_dir_all(output)?;
    let path = output.join("plan.json");
    let bytes = serde_json::to_vec_pretty(&plan)?;
    if path.exists() {
        ensure!(
            fs::read(&path)? == bytes,
            "study plan is immutable; changed inputs need a versioned output directory"
        );
    } else {
        artefact::atomic_write(&path, &bytes)?;
    }
    Ok(plan)
}

pub fn load_plan(path: &Path) -> Result<StudyPlan> {
    let plan: StudyPlan = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        matches!(plan.schema_version, 1 | 2) && !plan.held_out_used && !plan.production_eligible,
        "invalid study plan or unapproved held-out/production claim"
    );
    plan.config.verify(&std::env::current_dir()?)?;
    config::validate_inputs(&plan.inputs)?;
    Ok(plan)
}

/// Workers independently verify the exact frozen arm, budgets and source bundle.
pub fn validate_worker_request(path: &Path) -> Result<WorkerRequest> {
    let request: WorkerRequest = serde_json::from_slice(&fs::read(path)?)?;
    let root = std::env::current_dir()?.canonicalize()?;
    let plan_path = request.study_plan.verify(&root)?;
    let plan = load_plan(&plan_path)?;
    let arm = plan
        .arms
        .iter()
        .find(|a| a.id == request.arm.id && a.applicable)
        .context("worker arm is absent or inapplicable in the frozen study")?;
    let fidelity = training_for(&plan, &request.arm, request.fidelity);
    let output = path
        .canonicalize()?
        .parent()
        .context("worker request directory missing")?
        .strip_prefix(&root)?
        .to_owned();
    ensure!(
        request.schema_version == plan.schema_version
            && serde_json::to_value(arm)? == serde_json::to_value(&request.arm)?
            && serde_json::to_value(&fidelity)? == serde_json::to_value(&request.training)?
            && serde_json::to_value(&plan.inputs)? == serde_json::to_value(&request.inputs)?
            && request.resources == resources_for(&plan.inputs, &request.arm, request.fidelity)
            && request.teacher_targets_cache_only
                == plan
                    .inputs
                    .student_resources
                    .iter()
                    .find(|p| p.student == request.arm.student)
                    .is_some_and(|p| p.teacher_targets_cache_only)
            && fidelity_inapplicable_reasons(&request.arm, &request.training).is_empty()
            && request.output == output,
        "worker request differs from its frozen arm, inputs, resources or output"
    );
    request.inputs.evidence_bundle.verify(&root)?;
    request.inputs.student_base.verify(&root)?;
    Ok(request)
}

#[cfg(test)]
mod tests;
