//! Local-study training evidence, kept separate from the provider assessment.
//!
//! Public train contracts, original model outputs and trusted container audits
//! are independently bound. An advertised signal is never a recorded trace.
use crate::{
    artefact::{self, Artefact},
    comparative::{
        self, CompareOptions, ComparisonSplit, Observation, RegistryModel, WeightFormat,
    },
    hub,
    model::{self, RuntimeOptions},
    sandbox::SandboxReport,
    tasks::{self, Split, Task},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

mod collect;
mod eligibility;
mod preference;
mod rollouts;
mod validate;
mod variants;
pub use collect::collect;
pub use eligibility::{CONTENT_ROLE_POLICY, ContentEligibility, ContentRole, classify_content};
pub use preference::{
    PreferencePromptTransform, common_preference_prompt, negative_pair_is_eligible,
    preference_conditioning,
};
pub(crate) use rollouts::RecordedModel;
pub use rollouts::{
    RolloutCollectOptions, RolloutInitialMetadata, RolloutMode, collect_rollouts,
    resolve_rollout_initial_checkpoint, resolve_rollout_initial_metadata,
};
pub use validate::load_verified;
pub use variants::{ContentCollectOptions, collect_content};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ContentKind {
    CompleteSequence,
    Process,
    ToolTrajectory,
    Context,
    Preference,
}

/// Supported means a native implementation can acquire it; available requires
/// actual, immutable evidence. The reason remains mandatory in both states.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SignalAvailability {
    pub supported: bool,
    pub available: bool,
    pub reason: String,
    pub artefacts: Vec<Artefact>,
}
impl SignalAvailability {
    fn absent(supported: bool, reason: &str) -> Self {
        Self {
            supported,
            available: false,
            reason: reason.into(),
            artefacts: vec![],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceCapabilities {
    pub complete_sequences: SignalAvailability,
    pub logits: SignalAvailability,
    pub hidden_states: SignalAvailability,
    pub attention: SignalAvailability,
    pub process: SignalAvailability,
    pub tool_trajectories: SignalAvailability,
    pub context: SignalAvailability,
    pub preferences: SignalAvailability,
    pub on_policy: SignalAvailability,
    pub self_distillation: SignalAvailability,
    pub tokeniser_sha256: String,
    pub student_tokeniser_sha256: String,
    pub tokeniser_aligned: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalRightsDecision {
    pub eligible: bool,
    pub scope: String,
    pub licence: String,
    pub licence_artefacts: Vec<Artefact>,
    pub inherited_restrictions: Vec<String>,
    pub reasoning: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherEvidence {
    pub id: String,
    pub model: RegistryModel,
    pub assets: Artefact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_checkpoint: Option<Artefact>,
    pub capabilities: EvidenceCapabilities,
    pub rights: LocalRightsDecision,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StudentRolloutProvenance {
    pub student_checkpoint_sha256: String,
    pub student_raw_rollout: Artefact,
    pub teacher_feedback: Artefact,
    pub regraded_sandbox_report: Artefact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceSample {
    pub id: String,
    pub task: Task,
    pub teacher_id: String,
    pub prompt: String,
    pub completion: String,
    pub content: ContentKind,
    pub observation: Artefact,
    pub raw_generation: Artefact,
    pub sandbox_report: Artefact,
    pub audit: Artefact,
    pub sequence_sha256: String,
    pub provenance_kind: String,
    pub student_rollout: Option<StudentRolloutProvenance>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceCandidate {
    pub task_id: String,
    pub teacher_id: String,
    pub completion: String,
    pub observation: Artefact,
    pub raw_generation: Artefact,
    pub sandbox_report: Artefact,
    pub audit: Artefact,
    pub passed: bool,
    pub quarantined: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreferencePair {
    pub id: String,
    pub task_id: String,
    pub chosen_sample_id: String,
    pub rejected: EvidenceCandidate,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected_eligibility: Option<ContentEligibility>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_transform: Option<PreferencePromptTransform>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceBundle {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub created_unix_ms: i64,
    pub teachers: Vec<TeacherEvidence>,
    pub samples: Vec<EvidenceSample>,
    pub candidates: Vec<EvidenceCandidate>,
    pub preferences: Vec<PreferencePair>,
    pub input_comparisons: Vec<Artefact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content_records: Vec<Artefact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rollout_records: Vec<Artefact>,
    pub capabilities: BTreeMap<String, EvidenceCapabilities>,
    pub train_corpus_sha256: String,
    pub limitations: Vec<String>,
}
pub struct CollectOptions {
    pub registry: PathBuf,
    pub comparisons: Vec<PathBuf>,
    pub output: PathBuf,
    pub runtime: RuntimeOptions,
    pub max_tokens: usize,
    pub seed: u64,
    pub models: Vec<String>,
}

#[derive(Deserialize)]
struct EvidencePlan {
    models: Vec<FrozenTeacher>,
    tasks: Vec<Task>,
    split: ComparisonSplit,
    #[serde(default)]
    pack: Option<String>,
}
#[derive(Deserialize)]
struct FrozenTeacher {
    model: RegistryModel,
    assets: Artefact,
}

fn relative(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "evidence paths must remain relative without traversal"
    );
    Ok(())
}
fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    relative(path)?;
    Artefact::capture(root, path)
}
fn json_equal(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}

fn train_tasks() -> Vec<Task> {
    tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::Train)
        .collect()
}
fn train_hash() -> Result<String> {
    Ok(artefact::hash(&serde_json::to_vec(&train_tasks())?))
}

fn validate_train_plan(plan: &EvidencePlan) -> Result<()> {
    ensure!(
        plan.split == ComparisonSplit::Train
            && plan.pack.as_deref().is_none_or(|p| p == "original"),
        "only original training sources may become local-study evidence"
    );
    let expected = train_tasks();
    ensure!(
        expected.len() == 48 && plan.tasks.len() == 48,
        "local-study collection requires all 48 training tasks, not 48 sources"
    );
    for (a, b) in plan.tasks.iter().zip(&expected) {
        ensure!(
            json_equal(a, b)?,
            "training task identity, order, source, lineage or oracle changed"
        );
    }
    ensure!(
        expected
            .iter()
            .map(|t| &t.source_id)
            .collect::<BTreeSet<_>>()
            .len()
            == 12,
        "unexpected training source count"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_training_identity_rejects_planted_or_mutated_sources() -> Result<()> {
        let mut plan = EvidencePlan {
            models: vec![],
            tasks: train_tasks(),
            split: ComparisonSplit::Train,
            pack: None,
        };
        validate_train_plan(&plan)?;
        plan.tasks[0] = tasks::corpus()
            .into_iter()
            .find(|t| t.split == Split::HeldOut)
            .context("held-out fixture absent")?;
        assert!(validate_train_plan(&plan).is_err());
        plan.tasks = train_tasks();
        plan.tasks[0].lineage_id.push_str("-mutation");
        assert!(validate_train_plan(&plan).is_err());
        plan.tasks = train_tasks();
        plan.tasks[0].oracle_sha256 = "a".repeat(64);
        assert!(validate_train_plan(&plan).is_err());
        Ok(())
    }
    #[test]
    fn development_qualification_and_path_escape_fail_before_collection() -> Result<()> {
        let mut plan = EvidencePlan {
            models: vec![],
            tasks: train_tasks(),
            split: ComparisonSplit::Development,
            pack: None,
        };
        assert!(validate_train_plan(&plan).is_err());
        plan.split = ComparisonSplit::Train;
        plan.pack = Some("qualification".into());
        assert!(validate_train_plan(&plan).is_err());
        assert!(relative(Path::new("../escaped")).is_err());
        assert!(relative(Path::new("/absolute")).is_err());
        assert!(relative(Path::new("reports/local-study")).is_ok());
        Ok(())
    }
}
