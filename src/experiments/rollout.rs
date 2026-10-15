//! Per-teacher checkpoint and content compatibility for recorded local rollouts.
use super::*;
use crate::teacher_evidence::{self, ContentKind, EvidenceBundle, RolloutInitialMetadata};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutProfile {
    pub initial: Option<RolloutInitialMetadata>,
    pub reason: String,
    pub complete_sequences: bool,
    pub contents: Vec<ContentMethod>,
    pub complete_sequence_preferences: bool,
    pub preference_contents: Vec<ContentMethod>,
}

fn has_preference_choice(profile: &RolloutProfile) -> bool {
    profile.complete_sequence_preferences || !profile.preference_contents.is_empty()
}

/// Caller supplies an independently verified bundle. Recorded positive sample
/// provenance, rather than inherited offline capability flags, establishes the
/// exact checkpoint/content intersection available to a rollout-based arm.
pub fn rollout_profile(
    bundle: &EvidenceBundle,
    teacher_id: &str,
) -> Result<Option<RolloutProfile>> {
    let samples = bundle
        .samples
        .iter()
        .filter(|sample| sample.teacher_id == teacher_id && sample.student_rollout.is_some())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return Ok(None);
    }
    let initial =
        teacher_evidence::resolve_rollout_initial_metadata(bundle, &[teacher_id.to_owned()]);
    let (initial, reason) = match initial {
        Ok(initial) => (
            Some(initial),
            "Verified checkpoint-bound rollout origin and actual content rows".into(),
        ),
        Err(error) => (
            None,
            format!("Rollout origin cannot be used as one homogeneous initial student: {error:#}"),
        ),
    };
    let mut contents = BTreeSet::new();
    let mut preference_contents = BTreeSet::new();
    let mut complete_sequences = false;
    let mut complete_sequence_preferences = false;
    for sample in samples {
        let paired = bundle
            .preferences
            .iter()
            .any(|pair| pair.chosen_sample_id == sample.id);
        let content = match sample.content {
            ContentKind::CompleteSequence => {
                complete_sequences = true;
                complete_sequence_preferences |= paired;
                None
            }
            ContentKind::Process => Some(ContentMethod::Process),
            ContentKind::Context => Some(ContentMethod::Context),
            ContentKind::ToolTrajectory => Some(ContentMethod::ToolTrajectory),
            ContentKind::Preference => None,
        };
        if let Some(content) = content {
            contents.insert(content);
            if paired {
                preference_contents.insert(content);
            }
        }
    }
    Ok(Some(RolloutProfile {
        initial,
        reason,
        complete_sequences,
        contents: contents.into_iter().collect(),
        complete_sequence_preferences,
        preference_contents: preference_contents.into_iter().collect(),
    }))
}

pub(super) fn initial_for(arm: &ArmSpec, inputs: &StudyInputs) -> Option<RolloutInitialMetadata> {
    if arm.data_policy != DataPolicy::OnPolicy && !arm.self_distillation {
        return None;
    }
    let mut initial = None;
    for id in &arm.teacher_ids {
        let next = inputs
            .teachers
            .iter()
            .find(|teacher| &teacher.id == id)?
            .rollout_data
            .as_ref()?
            .initial
            .as_ref()?;
        if let Some(previous) = &initial {
            if serde_json::to_value(previous).ok()? != serde_json::to_value(next).ok()? {
                return None;
            }
        } else {
            initial = Some(next.clone());
        }
    }
    initial
}

pub(super) fn reasons(
    arm: &ArmSpec,
    teacher: &TeacherProfile,
    inputs: &StudyInputs,
) -> Vec<String> {
    if arm.data_policy != DataPolicy::OnPolicy && !arm.self_distillation {
        return vec![];
    }
    let Some(profile) = &teacher.rollout_data else {
        return vec![format!(
            "teacher {} has no verified checkpoint/content rollout profile",
            teacher.id
        )];
    };
    let Some(initial) = &profile.initial else {
        return vec![format!(
            "teacher {} rollout initial student unavailable: {}",
            teacher.id, profile.reason
        )];
    };
    let mut reasons = Vec::new();
    if serde_json::to_value(initial.student_kind).ok() != serde_json::to_value(arm.student).ok() {
        reasons.push(format!(
            "teacher {} rollout student architecture {:?} differs from arm {:?}",
            teacher.id, initial.student_kind, arm.student
        ));
    }
    let base = inputs
        .students
        .iter()
        .find(|student| student.architecture == arm.student)
        .and_then(|student| student.base_assets.as_ref())
        .and_then(|assets| assets.path.parent());
    if base != Some(initial.student_base.as_path()) {
        reasons.push(format!(
            "teacher {} rollout base {} differs from this student's frozen base",
            teacher.id,
            initial.student_base.display()
        ));
    }
    let selected = arm
        .contents
        .iter()
        .copied()
        .filter(|content| *content != ContentMethod::Preference)
        .collect::<Vec<_>>();
    let preference = arm.contents.contains(&ContentMethod::Preference);
    if selected.is_empty() {
        if !preference && !profile.complete_sequences {
            reasons.push(format!(
                "teacher {} has no checkpoint-bound complete-sequence rollout sample",
                teacher.id
            ));
        }
        if preference && !has_preference_choice(profile) {
            reasons.push(format!("teacher {} has no genuine preference pair chosen from any checkpoint-bound rollout content",teacher.id));
        }
    }
    for content in selected {
        if !profile.contents.contains(&content) {
            reasons.push(format!("teacher {} has no checkpoint-bound {content:?} rollout sample; offline content cannot substitute",teacher.id));
        }
        if preference && !profile.preference_contents.contains(&content) {
            reasons.push(format!("teacher {} has no genuine preference pair for checkpoint-bound {content:?} rollout content",teacher.id));
        }
    }
    if arm.rollout_initial.is_none() {
        reasons.push(
            "selected rollout teachers do not share one verified initial student identity".into(),
        );
    }
    reasons
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pure_preference_accepts_verified_correction_pairs_without_relabelling_them_sequences() {
        let mut profile = RolloutProfile {
            initial: None,
            reason: "labelled applicability fixture; no model executed".into(),
            complete_sequences: false,
            contents: vec![ContentMethod::ToolTrajectory],
            complete_sequence_preferences: false,
            preference_contents: vec![],
        };
        assert!(!has_preference_choice(&profile));
        profile
            .preference_contents
            .push(ContentMethod::ToolTrajectory);
        assert!(has_preference_choice(&profile));
        assert!(!profile.complete_sequences);
        assert!(!profile.complete_sequence_preferences);
        profile.preference_contents.clear();
        profile.contents.push(ContentMethod::Context);
        assert!(!has_preference_choice(&profile));
    }
}
