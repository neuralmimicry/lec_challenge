//! Both response likelihoods use one declared prompt, independent of how their
//! original generations were conditioned. These bytes are frozen before Adam.
use super::{DistillationWorkerSpec, alignment, student};
use crate::{
    artefact,
    teacher_evidence::{self, EvidenceBundle, EvidenceSample, PreferencePromptTransform},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct FrozenPreference {
    pub pair_id: String,
    pub chosen_sample_id: String,
    pub common_prompt: String,
    pub common_prompt_sha256: String,
    pub generation_prompt_transform: Option<PreferencePromptTransform>,
    pub chosen: alignment::EncodedSequence,
    pub rejected: alignment::EncodedSequence,
}

fn encode_responses(
    spec: &DistillationWorkerSpec,
    tokenizer: &Tokenizer,
    prompt: &str,
    chosen: &str,
    rejected: &str,
) -> Result<(alignment::EncodedSequence, alignment::EncodedSequence)> {
    let prefix = student::prefix(spec, prompt);
    let encode = |completion| {
        alignment::encode_with_policy(
            tokenizer,
            &prefix,
            completion,
            student::eos(spec),
            spec.max_sequence_tokens,
            spec.sequence_policy,
        )
    };
    let chosen = encode(chosen).context("preference chosen student sequence")?;
    let rejected = encode(rejected).context("preference rejected student sequence")?;
    validate_shared_context(&chosen, &rejected)?;
    Ok((chosen, rejected))
}

fn validate_shared_context(
    chosen: &alignment::EncodedSequence,
    rejected: &alignment::EncodedSequence,
) -> Result<()> {
    fn context(sequence: &alignment::EncodedSequence) -> Result<(usize, &[u32])> {
        let first = *sequence
            .feature_positions
            .first()
            .context("preference response has no assistant boundary")? as usize;
        Ok((
            sequence
                .window
                .as_ref()
                .map_or(0, |window| window.original_input_start),
            sequence
                .input_ids
                .get(..first)
                .context("preference assistant boundary outside input")?,
        ))
    }
    // AssistantWindow retains a response-length-dependent prompt tail. A
    // shared textual prompt alone is insufficient if those tails differ.
    ensure!(
        context(chosen)? == context(rejected)?,
        "DPO assistant windows retain different conditioning; use complete in-capacity responses or an explicitly shared pair window"
    );
    Ok(())
}

pub(super) fn prepare(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    sample: &EvidenceSample,
    tokenizer: &Tokenizer,
) -> Result<FrozenPreference> {
    let pair = bundle
        .preferences
        .iter()
        .find(|pair| {
            pair.chosen_sample_id == sample.id && teacher_evidence::negative_pair_is_eligible(pair)
        })
        .context("sample lacks independently verified oracle-negative preference pair")?;
    let prompt = teacher_evidence::common_preference_prompt(bundle, pair)?;
    let (chosen, rejected) = encode_responses(
        spec,
        tokenizer,
        &prompt,
        &sample.completion,
        &pair.rejected.completion,
    )?;
    Ok(FrozenPreference {
        pair_id: pair.id.clone(),
        chosen_sample_id: sample.id.clone(),
        common_prompt_sha256: artefact::hash(prompt.as_bytes()),
        common_prompt: prompt,
        generation_prompt_transform: pair.prompt_transform.clone(),
        chosen,
        rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distillation::{
            DataPolicy, DistillationStudent, LossKind, SequencePolicy, TeacherTargetPolicy,
            WeightedLoss,
        },
        model::RuntimeOptions,
        tasks,
        teacher_evidence::{ContentKind, preference_conditioning},
    };
    #[test]
    fn different_generation_contexts_share_exact_preference_prefix_and_assistant_masks() {
        let mut spec = DistillationWorkerSpec {
            schema_version: 1,
            arm_id: "labelled-test-only".into(),
            student_base: "unused".into(),
            student_kind: DistillationStudent::ConventionalQwenLora,
            evidence_bundle: "unused".into(),
            output: "unused".into(),
            seed: 1,
            steps: 1,
            max_sequence_tokens: 1024,
            sequence_policy: SequencePolicy::RejectOverlength,
            rank: 1,
            alpha: 1.,
            learning_rate: 0.001,
            temperature: 1.,
            dpo_beta: 0.1,
            losses: vec![WeightedLoss {
                kind: LossKind::PreferenceDpo,
                weight: 1.,
            }],
            teacher_ids: vec!["test".into()],
            contents: vec![ContentKind::Preference],
            teacher_runtime: RuntimeOptions::default(),
            teacher_target_policy: TeacherTargetPolicy::CacheOnly,
            student_runtime: RuntimeOptions::default(),
            initial_checkpoint: None,
            data_policy: DataPolicy::Offline,
            progressive: false,
            self_distillation: false,
            health_check: None,
        };
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("chosen".into(), 1),
                    ("rejected".into(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let task = tasks::corpus().remove(0);
        let rich = format!("PRIVILEGED_SENTINEL correction context\n{}", task.prompt());
        let (common, transform) = preference_conditioning(&task, &rich, &task.prompt());
        let (chosen, rejected) =
            encode_responses(&spec, &tokenizer, &common, "chosen", "rejected").unwrap();
        assert_eq!(chosen.prediction_positions, rejected.prediction_positions);
        let boundary = chosen.prediction_positions[0] as usize + 1;
        assert_eq!(
            &chosen.input_ids[..boundary],
            &rejected.input_ids[..boundary]
        );
        assert_eq!(chosen.target_spans, vec![Some((0, 6)), None]);
        assert_eq!(rejected.target_spans, vec![Some((0, 8)), None]);
        assert_ne!(chosen.targets[0], rejected.targets[0]);
        assert_ne!(
            transform.chosen_generation_prompt_sha256,
            transform.rejected_generation_prompt_sha256
        );
        assert!(rich.contains("PRIVILEGED_SENTINEL"));
        assert!(!common.contains("PRIVILEGED_SENTINEL"));
        spec.max_sequence_tokens = 4;
        spec.sequence_policy = SequencePolicy::AssistantWindow;
        // The short answer retains two prompt tokens, while the long answer
        // retains one. Rejecting this prevents unequal DPO conditioning.
        assert!(
            encode_responses(
                &spec,
                &tokenizer,
                &common,
                "chosen",
                "rejected rejected rejected rejected"
            )
            .is_err()
        );
        // Both long answers retain the same one-token prompt tail. Their
        // different original response lengths do not alter conditioning.
        let (chosen, rejected) = encode_responses(
            &spec,
            &tokenizer,
            &common,
            "chosen chosen chosen",
            "rejected rejected rejected rejected",
        )
        .unwrap();
        validate_shared_context(&chosen, &rejected).unwrap();
        assert!(!chosen.window.unwrap().complete_assistant);
        assert!(!rejected.window.unwrap().complete_assistant);
    }
}
