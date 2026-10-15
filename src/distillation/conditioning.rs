//! Context distillation gives privileged examples to the recorded teacher and
//! the original task alone to the student. Teacher evidence is never rewritten.
use crate::{
    artefact,
    teacher_evidence::{ContentKind, EvidenceSample},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

pub const CONTEXT_POLICY: &str = "context-distillation-teacher-privileged-v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptTransform {
    pub schema_version: u32,
    pub policy: String,
    pub sample_id: String,
    pub teacher_id: String,
    pub task_id: String,
    pub source_id: String,
    pub teacher_prompt_sha256: String,
    pub student_prompt_sha256: String,
    pub teacher_prompt_bytes: usize,
    pub student_prompt_bytes: usize,
    pub task_start_in_teacher_bytes: usize,
    pub task_end_in_teacher_bytes: usize,
    pub omitted_context_bytes: usize,
    /// Hash of the exact omitted prefix and suffix, with explicit lengths, so
    /// a boundary change cannot produce an ambiguous concatenation identity.
    pub omitted_context_sha256: String,
}

pub fn student_prompt(sample: &EvidenceSample) -> String {
    if sample.content == ContentKind::Context {
        sample.task.prompt()
    } else {
        sample.prompt.clone()
    }
}

/// Derive only from an independently verified evidence sample. Both prompt
/// identities are frozen before optimisation; a plain task cannot masquerade
/// as a privileged-context example merely by carrying the Context label.
pub fn transform(sample: &EvidenceSample) -> Result<Option<PromptTransform>> {
    if sample.content != ContentKind::Context {
        return Ok(None);
    }
    let student = student_prompt(sample);
    let start = sample
        .prompt
        .rfind(&student)
        .context("context evidence does not retain the exact original student task")?;
    let end = start + student.len();
    ensure!(
        sample.prompt.len() > student.len(),
        "context evidence contains no privileged teacher context"
    );
    let prefix = &sample.prompt[..start];
    let suffix = &sample.prompt[end..];
    let omitted = serde_json::to_vec(&(prefix.len(), prefix, suffix.len(), suffix))?;
    Ok(Some(PromptTransform {
        schema_version: 1,
        policy: CONTEXT_POLICY.into(),
        sample_id: sample.id.clone(),
        teacher_id: sample.teacher_id.clone(),
        task_id: sample.task.id.clone(),
        source_id: sample.task.source_id.clone(),
        teacher_prompt_sha256: artefact::hash(sample.prompt.as_bytes()),
        student_prompt_sha256: artefact::hash(student.as_bytes()),
        teacher_prompt_bytes: sample.prompt.len(),
        student_prompt_bytes: student.len(),
        task_start_in_teacher_bytes: start,
        task_end_in_teacher_bytes: end,
        omitted_context_bytes: prefix.len() + suffix.len(),
        omitted_context_sha256: artefact::hash(&omitted),
    }))
}

pub fn verify_transform(sample: &EvidenceSample, recorded: &PromptTransform) -> Result<()> {
    ensure!(
        transform(sample)?.as_ref() == Some(recorded),
        "recorded context prompt transform differs from verified teacher/student bytes"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artefact::Artefact,
        distillation::alignment,
        tasks::{self, Split},
    };
    fn fixture() -> EvidenceSample {
        let task = tasks::corpus()
            .into_iter()
            .find(|task| task.split == Split::Train)
            .unwrap();
        let artefact = Artefact {
            path: "labelled-conditioning-fixture-only.json".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        EvidenceSample {
            id: "labelled-sample".into(),
            teacher_id: "labelled-teacher".into(),
            prompt: format!(
                "PRIVILEGED_SENTINEL: verified examples belong only to the teacher.\nNew task:\n{}",
                task.prompt()
            ),
            task,
            completion: "answer result".into(),
            content: ContentKind::Context,
            observation: artefact.clone(),
            raw_generation: artefact.clone(),
            sandbox_report: artefact.clone(),
            audit: artefact,
            sequence_sha256: "b".repeat(64),
            provenance_kind: "labelled policy fixture; no generated evidence".into(),
            student_rollout: None,
        }
    }
    #[test]
    fn privileged_context_is_removed_only_from_student_and_bound_to_exact_teacher_bytes() {
        let sample = fixture();
        let teacher = sample.prompt.clone();
        let student = student_prompt(&sample);
        assert!(!student.contains("PRIVILEGED_SENTINEL"));
        assert_eq!(student, sample.task.prompt());
        assert_eq!(sample.prompt, teacher);
        let proof = transform(&sample).unwrap().unwrap();
        verify_transform(&sample, &proof).unwrap();
        assert!(proof.omitted_context_bytes > 0);
        assert_eq!(
            proof.teacher_prompt_bytes - proof.student_prompt_bytes,
            proof.omitted_context_bytes
        );
        let mut altered = proof.clone();
        altered.student_prompt_sha256 = "c".repeat(64);
        assert!(verify_transform(&sample, &altered).is_err());
        let mut plain = sample.clone();
        plain.prompt = plain.task.prompt();
        assert!(transform(&plain).is_err());
        let mut feedback = sample;
        feedback.content = ContentKind::ToolTrajectory;
        assert_eq!(student_prompt(&feedback), teacher);
        assert!(transform(&feedback).unwrap().is_none());
    }
    #[test]
    fn assistant_byte_alignment_survives_actual_different_teacher_student_prefix_lengths() {
        let sample = fixture();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("answer".into(), 1),
                    ("result".into(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let teacher = alignment::encode(
            &tokenizer,
            &format!("{}\n", sample.prompt),
            &sample.completion,
            99,
            4096,
        )
        .unwrap();
        let student = alignment::encode(
            &tokenizer,
            &format!("{}\n", student_prompt(&sample)),
            &sample.completion,
            99,
            4096,
        )
        .unwrap();
        assert_ne!(student.input_ids.len(), teacher.input_ids.len());
        assert_eq!(student.target_spans, teacher.target_spans);
        assert_eq!(student.targets, teacher.targets);
        assert_eq!(
            alignment::prediction_rows(&student, &teacher).unwrap(),
            vec![0, 1, 2]
        );
        let anchors = alignment::feature_anchors(&student, &teacher).unwrap();
        assert_eq!(anchors.byte_boundary_matched, 2);
        assert_ne!(anchors.student_rows, anchors.teacher_rows);
    }
}
