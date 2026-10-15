//! Byte offsets, rather than token indices, align representations across models.
//! Full-vocabulary KL additionally requires identical vocabulary IDs and spans.
use super::SequencePolicy;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SequenceWindow {
    pub original_input_tokens: usize,
    pub original_input_start: usize,
    pub original_input_end: usize,
    pub context_tokens_retained: usize,
    pub assistant_tokens_omitted: usize,
    pub complete_assistant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedSequence {
    pub input_ids: Vec<u32>,
    pub prediction_positions: Vec<u32>,
    pub targets: Vec<u32>,
    /// None identifies the separately supervised EOS token.
    pub target_spans: Vec<Option<(usize, usize)>>,
    pub feature_positions: Vec<u32>,
    pub feature_byte_ends: Vec<usize>,
    pub window: Option<SequenceWindow>,
}

pub fn encode(
    tokenizer: &Tokenizer,
    prefix: &str,
    completion: &str,
    eos: u32,
    max_tokens: usize,
) -> Result<EncodedSequence> {
    encode_with_policy(
        tokenizer,
        prefix,
        completion,
        eos,
        max_tokens,
        SequencePolicy::RejectOverlength,
    )
}

pub fn encode_with_policy(
    tokenizer: &Tokenizer,
    prefix: &str,
    completion: &str,
    eos: u32,
    max_tokens: usize,
    policy: SequencePolicy,
) -> Result<EncodedSequence> {
    ensure!(
        !completion.trim().is_empty(),
        "empty distillation completion"
    );
    let text = format!("{prefix}{completion}");
    let encoded = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    ensure!(
        !encoded.is_empty() && max_tokens >= 4,
        "invalid distillation token capacity"
    );
    let boundary = prefix.len();
    let mut out = EncodedSequence {
        input_ids: encoded.get_ids().to_vec(),
        prediction_positions: vec![],
        targets: vec![],
        target_spans: vec![],
        feature_positions: vec![],
        feature_byte_ends: vec![],
        window: None,
    };
    for (i, (&(first, end), &id)) in encoded
        .get_offsets()
        .iter()
        .zip(encoded.get_ids())
        .enumerate()
    {
        if first < boundary || end <= first {
            continue;
        }
        ensure!(end <= text.len(), "token offset lies outside original text");
        out.feature_positions.push(u32::try_from(i)?);
        out.feature_byte_ends.push(end - boundary);
        if i > 0 {
            out.prediction_positions.push(u32::try_from(i - 1)?);
            out.targets.push(id);
            out.target_spans
                .push(Some((first - boundary, end - boundary)));
        }
    }
    ensure!(
        !out.targets.is_empty(),
        "no complete assistant tokens remain after boundary masking"
    );
    out.prediction_positions
        .push(u32::try_from(out.input_ids.len() - 1)?);
    out.targets.push(eos);
    out.target_spans.push(None);
    if out.input_ids.len() + 1 > max_tokens {
        ensure!(
            policy == SequencePolicy::AssistantWindow,
            "distillation sequence exceeds its frozen token cap"
        );
        let original = out.input_ids.len();
        let first = *out
            .feature_positions
            .first()
            .context("assistant window has no token anchor")? as usize;
        ensure!(
            first > 0,
            "assistant window requires preceding original context"
        );
        let count = (original - first).min(max_tokens - 2);
        let start = first.saturating_sub((max_tokens - 1) - count);
        let end = first + count;
        let mut positions = Vec::new();
        let mut targets = Vec::new();
        let mut spans = Vec::new();
        for ((&position, &target), &span) in out
            .prediction_positions
            .iter()
            .zip(&out.targets)
            .zip(&out.target_spans)
        {
            let p = position as usize;
            if p >= start
                && ((span.is_some() && p < end - 1) || (span.is_none() && end == original))
            {
                positions.push(u32::try_from(p - start)?);
                targets.push(target);
                spans.push(span);
            }
        }
        let features = out
            .feature_positions
            .iter()
            .copied()
            .zip(out.feature_byte_ends.iter().copied())
            .filter(|(p, _)| (*p as usize) >= start && (*p as usize) < end)
            .collect::<Vec<_>>();
        out.feature_positions = features.iter().map(|(p, _)| *p - start as u32).collect();
        out.feature_byte_ends = features.iter().map(|(_, end)| *end).collect();
        out.input_ids = out.input_ids[start..end].to_vec();
        out.prediction_positions = positions;
        out.targets = targets;
        out.target_spans = spans;
        out.window = Some(SequenceWindow {
            original_input_tokens: original,
            original_input_start: start,
            original_input_end: end,
            context_tokens_retained: first - start,
            assistant_tokens_omitted: original - end,
            complete_assistant: end == original,
        });
        ensure!(
            !out.targets.is_empty(),
            "assistant window removed every learnable target"
        );
    }
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnchorAlignment {
    pub student_rows: Vec<u32>,
    pub teacher_rows: Vec<u32>,
    pub student_available: usize,
    pub teacher_available: usize,
    pub byte_boundary_matched: usize,
}

pub fn feature_anchors(
    student: &EncodedSequence,
    teacher: &EncodedSequence,
) -> Result<AnchorAlignment> {
    ensure!(
        student.feature_positions.len() == student.feature_byte_ends.len()
            && teacher.feature_positions.len() == teacher.feature_byte_ends.len(),
        "feature offset accounting differs"
    );
    let teacher_by_byte = teacher
        .feature_byte_ends
        .iter()
        .zip(&teacher.feature_positions)
        .map(|(&end, &position)| (end, position))
        .collect::<BTreeMap<_, _>>();
    let mut student_rows = Vec::new();
    let mut teacher_rows = Vec::new();
    for (&end, &position) in student
        .feature_byte_ends
        .iter()
        .zip(&student.feature_positions)
    {
        if let Some(&target) = teacher_by_byte.get(&end) {
            student_rows.push(position);
            teacher_rows.push(target);
        }
    }
    ensure!(
        !student_rows.is_empty(),
        "no assistant byte-boundary alignment exists"
    );
    let matched = student_rows.len();
    ensure!(
        matched * 2
            >= student
                .feature_positions
                .len()
                .min(teacher.feature_positions.len()),
        "less than half of the smaller tokenisation shares assistant byte boundaries"
    );
    Ok(AnchorAlignment {
        student_rows,
        teacher_rows,
        student_available: student.feature_positions.len(),
        teacher_available: teacher.feature_positions.len(),
        byte_boundary_matched: matched,
    })
}

pub fn prediction_rows(student: &EncodedSequence, teacher: &EncodedSequence) -> Result<Vec<u32>> {
    let teacher_by_span = teacher
        .target_spans
        .iter()
        .enumerate()
        .map(|(i, &span)| (span, i))
        .collect::<BTreeMap<_, _>>();
    student
        .target_spans
        .iter()
        .zip(&student.targets)
        .map(|(&span, &id)| {
            let &row = teacher_by_span
                .get(&span)
                .context("KL tokenisation differs at an assistant byte span")?;
            ensure!(
                teacher.targets[row] == id,
                "KL target token ID differs despite aligned bytes"
            );
            Ok(u32::try_from(row)?)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sequence(positions: &[u32], ends: &[usize]) -> EncodedSequence {
        EncodedSequence {
            input_ids: vec![1, 2, 3],
            prediction_positions: vec![0, 1],
            targets: vec![2, 3],
            target_spans: vec![Some((0, 2)), None],
            feature_positions: positions.to_vec(),
            feature_byte_ends: ends.to_vec(),
            window: None,
        }
    }
    #[test]
    fn feature_alignment_uses_original_bytes_across_different_chat_prefixes() {
        let a = sequence(&[10, 11, 12], &[2, 4, 6]);
        let b = sequence(&[20, 21], &[4, 6]);
        let aligned = feature_anchors(&a, &b).unwrap();
        assert_eq!(aligned.student_rows, vec![11, 12]);
        assert_eq!(aligned.teacher_rows, vec![20, 21]);
        assert_eq!(aligned.student_available, 3);
    }
    #[test]
    fn kl_cannot_hide_missing_spans_or_different_token_ids() {
        let a = sequence(&[1], &[2]);
        let mut b = a.clone();
        assert_eq!(prediction_rows(&a, &b).unwrap(), vec![0, 1]);
        b.targets[0] = 7;
        assert!(prediction_rows(&a, &b).is_err());
        b = a.clone();
        b.target_spans[0] = Some((0, 1));
        assert!(prediction_rows(&a, &b).is_err());
    }

    #[test]
    fn declared_assistant_windows_preserve_real_context_offsets_and_never_invent_eos() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("p".into(), 1),
                    ("a".into(), 2),
                    ("b".into(), 3),
                    ("c".into(), 4),
                    ("d".into(), 5),
                    ("e".into(), 6),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        assert!(encode(&tokenizer, "p p p p ", "a b c d e", 99, 4).is_err());
        let result = encode_with_policy(
            &tokenizer,
            "p p p p ",
            "a b c d e",
            99,
            4,
            SequencePolicy::AssistantWindow,
        )
        .unwrap();
        assert_eq!(result.input_ids, vec![1, 2, 3]);
        assert_eq!(result.targets, vec![2, 3]);
        assert_eq!(result.prediction_positions, vec![0, 1]);
        assert_eq!(result.target_spans, vec![Some((0, 1)), Some((2, 3))]);
        let window = result.window.unwrap();
        assert_eq!(window.context_tokens_retained, 1);
        assert_eq!(window.assistant_tokens_omitted, 3);
        assert!(!window.complete_assistant);
        let short = encode_with_policy(
            &tokenizer,
            "p p p p ",
            "a",
            99,
            4,
            SequencePolicy::AssistantWindow,
        )
        .unwrap();
        assert_eq!(short.targets, vec![2, 99]);
        assert!(short.window.unwrap().complete_assistant);
    }
}
