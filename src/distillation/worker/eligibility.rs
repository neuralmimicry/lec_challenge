//! Project the actual worker selection without model tensors or optimisation.
use super::*;
use serde_json::{Value, json};

pub(super) fn sequence_summary(sequence: &EncodedSequence) -> Value {
    json!({
        "retained_input_tokens":sequence.input_ids.len(),
        "original_input_tokens":sequence.window.as_ref().map_or(sequence.input_ids.len(),|window|window.original_input_tokens),
        "original_tokens_including_eos":sequence.window.as_ref().map_or(sequence.input_ids.len(),|window|window.original_input_tokens)+1,
        "supervised_tokens":sequence.targets.len(),
        "assistant_tokens_supervised":sequence.target_spans.iter().filter(|span|span.is_some()).count(),
        "eos_supervised":sequence.target_spans.iter().any(|span|span.is_none()),
        "window":sequence.window,
    })
}

fn rejected_summary(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    sample: &EvidenceSample,
    tokenizer: &Tokenizer,
) -> Result<Value> {
    // Ask the same encoders for the complete sequence only to measure the
    // rejected example. No shortened sequence can enter the actual selection.
    let mut complete = spec.clone();
    complete.max_sequence_tokens = usize::MAX;
    complete.sequence_policy = SequencePolicy::RejectOverlength;
    let pair = spec
        .losses
        .iter()
        .any(|loss| loss.kind == LossKind::PreferenceDpo)
        .then(|| preference::prepare(&complete, bundle, sample, tokenizer))
        .transpose()?;
    let sequence = if spec
        .losses
        .iter()
        .all(|loss| loss.kind == LossKind::PreferenceDpo)
    {
        pair.as_ref()
            .context("preference encoding absent")?
            .chosen
            .clone()
    } else {
        encode_sample(&complete, sample, tokenizer)?
    };
    let mut summary = sequence_summary(&sequence);
    if let Some(pair) = pair {
        summary["preference_chosen"] = sequence_summary(&pair.chosen);
        summary["preference_rejected"] = sequence_summary(&pair.rejected);
    }
    Ok(summary)
}

/// Verify original evidence and run precisely the worker's data selection.
/// Asset checks hash existing files; this function never loads model weights,
/// acquires teacher targets, writes a training freeze or creates an optimiser.
pub fn preflight_eligibility(spec: &DistillationWorkerSpec) -> Result<Value> {
    validate(spec)?;
    let root = std::env::current_dir()?;
    let assets = student::assets(spec)?;
    let bundle = teacher_evidence::load_verified(&spec.evidence_bundle)?;
    validate_origins(&bundle)?;
    let tokenizer = Tokenizer::from_file(spec.student_base.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let selected = choose(spec, &bundle, &tokenizer)?;
    let mut accepted = BTreeMap::new();
    let mut excluded = BTreeMap::new();
    for sample in &bundle.samples {
        let mut row = if let Some(sequence) = selected.eligible_sequences.get(&sample.id) {
            let mut row = sequence.clone();
            row["selected_for_updates"] = json!(selected.samples.contains_key(&sample.id));
            row
        } else if let Some(reason) = selected.rejected_overlength.get(&sample.id) {
            let mut row = rejected_summary(spec, &bundle, sample, &tokenizer)?;
            row["reason"] = json!(reason);
            row
        } else {
            continue;
        };
        row["task_id"] = json!(sample.task.id);
        row["source_id"] = json!(sample.task.source_id);
        row["teacher_id"] = json!(sample.teacher_id);
        row["content"] = json!(sample.content);
        if selected.eligible_sequences.contains_key(&sample.id) {
            accepted.insert(sample.id.clone(), row);
        } else {
            excluded.insert(sample.id.clone(), row);
        }
    }
    let selection =
        json!({"accepted":accepted,"excluded_overlength":excluded,"order":selected.order});
    Ok(json!({
        "schema_version":1,
        "evidence_kind":"verified_native_worker_selection_preflight",
        "arm_id":spec.arm_id,
        "canonical_request_sha256":artefact::hash(&serde_json::to_vec(spec)?),
        "evidence_bundle":Artefact::capture(&root,spec.evidence_bundle.clone())?,
        "base_assets":Artefact::capture(&root,spec.student_base.join("assets.json"))?,
        "base_weight_sha256":assets.files.get("model.safetensors"),
        "tokeniser":Artefact::capture(&root,spec.student_base.join("tokenizer.json"))?,
        "engine_sha256":artefact::file_hash(&std::env::current_exe()?)?,
        "sequence_policy":spec.sequence_policy,"max_sequence_tokens":spec.max_sequence_tokens,
        "accepted_samples":accepted.len(),"excluded_overlength_samples":excluded.len(),
        "selected_unique_samples":selected.samples.len(),"planned_updates":selected.order.len(),
        "selection_sha256":artefact::hash(&serde_json::to_vec(&selection)?),"selection":selection,
        "model_weights_allocated":false,"optimiser_created":false,"teacher_targets_acquired":false,
        "held_out_used":false,"production_eligible":false,
        "limitation":"Eligibility and exact realised order only; this is neither a trained checkpoint nor a measured quality result."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_non_preference_losses_keep_identical_rows_order_and_overlength_exclusions() {
        let artefact = Artefact {
            path: "labelled-policy-fixture-never-loaded".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        let absent = json!({"supported":false,"available":false,"reason":"labelled unit fixture; no teacher execution","artefacts":[]});
        let teacher = serde_json::from_value(json!({
            "id":"labelled_teacher",
            "model":{"id":"labelled_teacher","model_id":"labelled/fixture","revision":"1".repeat(40),"directory":"unused","architecture":"qwen2","role":"labelled_fixture","licence":"unit_fixture_only","developer_id":"unit_fixture"},
            "assets":artefact,
            "capabilities":{"complete_sequences":absent,"logits":absent,"hidden_states":absent,"attention":absent,"process":absent,"tool_trajectories":absent,"context":absent,"preferences":absent,"on_policy":absent,"self_distillation":absent,"tokeniser_sha256":"a".repeat(64),"student_tokeniser_sha256":"a".repeat(64),"tokeniser_aligned":true},
            "rights":{"eligible":true,"scope":"labelled unit fixture only","licence":"unit_fixture_only","licence_artefacts":[],"inherited_restrictions":[],"reasoning":"No model outputs or training evidence are created by this policy test."},
        })).unwrap();
        let task = crate::tasks::corpus()
            .into_iter()
            .find(|task| task.split == crate::tasks::Split::Train)
            .unwrap();
        let samples = [
            ("short-a", "answer result".to_owned()),
            ("short-b", "result answer answer".to_owned()),
            ("long", "answer ".repeat(1000)),
        ]
        .into_iter()
        .map(|(id, completion)| EvidenceSample {
            id: id.into(),
            task: task.clone(),
            teacher_id: "labelled_teacher".into(),
            prompt: "labelled prompt".into(),
            completion,
            content: ContentKind::CompleteSequence,
            observation: artefact.clone(),
            raw_generation: artefact.clone(),
            sandbox_report: artefact.clone(),
            audit: artefact.clone(),
            sequence_sha256: "b".repeat(64),
            provenance_kind: "labelled policy fixture, never training evidence".into(),
            student_rollout: None,
        })
        .collect();
        let bundle = EvidenceBundle {
            schema_version: 1,
            evidence_kind: "labelled policy fixture only".into(),
            created_unix_ms: 0,
            teachers: vec![teacher],
            samples,
            candidates: vec![],
            preferences: vec![],
            input_comparisons: vec![],
            content_records: vec![],
            rollout_records: vec![],
            capabilities: BTreeMap::new(),
            train_corpus_sha256: String::new(),
            limitations: vec![],
        };
        let mut spec: DistillationWorkerSpec = serde_json::from_value(json!({
            "schema_version":1,"arm_id":"labelled_fixture","student_base":"unused","student_kind":"conventional_qwen_lora",
            "evidence_bundle":"unused","output":"unused","seed":1729,"steps":4,"max_sequence_tokens":512,
            "sequence_policy":"reject_overlength","rank":2,"alpha":4.0,"learning_rate":0.0002,"temperature":2.0,"dpo_beta":0.1,
            "losses":[{"kind":"sequence_sft","weight":1.0}],"teacher_ids":["labelled_teacher"],"contents":["complete_sequence"],
            "teacher_runtime":{"device":"cpu","dtype":"f32"},"teacher_target_policy":"cache_only",
            "student_runtime":{"device":"cpu","dtype":"f32"},"initial_checkpoint":null,"data_policy":"offline","progressive":false,"self_distillation":false,
        })).unwrap();
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
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        let sft = choose(&spec, &bundle, &tokenizer).unwrap();
        spec.losses.extend(
            [
                LossKind::SoftTokenKl,
                LossKind::HiddenFeature,
                LossKind::AttentionRelation,
            ]
            .map(|kind| WeightedLoss { kind, weight: 1.0 }),
        );
        let composed = choose(&spec, &bundle, &tokenizer).unwrap();
        assert_eq!(sft.eligible_sequences.len(), 2);
        assert_eq!(sft.eligible_sequences, composed.eligible_sequences);
        assert_eq!(sft.rejected_overlength, composed.rejected_overlength);
        assert_eq!(
            serde_json::to_value(sft.order).unwrap(),
            serde_json::to_value(composed.order).unwrap()
        );
        assert_eq!(
            composed
                .rejected_overlength
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            vec!["long".to_owned()]
        );
        let long = bundle
            .samples
            .iter()
            .find(|sample| sample.id == "long")
            .unwrap();
        let measured = rejected_summary(&spec, &bundle, long, &tokenizer).unwrap();
        assert!(measured["original_input_tokens"].as_u64().unwrap() > 512);
        assert_eq!(measured["window"], Value::Null);
    }
}
