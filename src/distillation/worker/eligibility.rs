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
        "prediction_positions":sequence.prediction_positions,
        "supervised_byte_spans":sequence.target_spans,
        "input_ids_sha256":artefact::hash(&serde_json::to_vec(&sequence.input_ids).expect("token IDs serialise")),
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
    if spec.teacher_target_policy == TeacherTargetPolicy::CacheOnly {
        targets::prepare(
            spec,
            &bundle,
            &selected.samples,
            &selected.encoded,
            &tokenizer,
        )?;
    }
    let mut accepted = BTreeMap::new();
    let mut excluded = BTreeMap::new();
    let mut outside = BTreeMap::new();
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
            outside.insert(sample.id.clone(),json!({"reason":"Outside the declared teacher/content/verified preference/rollout selection","task_id":sample.task.id,"source_id":sample.task.source_id,"teacher_id":sample.teacher_id,"content":sample.content}));
            continue;
        };
        row["task_id"] = json!(sample.task.id);
        row["source_id"] = json!(sample.task.source_id);
        row["teacher_id"] = json!(sample.teacher_id);
        row["content"] = json!(sample.content);
        row["content_origin"] = json!({"kind":sample.provenance_kind,"observation":sample.observation,"raw_generation":sample.raw_generation,"sequence_sha256":sample.sequence_sha256});
        if spec.schema_version >= 2 && targets::required(spec) {
            row["teacher_sequence"] = match targets::encode_teacher(spec, &bundle, sample) {
                Ok(sequence) => sequence_summary(&sequence),
                Err(error) => json!({"rejection":format!("{error:#}")}),
            };
        }
        if selected.eligible_sequences.contains_key(&sample.id) {
            accepted.insert(sample.id.clone(), row);
        } else {
            excluded.insert(sample.id.clone(), row);
        }
    }
    let selection =
        json!({"accepted":accepted,"excluded_overlength":excluded,"order":selected.order});
    Ok(json!({
        "schema_version":2,
        "evidence_kind":"verified_native_worker_selection_preflight",
        "arm_id":spec.arm_id,
        "canonical_request_sha256":artefact::hash(&serde_json::to_vec(spec)?),
        "evidence_bundle":Artefact::capture(&root,spec.evidence_bundle.clone())?,
        "base_assets":Artefact::capture(&root,spec.student_base.join("assets.json"))?,
        "base_weight_sha256":assets.files.get("model.safetensors"),
        "tokeniser":Artefact::capture(&root,spec.student_base.join("tokenizer.json"))?,
        "engine_sha256":artefact::file_hash(&std::env::current_exe()?)?,
        "sequence_policy":spec.sequence_policy,"max_sequence_tokens":spec.max_sequence_tokens,
        "teacher_sequence_cap_including_eos":spec.max_sequence_tokens,
        "student_sequence_cap_including_eos":spec.max_sequence_tokens,
        "teacher_target_policy":spec.teacher_target_policy,"target_requirements":spec.losses,
        "accepted_samples":accepted.len(),"excluded_overlength_samples":excluded.len(),
        "excluded_inapplicable_samples":outside,
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
        let mut bundle = EvidenceBundle {
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
        assert_eq!(
            composed.rejected_overlength["long"],
            "distillation sequence exceeds its frozen token cap",
            "schema-1 checkpoint readers must preserve their historical frozen exclusion strings"
        );
        let long = bundle
            .samples
            .iter()
            .find(|sample| sample.id == "long")
            .unwrap();
        let measured = rejected_summary(&spec, &bundle, long, &tokenizer).unwrap();
        assert!(measured["original_input_tokens"].as_u64().unwrap() > 512);
        assert_eq!(measured["window"], Value::Null);
        // Labelled auxiliary files exercise the real weight-free teacher encoder.
        // No model weights, provider output or sandbox success is fabricated.
        let auxiliary = tempfile::tempdir().unwrap();
        tokenizer
            .save(auxiliary.path().join("tokenizer.json"), false)
            .unwrap();
        let template = json!({"chat_template":"{% for message in messages %}{{ message['content'] }} {% endfor %}assistant "});
        fs::write(
            auxiliary.path().join("tokenizer_config.json"),
            serde_json::to_vec(&template).unwrap(),
        )
        .unwrap();
        fs::write(auxiliary.path().join("config.json"),br#"{"vocab_size":100,"hidden_size":4,"num_attention_heads":1,"max_position_embeddings":1024,"eos_token_id":99}"#).unwrap();
        let manifest = json!({"model_id":"labelled/fixture","revision":"1".repeat(40),"licence":"unit_fixture_only","files":{
            "tokenizer.json":artefact::file_hash(&auxiliary.path().join("tokenizer.json")).unwrap(),
            "tokenizer_config.json":artefact::file_hash(&auxiliary.path().join("tokenizer_config.json")).unwrap()
        }});
        fs::write(
            auxiliary.path().join("assets.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        bundle.teachers[0].model.directory = auxiliary.path().to_owned();
        let mut context = bundle.samples[0].clone();
        context.content = ContentKind::Context;
        context.prompt = format!("{} {}", "privileged ".repeat(700), context.task.prompt());
        spec.schema_version = 2;
        spec.contents = vec![ContentKind::Context];
        spec.losses = vec![WeightedLoss {
            kind: LossKind::HiddenFeature,
            weight: 1.,
        }];
        let short_student = encode_sample(&spec, &context, &tokenizer).unwrap();
        assert!(short_student.input_ids.len() < 512);
        let error = targets::encode_teacher(&spec, &bundle, &context).unwrap_err();
        let capacity = error
            .downcast_ref::<alignment::SequenceCapacityError>()
            .unwrap();
        assert_eq!(capacity.cap, 512);
        assert!(capacity.measured_tokens_including_eos > 700);
        bundle.samples = vec![context];
        let error = choose(&spec, &bundle, &tokenizer).err().unwrap();
        assert!(format!("{error:#}").contains("teacher sequence"));
        // A fitting teacher with absent immutable traces is rejected in cache-only
        // mode before a model is loaded; no trace may be invented by preflight.
        bundle.samples[0].content = ContentKind::CompleteSequence;
        bundle.samples[0].prompt = "labelled prompt".into();
        spec.contents = vec![ContentKind::CompleteSequence];
        spec.student_base = auxiliary.path().to_owned();
        let selected = choose(&spec, &bundle, &tokenizer).unwrap();
        let error = targets::prepare(
            &spec,
            &bundle,
            &selected.samples,
            &selected.encoded,
            &tokenizer,
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains("missing"));
        let raw_directory = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
        let raw_path = raw_directory.path().join("labelled-negative.json");
        let sample = &mut bundle.samples[0];
        sample.content = ContentKind::Process;
        let original_prompt = sample.task.prompt();
        fs::write(&raw_path,serde_json::to_vec(&json!({"fixture_only":true,"evidence_kind":"actual_local_model_generation","task_id":sample.task.id,"request":{"prompt":original_prompt}})).unwrap()).unwrap();
        let relative = raw_path
            .strip_prefix(std::env::current_dir().unwrap())
            .unwrap()
            .to_owned();
        let raw = Artefact::capture(&std::env::current_dir().unwrap(), relative).unwrap();
        let (prompt, transform) = teacher_evidence::preference_conditioning(
            &sample.task,
            &sample.prompt,
            &original_prompt,
        );
        let negative = teacher_evidence::EvidenceCandidate {
            task_id: sample.task.id.clone(),
            teacher_id: sample.teacher_id.clone(),
            completion: "answer answer".into(),
            observation: sample.observation.clone(),
            raw_generation: raw,
            sandbox_report: sample.sandbox_report.clone(),
            audit: sample.audit.clone(),
            passed: false,
            quarantined: true,
        };
        let eligibility = teacher_evidence::ContentEligibility {
            policy: teacher_evidence::CONTENT_ROLE_POLICY.into(),
            role: teacher_evidence::ContentRole::OracleNegative,
            report: negative.sandbox_report.clone(),
            audit: negative.audit.clone(),
            completion_sha256: artefact::hash(negative.completion.as_bytes()),
            generation_complete: true,
            original_quarantined: true,
            reason: "Labelled selector fixture; never independently verified training evidence"
                .into(),
        };
        bundle.preferences = vec![teacher_evidence::PreferencePair {
            id: "labelled-pair".into(),
            task_id: sample.task.id.clone(),
            chosen_sample_id: sample.id.clone(),
            rejected: negative,
            reason: "Labelled unit fixture".into(),
            rejected_eligibility: Some(eligibility),
            preference_prompt: Some(prompt),
            prompt_transform: Some(transform),
        }];
        spec.contents = vec![ContentKind::Preference];
        spec.losses = vec![WeightedLoss {
            kind: LossKind::PreferenceDpo,
            weight: 1.,
        }];
        let preference_only = choose(&spec, &bundle, &tokenizer).unwrap();
        assert_eq!(preference_only.samples.len(), 1);
        assert_eq!(
            preference_only.samples.values().next().unwrap().content,
            ContentKind::Process
        );
        assert!(
            preference_only
                .order
                .iter()
                .all(|step| step.sample_id == bundle.samples[0].id)
        );
        bundle.preferences[0].chosen_sample_id = "unlinked".into();
        assert!(choose(&spec, &bundle, &tokenizer).is_err());
    }
}
