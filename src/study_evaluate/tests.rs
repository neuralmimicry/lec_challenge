//! Failure-only evidence fixtures; no checkpoint, model run or learning is invented.
use super::*;
fn placeholder() -> Artefact {
    Artefact {
        path: "labelled-unit-fixture-only.json".into(),
        sha256: "a".repeat(64),
        bytes: 0,
    }
}
pub(super) fn plan() -> Plan {
    let tasks = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::Train)
        .take(2)
        .collect();
    Plan {
        schema_version: 1,
        evidence_kind: "labelled_unit_fixture".into(),
        frozen_unix_ms: 1,
        checkpoint: Checkpoint {
            checkpoint_id: "fixture-checkpoint".into(),
            tensor: placeholder(),
            manifest: placeholder(),
            base_assets: placeholder(),
            tokenizer: placeholder(),
            architecture: StudentArchitecture::ConventionalQwenLora,
        },
        runtime: serde_json::json!({"device":"cpu","dtype":"F32"}),
        engine: placeholder(),
        backend_artefacts: Vec::new(),
        split: StudySplit::Train,
        tasks,
        seed: 1729,
        max_new_tokens: 512,
        temperature: 0.2,
        top_p: 0.95,
        timeout_seconds: 120,
        maximum_concurrent_graders: 1,
        sandbox: SandboxConfig::default(),
        sandbox_image_id: "fixture".into(),
        sandbox_policy_sha256: "fixture".into(),
        sandbox_preflight: placeholder(),
        production_eligible: false,
    }
}
fn raw(plan: &Plan) -> RawGeneration {
    let task = &plan.tasks[0];
    RawGeneration {
        schema_version: 1,
        plan_sha256: "fixture-plan".into(),
        task_id: task.id.clone(),
        source_id: task.source_id.clone(),
        request: plan.request(task),
        generation: None,
        model_error: Some("Injected unit-test error; no model was run".into()),
    }
}
#[test]
fn raw_resume_rejects_changed_seed_task_checkpoint_and_output_limit() {
    let plan = plan();
    let record = raw(&plan);
    evidence::raw(&record, &plan.tasks[0], &plan, "fixture-plan").unwrap();
    let mut changed = record.clone();
    changed.request.seed += 1;
    assert!(evidence::raw(&changed, &plan.tasks[0], &plan, "fixture-plan").is_err());
    assert!(evidence::raw(&record, &plan.tasks[1], &plan, "fixture-plan").is_err());
    changed = record.clone();
    changed.request.max_new_tokens += 1;
    assert!(evidence::raw(&changed, &plan.tasks[0], &plan, "fixture-plan").is_err());
    changed = record;
    changed.generation = Some(Generation {
        completion: String::new(),
        checkpoint_id: "another-checkpoint".into(),
        finish_reason: "timeout".into(),
        input_tokens: 1,
        output_tokens: 0,
        elapsed_ms: 1,
        complete: false,
    });
    changed.model_error = None;
    assert!(evidence::raw(&changed, &plan.tasks[0], &plan, "fixture-plan").is_err());
}
#[test]
fn missing_and_failed_outputs_remain_in_the_full_denominator() {
    let plan = plan();
    let generated = raw(&plan);
    let task = &plan.tasks[0];
    let row = Observation {
        schema_version: 1,
        plan_sha256: "fixture-plan".into(),
        task_id: task.id.clone(),
        source_id: task.source_id.clone(),
        success: false,
        raw_generation: placeholder(),
        sandbox_report: None,
        grading_error: None,
    };
    let summary = evidence::summary(
        &plan,
        std::slice::from_ref(&row),
        std::slice::from_ref(&generated),
        "fixture-plan",
    )
    .unwrap();
    assert_eq!(summary["planned"], 2);
    assert_eq!(summary["observed"], 1);
    assert_eq!(summary["coverage_complete"], false);
    assert_eq!(summary["passed"], 0);
    assert_eq!(summary["infrastructure_or_generation_errors"], 1);
    assert!(
        evidence::summary(
            &plan,
            &[row.clone(), row],
            &[generated.clone(), generated],
            "fixture-plan"
        )
        .is_err()
    );
}
#[test]
fn cached_raw_error_cannot_be_relabelled_as_a_trusted_pass() {
    let plan = plan();
    let tmp = tempfile::tempdir().unwrap();
    let generated = raw(&plan);
    artefact::write_json(&tmp.path().join("raw.json"), &generated).unwrap();
    let task = &plan.tasks[0];
    let mut row = Observation {
        schema_version: 1,
        plan_sha256: "fixture-plan".into(),
        task_id: task.id.clone(),
        source_id: task.source_id.clone(),
        success: false,
        raw_generation: Artefact::capture(tmp.path(), "raw.json").unwrap(),
        sandbox_report: None,
        grading_error: None,
    };
    evidence::observation(&row, task, &plan, "fixture-plan", tmp.path()).unwrap();
    row.success = true;
    assert!(evidence::observation(&row, task, &plan, "fixture-plan", tmp.path()).is_err());
}

#[test]
fn optional_backend_pins_reject_changed_or_duplicate_files() {
    let mut plan = plan();
    let tmp = tempfile::tempdir().unwrap();
    artefact::atomic_write(&tmp.path().join("fixture.bin"), b"labelled fixture only").unwrap();
    let file = Artefact::capture(tmp.path(), "fixture.bin").unwrap();
    plan.backend_artefacts = vec![file.clone()];
    backend::verify(&plan, tmp.path()).unwrap();
    plan.backend_artefacts.push(file);
    assert!(backend::verify(&plan, tmp.path()).is_err());
    plan.backend_artefacts.pop();
    artefact::atomic_write(&tmp.path().join("fixture.bin"), b"modified fixture only").unwrap();
    assert!(backend::verify(&plan, tmp.path()).is_err());
}

#[test]
fn cpp_metadata_cannot_claim_unfrozen_native_code_or_tensors() {
    let plan = plan();
    let tmp = tempfile::tempdir().unwrap();
    let mut metadata = serde_json::json!({
        "engine":"labelled_unit_fixture_only",
        "checkpoint_id":plan.checkpoint.checkpoint_id,
        "requested":plan.runtime,
    });
    backend::execution(&metadata, &plan, tmp.path()).unwrap();
    metadata["engine"] = "pinned_bitnet_cpp_supervised_by_rust".into();
    assert!(backend::execution(&metadata, &plan, tmp.path()).is_err());
    metadata["engine"] = "labelled_unit_fixture_only".into();
    metadata["requested"]["device"] = "cuda".into();
    assert!(backend::execution(&metadata, &plan, tmp.path()).is_err());
}

#[test]
fn older_rust_only_plans_deserialise_with_no_native_backend_pins() {
    let mut value = serde_json::to_value(plan()).unwrap();
    value.as_object_mut().unwrap().remove("backend_artefacts");
    let old: Plan = serde_json::from_value(value).unwrap();
    assert!(old.backend_artefacts.is_empty());
}

#[test]
fn alleged_native_export_cannot_bypass_its_implementation_binding() {
    let mut plan = plan();
    plan.checkpoint.architecture = StudentArchitecture::TernaryBitnet;
    let tmp = tempfile::tempdir().unwrap();
    artefact::write_json(
        &tmp.path().join("fixture.json"),
        &serde_json::json!({
            "evidence_kind":"native_official_bitnet_qat_gguf_export",
            "schema_version":1,"production_eligible":false,
            "binding":{"codec":"native_hf_axis_to_i2s_v1",
                "implementation_sha256":"invented unit fixture"},
            "tensors":1
        }),
    )
    .unwrap();
    plan.backend_artefacts = vec![Artefact::capture(tmp.path(), "fixture.json").unwrap()];
    assert!(backend::verify(&plan, tmp.path()).is_err());
}

#[test]
fn baseline_comparison_rejects_different_decoding_runtime_or_task_denominator() {
    let original = plan();
    let mut candidate = original.clone();
    compare::matched_protocol(&original, &candidate).unwrap();
    candidate.seed += 1;
    assert!(compare::matched_protocol(&original, &candidate).is_err());
    candidate = original.clone();
    candidate.runtime["device"] = "cuda".into();
    assert!(compare::matched_protocol(&original, &candidate).is_err());
    candidate = original.clone();
    candidate.tasks.pop();
    assert!(compare::matched_protocol(&original, &candidate).is_err());
}
