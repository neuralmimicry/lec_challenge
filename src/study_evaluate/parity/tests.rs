//! Synthetic policy fixtures exercise verification; no model was run here.
use super::*;

fn file() -> Artefact {
    Artefact {
        path: "labelled-unit-policy-fixture".into(),
        sha256: "a".repeat(64),
        bytes: 0,
    }
}

fn protocol() -> NumericalParityPlan {
    NumericalParityPlan {
        schema_version: 2,
        evidence_kind: "labelled_unit_policy_fixture_only".into(),
        checkpoint: Checkpoint {
            checkpoint_id: "a".repeat(64),
            tensor: file(),
            manifest: file(),
            base_assets: file(),
            tokenizer: file(),
            architecture: StudentArchitecture::TernaryBitnet,
        },
        gguf: file(),
        export_receipt: file(),
        cpp_engine_manifest: file(),
        cpp_protocol: NUMERICAL_PARITY_CPP_PROTOCOL.into(),
        cpp_kv_dtype: "F32".into(),
        diagnostic_executable: file(),
        rust_implementation_sha256: "fixture".into(),
        implementation_sources: BTreeMap::new(),
        prompts: vec![NumericalParityPrompt {
            prompt: NUMERICAL_PARITY_PROMPTS[0].into(),
            input_ids: vec![128000, 42],
        }],
        native_top_tokens: 16,
        maximum_absolute_log_probability_error: 0.10,
        require_equal_top_prediction: true,
        production_eligible: false,
    }
}

fn observations(plan: &NumericalParityPlan) -> (serde_json::Value, RustLogProbabilityProbe) {
    let logprobs = (1..=20)
        .rev()
        .map(|weight| (f64::from(weight) / 210.).ln())
        .collect::<Vec<_>>();
    let top = logprobs
        .iter()
        .take(16)
        .enumerate()
        .map(|(id, p)| serde_json::json!({"id":id,"logprob":p}))
        .collect::<Vec<_>>();
    let native = serde_json::json!({"model":plan.gguf.path,
        "prompt":crate::bitnet::formatted_prompt(&plan.prompts[0].prompt),
        "tokens_evaluated":2,"tokens_predicted":1,"truncated":false,"tokens":[0],
        "timings":{"cache_n":0,"prompt_n":2},
        "generation_settings":{"temperature":0.,"top_p":1.,"seed":1729,"n_predict":1,"n_probs":16,
            "post_sampling_probs":false,"stream":false,"repeat_penalty":1.,"frequency_penalty":0.,
            "presence_penalty":0.,"grammar":"","logit_bias":[],"lora":[]},
        "completion_probabilities":[{"id":0,"logprob":logprobs[0],"top_logprobs":top}]});
    (
        native,
        RustLogProbabilityProbe {
            schema_version: 1,
            evidence_kind: "actual_full_vocabulary_rust_next_token_log_probabilities".into(),
            plan_sha256: "labelled-fixture-plan".into(),
            prompt_index: 0,
            input_ids: plan.prompts[0].input_ids.clone(),
            log_probabilities: logprobs,
        },
    )
}

fn compare(
    native: &serde_json::Value,
    rust: &RustLogProbabilityProbe,
) -> Result<serde_json::Value> {
    probes::compare(&protocol(), "labelled-fixture-plan", 0, 20, native, rust)
}

#[test]
fn raw_full_distribution_controls_result_and_preserves_a_failed_tolerance() {
    let plan = protocol();
    let (mut native, rust) = observations(&plan);
    assert_eq!(compare(&native, &rust).unwrap()["passed"], true);
    for row in native["completion_probabilities"][0]["top_logprobs"]
        .as_array_mut()
        .unwrap()
    {
        row["logprob"] = (row["logprob"].as_f64().unwrap() - 0.2).into();
    }
    native["completion_probabilities"][0]["logprob"] =
        native["completion_probabilities"][0]["top_logprobs"][0]["logprob"].clone();
    let result = compare(&native, &rust).unwrap();
    assert_eq!(result["passed"], false);
    assert_eq!(result["top_prediction_equal"], true);
    assert!(
        result["maximum_absolute_log_probability_error"]
            .as_f64()
            .unwrap()
            > 0.19
    );
}

#[test]
fn raw_probe_cannot_substitute_top_k_for_complete_rust_mass() {
    let (native, mut rust) = observations(&protocol());
    rust.log_probabilities.truncate(16);
    assert!(compare(&native, &rust).is_err());
    let (_, mut rust) = observations(&protocol());
    rust.log_probabilities[19] = f64::NAN;
    assert!(compare(&native, &rust).is_err());
    let (_, mut rust) = observations(&protocol());
    rust.log_probabilities.iter_mut().for_each(|p| *p -= 0.1);
    assert!(compare(&native, &rust).is_err());
}

#[test]
fn raw_probe_rejects_duplicate_ids_missing_tokens_and_changed_conditioning() {
    let (native, rust) = observations(&protocol());
    let mut corrupt = native.clone();
    corrupt["completion_probabilities"][0]["top_logprobs"][15]["id"] = 0.into();
    assert!(compare(&corrupt, &rust).is_err());
    corrupt = native.clone();
    corrupt["completion_probabilities"][0]["top_logprobs"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(compare(&corrupt, &rust).is_err());
    corrupt = native.clone();
    corrupt["generation_settings"]["post_sampling_probs"] = true.into();
    assert!(compare(&corrupt, &rust).is_err());
    corrupt = native.clone();
    corrupt["prompt"] = "different conditioning".into();
    assert!(compare(&corrupt, &rust).is_err());
    let mut corrupt = rust;
    corrupt.input_ids.push(7);
    assert!(compare(&native, &corrupt).is_err());
}

#[test]
fn different_top_token_fails_even_with_small_log_probability_discrepancies() {
    let (mut native, rust) = observations(&protocol());
    let candidates = native["completion_probabilities"][0]["top_logprobs"]
        .as_array_mut()
        .unwrap();
    candidates[0]["id"] = 1.into();
    candidates[1]["id"] = 0.into();
    native["completion_probabilities"][0]["id"] = 1.into();
    native["tokens"][0] = 1.into();
    let result = compare(&native, &rust).unwrap();
    assert!(
        result["maximum_absolute_log_probability_error"]
            .as_f64()
            .unwrap()
            < 0.10
    );
    assert_eq!(result["passed"], false);
}

#[test]
fn schema_one_and_incomplete_receipts_never_become_numerical_proof() {
    let report = NumericalParityReport {
        schema_version: 1,
        evidence_kind: "actual_original_bitnet_rust_i2s_numerical_parity".into(),
        plan: file(),
        observations: vec![],
        cpp_execution: serde_json::Value::Null,
        rust_execution: serde_json::Value::Null,
        comparison_completed: true,
        passed: true,
        production_eligible: false,
    };
    let tmp = tempfile::tempdir().unwrap();
    assert!(reconstruct(&report, tmp.path()).is_err());
    let mut report = report;
    report.schema_version = 2;
    report.comparison_completed = false;
    assert!(reconstruct(&report, tmp.path()).is_err());
}

#[test]
fn missing_bitnet_receipt_is_unavailable_while_conventional_is_unchanged() {
    let root = std::env::current_dir().unwrap();
    let tmp = tempfile::tempdir_in(&root).unwrap();
    let mut evaluation = super::super::tests::plan();
    evaluation.checkpoint.architecture = StudentArchitecture::TernaryBitnet;
    artefact::write_json(&tmp.path().join("plan.json"), &evaluation).unwrap();
    let result = verify_numerical_equivalence(tmp.path()).unwrap();
    assert!(result.required && !result.qualified);
    assert_eq!(result.status, "unavailable");
    assert!(result.reasons[0].contains("missing"));
    evaluation.checkpoint.architecture = StudentArchitecture::ConventionalQwenLora;
    artefact::write_json(&tmp.path().join("plan.json"), &evaluation).unwrap();
    let result = verify_numerical_equivalence(tmp.path()).unwrap();
    assert!(!result.required && result.qualified);
    assert_eq!(result.status, "not_required");
}
