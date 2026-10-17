//! Selection fixtures test policy only; no model quality or provider evidence is fabricated.
use super::*;

fn trial(id: &str, source: &str, passed: u64, qualified: bool) -> Trial {
    Trial {
        id: id.into(),
        model_id: format!("fixture/{id}"),
        source_model_id: source.into(),
        revision: "1".repeat(40),
        directory: PathBuf::from("offline_bundle/teacher-search").join(id),
        role: "coding_teacher".into(),
        status: "labelled_fixture".into(),
        passed: Some(passed),
        planned: 16,
        errors: 0,
        qualified,
        qualification_comparison: None,
        qualification_report: None,
        qualification_passed: qualified.then_some(passed),
        qualification_errors: qualified.then_some(0),
        comparison: None,
        report: None,
        reason: None,
        elapsed_seconds: None,
        imported: false,
        pruned: false,
    }
}
fn fixture_config() -> SearchConfig {
    SearchConfig {
        schema_version: 1,
        target_models: 3,
        minimum_pass_rate: 0.8,
        maximum_iterations: 1,
        maximum_download_bytes: 32 << 30,
        minimum_free_disk_bytes: 4 << 30,
        device: "auto".into(),
        gpu_ordinal: 0,
        context_tokens: 1024,
        max_new_tokens: 512,
        seed: 1729,
        managed_roots: vec!["offline_bundle/teacher-search".into()],
        keep_models: vec![],
        initial_registry: "config/comparative-models.toml".into(),
        initial_comparisons: vec![],
        candidates: vec![Candidate {
            id: "fixture".into(),
            model_id: "fixture/quantised".into(),
            revision: "1".repeat(40),
            source_model_id: "fixture/original".into(),
            source_revision: "2".repeat(40),
            filename: "fixture.Q4_K_M.gguf".into(),
            sha256: "3".repeat(64),
            bytes: 4 << 30,
            directory: "offline_bundle/teacher-search/fixture".into(),
            architecture: "qwen2".into(),
            licence: "fixture-only".into(),
            developer_id: "fixture-only".into(),
            dense_embedding_bytes: 1 << 30,
            kv_bytes_per_token: 1024,
        }],
    }
}
fn state(trials: Vec<Trial>) -> State {
    State {
        schema_version: 1,
        freeze_sha256: "fixture".into(),
        initialised: true,
        trials,
        download_bytes_reserved: 0,
        cached_only: true,
        connectivity_reason: Some("labelled offline fixture".into()),
        events: vec![],
    }
}

#[test]
fn duplicate_quantisation_variants_do_not_inflate_qualified_target() {
    let config = fixture_config();
    let state = state(vec![
        trial("q4", "same-source", 13, true),
        trial("q6", "same-source", 14, true),
        trial("other", "another-source", 13, true),
    ]);
    let report = report(&state, &config, "fixture");
    assert_eq!(report["qualified_models"].as_array().unwrap().len(), 2);
    assert_eq!(report["qualified_target_met"], false);
    let retained = retained(&state, &config);
    assert!(retained.contains("q6") && !retained.contains("q4"));
}

#[tokio::test]
async fn offline_and_failed_connectivity_preserve_inactive_model_bytes() {
    let directory = tempfile::tempdir_in("offline_bundle").unwrap();
    let weight = directory.path().join("fixture.safetensors");
    fs::write(&weight, b"LABELLED weight fixture; never a real model").unwrap();
    let original = fs::read(&weight).unwrap();
    let mut inactive = trial("inactive", "source", 0, false);
    inactive.directory = directory
        .path()
        .canonicalize()
        .unwrap()
        .strip_prefix(std::env::current_dir().unwrap())
        .unwrap()
        .to_path_buf();
    let mut config = fixture_config();
    config.managed_roots = vec![PathBuf::from("offline_bundle")];
    let mut state = state(vec![trial("stronger", "other-source", 16, true), inactive]);
    let output = tempfile::tempdir().unwrap();
    workflow::prune_scored_with_probe(output.path(), &mut state, &config, || async {
        panic!("explicit offline must perform zero connectivity probes")
    })
    .await
    .unwrap();
    assert_eq!(fs::read(&weight).unwrap(), original);
    assert!(!state.trials[1].pruned);
    state.cached_only = false;
    let probes = std::sync::atomic::AtomicUsize::new(0);
    workflow::prune_scored_with_probe(output.path(), &mut state, &config, || async {
        probes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        false
    })
    .await
    .unwrap();
    assert_eq!(probes.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert!(state.cached_only);
    assert!(!state.trials[1].pruned);
    assert_eq!(fs::read(&weight).unwrap(), original);
    assert_eq!(
        state.events.last().unwrap()["kind"],
        "pruning_skipped_no_connectivity"
    );
}

#[test]
fn best_unqualified_incumbent_and_assessment_baseline_remain_protected() {
    let config = fixture_config();
    let mut baseline = trial("baseline", "baseline", 0, false);
    baseline.role = "student_baseline".into();
    let state = state(vec![
        baseline,
        trial("best", "best-source", 2, false),
        trial("poor", "poor-source", 0, false),
    ]);
    let keep = retained(&state, &config);
    assert!(keep.contains("best") && keep.contains("baseline") && !keep.contains("poor"));
    assert_eq!(
        report(&state, &config, "fixture")["qualified_target_met"],
        false
    );
}

#[test]
fn invalid_paths_unbounded_limits_and_residency_overflow_fail_closed() {
    let mut config = fixture_config();
    config.validate().unwrap();
    config.candidates[0].directory = "offline_bundle/base/fixture".into();
    assert!(config.validate().is_err());
    config.candidates[0].directory = "../outside".into();
    assert!(config.validate().is_err());
    config = fixture_config();
    config.maximum_iterations = 25;
    assert!(config.validate().is_err());
    config = fixture_config();
    config.minimum_pass_rate = f64::NAN;
    assert!(config.validate().is_err());
    config = fixture_config();
    config.candidates[0].kv_bytes_per_token = u64::MAX;
    assert!(config.candidates[0].residency(1024).is_err());
}

#[test]
fn complete_development_threshold_does_not_mean_held_out_promotion() {
    let config = fixture_config();
    let state = state(vec![
        trial("one", "one", 13, true),
        trial("two", "two", 13, true),
        trial("three", "three", 13, true),
    ]);
    let report = report(&state, &config, "fixture");
    assert_eq!(report["qualified_target_met"], true);
    assert_eq!(report["promotable"], false);
    assert_eq!(report["held_out_access"], false);
    assert!(report["cost"].as_str().unwrap().contains("unmeasured"));
}

#[test]
fn original_selection_alone_cannot_qualify_and_fresh_regression_loses_to_qualified() {
    let config = fixture_config();
    let original_only = trial("original-only", "source-a", 16, false);
    let report = report(
        &state(vec![original_only]),
        &config,
        "labelled_policy_fixture",
    );
    assert_eq!(report["qualified_models"].as_array().unwrap().len(), 0);
    assert_eq!(report["qualified_target_met"], false);
    let mut regression = trial("regression", "source-a", 16, false);
    regression.qualification_passed = Some(0);
    regression.qualification_errors = Some(0);
    let mut qualified = trial("qualified", "source-b", 15, true);
    qualified.qualification_passed = Some(14);
    let keep = retained(&state(vec![regression.clone(), qualified.clone()]), &config);
    assert!(keep.contains("qualified") && !keep.contains("regression"));
    assert!(selection_order(&qualified, &regression).is_gt());
    let mut imbalanced = trial("imbalanced", "source-c", 16, true);
    imbalanced.qualification_passed = Some(13);
    assert!(selection_order(&qualified, &imbalanced).is_gt());
}

#[test]
fn saved_identity_and_qualification_cannot_be_relabelled() {
    let config = fixture_config();
    let registry = Registry {
        schema_version: 1,
        models: vec![],
    };
    let output = Path::new("reports/fixture-search");
    let candidate = &config.candidates[0];
    let mut saved = trial(&candidate.id, &candidate.source_model_id, 0, false);
    saved.model_id = candidate.model_id.clone();
    saved.directory = candidate.directory.clone();
    saved.passed = None;
    evidence::expected_trial_model(&saved, &config, &registry, output).unwrap();
    for changed in [
        "source",
        "model",
        "directory",
        "role",
        "qualification",
        "score",
        "pruned",
    ] {
        let mut forged = saved.clone();
        match changed {
            "source" => forged.source_model_id = "invented/independent-source".into(),
            "model" => forged.model_id = "invented/model".into(),
            "directory" => forged.directory = "offline_bundle/base".into(),
            "role" => forged.role = "student_baseline".into(),
            "qualification" => forged.qualified = true,
            "score" => forged.passed = Some(16),
            "pruned" => forged.pruned = true,
            _ => unreachable!(),
        }
        assert!(
            evidence::expected_trial_model(&forged, &config, &registry, output).is_err(),
            "accepted forged {changed}"
        );
    }
}

#[test]
fn resumed_runtime_uses_frozen_device_without_model_or_hardware_access() {
    let mut config = fixture_config();
    let candidate = config.candidates[0].clone();
    let tmp = tempfile::tempdir().unwrap();
    let registry_path = tmp.path().join("registry.toml");
    fs::write(&registry_path, "labelled registry fixture").unwrap();
    let runtime = model::RuntimeOptions {
        device: model::ExecutionDevice::Cuda { ordinal: 7 },
        dtype: model::InferenceDtype::F32,
    };
    let corpus = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::Development)
        .collect::<Vec<_>>();
    let plan = serde_json::json!({"models":[{"model":candidate.registry_model()}],"tasks":corpus,"split":"development","registry":{"path":registry_path,"sha256":artefact::file_hash(&registry_path).unwrap()},"runtime":runtime,"max_tokens":512,"seed":1729,"engine_binary_sha256":engine_sha256().unwrap()});
    let plan_path = tmp.path().join("plan.json");
    artefact::write_json(&plan_path, &plan).unwrap();
    let resumed =
        evidence::frozen_runtime(&plan_path, &candidate, &config, &registry_path).unwrap();
    assert_eq!(resumed.options, runtime);
    assert!(
        !candidate.directory.exists(),
        "fixture unexpectedly overlaps real assets"
    );
    config.device = "cpu".into();
    assert!(evidence::frozen_runtime(&plan_path, &candidate, &config, &registry_path).is_err());
}

#[test]
fn historical_oracle_hash_is_preserved_without_weakening_contract_identity() {
    let config = fixture_config();
    let model = config.candidates[0].registry_model();
    let tmp = tempfile::tempdir().unwrap();
    let registry_path = tmp.path().join("registry.toml");
    fs::write(&registry_path, "labelled historical registry fixture").unwrap();
    let corpus = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::Development)
        .collect::<Vec<_>>();
    let mut plan = serde_json::json!({"models":[{"model":model}],"tasks":corpus,"split":"development","registry":{"path":registry_path,"sha256":artefact::file_hash(&registry_path).unwrap()},"max_tokens":512,"seed":1729});
    plan["tasks"][0]["oracle_sha256"] = serde_json::json!("labelled-original-source-hash");
    evidence::verify_plan_identity(&plan, &model, &config, true, false, &registry_path).unwrap();
    assert!(
        evidence::verify_plan_identity(&plan, &model, &config, false, false, &registry_path)
            .is_err()
    );
    plan["tasks"][0]["specification"] = serde_json::json!("changed contract must be refused");
    assert!(
        evidence::verify_plan_identity(&plan, &model, &config, true, false, &registry_path)
            .is_err()
    );
}

#[tokio::test]
async fn explicit_offline_missing_assets_reports_shortfall_without_inference() {
    let mut config = fixture_config();
    let assets = tempfile::tempdir_in("offline_bundle").unwrap();
    let cwd = std::env::current_dir().unwrap();
    let asset_path = assets.path().strip_prefix(&cwd).unwrap_or(assets.path());
    config.managed_roots = vec![asset_path.to_owned()];
    config.candidates[0].directory = asset_path.join("missing-fixture");
    config.initial_comparisons.clear();
    let output = tempfile::tempdir_in("target").unwrap();
    let output_path = output.path().strip_prefix(&cwd).unwrap_or(output.path());
    let config_path = output_path.join("search.toml");
    fs::write(&config_path, toml::to_string_pretty(&config).unwrap()).unwrap();
    let report = run_with_options(&config_path, &output_path.join("run"), true)
        .await
        .unwrap();
    assert_eq!(report["status"], "cached_only_shortfall");
    assert_eq!(report["trials"][0]["status"], "waiting_for_cached_assets");
    assert_eq!(report["trials"][0]["passed"], serde_json::Value::Null);
    assert_eq!(report["download_bytes_reserved"], 0);
    assert!(
        !config.candidates[0]
            .directory
            .join("gguf-plan.json")
            .exists()
    );
}

#[test]
fn committed_search_roster_is_bounded_and_source_deduplicated() {
    let config = SearchConfig::load(Path::new("config/teacher-search.toml")).unwrap();
    assert_eq!(config.candidates.len(), 6);
    assert_eq!(
        config
            .candidates
            .iter()
            .map(|c| &c.source_model_id)
            .collect::<BTreeSet<_>>()
            .len(),
        5
    );
    let logical = config
        .candidates
        .iter()
        .map(|c| c.bytes + (64 << 20))
        .sum::<u64>();
    assert!(logical <= config.maximum_download_bytes);
}
