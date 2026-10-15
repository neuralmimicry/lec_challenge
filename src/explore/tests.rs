//! Protocol-only QA fixtures; no provider call, model run or learning is invented.
use super::*;

fn placeholder() -> Artefact {
    Artefact {
        path: "labelled-unit-fixture-only".into(),
        sha256: "a".repeat(64),
        bytes: 0,
    }
}
fn selection() -> selection::Selection {
    selection::Selection {
        schema_version: 1,
        evidence_kind: "labelled-unit-fixture-only".into(),
        registry: placeholder(),
        search_freeze: placeholder(),
        search_state_snapshot: placeholder(),
        search_report_snapshot: placeholder(),
        models: vec![],
        trials: vec![],
        runtime: RuntimeOptions::default(),
        target_models: 3,
        minimum_pass_rate: 0.8,
        qualified_target_met: false,
        held_out_used: false,
        production_eligible: false,
    }
}

#[test]
fn invocation_budget_and_offline_permission_do_not_change_frozen_semantics() {
    let original = ExploreOptions::default();
    let mut invocation = original.clone();
    invocation.offline = true;
    invocation.maximum_run_seconds = 60;
    assert_eq!(
        workflow::semantics(&original).unwrap(),
        workflow::semantics(&invocation).unwrap()
    );
    invocation.content_max_tokens += 1;
    assert_ne!(
        workflow::semantics(&original).unwrap(),
        workflow::semantics(&invocation).unwrap()
    );
}

#[test]
fn workspace_paths_reject_escape_without_accessing_model_assets() {
    for path in ["../other", "/tmp/other", "reports/../other", ""] {
        assert!(relative(Path::new(path)).is_err());
    }
    relative(Path::new("reports/local-study/explore/round-001")).unwrap();
}

#[test]
fn train_comparison_reuse_rejects_development_and_changed_decoding() {
    let selection = selection();
    let tasks = crate::tasks::corpus()
        .into_iter()
        .filter(|task| task.split == crate::tasks::Split::Train)
        .collect::<Vec<_>>();
    let mut plan = json!({"split":"development","tasks":tasks,"runtime":selection.runtime,"max_tokens":512,"seed":1729,"temperature":0.2,"top_p":0.95});
    let tmp = tempfile::tempdir().unwrap();
    assert!(!workflow::reusable(&plan, &selection, 512, tmp.path()).unwrap());
    plan["split"] = "train".into();
    plan["seed"] = 2718.into();
    assert!(!workflow::reusable(&plan, &selection, 512, tmp.path()).unwrap());
    plan["seed"] = 1729.into();
    plan["generation_timeout_seconds"] = 121.into();
    assert!(!workflow::reusable(&plan, &selection, 512, tmp.path()).unwrap());
}

#[test]
fn defaults_are_finite_and_never_open_the_sealed_gate() {
    let defaults = ExploreOptions::default();
    assert_eq!(defaults.maximum_run_seconds, 86400);
    assert_eq!(defaults.collection_max_tokens, 512);
    assert_eq!(defaults.content_maximum_tasks, 48);
    assert_eq!(defaults.top_groups, 2);
    assert!(defaults.output.starts_with("reports/local-study"));
    assert_eq!(
        runtime_arguments(&RuntimeOptions::default()),
        ["--device", "cpu", "--gpu", "0", "--dtype", "f32"]
    );
}
