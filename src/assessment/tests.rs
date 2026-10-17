//! Adversarial log fixtures cannot become completed assessment evidence.
use super::*;

#[test]
fn missing_inputs_and_an_aggregate_ship_label_do_not_establish_completion() {
    let root = tempfile::tempdir().unwrap();
    artefact::write_json(
        &root.path().join("reports/eval.json"),
        &json!({"decision":"SHIP","effect":1.0}),
    )
    .unwrap();
    let audit = inspect(root.path(), Path::new("reports/missing-exploration.json")).unwrap();
    assert!(!audit.original_request_satisfied);
    assert_eq!(
        audit
            .requirements
            .iter()
            .find(|r| r.id == "tasks")
            .unwrap()
            .status,
        EvidenceStatus::Verified
    );
    assert_ne!(
        audit
            .requirements
            .iter()
            .find(|r| r.id == "gate")
            .unwrap()
            .status,
        EvidenceStatus::Verified
    );
    assert_eq!(
        audit
            .requirements
            .iter()
            .filter(|r| r.id.starts_with("drill_"))
            .count(),
        9
    );
}

#[test]
fn fixture_success_duplicate_drills_and_report_complete_flags_cannot_pass_the_audit() {
    let root = tempfile::tempdir().unwrap();
    let names = [
        "kill_orchestrator_mid_round",
        "provider_error_then_rate_limit",
        "grader_gaming",
        "escape_attempts",
        "planted_heldout_task",
        "deliberately_worse_checkpoint",
        "train_gain_heldout_loss",
        "seed_only_checkpoints",
        "round_budget_exceeded",
    ];
    let mut cases=names.into_iter().map(|name|json!({"name":name,"status":"PASSED","evidence_kind":"synthetic_fixture","detail":{"passed":true}})).collect::<Vec<_>>();
    cases.push(cases[0].clone());
    artefact::write_json(
        &root.path().join("reports/drill.json"),
        &json!({"complete":true,"cases":cases}),
    )
    .unwrap();
    let audit = inspect(root.path(), Path::new("absent.json")).unwrap();
    assert!(!audit.original_request_satisfied);
    assert!(
        audit
            .requirements
            .iter()
            .filter(|r| r.id.starts_with("drill_"))
            .all(|r| r.status == EvidenceStatus::Invalid)
    );
}

#[test]
fn malformed_reports_are_retained_as_errors_and_development_observations_stay_exploratory() {
    let root = tempfile::tempdir().unwrap();
    artefact::atomic_write(
        &root.path().join("reports/drill.json"),
        b"malformed fixture",
    )
    .unwrap();
    artefact::write_json(&root.path().join("reports/exploration.json"),&json!({"status":"completed_with_teacher_target_shortfall","success":false,
        "held_out_used":false,"stages":{"failed-stage":{"status":"verified_complete","details":{"healthy":false}}},
        "details":{"baseline_comparisons":[{"profile":"full","candidate_passed":1,"baseline_passed":0,"planned_tasks":16,"quality_comparison_valid":true,"held_out_used":false}]}})).unwrap();
    let audit = inspect(root.path(), Path::new("reports/exploration.json")).unwrap();
    assert!(!audit.original_request_satisfied);
    let development = &audit.exploration.as_ref().unwrap()["reported_development_comparisons"][0];
    assert_eq!(development["candidate_passed"], 1);
    assert_eq!(development["assessment_promotion"], false);
    assert_eq!(
        audit
            .requirements
            .iter()
            .find(|r| r.id == "drill_grader_gaming")
            .unwrap()
            .status,
        EvidenceStatus::Invalid
    );
    assert!(
        audit
            .inputs
            .iter()
            .any(|a| a.path == Path::new("reports/drill.json"))
    );
}

#[test]
fn repeated_audits_do_not_overwrite_previous_observations() {
    let root = tempfile::tempdir().unwrap();
    let report = inspect(root.path(), Path::new("absent.json")).unwrap();
    let output = root.path().join("audit");
    write(&report, &output).unwrap();
    let before = fs::read(output.join("audit.json")).unwrap();
    assert!(write(&report, &output).is_err());
    assert_eq!(fs::read(output.join("audit.json")).unwrap(), before);
}
