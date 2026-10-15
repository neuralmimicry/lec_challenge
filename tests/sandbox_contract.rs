//! Explicit container QA. These tests execute real Rust candidates and require
//! the pinned, locally bootstrapped image and a verified Linux runtime.
use anyhow::{Result, ensure};
use lec_distil::{
    sandbox::{self, SandboxConfig},
    tasks,
};
use serde_json::json;
use std::{fs, path::Path};

async fn configuration() -> Result<SandboxConfig> {
    let mut config = SandboxConfig::default();
    config.expected_image_id = sandbox::policy_evidence(&config).await?.1;
    config.report_dir = "reports/sandbox-qa/runs".into();
    Ok(config)
}

#[tokio::test]
#[ignore = "loads the complete verified local OCI archive; never downloads an image"]
async fn offline_oci_archive_round_trip() -> Result<()> {
    let configuration = SandboxConfig::default();
    let loaded =
        sandbox::restore_image_bundle(Path::new("offline_bundle/images"), &configuration).await?;
    ensure!(
        sandbox::policy_evidence(&configuration).await?.1 == loaded,
        "loaded immutable identity mismatch"
    );
    fs::create_dir_all("reports/sandbox-qa")?;
    fs::write(
        "reports/sandbox-qa/oci-round-trip.json",
        serde_json::to_vec_pretty(
            &json!({"passed":true,"evidence_kind":"actual_hash_verified_offline_oci_load","image_id":loaded,"archive":"offline_bundle/images/lec-rust-sandbox.tar","network_downloads":0}),
        )?,
    )?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires the explicitly bootstrapped OCI image and actual kernel controls"]
async fn real_reference_candidates_across_all_splits_and_evidence_tampering() -> Result<()> {
    let config = configuration().await?;
    let mut reports = vec![];
    for (task_id, path) in [
        ("ledger.sum", "sandbox/examples/sum.rs"),
        ("encoding.rot13", "sandbox/examples/rot13.rs"),
        ("geometry.box", "sandbox/examples/bounding_box.rs"),
    ] {
        let task = tasks::corpus()
            .into_iter()
            .find(|t| t.id == task_id)
            .expect("reference task");
        let code = fs::read_to_string(path)?;
        let report = sandbox::grade(&task, &code, &config).await?;
        ensure!(
            report.passed && report.trace_complete && !report.quarantined,
            "reference {} failed: {:?}",
            task_id,
            report.failure
        );
        sandbox::verify_report(&report, &config, &task, &code)?;
        let mut forged = report.clone();
        forged.cases_passed -= 1;
        ensure!(
            sandbox::verify_report(&forged, &config, &task, &code).is_err(),
            "aggregate score tampering accepted"
        );
        ensure!(
            sandbox::verify_report(
                &report,
                &config,
                &task,
                "pub fn solve(_:serde_json::Value)->serde_json::Value{serde_json::Value::Null}"
            )
            .is_err(),
            "candidate substitution accepted"
        );
        let mut substituted = report.clone();
        substituted.image_id = "forged-image".into();
        ensure!(
            sandbox::verify_report(&substituted, &config, &task, &code).is_err(),
            "image substitution accepted"
        );
        reports.push(report);
    }
    fs::create_dir_all("reports/sandbox-qa")?;
    fs::write(
        "reports/sandbox-qa/references.json",
        serde_json::to_vec_pretty(
            &json!({"evidence_kind":"actual_reference_candidate_execution","passed":true,"reports":reports,"tampering_rejected":["aggregate_score","candidate_code","immutable_image"]}),
        )?,
    )?;
    Ok(())
}

#[tokio::test]
#[ignore = "executes oversized, hung and alias-probing programmes in fresh containers"]
async fn real_output_deadline_and_alias_defences() -> Result<()> {
    let config = configuration().await?;
    let task = tasks::corpus().remove(0);
    let mut reports = vec![];
    let oversized = "pub fn solve(input:serde_json::Value)->serde_json::Value{use std::io::Write; let _=std::io::stderr().write_all(&vec![b'x';300000]);serde_json::json!(input.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).sum::<i64>())}";
    let report = sandbox::grade(&task, oversized, &config).await?;
    ensure!(
        !report.passed
            && report.quarantined
            && report.trace_complete
            && report
                .failure
                .as_deref()
                .unwrap_or("")
                .contains("output cap"),
        "oversized stderr was not observed and rejected: {:?}",
        report.failure
    );
    sandbox::verify_report(&report, &config, &task, oversized)?;
    reports.push(report);
    let aliases = "pub fn solve(input:serde_json::Value)->serde_json::Value{let _=std::os::unix::fs::symlink(\"/proc/1/fd/1\",\"/tmp/audit-alias\");let _=std::fs::read(\"/lib/../../proc/1/fd/1\");serde_json::json!(input.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).sum::<i64>())}";
    let report = sandbox::grade(&task, aliases, &config).await?;
    ensure!(
        !report.passed && report.quarantined && report.trace_complete,
        "alias candidate was not rejected"
    );
    ensure!(
        report.audit_events.iter().any(|e| e.category == "process")
            && report
                .audit_events
                .iter()
                .any(|e| e.target == "/lib/../../proc/1/fd/1"),
        "alias attempts absent from trusted trace"
    );
    sandbox::verify_report(&report, &config, &task, aliases)?;
    reports.push(report);
    let hung =
        "pub fn solve(_:serde_json::Value)->serde_json::Value{loop{std::hint::black_box(7);}}";
    let report = sandbox::grade(&task, hung, &config).await?;
    ensure!(
        !report.passed
            && report.quarantined
            && report
                .failure
                .as_deref()
                .unwrap_or("")
                .contains("wall-clock deadline"),
        "hung candidate not stopped by host deadline: {:?}",
        report.failure
    );
    // It is deliberately not presented as a complete candidate syscall trace.
    ensure!(
        !report.trace_complete,
        "a killed supervisor cannot supply a complete trace"
    );
    let timing = report
        .timeout_evidence
        .as_ref()
        .expect("host timeout timing evidence");
    ensure!(
        timing.wait_elapsed_ms >= timing.deadline_ms
            && timing.wait_elapsed_ms <= timing.deadline_ms + 1000,
        "host timeout scheduling exceeded its declared bound"
    );
    ensure!(
        timing.kill_acknowledged && timing.kill_request_ms <= 5000 && timing.container_removed,
        "bounded SIGKILL/removal was not observed"
    );
    reports.push(report);
    fs::create_dir_all(Path::new("reports/sandbox-qa"))?;
    fs::write(
        "reports/sandbox-qa/adversarial.json",
        serde_json::to_vec_pretty(
            &json!({"evidence_kind":"actual_adversarial_candidate_execution","passed":true,"reports":reports}),
        )?,
    )?;
    Ok(())
}
