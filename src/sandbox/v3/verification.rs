//! Reconstruct every retained native lifecycle/drain artefact before grading.
use super::*;
fn verify_stream(evidence: &stream::StreamEvidence) -> Result<Vec<u8>> {
    let metadata = fs::metadata(&evidence.path)?;
    ensure!(
        metadata.len() <= evidence.cap as u64,
        "runtime stream grew beyond cap"
    );
    let bytes = fs::read(&evidence.path)?;
    ensure!(
        bytes.len() == evidence.bytes_retained
            && tasks::hash(&bytes) == evidence.sha256
            && evidence.bytes_seen >= bytes.len() as u64
            && !evidence.overflow
            && evidence.eof
            && evidence.error.is_none(),
        "runtime stream evidence changed or incomplete"
    );
    Ok(bytes)
}

/// Recheck the caller's durable run identity against the actual lifecycle.
pub(super) fn verify_prepared_binding(
    prepared: &PreparedGrade,
    report: &SandboxReport,
) -> Result<()> {
    let reference = report
        .lifecycle_evidence
        .as_ref()
        .context("prepared report lacks lifecycle receipt")?;
    let bytes = fs::read(&reference.receipt_path)?;
    ensure!(
        tasks::hash(&bytes) == reference.receipt_sha256,
        "prepared lifecycle receipt changed"
    );
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    ensure!(
        receipt.run_id == prepared.run_id
            && receipt.nonce == prepared.ownership_nonce
            && receipt.code_sha256 == prepared.code_sha256
            && receipt.policy_sha256 == prepared.policy_sha256,
        "prepared runtime identity differs"
    );
    Ok(())
}
pub(super) fn verify(report: &SandboxReport, cfg: &SandboxConfig) -> Result<bool> {
    let reference = report
        .lifecycle_evidence
        .as_ref()
        .context("controller three lifecycle evidence absent")?;
    let bytes = fs::read(&reference.receipt_path)?;
    ensure!(
        reference.receipt_path.canonicalize()?
            == cfg
                .report_dir
                .join(format!("{}.lifecycle.json", report.run_id))
                .canonicalize()?,
        "lifecycle receipt path differs from exact run"
    );
    ensure!(
        bytes.len() <= 8 << 20 && tasks::hash(&bytes) == reference.receipt_sha256,
        "lifecycle receipt changed"
    );
    let receipt: Receipt = serde_json::from_slice(&bytes)?;
    ensure!(
        receipt.schema_version == 3
            && receipt.run_id == report.run_id
            && receipt.image_id == report.image_id
            && receipt.policy_sha256 == report.policy_sha256
            && receipt.code_sha256 == report.code_sha256
            && receipt.native_timeout_secs == 80
            && receipt.candidate_timeout_secs == cfg.wall_timeout_secs
            && receipt.pid1_watchdog_timeout_secs == 20
            && receipt.container_removed
            && receipt.failure.is_none(),
        "owned lifecycle incomplete or mismatched"
    );
    let id = receipt
        .container_id
        .as_ref()
        .context("owned container ID absent")?;
    let terminal = receipt
        .terminal_inspection
        .as_ref()
        .context("owned terminal inspection absent")?;
    ensure!(
        terminal["Id"] == *id
            && terminal["Image"]
                .as_str()
                .map(|s| s.trim_start_matches("sha256:"))
                == Some(receipt.image_id.trim_start_matches("sha256:"))
            && terminal["Name"].as_str().map(|s| s.trim_start_matches('/'))
                == Some(receipt.container_name.as_str()),
        "terminal identity differs"
    );
    for (key, expected) in [
        ("org.lec.sandbox.nonce", receipt.nonce.as_str()),
        ("org.lec.sandbox.scope", receipt.scope.as_str()),
        ("org.lec.sandbox.controller", "3"),
    ] {
        ensure!(
            terminal["Config"]["Labels"][key] == expected,
            "terminal ownership label changed"
        );
    }
    let mut terminal_raw = None;
    for call in &receipt.calls {
        for stream in [&call.stdout, &call.stderr] {
            ensure!(
                stream
                    .path
                    .parent()
                    .context("stream parent missing")?
                    .canonicalize()?
                    == cfg.report_dir.canonicalize()?
                    && stream
                        .path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .is_some_and(|s| s.starts_with(&format!("{}.", report.run_id))),
                "runtime metadata stream escaped exact run"
            );
        }
        if call.operation == "terminal-inspect" {
            let entries: Vec<Value> = serde_json::from_slice(&verify_stream(&call.stdout)?)?;
            ensure!(
                entries.len() == 1 && terminal_raw.is_none(),
                "terminal raw inspection missing or ambiguous"
            );
            terminal_raw = Some(entries[0].clone());
        }
        verify_stream(&call.stdout)?;
        verify_stream(&call.stderr)?;
        ensure!(
            !call.deadline_exceeded,
            "native metadata operation exceeded deadline"
        );
    }
    ensure!(
        terminal_raw.as_ref() == Some(terminal),
        "terminal snapshot differs from actual retained native inspection"
    );
    let audit = receipt
        .audit
        .as_ref()
        .context("retained audit binding absent")?;
    ensure!(
        audit.path.canonicalize()? == report.audit_path.canonicalize()?
            && audit.sha256 == report.audit_sha256,
        "report audit differs from durable drain"
    );
    let binding = receipt
        .cgroup
        .as_ref()
        .context("cached exact cgroup binding absent")?;
    ensure!(
        binding.container_id == *id
            && binding.device > 0
            && binding.inode > 0
            && binding
                .directory
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|leaf| leaf == id
                    || leaf == format!("libpod-{id}.scope")
                    || leaf == format!("docker-{id}.scope")),
        "cached cgroup does not bind this exact owned container"
    );
    let marker = eof_marker(
        &verify_stream(receipt.stderr.as_ref().context("runtime stderr missing")?)?,
        cfg,
        receipt
            .input_release_unix_ms
            .context("input release absent")?,
    )?;
    ensure!(
        receipt.eof_timer_metadata.as_ref() == Some(&marker),
        "retained trusted EOF marker changed"
    );
    let process = receipt
        .process_binding
        .as_ref()
        .context("exact namespace process binding absent")?;
    ensure!(
        process["namespace_pid1_verified"] == true
            && receipt.gated_inspection.as_ref().is_some_and(
                |inspection| inspection["State"]["Pid"] == process["host_pid"]
                    && inspection["Id"] == *id
                    && inspection["Config"]["Timeout"] == 80
            ),
        "gated namespace process identity differs"
    );
    if receipt.classification == "trusted_pid1_candidate_deadline" {
        let audit_value: Value = serde_json::from_slice(&verify_stream(audit)?)?;
        audit_timer_metadata(&audit_value, &marker, cfg)?;
        let elapsed = audit_value["deadline_elapsed_ms"]
            .as_u64()
            .context("watchdog elapsed absent")?;
        let timing = report
            .timeout_evidence
            .as_ref()
            .context("watchdog timeout timing absent")?;
        ensure!(
            cfg.wall_timeout_secs == 20
                && audit_value["deadline_exceeded"] == true
                && elapsed >= 20000
                && receipt.pid1_watchdog.as_ref() == Some(&audit_value)
                && terminal["State"]["ExitCode"] == 124
                && terminal["State"]["Running"] == false
                && timing.deadline_ms == 20000
                && timing.wait_elapsed_ms == elapsed
                && !timing.kill_acknowledged
                && timing.container_removed
                && !report.passed
                && report.quarantined
                && !report.trace_complete
                && report.cases_passed == 0,
            "independent watchdog deadline not proved"
        );
        return Ok(true);
    }
    verify_stream(audit)?;
    verify_stream(
        receipt
            .stderr
            .as_ref()
            .context("retained runtime stderr absent")?,
    )?;
    if receipt.classification == "candidate_wall_deadline" {
        let kill = receipt
            .kill
            .as_ref()
            .context("cached deadline kill absent")?;
        let timing = report
            .timeout_evidence
            .as_ref()
            .context("candidate timeout receipt absent")?;
        ensure!(
            receipt.guard_stop_reason.as_deref() == Some("execution_wall_deadline")
                && kill.populated_before
                && kill.kill_written
                && kill.empty_observed
                && kill.failure.is_none()
                && kill.elapsed_from_release_ms >= cfg.wall_timeout_secs * 1000
                && timing.kill_acknowledged
                && timing.container_removed
                && timing.deadline_ms == cfg.wall_timeout_secs * 1000
                && timing.wait_elapsed_ms == kill.elapsed_from_release_ms
                && !report.passed
                && report.quarantined
                && !report.trace_complete
                && report.cases_passed == 0,
            "unproved cached candidate timeout"
        );
        return Ok(true);
    }
    ensure!(
        receipt.classification == "completed_supervisor_audit"
            && terminal["State"]["Running"] == false
            && terminal["State"]["ExitCode"] == 0,
        "trusted supervisor did not complete normally"
    );
    let audit_value: Value = serde_json::from_slice(&verify_stream(audit)?)?;
    audit_timer_metadata(&audit_value, &marker, cfg)?;
    ensure!(
        audit_value["deadline_exceeded"] == false
            && audit_value["deadline_elapsed_ms"]
                .as_u64()
                .is_some_and(|elapsed| elapsed < 20000),
        "normal audit timer invalid"
    );
    Ok(false)
}
