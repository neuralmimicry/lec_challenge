//! Caller-bound durable grading identities, published before runtime launch.
use super::*;
use std::io::Write;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedGrade {
    pub schema_version: u32,
    pub run_id: String,
    pub ownership_nonce: String,
    pub attempt_key: String,
    pub receipt_path: PathBuf,
    pub report_path: PathBuf,
    pub task: Task,
    pub code: String,
    pub config: SandboxConfig,
    pub code_sha256: String,
    pub policy_sha256: String,
    pub image_id: String,
}
fn publish(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("prepared grade parent absent")?;
    let temporary = parent.join(format!(".{}.pending", uuid::Uuid::new_v4()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::hard_link(&temporary, path)?;
    fs::remove_file(&temporary)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}
pub fn prepare_grade(
    task: &Task,
    code: &str,
    cfg: &SandboxConfig,
    attempt_key: &str,
    receipt_path: &Path,
) -> Result<PreparedGrade> {
    ensure!(
        cfg.controller_version == 3,
        "prepared grading requires host controller three"
    );
    validate_execution_platform(cfg)?;
    ensure!(
        !attempt_key.trim().is_empty() && attempt_key.len() <= 4096,
        "invalid grade attempt key"
    );
    ensure!(
        !cfg.expected_image_id.is_empty(),
        "prepared grading needs frozen image identity"
    );
    fs::create_dir_all(
        receipt_path
            .parent()
            .context("prepared receipt parent absent")?,
    )?;
    fs::create_dir_all(&cfg.report_dir)?;
    if receipt_path.try_exists()? {
        let prepared: PreparedGrade = serde_json::from_slice(&fs::read(receipt_path)?)?;
        verify_prepared(&prepared)?;
        ensure!(
            prepared.attempt_key == attempt_key
                && prepared.code == code
                && serde_json::to_vec(&prepared.task)? == serde_json::to_vec(task)?
                && serde_json::to_vec(&prepared.config)? == serde_json::to_vec(cfg)?,
            "prepared attempt identity changed"
        );
        return Ok(prepared);
    }
    let run_id = uuid::Uuid::new_v4().to_string();
    let prepared = PreparedGrade {
        schema_version: 3,
        report_path: cfg.report_dir.join(format!("{run_id}.json")),
        run_id,
        ownership_nonce: uuid::Uuid::new_v4().to_string(),
        attempt_key: attempt_key.into(),
        receipt_path: receipt_path.into(),
        task: task.clone(),
        code: code.into(),
        config: cfg.clone(),
        code_sha256: tasks::hash(code),
        policy_sha256: policy_sha256(cfg)?,
        image_id: cfg.expected_image_id.clone(),
    };
    publish(receipt_path, &serde_json::to_vec_pretty(&prepared)?)?;
    Ok(prepared)
}
pub fn verify_prepared(prepared: &PreparedGrade) -> Result<()> {
    ensure!(
        prepared.schema_version == 3
            && prepared.config.controller_version == 3
            && !prepared.attempt_key.trim().is_empty()
            && uuid::Uuid::parse_str(&prepared.run_id).is_ok()
            && uuid::Uuid::parse_str(&prepared.ownership_nonce).is_ok()
            && prepared.code_sha256 == tasks::hash(&prepared.code)
            && prepared.policy_sha256 == policy_sha256(&prepared.config)?
            && prepared.image_id == prepared.config.expected_image_id
            && prepared.report_path
                == prepared
                    .config
                    .report_dir
                    .join(format!("{}.json", prepared.run_id)),
        "prepared grade binding changed"
    );
    ensure!(
        fs::read(&prepared.receipt_path)? == serde_json::to_vec_pretty(prepared)?,
        "prepared grade receipt changed"
    );
    Ok(())
}
pub fn verify_prepared_report(prepared: &PreparedGrade, report: &SandboxReport) -> Result<()> {
    verify_prepared(prepared)?;
    ensure!(
        report.run_id == prepared.run_id
            && report.policy_sha256 == prepared.policy_sha256
            && report.image_id == prepared.image_id,
        "report differs from prepared attempt"
    );
    v3::verify_prepared_binding(prepared, report)?;
    verify_report(report, &prepared.config, &prepared.task, &prepared.code)
}
pub async fn execute_prepared(prepared: &PreparedGrade) -> Result<SandboxReport> {
    let prepared = prepared.clone();
    tokio::task::spawn_blocking({
        let prepared = prepared.clone();
        move || verify_prepared(&prepared)
    })
    .await??;
    let existing = tokio::task::spawn_blocking({
        let prepared = prepared.clone();
        move || -> Result<Option<SandboxReport>> {
            if !prepared.report_path.try_exists()? {
                return Ok(None);
            }
            let report: SandboxReport = serde_json::from_slice(&fs::read(&prepared.report_path)?)?;
            verify_prepared_report(&prepared, &report)?;
            Ok(Some(report))
        }
    })
    .await??;
    if let Some(report) = existing {
        return Ok(report);
    }
    let launch_path = prepared
        .config
        .report_dir
        .join(format!("{}.launch.json", prepared.run_id));
    ensure!(
        !launch_path.try_exists()?,
        "prepared attempt was already launched without a complete verified report; preserve interrupted infrastructure evidence"
    );
    let bytes = serde_json::to_vec_pretty(
        &json!({"schema_version":3,"run_id":prepared.run_id,"prepared_receipt":prepared.receipt_path,
        "prepared_sha256":tasks::hash(serde_json::to_vec_pretty(&prepared)?),"owner_pid":std::process::id(),"launch_unix_ms":crate::ledger::unix_millis()}),
    )?;
    tokio::task::spawn_blocking(move || publish(&launch_path, &bytes)).await??;
    let lease = runtime_lease::shared(&prepared.config.runtime).await?;
    ensure!(
        lease.supports_independent_wall_timeout(),
        "prepared grading requires native Podman startup/crash ceiling"
    );
    owned::run(lease, async move {
        recovery::recover(&prepared.config).await?;
        grade_identified(
            &prepared.task,
            &prepared.code,
            &prepared.config,
            Some((&prepared.run_id, &prepared.ownership_nonce)),
        )
        .await
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_caller_attempt_keeps_its_exact_identity_and_rejects_code_substitution() -> Result<()> {
        let root = tempfile::tempdir()?;
        let task = tasks::corpus().remove(0);
        let cfg = SandboxConfig {
            expected_image_id: "a".repeat(64),
            report_dir: root.path().join("reports"),
            ..Default::default()
        };
        let path = root.path().join("prepared.json");
        let a = prepare_grade(
            &task,
            "pub fn solve(_:serde_json::Value)->serde_json::Value{serde_json::Value::Null}",
            &cfg,
            "fixture-attempt",
            &path,
        )?;
        let b = prepare_grade(&task, &a.code, &cfg, "fixture-attempt", &path)?;
        assert_eq!(a.run_id, b.run_id);
        assert_eq!(a.ownership_nonce, b.ownership_nonce);
        assert!(!a.report_path.exists());
        verify_prepared(&a)?;
        assert!(prepare_grade(&task, "different code", &cfg, "fixture-attempt", &path).is_err());
        assert!(prepare_grade(&task, &a.code, &cfg, "different attempt", &path).is_err());
        Ok(())
    }
    #[test]
    fn greater_than_trusted_watchdog_budget_is_rejected_before_publication() -> Result<()> {
        let root = tempfile::tempdir()?;
        let task = tasks::corpus().remove(0);
        let cfg = SandboxConfig {
            wall_timeout_secs: 21,
            expected_image_id: "a".repeat(64),
            report_dir: root.path().join("reports"),
            ..Default::default()
        };
        let path = root.path().join("prepared.json");
        assert!(prepare_grade(&task, "candidate", &cfg, "fixture-attempt", &path).is_err());
        assert!(!path.exists());
        Ok(())
    }
}
