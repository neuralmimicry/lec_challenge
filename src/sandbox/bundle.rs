//! Exact archive restoration; validation is shared with the read-only doctor.
use super::*;
use crate::sandbox_archive::verify_image_bundle;
pub async fn restore_image_bundle(directory: &Path, cfg: &SandboxConfig) -> Result<String> {
    let (archive, image_id) = verify_image_bundle(directory, cfg).await?;
    let _runtime_lease = runtime_lease::exclusive(&cfg.runtime).await?;
    tracing::info!(archive=%archive.display(),"loading verified offline OCI image");
    let output = timeout(
        Duration::from_secs(300),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["load", "--input"])
            .arg(&archive)
            .output(),
    )
    .await
    .context("offline OCI load exceeded five minutes")??;
    ensure!(
        output.status.success(),
        "offline OCI load failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let identity = Command::new(&cfg.runtime)
        .kill_on_drop(true)
        .args(["image", "inspect", "--format", "{{.Id}}", &image_id])
        .output();
    let identity = timeout(Duration::from_secs(30), identity)
        .await
        .context("loaded image inspection exceeded metadata deadline")??;
    ensure!(
        identity.status.success()
            && String::from_utf8_lossy(&identity.stdout)
                .trim()
                .trim_start_matches("sha256:")
                == image_id.trim_start_matches("sha256:"),
        "loaded OCI image ID mismatch"
    );
    // Restore the human-readable alias only after verifying immutable identity.
    let tagged = timeout(
        Duration::from_secs(30),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["tag", &image_id, &cfg.image])
            .status(),
    )
    .await
    .context("sandbox alias restoration deadline exceeded")??;
    ensure!(tagged.success(), "could not restore sandbox image alias");
    Ok(image_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn archive_checksum_is_checked_before_the_runtime_can_be_invoked() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("image.tar"),
            b"labelled invalid archive fixture",
        )
        .unwrap();
        let bytes = fs::metadata(directory.path().join("image.tar"))
            .unwrap()
            .len();
        fs::write(directory.path().join("manifest.json"),serde_json::to_vec(&json!({"schema_version":1,"archive":"image.tar","bytes":bytes,"sha256":"0".repeat(64),"image_id":"1".repeat(64)})).unwrap()).unwrap();
        let cfg = SandboxConfig {
            runtime: "/this/runtime/must/not/be/invoked".into(),
            policy_version: 1,
            architecture: String::new(),
            ..Default::default()
        };
        let error = restore_image_bundle(directory.path(), &cfg)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("checksum mismatch"), "{error}");
    }
    #[tokio::test]
    async fn archive_paths_cannot_leave_the_bundle() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("manifest.json"),serde_json::to_vec(&json!({"schema_version":1,"archive":"../image.tar","bytes":0,"sha256":"0".repeat(64),"image_id":"1".repeat(64)})).unwrap()).unwrap();
        let config = SandboxConfig {
            policy_version: 1,
            architecture: String::new(),
            ..Default::default()
        };
        let error = restore_image_bundle(directory.path(), &config)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("path escapes"), "{error}");
    }
    #[tokio::test]
    async fn legacy_archive_is_rejected_before_loading_under_portable_arm_policy() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("manifest.json"), serde_json::to_vec(&json!({"schema_version":1,"archive":"image.tar","bytes":0,"sha256":"0".repeat(64),"image_id":"1".repeat(64)})).unwrap()).unwrap();
        let config = SandboxConfig {
            policy_version: 2,
            architecture: "aarch64".into(),
            runtime: "/this/runtime/must/not/be/invoked".into(),
            ..Default::default()
        };
        let error = restore_image_bundle(directory.path(), &config)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("policy/architecture"), "{error}");
    }
}
