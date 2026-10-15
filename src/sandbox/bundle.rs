//! Hash-verified, network-free OCI restoration for a newly unpacked bundle.
use super::*;

#[derive(Deserialize)]
struct ImageManifest {
    schema_version: u32,
    archive: String,
    bytes: u64,
    sha256: String,
    image_id: String,
}

/// Restore only the archived image whose complete bytes match the manifest.
/// This function never pulls an image or builds/downloads a dependency.
pub async fn restore_image_bundle(directory: &Path, cfg: &SandboxConfig) -> Result<String> {
    let manifest: ImageManifest = serde_json::from_slice(
        &tokio::fs::read(directory.join("manifest.json"))
            .await
            .context("offline OCI manifest missing")?,
    )?;
    ensure!(
        manifest.schema_version == 1 && manifest.image_id.trim_start_matches("sha256:").len() == 64,
        "invalid OCI bundle manifest"
    );
    ensure!(
        cfg.expected_image_id.is_empty()
            || cfg.expected_image_id.trim_start_matches("sha256:")
                == manifest.image_id.trim_start_matches("sha256:"),
        "OCI bundle differs from the frozen image"
    );
    let relative = Path::new(&manifest.archive);
    ensure!(
        !relative.is_absolute()
            && relative
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "OCI archive path escapes bundle"
    );
    let root = directory.canonicalize()?;
    let archive = root.join(relative).canonicalize()?;
    ensure!(
        archive.starts_with(&root),
        "OCI archive symlink escapes bundle"
    );
    ensure!(
        tokio::fs::metadata(&archive).await?.len() == manifest.bytes,
        "OCI archive size mismatch"
    );
    let hash_path = archive.clone();
    let hash =
        tokio::task::spawn_blocking(move || crate::artefact::file_hash(&hash_path)).await??;
    ensure!(hash == manifest.sha256, "OCI archive checksum mismatch");
    tracing::info!(archive=%archive.display(),bytes=manifest.bytes,"loading verified offline OCI image");
    let output = timeout(
        Duration::from_secs(300),
        Command::new(&cfg.runtime)
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
        .args([
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            &manifest.image_id,
        ])
        .output()
        .await?;
    ensure!(
        identity.status.success()
            && String::from_utf8_lossy(&identity.stdout)
                .trim()
                .trim_start_matches("sha256:")
                == manifest.image_id.trim_start_matches("sha256:"),
        "loaded OCI image ID mismatch"
    );
    // Restore the human-readable alias only after verifying immutable identity.
    let tagged = Command::new(&cfg.runtime)
        .args(["tag", &manifest.image_id, &cfg.image])
        .status()
        .await?;
    ensure!(tagged.success(), "could not restore sandbox image alias");
    Ok(manifest.image_id)
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
        let error = restore_image_bundle(directory.path(), &SandboxConfig::default())
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("path escapes"), "{error}");
    }
}
