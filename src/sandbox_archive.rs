//! Hash-verified, network-free OCI restoration for a newly unpacked bundle.
use crate::sandbox::SandboxConfig;
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn legacy_policy_version() -> u32 {
    1
}

#[derive(Deserialize)]
struct ImageManifest {
    schema_version: u32,
    #[serde(default = "legacy_policy_version")]
    policy_version: u32,
    #[serde(default = "legacy_image_architecture")]
    architecture: String,
    archive: String,
    bytes: u64,
    sha256: String,
    image_id: String,
}
fn legacy_image_architecture() -> String {
    "x86_64".into()
}

/// Restore only the archived image whose complete bytes match the manifest.
/// This function never pulls an image or builds/downloads a dependency.
pub async fn verify_image_bundle(
    directory: &Path,
    cfg: &SandboxConfig,
) -> Result<(PathBuf, String)> {
    let selected = directory.join(format!(
        "policy-{}/{}",
        cfg.policy_version,
        if cfg.policy_version == 1 {
            "x86_64"
        } else {
            &cfg.architecture
        }
    ));
    let directory = if selected.join("manifest.json").try_exists()? {
        selected.as_path()
    } else {
        directory
    };
    let manifest: ImageManifest = serde_json::from_slice(
        &tokio::fs::read(directory.join("manifest.json"))
            .await
            .context("offline OCI manifest missing")?,
    )?;
    ensure!(
        manifest.schema_version == 1
            && manifest.image_id.trim_start_matches("sha256:").len() == 64
            && manifest
                .image_id
                .trim_start_matches("sha256:")
                .bytes()
                .all(|b| b.is_ascii_hexdigit()),
        "invalid OCI bundle manifest"
    );
    ensure!(
        manifest.policy_version == cfg.policy_version
            && manifest.architecture
                == if cfg.policy_version == 1 {
                    "x86_64"
                } else {
                    &cfg.architecture
                },
        "OCI bundle policy/architecture differs from the configured sandbox"
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
        tokio::fs::metadata(&archive).await?.is_file()
            && tokio::fs::metadata(&archive).await?.len() == manifest.bytes,
        "OCI archive size mismatch"
    );
    let hash_path = archive.clone();
    let hash =
        tokio::task::spawn_blocking(move || crate::artefact::file_hash(&hash_path)).await??;
    ensure!(hash == manifest.sha256, "OCI archive checksum mismatch");
    Ok((archive, manifest.image_id))
}
