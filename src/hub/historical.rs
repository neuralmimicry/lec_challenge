//! Historical identities for already-recorded generations, never model loading.
//!
//! All metadata, plans, tokenisers, templates and licence bytes are revalidated.
//! LFS tensor bytes are deliberately not read: the caller must independently
//! bind the returned original manifest to its immutable generation evidence.
//! Every native loader still uses the full validators before new inference.

use super::*;
use std::collections::BTreeSet;

/// Revalidate a recorded dense teacher's original metadata without reading LFS
/// tensor contents. This proves an historical identity, not runnable weights.
pub fn validate_downloaded_asset_metadata(directory: &Path) -> Result<AssetManifest> {
    validate_downloaded_assets_with_policy(directory, false, true)
}

/// Revalidate both pinned GGUF repositories and every original auxiliary file,
/// without reading the selected LFS weight. Durable owned pruning is supported.
/// The returned proof cannot replace `validate_gguf_assets` before inference.
pub fn validate_gguf_asset_metadata(directory: &Path) -> Result<HubGgufProvenance> {
    super::gguf::validate_gguf_assets_with_policy(directory, &BTreeSet::new(), true)
}

pub(super) fn verify_weight_metadata(
    directory: &Path,
    manifest_name: &str,
    planned: &PlannedFile,
    recorded_sha256: &str,
) -> Result<()> {
    ensure!(
        valid_hex(recorded_sha256, 64),
        "historical weight SHA-256 is invalid"
    );
    let path = directory.join(&planned.filename);
    ensure_safe_path(&path)?;
    match &planned.publisher_identity {
        PublisherIdentity::LfsSha256(expected) => ensure!(
            expected == recorded_sha256,
            "historical weight identity differs from publisher LFS SHA-256"
        ),
        // Git hashes cannot establish the separate recorded SHA-256 without
        // reading bytes. Such small/unusual weights retain full verification.
        PublisherIdentity::GitBlobSha1(_) => {
            ensure!(
                verify_file(&path, planned)? == recorded_sha256,
                "historical Git weight checksum mismatch"
            );
            return Ok(());
        }
    }
    match fs::metadata(&path) {
        Ok(metadata) => ensure!(
            metadata.is_file() && metadata.len() == planned.bytes,
            "historical weight is not a regular file of the recorded size"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            verify_prune_authority(directory, manifest_name, planned, recorded_sha256)?;
        }
        Err(error) => return Err(error).context("inspect historical weight"),
    }
    Ok(())
}

/// An absent weight needs the same durable ownership/manifest tuple used by
/// pruning. Merely deleting a file, or adding a marker with a guessed hash,
/// cannot turn incomplete downloads into historical evidence.
fn verify_prune_authority(
    directory: &Path,
    manifest_name: &str,
    planned: &PlannedFile,
    recorded_sha256: &str,
) -> Result<()> {
    let read = |name: &str| -> Result<Vec<u8>> {
        let path = directory.join(name);
        ensure_safe_path(&path)?;
        ensure!(
            fs::metadata(&path)?.is_file() && fs::metadata(&path)?.len() <= 1024 * 1024,
            "historical prune identity oversized or not a regular file"
        );
        Ok(fs::read(path)?)
    };
    let pruned: PrunedWeightsProof = serde_json::from_slice(&read(".pruned-weights.json")?)?;
    let owner: serde_json::Value = serde_json::from_slice(&read(".teacher-search-owned.json")?)?;
    let workspace = std::env::current_dir()?.canonicalize()?;
    let absolute = directory.canonicalize()?;
    let relative = absolute
        .strip_prefix(&workspace)
        .context("historical pruned teacher lies outside its workspace")?;
    let manifest_bytes = read(manifest_name)?;
    let manifest: AssetManifest = serde_json::from_slice(&manifest_bytes)?;
    ensure!(
        pruned.schema_version == 1
            && pruned.created_unix_ms > 0
            && matches!(pruned.removal_state.as_str(), "planned" | "completed")
            && pruned.directory == relative
            && pruned.manifest_filename == manifest_name
            && pruned.manifest_sha256 == hex::encode(Sha256::digest(&manifest_bytes))
            && pruned.model_id == manifest.model_id
            && pruned.revision == manifest.revision
            && owner["schema_version"] == 1
            && owner["directory"] == serde_json::to_value(relative)?
            && owner["candidate_id"] == pruned.candidate_id
            && !pruned.candidate_id.is_empty()
            && owner["model_id"] == pruned.model_id
            && owner["revision"] == pruned.revision,
        "historical prune authority differs from immutable ownership/manifest"
    );
    let mut declared = BTreeSet::new();
    let mut matched = false;
    for file in pruned.removed_weights {
        ensure!(
            (allowed_filename(&file.filename) && file.filename.ends_with(".safetensors")
                || file.filename.ends_with(".gguf")
                    && !file.filename.contains(['/', '\\'])
                    && !file.filename.contains(".."))
                && declared.insert(file.filename.clone())
                && manifest.files.get(&file.filename) == Some(&file.sha256),
            "historical prune authority names an unexpected or duplicate weight"
        );
        if file.filename == planned.filename {
            ensure!(
                file.sha256 == recorded_sha256 && file.bytes == planned.bytes,
                "historical pruned weight differs from publisher-bound identity"
            );
            matched = true;
        }
    }
    ensure!(matched, "missing historical weight has no prune authority");
    Ok(())
}
