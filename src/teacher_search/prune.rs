//! Remove only explicitly owned model-cache weights; preserve their evidence.
//!
//! The caller holds the global pool lease and establishes comparison/registry
//! adoption eligibility. This module independently checks directory, ownership,
//! immutable model identity and the per-download lock before touching files.

use crate::hub::{PrunedWeight, PrunedWeightsProof};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime},
};

const MARKER: &str = ".teacher-search-owned.json";
const MAX_IDENTITY_BYTES: u64 = 1024 * 1024;
const PARTIAL_AGE: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Ownership {
    schema_version: u32,
    directory: PathBuf,
    candidate_id: String,
    model_id: String,
    revision: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PruneReport {
    pub candidate_id: String,
    pub directory: PathBuf,
    pub dry_run: bool,
    /// Logical bytes actually removed, measured from the file rather than config.
    pub reclaimed_bytes: u64,
    pub planned_bytes: u64,
    /// Unix inode blocks removed; this is not a measured free-space delta.
    pub allocated_bytes_removed: u64,
    pub files: Vec<PathBuf>,
    pub retained_files: Vec<PathBuf>,
    pub metadata_preserved: bool,
}

fn relative(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "model cache path must be a traversal-free relative workspace path"
    );
    Ok(())
}

fn safe_path(workspace: &Path, relative_path: &Path) -> Result<PathBuf> {
    relative(relative_path)?;
    let mut absolute = workspace.to_owned();
    for part in relative_path.components() {
        absolute.push(part.as_os_str());
        match fs::symlink_metadata(&absolute) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "symlink in model-cache path"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("inspect owned model-cache path"),
        }
    }
    ensure!(
        absolute.starts_with(workspace),
        "model cache leaves its workspace"
    );
    Ok(absolute)
}

fn protected_path(workspace: &Path, model_dir: &Path, protected: &[PathBuf]) -> Result<()> {
    relative(model_dir)?;
    for path in [
        PathBuf::from("offline_bundle/base"),
        PathBuf::from("offline_bundle/students"),
        PathBuf::from("checkpoints"),
    ]
    .iter()
    .chain(protected)
    {
        let path = if path.is_absolute() {
            path.strip_prefix(workspace)
                .context("protected path lies outside the workspace")?
        } else {
            path.as_path()
        };
        relative(path)?;
        ensure!(
            !model_dir.starts_with(path) && !path.starts_with(model_dir),
            "model cache intersects a protected directory or its ancestor"
        );
    }
    Ok(())
}

fn valid_candidate(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

fn valid_identity(model: &str, revision: &str) -> bool {
    let pieces: Vec<_> = model.split('/').collect();
    pieces.len() == 2
        && pieces.iter().all(|piece| valid_candidate(piece))
        && revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn json_identity(path: &Path) -> Result<Option<(String, String)>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                metadata.is_file()
                    && !metadata.file_type().is_symlink()
                    && metadata.len() <= MAX_IDENTITY_BYTES,
                "model identity document is unsafe or oversized"
            );
            let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
            Ok(Some((
                value["model_id"]
                    .as_str()
                    .context("model identity missing repository")?
                    .into(),
                value["revision"]
                    .as_str()
                    .context("model identity missing immutable revision")?
                    .into(),
            )))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).context("inspect model identity document"),
    }
}

fn check_published_identity(directory: &Path, model_id: &str, revision: &str) -> Result<bool> {
    let mut found = false;
    for name in [
        "assets.json",
        "gguf-assets.json",
        "hub-plan.json",
        "gguf-plan.json",
    ] {
        if let Some((model, pin)) = json_identity(&directory.join(name))? {
            ensure!(
                model == model_id && pin == revision,
                "owned directory differs from the requested repository/revision tuple"
            );
            found = true;
        }
    }
    Ok(found)
}

fn inventory(directory: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        ensure!(
            output.len() < 10_000,
            "model-cache directory exceeds the bounded inventory"
        );
        let metadata = fs::symlink_metadata(entry.path())?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "symlink inside owned model-cache directory"
        );
        if metadata.is_dir() {
            inventory(&entry.path(), output)?;
        } else {
            ensure!(
                metadata.is_file(),
                "special file inside owned model-cache directory"
            );
            output.push(entry.path());
        }
    }
    Ok(())
}

fn atomic_marker(path: &Path, ownership: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("ownership marker lacks parent")?;
    let temporary = parent.join(format!(
        ".teacher-search-owned.{}.tmp",
        uuid::Uuid::new_v4()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let mut file = options.open(&temporary)?;
    file.write_all(&serde_json::to_vec_pretty(ownership)?)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(super) fn claim_directory(
    model_dir: &Path,
    candidate_id: &str,
    model_id: &str,
    revision: &str,
) -> Result<()> {
    claim_in(
        &std::env::current_dir()?.canonicalize()?,
        model_dir,
        candidate_id,
        model_id,
        revision,
    )
}

fn claim_in(
    workspace: &Path,
    model_dir: &Path,
    candidate_id: &str,
    model_id: &str,
    revision: &str,
) -> Result<()> {
    ensure!(
        valid_candidate(candidate_id) && valid_identity(model_id, revision),
        "invalid owned candidate identity"
    );
    protected_path(workspace, model_dir, &[])?;
    let directory = safe_path(workspace, model_dir)?;
    fs::create_dir_all(&directory)?;
    let directory = safe_path(workspace, model_dir)?;
    let mut entries = Vec::new();
    inventory(&directory, &mut entries)?;
    let expected = Ownership {
        schema_version: 1,
        directory: model_dir.to_owned(),
        candidate_id: candidate_id.into(),
        model_id: model_id.into(),
        revision: revision.into(),
    };
    let marker = directory.join(MARKER);
    if marker.exists() {
        let metadata = fs::metadata(&marker)?;
        ensure!(
            metadata.len() <= MAX_IDENTITY_BYTES,
            "ownership marker oversized"
        );
        let actual: Ownership = serde_json::from_slice(&fs::read(marker)?)?;
        ensure!(
            actual == expected,
            "directory already belongs to another immutable candidate"
        );
        check_published_identity(&directory, model_id, revision)?;
        return Ok(());
    }
    let identified = check_published_identity(&directory, model_id, revision)?;
    ensure!(
        entries.is_empty() || identified,
        "cannot adopt an unidentified existing model-cache directory"
    );
    atomic_marker(&marker, &expected)
}

fn weight_filename(name: &str) -> bool {
    let safe = !name.is_empty()
        && !name.contains("..")
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte));
    safe && (name.ends_with(".gguf")
        || name == "model.safetensors"
        || (name.starts_with("model-") && name.ends_with(".safetensors")))
}

fn stale_untracked_partial(
    directory: &Path,
    name: &str,
    metadata: &fs::Metadata,
    now: SystemTime,
) -> Result<bool> {
    let Some(stem) = name
        .strip_suffix(".partial")
        .or_else(|| name.strip_suffix(".download"))
    else {
        return Ok(false);
    };
    if !weight_filename(stem.strip_prefix('.').unwrap_or(stem)) {
        return Ok(false);
    }
    if directory.join(format!("{name}.json")).exists() {
        return Ok(false);
    }
    Ok(now.duration_since(metadata.modified()?).unwrap_or_default() > PARTIAL_AGE)
}

struct DownloadLease(File);
fn download_lease(directory: &Path) -> Result<DownloadLease> {
    let path = directory.join(".hub-download.lock");
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: flock borrows a live descriptor without taking ownership.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "active downloader prevents pruning this candidate"
        );
    }
    #[cfg(not(unix))]
    anyhow::bail!("safe pruning requires the supported Unix lease backend");
    Ok(DownloadLease(file))
}

impl Drop for DownloadLease {
    fn drop(&mut self) {
        let _ = self.0.sync_all();
    }
}

pub(super) fn prune_candidate(
    model_dir: &Path,
    protected: &[PathBuf],
    candidate_id: &str,
    dry_run: bool,
) -> Result<PruneReport> {
    prune_in(
        &std::env::current_dir()?.canonicalize()?,
        model_dir,
        protected,
        candidate_id,
        dry_run,
    )
}

pub(super) fn verify_pruned_directory(
    model_dir: &Path,
    candidate_id: &str,
    model_id: &str,
    revision: &str,
) -> Result<bool> {
    let workspace = std::env::current_dir()?.canonicalize()?;
    protected_path(&workspace, model_dir, &[])?;
    let directory = safe_path(&workspace, model_dir)?;
    let marker = directory.join(".pruned-weights.json");
    if !marker.exists() {
        return Ok(false);
    }
    let mut files = Vec::new();
    inventory(&directory, &mut files)?;
    ensure!(
        fs::metadata(&marker)?.len() <= MAX_IDENTITY_BYTES,
        "prune authority oversized"
    );
    let proof: PrunedWeightsProof = serde_json::from_slice(&fs::read(marker)?)?;
    let owner: Ownership = serde_json::from_slice(&fs::read(directory.join(MARKER))?)?;
    ensure!(
        proof.schema_version == 1
            && matches!(proof.removal_state.as_str(), "planned" | "completed")
            && proof.directory == model_dir
            && proof.candidate_id == candidate_id
            && proof.model_id == model_id
            && proof.revision == revision
            && owner.schema_version == 1
            && owner.directory == model_dir
            && owner.candidate_id == candidate_id
            && owner.model_id == model_id
            && owner.revision == revision
            && matches!(
                proof.manifest_filename.as_str(),
                "assets.json" | "gguf-assets.json"
            ),
        "prune authority differs from immutable ownership tuple"
    );
    let manifest_path = directory.join(&proof.manifest_filename);
    ensure!(
        crate::artefact::file_hash(&manifest_path)? == proof.manifest_sha256,
        "retained manifest changed after intentional prune"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(manifest_path)?)?;
    ensure!(
        manifest["model_id"] == model_id && manifest["revision"] == revision,
        "pruned manifest identity changed"
    );
    let mut missing = false;
    let mut names = std::collections::BTreeSet::new();
    for file in proof.removed_weights {
        ensure!(
            weight_filename(&file.filename)
                && names.insert(file.filename.clone())
                && manifest["files"][&file.filename] == file.sha256,
            "prune authority names an unexpected weight"
        );
        let path = directory.join(&file.filename);
        if path.exists() {
            ensure!(
                fs::metadata(&path)?.len() == file.bytes
                    && crate::artefact::file_hash(&path)? == file.sha256,
                "existing pruned weight has changed bytes"
            );
        } else {
            missing = true;
        }
    }
    Ok(missing)
}

fn prune_in(
    workspace: &Path,
    model_dir: &Path,
    protected: &[PathBuf],
    candidate_id: &str,
    dry_run: bool,
) -> Result<PruneReport> {
    ensure!(valid_candidate(candidate_id), "invalid prune candidate ID");
    protected_path(workspace, model_dir, protected)?;
    let directory = safe_path(workspace, model_dir)?;
    let mut entries = Vec::new();
    inventory(&directory, &mut entries)?;
    let marker = directory.join(MARKER);
    let metadata = fs::symlink_metadata(&marker)
        .context("model directory is not explicitly owned by teacher search")?;
    ensure!(
        metadata.is_file() && metadata.len() <= MAX_IDENTITY_BYTES,
        "unsafe ownership marker"
    );
    let ownership: Ownership = serde_json::from_slice(&fs::read(marker)?)?;
    ensure!(
        ownership.schema_version == 1
            && ownership.directory == model_dir
            && ownership.candidate_id == candidate_id
            && valid_identity(&ownership.model_id, &ownership.revision),
        "owned candidate ID/repository/revision tuple mismatch"
    );
    ensure!(
        check_published_identity(&directory, &ownership.model_id, &ownership.revision)?,
        "pruning requires retained published model identity metadata"
    );
    let _lease = download_lease(&directory)?;
    let mut report = PruneReport {
        candidate_id: candidate_id.into(),
        directory: model_dir.into(),
        dry_run,
        reclaimed_bytes: 0,
        planned_bytes: 0,
        allocated_bytes_removed: 0,
        files: vec![],
        retained_files: vec![],
        metadata_preserved: true,
    };
    let now = SystemTime::now();
    let mut remove = Vec::new();
    for path in entries {
        let name = path
            .file_name()
            .context("owned file lacks filename")?
            .to_str()
            .context("non-UTF8 owned cache filename")?;
        let metadata = fs::symlink_metadata(&path)?;
        let top_level = path.parent() == Some(directory.as_path());
        if top_level
            && (weight_filename(name) || stale_untracked_partial(&directory, name, &metadata, now)?)
        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                ensure!(
                    metadata.nlink() == 1,
                    "hard-linked candidate weight cannot be accounted as reclaimed"
                );
            }
            report.planned_bytes = report
                .planned_bytes
                .checked_add(metadata.len())
                .context("prune byte total overflow")?;
            report.files.push(path.strip_prefix(workspace)?.into());
            remove.push((path, metadata));
        } else {
            report
                .retained_files
                .push(path.strip_prefix(workspace)?.into());
        }
    }
    if !dry_run {
        let manifest_filename = if directory.join("gguf-assets.json").exists() {
            "gguf-assets.json"
        } else {
            "assets.json"
        };
        let manifest_path = directory.join(manifest_filename);
        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let mut removed_weights = Vec::new();
        for (path, before) in &remove {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid owned weight name")?;
            if !weight_filename(name) {
                continue;
            }
            let expected = manifest["files"][name]
                .as_str()
                .context("candidate weight is absent from the retained asset manifest")?;
            let actual = crate::artefact::file_hash(path)?;
            ensure!(
                actual == expected,
                "candidate weight changed since its retained manifest; refusing prune/rehydration authority"
            );
            removed_weights.push(PrunedWeight {
                filename: name.into(),
                sha256: actual,
                bytes: before.len(),
            });
        }
        let mut proof = PrunedWeightsProof {
            schema_version: 1,
            candidate_id: candidate_id.into(),
            model_id: ownership.model_id.clone(),
            revision: ownership.revision.clone(),
            directory: model_dir.into(),
            manifest_filename: manifest_filename.into(),
            manifest_sha256: crate::artefact::file_hash(&manifest_path)?,
            created_unix_ms: crate::ledger::unix_millis(),
            removal_state: "planned".into(),
            removed_weights,
        };
        let proof_path = directory.join(".pruned-weights.json");
        if proof_path.exists() {
            ensure!(
                fs::metadata(&proof_path)?.len() <= MAX_IDENTITY_BYTES,
                "previous prune authority oversized"
            );
            let previous: PrunedWeightsProof = serde_json::from_slice(&fs::read(&proof_path)?)?;
            ensure!(
                previous.schema_version == 1
                    && previous.candidate_id == proof.candidate_id
                    && previous.model_id == proof.model_id
                    && previous.revision == proof.revision
                    && previous.directory == proof.directory
                    && previous.manifest_filename == proof.manifest_filename
                    && previous.manifest_sha256 == proof.manifest_sha256,
                "previous prune authority differs from immutable model metadata"
            );
            for weight in previous.removed_weights {
                ensure!(
                    weight_filename(&weight.filename)
                        && manifest["files"][&weight.filename] == weight.sha256,
                    "previous prune authority names an unrecognised weight"
                );
                if !directory.join(&weight.filename).exists()
                    && !proof
                        .removed_weights
                        .iter()
                        .any(|file| file.filename == weight.filename)
                {
                    proof.removed_weights.push(weight);
                }
            }
            let history = directory.join(".prune-history");
            fs::create_dir_all(&history)?;
            let original = fs::read(&proof_path)?;
            let archived = history.join(format!("pruned-{}.json", uuid::Uuid::new_v4()));
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(archived)?;
            file.write_all(&original)?;
            file.sync_all()?;
            File::open(history)?.sync_all()?;
        }
        // Durable intent also covers a crash between two authorised removals.
        atomic_marker(&proof_path, &proof)?;
        for (path, before) in remove {
            let after = fs::symlink_metadata(&path)?;
            ensure!(
                after.is_file() && after.len() == before.len(),
                "candidate changed during guarded prune"
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                ensure!(
                    after.dev() == before.dev()
                        && after.ino() == before.ino()
                        && after.nlink() == 1,
                    "candidate inode changed during guarded prune"
                );
                report.allocated_bytes_removed = report
                    .allocated_bytes_removed
                    .checked_add(
                        before
                            .blocks()
                            .checked_mul(512)
                            .context("allocated prune block overflow")?,
                    )
                    .context("allocated prune byte overflow")?;
            }
            fs::remove_file(path)?;
            report.reclaimed_bytes = report
                .reclaimed_bytes
                .checked_add(before.len())
                .context("reclaimed byte total overflow")?;
        }
        File::open(&directory)?.sync_all()?;
        proof.removal_state = "completed".into();
        atomic_marker(&proof_path, &proof)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path) -> Result<PathBuf> {
        let path = PathBuf::from("models/candidate");
        claim_in(
            root,
            &path,
            "candidate",
            "organisation/model",
            &"a".repeat(40),
        )?;
        fs::write(
            root.join(&path).join("gguf-assets.json"),
            serde_json::to_vec(
                &serde_json::json!({"model_id":"organisation/model","revision":"a".repeat(40),"files":{}}),
            )?,
        )?;
        Ok(path)
    }

    fn record_weight(directory: &Path, name: &str) -> Result<()> {
        let path = directory.join("gguf-assets.json");
        let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        manifest["files"][name] =
            serde_json::json!(crate::artefact::file_hash(&directory.join(name))?);
        fs::write(path, serde_json::to_vec(&manifest)?)?;
        Ok(())
    }

    #[test]
    fn measured_prune_preserves_metadata_and_dry_run_is_non_destructive() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = fixture(root.path())?;
        let directory = root.path().join(&path);
        fs::write(directory.join("model-q4.gguf"), b"twelve bytes")?;
        record_weight(&directory, "model-q4.gguf")?;
        fs::write(directory.join("config.json"), b"metadata")?;
        fs::create_dir(directory.join("source"))?;
        fs::write(directory.join("source/LICENSE"), b"licence")?;
        let preview = prune_in(root.path(), &path, &[], "candidate", true)?;
        assert_eq!(preview.planned_bytes, 12);
        assert_eq!(preview.reclaimed_bytes, 0);
        assert!(directory.join("model-q4.gguf").exists());
        let actual = prune_in(root.path(), &path, &[], "candidate", false)?;
        assert_eq!(actual.reclaimed_bytes, 12);
        assert!(!directory.join("model-q4.gguf").exists());
        assert_eq!(fs::read(directory.join("config.json"))?, b"metadata");
        assert_eq!(fs::read(directory.join("source/LICENSE"))?, b"licence");
        assert!(directory.join(MARKER).exists());
        let proof: PrunedWeightsProof =
            serde_json::from_slice(&fs::read(directory.join(".pruned-weights.json"))?)?;
        assert_eq!(proof.removal_state, "completed");
        assert_eq!(proof.removed_weights[0].bytes, 12);
        assert_eq!(
            proof.manifest_sha256,
            crate::artefact::file_hash(&directory.join("gguf-assets.json"))?
        );
        prune_in(root.path(), &path, &[], "candidate", false)?;
        let repeated: PrunedWeightsProof =
            serde_json::from_slice(&fs::read(directory.join(".pruned-weights.json"))?)?;
        assert_eq!(
            repeated.removed_weights.len(),
            1,
            "repeat prune retains authority for already absent weights"
        );
        assert!(directory.join(".prune-history").is_dir());
        Ok(())
    }

    #[test]
    fn traversal_protected_ancestors_and_identity_changes_fail_closed() -> Result<()> {
        let root = tempfile::tempdir()?;
        assert!(
            claim_in(
                root.path(),
                Path::new("../outside"),
                "candidate",
                "organisation/model",
                &"a".repeat(40)
            )
            .is_err()
        );
        assert!(
            claim_in(
                root.path(),
                Path::new("offline_bundle/base"),
                "candidate",
                "organisation/model",
                &"a".repeat(40)
            )
            .is_err()
        );
        assert!(
            claim_in(
                root.path(),
                Path::new("checkpoints/adapter"),
                "candidate",
                "organisation/model",
                &"a".repeat(40)
            )
            .is_err()
        );
        let path = fixture(root.path())?;
        assert!(
            prune_in(
                root.path(),
                &path,
                &[path.join("protected-adapter")],
                "candidate",
                false
            )
            .is_err()
        );
        assert!(prune_in(root.path(), &path, &[], "different-candidate", false).is_err());
        assert!(
            claim_in(
                root.path(),
                &path,
                "candidate",
                "organisation/model",
                &"b".repeat(40)
            )
            .is_err()
        );
        fs::write(root.path().join(&path).join("gguf-assets.json"), br#"{"model_id":"different/model","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#)?;
        assert!(prune_in(root.path(), &path, &[], "candidate", false).is_err());
        Ok(())
    }

    #[test]
    fn stale_untracked_weight_partials_only_are_removed() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = fixture(root.path())?;
        let directory = root.path().join(&path);
        for name in [
            ".old.gguf.partial",
            ".tracked.gguf.partial",
            ".recent.gguf.partial",
            ".tokenizer.json.partial",
        ] {
            fs::write(directory.join(name), b"partial")?;
        }
        let old = SystemTime::now() - Duration::from_secs(25 * 60 * 60);
        for name in [
            ".old.gguf.partial",
            ".tracked.gguf.partial",
            ".tokenizer.json.partial",
        ] {
            File::open(directory.join(name))?.set_modified(old)?;
        }
        fs::write(
            directory.join(".tracked.gguf.partial.json"),
            b"tracked identity",
        )?;
        let report = prune_in(root.path(), &path, &[], "candidate", false)?;
        assert_eq!(report.reclaimed_bytes, 7);
        assert!(!directory.join(".old.gguf.partial").exists());
        for name in [
            ".tracked.gguf.partial",
            ".recent.gguf.partial",
            ".tokenizer.json.partial",
            ".tracked.gguf.partial.json",
        ] {
            assert!(directory.join(name).exists());
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_hardlinks_and_active_downloads_prevent_any_deletion() -> Result<()> {
        let root = tempfile::tempdir()?;
        let path = fixture(root.path())?;
        let directory = root.path().join(&path);
        let outside = root.path().join("outside.gguf");
        fs::write(&outside, b"protected")?;
        std::os::unix::fs::symlink(&outside, directory.join("model.gguf"))?;
        assert!(prune_in(root.path(), &path, &[], "candidate", false).is_err());
        fs::remove_file(directory.join("model.gguf"))?;
        fs::hard_link(&outside, directory.join("model.gguf"))?;
        assert!(prune_in(root.path(), &path, &[], "candidate", false).is_err());
        fs::remove_file(directory.join("model.gguf"))?;
        fs::write(directory.join("model.gguf"), b"owned")?;
        let lease = download_lease(&directory)?;
        assert!(prune_in(root.path(), &path, &[], "candidate", false).is_err());
        drop(lease);
        assert!(directory.join("model.gguf").exists());
        assert_eq!(fs::read(outside)?, b"protected");
        Ok(())
    }

    #[test]
    fn prune_authority_is_measured_and_cannot_hide_present_corruption() -> Result<()> {
        let temporary = tempfile::tempdir_in("target")?;
        let directory = temporary.path();
        let relative = if directory.is_absolute() {
            directory.strip_prefix(std::env::current_dir()?)?.to_owned()
        } else {
            directory.to_owned()
        };
        claim_directory(
            &relative,
            "authority-fixture",
            "organisation/model",
            &"a".repeat(40),
        )?;
        fs::write(directory.join("model.gguf"), b"measured weight")?;
        fs::write(
            directory.join("gguf-assets.json"),
            serde_json::to_vec(
                &serde_json::json!({"model_id":"organisation/model","revision":"a".repeat(40),"files":{"model.gguf":crate::artefact::file_hash(&directory.join("model.gguf"))?}}),
            )?,
        )?;
        prune_candidate(&relative, &[], "authority-fixture", false)?;
        assert!(verify_pruned_directory(
            &relative,
            "authority-fixture",
            "organisation/model",
            &"a".repeat(40)
        )?);
        fs::write(directory.join("model.gguf"), b"modified weight")?;
        assert!(
            verify_pruned_directory(
                &relative,
                "authority-fixture",
                "organisation/model",
                &"a".repeat(40)
            )
            .is_err()
        );
        fs::write(directory.join("model.gguf"), b"measured weight")?;
        assert!(
            !verify_pruned_directory(
                &relative,
                "authority-fixture",
                "organisation/model",
                &"a".repeat(40)
            )?,
            "restored weight cannot be hidden by a forged pruned state"
        );
        Ok(())
    }
}
