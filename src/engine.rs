//! Content-addressed engine snapshots protect resumable work from later builds.
use crate::artefact;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::fd::AsRawFd,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
pub mod cpu;

/// Kernel-released writer ownership also survives abrupt process termination.
pub struct RunLease(File);
impl RunLease {
    pub fn acquire(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)?;
        // SAFETY: the file descriptor stays open for the complete lease.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another process owns run progress at {}",
            path.display()
        );
        Ok(Self(file))
    }
}
impl Drop for RunLease {
    fn drop(&mut self) {
        // SAFETY: the retained descriptor is valid until after this call.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PinnedEngine {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_build: Option<cpu::CpuBuildReceipt>,
}

impl PinnedEngine {
    pub fn verify(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.path)?;
        ensure!(metadata.is_file(), "engine must be a regular file");
        ensure!(metadata.len() == self.bytes, "engine length changed");
        ensure!(
            metadata.permissions().mode() & 0o222 == 0,
            "pinned engine must have no write bits"
        );
        ensure!(
            metadata.permissions().mode() & 0o111 != 0,
            "pinned engine is not executable"
        );
        ensure!(
            artefact::file_hash(&self.path)? == self.sha256,
            "pinned engine checksum changed"
        );
        if let Some(cpu) = &self.cpu_build {
            cpu.verify(self)?;
        }
        let receipt = self
            .path
            .parent()
            .context("engine parent absent")?
            .join("engine.json");
        if receipt.exists() {
            let recorded: Self = serde_json::from_slice(&fs::read(receipt)?)?;
            ensure!(
                recorded.sha256 == self.sha256
                    && recorded.bytes == self.bytes
                    && recorded.cpu_build == self.cpu_build,
                "pinned engine metadata or CPU requirements changed"
            );
        }
        Ok(())
    }
}

/// Copy through an open descriptor, fsync and publish without replacing an
/// existing snapshot. Cargo may replace its output while another run is active;
/// that output is therefore never the executable used for durable trials.
pub fn pin_executable(source: &Path, root: &Path) -> Result<PinnedEngine> {
    pin_executable_configured(source, root, None)
}

pub fn pin_executable_with_cpu(
    source: &Path,
    root: &Path,
    build: cpu::CpuBuildConfig,
) -> Result<PinnedEngine> {
    build.verify_host()?;
    pin_executable_configured(source, root, Some(build))
}

fn pin_executable_configured(
    source: &Path,
    root: &Path,
    build: Option<cpu::CpuBuildConfig>,
) -> Result<PinnedEngine> {
    fs::create_dir_all(root)?;
    let root = root.canonicalize()?;
    let temporary = root.join(format!(".engine-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<PinnedEngine> {
        let mut input = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(source)
            .with_context(|| format!("open built engine {}", source.display()))?;
        ensure!(
            input.metadata()?.is_file(),
            "built engine is not a regular file"
        );
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        let mut buffer = [0u8; 65536];
        loop {
            let count = input.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            output.write_all(&buffer[..count])?;
        }
        output.sync_all()?;
        let sha256 = artefact::file_hash(&temporary)?;
        let bytes = output.metadata()?.len();
        ensure!(bytes > 0, "built engine is empty");
        ensure!(
            artefact::file_hash(source)? == sha256,
            "build output changed during pinning; retry the launcher"
        );
        output.set_permissions(fs::Permissions::from_mode(0o555))?;
        output.sync_all()?;
        let directory = root.join(&sha256);
        fs::create_dir_all(&directory)?;
        ensure!(
            fs::symlink_metadata(&directory)?.is_dir(),
            "engine snapshot directory cannot be a symbolic link"
        );
        let path = directory.join("lec-distil");
        match fs::hard_link(&temporary, &path) {
            Ok(()) => File::open(&directory)?.sync_all()?,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        let source_receipt = source
            .parent()
            .context("source engine parent absent")?
            .join("engine.json");
        let inherited_cpu = if source_receipt.exists() {
            let mut original: PinnedEngine = serde_json::from_slice(&fs::read(source_receipt)?)?;
            ensure!(
                original.sha256 == sha256 && original.bytes == bytes,
                "source engine metadata belongs to another executable"
            );
            original.path = source.to_owned();
            original.verify()?;
            original.cpu_build
        } else {
            None
        };
        let requested_cpu = build
            .map(|build| cpu::CpuBuildReceipt {
                build,
                executable_sha256: sha256.clone(),
                executable_bytes: bytes,
            })
            .or(inherited_cpu);
        let receipt = directory.join("engine.json");
        let cpu_build = if receipt.exists() {
            let recorded: PinnedEngine = serde_json::from_slice(&fs::read(&receipt)?)?;
            ensure!(
                recorded.sha256 == sha256 && recorded.bytes == bytes,
                "existing engine metadata belongs to another executable"
            );
            if let Some(requested) = &requested_cpu {
                ensure!(
                    recorded.cpu_build.as_ref() == Some(requested),
                    "existing engine has a different frozen CPU build contract"
                );
            }
            recorded.cpu_build
        } else {
            requested_cpu
        };
        let pinned = PinnedEngine {
            path,
            sha256,
            bytes,
            cpu_build,
        };
        pinned.verify()?;
        if !receipt.exists() {
            artefact::write_json(&receipt, &pinned)?;
        }
        File::open(&root)?.sync_all()?;
        Ok(pinned)
    })();
    let _ = fs::remove_file(&temporary);
    result
}

/// Resolve a previous run's exact engine rather than rebuilding it on resume.
pub fn resolve(root: &Path, sha256: &str) -> Result<PinnedEngine> {
    ensure!(
        sha256.len() == 64
            && sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "invalid engine identity"
    );
    let path = root.join(sha256).join("lec-distil").canonicalize()?;
    ensure!(
        path.starts_with(root.canonicalize()?),
        "engine escapes snapshot root"
    );
    verified_snapshot(&path, sha256, fs::metadata(&path)?.len())
}

/// Restore optional build requirements before checking an executable. Callers
/// with older hash-only records still enforce the retained engine metadata.
pub fn verified_snapshot(path: &Path, sha256: &str, bytes: u64) -> Result<PinnedEngine> {
    let receipt = path
        .parent()
        .context("engine parent absent")?
        .join("engine.json");
    let pinned = if receipt.exists() {
        let recorded: PinnedEngine = serde_json::from_slice(&fs::read(&receipt)?)?;
        ensure!(
            recorded.sha256 == sha256 && recorded.bytes == bytes,
            "engine snapshot metadata changed its executable identity"
        );
        PinnedEngine {
            path: path.to_owned(),
            ..recorded
        }
    } else {
        PinnedEngine {
            path: path.to_owned(),
            sha256: sha256.into(),
            bytes,
            cpu_build: None,
        }
    };
    pinned.verify()?;
    Ok(pinned)
}

/// An explicit fresh run receives a sibling directory. Keeping the previous
/// path intact preserves all content-addressed links in its evidence graph.
pub fn prepare_fresh_output(path: &Path) -> Result<PathBuf> {
    if !path.try_exists()? {
        return Ok(path.to_owned());
    }
    ensure!(
        fs::symlink_metadata(path)?.is_dir(),
        "run output must be a real directory"
    );
    let _leases = [
        ".search.lock",
        ".explore.lock",
        ".scheduler.lock",
        ".process.lock",
        ".evaluation.lock",
        ".study-evaluation.lock",
        ".study-evaluation-preparation.lock",
        ".training.lock",
        ".content-collection.lock",
        ".rollout-collection.lock",
        ".worker.lock",
    ]
    .into_iter()
    .map(|name| path.join(name))
    .filter(|lock| lock.exists())
    .map(|lock| RunLease::acquire(&lock))
    .collect::<Result<Vec<_>>>()?;
    let name = path
        .file_name()
        .context("run output needs a directory name")?
        .to_string_lossy();
    let fresh = path.with_file_name(format!("{name}.fresh-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&fresh)?;
    File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?
    .sync_all()?;
    Ok(fresh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_contract_survives_repin_and_corruption_cannot_silently_reset_it() {
        if std::env::consts::ARCH != "x86_64" {
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("build");
        fs::write(
            &source,
            b"labelled CPU receipt fixture; no engine was executed",
        )
        .unwrap();
        let root = directory.path().join("engines");
        let config = cpu::fixture_config(cpu::CpuProfile::Portable);
        let original = pin_executable_with_cpu(&source, &root, config).unwrap();
        let repinned = pin_executable(&original.path, &root).unwrap();
        assert_eq!(repinned.cpu_build, original.cpu_build);
        let copied =
            pin_executable(&original.path, &directory.path().join("another-root")).unwrap();
        assert_eq!(copied.cpu_build, original.cpu_build);
        let mut changed = original.clone();
        changed.cpu_build.as_mut().unwrap().executable_sha256 = "b".repeat(64);
        artefact::write_json(
            &original.path.parent().unwrap().join("engine.json"),
            &changed,
        )
        .unwrap();
        assert!(original.verify().is_err());
        assert!(resolve(&root, &original.sha256).is_err());
    }

    #[test]
    fn absent_cpu_field_preserves_historical_engine_json_shape() {
        let old =
            serde_json::json!({"path":"labelled-fixture-only","sha256":"a".repeat(64),"bytes":10});
        let engine: PinnedEngine = serde_json::from_value(old.clone()).unwrap();
        assert!(engine.cpu_build.is_none());
        assert_eq!(serde_json::to_value(engine).unwrap(), old);
    }

    #[test]
    fn later_builds_cannot_change_a_pinned_run_and_corruption_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("build");
        fs::write(&source, b"original engine").unwrap();
        let root = directory.path().join("engines");
        let original = pin_executable(&source, &root).unwrap();
        fs::write(&source, b"later engine").unwrap();
        let later = pin_executable(&source, &root).unwrap();
        assert_ne!(original.sha256, later.sha256);
        assert_eq!(fs::read(&original.path).unwrap(), b"original engine");
        resolve(&root, &original.sha256).unwrap();
        fs::set_permissions(&original.path, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(&original.path, b"corrupt").unwrap();
        assert!(resolve(&root, &original.sha256).is_err());
        assert!(pin_executable(&source, &root).is_ok());
    }

    #[test]
    fn fresh_preserves_original_paths_and_starts_in_a_separate_empty_directory() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("round");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("checkpoint"), b"saved step").unwrap();
        let fresh = prepare_fresh_output(&output).unwrap();
        assert_ne!(fresh, output);
        assert_eq!(fs::read_dir(&fresh).unwrap().count(), 0);
        assert_eq!(fs::read(output.join("checkpoint")).unwrap(), b"saved step");
        let new = directory.path().join("new-output");
        assert_eq!(prepare_fresh_output(&new).unwrap(), new);
    }

    #[test]
    fn explicit_fresh_rejects_an_active_run() {
        let directory = tempfile::tempdir().unwrap();
        let lock = RunLease::acquire(&directory.path().join(".evaluation.lock")).unwrap();
        assert!(prepare_fresh_output(directory.path()).is_err());
        drop(lock);
    }
}
