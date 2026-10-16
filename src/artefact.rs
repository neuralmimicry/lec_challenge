//! Durable, checksum-addressed artifacts shared across trust boundaries.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

/// Paths are relative to an evidence directory; traversal and symlinks are rejected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artefact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

pub fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub fn file_hash(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(hex::encode(digest.finalize()))
}

impl Artefact {
    pub fn capture(root: &Path, relative: impl Into<PathBuf>) -> Result<Self> {
        let path = relative.into();
        let full = safe_path(root, &path)?;
        Ok(Self {
            path,
            sha256: file_hash(&full)?,
            bytes: fs::metadata(full)?.len(),
        })
    }
    pub fn verify(&self, root: &Path) -> Result<PathBuf> {
        ensure!(self.sha256.len() == 64, "invalid SHA-256");
        let full = safe_path(root, &self.path)?;
        ensure!(
            fs::metadata(&full)?.len() == self.bytes,
            "artifact size mismatch: {}",
            self.path.display()
        );
        ensure!(
            file_hash(&full)? == self.sha256,
            "artifact checksum mismatch: {}",
            self.path.display()
        );
        Ok(full)
    }
}

fn safe_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    use std::path::Component;
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "artifact path must be relative without traversal"
    );
    let canonical_root = root.canonicalize()?;
    let full = root.join(relative);
    let resolved = full.canonicalize()?;
    ensure!(
        resolved.starts_with(canonical_root),
        "artifact escapes evidence root"
    );
    Ok(resolved)
}

/// fsync the payload and directory before exposing its name as durable evidence.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(tmp);
    }
    result
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checksum_and_traversal_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("run.json");
        atomic_write(&p, b"real").unwrap();
        let a = Artefact::capture(dir.path(), "run.json").unwrap();
        a.verify(dir.path()).unwrap();
        atomic_write(&p, b"fake").unwrap();
        assert!(a.verify(dir.path()).is_err());
        assert!(Artefact::capture(dir.path(), "../outside").is_err());
    }

    #[test]
    fn persisted_task_rates_retain_exact_bits_for_independent_reconstruction() {
        // The default JSON parser rounded the actual 11/48 score down by one
        // ULP. A strict verifier must keep exact equality and fix parsing,
        // rather than accepting an arbitrary numerical tolerance for reports.
        for denominator in 1..=192 {
            for numerator in 0..=denominator {
                let expected = numerator as f64 / denominator as f64;
                let bytes = serde_json::to_vec(&serde_json::json!({
                    "passed": numerator,
                    "planned": denominator,
                    "pass_rate": expected,
                }))
                .unwrap();
                let read: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(
                    expected.to_bits(),
                    read["pass_rate"].as_f64().unwrap().to_bits(),
                    "persisted rate changed for {numerator}/{denominator}"
                );
                assert_eq!(serde_json::to_vec(&read).unwrap(), bytes);
            }
        }
    }
}
