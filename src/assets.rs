//! Non-destructive inventory of pinned assets. No acquisition or pruning.
use crate::{artefact, model};
use anyhow::{Result, ensure};
use serde_json::{Value, json};
use std::{fs, path::Path};

pub fn inspect_file(path: &Path, expected: &str) -> Result<Value> {
    let state = match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => "missing",
        Err(error) => return Err(error.into()),
        Ok(metadata) if metadata.file_type().is_dir() => "directory_conflict",
        Ok(metadata) if !metadata.file_type().is_file() => "non_regular_conflict",
        Ok(_) if artefact::file_hash(path)? == expected => "valid_regular_file",
        Ok(_) => "corrupt_regular_file",
    };
    Ok(json!({"path":path,"state":state,"expected_sha256":expected}))
}

pub fn inspect_base(base: &Path, quarantine: bool) -> Result<Value> {
    let mut files = Vec::new();
    for (name, expected) in model::PINNED_FILES {
        let path = base.join(name);
        let mut row = inspect_file(&path, expected)?;
        if quarantine
            && matches!(
                row["state"].as_str(),
                Some("directory_conflict" | "non_regular_conflict" | "corrupt_regular_file")
            )
        {
            let destination = base.join(format!("{name}.quarantine-{}", uuid::Uuid::new_v4()));
            ensure!(
                !destination.exists(),
                "quarantine destination already exists"
            );
            fs::rename(&path, &destination)?;
            row["quarantined_to"] = json!(destination);
        }
        files.push(row);
    }
    Ok(
        json!({"schema_version":1,"files":files,"network_used":false,"downloads_performed":0,"quarantine_explicitly_requested":quarantine,"note":"Quarantine renames conflicts without deleting their contents. Bootstrap-online can restore absent pinned payloads; valid manifests and payloads are preserved."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_states_preserve_directory_conflicts_and_corrupt_payloads() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("payload");
        let hash = artefact::hash(b"verified");
        assert_eq!(inspect_file(&path, &hash)?["state"], "missing");
        let dangling = tmp.path().join("dangling");
        std::os::unix::fs::symlink(tmp.path().join("absent-target"), &dangling)?;
        assert_eq!(
            inspect_file(&dangling, &hash)?["state"],
            "non_regular_conflict"
        );
        assert!(fs::symlink_metadata(&dangling)?.file_type().is_symlink());
        fs::create_dir(&path)?;
        fs::write(path.join("ownership"), b"preserve")?;
        assert_eq!(inspect_file(&path, &hash)?["state"], "directory_conflict");
        assert_eq!(fs::read(path.join("ownership"))?, b"preserve");
        let file = tmp.path().join("file");
        fs::write(&file, b"bad")?;
        assert_eq!(inspect_file(&file, &hash)?["state"], "corrupt_regular_file");
        fs::write(&file, b"verified")?;
        assert_eq!(inspect_file(&file, &hash)?["state"], "valid_regular_file");
        Ok(())
    }
}
