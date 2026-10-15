//! Hash source and embedded inputs without including generated model artefacts.
use anyhow::Result;
use lec_distil::artefact;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceInput {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

fn source_files(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            source_files(&entry.path(), paths)?;
        } else if kind.is_file()
            && (entry.path().extension().is_some_and(|e| {
                matches!(e.to_str(), Some("rs" | "toml" | "json" | "lock" | "patch"))
            }) || entry
                .file_name()
                .to_string_lossy()
                .starts_with("Containerfile"))
        {
            paths.push(entry.path());
        }
    }
    Ok(())
}

pub fn capture() -> Result<Vec<SourceInput>> {
    let root = std::env::current_dir()?;
    let mut paths = vec![PathBuf::from("Cargo.toml"), PathBuf::from("Cargo.lock")];
    for directory in [
        "src",
        "examples",
        "tests",
        "config",
        "sandbox/policies",
        "docs/bitnet",
    ] {
        if Path::new(directory).is_dir() {
            source_files(Path::new(directory), &mut paths)?;
        }
    }
    for file in [
        "build.rs",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "docs/comparative-models/provenance/deepseek-coder-1.3b/config.json",
        "docs/comparative-models/provenance/deepseek-coder-1.3b/tokenizer_config.json",
    ] {
        if Path::new(file).is_file() {
            paths.push(file.into());
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let input = artefact::Artefact::capture(&root, path)?;
            Ok(SourceInput {
                path: input.path,
                sha256: input.sha256,
                bytes: input.bytes,
            })
        })
        .collect()
}
