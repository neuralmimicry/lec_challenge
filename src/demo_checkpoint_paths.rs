//! Separate reproducible training work from immutable bundled replay evidence.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
};

const LABELS: [(&str, &str); 4] = [
    ("candidate", "candidate"),
    ("seed-control", "seed-control"),
    ("degraded-control", "degraded"),
    ("overfit-control", "overfit"),
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct TrainingInputs {
    pub dataset_manifest_sha256: String,
    pub heldout_index_sha256: String,
    pub base_sha256: String,
    pub tokenizer_sha256: String,
    pub training_configuration_hashes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Allocation {
    schema_version: u32,
    input_sha256: String,
    run_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Published {
    schema_version: u32,
    input_sha256: String,
    paths: BTreeMap<String, PathBuf>,
    tensor_sha256: BTreeMap<String, String>,
}

pub(super) struct Selection {
    pub paths: BTreeMap<String, PathBuf>,
    /// Keeping this file open holds the lease across training and evaluation.
    _lease: Option<File>,
    input_sha256: String,
    recorded_tensor_sha256: Option<BTreeMap<String, String>>,
}

/// Every component must stay within the project tree. Existing
/// symbolic links are refused even if they currently resolve inside the tree.
fn local_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    let components = relative.components().collect::<Vec<_>>();
    ensure!(
        !components.is_empty() && components.iter().all(|c| matches!(c, Component::Normal(_))),
        "checkpoint bookkeeping path must be relative without traversal"
    );
    let mut path = root.to_owned();
    for component in components {
        path.push(component.as_os_str());
        match fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "checkpoint bookkeeping refuses symbolic links: {}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(path)
}

fn allocation_root(allocation: &Allocation) -> Result<PathBuf> {
    ensure!(
        allocation.schema_version == 1,
        "unsupported training allocation"
    );
    let id = uuid::Uuid::parse_str(&allocation.run_id)?;
    ensure!(
        id.to_string() == allocation.run_id && id.get_version_num() == 4,
        "invalid training execution identifier"
    );
    Ok(PathBuf::from("checkpoints/retrained").join(&allocation.run_id))
}

/// An empty application state always allocates fresh optimisation outputs,
/// regardless of which completed adapters the delivery already contains.
pub(super) fn select(root: &Path, fast: bool, inputs: &TrainingInputs) -> Result<Selection> {
    ensure!(
        inputs
            .training_configuration_hashes
            .keys()
            .map(String::as_str)
            .eq([
                "candidate",
                "degraded-control",
                "overfit-control",
                "seed-control"
            ]),
        "checkpoint selection requires all four predeclared configurations"
    );
    let input_sha256 = artefact::hash(&serde_json::to_vec(inputs)?);
    if fast {
        let published = local_path(root, Path::new("checkpoints/demo-published.json"))?;
        let (paths, recorded_tensor_sha256) = if published.exists() {
            let saved: Published = serde_json::from_slice(&fs::read(published)?)?;
            ensure!(
                saved.schema_version == 1 && saved.input_sha256 == input_sha256,
                "published replay checkpoints differ from the original training inputs"
            );
            ensure!(
                saved
                    .paths
                    .keys()
                    .eq(inputs.training_configuration_hashes.keys())
                    && saved.tensor_sha256.keys().eq(saved.paths.keys()),
                "published replay checkpoint coverage is incomplete"
            );
            for path in saved.paths.values() {
                ensure!(
                    path.starts_with("checkpoints"),
                    "published checkpoint is outside the included evidence tree"
                );
                local_path(root, &path.join("manifest.json"))?;
            }
            (saved.paths, Some(saved.tensor_sha256))
        } else {
            (
                LABELS
                    .iter()
                    .map(|(label, directory)| {
                        (
                            (*label).to_owned(),
                            PathBuf::from("checkpoints").join(directory),
                        )
                    })
                    .collect(),
                None,
            )
        };
        for path in paths.values() {
            local_path(root, &path.join("manifest.json"))?;
        }
        return Ok(Selection {
            paths,
            _lease: None,
            input_sha256,
            recorded_tensor_sha256,
        });
    }
    let state = local_path(root, Path::new("state/demo-training"))?;
    fs::create_dir_all(&state)?;
    let lock_path = local_path(root, Path::new("state/demo-training/lease.lock"))?;
    let lease = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    // The kernel releases this advisory lease when a killed orchestrator exits.
    ensure!(
        unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another full demo owns this training state"
    );
    let pointer = state.join(format!("{input_sha256}.json"));
    local_path(
        root,
        &PathBuf::from("state/demo-training").join(format!("{input_sha256}.json")),
    )?;
    let allocation: Allocation = if pointer.exists() {
        serde_json::from_slice(&fs::read(&pointer)?)?
    } else {
        let allocation = Allocation {
            schema_version: 1,
            input_sha256: input_sha256.clone(),
            run_id: uuid::Uuid::new_v4().to_string(),
        };
        let relative = allocation_root(&allocation)?;
        let parent = local_path(root, Path::new("checkpoints/retrained"))?;
        fs::create_dir_all(parent)?;
        // create_dir, rather than create_dir_all, forbids adopting old outputs.
        let directory = local_path(root, &relative)?;
        fs::create_dir(&directory)?;
        artefact::write_json(&directory.join("allocation.json"), &allocation)?;
        artefact::write_json(&pointer, &allocation)?;
        allocation
    };
    ensure!(
        allocation.input_sha256 == input_sha256,
        "training state input changed"
    );
    let relative = allocation_root(&allocation)?;
    let directory = local_path(root, &relative)?;
    let marker = local_path(root, &relative.join("allocation.json"))?;
    let owned: Allocation = serde_json::from_slice(
        &fs::read(&marker).context("training allocation ownership marker is absent")?,
    )?;
    ensure!(
        owned == allocation && directory.is_dir(),
        "training state does not own these checkpoint outputs"
    );
    let paths = LABELS
        .iter()
        .map(|(label, _)| ((*label).to_owned(), relative.join(label)))
        .collect::<BTreeMap<_, _>>();
    for path in paths.values() {
        local_path(root, &path.join("manifest.json"))?;
        local_path(root, &path.join("progress.json"))?;
    }
    Ok(Selection {
        paths,
        _lease: Some(lease),
        input_sha256,
        recorded_tensor_sha256: None,
    })
}

impl Selection {
    pub fn verify_recorded_tensors(&self, tensors: &BTreeMap<String, String>) -> Result<()> {
        if let Some(recorded) = &self.recorded_tensor_sha256 {
            ensure!(
                recorded == tensors,
                "published replay tensor identity changed"
            );
        }
        Ok(())
    }

    /// Seal once, after checkpoint validation and the immutable model freeze.
    /// A repeated deterministic full run keeps the original replay allocation.
    pub fn publish(&self, root: &Path, tensors: &BTreeMap<String, String>) -> Result<()> {
        ensure!(
            tensors.keys().eq(self.paths.keys()),
            "checkpoint tensor coverage is incomplete"
        );
        self.verify_recorded_tensors(tensors)?;
        let path = local_path(root, Path::new("checkpoints/demo-published.json"))?;
        if path.exists() {
            let existing: Published = serde_json::from_slice(&fs::read(&path)?)?;
            ensure!(
                existing.schema_version == 1
                    && existing.input_sha256 == self.input_sha256
                    && existing.tensor_sha256 == *tensors,
                "published checkpoint evidence is immutable; changed training requires fresh independent sources"
            );
        } else if self._lease.is_some() {
            artefact::write_json(
                &path,
                &Published {
                    schema_version: 1,
                    input_sha256: self.input_sha256.clone(),
                    paths: self.paths.clone(),
                    tensor_sha256: tensors.clone(),
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn inputs() -> TrainingInputs {
        TrainingInputs {
            dataset_manifest_sha256: artefact::hash(b"original dataset"),
            heldout_index_sha256: artefact::hash(b"original sealed index"),
            base_sha256: artefact::hash(b"base"),
            tokenizer_sha256: artefact::hash(b"tokenizer"),
            training_configuration_hashes: LABELS
                .iter()
                .map(|(label, _)| ((*label).into(), artefact::hash(label.as_bytes())))
                .collect(),
        }
    }
    fn tensors() -> BTreeMap<String, String> {
        LABELS
            .iter()
            .map(|(label, _)| ((*label).into(), artefact::hash(label.as_bytes())))
            .collect()
    }

    #[test]
    fn bundled_manifests_never_shortcut_fresh_full_training() {
        let root = tempfile::tempdir().unwrap();
        for (_, directory) in LABELS {
            let path = root.path().join("checkpoints").join(directory);
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("manifest.json"), b"bundled evidence").unwrap();
        }
        let full = select(root.path(), false, &inputs()).unwrap();
        let fast = select(root.path(), true, &inputs()).unwrap();
        for (label, directory) in LABELS {
            assert!(full.paths[label].starts_with("checkpoints/retrained"));
            assert!(
                !root
                    .path()
                    .join(&full.paths[label])
                    .join("manifest.json")
                    .exists()
            );
            assert_eq!(
                fast.paths[label],
                PathBuf::from("checkpoints").join(directory)
            );
        }
        assert!(
            select(root.path(), false, &inputs()).is_err(),
            "concurrent full runs must not share optimiser state"
        );
    }

    #[test]
    fn restart_reuses_only_its_own_allocation_and_cleared_state_trains_again() {
        let root = tempfile::tempdir().unwrap();
        let full = select(root.path(), false, &inputs()).unwrap();
        let paths = full.paths.clone();
        let completed = root.path().join(&paths["candidate"]);
        fs::create_dir_all(&completed).unwrap();
        fs::write(completed.join("manifest.json"), b"locally completed work").unwrap();
        drop(full);
        let restarted = select(root.path(), false, &inputs()).unwrap();
        assert_eq!(restarted.paths, paths);
        assert_eq!(
            fs::read(completed.join("manifest.json")).unwrap(),
            b"locally completed work"
        );
        drop(restarted);
        fs::remove_dir_all(root.path().join("state")).unwrap();
        assert_ne!(select(root.path(), false, &inputs()).unwrap().paths, paths);
    }

    #[test]
    fn dataset_or_configuration_change_gets_separate_optimiser_state() {
        let root = tempfile::tempdir().unwrap();
        let first = select(root.path(), false, &inputs()).unwrap().paths;
        let mut changed = inputs();
        changed.dataset_manifest_sha256 = artefact::hash(b"different dataset");
        let second = select(root.path(), false, &changed).unwrap().paths;
        changed = inputs();
        changed
            .training_configuration_hashes
            .insert("candidate".into(), artefact::hash(b"different training"));
        let third = select(root.path(), false, &changed).unwrap().paths;
        assert_ne!(first, second);
        assert_ne!(first, third);
        assert_ne!(second, third);
    }

    #[test]
    fn published_replay_uses_included_paths_and_blocks_tensor_changes() {
        let root = tempfile::tempdir().unwrap();
        let full = select(root.path(), false, &inputs()).unwrap();
        full.publish(root.path(), &tensors()).unwrap();
        let replay = select(root.path(), true, &inputs()).unwrap();
        assert_eq!(replay.paths, full.paths);
        replay.verify_recorded_tensors(&tensors()).unwrap();
        let mut changed = tensors();
        changed.insert("candidate".into(), artefact::hash(b"different tensor"));
        assert!(replay.verify_recorded_tensors(&changed).is_err());
        assert!(full.publish(root.path(), &changed).is_err());
        let mut different_inputs = inputs();
        different_inputs.dataset_manifest_sha256 = artefact::hash(b"different dataset");
        assert!(select(root.path(), true, &different_inputs).is_err());
    }

    #[test]
    fn ownership_marker_and_symlinked_outputs_fail_closed() {
        let root = tempfile::tempdir().unwrap();
        let full = select(root.path(), false, &inputs()).unwrap();
        let allocation = root.path().join(full.paths["candidate"].parent().unwrap());
        drop(full);
        fs::remove_file(allocation.join("allocation.json")).unwrap();
        assert!(select(root.path(), false, &inputs()).is_err());
        let other = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(
            other.path(),
            root.path().join("checkpoints/demo-published.json"),
        )
        .unwrap();
        assert!(select(root.path(), true, &inputs()).is_err());
    }
}
