//! Select native inference artefacts before acquiring or changing any assets.
//! A frozen evaluation or numerical report is authoritative; the shared export
//! cache is used only for new evaluations without an explicit recorded tuple.
use crate::{
    artefact::{self, Artefact},
    study_evaluate::parity::{NumericalParityPlan, NumericalParityReport},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BitNetBackendOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_manifest: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gguf: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parity_report: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct PreparedBitNetBackend {
    pub engine_manifest: PathBuf,
    pub gguf: PathBuf,
    pub backend_artefacts: Vec<Artefact>,
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(root, path.canonicalize()?.strip_prefix(root)?)
}
fn same(left: &Artefact, right: &Artefact) -> Result<bool> {
    Ok(serde_json::to_value(left)? == serde_json::to_value(right)?)
}
fn read_json<T: serde::de::DeserializeOwned>(root: &Path, file: &Artefact) -> Result<T> {
    ensure!(
        file.bytes <= 32 << 20,
        "native backend JSON receipt exceeds bound"
    );
    Ok(serde_json::from_slice(&fs::read(file.verify(root)?)?)?)
}
fn no_overrides(selection: &BitNetBackendOptions) -> bool {
    selection.engine_manifest.is_none()
        && selection.gguf.is_none()
        && selection.parity_report.is_none()
}
fn check_override(
    root: &Path,
    requested: Option<&Path>,
    expected: &Artefact,
    label: &str,
) -> Result<()> {
    if let Some(path) = requested {
        ensure!(
            same(&capture(root, path)?, expected)?,
            "explicit {label} conflicts with the frozen native backend"
        );
    }
    Ok(())
}

/// Independently verify raw numerical observations, then recover their exact
/// native tuple. A completed failed comparison is useful descriptive evidence.
fn from_parity(
    root: &Path,
    report_path: &Path,
) -> Result<(PreparedBitNetBackend, NumericalParityPlan)> {
    crate::study_evaluate::parity::verify_bitnet_parity_report(report_path)?;
    let report: NumericalParityReport = read_json(root, &capture(root, report_path)?)?;
    let plan: NumericalParityPlan = read_json(root, &report.plan)?;
    let prepared = PreparedBitNetBackend {
        engine_manifest: plan.cpp_engine_manifest.path.clone(),
        gguf: plan.gguf.path.clone(),
        backend_artefacts: vec![
            plan.cpp_engine_manifest.clone(),
            plan.gguf.clone(),
            plan.export_receipt.clone(),
        ],
    };
    Ok((prepared, plan))
}

/// Resolve only recorded files. This read-only path deliberately has no native
/// host/ISA check: historical evidence can be inspected on another platform.
/// `None` retains a historical Rust-only plan and never bootstraps a substitute.
pub fn resolve_frozen_backend(
    frozen_artefacts: &[Artefact],
    selection: &BitNetBackendOptions,
) -> Result<Option<PreparedBitNetBackend>> {
    let root = std::env::current_dir()?.canonicalize()?;
    resolve_frozen(&root, frozen_artefacts, selection)
}

fn resolve_frozen(
    root: &Path,
    files: &[Artefact],
    selection: &BitNetBackendOptions,
) -> Result<Option<PreparedBitNetBackend>> {
    // Each large file is hashed once during this resolution. This map exists
    // only for the call; later execution and independent readers verify again.
    let mut verified = BTreeMap::new();
    for file in files {
        ensure!(
            !verified.contains_key(&file.path),
            "duplicate frozen native backend artefact"
        );
        verified.insert(file.path.clone(), (file, file.verify(root)?));
    }
    let mut engine = None;
    let mut export = None;
    for file in files {
        let path = &verified[&file.path].1;
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        ensure!(
            file.bytes <= 32 << 20,
            "native backend JSON receipt exceeds bound"
        );
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        if value.get("engine_commit").is_some() && value.get("llama_commit").is_some() {
            ensure!(engine.is_none(), "multiple frozen native inference engines");
            crate::bitnet::cpp::verify_engine(path)?;
            engine = Some(file.clone());
        }
        if value["evidence_kind"] == "native_official_bitnet_qat_gguf_export" {
            ensure!(
                export.is_none(),
                "multiple frozen native GGUF export receipts"
            );
            ensure!(
                value["schema_version"] == 1
                    && value["production_eligible"] == false
                    && value["binding"]["codec"] == "native_hf_axis_to_i2s_v1"
                    && value["binding"]["implementation_sha256"]
                        .as_str()
                        .is_some_and(|hash| hash.len() == 64
                            && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
                    && value["tensors"].as_u64().is_some_and(|count| count > 0),
                "frozen GGUF export receipt has unsupported inputs"
            );
            let gguf: Artefact = serde_json::from_value(value["gguf"].clone())?;
            ensure!(
                verified
                    .get(&gguf.path)
                    .is_some_and(|(file, _)| same(file, &gguf).unwrap_or(false)),
                "frozen GGUF is absent from backend artefacts"
            );
            export = Some((gguf, file.clone()));
        }
    }
    ensure!(
        engine.is_some() == export.is_some(),
        "frozen native engine and GGUF export must be recorded together"
    );
    let Some(engine) = engine else {
        ensure!(
            no_overrides(selection),
            "a historical Rust-only evaluation cannot accept native backend overrides; use a fresh evaluation output"
        );
        return Ok(None);
    };
    let (gguf, receipt) = export.context("frozen native GGUF export absent")?;
    for (requested, expected, label) in [
        (&selection.engine_manifest, &engine, "engine manifest"),
        (&selection.gguf, &gguf, "GGUF"),
    ] {
        if let Some(path) = requested {
            ensure!(
                path.canonicalize()? == verified[&expected.path].1,
                "explicit {label} conflicts with the frozen native backend"
            );
        }
    }
    if let Some(report) = &selection.parity_report {
        let (_, plan) = from_parity(root, report)?;
        ensure!(
            same(&engine, &plan.cpp_engine_manifest)?
                && same(&gguf, &plan.gguf)?
                && same(&receipt, &plan.export_receipt)?,
            "numerical report conflicts with the frozen native backend tuple"
        );
    }
    Ok(Some(PreparedBitNetBackend {
        engine_manifest: engine.path.clone(),
        gguf: gguf.path.clone(),
        backend_artefacts: files.to_vec(),
    }))
}

/// Existing explicit outputs need their exact original source/adapter binding
/// before a missing default engine is allowed to change the build cache. A new
/// output is allowed; a partially assembled one needs the exporter's durable
/// binding and is recovered by that exporter rather than overwritten here.
fn preflight_explicit_export(
    root: &Path,
    gguf: &Path,
    checkpoint: &crate::study_evaluate::Checkpoint,
) -> Result<()> {
    // Match the exporter's writable-path contract before bootstrap can touch
    // state. Historical read-only tuples do not enter this new-output path.
    for path in [
        gguf.to_owned(),
        gguf.with_extension("export.json"),
        gguf.with_extension("gguf.parts").join("binding.json"),
        gguf.with_extension("gguf.export.lock"),
    ] {
        ensure!(
            !path.as_os_str().is_empty()
                && path
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
            "explicit export paths must remain relative to the workspace"
        );
        let mut current = root.to_owned();
        for component in path.components() {
            current.push(component);
            match fs::symlink_metadata(&current) {
                Ok(metadata) => {
                    ensure!(
                        !metadata.file_type().is_symlink()
                            && (metadata.is_file() || metadata.is_dir()),
                        "unsafe explicit export path"
                    );
                    #[cfg(unix)]
                    if metadata.is_file() {
                        use std::os::unix::fs::MetadataExt;
                        ensure!(metadata.nlink() == 1, "hard-linked explicit export file");
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    ensure!(
        gguf.extension()
            .is_some_and(|extension| extension == "gguf"),
        "BitNet export requires a GGUF destination"
    );
    let gguf = root.join(gguf);
    let receipt_path = gguf.with_extension("export.json");
    let binding_path = gguf.with_extension("gguf.parts").join("binding.json");
    let receipt = if receipt_path.exists() {
        ensure!(
            fs::metadata(&receipt_path)?.len() <= 32 << 20,
            "explicit GGUF receipt exceeds bound"
        );
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&receipt_path)?)?;
        ensure!(
            value["schema_version"] == 1
                && value["evidence_kind"] == "native_official_bitnet_qat_gguf_export"
                && value["production_eligible"] == false
                && value["tensors"].as_u64().is_some_and(|count| count > 0),
            "explicit GGUF receipt has unsupported inputs"
        );
        let file: Artefact = serde_json::from_value(value["gguf"].clone())?;
        ensure!(
            file.verify(root)? == gguf.canonicalize()?,
            "explicit GGUF receipt names another destination"
        );
        Some(value)
    } else {
        None
    };
    ensure!(
        !gguf.exists() || receipt.is_some() || binding_path.exists(),
        "existing explicit GGUF has neither a receipt nor resumable export authority"
    );
    let binding = if let Some(receipt) = &receipt {
        Some(receipt["binding"].clone())
    } else if binding_path.exists() {
        ensure!(
            fs::metadata(&binding_path)?.len() <= 32 << 20,
            "explicit GGUF fragment binding exceeds bound"
        );
        Some(serde_json::from_slice(&fs::read(binding_path)?)?)
    } else {
        None
    };
    if let Some(binding) = binding {
        let base: serde_json::Value =
            serde_json::from_slice(&fs::read(checkpoint.base_assets.verify(root)?)?)?;
        ensure!(
            binding["source"] == base
                && binding["codec"] == "native_hf_axis_to_i2s_v1"
                && binding["implementation_sha256"]
                    .as_str()
                    .is_some_and(|hash| hash.len() == 64
                        && hash.bytes().all(|byte| byte.is_ascii_hexdigit())),
            "explicit GGUF belongs to another base or exporter protocol"
        );
        if same(&checkpoint.manifest, &checkpoint.base_assets)? {
            ensure!(
                binding["adapter"].is_null(),
                "explicit baseline GGUF silently includes an adapter"
            );
        } else {
            ensure!(
                binding["adapter"] == serde_json::to_value(&checkpoint.tensor)?,
                "explicit GGUF belongs to another checkpoint"
            );
            if let Some(receipt) = receipt {
                ensure!(
                    receipt["changed_effective_trits"]
                        .as_u64()
                        .is_some_and(|count| count > 0),
                    "explicit adapter GGUF has no changed effective trits"
                );
            }
        }
    }
    Ok(())
}

pub(super) fn cache_path(tensor_sha256: &str, exporter_sha256: &str) -> Result<PathBuf> {
    for hash in [tensor_sha256, exporter_sha256] {
        ensure!(
            hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid BitNet cache content identity"
        );
    }
    Ok(Path::new("state/bitnet-exports")
        .join(tensor_sha256)
        .join(exporter_sha256)
        .join("model.i2_s.gguf"))
}

/// Prepare a new native backend off the asynchronous executor. Validate all
/// checkpoint and override identities before bootstrap/export mutations. The
/// exporter's exclusive ownership lock and resumable fragments govern cache
/// concurrency; evaluations merely refer to the resulting immutable receipt.
pub async fn prepare_backend(
    base: &Path,
    checkpoint: Option<&Path>,
    evaluation_output: &Path,
    selection: &BitNetBackendOptions,
    offline: bool,
    jobs: usize,
) -> Result<PreparedBitNetBackend> {
    let output = evaluation_output.to_owned();
    let validation_base = base.to_owned();
    let validation_checkpoint = checkpoint.map(Path::to_owned);
    let validation_selection = selection.clone();
    let (identity, prior) = tokio::task::spawn_blocking(move || -> Result<_> {
        let base = validation_base;
        let checkpoint = validation_checkpoint;
        let selection = validation_selection;
        let root = std::env::current_dir()?.canonicalize()?;
        let identity = super::parity_run::checkpoint_identity(&base, checkpoint.as_deref(), &root)?;
        let saved = output.join("plan.json");
        if saved.exists() {
            let plan: serde_json::Value = serde_json::from_slice(&fs::read(saved)?)?;
            ensure!(
                plan["checkpoint"] == serde_json::to_value(&identity)?,
                "saved native evaluation belongs to another checkpoint/base/tokeniser"
            );
            let files: Vec<Artefact> = serde_json::from_value(
                plan.get("backend_artefacts")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!([])),
            )?;
            let prepared = resolve_frozen(&root, &files, &selection)?
                .context("saved evaluation uses historical Rust inference; retain that runtime")?;
            return Ok((identity, Some(prepared)));
        }
        if let Some(report) = &selection.parity_report {
            let (prepared, plan) = from_parity(&root, report)?;
            ensure!(
                serde_json::to_value(&identity)? == serde_json::to_value(&plan.checkpoint)?,
                "numerical report belongs to another checkpoint/base/tokeniser"
            );
            check_override(
                &root,
                selection.engine_manifest.as_deref(),
                &plan.cpp_engine_manifest,
                "engine manifest",
            )?;
            check_override(&root, selection.gguf.as_deref(), &plan.gguf, "GGUF")?;
            return Ok((identity, Some(prepared)));
        }
        // Inspect a requested engine before any expensive build or export.
        if let Some(engine) = &selection.engine_manifest {
            super::cpp::verify_engine(engine)?;
        }
        if let Some(gguf) = &selection.gguf {
            preflight_explicit_export(&root, gguf, &identity)?;
        }
        Ok((identity, None))
    })
    .await??;
    if let Some(prepared) = prior {
        return Ok(prepared);
    }
    // Clone the inputs again only after the immutable resolution has succeeded.
    let engine_manifest = if let Some(path) = &selection.engine_manifest {
        path.clone()
    } else {
        super::cpp::bootstrap_engine(offline, jobs).await?;
        PathBuf::from("state/engines/bitnet-build/engine-manifest.json")
    };
    let gguf = selection.gguf.clone().map(Ok).unwrap_or_else(|| {
        cache_path(
            &identity.tensor.sha256,
            &artefact::hash(include_bytes!("export.rs")),
        )
    })?;
    let base = base.to_owned();
    let checkpoint = checkpoint.map(Path::to_owned);
    tokio::task::spawn_blocking(move || -> Result<_> {
        let root = std::env::current_dir()?.canonicalize()?;
        super::cpp::verify_engine(&engine_manifest)?;
        let exported = super::export::export(&base, checkpoint.as_deref(), &gguf)?;
        let exported_file: Artefact = serde_json::from_value(exported["gguf"].clone())?;
        let engine_file = capture(&root, &engine_manifest)?;
        let export_file = capture(&root, &gguf.with_extension("export.json"))?;
        Ok(PreparedBitNetBackend {
            engine_manifest: engine_file.path.clone(),
            gguf: exported_file.path.clone(),
            backend_artefacts: vec![engine_file, exported_file, export_file],
        })
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn historical_rust_plan_rejects_every_native_override_without_mutation() {
        let root = tempfile::tempdir().unwrap();
        assert!(
            resolve_frozen(root.path(), &[], &Default::default())
                .unwrap()
                .is_none()
        );
        for selection in [
            BitNetBackendOptions {
                engine_manifest: Some("missing-engine.json".into()),
                ..Default::default()
            },
            BitNetBackendOptions {
                gguf: Some("missing.gguf".into()),
                ..Default::default()
            },
            BitNetBackendOptions {
                parity_report: Some("missing-parity.json".into()),
                ..Default::default()
            },
        ] {
            assert!(resolve_frozen(root.path(), &[], &selection).is_err());
        }
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    }
    #[test]
    fn shared_cache_separates_changed_tensors_and_exporters_and_rejects_paths() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        assert_ne!(cache_path(&a, &a).unwrap(), cache_path(&b, &a).unwrap());
        assert_ne!(cache_path(&a, &a).unwrap(), cache_path(&a, &b).unwrap());
        assert_eq!(cache_path(&a, &a).unwrap(), cache_path(&a, &a).unwrap());
        assert!(cache_path("../../escape", &a).is_err());
    }
    #[test]
    fn frozen_receipts_reject_duplicates_and_corruption() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("unrelated.json"), b"{}").unwrap();
        let file = Artefact::capture(root.path(), "unrelated.json").unwrap();
        assert!(
            resolve_frozen(
                root.path(),
                &[file.clone(), file.clone()],
                &Default::default()
            )
            .is_err()
        );
        fs::write(root.path().join("unrelated.json"), b"[]").unwrap();
        assert!(resolve_frozen(root.path(), &[file], &Default::default()).is_err());
    }
    #[test]
    fn explicit_export_preflight_rejects_unclaimed_and_wrong_checkpoint_bytes() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("assets.json"), b"{}").unwrap();
        fs::write(root.path().join("tensor.bin"), b"original tensor").unwrap();
        let base = Artefact::capture(root.path(), "assets.json").unwrap();
        let tensor = Artefact::capture(root.path(), "tensor.bin").unwrap();
        let checkpoint = crate::study_evaluate::Checkpoint {
            checkpoint_id: tensor.sha256.clone(),
            tensor: tensor.clone(),
            manifest: base.clone(),
            base_assets: base,
            tokenizer: tensor,
            architecture: crate::experiments::StudentArchitecture::TernaryBitnet,
        };
        let output = Path::new("new/model.gguf");
        preflight_explicit_export(root.path(), output, &checkpoint).unwrap();
        assert!(!root.path().join("new").exists());
        fs::create_dir(root.path().join("new")).unwrap();
        fs::write(root.path().join(output), b"actual test bytes").unwrap();
        assert!(preflight_explicit_export(root.path(), output, &checkpoint).is_err());
        let mut receipt = serde_json::json!({"schema_version":1,"evidence_kind":"native_official_bitnet_qat_gguf_export",
            "production_eligible":false,"tensors":1,"gguf":Artefact::capture(root.path(),output).unwrap(),
            "binding":{"source":{"wrong_base":true},"adapter":null,"codec":"native_hf_axis_to_i2s_v1","implementation_sha256":"a".repeat(64)}});
        let receipt_path = root.path().join(output.with_extension("export.json"));
        artefact::write_json(&receipt_path, &receipt).unwrap();
        assert!(preflight_explicit_export(root.path(), output, &checkpoint).is_err());
        receipt["binding"]["source"] = serde_json::json!({});
        artefact::write_json(&receipt_path, &receipt).unwrap();
        preflight_explicit_export(root.path(), output, &checkpoint).unwrap();
        receipt["binding"]["adapter"] = serde_json::to_value(&checkpoint.tensor).unwrap();
        artefact::write_json(&receipt_path, &receipt).unwrap();
        assert!(preflight_explicit_export(root.path(), output, &checkpoint).is_err());
        assert!(
            preflight_explicit_export(root.path(), Path::new("../escape.gguf"), &checkpoint)
                .is_err()
        );
        assert!(
            serde_json::from_value::<BitNetBackendOptions>(serde_json::json!({"ggf":"typo"}))
                .is_err()
        );
    }
}
