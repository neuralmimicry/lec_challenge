//! Bind optional native inference code and exported tensors independently of
//! the Rust orchestration binary. Ports, logs and affinity are observations,
//! while executable and tensor identities are immutable evaluation inputs.
use super::*;

fn contains(files: &[Artefact], expected: &Artefact) -> Result<bool> {
    for file in files {
        if same(file, expected)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_sha256(value: &serde_json::Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Inspect small JSON receipts only; never read an entire GGUF into memory.
fn receipts(plan: &Plan, root: &Path) -> Result<(Option<Artefact>, Option<Artefact>)> {
    let mut paths = BTreeSet::new();
    let mut engine = None;
    let mut export = None;
    for file in &plan.backend_artefacts {
        ensure!(paths.insert(&file.path), "duplicate study backend artefact");
        let path = file.verify(root)?;
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        ensure!(
            file.bytes <= 16 << 20,
            "native backend JSON receipt too large"
        );
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        if value.get("engine_commit").is_some() && value.get("llama_commit").is_some() {
            ensure!(
                engine.is_none(),
                "multiple native inference engines in one study evaluation"
            );
            crate::bitnet::cpp::verify_engine(&path)?;
            engine = Some(file.clone());
        }
        if value["evidence_kind"] == "native_official_bitnet_qat_gguf_export" {
            // The receipt itself freezes this implementation hash. Comparing
            // it with today's source would invalidate legitimate old runs
            // after rebuilding the launcher; their Rust engine is pinned too.
            ensure!(
                export.is_none(),
                "multiple native GGUF exports in one study evaluation"
            );
            ensure!(
                plan.checkpoint.architecture == StudentArchitecture::TernaryBitnet
                    && value["schema_version"] == 1
                    && value["production_eligible"] == false
                    && value["binding"]["codec"] == "native_hf_axis_to_i2s_v1"
                    && is_sha256(&value["binding"]["implementation_sha256"])
                    && value["tensors"].as_u64().is_some_and(|n| n > 0),
                "native GGUF export receipt has unsupported inputs or implementation"
            );
            let base: serde_json::Value =
                serde_json::from_slice(&fs::read(plan.checkpoint.base_assets.verify(root)?)?)?;
            ensure!(
                value["binding"]["source"] == base,
                "native GGUF export used another base"
            );
            let adapter = &value["binding"]["adapter"];
            if same(&plan.checkpoint.manifest, &plan.checkpoint.base_assets)? {
                ensure!(
                    adapter.is_null(),
                    "base GGUF evaluation silently includes an adapter"
                );
            } else {
                ensure!(
                    *adapter == serde_json::to_value(&plan.checkpoint.tensor)?
                        && value["changed_effective_trits"]
                            .as_u64()
                            .is_some_and(|n| n > 0),
                    "native GGUF export lacks this checkpoint's changed effective trits"
                );
            }
            let gguf: Artefact = serde_json::from_value(value["gguf"].clone())?;
            gguf.verify(root)?;
            ensure!(
                contains(&plan.backend_artefacts, &gguf)?,
                "native GGUF is absent from frozen backend artefacts"
            );
            export = Some(gguf);
        }
    }
    ensure!(
        engine.is_some() == export.is_some(),
        "native engine and GGUF export receipt must be frozen together"
    );
    Ok((engine, export))
}

pub(super) fn verify(plan: &Plan, root: &Path) -> Result<()> {
    receipts(plan, root)?;
    Ok(())
}

pub(super) fn execution(metadata: &serde_json::Value, plan: &Plan, root: &Path) -> Result<()> {
    ensure!(
        metadata["checkpoint_id"] == plan.checkpoint.checkpoint_id
            && metadata["requested"] == plan.runtime,
        "loaded study backend changed checkpoint or requested runtime"
    );
    let (engine, gguf) = receipts(plan, root)?;
    if metadata["engine"] == "pinned_bitnet_cpp_supervised_by_rust" {
        let expected_engine =
            engine.context("C++ backend has no frozen native executable receipt")?;
        let expected_gguf = gguf.context("C++ backend has no frozen GGUF/export receipt")?;
        ensure!(
            metadata["engine_manifest"] == serde_json::to_value(expected_engine)?
                && metadata["gguf"] == serde_json::to_value(expected_gguf)?
                && metadata["original_tokeniser_parity_verified"] == true
                && metadata["kv_cache"] == true
                && metadata["actual_device"] == "Cpu"
                && metadata["actual_dtype"] == "F32"
                && metadata["cuda_active"] == false,
            "C++ inference changed its frozen native engine, exported tensors or verified runtime"
        );
    } else {
        ensure!(
            engine.is_none(),
            "frozen native C++ backend was replaced by another inference engine"
        );
    }
    Ok(())
}
