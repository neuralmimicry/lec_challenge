//! Generation-only imports do not depend on the original grading being healthy.
use super::*;
use std::io::Read;

fn tensor_header(path: &Path) -> Result<()> {
    let mut file = fs::File::open(path)?;
    let total = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let length = u64::from_le_bytes(prefix);
    ensure!(
        length > 0 && length <= 16 << 20 && length + 8 <= total,
        "invalid bounded safetensors header length"
    );
    let mut bytes = vec![0u8; length as usize];
    file.read_exact(&mut bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&bytes)?;
    let mut ranges = Vec::new();
    for (name, tensor) in header
        .as_object()
        .context("tensor header is not an object")?
    {
        if name == "__metadata__" {
            continue;
        }
        let width = match tensor["dtype"].as_str() {
            Some("F64" | "I64" | "U64") => 8,
            Some("F32" | "I32" | "U32") => 4,
            Some("F16" | "BF16" | "I16" | "U16") => 2,
            Some("I8" | "U8" | "BOOL") => 1,
            _ => anyhow::bail!("unsupported imported tensor dtype"),
        };
        let elements = tensor["shape"]
            .as_array()
            .context("tensor shape absent")?
            .iter()
            .try_fold(1u64, |product, dimension| {
                product
                    .checked_mul(dimension.as_u64().context("invalid tensor dimension")?)
                    .context("tensor shape overflow")
            })?;
        let offsets = tensor["data_offsets"]
            .as_array()
            .context("tensor offsets absent")?;
        ensure!(offsets.len() == 2, "invalid tensor offset pair");
        let begin = offsets[0].as_u64().context("invalid tensor start")?;
        let end = offsets[1].as_u64().context("invalid tensor end")?;
        ensure!(
            begin <= end
                && end <= total - length - 8
                && end - begin
                    == elements
                        .checked_mul(width)
                        .context("tensor byte overflow")?,
            "tensor shape/extent differs"
        );
        ranges.push((begin, end));
    }
    ensure!(
        !ranges.is_empty(),
        "imported checkpoint contains no tensors"
    );
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        ensure!(start == end, "tensor ranges overlap or leave gaps");
        end = next;
    }
    ensure!(end == total - length - 8, "unbound tensor payload remains");
    Ok(())
}

fn checkpoint(identity: &Checkpoint, root: &Path) -> Result<()> {
    ensure!(
        identity.checkpoint_id == identity.tensor.sha256,
        "imported checkpoint ID differs from its tensor"
    );
    let assets_path = identity.base_assets.verify(root)?;
    let base = assets_path.parent().context("base directory absent")?;
    let assets = match identity.architecture {
        StudentArchitecture::ConventionalQwenLora => crate::model::validate_assets(base)?,
        StudentArchitecture::TernaryBitnet => crate::hub::validate_bitnet_student_assets(base)?,
    };
    identity.tokenizer.verify(root)?;
    ensure!(
        assets.files.get("tokenizer.json") == Some(&identity.tokenizer.sha256),
        "imported tokeniser differs"
    );
    tensor_header(&identity.tensor.verify(root)?)?;
    let manifest = identity.manifest.verify(root)?;
    if assets.files.get("model.safetensors") == Some(&identity.checkpoint_id) {
        ensure!(
            same(&identity.manifest, &identity.base_assets)?,
            "imported base has another manifest"
        );
    } else {
        let proof = crate::distillation::checkpoint::validate_checkpoint_metadata(
            manifest.parent().context("adapter directory absent")?,
            &assets,
        )?;
        ensure!(
            proof.adapter_sha256 == identity.checkpoint_id
                && serde_json::to_value(proof.spec.student_kind)?
                    == serde_json::to_value(identity.architecture)?,
            "imported adapter architecture or identity differs"
        );
    }
    Ok(())
}

fn strict_execution(metadata: &serde_json::Value, plan: &Plan, root: &Path) -> Result<()> {
    backend::execution(metadata, plan, root)?;
    if metadata["engine"] == "native_rust_candle" {
        let runtime: crate::model::RuntimeOptions = serde_json::from_value(plan.runtime.clone())?;
        let (device, cuda) = match runtime.device {
            crate::model::ExecutionDevice::Cpu => ("Cpu".to_owned(), false),
            crate::model::ExecutionDevice::Cuda { ordinal } => {
                (format!("Cuda {{ gpu_id: {ordinal} }}"), true)
            }
        };
        let dtype = match runtime.dtype {
            crate::model::InferenceDtype::F32 => "F32",
            crate::model::InferenceDtype::F16 => "F16",
        };
        ensure!(
            metadata["actual_device"] == device
                && metadata["actual_dtype"] == dtype
                && metadata["cpu_fallback"] == false
                && metadata["cuda_active"] == cuda,
            "imported actual native runtime differs from original request"
        );
    } else {
        ensure!(
            metadata["engine"] == "pinned_bitnet_cpp_supervised_by_rust",
            "unknown imported generation engine"
        );
    }
    Ok(())
}

pub(super) fn collect(root: &Path, directory: &Path) -> Result<OriginalEvaluation> {
    let directory = root_relative(root, directory)?;
    let plan_file = capture(root, &directory.join("plan.json"))?;
    let plan: Plan = read(&plan_file.verify(root)?)?;
    evidence::plan_structure(&plan, root)?;
    checkpoint(&plan.checkpoint, root)?;
    let mut receipts = Vec::new();
    let main = directory.join("execution.json");
    if main.exists() {
        receipts.push(capture(root, &main)?);
    }
    let attempts = directory.join("execution-attempts");
    if attempts.exists() {
        let mut paths = fs::read_dir(&attempts)?
            .map(|x| x.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        for path in paths {
            if path.extension().is_some_and(|e| e == "json") {
                receipts.push(capture(root, &path)?);
            }
        }
    }
    for receipt in &receipts {
        strict_execution(
            &read::<serde_json::Value>(&receipt.verify(root)?)?,
            &plan,
            root,
        )?;
    }
    let mut rows = Vec::new();
    let mut any_generation = false;
    for task in &plan.tasks {
        let artifact = capture(
            root,
            &directory
                .join("generations")
                .join(format!("{}.json", task.id)),
        )?;
        let raw: RawGeneration = read(&artifact.verify(root)?)?;
        evidence::raw(&raw, task, &plan, &plan_file.sha256)?;
        any_generation |= raw.generation.is_some();
        rows.push(ImportedGeneration {
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            original_plan_sha256: plan_file.sha256.clone(),
            request_sha256: artefact::hash(&serde_json::to_vec(&raw.request)?),
            extracted_code_sha256: raw
                .generation
                .as_ref()
                .filter(|_| usable(&raw))
                .map(|g| tasks::hash(crate::evaluate::extract_code(&g.completion))),
            original_raw: artifact,
        });
    }
    ensure!(
        !any_generation || main.exists(),
        "imported generations lack original loaded-runtime evidence"
    );
    Ok(OriginalEvaluation {
        directory,
        plan: plan_file,
        generation_plan: plan,
        execution_receipts: receipts,
        rows,
    })
}

pub(super) fn verify(root: &Path, original: &OriginalEvaluation) -> Result<()> {
    original.plan.verify(root)?;
    let actual: Plan = read(&root.join(&original.plan.path))?;
    ensure!(
        same(&actual, &original.generation_plan)?,
        "imported original plan snapshot changed"
    );
    evidence::plan_structure(&actual, root)?;
    checkpoint(&actual.checkpoint, root)?;
    ensure!(
        original.rows.len() == actual.tasks.len(),
        "imported original denominator incomplete"
    );
    let mut seen = BTreeSet::new();
    for (task, row) in actual.tasks.iter().zip(&original.rows) {
        ensure!(seen.insert(&row.task_id), "duplicate imported task");
        original_raw(root, original, row, task)?;
    }
    ensure!(
        original
            .execution_receipts
            .iter()
            .any(|a| a.path == original.directory.join("execution.json"))
            || original
                .rows
                .iter()
                .all(|r| read::<RawGeneration>(&root.join(&r.original_raw.path))
                    .is_ok_and(|raw| raw.generation.is_none())),
        "original loaded runtime is absent"
    );
    for receipt in &original.execution_receipts {
        strict_execution(
            &read::<serde_json::Value>(&receipt.verify(root)?)?,
            &actual,
            root,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_metadata_validation_never_requires_tensor_storage() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("tiny.safetensors");
        let header = serde_json::to_vec(
            &serde_json::json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
        )
        .unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0; 8]);
        fs::write(&path, &bytes).unwrap();
        tensor_header(&path).unwrap();
        bytes.pop();
        fs::write(&path, &bytes).unwrap();
        assert!(tensor_header(&path).is_err());
    }
}
