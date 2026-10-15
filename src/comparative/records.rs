//! Durable raw inference is separate from grading, preserving its first observation.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct GenerationRecord {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub recorded_unix_ms: i64,
    pub model_id: String,
    pub task_id: String,
    pub source_id: String,
    pub specification_sha256: String,
    pub seed: u64,
    pub plan_sha256: String,
    pub request: GenerationRequest,
    pub generation: Option<Generation>,
    pub generation_error: Option<String>,
    /// None only when model loading failed before an execution device existed.
    pub execution: Option<serde_json::Value>,
}

impl GenerationRecord {
    pub(super) fn new(
        frozen: &FrozenModel,
        task: &Task,
        plan: &Plan,
        hash: &str,
        result: Result<Generation>,
        execution: Option<serde_json::Value>,
    ) -> Self {
        let (generation, generation_error) = match result {
            Ok(output) => (Some(output), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        };
        Self {
            schema_version: 1,
            evidence_kind: "actual_local_model_generation".into(),
            recorded_unix_ms: crate::ledger::unix_millis(),
            model_id: frozen.model.id.clone(),
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            specification_sha256: task.specification_sha256.clone(),
            seed: plan.seed,
            plan_sha256: hash.to_owned(),
            request: request(task, plan),
            generation,
            generation_error,
            execution,
        }
    }
}

fn raw_path(root: &Path, path: &Path) -> Result<PathBuf> {
    ensure!(
        path.components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
            && !path.as_os_str().is_empty(),
        "raw generation path must stay inside its evidence root"
    );
    let mut prefix = root.to_owned();
    for component in path.components() {
        prefix.push(component);
        match fs::symlink_metadata(&prefix) {
            Ok(metadata) => ensure!(
                !metadata.file_type().is_symlink(),
                "symlink in raw generation evidence path"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(prefix)
}

pub(super) fn verify_generation(
    raw: &GenerationRecord,
    frozen: &FrozenModel,
    task: &Task,
    plan: &Plan,
    hash: &str,
    root: &Path,
) -> Result<()> {
    ensure!(
        raw.schema_version == 1
            && raw.recorded_unix_ms > 0
            && (raw.evidence_kind == "actual_local_model_generation"
                || (cfg!(test) && raw.evidence_kind == "labelled_synthetic_generation_fixture")),
        "unsupported raw generation evidence"
    );
    ensure!(
        raw.model_id == frozen.model.id
            && raw.task_id == task.id
            && raw.source_id == task.source_id
            && raw.specification_sha256 == task.specification_sha256
            && raw.seed == plan.seed
            && raw.plan_sha256 == hash
            && serde_json::to_value(&raw.request)? == serde_json::to_value(request(task, plan))?,
        "raw generation differs from the frozen request"
    );
    if let Some(execution) = &raw.execution {
        let requested: RuntimeOptions = serde_json::from_value(plan.runtime.clone())?;
        let cuda = matches!(requested.device, model::ExecutionDevice::Cuda { .. });
        let dtype = match requested.dtype {
            model::InferenceDtype::F16 => "F16",
            model::InferenceDtype::F32 => "F32",
        };
        ensure!(
            execution["engine"] == "native_rust_candle"
                && execution["requested"] == plan.runtime
                && execution["actual_dtype"] == dtype
                && execution["cuda_active"] == cuda
                && execution["cpu_fallback"] == false
                && execution["model_id"] == frozen.model.model_id
                && execution["revision"] == frozen.model.revision
                && execution["actual_device"]
                    .as_str()
                    .is_some_and(|d| !d.is_empty())
                && (!cuda || execution["cuda_feature_compiled"] == true),
            "actual execution differs from requested device/dtype"
        );
    }
    if let Some(generation) = &raw.generation {
        ensure!(
            raw.generation_error.is_none() && raw.execution.is_some(),
            "output lacks execution evidence or also claims an error"
        );
        let assets: model::AssetManifest =
            serde_json::from_slice(&fs::read(frozen.assets.verify(root)?)?)?;
        ensure!(
            generation.checkpoint_id == comparative_checkpoint_id(&assets)?,
            "generation belongs to another tensor set"
        );
        ensure!(
            raw.execution
                .as_ref()
                .is_some_and(|e| e["checkpoint_id"] == generation.checkpoint_id),
            "execution metadata belongs to another tensor set"
        );
        ensure!(
            matches!(
                generation.finish_reason.as_str(),
                "stop" | "length" | "timeout"
            ) && generation.complete == (generation.finish_reason == "stop")
                && generation.output_tokens <= raw.request.max_new_tokens
                && generation.input_tokens > 0
                && generation
                    .input_tokens
                    .checked_add(raw.request.max_new_tokens)
                    .is_some_and(|n| n <= 4096),
            "invalid generation termination/token bounds"
        );
        ensure!(
            generation.completion.trim().is_empty() || generation.output_tokens > 0,
            "nonempty decoded output has no token evidence"
        );
    } else {
        ensure!(
            raw.generation_error
                .as_ref()
                .is_some_and(|e| !e.trim().is_empty()),
            "missing generation without an explicit failure"
        );
    }
    Ok(())
}

pub(super) fn load_generation(
    path: &Path,
    frozen: &FrozenModel,
    task: &Task,
    plan: &Plan,
    hash: &str,
    root: &Path,
) -> Result<Option<(GenerationRecord, Artefact)>> {
    if !raw_path(root, path)?.exists() {
        return Ok(None);
    }
    let artifact = Artefact::capture(root, path)?;
    let raw: GenerationRecord = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
    verify_generation(&raw, frozen, task, plan, hash, root)?;
    Ok(Some((raw, artifact)))
}

pub(super) fn publish_generation(
    path: &Path,
    raw: &GenerationRecord,
    frozen: &FrozenModel,
    task: &Task,
    plan: &Plan,
    hash: &str,
    root: &Path,
) -> Result<Artefact> {
    verify_generation(raw, frozen, task, plan, hash, root)?;
    let target = raw_path(root, path)?;
    ensure!(
        !target.exists(),
        "original raw generation cannot be overwritten"
    );
    artefact::write_json(&target, raw)?;
    Artefact::capture(root, path)
}
