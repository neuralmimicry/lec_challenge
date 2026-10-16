//! Explicit GGUF teacher contexts: validate bounds before device allocation.
use super::*;

fn supported(teacher: &TeacherEvidence) -> Result<()> {
    ensure!(
        teacher.adapter_checkpoint.is_none()
            && teacher.model.architecture != "bitnet"
            && teacher.model.directory.join("gguf-assets.json").is_file(),
        "an explicit teacher context requires an original GGUF teacher; dense, adapted and BitNet overrides are unsupported"
    );
    Ok(())
}

fn bounds(requested: usize, source: usize, published: usize) -> Result<()> {
    ensure!(
        (1..=4096).contains(&requested),
        "explicit teacher context must be 1–4096 tokens"
    );
    ensure!(
        requested <= source,
        "explicit teacher context {requested} exceeds original source maximum {source}"
    );
    ensure!(
        requested <= published,
        "explicit teacher context {requested} exceeds GGUF publisher maximum {published}"
    );
    Ok(())
}

/// Read provenance/configuration and metadata, then apply the same guarded
/// hardware capacities used by explicit platform selection. No model is loaded.
pub(super) fn preflight(
    teacher: &TeacherEvidence,
    options: &RuntimeOptions,
    requested: usize,
) -> Result<()> {
    supported(teacher)?;
    ensure!(
        options.dtype == model::InferenceDtype::F32,
        "explicit GGUF teacher contexts require F32 activations"
    );
    bounds(requested, usize::MAX, usize::MAX)?;
    let proof = hub::validate_gguf_assets(&teacher.model.directory)?;
    let cfg: model::QwenConfig = serde_json::from_slice(&fs::read(
        teacher
            .model
            .directory
            .join(&proof.source_assets.directory)
            .join("config.json"),
    )?)?;
    cfg.validate()?;
    let mut file = fs::File::open(teacher.model.directory.join(&proof.filename))?;
    let gguf = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let published = gguf
        .metadata
        .get(&format!("{}.context_length", cfg.model_type))
        .context("GGUF teacher context metadata is absent")?
        .to_u32()? as usize;
    bounds(requested, cfg.max_position_embeddings, published)?;
    let gpu = model::estimate_quantised_memory(&cfg, proof.file_bytes, requested, true)?;
    let cpu = model::estimate_quantised_memory(&cfg, proof.file_bytes, requested, false)?;
    let hardware = crate::platform::detect()?;
    let (device, ordinal) = match options.device {
        model::ExecutionDevice::Cpu => ("cpu", 0),
        model::ExecutionDevice::Cuda { ordinal } => ("cuda", ordinal),
    };
    crate::platform::select_runtime(&hardware, device, ordinal, true, gpu.conservative_total_bytes, cpu.conservative_total_bytes)
        .with_context(|| format!("explicit {requested}-token teacher context failed current RAM/VRAM guard (GPU estimate {} bytes; CPU estimate {} bytes)", gpu.conservative_total_bytes, cpu.conservative_total_bytes))?;
    Ok(())
}

fn request_bound(input_tokens: usize, max_new_tokens: usize, context_tokens: usize) -> Result<()> {
    let total = input_tokens
        .checked_add(max_new_tokens)
        .context("teacher request token count overflow")?;
    ensure!(
        total <= context_tokens,
        "teacher request requires {input_tokens} input + {max_new_tokens} output = {total} tokens, exceeding explicit context {context_tokens}"
    );
    Ok(())
}

/// Count the original template/tokeniser exactly before loading any weights.
/// Historical `None` requests retain their existing runtime error behaviour.
pub(crate) fn validate_teacher_request(
    teacher: &TeacherEvidence,
    context_tokens: Option<usize>,
    request: &GenerationRequest,
) -> Result<()> {
    if let Some(context_tokens) = context_tokens {
        supported(teacher)?;
        bounds(context_tokens, usize::MAX, usize::MAX)?;
        let input =
            model::comparative_input_tokens(&teacher.model.directory, true, &request.prompt)?;
        request_bound(input, request.max_new_tokens, context_tokens)?;
    }
    Ok(())
}

/// Reverification checks the actual runtime cap, without consulting transient
/// hardware availability or allocating the historical teacher again.
pub(crate) fn validate_context_observation(
    context_tokens: Option<usize>,
    execution: Option<&serde_json::Value>,
    generation: Option<&Generation>,
    max_new_tokens: usize,
) -> Result<()> {
    if let Some(context_tokens) = context_tokens {
        bounds(context_tokens, usize::MAX, usize::MAX)?;
        if let Some(execution) = execution {
            ensure!(
                execution["context_cap"].as_u64() == Some(context_tokens as u64),
                "recorded teacher runtime context differs from its explicit plan"
            );
        }
        if let Some(generation) = generation {
            ensure!(
                execution.is_some(),
                "explicit-context generation lacks runtime metadata"
            );
            request_bound(generation.input_tokens, max_new_tokens, context_tokens)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_context_rejects_runtime_source_and_publisher_overflow() {
        bounds(2048, 32768, 4096).unwrap();
        for (requested, source, published) in [
            (0, 4096, 4096),
            (4097, 32768, 32768),
            (2048, 1024, 4096),
            (2048, 32768, 1024),
        ] {
            assert!(bounds(requested, source, published).is_err());
        }
        request_bound(1280, 768, 2048).unwrap();
        assert!(request_bound(1281, 768, 2048).is_err());
        assert!(request_bound(usize::MAX, 1, 4096).is_err());
    }

    #[test]
    fn explicit_context_evidence_binds_the_actual_cap_without_changing_legacy() {
        let old = serde_json::json!({"context_cap":1024});
        validate_context_observation(None, Some(&old), None, 768).unwrap();
        assert!(validate_context_observation(Some(2048), Some(&old), None, 768).is_err());
        let actual = serde_json::json!({"context_cap":2048});
        validate_context_observation(Some(2048), Some(&actual), None, 768).unwrap();
    }
}
