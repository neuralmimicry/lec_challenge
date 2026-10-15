//! Native GGUF inference reuses the validated transformer attention/RMS core.
//! Quantised matrix weights stay on the requested device; no implicit offload.
use super::*;
use candle_core::quantized::{GgmlDType, QMatMul, gguf_file};
use std::io::{Read, Seek};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantisedMemoryEstimate {
    pub quantised_file_bytes: u64,
    pub dense_embedding_bytes: u64,
    pub kv_cache_bytes: u64,
    pub attention_score_bytes: u64,
    pub activation_workspace_bytes: u64,
    pub runtime_reserve_bytes: u64,
    pub conservative_total_bytes: u64,
    pub context_tokens: usize,
    pub assumptions: String,
}

/// Conservatively retains the entire file size as resident weights, then adds
/// the expanded GPU embedding, float32 KV/scores and temporary activations.
/// Actual allocation and driver free memory still decide whether loading works.
pub fn estimate_quantised_memory(
    cfg: &QwenConfig,
    file_bytes: u64,
    context_tokens: usize,
    gpu: bool,
) -> Result<QuantisedMemoryEstimate> {
    cfg.validate()?;
    ensure!(
        context_tokens > 0 && context_tokens <= cfg.max_position_embeddings,
        "invalid memory-plan context length"
    );
    let bytes = |count: u128| -> Result<u64> {
        u64::try_from(count).context("quantised memory estimate overflow")
    };
    let embedding =
        bytes(cfg.vocab_size as u128 * cfg.hidden_size as u128 * if gpu { 2 } else { 4 })?;
    let kv = bytes(
        2 * cfg.num_hidden_layers as u128
            * cfg.num_key_value_heads as u128
            * (cfg.hidden_size / cfg.num_attention_heads) as u128
            * context_tokens as u128
            * 4,
    )?;
    let scores = bytes(
        cfg.num_attention_heads as u128 * context_tokens as u128 * context_tokens as u128 * 4,
    )?;
    let workspace = bytes(
        context_tokens as u128
            * (3 * cfg.intermediate_size as u128 + 2 * cfg.hidden_size as u128)
            * 4,
    )?;
    let reserve = if gpu {
        256 * 1024 * 1024
    } else {
        64 * 1024 * 1024
    };
    let total = bytes(
        file_bytes as u128
            + embedding as u128
            + kv as u128
            + scores as u128
            + workspace as u128
            + reserve as u128,
    )?;
    Ok(QuantisedMemoryEstimate {
        quantised_file_bytes:file_bytes, dense_embedding_bytes:embedding, kv_cache_bytes:kv,
        attention_score_bytes:scores, activation_workspace_bytes:workspace, runtime_reserve_bytes:reserve,
        conservative_total_bytes:total, context_tokens,
        assumptions:"Full quantised weights on the selected device; embedding expanded to GPU float16 or CPU float32; float32 activations/KV/attention; no hybrid offload. Driver/allocator/workspace behaviour may require additional headroom.".into(),
    })
}

fn metadata_usize(content: &gguf_file::Content, key: &str) -> Result<usize> {
    Ok(content
        .metadata
        .get(key)
        .with_context(|| format!("GGUF metadata missing {key}"))?
        .to_u32()? as usize)
}

fn validate_metadata(
    content: &gguf_file::Content,
    cfg: &QwenConfig,
    context_tokens: usize,
) -> Result<()> {
    cfg.validate()?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .context("GGUF architecture is absent")?
        .to_string()?;
    ensure!(
        architecture == &cfg.model_type,
        "GGUF architecture differs from the verified original base"
    );
    for (suffix, expected) in [
        ("embedding_length", cfg.hidden_size),
        ("block_count", cfg.num_hidden_layers),
        ("feed_forward_length", cfg.intermediate_size),
        ("attention.head_count", cfg.num_attention_heads),
        ("attention.head_count_kv", cfg.num_key_value_heads),
    ] {
        ensure!(
            metadata_usize(content, &format!("{architecture}.{suffix}"))? == expected,
            "GGUF/source dimension differs: {suffix}"
        );
    }
    let epsilon = content
        .metadata
        .get(&format!("{architecture}.attention.layer_norm_rms_epsilon"))
        .context("GGUF RMS epsilon is absent")?
        .to_f32()? as f64;
    ensure!(
        (epsilon - cfg.rms_norm_eps).abs() <= cfg.rms_norm_eps * 1e-5,
        "GGUF/source RMS epsilon differs"
    );
    let theta = content
        .metadata
        .get(&format!("{architecture}.rope.freq_base"))
        .map(|value| value.to_f32())
        .transpose()?
        .unwrap_or(10000.) as f64;
    ensure!(
        (theta - cfg.rope_theta).abs() <= cfg.rope_theta * 1e-6,
        "GGUF/source rotary base differs"
    );
    let factor = content
        .metadata
        .get(&format!("{architecture}.rope.scaling.factor"))
        .map(|value| value.to_f32())
        .transpose()?
        .unwrap_or(1.) as f64;
    ensure!(
        (factor
            - cfg
                .rope_scaling
                .as_ref()
                .map(|value| value.factor)
                .unwrap_or(1.))
        .abs()
            < 1e-6,
        "GGUF/source rotary scaling differs"
    );
    if let Some(kind) = content
        .metadata
        .get(&format!("{architecture}.rope.scaling.type"))
    {
        let kind = kind.to_string()?;
        ensure!(
            (factor == 1. && matches!(kind.as_str(), "none" | "linear"))
                || (factor > 1. && kind == "linear"),
            "unsupported GGUF rotary scaling semantics"
        );
    }
    if let Some(dimension) = content
        .metadata
        .get(&format!("{architecture}.rope.dimension_count"))
    {
        ensure!(
            dimension.to_u32()? as usize == cfg.hidden_size / cfg.num_attention_heads,
            "partial GGUF rotary dimensions are unsupported"
        );
    }
    ensure!(
        metadata_usize(content, &format!("{architecture}.context_length"))?
            >= context_tokens.min(cfg.max_position_embeddings),
        "GGUF context is shorter than the planned runtime"
    );
    Ok(())
}

pub(super) fn validate_tokeniser_metadata(
    content: &gguf_file::Content,
    tokenizer: &Tokenizer,
    source: &serde_json::Value,
    eos_ids: &[u32],
) -> Result<()> {
    let eos = metadata_usize(content, "tokenizer.ggml.eos_token_id")?;
    ensure!(
        eos_ids.iter().any(|id| *id as usize == eos),
        "GGUF EOS differs from the original source generation contract"
    );
    if let Some(expected) = source
        .get("bos_token_id")
        .and_then(serde_json::Value::as_u64)
    {
        ensure!(
            metadata_usize(content, "tokenizer.ggml.bos_token_id")? as u64 == expected,
            "GGUF BOS differs from the original source model"
        );
    }
    let tokens = match content.metadata.get("tokenizer.ggml.tokens") {
        Some(gguf_file::Value::Array(values)) => values,
        _ => anyhow::bail!(
            "GGUF embedded vocabulary is absent; original token-ID mapping cannot be checked"
        ),
    };
    for (token, id) in tokenizer.get_vocab(true) {
        let actual = tokens
            .get(id as usize)
            .with_context(|| format!("GGUF vocabulary omits original token ID {id}"))?
            .to_string()?;
        ensure!(
            actual == &token,
            "GGUF/original token string differs at ID {id}"
        );
    }
    Ok(())
}

fn shape(content: &gguf_file::Content, name: &str, expected: &[usize]) -> Result<()> {
    ensure!(
        content
            .tensor_infos
            .get(name)
            .with_context(|| format!("GGUF tensor missing {name}"))?
            .shape
            .dims()
            == expected,
        "GGUF tensor dimensions disagree: {name}"
    );
    Ok(())
}

fn small_tensor<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    expected: &[usize],
    device: &Device,
) -> Result<Tensor> {
    shape(content, name, expected)?;
    Ok(content.tensor(reader, name, device)?.dequantize(device)?)
}

fn linear<R: Read + Seek>(
    content: &gguf_file::Content,
    reader: &mut R,
    name: &str,
    out: usize,
    input: usize,
    bias: bool,
    device: &Device,
) -> Result<LinearBackend> {
    let weight_name = format!("{name}.weight");
    shape(content, &weight_name, &[out, input])?;
    let tensor = content.tensor(reader, &weight_name, device)?;
    ensure!(
        matches!(
            tensor.dtype(),
            GgmlDType::F32
                | GgmlDType::F16
                | GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ),
        "unsupported quantised matrix type"
    );
    // Construct directly so CANDLE_DEQUANTIZE_ALL cannot silently expand a
    // multi-gigabyte matrix or change the recorded execution contract.
    let weight = if matches!(tensor.dtype(), GgmlDType::F32 | GgmlDType::F16) {
        QMatMul::Tensor(tensor.dequantize(device)?)
    } else {
        QMatMul::QTensor(Arc::new(tensor))
    };
    let bias = if bias {
        Some(small_tensor(
            content,
            reader,
            &format!("{name}.bias"),
            &[out],
            device,
        )?)
    } else {
        None
    };
    Ok(LinearBackend::Quantised { weight, bias })
}

impl QwenModel {
    pub(super) fn from_gguf<R: Read + Seek>(
        cfg: QwenConfig,
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        file_bytes: u64,
        context_tokens: usize,
    ) -> Result<Self> {
        validate_metadata(content, &cfg, context_tokens)?;
        let bias = cfg.attention_bias.unwrap_or(cfg.model_type == "qwen2");
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_dimension = head_dim * cfg.num_key_value_heads;
        shape(
            content,
            "token_embd.weight",
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        let embedding_tensor = content.tensor(reader, "token_embd.weight", device)?;
        let embedding = if device.is_cuda() {
            if matches!(embedding_tensor.dtype(), GgmlDType::F32 | GgmlDType::F16) {
                embedding_tensor.dequantize(device)?.to_dtype(DType::F16)?
            } else {
                embedding_tensor.dequantize_f16(device)?
            }
        } else {
            embedding_tensor.dequantize(device)?
        };
        // Drop the quantised embedding allocation before loading layer weights;
        // tied output weights are loaded separately only when actually required.
        drop(embedding_tensor);
        let embedding = Embedding::new(embedding, cfg.hidden_size);
        let output_name = if content.tensor_infos.contains_key("output.weight") {
            "output"
        } else {
            ensure!(
                cfg.tie_word_embeddings,
                "untied source requires a GGUF output matrix"
            );
            "token_embd"
        };
        let lm_head = linear(
            content,
            reader,
            output_name,
            cfg.vocab_size,
            cfg.hidden_size,
            false,
            device,
        )?;
        let norm = LayerNorm::rms_norm(
            small_tensor(
                content,
                reader,
                "output_norm.weight",
                &[cfg.hidden_size],
                device,
            )?,
            cfg.rms_norm_eps,
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let p = format!("blk.{index}");
            layers.push(Layer {
                q: Projection {
                    base: linear(
                        content,
                        reader,
                        &format!("{p}.attn_q"),
                        cfg.hidden_size,
                        cfg.hidden_size,
                        bias,
                        device,
                    )?,
                    adapter: None,
                },
                k: linear(
                    content,
                    reader,
                    &format!("{p}.attn_k"),
                    kv_dimension,
                    cfg.hidden_size,
                    bias,
                    device,
                )?,
                v: Projection {
                    base: linear(
                        content,
                        reader,
                        &format!("{p}.attn_v"),
                        kv_dimension,
                        cfg.hidden_size,
                        bias,
                        device,
                    )?,
                    adapter: None,
                },
                o: linear(
                    content,
                    reader,
                    &format!("{p}.attn_output"),
                    cfg.hidden_size,
                    cfg.hidden_size,
                    cfg.model_type == "llama" && bias,
                    device,
                )?,
                gate: linear(
                    content,
                    reader,
                    &format!("{p}.ffn_gate"),
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    false,
                    device,
                )?,
                up: linear(
                    content,
                    reader,
                    &format!("{p}.ffn_up"),
                    cfg.intermediate_size,
                    cfg.hidden_size,
                    false,
                    device,
                )?,
                down: linear(
                    content,
                    reader,
                    &format!("{p}.ffn_down"),
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    false,
                    device,
                )?,
                norm1: LayerNorm::rms_norm(
                    small_tensor(
                        content,
                        reader,
                        &format!("{p}.attn_norm.weight"),
                        &[cfg.hidden_size],
                        device,
                    )?,
                    cfg.rms_norm_eps,
                ),
                norm2: LayerNorm::rms_norm(
                    small_tensor(
                        content,
                        reader,
                        &format!("{p}.ffn_norm.weight"),
                        &[cfg.hidden_size],
                        device,
                    )?,
                    cfg.rms_norm_eps,
                ),
                kv: None,
            });
        }
        let scale = cfg
            .rope_scaling
            .as_ref()
            .map(|value| value.factor)
            .unwrap_or(1.);
        let inv = Tensor::from_vec(
            (0..head_dim)
                .step_by(2)
                .map(|i| (1. / (scale * cfg.rope_theta.powf(i as f64 / head_dim as f64))) as f32)
                .collect::<Vec<_>>(),
            (1, head_dim / 2),
            device,
        )?;
        let context = cfg.max_position_embeddings.min(context_tokens);
        ensure!(
            context > 0,
            "quantised context must contain at least one token"
        );
        let time = Tensor::arange(0_u32, context as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((context, 1))?;
        let phase = time.matmul(&inv)?;
        let mut types = BTreeMap::<String, usize>::new();
        for info in content.tensor_infos.values() {
            *types.entry(format!("{:?}", info.ggml_dtype)).or_default() += 1;
        }
        ensure!(
            types
                .iter()
                .any(|(kind, count)| kind.starts_with('Q') && *count > 0),
            "GGUF contains no quantised weights"
        );
        let plan = estimate_quantised_memory(&cfg, file_bytes, context, device.is_cuda())?;
        let interleaved_rotary = cfg.model_type == "llama";
        Ok(Self {
            cfg,
            embedding,
            layers,
            norm,
            lm_head,
            cos: phase.cos()?,
            sin: phase.sin()?,
            device: device.clone(),
            dtype: DType::F32,
            quantisation: Some(
                serde_json::json!({"format":"gguf","tensor_types":types,"matrix_backend":"native_candle_QMatMul","all_weights_dequantised":false,"embedding_precision":if device.is_cuda(){"F16"}else{"F32"},"kv_precision":"F32","hybrid_cpu_offload":false,"memory_estimate":plan}),
            ),
            interleaved_rotary,
            adapter_vars: BTreeMap::new(),
            _asset_lease: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QTensor;
    use std::io::Cursor;

    fn configuration() -> QwenConfig {
        QwenConfig {
            model_type: "qwen2".into(),
            attention_bias: None,
            rope_scaling: None,
            hidden_act: "silu".into(),
            mlp_bias: false,
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 32,
            rope_theta: 10000.,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            use_sliding_window: false,
        }
    }

    fn fixture() -> Cursor<Vec<u8>> {
        let mut tensors = BTreeMap::<String, QTensor>::new();
        let mut state = StableRng::new(47);
        for (name, out, input) in [
            ("token_embd.weight", 64, 32),
            ("blk.0.attn_q.weight", 32, 32),
            ("blk.0.attn_k.weight", 16, 32),
            ("blk.0.attn_v.weight", 16, 32),
            ("blk.0.attn_output.weight", 32, 32),
            ("blk.0.ffn_gate.weight", 64, 32),
            ("blk.0.ffn_up.weight", 64, 32),
            ("blk.0.ffn_down.weight", 32, 64),
        ] {
            let values = (0..out * input)
                .map(|_| ((state.uniform() * 2. - 1.) * 0.05) as f32)
                .collect::<Vec<_>>();
            let tensor = Tensor::from_vec(values, (out, input), &Device::Cpu).unwrap();
            tensors.insert(
                name.into(),
                QTensor::quantize(&tensor, GgmlDType::Q4_0).unwrap(),
            );
        }
        for (name, size, one) in [
            ("output_norm.weight", 32, true),
            ("blk.0.attn_norm.weight", 32, true),
            ("blk.0.ffn_norm.weight", 32, true),
            ("blk.0.attn_q.bias", 32, false),
            ("blk.0.attn_k.bias", 16, false),
            ("blk.0.attn_v.bias", 16, false),
        ] {
            let tensor =
                Tensor::from_vec(vec![if one { 1_f32 } else { 0. }; size], size, &Device::Cpu)
                    .unwrap();
            tensors.insert(
                name.into(),
                QTensor::quantize(&tensor, GgmlDType::F32).unwrap(),
            );
        }
        let metadata = [
            (
                "general.architecture",
                gguf_file::Value::String("qwen2".into()),
            ),
            ("qwen2.embedding_length", gguf_file::Value::U32(32)),
            ("qwen2.block_count", gguf_file::Value::U32(1)),
            ("qwen2.feed_forward_length", gguf_file::Value::U32(64)),
            ("qwen2.attention.head_count", gguf_file::Value::U32(4)),
            ("qwen2.attention.head_count_kv", gguf_file::Value::U32(2)),
            ("qwen2.context_length", gguf_file::Value::U32(32)),
            (
                "qwen2.attention.layer_norm_rms_epsilon",
                gguf_file::Value::F32(1e-6),
            ),
            ("qwen2.rope.freq_base", gguf_file::Value::F32(10000.)),
        ];
        let metadata = metadata
            .iter()
            .map(|(key, value)| (*key, value))
            .collect::<Vec<_>>();
        let tensors = tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let mut bytes = Cursor::new(Vec::new());
        gguf_file::write(&mut bytes, &metadata, &tensors).unwrap();
        bytes.set_position(0);
        bytes
    }

    #[test]
    fn real_quantised_tensor_round_trip_runs_the_shared_cpu_transformer() {
        let mut bytes = fixture();
        let length = bytes.get_ref().len() as u64;
        let content = gguf_file::Content::read(&mut bytes).unwrap();
        let mut model = QwenModel::from_gguf(
            configuration(),
            &content,
            &mut bytes,
            &Device::Cpu,
            length,
            32,
        )
        .unwrap();
        assert!(model.adapter_vars.is_empty());
        assert_eq!(
            model.quantisation.as_ref().unwrap()["all_weights_dequantised"],
            false
        );
        assert!(matches!(
            &model.layers[0].q.base,
            LinearBackend::Quantised {
                weight: QMatMul::QTensor(_),
                ..
            }
        ));
        let input = Tensor::from_slice(&[1_u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let hidden = model.hidden(&input, 0, false).unwrap();
        let logits = model.logits(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 4, 64]);
        assert!(
            logits
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        let full = logits.i((0, 3, ..)).unwrap().to_vec1::<f32>().unwrap();
        model.clear_cache();
        let mut last = Vec::new();
        for (offset, token) in [1_u32, 2, 3, 4].into_iter().enumerate() {
            let input = Tensor::from_slice(&[token], (1, 1), &Device::Cpu).unwrap();
            let hidden = model.hidden(&input, offset, true).unwrap();
            last = model
                .logits(&hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
        }
        let largest = full
            .iter()
            .zip(&last)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(
            largest < 0.01,
            "quantised full/cached logits diverged: {largest}"
        );
    }

    #[test]
    fn gguf_metadata_cannot_override_original_architecture_or_tensor_dimensions() {
        let mut bytes = fixture();
        let mut content = gguf_file::Content::read(&mut bytes).unwrap();
        let cfg = configuration();
        validate_metadata(&content, &cfg, 32).unwrap();
        content.metadata.insert(
            "general.architecture".into(),
            gguf_file::Value::String("llama".into()),
        );
        assert!(validate_metadata(&content, &cfg, 32).is_err());
        content.metadata.insert(
            "general.architecture".into(),
            gguf_file::Value::String("qwen2".into()),
        );
        content.metadata.insert(
            "qwen2.rope.freq_base".into(),
            gguf_file::Value::F32(1000000.),
        );
        assert!(validate_metadata(&content, &cfg, 32).is_err());
        assert!(shape(&content, "blk.0.attn_q.weight", &[64, 32]).is_err());
    }

    #[test]
    fn requested_context_cannot_exceed_the_quantised_publishers_contract() {
        let mut bytes = fixture();
        let mut content = gguf_file::Content::read(&mut bytes).unwrap();
        let cfg = configuration();
        content
            .metadata
            .insert("qwen2.context_length".into(), gguf_file::Value::U32(16));
        validate_metadata(&content, &cfg, 16).unwrap();
        assert!(validate_metadata(&content, &cfg, 17).is_err());
    }

    #[test]
    fn memory_plan_accounts_for_embedding_expansion_and_quadratic_attention() {
        let cfg = configuration();
        let small = estimate_quantised_memory(&cfg, 1_000_000, 8, true).unwrap();
        let larger = estimate_quantised_memory(&cfg, 1_000_000, 16, true).unwrap();
        assert_eq!(small.dense_embedding_bytes, 64 * 32 * 2);
        assert_eq!(larger.kv_cache_bytes, small.kv_cache_bytes * 2);
        assert_eq!(
            larger.attention_score_bytes,
            small.attention_score_bytes * 4
        );
        assert!(
            small.conservative_total_bytes
                > small.quantised_file_bytes + small.dense_embedding_bytes
        );
        assert!(estimate_quantised_memory(&cfg, 1_000_000, 33, true).is_err());
    }

    #[test]
    fn gguf_vocabulary_and_termination_cannot_override_the_original_tokeniser() {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<unk>".to_owned(), 0),
                    ("a".to_owned(), 1),
                    ("b".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let tokenizer = Tokenizer::new(model);
        let mut bytes = fixture();
        let mut content = gguf_file::Content::read(&mut bytes).unwrap();
        content.metadata.insert(
            "tokenizer.ggml.eos_token_id".into(),
            gguf_file::Value::U32(2),
        );
        content.metadata.insert(
            "tokenizer.ggml.bos_token_id".into(),
            gguf_file::Value::U32(1),
        );
        let tokens = || {
            gguf_file::Value::Array(
                ["<unk>", "a", "b"]
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).into()))
                    .collect(),
            )
        };
        content
            .metadata
            .insert("tokenizer.ggml.tokens".into(), tokens());
        let source = serde_json::json!({"bos_token_id":1});
        validate_tokeniser_metadata(&content, &tokenizer, &source, &[2]).unwrap();
        assert!(validate_tokeniser_metadata(&content, &tokenizer, &source, &[3]).is_err());
        assert!(
            validate_tokeniser_metadata(
                &content,
                &tokenizer,
                &serde_json::json!({"bos_token_id":0}),
                &[2]
            )
            .is_err()
        );
        content.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            gguf_file::Value::Array(
                ["<unk>", "b", "a"]
                    .iter()
                    .map(|token| gguf_file::Value::String((*token).into()))
                    .collect(),
            ),
        );
        assert!(validate_tokeniser_metadata(&content, &tokenizer, &source, &[2]).is_err());
        content.metadata.remove("tokenizer.ggml.tokens");
        assert!(validate_tokeniser_metadata(&content, &tokenizer, &source, &[2]).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit actual CUDA quantised-kernel QA; requires a free supported GPU"]
    fn actual_cuda_quantised_round_trip_matches_cpu_without_dequantising_weights() {
        let mut bytes = fixture();
        let length = bytes.get_ref().len() as u64;
        let content = gguf_file::Content::read(&mut bytes).unwrap();
        let mut cpu = QwenModel::from_gguf(
            configuration(),
            &content,
            &mut bytes,
            &Device::Cpu,
            length,
            32,
        )
        .unwrap();
        let device = Device::new_cuda(0).expect("explicit CUDA device; CPU fallback is forbidden");
        let mut gpu =
            QwenModel::from_gguf(configuration(), &content, &mut bytes, &device, length, 32)
                .unwrap();
        assert!(gpu.device().is_cuda());
        assert!(matches!(
            &gpu.layers[0].q.base,
            LinearBackend::Quantised {
                weight: QMatMul::QTensor(_),
                ..
            }
        ));
        let cpu_input = Tensor::from_slice(&[1_u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let gpu_input = cpu_input.to_device(&device).unwrap();
        let cpu_hidden = cpu.hidden(&cpu_input, 0, false).unwrap();
        let gpu_hidden = gpu.hidden(&gpu_input, 0, false).unwrap();
        let reference = cpu
            .logits(&cpu_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let actual = gpu
            .logits(&gpu_hidden)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(actual.iter().all(|value| value.is_finite()));
        let largest = reference
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0_f32, f32::max);
        assert!(
            largest < 0.02,
            "CUDA/CPU quantised logits diverged: {largest}"
        );
        let nvidia = std::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .unwrap();
        assert!(nvidia.status.success());
        let report = serde_json::json!({"evidence_kind":"actual_native_cuda_Q4_GGUF_fixture_forward_compared_with_CPU","assessment_model_or_quality_evidence":false,"actual_device":format!("{:?}",device.location()),"activation_dtype":"F32","matrix_weights_remain_quantised":true,"largest_absolute_logit_difference":largest,"compared_logit_count":actual.len(),"process_id":std::process::id(),"nvidia_smi":String::from_utf8_lossy(&nvidia.stdout),"cpu_logits_first_eight":&reference[..8],"cuda_logits_first_eight":&actual[..8]});
        let output = std::path::PathBuf::from(format!(
            "reports/comparative/search/quantised-kernel-qa/{}.json",
            uuid::Uuid::new_v4()
        ));
        crate::artefact::write_json(&output, &report).unwrap();
        println!(
            "{}\nreport={}",
            serde_json::to_string_pretty(&report).unwrap(),
            output.display()
        );
    }
}
