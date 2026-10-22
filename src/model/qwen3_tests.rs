//! Independent numerical checks for explicit-width, per-head-normalised Qwen3.
//!
//! These tiny synthetic models test implementation correctness. They do not
//! establish publisher numerical parity, task quality or assessment promotion.
use super::*;
use std::path::PathBuf;

fn configuration() -> QwenConfig {
    QwenConfig {
        model_type: "qwen3".into(),
        attention_bias: Some(false),
        head_dim: Some(16),
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

fn weights() -> HashMap<String, Tensor> {
    let mut result = HashMap::new();
    let mut rng = StableRng::new(47);
    for (name, rows, columns) in [
        ("model.embed_tokens.weight", 64, 32),
        ("model.layers.0.self_attn.q_proj.weight", 64, 32),
        ("model.layers.0.self_attn.k_proj.weight", 32, 32),
        ("model.layers.0.self_attn.v_proj.weight", 32, 32),
        ("model.layers.0.self_attn.o_proj.weight", 32, 64),
        ("model.layers.0.mlp.gate_proj.weight", 64, 32),
        ("model.layers.0.mlp.up_proj.weight", 64, 32),
        ("model.layers.0.mlp.down_proj.weight", 32, 64),
    ] {
        let values = (0..rows * columns)
            .map(|_| ((rng.uniform() * 2. - 1.) * 0.15) as f32)
            .collect::<Vec<_>>();
        result.insert(
            name.into(),
            Tensor::from_vec(values, (rows, columns), &Device::Cpu).unwrap(),
        );
    }
    for (name, length, start, increment) in [
        ("model.norm.weight", 32, 0.8_f32, 0.01_f32),
        ("model.layers.0.input_layernorm.weight", 32, 0.9, 0.006),
        (
            "model.layers.0.post_attention_layernorm.weight",
            32,
            1.1,
            -0.004,
        ),
        ("model.layers.0.self_attn.q_norm.weight", 16, 0.5, 0.07),
        ("model.layers.0.self_attn.k_norm.weight", 16, 1.4, -0.045),
    ] {
        result.insert(
            name.into(),
            Tensor::from_vec(
                (0..length).map(|i| start + i as f32 * increment).collect(),
                length,
                &Device::Cpu,
            )
            .unwrap(),
        );
    }
    result
}

fn vector(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn tiny_labelled_task_overfits_assistant_tokens_and_reloads_in_a_fresh_process() -> Result<()> {
    use crate::{
        distillation::{SequencePolicy, alignment},
        train::{ResumableAdam, TrainConfig},
    };
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("instruction".into(), 1),
                ("answer".into(), 2),
                ("prompt".into(), 3),
                ("result".into(), 5),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(wordlevel);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    let sequence = alignment::encode_with_policy(
        &tokenizer,
        "instruction prompt ",
        "answer result",
        7,
        5,
        SequencePolicy::RejectOverlength,
    )?;
    assert_eq!(sequence.input_ids, vec![1, 3, 2, 5]);
    assert_eq!(sequence.prediction_positions, vec![1, 2, 3]);
    assert_eq!(sequence.targets, vec![2, 5, 7]);
    assert_eq!(
        sequence.target_spans,
        vec![Some((0, 6)), Some((7, 13)), None]
    );
    assert!(alignment::encode(&tokenizer, "instruction prompt ", "answer result", 7, 4).is_err());
    let tensors = weights();
    let frozen = tensors
        .iter()
        .map(|(k, v)| (k.clone(), (v.clone(), vector(v))))
        .collect::<BTreeMap<_, _>>();
    let mut adapted = QwenModel::from_weights(
        configuration(),
        tensors,
        Some(AdapterConfig { rank: 4, alpha: 8. }),
        1729,
        None,
        true,
    )?;
    let sample = crate::train::EncodedSample {
        id: "labelled numerical QA, never assessment evidence".into(),
        input_ids: sequence.input_ids.clone(),
        positions: sequence.prediction_positions.clone(),
        targets: sequence.targets.clone(),
    };
    let cfg = TrainConfig {
        learning_rate: 0.01,
        weight_decay: 0.,
        ..TrainConfig::default()
    };
    let mut optimiser = ResumableAdam::new(&adapted.adapter_vars, &cfg)?;
    let initial = crate::train::assistant_loss(&mut adapted, &sample)?.to_scalar::<f32>()?;
    let mut base = QwenModel::from_weights(configuration(), weights(), None, 1729, None, false)?;
    let baseline = crate::train::assistant_loss(&mut base, &sample)?.to_scalar::<f32>()?;
    assert!(
        (initial - baseline).abs() < 1e-6,
        "zero-B LoRA must initially equal adapter-disabled base"
    );
    let mut norms = Vec::new();
    for _ in 0..64 {
        let loss = crate::train::assistant_loss(&mut adapted, &sample)?;
        norms.push(optimiser.update(&adapted.adapter_vars, &loss)?);
    }
    let after = crate::train::assistant_loss(&mut adapted, &sample)?.to_scalar::<f32>()?;
    assert!(
        after < initial * 0.5,
        "real forward/backward/Adam must lower correctly shifted assistant NLL: {initial} -> {after}"
    );
    eprintln!(
        "labelled numerical fixture: assistant NLL {initial:.6} -> {after:.6}; no assessment evidence"
    );
    assert!(norms.iter().all(|n| n.is_finite() && *n > 0.));
    for (tensor, before) in frozen.values() {
        assert_eq!(vector(tensor), *before, "base tensors must remain frozen");
    }
    let input = Tensor::new(&[[1_u32, 3, 2, 5]], &Device::Cpu)?;
    let fixed_hidden = adapted.hidden(&input, 0, false)?;
    let fixed_logits = adapted.logits(&fixed_hidden)?;
    let directory = tempfile::tempdir()?;
    let adapter = adapted
        .adapter_vars
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect::<HashMap<_, _>>();
    candle_core::safetensors::save(&adapter, directory.path().join("adapter.safetensors"))?;
    fs::write(
        directory.path().join("expected.json"),
        serde_json::to_vec(&vector(&fixed_logits))?,
    )?;
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "model::qwen3_tests::fresh_process_reload_fixture",
            "--nocapture",
        ])
        .env("LEC_LABELLED_NUMERICAL_RELOAD", directory.path())
        .status()?;
    ensure!(status.success(), "fresh-process numerical reload failed");
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.path().join("reload.json"))?)?;
    assert!(receipt["maximum_absolute_logit_error"].as_f64().unwrap() <= 1e-6);
    assert!(
        receipt["adapter_disabled_maximum_difference"]
            .as_f64()
            .unwrap()
            > 0.001
    );
    Ok(())
}

#[test]
fn fresh_process_reload_fixture() -> Result<()> {
    let Some(directory) = std::env::var_os("LEC_LABELLED_NUMERICAL_RELOAD").map(PathBuf::from)
    else {
        return Ok(());
    };
    let adapter =
        candle_core::safetensors::load(directory.join("adapter.safetensors"), &Device::Cpu)?;
    let mut reloaded = QwenModel::from_weights(
        configuration(),
        weights(),
        Some(AdapterConfig { rank: 4, alpha: 8. }),
        1729,
        Some(adapter),
        false,
    )?;
    let input = Tensor::new(&[[1_u32, 3, 2, 5]], &Device::Cpu)?;
    let hidden = reloaded.hidden(&input, 0, false)?;
    let actual = vector(&reloaded.logits(&hidden)?);
    let expected: Vec<f32> = serde_json::from_slice(&fs::read(directory.join("expected.json"))?)?;
    let mut base = QwenModel::from_weights(configuration(), weights(), None, 1729, None, false)?;
    let hidden = base.hidden(&input, 0, false)?;
    let disabled = vector(&base.logits(&hidden)?);
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    let difference = actual
        .iter()
        .zip(disabled)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f32, f32::max);
    ensure!(
        max <= 1e-6 && difference > 0.001,
        "fixed-token logits differ after export/reload, or adapter was disabled"
    );
    fs::write(
        directory.join("reload.json"),
        serde_json::to_vec(
            &serde_json::json!({"maximum_absolute_logit_error":max,"adapter_disabled_maximum_difference":difference,"tolerance":1e-6,"justification":"Same deterministic CPU F32 kernels, weights, tokens and engine across independent processes; one micro-unit allows float serialisation round trips."}),
        )?,
    )?;
    Ok(())
}

fn matrix_product(matrix: &Tensor, input: &[f64]) -> Vec<f64> {
    let (_, columns) = matrix.dims2().unwrap();
    assert_eq!(columns, input.len());
    vector(matrix)
        .chunks_exact(columns)
        .map(|row| row.iter().zip(input).map(|(w, x)| f64::from(*w) * x).sum())
        .collect()
}

fn scalar_rms(input: &[f64], weight: &Tensor, epsilon: f64) -> Vec<f64> {
    let scale = (input.iter().map(|x| x * x).sum::<f64>() / input.len() as f64 + epsilon)
        .sqrt()
        .recip();
    input
        .iter()
        .zip(vector(weight))
        .map(|(x, w)| x * scale * f64::from(w))
        .collect()
}

fn scalar_rope(input: &[f64], position: usize, theta: f64) -> Vec<f64> {
    let half = input.len() / 2;
    let mut result = vec![0.; input.len()];
    for i in 0..half {
        let phase = position as f64 / theta.powf((2 * i) as f64 / input.len() as f64);
        let (sin, cos) = phase.sin_cos();
        result[i] = input[i] * cos - input[i + half] * sin;
        result[i + half] = input[i + half] * cos + input[i] * sin;
    }
    result
}

/// Deliberately scalar and independent of the production tensor graph. The
/// optional wrong ordering demonstrates that the fixture detects that defect.
fn scalar_logits(
    cfg: &QwenConfig,
    tensors: &HashMap<String, Tensor>,
    tokens: &[u32],
    norm_before_rope: bool,
) -> Vec<f64> {
    let get = |name: &str| &tensors[name];
    let project = |name: &str, row: &[f64]| {
        matrix_product(get(&format!("model.layers.0.{name}.weight")), row)
    };
    let embedding = vector(get("model.embed_tokens.weight"));
    let mut residual = tokens
        .iter()
        .map(|token| {
            embedding[*token as usize * cfg.hidden_size..(*token as usize + 1) * cfg.hidden_size]
                .iter()
                .map(|x| f64::from(*x))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let dim = cfg.head_dimension();
    let mut queries = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (position, row) in residual.iter().enumerate() {
        let norm = scalar_rms(
            row,
            get("model.layers.0.input_layernorm.weight"),
            cfg.rms_norm_eps,
        );
        let head_transform = |projected: Vec<f64>, weight: &Tensor| {
            projected
                .chunks_exact(dim)
                .flat_map(|head| {
                    if norm_before_rope {
                        scalar_rope(
                            &scalar_rms(head, weight, cfg.rms_norm_eps),
                            position,
                            cfg.rope_theta,
                        )
                    } else {
                        scalar_rms(
                            &scalar_rope(head, position, cfg.rope_theta),
                            weight,
                            cfg.rms_norm_eps,
                        )
                    }
                })
                .collect::<Vec<_>>()
        };
        queries.push(head_transform(
            project("self_attn.q_proj", &norm),
            get("model.layers.0.self_attn.q_norm.weight"),
        ));
        keys.push(head_transform(
            project("self_attn.k_proj", &norm),
            get("model.layers.0.self_attn.k_norm.weight"),
        ));
        values.push(project("self_attn.v_proj", &norm));
    }
    for (position, row) in residual.iter_mut().enumerate() {
        let mut attention = vec![0.; cfg.num_attention_heads * dim];
        for head in 0..cfg.num_attention_heads {
            let kv_head = head / (cfg.num_attention_heads / cfg.num_key_value_heads);
            let query = &queries[position][head * dim..(head + 1) * dim];
            let scores = keys[..=position]
                .iter()
                .map(|key| {
                    query
                        .iter()
                        .zip(&key[kv_head * dim..(kv_head + 1) * dim])
                        .map(|(q, k)| q * k)
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp = scores
                .iter()
                .map(|x| (x - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = exp.iter().sum::<f64>();
            for (source, probability) in exp.iter().map(|x| x / denominator).enumerate() {
                for component in 0..dim {
                    attention[head * dim + component] +=
                        probability * values[source][kv_head * dim + component];
                }
            }
        }
        for (x, update) in row.iter_mut().zip(project("self_attn.o_proj", &attention)) {
            *x += update;
        }
        let norm = scalar_rms(
            row,
            get("model.layers.0.post_attention_layernorm.weight"),
            cfg.rms_norm_eps,
        );
        let gate = project("mlp.gate_proj", &norm);
        let up = project("mlp.up_proj", &norm);
        let activated = gate
            .iter()
            .zip(up)
            .map(|(g, u)| g / (1. + (-g).exp()) * u)
            .collect::<Vec<_>>();
        for (x, update) in row.iter_mut().zip(project("mlp.down_proj", &activated)) {
            *x += update;
        }
    }
    residual
        .iter()
        .flat_map(|row| {
            matrix_product(
                get("model.embed_tokens.weight"),
                &scalar_rms(row, get("model.norm.weight"), cfg.rms_norm_eps),
            )
        })
        .collect()
}

fn maximum_error(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (f64::from(*a) - e).abs())
        .fold(0., f64::max)
}

#[test]
fn asymmetric_qwen3_matches_independent_scalar_attention_and_cached_decoding() {
    let cfg = configuration();
    let tensors = weights();
    let tokens = [1_u32, 3, 2, 5];
    let expected = scalar_logits(&cfg, &tensors, &tokens, true);
    let wrong_order = scalar_logits(&cfg, &tensors, &tokens, false);
    let ordering_signal = expected
        .iter()
        .zip(wrong_order)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(
        ordering_signal > 0.001,
        "the fixture must expose norm/RoPE reversal"
    );
    let mut model = QwenModel::from_weights(cfg.clone(), tensors, None, 17, None, false).unwrap();
    let input = Tensor::from_slice(&tokens, (1, tokens.len()), &Device::Cpu).unwrap();
    let hidden = model.hidden(&input, 0, false).unwrap();
    let logits = vector(&model.logits(&hidden).unwrap());
    let error = maximum_error(&logits, &expected);
    assert!(
        error < 3e-5,
        "explicit-width, head-normalised scalar discrepancy: {error}"
    );
    model.clear_cache();
    for (offset, token) in tokens.into_iter().enumerate() {
        let input = Tensor::from_slice(&[token], (1, 1), &Device::Cpu).unwrap();
        let hidden = model.hidden(&input, offset, true).unwrap();
        let actual = vector(&model.logits(&hidden).unwrap());
        let error = maximum_error(
            &actual,
            &expected[offset * cfg.vocab_size..(offset + 1) * cfg.vocab_size],
        );
        assert!(
            error < 3e-5,
            "cached scalar discrepancy at {offset}: {error}"
        );
    }
}

#[test]
fn qwen3_explicit_width_retains_real_adapter_gradients_and_frozen_head_norms() {
    let tensors = weights();
    let frozen = tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), (tensor.clone(), vector(tensor))))
        .collect::<BTreeMap<_, _>>();
    let mut model = QwenModel::from_weights(
        configuration(),
        tensors,
        Some(AdapterConfig { rank: 2, alpha: 4. }),
        1701,
        None,
        true,
    )
    .unwrap();
    let sample = crate::train::EncodedSample {
        id: "synthetic Qwen3 backward QA".into(),
        input_ids: vec![1, 3, 2, 5],
        positions: vec![1, 2, 3],
        targets: vec![2, 5, 7],
    };
    let loss = crate::train::assistant_loss(&mut model, &sample).unwrap();
    assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    let gradients = loss.backward().unwrap();
    assert_eq!(model.adapter_vars.len(), 4);
    for (name, variable) in &model.adapter_vars {
        let gradient = gradients
            .get(variable)
            .unwrap_or_else(|| panic!("missing gradient {name}"));
        let norm = gradient
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm.is_finite(), "non-finite gradient {name}");
        if name.contains("lora_B") {
            assert!(norm > 0., "head norm severed gradient {name}");
        }
    }
    // A central finite difference checks the actual query-adapter path through
    // head normalisation and rotary embeddings, beyond graph reachability.
    let query_b = model
        .adapter_vars
        .iter()
        .find(|(name, _)| name.contains("q_proj") && name.contains("lora_B"))
        .unwrap()
        .1
        .clone();
    let analytical = vector(gradients.get(&query_b).unwrap());
    let (index, derivative) = analytical
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .map(|(index, derivative)| (index, *derivative))
        .unwrap();
    let original = vector(&query_b);
    let epsilon = 0.002_f32;
    let mut perturbed_losses = Vec::new();
    for direction in [1_f32, -1.] {
        let mut perturbed = original.clone();
        perturbed[index] += direction * epsilon;
        query_b
            .set(&Tensor::from_vec(perturbed, query_b.dims2().unwrap(), &Device::Cpu).unwrap())
            .unwrap();
        perturbed_losses.push(
            crate::train::assistant_loss(&mut model, &sample)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
        );
    }
    query_b
        .set(&Tensor::from_vec(original, query_b.dims2().unwrap(), &Device::Cpu).unwrap())
        .unwrap();
    let numerical = (perturbed_losses[0] - perturbed_losses[1]) / (2. * epsilon);
    assert!(
        (numerical - derivative).abs() < 0.0002 + derivative.abs() * 0.03,
        "query adapter derivative disagrees: analytical={derivative}, numerical={numerical}"
    );
    let before = model
        .adapter_vars
        .iter()
        .map(|(name, tensor)| (name.clone(), vector(tensor)))
        .collect::<BTreeMap<_, _>>();
    let mut optimiser = crate::train::ResumableAdam::new(
        &model.adapter_vars,
        &crate::train::TrainConfig::default(),
    )
    .unwrap();
    let norm = optimiser.update(&model.adapter_vars, &loss).unwrap();
    assert!(norm.is_finite() && norm > 0.);
    assert!(
        model
            .adapter_vars
            .iter()
            .any(|(name, tensor)| before[name] != vector(tensor))
    );
    for (name, (tensor, original)) in frozen {
        assert_eq!(vector(&tensor), original, "base changed: {name}");
    }
}

#[test]
fn qwen3_requires_correct_head_norms_and_expanded_projection_shapes() {
    for name in [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
    ] {
        let mut absent = weights();
        absent.remove(name);
        assert!(
            QwenModel::from_weights(configuration(), absent, None, 17, None, false).is_err(),
            "accepted absent {name}"
        );
        let mut wrong = weights();
        wrong.insert(
            name.into(),
            Tensor::ones(32, DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(
            QwenModel::from_weights(configuration(), wrong, None, 17, None, false).is_err(),
            "accepted hidden-width {name}"
        );
    }
    for (name, shape) in [
        ("model.layers.0.self_attn.q_proj.weight", (32, 32)),
        ("model.layers.0.self_attn.o_proj.weight", (32, 32)),
    ] {
        let mut wrong = weights();
        wrong.insert(
            name.into(),
            Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(
            QwenModel::from_weights(configuration(), wrong, None, 17, None, false).is_err(),
            "accepted legacy projection {name}"
        );
    }
    for dimension in [0, 15, usize::MAX] {
        let mut cfg = configuration();
        cfg.head_dim = Some(dimension);
        assert!(
            cfg.validate().is_err(),
            "accepted invalid explicit head dimension {dimension}"
        );
    }
}

#[test]
fn legacy_configuration_omits_new_dimension_and_preserves_its_serialised_contract() {
    let mut legacy = configuration();
    legacy.model_type = "qwen2".into();
    legacy.head_dim = None;
    let encoded = serde_json::to_vec(&legacy).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert!(!value.as_object().unwrap().contains_key("head_dim"));
    let restored: QwenConfig = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.head_dimension(), 8);
    assert_eq!(encoded, serde_json::to_vec(&restored).unwrap());
    let original: QwenConfig = serde_json::from_str(include_str!(
        "../../docs/comparative-models/replacements/qwen3-4b-2507-base/config.json"
    ))
    .unwrap();
    original.validate().unwrap();
    assert_eq!(original.head_dimension(), 128);
    assert_eq!(
        original.num_attention_heads * original.head_dimension(),
        4096
    );
    assert_eq!(
        original.num_key_value_heads * original.head_dimension(),
        1024
    );
}

#[test]
fn original_qwen3_non_thinking_template_and_multiple_eos_are_preserved() {
    let configuration: serde_json::Value = serde_json::from_str(include_str!(
        "../../docs/comparative-models/replacements/qwen3-4b-2507-base/tokenizer_config.json"
    ))
    .unwrap();
    let actual = render_original_chat(
        configuration["chat_template"].as_str().unwrap(),
        &configuration,
        "fn add",
    )
    .unwrap();
    assert_eq!(
        actual,
        "<|im_start|>user\nfn add<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(configuration["add_bos_token"], false);
    let generation: serde_json::Value = serde_json::from_str(include_str!(
        "../../docs/comparative-models/replacements/qwen3-4b-2507-base/generation_config.json"
    ))
    .unwrap();
    assert_eq!(
        parse_eos_ids(&generation["eos_token_id"]).unwrap(),
        vec![151645, 151643]
    );
}
