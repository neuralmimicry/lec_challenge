//! Small random tensors exercise architecture/QAT invariants; no assessment claim.
use super::*;

fn fixture() -> (BitNetConfig, HashMap<String, Tensor>, BTreeMap<String, f64>) {
    let cfg = BitNetConfig {
        model_type: "bitnet".into(),
        hidden_act: "relu2".into(),
        vocab_size: 16,
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_position_embeddings: 32,
        rope_theta: 500000.,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
    };
    let mut weights = HashMap::new();
    let mut scales = BTreeMap::new();
    weights.insert(
        "model.embed_tokens.weight".into(),
        Tensor::from_vec(
            (0..128)
                .map(|i| ((i as f32 + 0.4) * 0.73).sin() * 0.2)
                .collect::<Vec<_>>(),
            (16, 8),
            &Device::Cpu,
        )
        .unwrap(),
    );
    for (name, size) in [
        ("model.norm.weight", 8),
        ("model.layers.0.input_layernorm.weight", 8),
        ("model.layers.0.post_attention_layernorm.weight", 8),
        ("model.layers.0.self_attn.attn_sub_norm.weight", 8),
        ("model.layers.0.mlp.ffn_sub_norm.weight", 12),
    ] {
        weights.insert(
            name.into(),
            Tensor::ones(size, DType::F32, &Device::Cpu).unwrap(),
        );
    }
    for (index, (name, out, input)) in [
        ("self_attn.q_proj", 8, 8),
        ("self_attn.k_proj", 4, 8),
        ("self_attn.v_proj", 4, 8),
        ("self_attn.o_proj", 8, 8),
        ("mlp.gate_proj", 12, 8),
        ("mlp.up_proj", 12, 8),
        ("mlp.down_proj", 8, 12),
    ]
    .into_iter()
    .enumerate()
    {
        let prefix = format!("model.layers.0.{name}");
        weights.insert(
            format!("{prefix}.weight"),
            Tensor::from_vec(
                (0..out * input)
                    .map(|i| (((i * 13 + index * 7) % 3) as f32 - 1.) * 0.25)
                    .collect::<Vec<_>>(),
                (out, input),
                &Device::Cpu,
            )
            .unwrap(),
        );
        scales.insert(prefix, 0.25);
    }
    (cfg, weights, scales)
}

#[test]
fn real_qat_graph_changes_reloads_and_retains_ternary_forward() -> Result<()> {
    let (cfg, weights, scales) = fixture();
    let frozen = weights
        .iter()
        .map(|(n, t)| {
            (
                n.clone(),
                t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut model = BitNetStudent::from_weights(
        cfg.clone(),
        weights.clone(),
        scales.clone(),
        Some(AdapterConfig { rank: 2, alpha: 4. }),
        31,
        None,
    )?;
    let tokens = Tensor::new(&[[1u32, 4, 7, 9]], &Device::Cpu)?;
    let before = model.forward(&tokens)?;
    let hidden = model.hidden(&tokens, 0, false)?;
    let selected = hidden.squeeze(0)?.narrow(0, 1, 2)?;
    assert_eq!(model.logits(&selected)?.dims(), [2, 16]);
    assert_eq!(
        model.logits(&selected)?.flatten_all()?.to_vec1::<f32>()?,
        before
            .squeeze(0)?
            .narrow(0, 1, 2)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    let loss = before.sqr()?.mean_all()?;
    assert!(loss.to_scalar::<f32>()?.is_finite());
    let gradients = loss.backward()?;
    let mut nonzero = 0.;
    for variable in model.adapter_vars.values() {
        if let Some(gradient) = gradients.get(variable.as_tensor()) {
            nonzero += gradient.abs()?.sum_all()?.to_scalar::<f32>()?;
            variable.set(&(variable.as_tensor() - (gradient * 2.)?)?)?;
        }
    }
    assert!(nonzero > 0., "actual QAT adapter gradients must exist");
    for (name, tensor) in &model.base_weights {
        assert_eq!(
            tensor.flatten_all()?.to_vec1::<f32>()?,
            frozen[name],
            "frozen BitNet base changed"
        );
    }
    for (name, tensor) in model.effective_weights()? {
        if name.contains("q_proj") || name.contains("v_proj") {
            assert!(
                tensor
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|x| [-0.25, 0., 0.25].contains(x)),
                "effective adapted projection escaped ternary lattice"
            );
        }
    }
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("adapter.safetensors");
    model.save_adapters(&path)?;
    let after = model.forward(&tokens)?.flatten_all()?.to_vec1::<f32>()?;
    let saved = candle_core::safetensors::load(&path, &Device::Cpu)?;
    let mut reloaded = BitNetStudent::from_weights(
        cfg,
        weights,
        scales,
        Some(AdapterConfig { rank: 2, alpha: 4. }),
        999,
        Some(saved),
    )?;
    assert_eq!(
        after,
        reloaded.forward(&tokens)?.flatten_all()?.to_vec1::<f32>()?
    );
    assert!(
        BitNetStudent::from_weights(
            fixture().0,
            fixture().1,
            fixture().2,
            None,
            1,
            Some(HashMap::new())
        )
        .is_ok()
    );
    Ok(())
}
