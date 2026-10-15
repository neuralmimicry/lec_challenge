//! Numerical operator QA on a tiny synthetic BitNet-shaped model.
//! These tensors are explicitly not pretrained weights or assessment samples.
use super::*;
use crate::artefact;

type MicroWeights = (BitNetConfig, HashMap<String, Tensor>, BTreeMap<String, f64>);
fn fixture() -> Result<MicroWeights> {
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
        )?,
    );
    for (name, size) in [
        ("model.norm.weight", 8),
        ("model.layers.0.input_layernorm.weight", 8),
        ("model.layers.0.post_attention_layernorm.weight", 8),
        ("model.layers.0.self_attn.attn_sub_norm.weight", 8),
        ("model.layers.0.mlp.ffn_sub_norm.weight", 12),
    ] {
        weights.insert(name.into(), Tensor::ones(size, DType::F32, &Device::Cpu)?);
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
            )?,
        );
        scales.insert(prefix, 0.25);
    }
    Ok((cfg, weights, scales))
}

pub fn run_operator_qa(output: &Path) -> Result<serde_json::Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    let report_path = output.join("operator-qa.json");
    if report_path.exists() {
        let old: serde_json::Value = serde_json::from_slice(&fs::read(&report_path)?)?;
        ensure!(
            old["evidence_kind"] == "synthetic_micro_bitnet_qat_numerical_qa"
                && old["genuine_ternary_training"] == true
                && old["changed_effective_weights"] == true
                && old["reloaded"] == true,
            "saved operator QA is incomplete or has another purpose"
        );
        let adapter: artefact::Artefact = serde_json::from_value(old["adapter"].clone())?;
        adapter.verify(&root)?;
        return Ok(old);
    }
    fs::create_dir_all(output)?;
    let (cfg, weights, scales) = fixture()?;
    let config = AdapterConfig { rank: 2, alpha: 4. };
    let mut model = BitNetStudent::from_weights(
        cfg.clone(),
        weights.clone(),
        scales.clone(),
        Some(config.clone()),
        31,
        None,
    )?;
    let before = model.effective_trit_identity()?;
    let input = Tensor::new(&[[1u32, 4, 7, 9]], &Device::Cpu)?;
    let mut losses = Vec::new();
    let mut norms = Vec::new();
    // Normalised SGD has a predeclared step length. It deliberately crosses
    // quantisation thresholds in this micro test; this is not a useful
    // full-size model hyperparameter or a claim of task-quality improvement.
    for _ in 0..8 {
        let loss = model.forward(&input)?.sqr()?.mean_all()?;
        let value = loss.to_scalar::<f32>()?;
        ensure!(value.is_finite(), "micro QAT loss is non-finite");
        let grads = loss.backward()?;
        let mut norm = 0_f64;
        for variable in model.adapter_vars.values() {
            if let Some(g) = grads.get(variable.as_tensor()) {
                norm += f64::from(g.sqr()?.sum_all()?.to_scalar::<f32>()?);
            }
        }
        norm = norm.sqrt();
        ensure!(norm.is_finite() && norm > 0., "micro QAT gradient absent");
        for variable in model.adapter_vars.values() {
            if let Some(g) = grads.get(variable.as_tensor()) {
                variable.set(&(variable.as_tensor() - (g * (32. / norm))?)?)?;
            }
        }
        losses.push(value);
        norms.push(norm);
    }
    let after = model.effective_trit_identity()?;
    ensure!(
        before.sha256 != after.sha256 && after.changed_from_frozen_base > 0,
        "only latent adapter values changed; effective ternary student is unchanged"
    );
    let adapter_path = output.join("micro-adapter.safetensors");
    model.save_adapters(&adapter_path)?;
    let adapter = artefact::Artefact::capture(&root, &adapter_path)?;
    let effective = model.effective_weights()?;
    let mut exact_trits = true;
    for (name, tensor) in effective
        .iter()
        .filter(|(n, _)| n.contains("q_proj") || n.contains("v_proj"))
    {
        let (rows, columns) = tensor.dims2()?;
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let packed = codec::pack_rows(&values, rows, columns, 0.25)?;
        let decoded = codec::unpack_rows(&packed, rows, columns)?
            .into_iter()
            .map(|x| x * 0.25)
            .collect::<Vec<_>>();
        exact_trits &= values == decoded;
        let path = output.join(format!("{}.trits", name.replace('.', "_")));
        artefact::atomic_write(&path, &packed)?;
    }
    ensure!(exact_trits, "micro QAT trit codec is not lossless");
    let expected = model.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
    let loaded = candle_core::safetensors::load(&adapter_path, &Device::Cpu)?;
    let mut reloaded =
        BitNetStudent::from_weights(cfg, weights, scales, Some(config), 999, Some(loaded))?;
    let reloaded_identity = reloaded.effective_trit_identity()?;
    let reload_exact = expected == reloaded.forward(&input)?.flatten_all()?.to_vec1::<f32>()?
        && after.sha256 == reloaded_identity.sha256;
    ensure!(reload_exact, "micro QAT fresh instance reload differs");
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"synthetic_micro_bitnet_qat_numerical_qa","synthetic_tensors":true,"official_full_base":false,"assessment_training":false,"genuine_ternary_training":true,"changed_effective_weights":true,"reloaded":reload_exact,"trit_codec_exact":exact_trits,"initial_effective":before,"final_effective":after,"losses":losses,"gradient_norms":norms,"steps":8,"optimiser":"normalised_sgd_step_length32","adapter":adapter,"production_eligible":false,"limitations":["Numerical QA on tiny synthetic tensors; no provider outputs, pretrained-model adaptation, task learning or quality gain established.","Full-size official BitNet training requires a separate real evidence-bound worker run, resource measurement and changed effective trits."]});
    artefact::write_json(&report_path, &report)?;
    Ok(report)
}
