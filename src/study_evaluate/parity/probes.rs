//! Reconstruct the fixed numerical criterion from full original observations.
use super::*;

pub(super) fn execution(report: &NumericalParityReport, plan: &NumericalParityPlan) -> Result<()> {
    let cpp = &report.cpp_execution;
    let rust = &report.rust_execution;
    ensure!(
        cpp["engine"] == "pinned_bitnet_cpp_supervised_by_rust"
            && cpp["architecture"] == "bitnet"
            && cpp["checkpoint_id"] == plan.checkpoint.checkpoint_id
            && cpp["engine_manifest"] == serde_json::to_value(&plan.cpp_engine_manifest)?
            && cpp["gguf"] == serde_json::to_value(&plan.gguf)?
            && cpp["actual_device"] == "Cpu"
            && cpp["actual_dtype"] == "F32"
            && cpp["cuda_active"] == false
            && cpp["key_cache_dtype"] == plan.cpp_kv_dtype
            && cpp["value_cache_dtype"] == plan.cpp_kv_dtype
            && cpp["inference_protocol"] == plan.cpp_protocol
            && cpp["kv_cache"] == true
            && cpp["original_tokeniser_parity_verified"] == true,
        "numerical parity native execution changed its engine/export/precision/protocol"
    );
    ensure!(
        rust["engine"] == "native_rust_candle"
            && rust["architecture"] == "bitnet"
            && rust["checkpoint_id"] == plan.checkpoint.checkpoint_id
            && rust["actual_device"] == "Cpu"
            && rust["actual_dtype"] == "F32"
            && rust["cuda_active"] == false
            && rust["weight_forward"] == "official_fixed_scale_ternary_effective_qat"
            && rust["activation_forward"] == "symmetric_per_token_int8_ties_even_ste",
        "numerical parity Rust execution changed its checkpoint or numerical forward"
    );
    Ok(())
}

fn native_protocol(
    native: &serde_json::Value,
    plan: &NumericalParityPlan,
    prompt: &NumericalParityPrompt,
) -> Result<()> {
    let settings = &native["generation_settings"];
    let model = Path::new(
        native["model"]
            .as_str()
            .context("native probe GGUF path absent")?,
    );
    let model_relative = if model.is_absolute() {
        model.strip_prefix(std::env::current_dir()?)?.to_owned()
    } else {
        model.to_owned()
    };
    ensure!(
        native["prompt"] == crate::bitnet::formatted_prompt(&prompt.prompt)
            && model_relative == plan.gguf.path
            && native["tokens_evaluated"].as_u64() == Some(prompt.input_ids.len() as u64)
            && native["timings"]["prompt_n"].as_u64() == Some(prompt.input_ids.len() as u64)
            && native["timings"]["cache_n"] == 0
            && native["tokens_predicted"] == 1
            && native["truncated"] == false
            && settings["temperature"].as_f64() == Some(0.)
            && settings["top_p"].as_f64() == Some(1.)
            && settings["seed"] == 1729
            && settings["n_predict"] == 1
            && settings["n_probs"] == NUMERICAL_PARITY_TOP_TOKENS
            && settings["post_sampling_probs"] == false
            && settings["stream"] == false
            && settings["repeat_penalty"].as_f64() == Some(1.)
            && settings["frequency_penalty"].as_f64() == Some(0.)
            && settings["presence_penalty"].as_f64() == Some(0.)
            && settings["grammar"] == ""
            && settings["logit_bias"].as_array().is_some_and(Vec::is_empty)
            && settings["lora"].as_array().is_some_and(Vec::is_empty),
        "native raw parity probe changed its prompt, GGUF, cache or pre-sampling protocol"
    );
    Ok(())
}

pub(super) fn compare(
    plan: &NumericalParityPlan,
    plan_sha256: &str,
    index: usize,
    vocabulary: usize,
    native: &serde_json::Value,
    rust: &RustLogProbabilityProbe,
) -> Result<serde_json::Value> {
    let prompt = plan
        .prompts
        .get(index)
        .context("undeclared numerical prompt")?;
    ensure!(
        rust.schema_version == 1
            && rust.evidence_kind == "actual_full_vocabulary_rust_next_token_log_probabilities"
            && rust.plan_sha256 == plan_sha256
            && rust.prompt_index == index
            && rust.input_ids == prompt.input_ids
            && vocabulary >= NUMERICAL_PARITY_TOP_TOKENS
            && rust.log_probabilities.len() == vocabulary
            && rust
                .log_probabilities
                .iter()
                .all(|p| p.is_finite() && *p <= 1e-5),
        "Rust raw parity probe is incomplete, non-finite or bound to another prompt/plan"
    );
    let probability_mass: f64 = rust.log_probabilities.iter().map(|p| p.exp()).sum();
    ensure!(
        (probability_mass - 1.).abs() <= 1e-4,
        "Rust full-vocabulary log probabilities are not normalised"
    );
    native_protocol(native, plan, prompt)?;
    let completions = native["completion_probabilities"]
        .as_array()
        .context("native pre-sampling probabilities absent")?;
    ensure!(
        completions.len() == 1,
        "native parity probe must contain exactly one next-token distribution"
    );
    let completion = &completions[0];
    let native_id = usize::try_from(
        completion["id"]
            .as_u64()
            .context("native top token absent")?,
    )?;
    let candidates = completion["top_logprobs"]
        .as_array()
        .context("native top log probabilities absent")?;
    ensure!(
        candidates.len() == NUMERICAL_PARITY_TOP_TOKENS,
        "native top-16 probability denominator differs"
    );
    let rust_top = rust
        .log_probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .context("Rust vocabulary empty")?
        .0;
    let mut seen = BTreeSet::new();
    let mut maximum = 0_f64;
    let mut previous = f64::INFINITY;
    let mut mass = 0_f64;
    let mut differences = Vec::new();
    for (position, candidate) in candidates.iter().enumerate() {
        let id = usize::try_from(
            candidate["id"]
                .as_u64()
                .context("native candidate token absent")?,
        )?;
        let logprob = candidate["logprob"]
            .as_f64()
            .context("native candidate log probability absent")?;
        ensure!(
            id < vocabulary
                && seen.insert(id)
                && logprob.is_finite()
                && logprob <= 1e-5
                && logprob <= previous
                && (position != 0 || id == native_id),
            "native top-16 distribution is duplicated, unsorted, non-finite or outside the original vocabulary"
        );
        if position == 0 {
            ensure!(
                completion["logprob"].as_f64() == Some(logprob),
                "native top probability differs from its distribution"
            );
        }
        previous = logprob;
        mass += logprob.exp();
        let reference = rust.log_probabilities[id];
        let error = (logprob - reference).abs();
        maximum = maximum.max(error);
        differences.push(serde_json::json!({"id":id,"native_log_probability":logprob,
            "rust_log_probability":reference,"absolute_error":error}));
    }
    ensure!(mass <= 1.0001, "native top-16 probability mass exceeds one");
    let tokens = native["tokens"]
        .as_array()
        .context("native output token IDs absent")?;
    ensure!(
        tokens.len() == 1 && tokens[0].as_u64() == Some(native_id as u64),
        "native greedy token differs from its top prediction"
    );
    let equal = rust_top == native_id;
    Ok(
        serde_json::json!({"prompt_index":index,"prompt":prompt.prompt,"input_tokens":prompt.input_ids.len(),
        "rust_top_token":rust_top,"native_top_token":native_id,"top_prediction_equal":equal,
        "maximum_absolute_log_probability_error":maximum,"native_top_tokens":differences,
        "full_rust_vocabulary":vocabulary,"full_rust_probability_mass":probability_mass,
        "passed":equal && maximum <= NUMERICAL_PARITY_MAXIMUM_ERROR}),
    )
}
