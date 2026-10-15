//! Numerical qualification is independent of descriptive task measurements.
//! A report's `passed` field never substitutes for its original model probes.
use super::*;

mod probes;
#[cfg(test)]
mod tests;

pub const NUMERICAL_PARITY_PROMPTS: [&str; 3] = [
    "Say hello in one short sentence.",
    "What is two plus three? Answer briefly.",
    "Complete the Rust expression: 3_i64 + 4_i64",
];
pub const NUMERICAL_PARITY_TOP_TOKENS: usize = 16;
pub const NUMERICAL_PARITY_MAXIMUM_ERROR: f64 = 0.10;
pub const NUMERICAL_PARITY_CPP_PROTOCOL: &str = "explicit_original_tokens_cpu_f32_kv_v2";
pub const NUMERICAL_PARITY_CPP_QUANT_PROTOCOL: &str =
    "explicit_original_tokens_cpu_f32_kv_quant_v3";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericalParityPrompt {
    pub prompt: String,
    pub input_ids: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericalParityPlan {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub checkpoint: Checkpoint,
    pub gguf: Artefact,
    pub export_receipt: Artefact,
    pub cpp_engine_manifest: Artefact,
    pub cpp_protocol: String,
    pub cpp_kv_dtype: String,
    pub diagnostic_executable: Artefact,
    pub rust_implementation_sha256: String,
    /// Original source names map to archived, checksum-bound source copies.
    pub implementation_sources: BTreeMap<String, Artefact>,
    pub prompts: Vec<NumericalParityPrompt>,
    pub native_top_tokens: usize,
    pub maximum_absolute_log_probability_error: f64,
    pub require_equal_top_prediction: bool,
    pub production_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RustLogProbabilityProbe {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub plan_sha256: String,
    pub prompt_index: usize,
    pub input_ids: Vec<u32>,
    /// Full original vocabulary; a retained top-k subset cannot qualify.
    pub log_probabilities: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericalParityObservation {
    pub prompt_index: usize,
    pub native_response: Artefact,
    pub rust_logprobs: Artefact,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericalParityReport {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub plan: Artefact,
    pub observations: Vec<NumericalParityObservation>,
    pub cpp_execution: serde_json::Value,
    pub rust_execution: serde_json::Value,
    pub comparison_completed: bool,
    pub passed: bool,
    pub production_eligible: bool,
}

/// Absence and invalidation are explicit exclusions, while preserving scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericalEquivalence {
    pub required: bool,
    pub qualified: bool,
    pub status: String,
    pub reasons: Vec<String>,
    pub attachment: Option<Artefact>,
    pub report: Option<Artefact>,
    pub reconstructed: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Attachment {
    schema_version: u32,
    evidence_kind: String,
    evaluation_plan: Artefact,
    report: Artefact,
    production_eligible: bool,
}

/// This identity deliberately becomes stale when the Rust numerical contract
/// changes. Archived measurements remain descriptive and are never rewritten.
pub fn numerical_parity_implementation() -> BTreeMap<String, String> {
    [
        ("Cargo.toml", include_bytes!("../../Cargo.toml").as_slice()),
        ("Cargo.lock", include_bytes!("../../Cargo.lock").as_slice()),
        ("src/bitnet.rs", include_bytes!("../bitnet.rs").as_slice()),
        (
            "src/bitnet/codec.rs",
            include_bytes!("../bitnet/codec.rs").as_slice(),
        ),
        (
            "src/bitnet/quant.rs",
            include_bytes!("../bitnet/quant.rs").as_slice(),
        ),
        (
            "src/bitnet/runtime.rs",
            include_bytes!("../bitnet/runtime.rs").as_slice(),
        ),
        (
            "src/bitnet/export.rs",
            include_bytes!("../bitnet/export.rs").as_slice(),
        ),
        (
            "src/bitnet/cpp.rs",
            include_bytes!("../bitnet/cpp.rs").as_slice(),
        ),
        (
            "src/bitnet/cpp/host.rs",
            include_bytes!("../bitnet/cpp/host.rs").as_slice(),
        ),
        (
            "src/bitnet/cpp/quantisation_v2.rs",
            include_bytes!("../bitnet/cpp/quantisation_v2.rs").as_slice(),
        ),
        (
            "src/bitnet/backend.rs",
            include_bytes!("../bitnet/backend.rs").as_slice(),
        ),
        (
            "src/bitnet/parity_run.rs",
            include_bytes!("../bitnet/parity_run.rs").as_slice(),
        ),
        (
            "docs/bitnet/engine-quantisation-v2.patch",
            include_bytes!("../../docs/bitnet/engine-quantisation-v2.patch").as_slice(),
        ),
        ("src/model.rs", include_bytes!("../model.rs").as_slice()),
        (
            "examples/bitnet_parity.rs",
            include_bytes!("../../examples/bitnet_parity.rs").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(path, bytes)| (path.to_owned(), artefact::hash(bytes)))
    .collect()
}

pub fn numerical_parity_implementation_sha256() -> Result<String> {
    Ok(artefact::hash(&serde_json::to_vec(
        &numerical_parity_implementation(),
    )?))
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(root, path.canonicalize()?.strip_prefix(root)?)
}

fn read_json<T: serde::de::DeserializeOwned>(file: &Artefact, root: &Path) -> Result<T> {
    ensure!(
        file.bytes <= 32 << 20,
        "numerical JSON artefact exceeds bound"
    );
    Ok(serde_json::from_slice(&fs::read(file.verify(root)?)?)?)
}

fn validate_plan(plan: &NumericalParityPlan, root: &Path) -> Result<usize> {
    ensure!(
        plan.schema_version == 2
            && plan.evidence_kind == "frozen_original_bitnet_rust_i2s_numerical_parity"
            && plan.checkpoint.architecture == StudentArchitecture::TernaryBitnet
            && plan.native_top_tokens == NUMERICAL_PARITY_TOP_TOKENS
            && plan.maximum_absolute_log_probability_error == NUMERICAL_PARITY_MAXIMUM_ERROR
            && plan.require_equal_top_prediction
            && plan.cpp_kv_dtype == "F32"
            && !plan.production_eligible,
        "numerical parity protocol differs from fixed schema-2 criteria"
    );
    // Numerical qualification hashes the original weights and verifies the
    // actual training manifest, without allocating a second full model merely
    // to compare retained log probabilities. Task verification remains separate.
    let checkpoint = &plan.checkpoint;
    ensure!(
        checkpoint.checkpoint_id == checkpoint.tensor.sha256,
        "numerical parity checkpoint identifier differs from its tensor"
    );
    let base_path = checkpoint.base_assets.verify(root)?;
    let assets = crate::hub::validate_bitnet_student_assets(
        base_path
            .parent()
            .context("numerical parity base directory absent")?,
    )?;
    checkpoint.tensor.verify(root)?;
    let checkpoint_manifest = checkpoint.manifest.verify(root)?;
    ensure!(
        assets.files.get("tokenizer.json") == Some(&checkpoint.tokenizer.sha256),
        "numerical parity tokeniser differs from original base"
    );
    if assets.files.get("model.safetensors") == Some(&checkpoint.checkpoint_id) {
        ensure!(
            same(&checkpoint.manifest, &checkpoint.base_assets)?,
            "numerical parity baseline includes an adapter manifest"
        );
    } else {
        let manifest = crate::distillation::checkpoint::validate_checkpoint(
            checkpoint_manifest
                .parent()
                .context("numerical parity adapter directory absent")?,
            &assets,
        )?;
        ensure!(
            manifest.adapter_sha256 == checkpoint.checkpoint_id
                && manifest.spec.student_kind
                    == crate::distillation::DistillationStudent::TernaryBitnet
                && !manifest.production_eligible,
            "numerical parity checkpoint manifest differs from actual ternary adapter"
        );
        crate::teacher_evidence::load_verified(&manifest.evidence_bundle.path)?;
    }
    let executable = plan.diagnostic_executable.verify(root)?;
    crate::engine::verified_snapshot(
        &executable,
        &plan.diagnostic_executable.sha256,
        plan.diagnostic_executable.bytes,
    )?;
    let expected = numerical_parity_implementation();
    ensure!(
        plan.rust_implementation_sha256 == numerical_parity_implementation_sha256()?
            && plan.implementation_sources.keys().eq(expected.keys()),
        "numerical parity Rust implementation is absent or stale"
    );
    for (name, file) in &plan.implementation_sources {
        file.verify(root)?;
        ensure!(
            expected.get(name) == Some(&file.sha256),
            "numerical parity archived implementation differs: {name}"
        );
    }
    let engine_path = plan.cpp_engine_manifest.verify(root)?;
    crate::bitnet::cpp::verify_engine(&engine_path)?;
    let engine: serde_json::Value = read_json(&plan.cpp_engine_manifest, root)?;
    let protocol = match engine["schema_version"].as_u64() {
        Some(1) => NUMERICAL_PARITY_CPP_PROTOCOL,
        Some(2) => NUMERICAL_PARITY_CPP_QUANT_PROTOCOL,
        _ => anyhow::bail!("numerical parity uses an unsupported native engine schema"),
    };
    ensure!(
        plan.cpp_protocol == protocol,
        "numerical parity native protocol differs from pinned engine"
    );
    plan.gguf.verify(root)?;
    let export: serde_json::Value = read_json(&plan.export_receipt, root)?;
    ensure!(
        export["schema_version"] == 1
            && export["evidence_kind"] == "native_official_bitnet_qat_gguf_export"
            && export["gguf"] == serde_json::to_value(&plan.gguf)?
            && export["production_eligible"] == false
            && export["binding"]["codec"] == "native_hf_axis_to_i2s_v1"
            && export["binding"]["implementation_sha256"]
                == plan.implementation_sources["src/bitnet/export.rs"].sha256,
        "numerical parity GGUF/export implementation differs"
    );
    let base: serde_json::Value = read_json(&plan.checkpoint.base_assets, root)?;
    ensure!(
        export["binding"]["source"] == base,
        "numerical parity export used another base"
    );
    if same(&plan.checkpoint.manifest, &plan.checkpoint.base_assets)? {
        ensure!(
            export["binding"]["adapter"].is_null(),
            "base numerical parity includes an adapter"
        );
    } else {
        ensure!(
            export["binding"]["adapter"] == serde_json::to_value(&plan.checkpoint.tensor)?
                && export["changed_effective_trits"]
                    .as_u64()
                    .is_some_and(|n| n > 0),
            "numerical parity export omits this adapter's effective changes"
        );
    }
    let tokenizer = tokenizers::Tokenizer::from_file(plan.checkpoint.tokenizer.verify(root)?)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    ensure!(
        plan.prompts.len() == NUMERICAL_PARITY_PROMPTS.len(),
        "numerical parity prompt denominator differs"
    );
    for (prompt, expected) in plan.prompts.iter().zip(NUMERICAL_PARITY_PROMPTS) {
        let ids = tokenizer
            .encode(crate::bitnet::formatted_prompt(expected), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ensure!(
            prompt.prompt == expected
                && prompt.input_ids == ids.get_ids()
                && !prompt.input_ids.is_empty(),
            "numerical parity prompt text or original token IDs differ"
        );
    }
    let config: serde_json::Value = serde_json::from_slice(&fs::read(
        plan.checkpoint
            .base_assets
            .verify(root)?
            .parent()
            .context("base directory missing")?
            .join("config.json"),
    )?)?;
    usize::try_from(
        config["vocab_size"]
            .as_u64()
            .context("BitNet vocabulary size missing")?,
    )
    .context("BitNet vocabulary size overflows")
}

fn reconstruct(report: &NumericalParityReport, root: &Path) -> Result<serde_json::Value> {
    ensure!(
        report.schema_version == 2
            && report.evidence_kind == "actual_original_bitnet_rust_i2s_numerical_parity"
            && report.comparison_completed
            && !report.production_eligible,
        "numerical parity requires a complete schema-2 report"
    );
    let plan: NumericalParityPlan = read_json(&report.plan, root)?;
    let vocabulary = validate_plan(&plan, root)?;
    probes::execution(report, &plan)?;
    ensure!(
        report.observations.len() == NUMERICAL_PARITY_PROMPTS.len(),
        "numerical parity observations are missing"
    );
    let mut observations = Vec::new();
    for (index, observation) in report.observations.iter().enumerate() {
        ensure!(
            observation.prompt_index == index,
            "numerical parity observations reordered or repeated"
        );
        let native: serde_json::Value = read_json(&observation.native_response, root)?;
        let rust: RustLogProbabilityProbe = read_json(&observation.rust_logprobs, root)?;
        observations.push(probes::compare(
            &plan,
            &report.plan.sha256,
            index,
            vocabulary,
            &native,
            &rust,
        )?);
    }
    let passed = observations.iter().all(|row| row["passed"] == true);
    ensure!(
        report.passed == passed,
        "numerical parity claimed result differs from raw probes"
    );
    Ok(
        serde_json::json!({"schema_version":2,"evidence_kind":"independently_reconstructed_bitnet_numerical_equivalence",
        "passed":passed,"plan":report.plan,"checkpoint_id":plan.checkpoint.checkpoint_id,"observations":observations,
        "maximum_absolute_log_probability_error":NUMERICAL_PARITY_MAXIMUM_ERROR,"native_top_tokens":NUMERICAL_PARITY_TOP_TOKENS,
        "quality_evaluation":false,"production_eligible":false}),
    )
}

pub fn verify_bitnet_parity_report(path: &Path) -> Result<serde_json::Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    let report: NumericalParityReport = read_json(&capture(&root, path)?, &root)?;
    reconstruct(&report, &root)
}

fn bound_report(evaluation: &Plan, report: &NumericalParityReport, root: &Path) -> Result<()> {
    let plan: NumericalParityPlan = read_json(&report.plan, root)?;
    ensure!(
        same(&evaluation.checkpoint, &plan.checkpoint)?,
        "numerical parity belongs to another checkpoint/base/tokeniser"
    );
    for file in [&plan.gguf, &plan.export_receipt, &plan.cpp_engine_manifest] {
        ensure!(
            evaluation
                .backend_artefacts
                .iter()
                .any(|actual| same(actual, file).unwrap_or(false)),
            "numerical parity backend identity differs from frozen evaluation"
        );
    }
    Ok(())
}

/// Attaching a failed measurement is useful evidence; it cannot qualify it.
pub fn attach_numerical_equivalence(
    evaluation_dir: &Path,
    report_path: &Path,
) -> Result<serde_json::Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    let evaluation: Plan = serde_json::from_slice(&fs::read(evaluation_dir.join("plan.json"))?)?;
    ensure!(
        evaluation.checkpoint.architecture == StudentArchitecture::TernaryBitnet,
        "numerical attachment is specific to BitNet"
    );
    let report_file = capture(&root, report_path)?;
    let report: NumericalParityReport = read_json(&report_file, &root)?;
    let reconstructed = reconstruct(&report, &root)?;
    bound_report(&evaluation, &report, &root)?;
    let attachment = Attachment {
        schema_version: 1,
        evidence_kind: "bound_bitnet_numerical_equivalence_attachment".into(),
        evaluation_plan: capture(&root, &evaluation_dir.join("plan.json"))?,
        report: report_file,
        production_eligible: false,
    };
    let path = evaluation_dir.join("numerical-equivalence.json");
    if path.exists() {
        let old: Attachment = serde_json::from_slice(&fs::read(&path)?)?;
        ensure!(
            same(&old, &attachment)?,
            "numerical equivalence attachment is already frozen"
        );
    } else {
        artefact::write_json(&path, &attachment)?;
    }
    Ok(reconstructed)
}

/// Missing, failed and stale numerical receipts exclude a BitNet score from
/// ranking without erasing its observed descriptive task outcomes.
pub fn verify_numerical_equivalence(evaluation_dir: &Path) -> Result<NumericalEquivalence> {
    let root = std::env::current_dir()?.canonicalize()?;
    let evaluation: Plan = serde_json::from_slice(&fs::read(evaluation_dir.join("plan.json"))?)?;
    let required = evaluation.checkpoint.architecture == StudentArchitecture::TernaryBitnet;
    let mut result = NumericalEquivalence {
        required,
        qualified: !required,
        status: if required {
            "unavailable"
        } else {
            "not_required"
        }
        .into(),
        reasons: Vec::new(),
        attachment: None,
        report: None,
        reconstructed: None,
    };
    if !required {
        return Ok(result);
    }
    let path = evaluation_dir.join("numerical-equivalence.json");
    let verification = (|| -> Result<serde_json::Value> {
        ensure!(
            path.exists(),
            "schema-2 numerical equivalence receipt is missing"
        );
        result.attachment = Some(capture(&root, &path)?);
        let attached: Attachment = serde_json::from_slice(&fs::read(&path)?)?;
        ensure!(
            attached.schema_version == 1
                && attached.evidence_kind == "bound_bitnet_numerical_equivalence_attachment"
                && !attached.production_eligible
                && same(
                    &attached.evaluation_plan,
                    &capture(&root, &evaluation_dir.join("plan.json"))?
                )?,
            "numerical equivalence attachment changed its frozen evaluation"
        );
        result.report = Some(attached.report.clone());
        let report: NumericalParityReport = read_json(&attached.report, &root)?;
        bound_report(&evaluation, &report, &root)?;
        let loaded: serde_json::Value =
            serde_json::from_slice(&fs::read(evaluation_dir.join("execution.json"))?)?;
        let protocol: NumericalParityPlan = read_json(&report.plan, &root)?;
        ensure!(
            loaded["inference_protocol"] == protocol.cpp_protocol
                && loaded["key_cache_dtype"] == protocol.cpp_kv_dtype
                && loaded["value_cache_dtype"] == protocol.cpp_kv_dtype,
            "numerical equivalence inference/KV protocol differs from evaluation"
        );
        backend::execution(&loaded, &evaluation, &root)?;
        reconstruct(&report, &root)
    })();
    match verification {
        Ok(reconstructed) => {
            result.qualified = reconstructed["passed"] == true;
            result.status = if result.qualified {
                "qualified"
            } else {
                "failed"
            }
            .into();
            if !result.qualified {
                result
                    .reasons
                    .push("raw Rust/C++ probes fail the fixed top-token/0.10 criterion".into());
            }
            result.reconstructed = Some(reconstructed);
        }
        Err(error) => result.reasons.push(format!("{error:#}")),
    }
    Ok(result)
}
