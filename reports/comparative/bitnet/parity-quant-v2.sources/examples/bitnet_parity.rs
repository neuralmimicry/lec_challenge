//! Freeze a numerical protocol before measured forwards and retain both raw
//! backends. A different executable cannot silently resume an unfinished probe.
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use clap::Parser;
use lec_distil::{
    artefact::{self, Artefact},
    bitnet::{self, CppRuntimeOptions},
    experiments::StudentArchitecture,
    study_evaluate::{Checkpoint, parity::*},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
    base: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "reports/comparative/bitnet/parity.json")]
    output: PathBuf,
    #[arg(long)]
    engine_manifest: Option<PathBuf>,
    /// A fresh export path preserves earlier codec implementation receipts.
    #[arg(long)]
    gguf: Option<PathBuf>,
    /// Explicit cached-only build of the separately versioned rounding fix.
    #[arg(long)]
    quantisation_v2: bool,
    /// Bind this report explicitly to a frozen checkpoint evaluation.
    #[arg(long)]
    evaluation: Vec<PathBuf>,
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(root, path.canonicalize()?.strip_prefix(root)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = std::env::current_dir()?.canonicalize()?;
    let current = std::env::current_exe()?.canonicalize()?;
    let pinned = lec_distil::engine::pin_executable(&current, &root.join("state/engines"))?;
    if current != pinned.path {
        // The measured process itself runs the immutable copy. Merely hashing
        // Cargo's replaceable output would not establish this binding.
        let status = std::process::Command::new(&pinned.path)
            .args(std::env::args_os().skip(1))
            .status()?;
        ensure!(
            status.success(),
            "pinned numerical diagnostic failed with {status}"
        );
        return Ok(());
    }
    let output = args.output.clone();
    let result = run(args, &root, capture(&root, &pinned.path)?);
    if let Err(error) = &result
        && !output.exists()
    {
        artefact::write_json(
            &output.with_extension("failure.json"),
            &serde_json::json!({
                "schema_version":2,"evidence_kind":"incomplete_bitnet_rust_i2s_numerical_parity",
                "passed":false,"comparison_completed":false,"failure":format!("{error:#}"),
                "quality_evaluation":false,"production_eligible":false
            }),
        )?;
    }
    result
}

fn checkpoint(args: &Args, root: &Path) -> Result<Checkpoint> {
    let assets = lec_distil::hub::validate_bitnet_student_assets(&args.base)?;
    let base_assets = capture(root, &args.base.join("assets.json"))?;
    let (checkpoint_id, tensor, manifest) = if let Some(path) = &args.checkpoint {
        let proof = lec_distil::distillation::checkpoint::validate_checkpoint(path, &assets)?;
        ensure!(
            proof.spec.student_kind == lec_distil::distillation::DistillationStudent::TernaryBitnet,
            "numerical diagnostic checkpoint is not the native ternary student"
        );
        (
            proof.adapter_sha256,
            capture(root, &path.join("adapter.safetensors"))?,
            capture(root, &path.join("manifest.json"))?,
        )
    } else {
        (
            assets.files["model.safetensors"].clone(),
            capture(root, &args.base.join("model.safetensors"))?,
            base_assets.clone(),
        )
    };
    Ok(Checkpoint {
        checkpoint_id,
        tensor,
        manifest,
        base_assets,
        tokenizer: capture(root, &args.base.join("tokenizer.json"))?,
        architecture: StudentArchitecture::TernaryBitnet,
    })
}

fn run(args: Args, root: &Path, diagnostic_executable: Artefact) -> Result<()> {
    if args.output.exists() {
        let verified = verify_bitnet_parity_report(&args.output)?;
        for directory in &args.evaluation {
            attach_numerical_equivalence(directory, &args.output)?;
        }
        println!("{}", serde_json::to_string_pretty(&verified)?);
        ensure!(
            verified["passed"] == true,
            "retained numerical comparison failed; no probes were repeated"
        );
        return Ok(());
    }
    if args.quantisation_v2 {
        let runtime = tokio::runtime::Runtime::new()?;
        runtime.block_on(bitnet::cpp::bootstrap_quantisation_v2(true, 2))?;
        drop(runtime);
    }
    let engine_path = args.engine_manifest.clone().unwrap_or_else(|| {
        PathBuf::from(if args.quantisation_v2 {
            "state/engines/bitnet-quant-v2-build/engine-manifest.json"
        } else {
            "state/engines/bitnet-build/engine-manifest.json"
        })
    });
    let engine = bitnet::cpp::verify_engine(&engine_path)?;
    let gguf = args.gguf.clone().unwrap_or_else(|| {
        args.checkpoint
            .as_ref()
            .map(|p| p.join("model.i2_s.gguf"))
            .unwrap_or_else(|| args.base.join("model.i2_s.gguf"))
    });
    let export = bitnet::export::export(&args.base, args.checkpoint.as_deref(), &gguf)?;
    let tokenizer = tokenizers::Tokenizer::from_file(args.base.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let prompts = NUMERICAL_PARITY_PROMPTS
        .iter()
        .map(|text| {
            let input_ids = tokenizer
                .encode(bitnet::formatted_prompt(text), false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();
            Ok(NumericalParityPrompt {
                prompt: (*text).to_owned(),
                input_ids,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let source_directory = args.output.with_extension("sources");
    let mut implementation_sources = BTreeMap::new();
    for (name, expected) in numerical_parity_implementation() {
        let bytes = fs::read(root.join(&name))?;
        ensure!(
            artefact::hash(&bytes) == expected,
            "source {name} differs from compiled diagnostic; rebuild or resume the original pinned executable"
        );
        let saved = source_directory.join(&name);
        if saved.exists() {
            ensure!(
                fs::read(&saved)? == bytes,
                "saved numerical implementation source changed"
            );
        } else {
            artefact::atomic_write(&saved, &bytes)?;
        }
        implementation_sources.insert(name, capture(root, &saved)?);
    }
    let plan = NumericalParityPlan {
        schema_version: 2,
        evidence_kind: "frozen_original_bitnet_rust_i2s_numerical_parity".into(),
        checkpoint: checkpoint(&args, root)?,
        gguf: serde_json::from_value(export["gguf"].clone())?,
        export_receipt: capture(root, &gguf.with_extension("export.json"))?,
        cpp_engine_manifest: capture(root, &engine_path)?,
        cpp_protocol: if engine.schema_version == 2 {
            NUMERICAL_PARITY_CPP_QUANT_PROTOCOL
        } else {
            NUMERICAL_PARITY_CPP_PROTOCOL
        }
        .into(),
        cpp_kv_dtype: "F32".into(),
        diagnostic_executable,
        rust_implementation_sha256: numerical_parity_implementation_sha256()?,
        implementation_sources,
        prompts,
        native_top_tokens: NUMERICAL_PARITY_TOP_TOKENS,
        maximum_absolute_log_probability_error: NUMERICAL_PARITY_MAXIMUM_ERROR,
        require_equal_top_prediction: true,
        production_eligible: false,
    };
    let plan_path = args.output.with_extension("plan.json");
    if plan_path.exists() {
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&plan_path)?)?
                == serde_json::to_value(&plan)?,
            "saved numerical protocol differs; use its original pinned executable or a fresh output"
        );
    } else {
        artefact::write_json(&plan_path, &plan)?;
    }
    let plan_file = capture(root, &plan_path)?;
    // All actual model forwards happen only after the complete plan is durable.
    let mut cpp = bitnet::BitNetCppRuntime::load_with_options(
        &args.base,
        args.checkpoint.as_deref(),
        CppRuntimeOptions {
            engine_manifest: engine_path,
            gguf,
            ..Default::default()
        },
    )?;
    let cpp_execution = cpp.execution_metadata();
    let mut rust = if let Some(path) = &args.checkpoint {
        bitnet::BitNetRuntime::load(&args.base, path)?
    } else {
        bitnet::BitNetRuntime::load_base(&args.base)?
    };
    let mut observations = Vec::new();
    let mut passed = true;
    for (index, prompt) in plan.prompts.iter().enumerate() {
        let native_path = args.output.with_extension(format!("probe-{index}.json"));
        let native: serde_json::Value = if native_path.exists() {
            serde_json::from_slice(&fs::read(&native_path)?)?
        } else {
            let raw = cpp.next_token_probe(&prompt.prompt, NUMERICAL_PARITY_TOP_TOKENS)?;
            artefact::write_json(&native_path, &raw)?;
            raw
        };
        let rust_path = args.output.with_extension(format!("rust-{index}.json"));
        let reference: RustLogProbabilityProbe = if rust_path.exists() {
            serde_json::from_slice(&fs::read(&rust_path)?)?
        } else {
            let ids = &prompt.input_ids;
            let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let hidden = rust
                .model
                .hidden(&input, 0, false)?
                .narrow(1, ids.len() - 1, 1)?;
            let logits = rust.model.logits(&hidden)?.flatten_all()?;
            let log_probabilities = candle_nn::ops::log_softmax(&logits, 0)?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect();
            let probe = RustLogProbabilityProbe {
                schema_version: 1,
                evidence_kind: "actual_full_vocabulary_rust_next_token_log_probabilities".into(),
                plan_sha256: plan_file.sha256.clone(),
                prompt_index: index,
                input_ids: ids.clone(),
                log_probabilities,
            };
            artefact::write_json(&rust_path, &probe)?;
            probe
        };
        ensure!(
            reference.plan_sha256 == plan_file.sha256
                && reference.prompt_index == index
                && reference.input_ids == prompt.input_ids,
            "retained Rust probe belongs to another frozen protocol"
        );
        let first = native["completion_probabilities"]
            .as_array()
            .and_then(|v| v.first())
            .context("native probability result absent")?;
        let top = reference
            .log_probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .context("empty Rust vocabulary")?
            .0;
        let candidates = first["top_logprobs"]
            .as_array()
            .context("native top16 probabilities absent")?;
        ensure!(
            candidates.len() == NUMERICAL_PARITY_TOP_TOKENS,
            "native top16 denominator changed"
        );
        let maximum = candidates
            .iter()
            .map(|candidate| {
                let id = candidate["id"].as_u64().context("native token ID absent")? as usize;
                let native = candidate["logprob"]
                    .as_f64()
                    .context("native log probability absent")?;
                Ok((native
                    - *reference
                        .log_probabilities
                        .get(id)
                        .context("native token outside Rust vocabulary")?)
                .abs())
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0_f64, f64::max);
        passed &=
            first["id"].as_u64() == Some(top as u64) && maximum <= NUMERICAL_PARITY_MAXIMUM_ERROR;
        observations.push(NumericalParityObservation {
            prompt_index: index,
            native_response: capture(root, &native_path)?,
            rust_logprobs: capture(root, &rust_path)?,
        });
    }
    let report = NumericalParityReport {
        schema_version: 2,
        evidence_kind: "actual_original_bitnet_rust_i2s_numerical_parity".into(),
        plan: plan_file,
        observations,
        cpp_execution,
        rust_execution: rust.execution_metadata(),
        comparison_completed: true,
        passed,
        production_eligible: false,
    };
    // A producer-computed aggregate is provisional until the independent
    // verifier reconstructs the fixed criterion from both retained backends.
    let provisional = args.output.with_extension("unverified.json");
    artefact::write_json(&provisional, &report)?;
    drop(rust);
    drop(cpp);
    let verified = verify_bitnet_parity_report(&provisional)?;
    artefact::write_json(&args.output, &report)?;
    for directory in &args.evaluation {
        attach_numerical_equivalence(directory, &args.output)?;
    }
    println!("{}", serde_json::to_string_pretty(&verified)?);
    ensure!(
        verified["passed"] == true,
        "native I2_S parity failed; full raw numerical evidence retained"
    );
    Ok(())
}
