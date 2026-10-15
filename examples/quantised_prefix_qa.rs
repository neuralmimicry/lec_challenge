//! Direct bounded numerical/prefix diagnostic. A length-truncated Generation is
//! retained as prefix evidence and is never called a successful coding answer.
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use lec_distil::{
    artefact, engine,
    model::{self, GenerationRequest, RuntimeOptions},
};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus},
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    #[arg(
        long,
        default_value = "offline_bundle/comparative/magicoder-s-ds-6_7b-q4_k_m"
    )]
    base: PathBuf,
    #[arg(
        long,
        default_value = "Write a Rust function add(a: i64, b: i64) -> i64. Return only the complete function."
    )]
    prompt: String,
    #[arg(long, default_value="cpu", value_parser=["cpu","cuda"])]
    device: String,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,
    #[arg(long, default_value_t = 120)]
    timeout_seconds: u64,
    /// Covers asset loading, the numerical forward and bounded generation.
    #[arg(long, default_value_t = 300)]
    wall_timeout_seconds: u64,
    #[arg(long)]
    cpp_reference: Option<PathBuf>,
    #[arg(
        long,
        default_value = "reports/comparative/quantised-prefix-qa/round-001"
    )]
    output: PathBuf,
    #[arg(long, hide = true)]
    worker: bool,
}

struct OwnedDiagnostic {
    child: Child,
    process_group: i32,
    reaped: bool,
}
impl Drop for OwnedDiagnostic {
    fn drop(&mut self) {
        if !self.reaped {
            // SAFETY: this is the isolated group of our retained, unreaped
            // child, so its identifier cannot have been reused by another run.
            unsafe { libc::kill(-self.process_group, libc::SIGKILL) };
            let _ = self.child.wait();
        }
    }
}

fn supervised(mut command: Command, deadline: Duration) -> Result<ExitStatus> {
    command.process_group(0);
    #[cfg(target_os = "linux")]
    {
        let expected_parent = std::process::id() as i32;
        // SAFETY: this hook performs only async-signal-safe libc operations
        // before exec. The exact-parent check closes the parent-death race.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != expected_parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
    }
    let child = command
        .spawn()
        .context("start isolated diagnostic worker")?;
    let process_group = child.id() as i32;
    let mut owned = OwnedDiagnostic {
        child,
        process_group,
        reaped: false,
    };
    let started = Instant::now();
    let mut heartbeat = Instant::now();
    loop {
        if let Some(status) = owned.child.try_wait()? {
            owned.reaped = true;
            return Ok(status);
        }
        ensure!(
            started.elapsed() < deadline,
            "whole diagnostic deadline exceeded after {:.1}s; owned process group terminated",
            started.elapsed().as_secs_f64()
        );
        if heartbeat.elapsed() >= Duration::from_secs(15) {
            eprintln!(
                "prefix-qa: supervised worker active {:.0}s / {:.0}s whole-process budget",
                started.elapsed().as_secs_f64(),
                deadline.as_secs_f64()
            );
            heartbeat = Instant::now();
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn compile_features() -> Vec<&'static str> {
    [
        ("avx", cfg!(target_feature = "avx")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("fma", cfg!(target_feature = "fma")),
        ("f16c", cfg!(target_feature = "f16c")),
        ("neon", cfg!(target_feature = "neon")),
    ]
    .into_iter()
    .filter_map(|(feature, enabled)| enabled.then_some(feature))
    .collect()
}

fn run(args: &Arguments) -> Result<Value> {
    ensure!(
        (1..=64).contains(&args.max_tokens) && (1..=256).contains(&args.threads),
        "invalid bounded prefix diagnostic"
    );
    ensure!(
        !args.output.exists(),
        "preserve previous prefix diagnostic; select a new output directory"
    );
    fs::create_dir_all(&args.output)?;
    let executable = std::env::current_exe()?.canonicalize()?;
    let executable_hash = artefact::file_hash(&executable)?;
    let pinned = engine::resolve(Path::new("state/engines/diagnostics"), &executable_hash)?;
    ensure!(
        pinned.path == executable,
        "diagnostic worker is not its immutable pinned executable"
    );
    ensure!(
        candle_core::utils::get_num_threads() == args.threads,
        "Candle thread pool differs from assigned CPU allocation"
    );
    let hardware = lec_distil::platform::detect()?;
    let runtime = if args.device == "cuda" {
        RuntimeOptions {
            device: model::ExecutionDevice::Cuda { ordinal: 0 },
            dtype: model::InferenceDtype::F32,
        }
    } else {
        RuntimeOptions::default()
    };
    eprintln!(
        "prefix-qa: validating and loading real quantised weights on {}",
        args.device
    );
    let load_start = Instant::now();
    let mut loaded = model::ModelRuntime::load_quantised_with_options(&args.base, runtime.clone())?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let rendered = loaded.formatted_prompt(&args.prompt)?;
    let ids = loaded
        .tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .get_ids()
        .to_vec();
    let request = GenerationRequest {
        prompt: args.prompt.clone(),
        max_new_tokens: args.max_tokens,
        seed: 1729,
        temperature: 0.,
        top_p: 1.,
        timeout_seconds: args.timeout_seconds,
    };
    let protocol = json!({"schema_version":1,"evidence_kind":"frozen_direct_quantised_synthetic_prefix_diagnostic","engine":{"path":executable,"sha256":artefact::file_hash(&executable)?},
        "runtime":runtime,"execution":loaded.execution_metadata(),"base":args.base,"base_manifest_sha256":artefact::file_hash(&args.base.join("gguf-assets.json"))?,
        "publisher_provenance_sha256":artefact::file_hash(&args.base.join("gguf-provenance.json"))?,"request":request,
        "original_rendered_prompt":rendered,"original_rendered_prompt_sha256":artefact::hash(rendered.as_bytes()),"original_prompt_token_ids":ids,
        "assigned_cpu_threads":args.threads,"whole_process_timeout_seconds":args.wall_timeout_seconds,"compiled_cpu_features":compile_features(),"hardware":hardware,
        "held_out_used":false,"coding_success_claim":false,"model_quality_or_learning_claim":false});
    artefact::write_json(&args.output.join("protocol.json"), &protocol)?;
    // The protocol is durable before this first forward pass. Record finite
    // logits independently of the sampler and retain an actual top-token view.
    let forward_start = Instant::now();
    let input = Tensor::from_slice(&ids, (1, ids.len()), loaded.model.device())?;
    let hidden = loaded.model.hidden(&input, 0, true)?;
    let logits = loaded
        .model
        .logits(&hidden.i((.., ids.len() - 1, ..))?)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    ensure!(
        !logits.is_empty() && logits.iter().all(|value| value.is_finite()),
        "prefix forward logits are absent or non-finite"
    );
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|(ai, a), (bi, b)| b.total_cmp(a).then(ai.cmp(bi)));
    let numerical = json!({"finite_logits":logits.len(),"first_forward_seconds":forward_start.elapsed().as_secs_f64(),"top8":ranked.iter().take(8).map(|(id,value)|json!({"id":id,"logit":value,"token":loaded.tokenizer.id_to_token(*id as u32)})).collect::<Vec<_>>()});
    artefact::write_json(&args.output.join("numerical.json"), &numerical)?;
    loaded.model.clear_cache();
    eprintln!(
        "prefix-qa: generating at most {} greedy tokens; incomplete answers remain prefix-only evidence",
        args.max_tokens
    );
    let generation = loaded.generate(request)?;
    artefact::write_json(&args.output.join("generation.json"), &generation)?;
    let diagnostic_complete = matches!(generation.finish_reason.as_str(), "stop" | "length")
        && !generation.completion.trim().is_empty()
        && (generation.complete || generation.output_tokens == args.max_tokens);
    let comparison = if let Some(reference) = &args.cpp_reference {
        let cpp: Value = serde_json::from_slice(&fs::read(reference.join("cpp-result.json"))?)?;
        let tokeniser: Value =
            serde_json::from_slice(&fs::read(reference.join("prompt-tokeniser-parity.json"))?)?;
        ensure!(
            cpp["exit_success"] == true
                && cpp["prompt_tokens"] == ids.len()
                && cpp["generation_limit_tokens"] == args.max_tokens
                && tokeniser["success"] == true
                && tokeniser["native_ids"] == json!(ids),
            "C++ reference differs from exact input/output protocol"
        );
        let exact = cpp["completion"].as_str() == Some(generation.completion.as_str());
        Some(
            json!({"cpp_reference":reference,"cpp_result_sha256":artefact::file_hash(&reference.join("cpp-result.json"))?,"cpp_completion":cpp["completion"],"byte_exact_prefix_match":exact,
            "limitation":"Identical prompt IDs and bounded greedy text; CPU/CUDA matrix and embedding arithmetic may differ. No long-decoding, task-quality or promotion conclusion follows."}),
        )
    } else {
        None
    };
    let report = json!({"schema_version":1,"evidence_kind":"actual_direct_quantised_numerical_and_prefix_diagnostic","prefix_diagnostic_complete":diagnostic_complete,
        "coding_success_claim":false,"model_quality_or_learning_claim":false,"held_out_used":false,"loading_seconds":load_seconds,"numerical":numerical,
        "generation":generation,"generation_is_complete_answer":generation.complete,"comparison":comparison,
        "limitation":"Length-truncated output is retained only as bounded numerical/prefix evidence. A missing, empty, timed-out or errored generation fails this diagnostic."});
    artefact::write_json(&args.output.join("report.json"), &report)?;
    ensure!(
        diagnostic_complete,
        "bounded prefix generation was empty, timed out or incomplete before its declared limit"
    );
    Ok(report)
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        (1..=3600).contains(&args.wall_timeout_seconds)
            && (1..=1800).contains(&args.timeout_seconds),
        "diagnostic wall/generation deadlines must be positive and bounded"
    );
    if !args.worker {
        let pinned = engine::pin_executable(
            &std::env::current_exe()?,
            Path::new("state/engines/diagnostics"),
        )?;
        let mut command = Command::new(&pinned.path);
        command
            .args(std::env::args_os().skip(1))
            .arg("--worker")
            .env("RAYON_NUM_THREADS", args.threads.to_string())
            .env("OMP_NUM_THREADS", args.threads.to_string())
            .env("MKL_NUM_THREADS", args.threads.to_string());
        let existed = args.output.exists();
        let result = supervised(command, Duration::from_secs(args.wall_timeout_seconds)).and_then(
            |status| {
                ensure!(
                    status.success(),
                    "diagnostic worker exited unsuccessfully: {status}"
                );
                Ok(())
            },
        );
        if !existed
            && !args.output.join("failure.json").exists()
            && let Err(error) = &result
        {
            fs::create_dir_all(&args.output)?;
            artefact::write_json(
                &args.output.join("failure.json"),
                &json!({
                    "schema_version":1,"prefix_diagnostic_complete":false,"coding_success_claim":false,
                    "engine_sha256":pinned.sha256,"error":format!("{error:#}")
                }),
            )?;
        }
        return result;
    }
    let existed = args.output.exists();
    let result = run(&args);
    match &result {
        Ok(report) => println!("{}", serde_json::to_string_pretty(report)?),
        Err(error) => {
            let failure = json!({"schema_version":1,"prefix_diagnostic_complete":false,"coding_success_claim":false,"error":format!("{error:#}")});
            if !existed {
                fs::create_dir_all(&args.output)?;
                artefact::write_json(&args.output.join("failure.json"), &failure)?;
            }
            eprintln!("{}", serde_json::to_string_pretty(&failure)?);
        }
    }
    result.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_process_deadline_kills_and_reaps_an_owned_worker() {
        let mut command = Command::new("/bin/sleep");
        command.arg("60");
        let started = Instant::now();
        let error = supervised(command, Duration::from_millis(100)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("whole diagnostic deadline exceeded")
        );
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
