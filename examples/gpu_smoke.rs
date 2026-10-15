//! Genuine CUDA inference QA on original weights, without opening task outcomes.
use anyhow::{Context, Result, ensure};
use lec_distil::{
    artefact,
    model::{GenerationRequest, ModelRuntime, RuntimeOptions},
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

fn observe_gpu(arguments: &[&str]) -> Result<String> {
    let result = Command::new("nvidia-smi").args(arguments).output()?;
    ensure!(
        result.status.success(),
        "nvidia-smi failed during actual CUDA QA"
    );
    Ok(String::from_utf8(result.stdout)?)
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let base = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| "offline_bundle/base".into());
    let comparative = arguments.iter().any(|value| value == "--comparative");
    let cpu_baseline = arguments.iter().any(|value| value == "--cpu-baseline");
    let output = PathBuf::from(format!(
        "reports/comparative/gpu-smoke/{}.json",
        uuid::Uuid::new_v4()
    ));
    let before = observe_gpu(&[
        "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap",
        "--format=csv,noheader",
    ])?;
    let loaded = Instant::now();
    let options = RuntimeOptions::cuda(0);
    let mut runtime = if comparative {
        ModelRuntime::load_comparative_with_options(&base, options)?
    } else {
        ModelRuntime::load_with_options(&base, None, options)?
    };
    runtime.model.device().synchronize()?;
    let load_seconds = loaded.elapsed().as_secs_f64();
    ensure!(
        runtime.model.device().is_cuda(),
        "actual GPU QA used a CPU device"
    );
    let process_observation = observe_gpu(&[
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])?;
    let process_id = std::process::id().to_string();
    let gpu_memory_mib = process_observation
        .lines()
        .find_map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            (fields.first().copied() == Some(process_id.as_str()))
                .then(|| fields.last()?.parse::<u64>().ok())
                .flatten()
        })
        .context("nvidia-smi did not observe this process allocating GPU memory")?;
    ensure!(
        gpu_memory_mib > 0,
        "actual CUDA process has no observed GPU memory"
    );
    let request = GenerationRequest {
        prompt: "Write a Rust function add(a: i64, b: i64) -> i64. Return only Rust source code."
            .into(),
        max_new_tokens: 128,
        seed: 1701,
        temperature: 0.,
        top_p: 1.,
        timeout_seconds: 120,
    };
    let generation = runtime.generate(request.clone())?;
    runtime.model.device().synchronize()?;
    ensure!(
        generation.complete && !generation.completion.trim().is_empty(),
        "actual CUDA smoke failed to produce complete output"
    );
    let source = lec_distil::evaluate::extract_code(&generation.completion);
    syn::parse_file(&source).context("CUDA smoke completion is not syntactically valid Rust")?;
    let warmed_generation = runtime.generate(request.clone())?;
    ensure!(
        warmed_generation.complete,
        "warmed CUDA inference was incomplete"
    );
    syn::parse_file(&lec_distil::evaluate::extract_code(
        &warmed_generation.completion,
    ))?;
    let cpu_generation = if cpu_baseline {
        let mut cpu = if comparative {
            ModelRuntime::load_comparative_with_options(&base, RuntimeOptions::default())?
        } else {
            ModelRuntime::load(&base, None)?
        };
        let result = cpu.generate(request)?;
        ensure!(result.complete, "CPU reference inference was incomplete");
        syn::parse_file(&lec_distil::evaluate::extract_code(&result.completion))?;
        Some(result)
    } else {
        None
    };
    let after = observe_gpu(&[
        "--query-gpu=name,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ])?;
    let report = json!({
        "evidence_kind":"actual_original_weights_native_rust_cuda_inference_qa",
        "assessment_training_or_heldout_evidence":false,
        "base_directory":base,
        "execution":runtime.execution_metadata(),
        "load_seconds":load_seconds,
        "process_id":std::process::id(),
        "observed_gpu_memory_mib":gpu_memory_mib,
        "nvidia_smi_before":before,
        "nvidia_smi_loaded_processes":process_observation,
        "nvidia_smi_after":after,
        "generation":generation,
        "warmed_generation":warmed_generation,
        "cpu_reference_generation":cpu_generation,
        "cpu_gpu_identical_source":cpu_generation.as_ref().map(|reference| lec_distil::evaluate::extract_code(&reference.completion) == source),
        "verification":"Complete outputs parsed as Rust source; no model-generated programme executed on the host. This QA prompt does not open a task split.",
        "output_report":output,
    });
    artefact::write_json(Path::new(&output), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
