//! Verify an already running local teacher without replacing its process.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact,
    comparative::{Registry, WeightFormat, comparative_checkpoint_id},
    model::{AssetManifest, Generation, GenerationRequest},
    serve::Health,
};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    #[arg(long, default_value = "http://127.0.0.1:8081")]
    endpoint: String,
    #[arg(long, default_value = "config/comparative-models.toml")]
    registry: PathBuf,
    #[arg(long, default_value = "qwen-coder-1_5b")]
    model: String,
    #[arg(long, default_value="auto", value_parser=["auto","cpu","cuda"])]
    expected_device: String,
    /// The supervisor's actual process ID, also required in the driver report.
    #[arg(long)]
    process_id: u32,
    #[arg(long, default_value = "reports/comparative/active-teacher")]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arguments::parse();
    ensure!(
        !args.output.join("report.json").exists(),
        "retain existing endpoint evidence; select a fresh output directory"
    );
    let registry = Registry::load(&args.registry)?;
    let model = registry
        .models
        .iter()
        .find(|m| m.id == args.model)
        .context("unknown registered teacher")?;
    let assets: AssetManifest = serde_json::from_slice(&fs::read(
        model.directory.join(model.weight_format.manifest_name()),
    )?)?;
    let checkpoint = comparative_checkpoint_id(&assets)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(125))
        .build()?;
    let endpoint = args.endpoint.trim_end_matches('/');
    eprintln!("teacher endpoint: waiting for checksum validation and model loading");
    let startup_deadline = Instant::now() + Duration::from_secs(180);
    let health: Health = loop {
        match client.get(format!("{endpoint}/health")).send().await {
            Ok(response) => break response.error_for_status()?.json().await?,
            Err(error) if error.is_connect() => {
                ensure!(
                    Instant::now() < startup_deadline,
                    "teacher startup deadline elapsed: {error}"
                );
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            Err(error) => return Err(error.into()),
        }
    };
    let cuda = health.execution["cuda_active"] == true;
    let expected_dtype = if model.weight_format == WeightFormat::Gguf || !cuda {
        "F32"
    } else {
        "F16"
    };
    ensure!(
        args.expected_device == "auto" || cuda == (args.expected_device == "cuda"),
        "endpoint uses another execution device"
    );
    ensure!(
        health.healthy
            && health.process_id == args.process_id
            && health.checksum_validated
            && health.immutable
            && !health.production_eligible
            && health.model_id == model.model_id
            && health.base_revision == model.revision
            && health.checkpoint_id == checkpoint
            && health.execution["actual_dtype"] == expected_dtype
            && health.execution["cpu_fallback"] == false,
        "running endpoint differs from the intended immutable CUDA teacher"
    );
    let request = GenerationRequest {
        prompt:
            "Write a Rust function add(a: i64, b: i64) -> i64. Return only the complete function."
                .into(),
        max_new_tokens: 128,
        seed: 1701,
        temperature: 0.,
        top_p: 1.,
        timeout_seconds: 120,
    };
    let generation: Generation = client
        .post(format!("{endpoint}/v1/generate"))
        .json(&request)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        generation.complete
            && generation.finish_reason == "stop"
            && !generation.completion.trim().is_empty()
            && generation.checkpoint_id == checkpoint,
        "running teacher failed to return complete checkpoint-bound output"
    );
    syn::parse_file(&lec_distil::evaluate::extract_code(&generation.completion))
        .context("teacher smoke completion is not syntactically valid Rust")?;
    let (processes, memory_mib) = if cuda {
        let driver = tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await?;
        ensure!(driver.status.success(), "GPU driver query failed");
        let processes = String::from_utf8(driver.stdout)?;
        let memory_mib = processes
            .lines()
            .find_map(|line| {
                let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
                (fields.len() == 3 && fields[0].parse::<u32>().ok() == Some(args.process_id))
                    .then(|| fields[2].parse::<u64>().ok())
                    .flatten()
            })
            .context("driver did not observe the supervisor's teacher process")?;
        ensure!(memory_mib > 0, "teacher has no observed GPU allocation");
        (processes, Some(memory_mib))
    } else {
        ensure!(
            health.execution["actual_device"] == "Cpu",
            "CPU endpoint did not report actual CPU execution"
        );
        (String::new(), None)
    };
    let report = serde_json::json!({
        "schema_version":1, "evidence_kind":"actual_existing_local_teacher_endpoint",
        "observed_unix_ms":lec_distil::ledger::unix_millis(),
        "endpoint":endpoint, "process_id":args.process_id,
        "health":health, "request":request, "generation":generation,
        "observed_gpu_memory_mib":memory_mib, "gpu_processes_csv":processes,
        "registry_sha256":artefact::file_hash(&args.registry)?,
        "assets_sha256":artefact::file_hash(&model.directory.join(model.weight_format.manifest_name()))?,
        "quality_assurance_passed":true, "promotable":false,
        "limitation":"One live original-model endpoint smoke; no learning, held-out result or assessment provider completion is claimed."
    });
    fs::create_dir_all(&args.output)?;
    artefact::write_json(&args.output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
