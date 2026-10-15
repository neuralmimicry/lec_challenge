//! Actual CUDA model endpoints, one resident teacher at a time.
//! This smoke checks serving identity; it makes no learning or promotion claim.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact,
    comparative::Registry,
    model::{Generation, GenerationRequest},
    serve::{Health, ServeConfig},
};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    #[arg(long, default_value = "config/comparative-models.toml")]
    registry: PathBuf,
    #[arg(long, default_value = "reports/comparative/gpu-serving-smoke")]
    output: PathBuf,
    #[arg(long, value_delimiter = ',')]
    models: Vec<String>,
}

struct Process(std::process::Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(
        !arguments.output.join("report.json").exists(),
        "retain existing smoke evidence and select a new output directory"
    );
    std::fs::create_dir_all(&arguments.output)?;
    let registry = Registry::load(&arguments.registry)?;
    let binary = std::env::current_exe()?
        .parent()
        .context("examples directory")?
        .parent()
        .context("profile directory")?
        .join("lec-distil");
    ensure!(binary.is_file(), "build the CUDA CLI beside this example");
    let binary_sha256 = artefact::file_hash(&binary)?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(125))
        .build()?;
    let mut results = Vec::new();
    for entry in registry
        .models
        .iter()
        .filter(|m| arguments.models.is_empty() || arguments.models.contains(&m.id))
    {
        eprintln!("gpu-serving-smoke: START {}", entry.model_id);
        let started = Instant::now();
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        drop(listener);
        let config = ServeConfig {
            address: address.to_string(),
            queue_capacity: 2,
            maximum_request_bytes: 32768,
            maximum_output_tokens: 256,
            timeout_seconds: 120,
        };
        let temporary = tempfile::tempdir()?;
        let config_path = temporary.path().join("serving.toml");
        std::fs::write(&config_path, toml::to_string(&config)?)?;
        let log_path = arguments.output.join(format!("{}.log", entry.id));
        let mut command = Command::new(&binary);
        command
            .args(["serve", "--device", "cuda", "--dtype", "f16", "--base"])
            .arg(&entry.directory)
            .arg("--config")
            .arg(&config_path);
        if entry.role != "student_baseline" {
            command.arg("--comparative");
        }
        let mut child = Process(
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::from(std::fs::File::create(&log_path)?))
                .spawn()?,
        );
        let url = format!("http://{address}/");
        let deadline = Instant::now() + Duration::from_secs(180);
        let health: Health = loop {
            ensure!(
                child.0.try_wait()?.is_none(),
                "CUDA endpoint exited; inspect {}",
                log_path.display()
            );
            if let Ok(response) = client.get(format!("{url}health")).send().await
                && response.status().is_success()
            {
                break response.json().await?;
            }
            ensure!(
                Instant::now() < deadline,
                "CUDA endpoint startup deadline elapsed"
            );
            tokio::time::sleep(Duration::from_millis(150)).await;
        };
        ensure!(
            health.healthy
                && health.checksum_validated
                && health.immutable
                && health.model_id == entry.model_id
                && health.base_revision == entry.revision,
            "CUDA endpoint identity/health mismatch"
        );
        ensure!(
            health.execution["cuda_active"] == true
                && health.execution["actual_dtype"] == "F16"
                && health.execution["cpu_fallback"] == false,
            "endpoint did not prove actual CUDA/F16 execution"
        );
        ensure!(
            entry.role == "student_baseline" || !health.production_eligible,
            "exploratory teacher was marked production eligible"
        );
        let request=GenerationRequest{prompt:"Write a Rust function add(a: i64, b: i64) -> i64. Return only the complete function.".into(),max_new_tokens:256,seed:1701,temperature:0.,top_p:1.,timeout_seconds:120};
        let response = client
            .post(format!("{url}v1/generate"))
            .json(&request)
            .send()
            .await?;
        ensure!(
            response.status().is_success(),
            "actual GPU endpoint generation returned {}",
            response.status()
        );
        let generation: Generation = response.json().await?;
        ensure!(
            generation.complete
                && generation.finish_reason == "stop"
                && !generation.completion.trim().is_empty()
                && generation.checkpoint_id == health.checkpoint_id,
            "GPU endpoint returned an incomplete or mismatched generation"
        );
        let snapshot = tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .await?;
        ensure!(
            snapshot.status.success(),
            "cannot record GPU process allocation"
        );
        results.push(serde_json::json!({"model":entry.id,"process_id":child.0.id(),"health":health,"request":request,"generation":generation,"gpu_processes_csv":String::from_utf8_lossy(&snapshot.stdout),"elapsed_seconds":started.elapsed().as_secs_f64(),"server_log":log_path}));
        artefact::write_json(&arguments.output.join("partial.json"), &results)?;
        eprintln!(
            "gpu-serving-smoke: PASSED {} {:.2}s",
            entry.model_id,
            started.elapsed().as_secs_f64()
        );
        drop(child);
    }
    ensure!(!results.is_empty(), "no registered smoke models selected");
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"actual_serial_cuda_f16_model_endpoints","binary_sha256":binary_sha256,"maximum_resident_models":1,"models":results,"quality_assurance_passed":true,"promotable":false,"limitation":"Actual GPU inference and endpoint identity checks only; no provider-derived assessment checkpoint, learning gain or quality promotion is claimed."});
    artefact::write_json(&arguments.output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
