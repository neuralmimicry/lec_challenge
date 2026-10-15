//! Reproducible native worker, target capture and real process-recovery QA.
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use lec_distil::{
    artefact,
    distillation::{self, DistillationStudent, DistillationWorkerSpec},
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[path = "distillation_qa/launch.rs"]
mod launch;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Mode,
}

#[derive(Subcommand)]
enum Mode {
    Pin,
    Worker {
        spec: PathBuf,
    },
    Prewarm {
        spec: PathBuf,
    },
    /// Verify the actual worker selection without allocating model tensors.
    Eligibility {
        spec: PathBuf,
        output: PathBuf,
    },
    Verify {
        spec: PathBuf,
        output: PathBuf,
    },
    Resume {
        spec: PathBuf,
        output: PathBuf,
    },
}

fn spec(path: &Path) -> Result<DistillationWorkerSpec> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn child(mode: &str, request: &Path, output: &Path, log: &Path) -> Result<Child> {
    let file = fs::File::create(log)?;
    let mut command = Command::new(std::env::current_exe()?);
    command.arg(mode).arg(request);
    if mode == "verify" {
        command.arg(output);
    }
    Ok(command
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file))
        .spawn()?)
}

fn equal_tensors(a: &Path, b: &Path) -> Result<usize> {
    let left = candle_core::safetensors::load(a, &Device::Cpu)?;
    let right = candle_core::safetensors::load(b, &Device::Cpu)?;
    ensure!(
        left.len() == right.len(),
        "resumed tensor inventory differs"
    );
    for (name, tensor) in &left {
        let other = right.get(name).context("resumed tensor absent")?;
        ensure!(
            tensor.dims() == other.dims()
                && tensor.flatten_all()?.to_vec1::<f32>()?
                    == other.flatten_all()?.to_vec1::<f32>()?,
            "resumed tensor differs from uninterrupted execution: {name}"
        );
    }
    Ok(left.len())
}

fn verify(request: &Path, output: &Path) -> Result<()> {
    let s = spec(request)?;
    launch::bind(request, &s, launch::Phase::Verify)?;
    let (identity, execution, logits) = match s.student_kind {
        DistillationStudent::ConventionalQwenLora => {
            let mut model =
                distillation::load_runtime(&s.student_base, &s.output, s.student_runtime)?;
            let input = Tensor::from_slice(&[1_u32, 2], (1, 2), model.model.device())?;
            let hidden = model.model.hidden(&input, 0, false)?;
            (
                model.checkpoint_id.clone(),
                model.execution_metadata(),
                model.model.logits(&hidden)?,
            )
        }
        DistillationStudent::TernaryBitnet => {
            let mut model = lec_distil::bitnet::BitNetRuntime::load(&s.student_base, &s.output)?;
            let input = Tensor::from_slice(&[1_u32, 2], (1, 2), &Device::Cpu)?;
            let hidden = model.model.hidden(&input, 0, false)?;
            (
                model.checkpoint_id.clone(),
                model.execution_metadata(),
                model.model.logits(&hidden)?,
            )
        }
    };
    let values = logits
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "fresh-process reload has invalid logits"
    );
    artefact::write_json(
        output,
        &serde_json::json!({
            "schema_version":1,"evidence_kind":"actual_fresh_process_checkpoint_reload",
            "checkpoint_sha256":identity,"process_id":std::process::id(),
            "engine_sha256":artefact::file_hash(&std::env::current_exe()?)?,
            "request":artefact::Artefact::capture(&std::env::current_dir()?, request.to_owned())?,
            "execution":execution,"finite_logits":values.len(),
            "limitation":"Two synthetic token IDs test reload and numerical finiteness only; this is not a task quality observation.",
            "production_eligible":false
        }),
    )?;
    println!(
        "Fresh-process reload verified {} finite logits",
        values.len()
    );
    Ok(())
}

fn resume(request: &Path, output: &Path) -> Result<()> {
    ensure!(
        !output.exists(),
        "resume QA requires an unused output directory"
    );
    let original = spec(request)?;
    ensure!(
        original.steps >= 3,
        "resume QA needs at least three optimiser steps"
    );
    fs::create_dir_all(output)?;
    let mut baseline = original.clone();
    baseline.output = output.join("uninterrupted");
    let mut resumed = original;
    resumed.output = output.join("interrupted");
    let a = output.join("uninterrupted-request.json");
    let b = output.join("interrupted-request.json");
    artefact::write_json(&a, &baseline)?;
    artefact::write_json(&b, &resumed)?;
    ensure!(
        child("worker", &a, output, &output.join("uninterrupted.log"))?
            .wait()?
            .success(),
        "uninterrupted worker failed; inspect saved log"
    );
    let started = Instant::now();
    let mut process = child("worker", &b, output, &output.join("interrupted.log"))?;
    let committed = loop {
        if let Some(status) = process.try_wait()? {
            anyhow::bail!("worker ended before forced interruption: {status}");
        }
        let progress = resumed.output.join("progress.json");
        if progress.exists() {
            let value: serde_json::Value = serde_json::from_slice(&fs::read(&progress)?)?;
            if value["steps"]
                .as_u64()
                .is_some_and(|s| s > 0 && s < resumed.steps as u64)
            {
                break value;
            }
        }
        if started.elapsed() > Duration::from_secs(600) {
            process.kill()?;
            let _ = process.wait();
            anyhow::bail!("no durable step observed within the bounded interruption window");
        }
        thread::sleep(Duration::from_millis(10));
    };
    process.kill()?; // Linux Child::kill sends SIGKILL; no cooperative checkpoint hook.
    let killed = process.wait()?;
    ensure!(
        !killed.success(),
        "forced interruption unexpectedly reported success"
    );
    let saved_adapter: artefact::Artefact = serde_json::from_value(committed["adapter"].clone())?;
    let saved_optimizer: artefact::Artefact =
        serde_json::from_value(committed["optimizer"].clone())?;
    let root = std::env::current_dir()?;
    saved_adapter.verify(&root)?;
    saved_optimizer.verify(&root)?;
    ensure!(
        child("worker", &b, output, &output.join("resumed.log"))?
            .wait()?
            .success(),
        "resumed worker failed; inspect saved log"
    );
    saved_adapter.verify(&root)?;
    saved_optimizer.verify(&root)?;
    let adapters = equal_tensors(
        &baseline.output.join("adapter.safetensors"),
        &resumed.output.join("adapter.safetensors"),
    )?;
    let moments = equal_tensors(
        &baseline.output.join("optimizer.safetensors"),
        &resumed.output.join("optimizer.safetensors"),
    )?;
    let am: serde_json::Value =
        serde_json::from_slice(&fs::read(baseline.output.join("manifest.json"))?)?;
    let bm: serde_json::Value =
        serde_json::from_slice(&fs::read(resumed.output.join("manifest.json"))?)?;
    for field in [
        "losses",
        "gradient_norms",
        "component_losses",
        "sample_ids",
        "realised_update_counts",
    ] {
        ensure!(
            am[field] == bm[field],
            "resumed numerical/order evidence differs: {field}"
        );
    }
    let reload = output.join("fresh-process-reload.json");
    ensure!(
        child("verify", &b, &reload, &output.join("reload.log"))?
            .wait()?
            .success(),
        "fresh-process reload failed; inspect saved log"
    );
    artefact::write_json(
        &output.join("report.json"),
        &serde_json::json!({
            "schema_version":1,"evidence_kind":"actual_native_distillation_sigkill_resume_qa",
            "success":true,"interrupted_after_steps":committed["steps"],"completed_steps":resumed.steps,
            "killed_status":killed.to_string(),"committed_adapter":saved_adapter,
            "committed_optimizer":saved_optimizer,"committed_files_unchanged":true,
            "final_adapter_tensors_bit_identical":adapters,"final_optimizer_tensors_bit_identical":moments,
            "losses_gradients_order_bit_identical":true,
            "fresh_process_reload":artefact::Artefact::capture(&root,reload)?,
            "engine_sha256":artefact::file_hash(&std::env::current_exe()?)?,
            "limitation":"Local-teacher operator screening; no independent provider or held-out learning claim.",
            "production_eligible":false
        }),
    )?;
    println!(
        "Actual SIGKILL/resume verified: {adapters} adapter tensors and {moments} optimiser tensors agree exactly"
    );
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Mode::Pin => println!(
            "{}",
            serde_json::to_string_pretty(&lec_distil::engine::pin_executable(
                &std::env::current_exe()?,
                Path::new("state/engines")
            )?)?
        ),
        Mode::Worker { spec: path } => {
            let request = spec(&path)?;
            launch::bind(&path, &request, launch::Phase::Worker)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&distillation::run_worker(&request)?)?
            );
        }
        Mode::Prewarm { spec: path } => {
            let request = spec(&path)?;
            launch::bind(&path, &request, launch::Phase::Prewarm)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&distillation::prepare_teacher_targets(&request)?)?
            );
        }
        Mode::Eligibility { spec: path, output } => {
            let mut report = distillation::preflight_eligibility(&spec(&path)?)?;
            report["request"] = serde_json::to_value(artefact::Artefact::capture(
                &std::env::current_dir()?,
                path,
            )?)?;
            if output.exists() {
                ensure!(
                    serde_json::from_slice::<serde_json::Value>(&fs::read(&output)?)? == report,
                    "eligibility report inputs changed; preserve it and use a new output"
                );
            } else {
                artefact::write_json(&output, &report)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Mode::Verify { spec, output } => verify(&spec, &output)?,
        Mode::Resume { spec, output } => resume(&spec, &output)?,
    }
    Ok(())
}
