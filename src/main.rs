//! Thin command boundary: trusted modules own decisions and durable evidence.
mod cli_demo;
use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use futures_util::{StreamExt, stream};
use lec_distil::{
    artefact, comparative, curate, drills, evaluate,
    fleet::{self, FleetConfig, FleetRunner, LiveTransport, ReplayTransport, Transport},
    gate::{self, GateConfig},
    ledger::Ledger,
    model, platform, preflight,
    sandbox::{self, SandboxConfig},
    serve,
    tasks::{self, Split},
    train,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Parser)]
#[command(
    version,
    about = "Native Rust, evidence-bound coding model distillation"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct ExecutionArguments {
    /// CUDA requests fail explicitly when this binary or host cannot support them.
    #[arg(long, value_parser = ["auto", "cpu", "cuda"])]
    device: Option<String>,
    #[arg(long, default_value_t = 0)]
    gpu: usize,
    #[arg(long, value_parser = ["f32", "f16"])]
    dtype: Option<String>,
}
impl ExecutionArguments {
    fn frozen_options(&self, saved: model::RuntimeOptions) -> Result<model::RuntimeOptions> {
        if let Some(device) = self.device.as_deref() {
            ensure!(
                device == "auto"
                    || (device == "cpu" && matches!(saved.device, model::ExecutionDevice::Cpu))
                    || (device == "cuda"
                        && matches!(saved.device, model::ExecutionDevice::Cuda { ordinal } if ordinal == self.gpu)),
                "requested execution device differs from the original run"
            );
        }
        if let Some(dtype) = self.dtype.as_deref() {
            ensure!(
                (dtype == "f32") == (saved.dtype == model::InferenceDtype::F32),
                "requested precision differs from the original run"
            );
        }
        eprintln!(
            "execution: resuming original frozen runtime {saved:?}; resource availability is checked before new computation"
        );
        Ok(saved)
    }

    fn options_from_plan(&self, path: &Path) -> Result<Option<model::RuntimeOptions>> {
        if !path.is_file() {
            return Ok(None);
        }
        let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        Ok(Some(self.frozen_options(serde_json::from_value(
            plan["runtime"].clone(),
        )?)?))
    }

    fn options_for_model(&self, base: &Path) -> Result<model::RuntimeOptions> {
        self.options_for_model_with_context(base, None)
    }

    fn options_for_model_with_context(
        &self,
        base: &Path,
        context_tokens: Option<usize>,
    ) -> Result<model::RuntimeOptions> {
        let quantised = base.join("gguf-assets.json").is_file();
        ensure!(
            quantised || context_tokens.is_none(),
            "explicit teacher context requires an original GGUF model"
        );
        let config_path = base.join(if quantised {
            "source/config.json"
        } else {
            "config.json"
        });
        let cfg: model::QwenConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let manifest: model::AssetManifest =
            serde_json::from_slice(&std::fs::read(base.join(if quantised {
                "gguf-assets.json"
            } else {
                "assets.json"
            }))?)?;
        let mut bytes = 0_u64;
        for name in manifest
            .files
            .keys()
            .filter(|name| name.ends_with(".gguf") || name.ends_with(".safetensors"))
        {
            ensure!(
                Path::new(name)
                    .components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
                "unsafe weight path"
            );
            bytes = bytes
                .checked_add(std::fs::metadata(base.join(name))?.len())
                .context("weight size overflow")?;
        }
        let (gpu_bytes, cpu_bytes) = if quantised {
            quantised_residency(&cfg, bytes, context_tokens)?
        } else {
            (
                bytes
                    .saturating_mul(if self.dtype.as_deref() == Some("f32") {
                        2
                    } else {
                        1
                    })
                    .saturating_add(768 * 1024 * 1024),
                bytes.saturating_mul(2).saturating_add(1024 * 1024 * 1024),
            )
        };
        self.options_with_estimate(quantised, gpu_bytes, cpu_bytes)
    }

    fn options_with_estimate(
        &self,
        quantised: bool,
        gpu_bytes: u64,
        cpu_bytes: u64,
    ) -> Result<model::RuntimeOptions> {
        let hardware = platform::detect()?;
        let selected = platform::select_runtime(
            &hardware,
            self.device.as_deref().unwrap_or("auto"),
            self.gpu,
            quantised,
            gpu_bytes,
            cpu_bytes,
        )?;
        eprintln!(
            "execution: {}; available CPU threads={}, usable RAM={} bytes",
            selected.reason, hardware.available_cpu_threads, hardware.usable_ram_bytes
        );
        let mut options = selected.options;
        if let Some(dtype) = &self.dtype {
            options.dtype = if dtype == "f16" {
                model::InferenceDtype::F16
            } else {
                model::InferenceDtype::F32
            };
        }
        Ok(options)
    }
}

/// Select the runtime using the same requested context that will be loaded.
/// Otherwise a larger context may appear to fit CUDA and fail after selection,
/// even when automatic execution could have used sufficient available RAM.
fn quantised_residency(
    cfg: &model::QwenConfig,
    weight_bytes: u64,
    context_tokens: Option<usize>,
) -> Result<(u64, u64)> {
    let context = context_tokens.unwrap_or(1024.min(cfg.max_position_embeddings));
    ensure!(
        (1..=4096).contains(&context) && context <= cfg.max_position_embeddings,
        "requested teacher context exceeds the native or source limit"
    );
    Ok((
        model::estimate_quantised_memory(cfg, weight_bytes, context, true)?
            .conservative_total_bytes,
        model::estimate_quantised_memory(cfg, weight_bytes, context, false)?
            .conservative_total_bytes,
    ))
}
#[derive(Subcommand)]
enum Command {
    /// Resume adaptive teacher selection and bounded local distillation studies.
    Explore {
        #[arg(long, default_value = "config/teacher-search.toml")]
        search_config: PathBuf,
        #[arg(long, default_value = "config/distillation-study.toml")]
        study_config: PathBuf,
        #[arg(long, default_value = "reports/local-study/explore/round-001")]
        output: PathBuf,
        #[arg(long, default_value = "reports/comparative/search/round-001")]
        search_output: PathBuf,
        #[arg(long, default_value_t = 86400)]
        maximum_run_seconds: u64,
        #[arg(long)]
        maximum_parallel_jobs: Option<usize>,
        #[arg(long, default_value_t = 48)]
        content_maximum_tasks: usize,
        #[arg(long, default_value_t = 2)]
        top_groups: usize,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        fresh: bool,
    },
    /// Detect CPU affinity, usable RAM and available accelerator memory.
    Hardware,
    /// Retain verified teachers and replace weak models with pinned Hub alternatives.
    SelectTeachers {
        #[arg(long, default_value = "config/teacher-search.toml")]
        config: PathBuf,
        #[arg(long, default_value = "reports/comparative/search/round-001")]
        output: PathBuf,
        #[arg(long)]
        offline: bool,
        /// Begin a fresh search in a new directory, preserving previous evidence.
        #[arg(long)]
        fresh: bool,
    },
    /// Collect independently graded local-teacher training evidence.
    CollectTeacherEvidence {
        #[arg(long)]
        registry: PathBuf,
        #[arg(long, default_value = "reports/local-study/evidence")]
        output: PathBuf,
        #[arg(long)]
        comparison: Vec<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long)]
        offline: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Acquire genuine process, context and tool-feedback variants on train only.
    CollectTeacherContent {
        #[arg(long, default_value = "reports/local-study/evidence/bundle.json")]
        bundle: PathBuf,
        #[arg(long, default_value = "reports/local-study/content")]
        output: PathBuf,
        /// Preserve the previous run and begin in a new sibling directory.
        #[arg(long)]
        fresh: bool,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "process,context,tool_trajectory"
        )]
        content: Vec<String>,
        #[arg(long, default_value_t = 48)]
        maximum_tasks: usize,
        #[arg(long, default_value_t = 768)]
        max_tokens: usize,
        /// Explicit total context for an original GGUF teacher; omission retains its existing cap.
        #[arg(long)]
        context_tokens: Option<usize>,
        #[arg(long)]
        offline: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Collect frozen-student attempts, actual corrective generations and regrades.
    CollectStudentRollouts {
        #[arg(
            long,
            default_value = "reports/local-study/qwen7b-evidence/bundle.json"
        )]
        bundle: PathBuf,
        #[arg(long, default_value = "reports/local-study/student-rollouts")]
        output: PathBuf,
        /// Preserve the previous run and begin in a new sibling directory.
        #[arg(long)]
        fresh: bool,
        #[arg(long, default_value = "offline_bundle/base")]
        student_base: PathBuf,
        /// Omit only to collect from the actual frozen base weights.
        #[arg(long)]
        student_checkpoint: Option<PathBuf>,
        /// Link an original batch without regenerating or regrading its outputs.
        #[arg(long)]
        reuse_student_attempts_from: Option<PathBuf>,
        #[arg(long, default_value="conventional_qwen_lora",value_parser=["conventional_qwen_lora","ternary_bitnet"])]
        student_kind: String,
        #[arg(long,default_value="on_policy",value_parser=["on_policy","self_distillation"])]
        mode: String,
        #[arg(long, value_delimiter = ',')]
        teachers: Vec<String>,
        #[arg(long, default_value_t = 4)]
        maximum_tasks: usize,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        /// Explicit correction-teacher context; the student runtime remains unchanged.
        #[arg(long)]
        teacher_context_tokens: Option<usize>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long,value_parser=["cpu","cuda"])]
        student_device: Option<String>,
        #[arg(long, default_value_t = 0)]
        student_gpu: usize,
        #[arg(long)]
        offline: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Freeze the finite, capability-checked local distillation catalogue.
    StudyPlan {
        #[arg(long, default_value = "config/distillation-study.toml")]
        config: PathBuf,
        #[arg(long, default_value = "reports/local-study/evidence/bundle.json")]
        bundle: PathBuf,
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
        bitnet_base: PathBuf,
        #[arg(long, default_value = "reports/local-study/study/round-001")]
        output: PathBuf,
        #[arg(long)]
        maximum_parallel_jobs: Option<usize>,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        fresh: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Resume a bounded set of real native training subprocesses.
    StudyRun {
        #[arg(long, default_value = "reports/local-study/study/round-001")]
        study: PathBuf,
        #[arg(long, default_value = "quick", value_parser = ["quick", "full"])]
        profile: String,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        selected_arms: Vec<String>,
        #[arg(long)]
        selection_evidence: Option<PathBuf>,
        #[arg(long, default_value_t = 1800)]
        maximum_run_seconds: u64,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        fresh: bool,
    },
    /// Rank verified development outcomes and freeze full-fidelity seed groups.
    StudySelect {
        #[arg(long, default_value = "reports/local-study/study/round-001")]
        study: PathBuf,
        #[arg(long, default_value_t = 2)]
        top_groups: usize,
    },
    /// Measure a real local student baseline or adapter on train/development.
    StudyEvaluate {
        #[arg(long)]
        base: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "qwen", value_parser = ["qwen", "bitnet"])]
        student: String,
        /// Explicit native BitNet engine receipt for a new evaluation protocol.
        #[arg(long)]
        bitnet_engine_manifest: Option<PathBuf>,
        /// Explicit exported GGUF path; existing exports are never replaced silently.
        #[arg(long)]
        bitnet_gguf: Option<PathBuf>,
        /// Exact checkpoint/backend numerical report to attach after evaluation.
        #[arg(long)]
        bitnet_parity_report: Option<PathBuf>,
        #[arg(long, default_value = "reports/local-study/evaluation")]
        output: PathBuf,
        #[arg(long, default_value = "development", value_parser = ["train", "development"])]
        split: String,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        fresh: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Regrade every saved train/development output under an explicit new host policy.
    StudyRegrade {
        /// Complete original evaluation directories; no task filtering or regeneration.
        #[arg(long)]
        original: Vec<PathBuf>,
        #[arg(long, default_value = "reports/local-study/regraded")]
        output: PathBuf,
        #[arg(long, default_value_t = 192)]
        expected_generations: usize,
        #[arg(long, default_value_t = 4)]
        maximum_concurrent_graders: usize,
        /// Reconstruct the completed batch without launching any container.
        #[arg(long)]
        verify_only: bool,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        fresh: bool,
    },
    /// Independently verify a completed original or derived local evaluation.
    StudyVerify {
        #[arg(long)]
        evaluation: PathBuf,
    },
    /// Descriptively compare complete local evaluations; never authorise ship.
    StudyCompare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long, required = true)]
        candidate: Vec<PathBuf>,
    },
    /// Internal immutable-request boundary used by the study scheduler.
    #[command(hide = true)]
    DistillationArmWorker {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    #[command(hide = true)]
    StudyEvaluateWorker {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        training_artifact: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Measure this host; no provider key values are read into the report.
    Preflight,
    /// Materialise and validate all 96 source-disjoint contracts and private cases.
    Tasks {
        #[arg(long, default_value = ".")]
        output: PathBuf,
    },
    /// Download pinned public model assets and build the candidate image explicitly.
    BootstrapOnline {
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long, default_value = "podman")]
        runtime: String,
    },
    /// Explicit authenticated, immutable-revision Hub download; no credential cache.
    HubDownload {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long, default_value_t = 2)]
        concurrency: usize,
        #[arg(long)]
        token_stdin: bool,
    },
    /// Compare locally hosted weights on exploratory train/development tasks only.
    CompareModels {
        #[arg(long, default_value = "config/comparative-models.toml")]
        config: PathBuf,
        #[arg(long, default_value = "reports/comparative/development")]
        output: PathBuf,
        #[arg(long, default_value = "development", value_parser = ["train", "development"])]
        split: String,
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Verify actual isolation and resource enforcement using executable probes.
    SandboxPreflight,
    /// Compile and grade Rust candidate code inside a fresh isolated container.
    Grade {
        #[arg(long)]
        task: String,
        #[arg(long)]
        code: PathBuf,
    },
    /// Acquire/replay all three configured businesses; persist responses before grading.
    Collect {
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        live: bool,
        #[arg(long)]
        authorised_usd: Option<String>,
        #[arg(long, default_value = "config/fleet.toml")]
        config: PathBuf,
        #[arg(long, default_value = "round-001")]
        round: String,
    },
    /// Build the dataset from verified real run/recording artifacts.
    Curate {
        #[arg(long, default_value = "reports/training-rows.json")]
        rows: PathBuf,
        #[arg(long, default_value = "data/curated")]
        output: PathBuf,
    },
    /// Execute genuine native Rust CPU LoRA forward/backward/optimiser steps.
    Train {
        #[arg(long, default_value = "data/curated")]
        dataset: PathBuf,
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long, default_value = "config/training.toml")]
        config: PathBuf,
        #[arg(long, default_value = "checkpoints/candidate")]
        output: PathBuf,
        /// Accepted for compatibility; validated partial training resumes by default.
        #[arg(long, hide = true, conflicts_with = "fresh")]
        resume: bool,
        /// Begin a fresh checkpoint in a new directory, preserving previous evidence.
        #[arg(long)]
        fresh: bool,
    },
    /// Generate with the real pinned base and optional validated trained adapter.
    Infer {
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Load a checksum-verified comparative model; assessment adapters stay separate.
        #[arg(long)]
        comparative: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    /// Freeze a complete decoding plan and evaluate actual checkpoint generations.
    Evaluate {
        #[arg(long,default_value="held-out",value_parser=["train","development","held-out"])]
        split: String,
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "reports/candidate-evaluation")]
        output: PathBuf,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    /// Two checkpoint directories (or complete verified evaluation manifests), one decision.
    Gate {
        #[arg(long)]
        incumbent: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
    },
    /// Simulate the actual promotion rule under declared development assumptions.
    Power {
        #[arg(long)]
        quick: bool,
    },
    Serve {
        #[arg(long, default_value = "offline_bundle/base")]
        base: PathBuf,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, default_value = "config/serving.toml")]
        config: PathBuf,
        #[arg(long)]
        comparative: bool,
        /// Serve a verified local-study adapter on an evaluation-only endpoint.
        #[arg(long, conflicts_with = "comparative", requires = "checkpoint")]
        local_study: bool,
        #[command(flatten)]
        execution: ExecutionArguments,
    },
    RouteInit {
        #[arg(long)]
        url: String,
        #[arg(long)]
        checkpoint_id: String,
    },
    StageEvaluation {
        #[arg(long)]
        url: String,
        #[arg(long)]
        checkpoint_id: String,
    },
    Canary {
        #[arg(long)]
        url: String,
        #[arg(long)]
        checkpoint_id: String,
        #[arg(long)]
        incumbent_evaluation: PathBuf,
        #[arg(long)]
        candidate_evaluation: PathBuf,
    },
    Rollback {
        #[arg(long, default_value = "operator requested rollback")]
        reason: String,
    },
    Costs {
        #[arg(long)]
        round: Option<String>,
    },
    Drill {
        #[arg(long)]
        offline: bool,
    },
    Demo {
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        use_recorded_checkpoint: bool,
    },
    /// Create an honest local delivery archive and asset inventory.
    Package,
    #[command(name = "__fleet-drill-child", hide = true)]
    FleetDrillChild {
        work_dir: PathBuf,
        #[arg(long)]
        unknown_window: bool,
    },
    #[command(name = "__infrastructure-failure", hide = true)]
    InfrastructureFailure,
}

fn ledger() -> Result<Ledger> {
    Ledger::open("state/ledger.sqlite", "state/artifacts")
}
fn heldout_index() -> PathBuf {
    PathBuf::from("grader_private/heldout-index.json")
}
async fn verified_sandbox() -> Result<SandboxConfig> {
    let mut cfg = SandboxConfig::default();
    let report = sandbox::preflight(&cfg).await?;
    ensure!(report.probes_passed, "sandbox enforcement probes failed");
    cfg.expected_image_id = report.image_id.clone();
    artefact::write_json(Path::new("reports/sandbox-preflight.json"), &report)?;
    Ok(cfg)
}
fn print(value: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
fn usd_nanos(value: &str) -> Result<i64> {
    ensure!(
        !value.starts_with('-') && !value.starts_with('+'),
        "spending cap must be nonnegative decimal USD"
    );
    let parts = value.split('.').collect::<Vec<_>>();
    ensure!(
        parts.len() <= 2 && !parts[0].is_empty() && parts[0].bytes().all(|b| b.is_ascii_digit()),
        "invalid USD amount"
    );
    let whole: i64 = parts[0].parse()?;
    let fraction = parts.get(1).copied().unwrap_or("");
    ensure!(
        fraction.len() <= 9 && fraction.bytes().all(|b| b.is_ascii_digit()),
        "USD precision is at most nine decimal places"
    );
    whole
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(format!("{fraction:0<9}").parse::<i64>().ok()?))
        .context("USD amount overflow")
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let mut cli = Cli::parse();
    if let Err(error) = prepare_fresh_command(&mut cli.command).await {
        let path = PathBuf::from("reports").join(format!(
            "fresh-preparation-failure-{}.json",
            uuid::Uuid::new_v4()
        ));
        let _ = artefact::write_json(
            &path,
            &serde_json::json!({"status":"FAILURE","error":format!("{error:#}"),"previous_run_preserved":true}),
        );
        eprintln!(
            "FAILURE preparing fresh output: {error:#}; report={}",
            path.display()
        );
        std::process::exit(1);
    }
    let failure_path = match &cli.command {
        Command::Explore { output, .. }
        | Command::SelectTeachers { output, .. }
        | Command::CollectTeacherEvidence { output, .. }
        | Command::CollectTeacherContent { output, .. }
        | Command::CollectStudentRollouts { output, .. }
        | Command::StudyPlan { output, .. }
        | Command::StudyEvaluate { output, .. }
        | Command::DistillationArmWorker { output, .. }
        | Command::StudyEvaluateWorker { output, .. }
        | Command::Train { output, .. }
        | Command::Evaluate { output, .. } => output.join("last-failure.json"),
        Command::StudyRegrade { .. }
        | Command::StudyVerify { .. }
        | Command::StudyCompare { .. } => {
            PathBuf::from("reports").join(format!("regrade-failure-{}.json", uuid::Uuid::new_v4()))
        }
        Command::StudyRun {
            study,
            profile,
            output,
            ..
        } => output
            .clone()
            .unwrap_or_else(|| study.join(profile))
            .join("last-failure.json"),
        Command::StudySelect { study, .. } => study.join("selection-failure.json"),
        _ => PathBuf::from("reports/last-failure.json"),
    };
    if let Err(error) = execute(cli.command).await {
        let failure = serde_json::json!({"status":"FAILURE","error":format!("{error:#}"),"observed_unix_ms":lec_distil::ledger::unix_millis(),"promotion":"incumbent_retained"});
        let _ = artefact::write_json(&failure_path, &failure);
        eprintln!("FAILURE: {error:#}");
        println!("{failure}");
        std::process::exit(1);
    }
}

/// Resolve an explicit new output before selecting the failure-report path.
/// Otherwise a failed fresh invocation would overwrite the original run's
/// last-failure file even though all of its numerical evidence was preserved.
async fn prepare_fresh_command(command: &mut Command) -> Result<()> {
    match command {
        Command::Explore { output, fresh, .. }
        | Command::SelectTeachers { output, fresh, .. }
        | Command::CollectTeacherContent { output, fresh, .. }
        | Command::CollectStudentRollouts { output, fresh, .. }
        | Command::StudyPlan { output, fresh, .. }
        | Command::StudyEvaluate { output, fresh, .. }
        | Command::StudyRegrade { output, fresh, .. }
        | Command::Train { output, fresh, .. }
            if *fresh =>
        {
            *output = fresh_run_output(output).await?;
            *fresh = false;
        }
        Command::StudyRun {
            study,
            profile,
            output,
            fresh,
            ..
        } if *fresh => {
            let requested = output.clone().unwrap_or_else(|| study.join(profile));
            *output = Some(fresh_run_output(&requested).await?);
            *fresh = false;
        }
        _ => {}
    }
    Ok(())
}

async fn execute(command: Command) -> Result<()> {
    match command {
        Command::Explore {
            search_config,
            study_config,
            output,
            search_output,
            maximum_run_seconds,
            maximum_parallel_jobs,
            content_maximum_tasks,
            top_groups,
            offline,
            fresh,
        } => {
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            let report = lec_distil::explore::run(lec_distil::explore::ExploreOptions {
                search_config,
                study_config,
                output,
                search_output,
                maximum_run_seconds,
                maximum_parallel_jobs,
                content_maximum_tasks,
                top_groups,
                offline,
                ..Default::default()
            })
            .await?;
            print(&report)?;
            ensure!(
                report["success"] == true,
                "local exploration remains incomplete or below its frozen teacher-quality target; progress retained"
            );
        }
        Command::Hardware => {
            print(&tokio::task::spawn_blocking(platform::detect).await??)?;
        }
        Command::SelectTeachers {
            config,
            output,
            offline,
            fresh,
        } => {
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            let report =
                lec_distil::teacher_search::run_with_options(&config, &output, offline).await?;
            print(&report)?;
            ensure!(
                report["qualified_target_met"] == true,
                "teacher selection has not met its declared quality target; retained best available model and saved progress"
            );
        }
        Command::CollectTeacherEvidence {
            registry,
            output,
            comparison,
            models,
            max_tokens,
            offline,
            execution,
        } => {
            eprintln!(
                "local teacher collection: {} — cached local weights only; no provider HTTP calls",
                if offline {
                    "explicit offline mode"
                } else {
                    "local execution"
                }
            );
            let roster = comparative::Registry::load(&registry)?;
            let teacher = roster
                .models
                .iter()
                .find(|m| models.is_empty() || models.contains(&m.id))
                .context("no selected teacher in registry")?;
            let previous_plan = comparison
                .first()
                .map(|p| p.join("plan.json"))
                .unwrap_or_else(|| output.join("collection/plan.json"));
            let runtime = if let Some(saved) = execution.options_from_plan(&previous_plan)? {
                saved
            } else {
                execution.options_for_model(&teacher.directory)?
            };
            let bundle = lec_distil::teacher_evidence::collect(
                lec_distil::teacher_evidence::CollectOptions {
                    registry,
                    comparisons: comparison,
                    output,
                    runtime,
                    max_tokens,
                    seed: 1729,
                    models,
                },
            )
            .await?;
            print(
                &serde_json::json!({"evidence_kind":bundle.evidence_kind,"teachers":bundle.teachers.len(),
                "samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"production_eligible":false}),
            )?;
        }
        Command::CollectTeacherContent {
            bundle,
            output,
            fresh,
            content,
            maximum_tasks,
            max_tokens,
            context_tokens,
            offline,
            execution,
        } => {
            use lec_distil::teacher_evidence::{self, ContentKind};
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            eprintln!(
                "teacher content collection: cached local weights only; explicit offline={offline}"
            );
            let input = bundle.clone();
            let verified =
                tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&input))
                    .await??;
            let teacher = verified
                .teachers
                .first()
                .context("evidence has no teacher")?;
            let runtime = if let Some(saved) =
                execution.options_from_plan(&output.join("content-policy.json"))?
            {
                saved
            } else {
                execution
                    .options_for_model_with_context(&teacher.model.directory, context_tokens)?
            };
            let content = content
                .iter()
                .map(|c| match c.as_str() {
                    "process" => Ok(ContentKind::Process),
                    "context" => Ok(ContentKind::Context),
                    "tool_trajectory" => Ok(ContentKind::ToolTrajectory),
                    _ => bail!("unsupported content collector {c}"),
                })
                .collect::<Result<Vec<_>>>()?;
            let content_output = output.clone();
            let _result =
                teacher_evidence::collect_content(teacher_evidence::ContentCollectOptions {
                    bundle,
                    output,
                    content,
                    runtime,
                    max_tokens,
                    context_tokens,
                    seed: 1729,
                    maximum_tasks,
                })
                .await?;
            let report: serde_json::Value = serde_json::from_slice(&std::fs::read(
                content_output.join("content-report.json"),
            )?)?;
            print(&report)?;
            ensure!(
                report["healthy_complete"] == true,
                "teacher content collection is incomplete or failed; original observations and valid partial evidence were retained"
            );
        }
        Command::CollectStudentRollouts {
            bundle,
            output,
            fresh,
            student_base,
            student_checkpoint,
            reuse_student_attempts_from,
            student_kind,
            mode,
            teachers,
            maximum_tasks,
            max_tokens,
            teacher_context_tokens,
            seed,
            student_device,
            student_gpu,
            offline,
            execution,
        } => {
            use lec_distil::{
                distillation::DistillationStudent,
                teacher_evidence::{self, RolloutMode},
            };
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            eprintln!(
                "student rollout collection: cached local weights only; explicit offline={offline}; frozen-student batch then offline updates"
            );
            let kind = match student_kind.as_str() {
                "conventional_qwen_lora" => DistillationStudent::ConventionalQwenLora,
                "ternary_bitnet" => DistillationStudent::TernaryBitnet,
                _ => bail!("unknown rollout student kind"),
            };
            let mode = match mode.as_str() {
                "on_policy" => RolloutMode::OnPolicy,
                "self_distillation" => RolloutMode::SelfDistillation,
                _ => bail!("unknown rollout mode"),
            };
            let input = bundle.clone();
            let verified =
                tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&input))
                    .await??;
            let policy = output.join("rollout-policy.json");
            let saved = if policy.exists() {
                Some(serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(policy)?,
                )?)
            } else {
                None
            };
            let student_runtime = if let Some(saved) = &saved {
                let runtime: model::RuntimeOptions =
                    serde_json::from_value(saved["student_runtime"].clone())?;
                if let Some(device) = student_device.as_deref() {
                    ensure!(
                        (device == "cpu" && runtime.device == model::ExecutionDevice::Cpu)
                            || (device == "cuda"
                                && runtime.device
                                    == model::ExecutionDevice::Cuda {
                                        ordinal: student_gpu
                                    }),
                        "requested rollout student device differs from frozen policy"
                    );
                }
                runtime
            } else {
                model::RuntimeOptions {
                    device: if student_device.as_deref() == Some("cuda") {
                        model::ExecutionDevice::Cuda {
                            ordinal: student_gpu,
                        }
                    } else {
                        model::ExecutionDevice::Cpu
                    },
                    dtype: model::InferenceDtype::F32,
                }
            };
            let teacher_runtime = if let Some(saved) = saved {
                execution
                    .frozen_options(serde_json::from_value(saved["teacher_runtime"].clone())?)?
            } else if mode == RolloutMode::SelfDistillation
                && kind == DistillationStudent::TernaryBitnet
            {
                execution.frozen_options(model::RuntimeOptions::default())?
            } else {
                let directory = if mode == RolloutMode::SelfDistillation {
                    student_base.as_path()
                } else {
                    verified
                        .teachers
                        .iter()
                        .find(|t| teachers.contains(&t.id))
                        .context("no named correction teacher in verified evidence")?
                        .model
                        .directory
                        .as_path()
                };
                execution.options_for_model_with_context(directory, teacher_context_tokens)?
            };
            let rollout_output = output.clone();
            let _result =
                teacher_evidence::collect_rollouts(teacher_evidence::RolloutCollectOptions {
                    bundle,
                    output,
                    student_base,
                    student_checkpoint,
                    student_kind: kind,
                    mode,
                    teacher_ids: teachers,
                    teacher_runtime,
                    teacher_context_tokens,
                    student_runtime,
                    max_tokens,
                    seed,
                    maximum_tasks,
                    reuse_student_attempts_from,
                })
                .await?;
            let report: serde_json::Value = serde_json::from_slice(&std::fs::read(
                rollout_output.join("rollout-report.json"),
            )?)?;
            print(&report)?;
            ensure!(
                report["healthy_complete"] == true,
                "student rollout collection is incomplete or failed; original observations and valid partial evidence were retained"
            );
        }
        Command::StudyPlan {
            config,
            bundle,
            base,
            bitnet_base,
            output,
            maximum_parallel_jobs,
            offline,
            fresh,
            execution,
        } => {
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            eprintln!("study planning: cached verified evidence only; explicit offline={offline}");
            let input = bundle.clone();
            let verified = tokio::task::spawn_blocking(move || {
                lec_distil::teacher_evidence::load_verified(&input)
            })
            .await??;
            let teacher = verified
                .teachers
                .first()
                .context("evidence has no teacher")?;
            let runtime = if output.join("plan.json").is_file() {
                execution.frozen_options(
                    lec_distil::experiments::load_plan(&output.join("plan.json"))?
                        .inputs
                        .teacher_runtime,
                )?
            } else {
                execution.options_for_model(&teacher.model.directory)?
            };
            let plan = tokio::task::spawn_blocking(move || {
                lec_distil::study::create_plan(lec_distil::study::PlanOptions {
                    config,
                    evidence_bundle: bundle,
                    base,
                    bitnet_base,
                    output,
                    maximum_parallel_jobs,
                    teacher_runtime: runtime,
                })
            })
            .await??;
            print(
                &serde_json::json!({"catalogue_arms":plan.arms.len(),"applicable_arms":plan.arms.iter().filter(|a|a.applicable).count(),
                "quick_selected":plan.quick_arms,"quick_coverage_gaps":plan.quick_coverage_gaps,
                "held_out_used":false,"production_eligible":false}),
            )?;
        }
        Command::StudyRegrade {
            original,
            output,
            expected_generations,
            maximum_concurrent_graders,
            verify_only,
            offline,
            fresh,
        } => {
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            eprintln!(
                "study-regrade: immutable local generations only; no model loading or regeneration; offline={offline}"
            );
            let report = if verify_only {
                tokio::task::spawn_blocking(move || {
                    lec_distil::study_evaluate::regrade::verify_batch(&output)
                })
                .await??
            } else {
                lec_distil::study_evaluate::regrade::run_batch(
                    lec_distil::study_evaluate::regrade::RegradeBatchOptions {
                        originals: original,
                        output,
                        expected_generations,
                        maximum_concurrent_graders,
                        sandbox: lec_distil::sandbox::SandboxConfig {
                            controller_version: 3,
                            ..Default::default()
                        },
                    },
                )
                .await?
            };
            print(&report)?;
            ensure!(
                report["healthy_complete"] == true,
                "derived regrading retains original generation or new infrastructure failures; complete evidence preserved"
            );
        }
        Command::StudyVerify { evaluation } => {
            let report = tokio::task::spawn_blocking(move || {
                lec_distil::study_evaluate::verify_completed(&evaluation)
            })
            .await??;
            print(&report)?;
            ensure!(
                report["healthy_complete"] == true,
                "verified evaluation retains generation or infrastructure failures; original evidence preserved"
            );
        }
        Command::StudyCompare {
            baseline,
            candidate,
        } => {
            let comparisons = tokio::task::spawn_blocking(move || {
                lec_distil::study_evaluate::compare_many(&baseline, &candidate)
            })
            .await??;
            let valid = !comparisons.is_empty()
                && comparisons
                    .iter()
                    .all(|row| row["quality_comparison_valid"] == true);
            print(
                &serde_json::json!({"schema_version":1,"evidence_kind":"independently_verified_descriptive_study_comparisons","status":if valid {"descriptive_comparison_complete"}else{"invalid_quality_comparison"},"all_quality_comparisons_valid":valid,"comparisons":comparisons,"inference_performed":false,"containers_started":false,"held_out_used":false,"production_eligible":false}),
            )?;
            ensure!(
                valid,
                "quality comparison is invalid; printed descriptive observations cannot establish improvement"
            );
        }
        Command::StudyEvaluate {
            base,
            checkpoint,
            student,
            bitnet_engine_manifest,
            bitnet_gguf,
            bitnet_parity_report,
            output,
            split,
            offline,
            fresh,
            execution,
        } => {
            let bitnet_backend = if bitnet_engine_manifest.is_some()
                || bitnet_gguf.is_some()
                || bitnet_parity_report.is_some()
            {
                ensure!(
                    student == "bitnet",
                    "BitNet backend flags require --student bitnet"
                );
                Some(lec_distil::bitnet::BitNetBackendOptions {
                    engine_manifest: bitnet_engine_manifest,
                    gguf: bitnet_gguf,
                    parity_report: bitnet_parity_report,
                })
            } else {
                None
            };
            let architecture = if student == "bitnet" {
                lec_distil::experiments::StudentArchitecture::TernaryBitnet
            } else {
                lec_distil::experiments::StudentArchitecture::ConventionalQwenLora
            };
            let base = base.unwrap_or_else(|| {
                PathBuf::from(if student == "bitnet" {
                    "offline_bundle/students/bitnet-2b-4t"
                } else {
                    "offline_bundle/base"
                })
            });
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            let saved_runtime = execution.options_from_plan(&output.join("plan.json"))?;
            let runtime = if let Some(saved) = saved_runtime {
                saved
            } else if student == "bitnet" {
                execution.frozen_options(model::RuntimeOptions::default())?
            } else {
                execution.options_for_model(&base)?
            };
            let hardware = tokio::task::spawn_blocking(platform::detect).await??;
            let cpu_threads = std::env::var("LEC_STUDY_CPU_THREADS")
                .ok()
                .map(|value| value.parse::<usize>())
                .transpose()
                .context("invalid assigned evaluation CPU thread count")?
                .unwrap_or(hardware.available_cpu_threads);
            ensure!(
                cpu_threads > 0 && cpu_threads <= hardware.available_cpu_threads,
                "assigned evaluation CPU threads exceed available affinity/cgroup capacity"
            );
            let report =
                lec_distil::study::evaluate_checkpoint(lec_distil::study::CheckpointEvalOptions {
                    base,
                    checkpoint,
                    architecture,
                    runtime,
                    output,
                    split: if split == "train" {
                        lec_distil::study_evaluate::StudySplit::Train
                    } else {
                        lec_distil::study_evaluate::StudySplit::Development
                    },
                    cpu_threads,
                    offline,
                    bitnet_backend,
                })
                .await?;
            print(&report)?;
            ensure!(
                report["healthy_complete"] == true,
                "local checkpoint evaluation has generation/infrastructure errors; observations retained"
            );
        }
        Command::StudyRun {
            study,
            profile,
            output,
            mut selected_arms,
            mut selection_evidence,
            maximum_run_seconds,
            offline,
            fresh,
        } => {
            let output = output.unwrap_or_else(|| study.join(&profile));
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            let root = std::env::current_dir()?;
            if profile == "full" && selected_arms.is_empty() {
                let path = selection_evidence.get_or_insert_with(|| study.join("selection.json"));
                let selection: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
                selected_arms = serde_json::from_value(selection["selected_arms"].clone())?;
            }
            let selection_evidence = selection_evidence
                .map(|path| artefact::Artefact::capture(&root, path))
                .transpose()?;
            let pinned = tokio::task::spawn_blocking(move || {
                lec_distil::engine::pin_executable(
                    &std::env::current_exe()?,
                    Path::new("state/engines"),
                )
            })
            .await??;
            let report = lec_distil::experiments::run(lec_distil::experiments::RunOptions {
                plan: study.join("plan.json"),
                output,
                fidelity: if profile == "quick" {
                    lec_distil::experiments::Fidelity::Quick
                } else {
                    lec_distil::experiments::Fidelity::Full
                },
                selected_arms,
                selection_evidence,
                worker_executable: pinned.path,
                worker_arguments: vec!["distillation-arm-worker".into()],
                offline,
                maximum_run_seconds,
            })
            .await?;
            print(&report)?;
            ensure!(
                report["selected_profile_succeeded"] == true,
                "study contains failed or budget-stopped work; progress and reports retained"
            );
        }
        Command::DistillationArmWorker { request, output } => {
            print(&lec_distil::study::run_worker(request, output).await?)?;
        }
        Command::StudyEvaluateWorker {
            request,
            training_artifact,
            output,
        } => {
            print(&lec_distil::study::evaluate_worker(request, training_artifact, output).await?)?;
        }
        Command::StudySelect { study, top_groups } => {
            let report = tokio::task::spawn_blocking(move || {
                lec_distil::study::select(
                    &study.join("plan.json"),
                    &study.join("quick"),
                    &study.join("selection.json"),
                    top_groups,
                )
            })
            .await??;
            print(&report)?;
            ensure!(
                report["selected_arms"]
                    .as_array()
                    .is_some_and(|arms| !arms.is_empty())
                    && report["selection_available"] != false,
                "no validated development group is available for selection; descriptive observations and exclusion reasons were retained"
            );
        }
        Command::Preflight => {
            let p = tokio::task::spawn_blocking(preflight::inspect).await??;
            artefact::write_json(Path::new("reports/preflight.json"), &p)?;
            print(&p)?;
        }
        Command::Tasks { output } => {
            print(&tasks::write_corpus(&output)?)?;
        }
        Command::BootstrapOnline { base, runtime } => {
            let (model_result, image_result) = tokio::join!(
                model::bootstrap(&base),
                sandbox::build_image(Path::new("."), &runtime)
            );
            let report = serde_json::json!({"model":model_result.as_ref().map(|a|serde_json::to_value(a).unwrap()).map_err(|e|format!("{e:#}")),"sandbox_image":image_result.as_ref().map_err(|e|format!("{e:#}"))});
            artefact::write_json(Path::new("reports/bootstrap.json"), &report)?;
            print(&report)?;
            model_result?;
            image_result?;
        }
        Command::HubDownload {
            model,
            revision,
            destination,
            concurrency,
            token_stdin,
        } => {
            print(
                &lec_distil::hub::download_model(lec_distil::hub::HubDownloadOptions {
                    model_id: model,
                    revision,
                    destination,
                    concurrency,
                    token_stdin,
                })
                .await?,
            )?;
        }
        Command::CompareModels {
            config,
            output,
            split,
            models,
            max_tokens,
            seed,
            execution,
        } => {
            let roster = comparative::Registry::load(&config)?;
            let selected = roster
                .models
                .iter()
                .filter(|model| models.is_empty() || models.contains(&model.id))
                .collect::<Vec<_>>();
            ensure!(!selected.is_empty(), "no models selected");
            let mut runtime_options = execution.options_from_plan(&output.join("plan.json"))?;
            if runtime_options.is_none() {
                for model in selected {
                    let options = execution.options_for_model(&model.directory)?;
                    if let Some(current) = &runtime_options {
                        ensure!(
                            *current == options,
                            "models require different hardware/precision; compare them in separate groups or use select-teachers"
                        );
                    } else {
                        runtime_options = Some(options);
                    }
                }
            }
            let split = if split == "train" {
                comparative::ComparisonSplit::Train
            } else {
                comparative::ComparisonSplit::Development
            };
            let report = comparative::run(comparative::CompareOptions {
                registry: config,
                output,
                split,
                runtime: runtime_options.context("missing selected runtime")?,
                max_tokens,
                seed,
                models,
            })
            .await?;
            print(&report)?;
            ensure!(
                report["coverage_complete"] == true,
                "comparative evaluation has missing observations"
            );
            ensure!(
                report["models"].as_array().is_some_and(|rows| rows
                    .iter()
                    .all(|row| row["infrastructure_or_generation_errors"] == 0)),
                "comparative evaluation contains generation/infrastructure failures; see the full denominator report"
            );
        }
        Command::SandboxPreflight => {
            let report = sandbox::preflight(&SandboxConfig::default()).await?;
            artefact::write_json(Path::new("reports/sandbox-preflight.json"), &report)?;
            print(&report)?;
            ensure!(report.probes_passed, "sandbox enforcement preflight failed");
        }
        Command::Grade { task, code } => {
            let task = tasks::corpus()
                .into_iter()
                .find(|t| t.id == task)
                .context("unknown task ID")?;
            let cfg = verified_sandbox().await?;
            let report = sandbox::grade(&task, &std::fs::read_to_string(code)?, &cfg).await?;
            print(&report)?;
            ensure!(report.passed, "candidate failed or was quarantined");
        }
        Command::Collect {
            offline,
            live,
            authorised_usd,
            config,
            round,
        } => {
            ensure!(offline ^ live, "select exactly one of --offline or --live");
            collect(offline, authorised_usd, config, round).await?;
        }
        Command::Curate { rows, output } => {
            let input: Vec<curate::TrainingRow> = serde_json::from_slice(&std::fs::read(rows)?)?;
            print(&curate::curate(&input, &output, &heldout_index())?)?;
        }
        Command::Train {
            dataset,
            base,
            config,
            output,
            resume: _,
            fresh,
        } => {
            let output = if fresh {
                fresh_run_output(&output).await?
            } else {
                output
            };
            let cfg = train::TrainConfig::load(&config)?;
            let report = tokio::task::spawn_blocking(move || {
                train::run(&dataset, &heldout_index(), &base, cfg, &output, !fresh)
            })
            .await??;
            print(&report)?;
        }
        Command::Infer {
            base,
            checkpoint,
            prompt,
            max_tokens,
            comparative,
            execution,
        } => {
            ensure!(
                !comparative || checkpoint.is_none(),
                "comparative models cannot use assessment adapters"
            );
            let options = execution.options_for_model(&base)?;
            let generated = tokio::task::spawn_blocking(move || {
                let mut m = if comparative {
                    model::ModelRuntime::load_comparative_with_options(&base, options)?
                } else {
                    model::ModelRuntime::load_with_options(&base, checkpoint.as_deref(), options)?
                };
                eprintln!("inference execution: {}", m.execution_metadata());
                m.generate(model::GenerationRequest {
                    prompt,
                    max_new_tokens: max_tokens,
                    seed: 1729,
                    temperature: 0.2,
                    top_p: 0.95,
                    timeout_seconds: 120,
                })
            })
            .await??;
            print(&generated)?;
            ensure!(
                generated.complete,
                "generation truncated, timed out or failed"
            );
        }
        Command::Evaluate {
            split,
            base,
            checkpoint,
            output,
            max_tokens,
        } => {
            let split = match split.as_str() {
                "train" => Split::Train,
                "development" => Split::Development,
                _ => Split::HeldOut,
            };
            let report = evaluate::run_split(
                base,
                checkpoint,
                std::env::current_dir()?,
                "config/gate.toml".into(),
                output,
                SandboxConfig::default(),
                max_tokens,
                split,
            )
            .await?;
            print(&report)?;
        }
        Command::Gate {
            incumbent,
            candidate,
        } => {
            gate_command(incumbent, candidate).await?;
        }
        Command::Power { quick } => {
            eprintln!("power: simulating the configured source-cluster gate; diagnostic={quick}");
            let mut cfg = GateConfig::load(Path::new("config/gate.toml"))?;
            if quick {
                cfg.power_trials = 100;
                cfg.bootstrap_trials = 1000;
            }
            let report = tokio::task::spawn_blocking(move || {
                gate::simulate_power(
                    &cfg,
                    8,
                    4,
                    &[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                    &[0.1, 0.3, 0.5],
                    &[0.0, 0.25, 0.5],
                )
            })
            .await??;
            artefact::write_json(Path::new("reports/power.json"), &report)?;
            artefact::atomic_write(Path::new("reports/power.md"), report.markdown().as_bytes())?;
            print(&report)?;
        }
        Command::Serve {
            base,
            checkpoint,
            config,
            comparative,
            local_study,
            execution,
        } => {
            ensure!(
                !comparative || checkpoint.is_none(),
                "comparative models cannot use assessment adapters"
            );
            let options = execution.options_for_model(&base)?;
            if comparative {
                serve::run_comparative(base, serve::ServeConfig::load(&config)?, options).await?;
            } else if local_study {
                serve::run_local_study(
                    base,
                    checkpoint.context("local-study checkpoint absent")?,
                    serve::ServeConfig::load(&config)?,
                    options,
                )
                .await?;
            } else {
                serve::run_with_options(
                    base,
                    checkpoint,
                    serve::ServeConfig::load(&config)?,
                    options,
                )
                .await?;
            }
        }
        Command::RouteInit { url, checkpoint_id } => {
            print(
                &serve::initialise_routing(
                    Path::new("state/routing.json"),
                    serve::VersionedEndpoint { url, checkpoint_id },
                )
                .await?,
            )?;
        }
        Command::StageEvaluation { url, checkpoint_id } => {
            print(
                &serve::stage_evaluation(
                    Path::new("state/routing.json"),
                    serve::VersionedEndpoint { url, checkpoint_id },
                )
                .await?,
            )?;
        }
        Command::Canary {
            url,
            checkpoint_id,
            incumbent_evaluation,
            candidate_evaluation,
        } => {
            print(
                &serve::stage_canary(
                    Path::new("state/routing.json"),
                    serve::VersionedEndpoint { url, checkpoint_id },
                    &incumbent_evaluation,
                    &candidate_evaluation,
                    &std::env::current_dir()?,
                    Path::new("config/gate.toml"),
                )
                .await?,
            )?;
        }
        Command::Rollback { reason } => {
            print(&serve::rollback(Path::new("state/routing.json"), &reason)?)?;
        }
        Command::Costs { round } => {
            let l = ledger()?;
            print(&lec_distil::cost::export(
                &l,
                round.as_deref(),
                Path::new("reports"),
            )?)?;
        }
        Command::Drill { offline } => {
            ensure!(offline, "drill currently requires --offline");
            let report = drills::run(Path::new("reports"), &SandboxConfig::default()).await?;
            print(
                &serde_json::json!({"complete":report.complete,"cases":report.cases.iter().map(|c|serde_json::json!({"name":c.name,"status":c.status,"evidence_kind":c.evidence_kind})).collect::<Vec<_>>(),"json":"reports/drill.json","markdown":"reports/drill.md"}),
            )?;
            ensure!(
                report.complete,
                "required drill evidence incomplete; see reports/drill.md"
            );
        }
        Command::Demo {
            offline,
            use_recorded_checkpoint,
        } => {
            ensure!(
                offline,
                "demo requires --offline; collection is a separate explicit command"
            );
            cli_demo::run(use_recorded_checkpoint).await?;
        }
        Command::Package => print(&lec_distil::package::build()?)?,
        Command::FleetDrillChild {
            work_dir,
            unknown_window,
        } => fleet::drill_child(&work_dir, unknown_window).await?,
        Command::InfrastructureFailure => {
            bail!("deliberately injected unhandled infrastructure failure")
        }
    }
    Ok(())
}

async fn fresh_run_output(output: &Path) -> Result<PathBuf> {
    let path = output.to_path_buf();
    let fresh =
        tokio::task::spawn_blocking(move || lec_distil::engine::prepare_fresh_output(&path))
            .await??;
    eprintln!(
        "explicit fresh run: previous evidence retained at {}; new output is {}",
        output.display(),
        fresh.display()
    );
    Ok(fresh)
}

async fn collect(
    offline: bool,
    authorised_usd: Option<String>,
    config_path: PathBuf,
    round: String,
) -> Result<()> {
    eprintln!("collection: START round={round} offline={offline}");
    std::fs::create_dir_all("state")?;
    let lock = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open("state/orchestrator.lock")?;
    use std::os::fd::AsRawFd;
    // Recovery may mark abandoned calls UNKNOWN only while this process owns the
    // orchestrator lease. Another live round must never be mistaken for a crash.
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another orchestrator owns this state directory"
    );
    let mut config = FleetConfig::load(config_path)?;
    config.require_three_businesses()?;
    let transport: Arc<dyn Transport> = if offline {
        Arc::new(ReplayTransport::load("recordings")?)
    } else {
        let cap = usd_nanos(authorised_usd.as_deref().context(
            "live collection requires --authorised-usd from explicit spending authority",
        )?)?;
        ensure!(cap > 0, "live cap must be positive");
        config.budgets.round_nanos = config.budgets.round_nanos.min(cap);
        Arc::new(LiveTransport::new(true)?)
    };
    let sandbox_config = verified_sandbox().await?;
    let l = ledger()?;
    l.recover_interrupted()?;
    let runner = Arc::new(FleetRunner::new(config.clone(), l.clone(), transport)?);
    let tasks = tasks::corpus()
        .into_iter()
        .filter(|t| t.split != Split::HeldOut)
        .collect::<Vec<_>>();
    let providers = config
        .providers
        .iter()
        .filter(|p| p.enabled)
        .map(|p| p.id.clone())
        .collect::<Vec<_>>();
    let planned=tasks.iter().flat_map(|t|providers.iter().enumerate().map(move|(slot,p)|(t.clone(),p.clone(),slot as u32))).map(|(task,provider,slot)|{
        let cap=config.providers.iter().find(|p|p.id==provider).context("missing planned provider")?.max_output_tokens;
        let request=fleet::FleetRequest{task_id:task.id.clone(),source_id:task.source_id.clone(),specification_sha256:task.specification_sha256.clone(),system_prompt:"Return only the requested Rust solve function. Do not attempt network, files or grader access.".into(),prompt:task.prompt(),sample_slot:slot,generation:fleet::GenerationParameters{max_output_tokens:cap,..Default::default()}};
        Ok((task,provider,request))
    }).collect::<Result<Vec<_>>>()?;
    let plan_jobs = planned
        .iter()
        .map(|(_, provider, request)| (provider.clone(), request.clone()))
        .collect::<Vec<_>>();
    // Cost completeness is tied to the immutable intended jobs, including eligible
    // failover request keys, rather than inferred from whichever attempts succeeded.
    l.record_round_plan(&fleet::build_round_plan(&round, &config, &plan_jobs)?)?;
    let round_ref = &round;
    let mut pending = stream::iter(planned.into_iter().map(|(task, provider, request)| {
        let runner = runner.clone();
        let sandbox_config = sandbox_config.clone();
        async move {
            let completion = runner.complete_on(round_ref, &provider, request).await?;
            let code = evaluate::extract_code(&completion.code);
            let report = sandbox::grade(&task, &code, &sandbox_config).await?;
            let proof_path = std::env::current_dir()?
                .join("state/attempt-proofs")
                .join(format!("{}.json", completion.attempt_id));
            let proof_ledger = runner.ledger.clone();
            let proof_id = completion.attempt_id.clone();
            let proof = tokio::task::spawn_blocking(move || proof_ledger.attempt_proof(&proof_id))
                .await??;
            artefact::write_json(&proof_path, &proof)?;
            let proof_sha256 = artefact::file_hash(&proof_path)?;
            let report_path =
                std::env::current_dir()?.join(format!("reports/sandbox/{}.json", report.run_id));
            let recording_path = std::env::current_dir()?.join(format!(
                "state/artifacts/{}.json",
                completion.response_artifact_sha256
            ));
            if completion.real_provider_recording {
                artefact::atomic_write(
                    &PathBuf::from("recordings")
                        .join(format!("{}.json", completion.response_artifact_sha256)),
                    &std::fs::read(&recording_path)?,
                )?;
            }
            let row = serde_json::to_value(curate::TrainingRow {
                task_id: task.id.clone(),
                source_id: task.source_id.clone(),
                lineage_id: task.lineage_id.clone(),
                specification: task.specification.clone(),
                code,
                run_id: report.run_id,
                attempt_id: completion.attempt_id.clone(),
                request_key: completion.request_key,
                provider_id: completion.provider_id,
                training_permission_decision_id: completion.training_permission_decision_id,
                training_permission_approved: completion.training_permission_approved,
                real_provider_recording: completion.real_provider_recording,
                passed: report.passed,
                trace_complete: report.trace_complete,
                quarantine_reasons: report.quarantine_reasons,
                sandbox_report_path: report_path,
                provider_recording_path: recording_path,
                response_artifact_sha256: completion.response_artifact_sha256,
                attempt_proof_path: proof_path,
                attempt_proof_sha256: proof_sha256,
            })?;
            artefact::write_json(
                &PathBuf::from("state/rows").join(format!("{}.json", completion.attempt_id)),
                &row,
            )?;
            Ok::<_, anyhow::Error>(row)
        }
    }))
    .buffer_unordered(config.max_parallel_tasks);
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(15));
    loop {
        let result = tokio::select! {result=pending.next()=>match result{Some(result)=>result,None=>break},_=heartbeat.tick()=>{eprintln!("collection: RUNNING round={round}; completed={}; failed={}",rows.len(),errors.len());continue;}};
        match result {
            Ok(row) => {
                eprintln!("collection: completed {}", row["task_id"]);
                rows.push(row)
            }
            Err(e) => {
                eprintln!("collection: FAILURE {e:#}");
                errors.push(format!("{e:#}"));
            }
        }
        artefact::write_json(Path::new("reports/training-rows.json"), &rows)?;
    }
    l.export_costs(Some(&round), "reports")?;
    artefact::write_json(
        Path::new("reports/collection.json"),
        &serde_json::json!({"round":round,"offline":offline,"completed":rows.len(),"errors":errors}),
    )?;
    ensure!(
        errors.is_empty(),
        "round incomplete: {} failures; progress saved",
        errors.len()
    );
    print(&serde_json::json!({"status":"complete","rows":rows.len(),"offline":offline}))
}

async fn gate_command(incumbent: PathBuf, candidate: PathBuf) -> Result<()> {
    let root = std::env::current_dir()?;
    let resolve = |path: PathBuf, label: &'static str| {
        let root = root.clone();
        async move {
            if path.is_file() {
                let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                if v["evidence_kind"] == "actual_checkpoint_generations" {
                    return Ok::<_, anyhow::Error>(path);
                }
            }
            let checkpoint = if path == Path::new("base") {
                None
            } else {
                let p = if path.is_file() {
                    path.parent()
                        .context("checkpoint parent missing")?
                        .to_owned()
                } else {
                    path
                };
                ensure!(
                    p.join("manifest.json").exists(),
                    "checkpoint manifest missing: {}",
                    p.display()
                );
                Some(p)
            };
            let out = PathBuf::from(format!("reports/{label}-evaluation"));
            evaluate::run(
                "offline_bundle/base".into(),
                checkpoint,
                root,
                "config/gate.toml".into(),
                out.clone(),
                SandboxConfig::default(),
                512,
            )
            .await?;
            Ok(out.join("evaluation.json"))
        }
    };
    let result = async {
        // Validate both inputs before evaluating either. An absent candidate
        // must never consume the sealed base-model evaluation first.
        for path in [&incumbent, &candidate] {
            ensure!(
                path == Path::new("base") || path.exists(),
                "checkpoint or verified evaluation missing: {}",
                path.display()
            );
        }
        let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
        let input_paths=[incumbent.clone(),candidate.clone()];let input_root=root.clone();let input_cfg=cfg.clone();
        let identities=tokio::task::spawn_blocking(move||->Result<Vec<serde_json::Value>>{
            let assets=model::validate_assets(Path::new("offline_bundle/base"))?;let mut identities=Vec::new();
            for path in &input_paths{
                if path==Path::new("base"){identities.push(serde_json::json!({"checkpoint_id":model::MODEL_SHA256}));continue;}
                if path.is_file(){let value:serde_json::Value=serde_json::from_slice(&std::fs::read(path)?)?;if value["evidence_kind"]=="actual_checkpoint_generations"{let evaluation:evaluate::Evaluation=serde_json::from_value(value)?;evaluate::verify(&evaluation,&input_root,Path::new("config/gate.toml"),&input_cfg)?;identities.push(serde_json::json!({"checkpoint_id":evaluation.checkpoint.checkpoint_id,"evaluation_sha256":artefact::file_hash(path)?}));continue;}}
                let directory=if path.is_dir(){path.as_path()}else{path.parent().context("checkpoint directory missing")?};
                let manifest:train::CheckpointManifest=serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
                train::validate_checkpoint(directory,&manifest,&assets)?;
                identities.push(serde_json::json!({"checkpoint_id":manifest.adapter_sha256,"manifest_sha256":artefact::file_hash(&directory.join("manifest.json"))?}));
            }Ok(identities)
        }).await??;
        let configured_power=std::fs::read("reports/power.json").ok().and_then(|b|serde_json::from_slice::<gate::PowerReport>(&b).ok()).is_some_and(|p|serde_json::to_value(&p.gate_configuration).ok()==serde_json::to_value(&cfg).ok());
        if !configured_power {
            ensure!(!Path::new("reports/incumbent-evaluation/preregistration.json").exists()&&!Path::new("reports/candidate-evaluation/preregistration.json").exists(),"configured power evidence missing after held-out access; original evidence or fresh confirmation required");
            eprintln!(
                "gate: computing predeclared detectability sensitivity before checkpoint evaluation"
            );
            let power = tokio::task::spawn_blocking(move || {
                gate::simulate_power(
                    &cfg,
                    8,
                    4,
                    &[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                    &[0.1, 0.3, 0.5],
                    &[0.0, 0.25, 0.5],
                )
            })
            .await??;
            artefact::write_json(Path::new("reports/power.json"), &power)?;
            artefact::atomic_write(Path::new("reports/power.md"), power.markdown().as_bytes())?;
        }
        let plan=serde_json::json!({"incumbent":identities[0],"candidate":identities[1],"gate_configuration_sha256":artefact::file_hash(Path::new("config/gate.toml"))?,"power_report_sha256":artefact::file_hash(Path::new("reports/power.json"))?,"max_new_tokens":512,"temperature":0.2,"top_p":0.95});
        let cached_pair=[&incumbent,&candidate].iter().all(|p|p.is_file()&&std::fs::read(p).ok().and_then(|b|serde_json::from_slice::<serde_json::Value>(&b).ok()).is_some_and(|v|v["evidence_kind"]=="actual_checkpoint_generations"));
        if !cached_pair{cli_demo::demo_freeze(Path::new("reports/gate-request-freeze.json"),&plan,&["incumbent","candidate"])?;}
        let a = resolve(incumbent, "incumbent").await?;
        let b = resolve(candidate, "candidate").await?;
        evaluate::gate_files(&a, &b, &root, Path::new("config/gate.toml"))
    }
    .await;
    match result {
        Ok(report) => {
            artefact::write_json(Path::new("reports/eval.json"), &report)?;
            artefact::atomic_write(
                Path::new("reports/eval.md"),
                format!("{}\nDetectability: see [the predeclared sensitivity report](power.md). Its full configuration, assumptions and Monte Carlo uncertainty are retained in power.json; no unmeasured scalar MDE is implied.\n",report.report.markdown()).as_bytes(),
            )?;
            print(&report)?;
            ensure!(
                report.report.decision != gate::Decision::InvalidEvaluation,
                "invalid evaluation"
            );
        }
        Err(e) => {
            let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
            let report = gate::GateReport::invalid(format!("{e:#}"), &cfg);
            artefact::write_json(Path::new("reports/eval.json"), &report)?;
            artefact::atomic_write(Path::new("reports/eval.md"), report.markdown().as_bytes())?;
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expanded_context_uses_ram_when_only_the_default_context_fits_vram() -> Result<()> {
        let cfg: model::QwenConfig = serde_json::from_str(include_str!(
            "../docs/comparative-models/replacements/qwen-coder-7b-base/config.json"
        ))?;
        // Synthetic weight size/capacity isolates the dispatch boundary; this
        // does not claim an allocation or execution on either architecture.
        let (small_gpu, small_cpu) = quantised_residency(&cfg, 4 << 30, None)?;
        let (large_gpu, large_cpu) = quantised_residency(&cfg, 4 << 30, Some(1536))?;
        ensure!(large_gpu > small_gpu && large_cpu > small_cpu);
        let guarded_midpoint = small_gpu + (large_gpu - small_gpu) / 2;
        let free_gpu = (guarded_midpoint + (128 << 20)) * 10 / 9;
        for architecture in ["x86_64", "aarch64"] {
            let hardware = platform::HardwareProfile {
                architecture: architecture.into(),
                operating_system: "linux".into(),
                available_cpu_threads: 16,
                host_available_ram_bytes: 32 << 30,
                available_ram_bytes: 32 << 30,
                usable_ram_bytes: 28 << 30,
                memory_constraint: "labelled boundary fixture".into(),
                gpus: vec![platform::GpuProfile {
                    backend: "cuda".into(),
                    ordinal: 0,
                    name: "labelled capacity fixture".into(),
                    total_memory_bytes: free_gpu + (1 << 30),
                    free_memory_bytes: free_gpu,
                    compute_capability: None,
                }],
                cuda_backend_compiled: true,
                gpu_probe_note: "synthetic dispatch QA only".into(),
            };
            ensure!(matches!(
                platform::select_runtime(&hardware, "auto", 0, true, small_gpu, small_cpu)?
                    .options
                    .device,
                model::ExecutionDevice::Cuda { ordinal: 0 }
            ));
            let expanded =
                platform::select_runtime(&hardware, "auto", 0, true, large_gpu, large_cpu)?;
            ensure!(matches!(
                expanded.options.device,
                model::ExecutionDevice::Cpu
            ));
            ensure!(expanded.reason.contains("VRAM"));
            ensure!(
                platform::select_runtime(&hardware, "cuda", 0, true, large_gpu, large_cpu).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn requested_context_cannot_be_cropped_to_make_a_model_fit() -> Result<()> {
        let mut cfg: model::QwenConfig = serde_json::from_str(include_str!(
            "../docs/comparative-models/replacements/qwen-coder-7b-base/config.json"
        ))?;
        for invalid in [0, 4097, usize::MAX] {
            ensure!(quantised_residency(&cfg, 4 << 30, Some(invalid)).is_err());
        }
        cfg.max_position_embeddings = 1024;
        ensure!(quantised_residency(&cfg, 4 << 30, Some(1536)).is_err());
        ensure!(quantised_residency(&cfg, 4 << 30, None).is_ok());
        Ok(())
    }

    #[test]
    fn local_study_serving_requires_a_checkpoint_and_cannot_mix_teacher_mode() -> Result<()> {
        ensure!(Cli::try_parse_from(["lec-distil", "serve", "--local-study"]).is_err());
        ensure!(
            Cli::try_parse_from([
                "lec-distil",
                "serve",
                "--local-study",
                "--comparative",
                "--checkpoint",
                "saved/adapter"
            ])
            .is_err()
        );
        let cli = Cli::try_parse_from([
            "lec-distil",
            "serve",
            "--local-study",
            "--checkpoint",
            "saved/adapter",
        ])?;
        ensure!(matches!(
            cli.command,
            Command::Serve {
                local_study: true,
                comparative: false,
                checkpoint: Some(_),
                ..
            }
        ));
        Ok(())
    }
    #[test]
    fn regrade_cli_retains_full_cohort_and_has_no_generation_controls() -> Result<()> {
        let cli = Cli::try_parse_from([
            "lec-distil",
            "study-regrade",
            "--original",
            "saved/base-development",
            "--original",
            "saved/base-train",
            "--offline",
        ])?;
        match cli.command {
            Command::StudyRegrade {
                original,
                expected_generations,
                maximum_concurrent_graders,
                verify_only,
                offline,
                ..
            } => {
                ensure!(
                    original
                        == vec![
                            PathBuf::from("saved/base-development"),
                            PathBuf::from("saved/base-train")
                        ]
                );
                ensure!(
                    expected_generations == 192
                        && maximum_concurrent_graders == 4
                        && !verify_only
                        && offline
                );
            }
            _ => bail!("unexpected regrade command"),
        }
        for flag in [
            "--task",
            "--seed",
            "--max-tokens",
            "--checkpoint",
            "--device",
        ] {
            ensure!(Cli::try_parse_from(["lec-distil", "study-regrade", flag, "1"]).is_err());
        }
        ensure!(matches!(
            Cli::try_parse_from(["lec-distil", "study-regrade", "--verify-only", "--offline"])?
                .command,
            Command::StudyRegrade {
                verify_only: true,
                offline: true,
                ..
            }
        ));
        Ok(())
    }
    #[test]
    fn study_inspection_cli_is_read_only_and_requires_complete_paths() -> Result<()> {
        ensure!(matches!(
            Cli::try_parse_from([
                "lec-distil",
                "study-verify",
                "--evaluation",
                "saved/derived"
            ])?
            .command,
            Command::StudyVerify { .. }
        ));
        let compared = Cli::try_parse_from([
            "lec-distil",
            "study-compare",
            "--baseline",
            "saved/base",
            "--candidate",
            "saved/sft",
            "--candidate",
            "saved/composed",
        ])?;
        ensure!(
            matches!(compared.command,Command::StudyCompare {candidate,..} if candidate.len()==2)
        );
        ensure!(
            Cli::try_parse_from(["lec-distil", "study-compare", "--baseline", "saved/base"])
                .is_err()
        );
        for command in ["study-verify", "study-compare"] {
            ensure!(Cli::try_parse_from(["lec-distil", command, "--device", "cuda"]).is_err());
        }
        Ok(())
    }
    #[tokio::test]
    async fn fresh_collectors_and_evaluations_preserve_original_failure_evidence() -> Result<()> {
        for name in [
            "collect-teacher-content",
            "collect-student-rollouts",
            "study-evaluate",
            "study-regrade",
        ] {
            let root = tempfile::tempdir()?;
            let original = root.path().join("original");
            std::fs::create_dir_all(&original)?;
            std::fs::write(
                original.join("last-failure.json"),
                b"original failed observation",
            )?;
            let mut cli = Cli::try_parse_from([
                "lec-distil",
                name,
                "--output",
                original.to_str().context("fixture path")?,
                "--fresh",
            ])?;
            prepare_fresh_command(&mut cli.command).await?;
            let path = match &cli.command {
                Command::CollectTeacherContent { output, fresh, .. }
                | Command::CollectStudentRollouts { output, fresh, .. }
                | Command::StudyEvaluate { output, fresh, .. }
                | Command::StudyRegrade { output, fresh, .. } => {
                    ensure!(!fresh, "fresh command would create another directory");
                    output.clone()
                }
                _ => anyhow::bail!("unexpected test command"),
            };
            ensure!(path != original && path.parent() == original.parent());
            artefact::atomic_write(&path.join("last-failure.json"), b"new failed observation")?;
            ensure!(
                std::fs::read(original.join("last-failure.json"))?
                    == b"original failed observation"
            );
            prepare_fresh_command(&mut cli.command).await?;
            ensure!(std::fs::read_dir(root.path())?.count() == 2);
        }
        Ok(())
    }
    #[test]
    fn explicit_teacher_context_and_explore_configuration_keep_legacy_defaults() -> Result<()> {
        for (command, flag) in [
            ("collect-teacher-content", "--context-tokens"),
            ("collect-student-rollouts", "--teacher-context-tokens"),
        ] {
            let legacy = Cli::try_parse_from(["lec-distil", command])?;
            match legacy.command {
                Command::CollectTeacherContent { context_tokens, .. } => {
                    ensure!(context_tokens.is_none())
                }
                Command::CollectStudentRollouts {
                    teacher_context_tokens,
                    ..
                } => ensure!(teacher_context_tokens.is_none()),
                _ => bail!("unexpected collector command"),
            }
            let explicit = Cli::try_parse_from(["lec-distil", command, flag, "2048"])?;
            match explicit.command {
                Command::CollectTeacherContent { context_tokens, .. } => {
                    ensure!(context_tokens == Some(2048))
                }
                Command::CollectStudentRollouts {
                    teacher_context_tokens,
                    ..
                } => ensure!(teacher_context_tokens == Some(2048)),
                _ => bail!("unexpected collector command"),
            }
        }
        let legacy = Cli::try_parse_from(["lec-distil", "explore"])?;
        if let Command::Explore {
            search_config,
            study_config,
            ..
        } = legacy.command
        {
            let defaults = lec_distil::explore::ExploreOptions::default();
            ensure!(
                search_config == defaults.search_config && study_config == defaults.study_config
            );
        } else {
            bail!("unexpected exploration command");
        }
        let explicit = Cli::try_parse_from([
            "lec-distil",
            "explore",
            "--search-config",
            "config/new-search.toml",
            "--study-config",
            "config/new-study.toml",
        ])?;
        if let Command::Explore {
            search_config,
            study_config,
            ..
        } = explicit.command
        {
            ensure!(
                search_config == Path::new("config/new-search.toml")
                    && study_config == Path::new("config/new-study.toml")
            );
        } else {
            bail!("unexpected exploration command");
        }
        Ok(())
    }
    #[test]
    fn decimal_spend_is_exact_and_rejects_overflow() {
        assert_eq!(usd_nanos("20.000000001").unwrap(), 20_000_000_001);
        for v in ["NaN", "-1", "1e3", "1.0000000001", "9223372036854775807"] {
            assert!(usd_nanos(v).is_err());
        }
    }
}
