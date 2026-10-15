//! Thin command boundary: trusted modules own decisions and durable evidence.
mod cli_demo;
use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use futures_util::{StreamExt, stream};
use lec_distil::{
    artefact, curate, drills, evaluate,
    fleet::{self, FleetConfig, FleetRunner, LiveTransport, ReplayTransport, Transport},
    gate::{self, GateConfig},
    ledger::Ledger,
    model, preflight,
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
#[derive(Subcommand)]
enum Command {
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
        #[arg(long)]
        resume: bool,
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
    let cli = Cli::parse();
    if let Err(error) = execute(cli.command).await {
        let failure = serde_json::json!({"status":"FAILURE","error":format!("{error:#}"),"observed_unix_ms":lec_distil::ledger::unix_millis(),"promotion":"incumbent_retained"});
        let _ = artefact::write_json(Path::new("reports/last-failure.json"), &failure);
        eprintln!("FAILURE: {error:#}");
        println!("{failure}");
        std::process::exit(1);
    }
}

async fn execute(command: Command) -> Result<()> {
    match command {
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
            resume,
        } => {
            let cfg = train::TrainConfig::load(&config)?;
            let report = tokio::task::spawn_blocking(move || {
                train::run(&dataset, &heldout_index(), &base, cfg, &output, resume)
            })
            .await??;
            print(&report)?;
        }
        Command::Infer {
            base,
            checkpoint,
            prompt,
            max_tokens,
        } => {
            let generated = tokio::task::spawn_blocking(move || {
                let mut m = model::ModelRuntime::load(&base, checkpoint.as_deref())?;
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
        } => serve::run(base, checkpoint, serve::ServeConfig::load(&config)?).await?,
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
    fn decimal_spend_is_exact_and_rejects_overflow() {
        assert_eq!(usd_nanos("20.000000001").unwrap(), 20_000_000_001);
        for v in ["NaN", "-1", "1e3", "1.0000000001", "9223372036854775807"] {
            assert!(usd_nanos(v).is_err());
        }
    }
}
