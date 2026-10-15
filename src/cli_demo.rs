//! Offline assessment orchestration: every phase is explicit and artifact-bound.
use super::{collect, heldout_index, ledger, print};
use anyhow::{Context, Result, bail, ensure};
use lec_distil::{
    artefact, curate, drills, evaluate,
    gate::{self, GateConfig},
    model, preflight,
    sandbox::{self, SandboxConfig},
    serve,
    tasks::{self, Split},
    train,
};
use std::path::{Path, PathBuf};

#[path = "demo_checkpoint_paths.rs"]
mod checkpoint_paths;

fn verify_cached_checkpoint_identity(
    cached_id: &str,
    cached_freeze_ms: i64,
    expected_id: &str,
    round_frozen_ms: i64,
) -> Result<()> {
    ensure!(
        cached_id == expected_id && cached_freeze_ms >= round_frozen_ms,
        "cached evaluation predates the round freeze or has different weights"
    );
    Ok(())
}

async fn demo_endpoint(checkpoint: Option<PathBuf>, label: &str) -> Result<serde_json::Value> {
    use std::{
        process::Stdio,
        time::{Duration, Instant},
    };
    let reserved = std::net::TcpListener::bind("127.0.0.1:0")?;
    let address = reserved.local_addr()?;
    drop(reserved);
    let temporary = tempfile::tempdir()?;
    let config_path = temporary.path().join("serving.toml");
    let config = serve::ServeConfig {
        address: address.to_string(),
        queue_capacity: 1,
        maximum_request_bytes: 2048,
        maximum_output_tokens: 128,
        timeout_seconds: 120,
    };
    std::fs::write(&config_path, toml::to_string(&config)?)?;
    let log_path = PathBuf::from(format!("reports/{label}-endpoint.log"));
    let log = std::fs::File::create(&log_path)?;
    let mut command = tokio::process::Command::new(std::env::current_exe()?);
    command
        .arg("serve")
        .arg("--base")
        .arg("offline_bundle/base")
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .kill_on_drop(true);
    if let Some(path) = &checkpoint {
        command.arg("--checkpoint").arg(path);
    }
    let mut child = command.spawn()?;
    let started = Instant::now();
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(130))
        .build()?;
    let url = format!("http://{address}/");
    let health: serve::Health = loop {
        if let Some(status) = child.try_wait()? {
            bail!(
                "{label} endpoint exited before health: {status}; inspect {}",
                log_path.display()
            );
        }
        if let Ok(response) = client.get(format!("{url}health")).send().await
            && response.status().is_success()
        {
            break response.json().await?;
        }
        ensure!(
            started.elapsed() < Duration::from_secs(120),
            "{label} endpoint startup timed out"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    let expected = match checkpoint {
        None => model::MODEL_SHA256.into(),
        Some(path) => {
            let manifest: train::CheckpointManifest =
                serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
            manifest.adapter_sha256
        }
    };
    ensure!(
        health.healthy
            && health.immutable
            && health.checksum_validated
            && health.checkpoint_id == expected,
        "{label} real endpoint health/identity failed"
    );
    let request = model::GenerationRequest {
        prompt: "Write a Rust function add(a: i64, b: i64) -> i64. Return only Rust source code."
            .into(),
        max_new_tokens: 128,
        seed: 1701,
        temperature: 0.,
        top_p: 1.,
        timeout_seconds: 120,
    };
    let generation: model::Generation = client
        .post(format!("{url}v1/generate"))
        .json(&request)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        generation.complete
            && !generation.completion.trim().is_empty()
            && generation.checkpoint_id == expected,
        "{label} endpoint failed to generate complete output"
    );
    let _ = child.kill().await;
    let _ = child.wait().await;
    let report = serde_json::json!({"evidence_kind":"actual_immutable_local_endpoint_smoke","checkpoint_id":expected,"health":health,"generation":generation,"elapsed_seconds":started.elapsed().as_secs_f64(),"log":log_path,"production_routing":"unchanged; smoke is evaluation-only"});
    artefact::write_json(
        &PathBuf::from(format!("reports/{label}-endpoint.json")),
        &report,
    )?;
    Ok(report)
}

async fn demo_evaluation(
    label: &str,
    checkpoint: Option<PathBuf>,
    expected_id: String,
    split: Split,
    round_frozen_ms: i64,
) -> Result<evaluate::Evaluation> {
    let root = std::env::current_dir()?;
    let output = PathBuf::from(format!("reports/{label}-evaluation"));
    let completed = output.join("evaluation.json");
    if completed.exists() {
        let evaluation: evaluate::Evaluation = serde_json::from_slice(&std::fs::read(&completed)?)?;
        let cached = evaluation.clone();
        let check_root = root.clone();
        tokio::task::spawn_blocking(move || {
            let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
            evaluate::verify_split(
                &cached,
                &check_root,
                Path::new("config/gate.toml"),
                &cfg,
                split,
            )
        })
        .await??;
        verify_cached_checkpoint_identity(
            &evaluation.checkpoint.checkpoint_id,
            evaluation.frozen_before_heldout_unix_ms,
            &expected_id,
            round_frozen_ms,
        )?;
        let prereg: evaluate::Preregistration =
            serde_json::from_slice(&std::fs::read(evaluation.preregistration.verify(&root)?)?)?;
        ensure!(
            prereg.inference.max_new_tokens == 512
                && prereg.inference.temperature == 0.2
                && prereg.inference.top_p == 0.95
                && prereg.inference.timeout_seconds == 120,
            "cached inference parameters differ from the frozen demo plan"
        );
        eprintln!("demo: verified cached evaluation {label}");
        return Ok(evaluation);
    }
    ensure!(
        !output.join("preregistration.json").exists(),
        "{label} evaluation was interrupted after preregistration; retain partial evidence, do not silently rerun sealed tasks"
    );
    eprintln!("demo: evaluating actual frozen checkpoint {label}, split={split:?}");
    let evaluated = evaluate::run_split(
        "offline_bundle/base".into(),
        checkpoint,
        root,
        "config/gate.toml".into(),
        output,
        SandboxConfig::default(),
        512,
        split,
    )
    .await?;
    ensure!(
        evaluated.checkpoint.checkpoint_id == expected_id,
        "evaluated checkpoint identity changed"
    );
    Ok(evaluated)
}

async fn demo_development_power(
    incumbent: &evaluate::Evaluation,
    candidate: &evaluate::Evaluation,
) -> Result<String> {
    use std::collections::BTreeMap;
    ensure!(
        incumbent.split == Split::Development && candidate.split == Split::Development,
        "power calibration requires development sources only"
    );
    let input = serde_json::json!({"incumbent_evaluation_sha256":artefact::file_hash(Path::new("reports/base-development-evaluation/evaluation.json"))?,"candidate_evaluation_sha256":artefact::file_hash(Path::new("reports/candidate-development-evaluation/evaluation.json"))?,"gate_configuration_sha256":artefact::file_hash(Path::new("config/gate.toml"))?,"rule":"paired_seed_discordance_plus_source_task_effect_ANOVA_ICC_with_predeclared_sensitivity_grid_v1"});
    let input_hash = artefact::hash(&serde_json::to_vec(&input)?);
    let calibration_path = Path::new("reports/development-calibration.json");
    if calibration_path.exists() {
        let previous: serde_json::Value =
            serde_json::from_slice(&std::fs::read(calibration_path)?)?;
        ensure!(
            previous["input_sha256"] == input_hash
                && previous["power_report_sha256"]
                    == artefact::file_hash(Path::new("reports/power.json"))?,
            "power calibration inputs or artifact changed after freezing"
        );
        return artefact::file_hash(calibration_path);
    }
    for label in [
        "incumbent",
        "candidate",
        "degraded-control",
        "overfit-control",
        "seed-control",
    ] {
        ensure!(
            !PathBuf::from(format!("reports/{label}-evaluation/preregistration.json")).exists(),
            "cannot create a new power calibration after held-out access"
        );
    }
    let base = incumbent
        .outcomes
        .iter()
        .map(|o| ((o.task_id.clone(), o.seed), o.success))
        .collect::<BTreeMap<_, _>>();
    let mut tasks = BTreeMap::<String, BTreeMap<String, Vec<f64>>>::new();
    let mut discordant = 0usize;
    for outcome in &candidate.outcomes {
        let incumbent_pass = *base
            .get(&(outcome.task_id.clone(), outcome.seed))
            .context("unpaired development outcome")?;
        discordant += usize::from(incumbent_pass != outcome.success);
        tasks
            .entry(outcome.source_id.clone())
            .or_default()
            .entry(outcome.task_id.clone())
            .or_default()
            .push(i32::from(outcome.success) as f64 - i32::from(incumbent_pass) as f64);
    }
    ensure!(
        !candidate.outcomes.is_empty() && tasks.len() >= 2,
        "insufficient development observations"
    );
    let q = discordant as f64 / candidate.outcomes.len() as f64;
    let source_task_effects = tasks
        .iter()
        .map(|(source, tasks)| {
            (
                source.clone(),
                tasks
                    .values()
                    .map(|seeds| seeds.iter().sum::<f64>() / seeds.len() as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let source_effects = source_task_effects
        .iter()
        .map(|(s, t)| (s.clone(), t.iter().sum::<f64>() / t.len() as f64))
        .collect::<BTreeMap<_, _>>();
    let mean = source_effects.values().sum::<f64>() / source_effects.len() as f64;
    let source_variance = source_effects
        .values()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / (source_effects.len() - 1) as f64;
    let task_count = source_task_effects.values().map(Vec::len).sum::<usize>();
    let within_ss = source_task_effects
        .iter()
        .map(|(source, task_effects)| {
            task_effects
                .iter()
                .map(|x| (x - source_effects[source]).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    let within_ms = if task_count > tasks.len() {
        within_ss / (task_count - tasks.len()) as f64
    } else {
        0.
    };
    let tasks_per_source = task_count as f64 / tasks.len() as f64;
    let between_ms = source_variance * tasks_per_source;
    let denominator = between_ms + (tasks_per_source - 1.) * within_ms;
    let source_correlation = if denominator > 0. {
        ((between_ms - within_ms) / denominator).clamp(0., 1.)
    } else {
        0.
    };
    let mut discordances = vec![q.max(0.01), 0.10, 0.30, 0.50];
    discordances.sort_by(f64::total_cmp);
    discordances.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let mut correlations = vec![0., source_correlation, 0.25, 0.50];
    correlations.sort_by(f64::total_cmp);
    correlations.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let mut heldout = BTreeMap::<String, usize>::new();
    for task in tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::HeldOut)
    {
        *heldout.entry(task.source_id).or_default() += 1;
    }
    let heldout_tasks = *heldout
        .values()
        .next()
        .context("held-out source corpus missing")?;
    ensure!(
        heldout.values().all(|n| *n == heldout_tasks),
        "power simulator currently requires balanced task sources"
    );
    let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
    let simulation_cfg = cfg.clone();
    let qs = discordances.clone();
    let rhos = correlations.clone();
    let n_sources = heldout.len();
    eprintln!(
        "demo: power calibrated on development; discordance={q:.4}, source ICC estimate={source_correlation:.4}; simulating the configured gate"
    );
    let started = std::time::Instant::now();
    let mut simulation = tokio::task::spawn_blocking(move || {
        gate::simulate_power(
            &simulation_cfg,
            n_sources,
            heldout_tasks,
            &[0., 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            &qs,
            &rhos,
        )
    });
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(15));
    let power = loop {
        tokio::select! {result=&mut simulation=>break result??,_=heartbeat.tick()=>eprintln!("demo: exact gate power simulation running ({:.0}s)",started.elapsed().as_secs_f64())}
    };
    artefact::write_json(Path::new("reports/power.json"), &power)?;
    artefact::atomic_write(Path::new("reports/power.md"), power.markdown().as_bytes())?;
    let calibration = serde_json::json!({"evidence_kind":"actual_development_checkpoint_calibration","input_sha256":input_hash,"input":input,"observed_unix_ms":lec_distil::ledger::unix_millis(),"observations":candidate.outcomes.len(),"tasks":task_count,"sources":source_effects.len(),"discordant_observations":discordant,"discordance":q,"source_effects":source_effects,"source_effect_variance":source_variance,"source_correlation_estimate":source_correlation,"sensitivity_discordances":discordances,"sensitivity_source_correlations":correlations,"power_configuration":cfg,"power_report_sha256":artefact::file_hash(Path::new("reports/power.json"))?,"simulation_elapsed_seconds":started.elapsed().as_secs_f64(),"limitations":["The small, single-author development source set gives a fragile ICC estimate; sensitivity assumptions remain necessary.","Decoding seeds repeat each task and are not counted as independent tasks or sources.","No observed discordance does not establish zero discordance; the simulation retains conservative alternative assumptions.","The predeclared primary candidate and promotion thresholds are unchanged by calibration; held-out outcomes never fit these assumptions."]});
    artefact::write_json(calibration_path, &calibration)?;
    artefact::file_hash(calibration_path)
}

pub(crate) fn demo_freeze(
    path: &Path,
    plan: &serde_json::Value,
    forbidden_prior_evaluations: &[&str],
) -> Result<(String, i64)> {
    let plan_hash = artefact::hash(&serde_json::to_vec(plan)?);
    let timestamp = if path.exists() {
        let frozen: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            frozen["plan_sha256"] == plan_hash && frozen["plan"] == *plan,
            "plan changed after freeze; fresh independent sources are required"
        );
        frozen["frozen_unix_ms"]
            .as_i64()
            .context("freeze timestamp missing")?
    } else {
        for label in forbidden_prior_evaluations {
            ensure!(
                !PathBuf::from(format!("reports/{label}-evaluation/preregistration.json")).exists(),
                "{label} access predates the required immutable plan freeze"
            );
        }
        let timestamp = lec_distil::ledger::unix_millis();
        artefact::write_json(
            path,
            &serde_json::json!({"frozen_unix_ms":timestamp,"plan_sha256":plan_hash,"plan":plan}),
        )?;
        timestamp
    };
    ensure!(timestamp > 0, "invalid plan freeze timestamp");
    Ok((plan_hash, timestamp))
}

async fn demo_actual_round(
    phases: &mut Vec<serde_json::Value>,
    use_recorded_checkpoint: bool,
) -> Result<()> {
    let row_bytes = std::fs::read("reports/training-rows.json")?;
    let data: Vec<curate::TrainingRow> = serde_json::from_slice(&row_bytes)?;
    let canonical = Path::new("data/curated");
    let destination = if canonical.join("manifest.json").exists() {
        PathBuf::from(format!("data/regraded/{}", artefact::hash(&row_bytes)))
    } else {
        canonical.to_owned()
    };
    let curated = curate::curate(&data, &destination, &heldout_index())?;
    let current = curate::validate_dataset(&destination, &heldout_index())?;
    let original = if destination != canonical {
        curate::validate_dataset(canonical, &heldout_index())?
    } else {
        curate::validate_dataset(&destination, &heldout_index())?
    };
    let semantic_hash = |rows: &[curate::TrainingSample]| -> Result<String> {
        let mut content=rows.iter().map(|s|serde_json::json!({"task":s.task_id,"source":s.source_id,"lineage":s.lineage_id,"specification":s.specification_sha256,"prompt":s.prompt,"code":s.code_sha256,"request":s.request_key,"provider":s.provider_id,"permission":s.permission_decision_id,"recording":s.provider_evidence_sha256})).collect::<Vec<_>>();
        content.sort_by_key(|v| v["task"].as_str().unwrap_or_default().to_owned());
        Ok(artefact::hash(&serde_json::to_vec(&content)?))
    };
    ensure!(
        semantic_hash(&current.rows)? == semantic_hash(&original.rows)?,
        "regraded winners differ from the original immutable training data; do not reuse or tune checkpoints on changed evidence"
    );
    let dataset_manifest = original.manifest;
    phases.push(serde_json::json!({"phase":"curation","status":"complete","regraded_manifest":curated.manifest,"active_training_manifest":dataset_manifest,"provenance":"New grading run IDs are retained separately; completed checkpoints retain their original validated training dataset, never fabricated stable run IDs."}));
    let assets =
        tokio::task::spawn_blocking(|| model::validate_assets(Path::new("offline_bundle/base")))
            .await??;
    let plans = [
        (
            "candidate",
            "config/training.toml",
            train::TrainingPurpose::Candidate,
            1701u64,
        ),
        (
            "seed-control",
            "config/training-seed-control.toml",
            train::TrainingPurpose::SeedControl,
            1702,
        ),
        (
            "degraded-control",
            "config/training-degraded-control.toml",
            train::TrainingPurpose::DegradedControl,
            1703,
        ),
        (
            "overfit-control",
            "config/training-overfit-control.toml",
            train::TrainingPurpose::OverfitControl,
            1704,
        ),
    ];
    let mut checkpoints = std::collections::BTreeMap::new();
    let mut config_hashes = std::collections::BTreeMap::new();
    let mut configurations = std::collections::BTreeMap::new();
    for (label, configuration, purpose, seed) in plans {
        let cfg = train::TrainConfig::load(Path::new(configuration))?;
        ensure!(
            cfg.purpose == purpose && cfg.seed == seed,
            "predeclared {label} training purpose/seed differs"
        );
        config_hashes.insert(label.to_owned(), artefact::hash(&serde_json::to_vec(&cfg)?));
        configurations.insert(label.to_owned(), cfg);
    }
    let root = std::env::current_dir()?;
    let selection = checkpoint_paths::select(
        &root,
        use_recorded_checkpoint,
        &checkpoint_paths::TrainingInputs {
            dataset_manifest_sha256: dataset_manifest.manifest_sha256.clone(),
            heldout_index_sha256: dataset_manifest.heldout_index_sha256.clone(),
            base_sha256: model::MODEL_SHA256.into(),
            tokenizer_sha256: model::TOKENIZER_SHA256.into(),
            training_configuration_hashes: config_hashes.clone(),
        },
    )?;
    // Every genuine control is trained and frozen before any held-out access.
    // Sequential model jobs bound RAM; each CPU GEMM uses the parallel backend.
    for (label, _, purpose, seed) in plans {
        let cfg = configurations[label].clone();
        let cfg_hash = &config_hashes[label];
        let directory = selection.paths[label].clone();
        let completed = directory.join("manifest.json");
        let reused = completed.exists();
        let manifest = if reused {
            let manifest: train::CheckpointManifest =
                serde_json::from_slice(&std::fs::read(&completed)?)?;
            train::validate_checkpoint(&directory, &manifest, &assets)?;
            manifest
        } else {
            ensure!(
                !use_recorded_checkpoint,
                "recorded-checkpoint fast mode requires all four actual checkpoint manifests; {label} is missing"
            );
            eprintln!("demo: genuine CPU training {label}; seed={seed}");
            let target = directory.clone();
            let configuration = cfg.clone();
            tokio::task::spawn_blocking(move || {
                train::run(
                    Path::new("data/curated"),
                    &heldout_index(),
                    Path::new("offline_bundle/base"),
                    configuration,
                    &target,
                    true,
                )
            })
            .await??
        };
        ensure!(
            &manifest.config_sha256 == cfg_hash
                && manifest.seed == seed
                && manifest.training_purpose == purpose
                && manifest.target_transform == purpose.target_transform()
                && manifest.steps == cfg.steps
                && manifest.dataset_sha256 == dataset_manifest.data_sha256
                && manifest.dataset_manifest_sha256 == dataset_manifest.manifest_sha256
                && manifest.heldout_index_sha256 == dataset_manifest.heldout_index_sha256,
            "completed {label} checkpoint does not match current immutable data/configuration"
        );
        phases.push(serde_json::json!({"phase":format!("training_{label}"),"status":if reused{if use_recorded_checkpoint{"skipped_using_verified_recorded_checkpoint"}else{"reused_verified_completed_checkpoint_from_application_state"}}else{"complete"},"checkpoint_path":directory,"tensor_sha256":manifest.adapter_sha256,"seed":seed,"purpose":purpose,"steps":manifest.steps}));
        checkpoints.insert(label.to_owned(), manifest);
    }
    let tensor_hashes = checkpoints
        .iter()
        .map(|(name, manifest)| (name.clone(), manifest.adapter_sha256.clone()))
        .collect::<std::collections::BTreeMap<_, _>>();
    selection.verify_recorded_tensors(&tensor_hashes)?;
    let plan = serde_json::json!({"schema_version":1,"dataset_manifest":dataset_manifest,"base_sha256":model::MODEL_SHA256,"tokenizer_sha256":model::TOKENIZER_SHA256,"candidate_selection":"predeclared_candidate_only; controls never replace it","checkpoint_tensor_hashes":checkpoints.iter().map(|(name,m)|(name.clone(),m.adapter_sha256.clone())).collect::<std::collections::BTreeMap<_,_>>(),"training_configuration_hashes":config_hashes,"gate_configuration_sha256":artefact::file_hash(Path::new("config/gate.toml"))?,"corpus_sha256":artefact::hash(&serde_json::to_vec(&tasks::corpus())?),"inference":{"max_new_tokens":512,"temperature":0.2,"top_p":0.95,"timeout_seconds":120},"heldout_evaluations":["incumbent","candidate","degraded-control","overfit-control","seed-control"]});
    let (model_plan_hash, model_frozen_ms) = demo_freeze(
        Path::new("reports/model-freeze.json"),
        &plan,
        &[
            "base-development",
            "candidate-development",
            "incumbent",
            "candidate",
            "degraded-control",
            "overfit-control",
            "seed-control",
        ],
    )?;
    // The model freeze rejects changed tensors before any new evaluation access.
    // Allocation paths are excluded from its identity so exact retraining can
    // reuse previously sealed observations without rewriting their provenance.
    selection.publish(&root, &tensor_hashes)?;
    phases.push(serde_json::json!({"phase":"all_checkpoint_and_primary_candidate_freeze","status":"complete","plan_sha256":model_plan_hash,"frozen_unix_ms":model_frozen_ms}));
    let development_base = demo_evaluation(
        "base-development",
        None,
        model::MODEL_SHA256.into(),
        Split::Development,
        model_frozen_ms,
    )
    .await?;
    let development_candidate = demo_evaluation(
        "candidate-development",
        Some(selection.paths["candidate"].clone()),
        checkpoints["candidate"].adapter_sha256.clone(),
        Split::Development,
        model_frozen_ms,
    )
    .await?;
    let calibration_sha256 =
        demo_development_power(&development_base, &development_candidate).await?;
    phases.push(serde_json::json!({"phase":"development_power_calibration","status":"complete","calibration_sha256":calibration_sha256,"candidate":"unchanged; no hyperparameter or seed selection"}));
    let confirmatory_plan = serde_json::json!({"model_plan_sha256":model_plan_hash,"model_freeze_sha256":artefact::file_hash(Path::new("reports/model-freeze.json"))?,"development_calibration_sha256":calibration_sha256,"power_report_sha256":artefact::file_hash(Path::new("reports/power.json"))?,"primary":"candidate versus base only","controls":"predeclared diagnostic comparisons; never used to select the primary candidate"});
    let freeze_path = Path::new("reports/round-freeze.json");
    let (plan_hash, frozen_ms) = demo_freeze(
        freeze_path,
        &confirmatory_plan,
        &[
            "incumbent",
            "candidate",
            "degraded-control",
            "overfit-control",
            "seed-control",
        ],
    )?;
    phases.push(serde_json::json!({"phase":"calibrated_confirmatory_heldout_freeze","status":"complete","plan_sha256":plan_hash,"frozen_unix_ms":frozen_ms}));
    let jobs = [
        (
            "base-train",
            None,
            model::MODEL_SHA256.to_owned(),
            Split::Train,
        ),
        (
            "overfit-train",
            Some(selection.paths["overfit-control"].clone()),
            checkpoints["overfit-control"].adapter_sha256.clone(),
            Split::Train,
        ),
        (
            "incumbent",
            None,
            model::MODEL_SHA256.to_owned(),
            Split::HeldOut,
        ),
        (
            "candidate",
            Some(selection.paths["candidate"].clone()),
            checkpoints["candidate"].adapter_sha256.clone(),
            Split::HeldOut,
        ),
        (
            "degraded-control",
            Some(selection.paths["degraded-control"].clone()),
            checkpoints["degraded-control"].adapter_sha256.clone(),
            Split::HeldOut,
        ),
        (
            "overfit-control",
            Some(selection.paths["overfit-control"].clone()),
            checkpoints["overfit-control"].adapter_sha256.clone(),
            Split::HeldOut,
        ),
        (
            "seed-control",
            Some(selection.paths["seed-control"].clone()),
            checkpoints["seed-control"].adapter_sha256.clone(),
            Split::HeldOut,
        ),
    ];
    let mut evaluations = std::collections::BTreeMap::new();
    for (label, checkpoint, id, split) in jobs {
        let evaluated = demo_evaluation(label, checkpoint, id, split, frozen_ms).await?;
        phases.push(serde_json::json!({"phase":format!("evaluation_{label}"),"status":"complete","split":split,"checkpoint_id":evaluated.checkpoint.checkpoint_id,"observations":evaluated.outcomes.len()}));
        evaluations.insert(label.to_owned(), evaluated);
    }
    let primary = evaluate::gate_files(
        Path::new("reports/incumbent-evaluation/evaluation.json"),
        Path::new("reports/candidate-evaluation/evaluation.json"),
        &root,
        Path::new("config/gate.toml"),
    )?;
    let degraded = evaluate::gate_files(
        Path::new("reports/incumbent-evaluation/evaluation.json"),
        Path::new("reports/degraded-control-evaluation/evaluation.json"),
        &root,
        Path::new("config/gate.toml"),
    )?;
    let overfit = evaluate::gate_files(
        Path::new("reports/incumbent-evaluation/evaluation.json"),
        Path::new("reports/overfit-control-evaluation/evaluation.json"),
        &root,
        Path::new("config/gate.toml"),
    )?;
    let seed = evaluate::gate_files(
        Path::new("reports/candidate-evaluation/evaluation.json"),
        Path::new("reports/seed-control-evaluation/evaluation.json"),
        &root,
        Path::new("config/gate.toml"),
    )?;
    let source_macro = |evaluation: &evaluate::Evaluation| {
        let mut sources = std::collections::BTreeMap::<String, (usize, usize)>::new();
        for outcome in &evaluation.outcomes {
            let counts = sources.entry(outcome.source_id.clone()).or_default();
            counts.0 += usize::from(outcome.success);
            counts.1 += 1;
        }
        sources
            .values()
            .map(|(pass, n)| *pass as f64 / *n as f64)
            .sum::<f64>()
            / sources.len() as f64
    };
    let train_gain =
        source_macro(&evaluations["overfit-train"]) - source_macro(&evaluations["base-train"]);
    artefact::write_json(Path::new("reports/eval.json"), &primary)?;
    artefact::atomic_write(
        Path::new("reports/eval.md"),
        primary.report.markdown().as_bytes(),
    )?;
    artefact::write_json(
        Path::new("reports/checkpoint-controls.json"),
        &serde_json::json!({"evidence_kind":"verified_actual_checkpoint_controls","round_freeze_sha256":artefact::file_hash(freeze_path)?,"degraded":degraded,"overfit":overfit,"overfit_train_gain":train_gain,"seed_pair":seed,"interpretation":"Observed control patterns are reported honestly; an unexpected pattern leaves its drill incomplete. No seed or target is chosen after seeing held-out results."}),
    )?;
    phases.push(serde_json::json!({"phase":"evaluation_gate","status":"complete","decision":primary.report.decision,"production":"incumbent retained unless separately authorised SHIP canary"}));
    let smoke = demo_endpoint(Some(selection.paths["candidate"].clone()), "candidate").await?;
    phases.push(
        serde_json::json!({"phase":"candidate_serving","status":"complete","evidence":smoke}),
    );
    // NO SHIP is a complete scientific outcome. A missing/invalid gate is not.
    ensure!(
        primary.report.decision != gate::Decision::InvalidEvaluation,
        "invalid primary evaluation"
    );
    Ok(())
}

pub(crate) async fn run(use_recorded_checkpoint: bool) -> Result<()> {
    std::fs::create_dir_all("state")?;
    let round_lease = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open("state/demo.lock")?;
    use std::os::fd::AsRawFd;
    // A complete round owns one publication lease. Independent CLI collection
    // still has its own ledger lease, without mistaking active work for a crash.
    ensure!(
        unsafe { libc::flock(round_lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another demo owns this application state"
    );
    let mut phases = Vec::new();
    let host = tokio::task::spawn_blocking(preflight::inspect).await??;
    artefact::write_json(Path::new("reports/preflight.json"), &host)?;
    phases.push(serde_json::json!({"phase":"host_inventory","status":"observed","assessment_ready":host.assessment_ready}));
    let corpus = tasks::write_corpus(Path::new("."))?;
    phases.push(serde_json::json!({"phase":"corpus","status":"complete","evidence":corpus}));
    let model_result =
        tokio::task::spawn_blocking(|| model::validate_assets(Path::new("offline_bundle/base")))
            .await?;
    let assets_ready = model_result.is_ok();
    phases.push(serde_json::json!({"phase":"base_assets","status":if assets_ready{"complete"}else{"failed"},"error":model_result.err().map(|e|format!("{e:#}"))}));
    // Base endpoint QA remains useful even while provider credentials or isolation
    // are unavailable. It never substitutes for a trained assessment checkpoint.
    let endpoint_ok = if assets_ready {
        let endpoint = demo_endpoint(None, "base").await;
        let ready = endpoint.is_ok();
        phases.push(serde_json::json!({"phase":"base_serving_qa","status":if endpoint.is_ok(){"complete"}else{"failed"},"evidence":endpoint.as_ref().ok(),"error":endpoint.as_ref().err().map(|e|format!("{e:#}"))}));
        ready
    } else {
        phases.push(serde_json::json!({"phase":"base_serving_qa","status":"blocked","reason":"pinned base assets are absent or invalid"}));
        false
    };
    let image_config = SandboxConfig::default();
    let image_ok = match sandbox::policy_evidence(&image_config).await {
        Ok((policy, image_id)) => {
            phases.push(serde_json::json!({"phase":"offline_container_image","status":"verified_existing","image_id":image_id,"policy_sha256":policy}));
            true
        }
        Err(existing_error) => {
            let restored =
                sandbox::restore_image_bundle(Path::new("offline_bundle/images"), &image_config)
                    .await;
            phases.push(serde_json::json!({"phase":"offline_container_image","status":if restored.is_ok(){"restored_verified_bundle"}else{"failed"},"image_id":restored.as_ref().ok(),"existing_image_error":format!("{existing_error:#}"),"error":restored.as_ref().err().map(|e|format!("{e:#}"))}));
            restored.is_ok()
        }
    };
    let collection = collect(
        true,
        None,
        "config/fleet.toml".into(),
        "offline-demo".into(),
    )
    .await;
    let collected = collection.is_ok();
    phases.push(serde_json::json!({"phase":"recorded_fleet_and_grading","status":if collected{"complete"}else{"failed"},"error":collection.err().map(|e|format!("{e:#}"))}));
    let pipeline = if collected && assets_ready {
        demo_actual_round(&mut phases, use_recorded_checkpoint).await
    } else {
        Err(anyhow::anyhow!(
            "upstream real recordings, rights-cleared dataset, assets or sandbox enforcement unavailable"
        ))
    };
    let pipeline_ok = pipeline.is_ok();
    if let Err(error) = pipeline {
        let reason = format!("{error:#}");
        let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
        let invalid =
            gate::GateReport::invalid(format!("Full round unavailable or failed: {reason}"), &cfg);
        let eval_path = Path::new("reports/eval.json");
        if eval_path.exists() {
            let previous = std::fs::read(eval_path)?;
            let archive = PathBuf::from(format!(
                "reports/previous-eval-{}.json",
                artefact::hash(&previous)
            ));
            if !archive.exists() {
                artefact::atomic_write(&archive, &previous)?;
            }
        }
        artefact::write_json(eval_path, &invalid)?;
        artefact::atomic_write(Path::new("reports/eval.md"), invalid.markdown().as_bytes())?;
        phases.push(serde_json::json!({"phase":"actual_training_evaluation_pipeline","status":"blocked_or_failed","reason":reason}));
        for name in [
            "missing_training_or_controls",
            "missing_real_checkpoint_evaluations",
            "missing_verified_gate_or_candidate_serving",
        ] {
            phases.push(serde_json::json!({"phase":name,"status":"blocked","reason":"missing upstream real evidence is never replaced with fixtures"}));
        }
    }
    let drill = drills::run(Path::new("reports"), &SandboxConfig::default()).await?;
    let complete = pipeline_ok && drill.complete && endpoint_ok && image_ok;
    phases.push(serde_json::json!({"phase":"nine_drills","status":if drill.complete{"complete"}else{"incomplete"}}));
    lec_distil::cost::export(&ledger()?, None, Path::new("reports"))?;
    phases.push(serde_json::json!({"phase":"disconnected_clean_directory_rehearsal","status":"external_rehearsal_required","reason":"This invocation is not evidence that it ran in a newly unpacked directory with outbound access disabled."}));
    let report = serde_json::json!({"status":if complete{"COMPLETE"}else{"INCOMPLETE"},"offline":true,"training_fast_mode":use_recorded_checkpoint,"phases":phases,"promotion":"only_verified_gate_can_authorise"});
    artefact::write_json(Path::new("reports/demo.json"), &report)?;
    print(&report)?;
    ensure!(
        complete,
        "full offline assessment incomplete; inspect reports/demo.json and reports/drill.md"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_evaluation_requires_same_tensors_and_consistent_freeze() {
        // Storage allocation does not enter tensor identity: an exact repeat
        // can use original evidence without rewriting its checkpoint path.
        let tensor = artefact::hash(b"actual immutable tensor bytes");
        verify_cached_checkpoint_identity(&tensor, 200, &tensor, 100).unwrap();
        verify_cached_checkpoint_identity(&tensor, 100, &tensor, 100).unwrap();
        assert!(verify_cached_checkpoint_identity(&tensor, 99, &tensor, 100).is_err());
        assert!(
            verify_cached_checkpoint_identity(
                &tensor,
                200,
                &artefact::hash(b"different retrained tensor bytes"),
                100,
            )
            .is_err()
        );
    }

    #[test]
    fn retraining_nondeterminism_preserves_original_freeze_and_blocks_access() {
        let root = tempfile::tempdir().unwrap();
        let freeze = root.path().join("model-freeze.json");
        let plan = serde_json::json!({"candidate_tensor_sha256":artefact::hash(b"original tensors"),"primary":"predeclared"});
        let identity = demo_freeze(&freeze, &plan, &[]).unwrap();
        let original = std::fs::read(&freeze).unwrap();
        assert_eq!(demo_freeze(&freeze, &plan, &[]).unwrap(), identity);
        let changed = serde_json::json!({"candidate_tensor_sha256":artefact::hash(b"nondeterministic retraining"),"primary":"predeclared"});
        assert!(demo_freeze(&freeze, &changed, &[]).is_err());
        assert_eq!(std::fs::read(&freeze).unwrap(), original);
    }

    #[test]
    fn evaluation_freeze_reuses_exact_plan_and_rejects_posthoc_change() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("freeze.json");
        let original = serde_json::json!({"candidate":"fixed-before-test","seed":1701});
        let first = demo_freeze(&path, &original, &[]).unwrap();
        assert_eq!(demo_freeze(&path, &original, &[]).unwrap(), first);
        assert!(
            demo_freeze(
                &path,
                &serde_json::json!({"candidate":"selected-after-test","seed":1702}),
                &[]
            )
            .is_err()
        );
        assert_eq!(demo_freeze(&path, &original, &[]).unwrap(), first);
    }
}
