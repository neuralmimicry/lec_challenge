//! Observable end-to-end drill assertions. Missing real evidence remains incomplete.
use crate::{
    artefact, curate, evaluate, fleet,
    gate::{self, Decision, GateConfig, Pair},
    sandbox::{self, SandboxConfig},
    tasks::{self, Split},
    train::{self, CheckpointManifest, TrainConfig, TrainingPurpose},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

async fn observed_phase<T>(name: &str, future: impl std::future::Future<Output = T>) -> T {
    let started = std::time::Instant::now();
    eprintln!("drill: START {name}");
    tokio::pin!(future);
    loop {
        tokio::select! {
            result=&mut future=>{eprintln!("drill: DONE {name} elapsed_seconds={:.2}",started.elapsed().as_secs_f64());return result;},
            _=tokio::time::sleep(Duration::from_secs(15))=>eprintln!("drill: RUNNING {name} elapsed_seconds={}",started.elapsed().as_secs()),
        }
    }
}
fn announce(case: Option<&DrillCase>) {
    if let Some(case) = case {
        eprintln!("drill: ASSERT {} {}", case.name, case.status);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DrillCase {
    pub name: String,
    pub status: String,
    pub evidence_kind: String,
    pub detail: Value,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct DrillReport {
    pub schema_version: u32,
    pub complete: bool,
    pub observed_unix_ms: i64,
    pub cases: Vec<DrillCase>,
    pub invariant_qa: Value,
}
fn blocked(name: &str, detail: impl Into<String>) -> DrillCase {
    DrillCase {
        name: name.into(),
        status: "INCOMPLETE".into(),
        evidence_kind: "missing_required_real_evidence".into(),
        detail: json!({"reason": detail.into()}),
    }
}
fn failed(name: &str, kind: &str, error: impl std::fmt::Display) -> DrillCase {
    DrillCase {
        name: name.into(),
        status: "FAILED".into(),
        evidence_kind: kind.into(),
        detail: json!({"error": error.to_string()}),
    }
}
fn passed(name: &str, kind: &str, detail: Value) -> DrillCase {
    DrillCase {
        name: name.into(),
        status: "PASSED".into(),
        evidence_kind: kind.into(),
        detail,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ControlPaths {
    incumbent_heldout: PathBuf,
    candidate_heldout: PathBuf,
    degraded_heldout: PathBuf,
    overfit_heldout: PathBuf,
    seed_heldout: PathBuf,
    incumbent_train: PathBuf,
    overfit_train: PathBuf,
    candidate_training_config: PathBuf,
    seed_training_config: PathBuf,
    degraded_training_config: PathBuf,
    overfit_training_config: PathBuf,
}

pub async fn run(report_dir: &Path, config: &SandboxConfig) -> Result<DrillReport> {
    eprintln!("drill: START all_nine_required_cases");
    std::fs::create_dir_all(report_dir)?;
    let work = report_dir
        .join("drill-work")
        .join(uuid::Uuid::new_v4().to_string());
    std::fs::create_dir_all(&work)?;
    let local = match observed_phase(
        "labelled_fleet_crash_unknown_failover_budget",
        fleet::run_local_drills(&work.join("fleet-fixtures")),
    )
    .await
    {
        Ok(value) => value,
        Err(error) => json!({"status":"FAILURE","reason":format!("{error:#}")}),
    };
    let recorded = match fleet::FleetConfig::load("config/fleet.toml") {
        Ok(config) => observed_phase(
            "real_recorded_fleet_crash_and_failover",
            fleet::run_recorded_drills(
                &work.join("recorded-fleet"),
                config,
                Path::new("recordings"),
            ),
        )
        .await
        .map_err(|e| format!("{e:#}")),
        Err(error) => Err(format!("{error:#}")),
    };
    let mut cases = Vec::new();
    for (name, field) in [
        ("kill_orchestrator_mid_round", "sigkill_resume"),
        ("provider_error_then_rate_limit", "error_then_rate_limit"),
    ] {
        cases.push(match &recorded {
            Ok(evidence) if evidence[field]["passed"] == true => passed(name, "real_recorded_replay_with_labelled_fault_injections", json!({"assertion":evidence[field],"unknown_window":evidence["unknown_window"],"artifacts":evidence["artifact_directory"]})),
            Ok(evidence) => failed(name, "real_recorded_replay", format!("required observable assertion missing: {evidence}")),
            Err(error) => blocked(name, format!("Original real recording replay is incomplete: {error}. Labelled local SIGKILL/UNKNOWN/retry assertions remain in invariant_qa, without substituting for real acquisition evidence.")),
        });
        announce(cases.last());
    }
    let preflight = observed_phase(
        "effective_sandbox_enforcement_preflight",
        sandbox::preflight(config),
    )
    .await;
    let mut frozen = config.clone();
    if let Ok(preflight) = &preflight {
        ensure!(
            frozen.expected_image_id.is_empty() || frozen.expected_image_id == preflight.image_id,
            "drill image differs from caller's frozen identity"
        );
        frozen.expected_image_id = preflight.image_id.clone();
        artefact::write_json(&work.join("sandbox-preflight.json"), preflight)?;
    }
    for (name, security) in [("grader_gaming", false), ("escape_attempts", true)] {
        if preflight.as_ref().is_ok_and(|p| p.probes_passed) {
            let result = if security {
                observed_phase(
                    "exact_network_filesystem_hidden_test_denials",
                    sandbox::drill_security(&frozen),
                )
                .await
            } else {
                observed_phase(
                    "executed_grader_gaming_candidates",
                    sandbox::drill_gaming(&frozen),
                )
                .await
            };
            cases.push(match result {
                Ok(value) if value["passed"] == true => passed(
                    name,
                    "executed_adversarial_candidates_with_trusted_exact_denial_assertions",
                    value,
                ),
                Ok(value) => failed(
                    name,
                    "executed_container_candidates",
                    format!("trusted helper omitted its passed assertion: {value}"),
                ),
                Err(error) => failed(name, "executed_container_candidates", format!("{error:#}")),
            });
        } else {
            cases.push(blocked(
                name,
                format!(
                    "Effective sandbox enforcement preflight did not pass: {}",
                    preflight
                        .as_ref()
                        .map(|p| serde_json::to_value(p).unwrap_or_default())
                        .unwrap_or_else(|e| json!({"error":format!("{e:#}")}))
                ),
            ));
        }
        announce(cases.last());
    }
    let contamination_dir = work.join("contamination");
    cases.push(
        match observed_phase(
            "planted_dataset_training_preflight",
            tokio::task::spawn_blocking(move || planted_dataset_drill(&contamination_dir)),
        )
        .await?
        {
            Ok(value) => passed(
                "planted_heldout_task",
                "temporary_planted_datasets_rejected_by_actual_training_entrypoint",
                value,
            ),
            Err(error) => failed(
                "planted_heldout_task",
                "contamination_fault_injection",
                format!("{error:#}"),
            ),
        },
    );
    announce(cases.last());
    let root = std::env::current_dir()?;
    let control_root = root.clone();
    let actual = observed_phase(
        "actual_checkpoint_control_evidence",
        tokio::task::spawn_blocking(move || actual_control_cases(&control_root)),
    )
    .await?;
    for case in actual {
        cases.push(case);
        announce(cases.last());
    }
    let budget_ok = local["budget_stop"]["passed"] == true
        && local["budget_stop"]["transport_calls"] == 0
        && local["budget_stop"]["progress_saved"] == true;
    cases.push(if budget_ok {
        passed(
            "round_budget_exceeded",
            "deterministic_budget_fault_through_production_reservation_and_no_dispatch_assertions",
            local["budget_stop"].clone(),
        )
    } else {
        failed(
            "round_budget_exceeded",
            "budget_fault_injection",
            "no-dispatch/durability assertion failed or missing",
        )
    });
    announce(cases.last());
    let infrastructure = observed_phase(
        "unhandled_infrastructure_failure_and_unchanged_production",
        infrastructure_failure_drill(&work.join("infrastructure"), &root),
    )
    .await;
    let infrastructure = match infrastructure {
        Ok(value) => value,
        Err(error) => json!({"assertion_passed":false,"error":format!("{error:#}")}),
    };
    let fixture_result = gate_fixtures();
    let fixture_ok = fixture_result.is_ok();
    let fixture_evidence =
        fixture_result.unwrap_or_else(|e| json!({"status":"FAILURE","error":format!("{e:#}")}));
    let qa_ok = infrastructure["assertion_passed"] == true
        && fixture_ok
        && local["sigkill_resume"]["passed"] == true
        && local["unknown_window"]["passed"] == true
        && local["error_then_rate_limit"]["passed"] == true
        && budget_ok;
    let invariant_qa = json!({"all_assertions_passed":qa_ok,"fleet":local,"gate_fixtures":fixture_evidence,"unhandled_infrastructure_error":infrastructure,"evidence_rule":"Fixtures and tensor QA do not establish provider-trained checkpoint performance."});
    ensure!(cases.len() == 9, "drill lost a required case");
    let complete = qa_ok && cases.iter().all(|case| case.status == "PASSED");
    let report = DrillReport {
        schema_version: 2,
        complete,
        observed_unix_ms: crate::ledger::unix_millis(),
        cases,
        invariant_qa,
    };
    artefact::write_json(&report_dir.join("drill.json"), &report)?;
    let mut markdown = format!(
        "# Required drill evidence\n\nOverall: **{}**. Missing evidence is never successful handling.\n\n| Case | Status | Evidence |\n| --- | --- | --- |\n",
        if complete { "COMPLETE" } else { "INCOMPLETE" }
    );
    for case in &report.cases {
        markdown.push_str(&format!(
            "| {} | {} | {} |\n",
            case.name, case.status, case.evidence_kind
        ));
    }
    markdown.push_str("\nThe JSON report retains exact assertions, measured effects, paths and missing artifacts. All nine cases are checked dynamically against the configured real evidence paths. Local transport/score fixtures are separate implementation QA. A missing or unexpected real control pattern remains incomplete; no held-out seed/control search is performed.\n");
    artefact::atomic_write(&report_dir.join("drill.md"), markdown.as_bytes())?;
    Ok(report)
}

fn read_evaluation(path: &Path) -> Result<evaluate::Evaluation> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn require_files(paths: &[&Path]) -> Result<()> {
    let missing: Vec<_> = paths
        .iter()
        .filter(|p| !p.is_file())
        .map(|p| p.display().to_string())
        .collect();
    ensure!(
        missing.is_empty(),
        "missing required real artifacts: {}",
        missing.join(", ")
    );
    Ok(())
}
fn control_manifest(
    path: &Path,
    root: &Path,
    expected: TrainingPurpose,
    configuration: &Path,
) -> Result<CheckpointManifest> {
    let evaluation = read_evaluation(path)?;
    let manifest: CheckpointManifest = serde_json::from_slice(&std::fs::read(
        evaluation.checkpoint.metadata.verify(root)?,
    )?)?;
    let config = TrainConfig::load(configuration)?;
    ensure!(
        manifest.training_purpose == expected && config.purpose == expected,
        "control purpose does not match predeclared role"
    );
    ensure!(
        manifest.config_sha256 == artefact::hash(&serde_json::to_vec(&config)?)
            && manifest.seed == config.seed,
        "control training configuration/seed differs from predeclared configuration"
    );
    ensure!(
        manifest.adapter_sha256 == evaluation.checkpoint.checkpoint_id,
        "evaluation/control tensor mismatch"
    );
    Ok(manifest)
}
fn actual_control_cases(root: &Path) -> Vec<DrillCase> {
    let names = [
        "deliberately_worse_checkpoint",
        "train_gain_heldout_loss",
        "seed_only_checkpoints",
    ];
    let paths = std::fs::read_to_string(root.join("config/drill-controls.toml"))
        .context("control path configuration missing")
        .and_then(|text| toml::from_str::<ControlPaths>(&text).map_err(Into::into));
    let paths = match paths {
        Ok(paths) => paths,
        Err(error) => {
            return names
                .iter()
                .map(|name| blocked(name, format!("{error:#}")))
                .collect();
        }
    };
    let gate_path = root.join("config/gate.toml");
    let worse = (|| -> Result<Value> {
        require_files(&[
            &paths.incumbent_heldout,
            &paths.degraded_heldout,
            &paths.degraded_training_config,
        ])?;
        let control = control_manifest(
            &paths.degraded_heldout,
            root,
            TrainingPurpose::DegradedControl,
            &paths.degraded_training_config,
        )?;
        let report = evaluate::gate_files(
            &paths.incumbent_heldout,
            &paths.degraded_heldout,
            root,
            &gate_path,
        )?;
        let pattern = report.report.effect.is_some_and(|effect| effect < 0.0)
            && report.report.decision == Decision::NoShipRegression;
        Ok(
            json!({"required_pattern_observed":pattern,"actual_checkpoint":control.adapter_sha256,"training_purpose":control.training_purpose,"gate":report,"interpretation":if pattern{"Actual negative held-out effect blocked."}else{"Actual control did not show the required regression; retain its measured result and mark the drill incomplete."}}),
        )
    })();
    let overfit = (|| -> Result<Value> {
        require_files(&[
            &paths.incumbent_heldout,
            &paths.overfit_heldout,
            &paths.incumbent_train,
            &paths.overfit_train,
            &paths.overfit_training_config,
        ])?;
        let control = control_manifest(
            &paths.overfit_heldout,
            root,
            TrainingPurpose::OverfitControl,
            &paths.overfit_training_config,
        )?;
        let heldout = evaluate::gate_files(
            &paths.incumbent_heldout,
            &paths.overfit_heldout,
            root,
            &gate_path,
        )?;
        let cfg = GateConfig::load(&gate_path)?;
        let base = read_evaluation(&paths.incumbent_train)?;
        let candidate = read_evaluation(&paths.overfit_train)?;
        ensure!(
            base.split == Split::Train && candidate.split == Split::Train,
            "train control uses another split"
        );
        evaluate::verify(&base, root, &gate_path, &cfg)?;
        evaluate::verify(&candidate, root, &gate_path, &cfg)?;
        ensure!(
            base.checkpoint.checkpoint_id == heldout.incumbent_checkpoint_id
                && candidate.checkpoint.checkpoint_id == heldout.candidate_checkpoint_id,
            "train/held-out observations use different actual checkpoints"
        );
        let base_plan: evaluate::Preregistration =
            serde_json::from_slice(&std::fs::read(base.preregistration.verify(root)?)?)?;
        let next_plan: evaluate::Preregistration =
            serde_json::from_slice(&std::fs::read(candidate.preregistration.verify(root)?)?)?;
        ensure!(
            base_plan.inference == next_plan.inference
                && base_plan.sandbox_image_id == next_plan.sandbox_image_id
                && base_plan.sandbox_policy_sha256 == next_plan.sandbox_policy_sha256,
            "train control comparison changed inference/security settings"
        );
        let training = paired_diagnostic(&base, &candidate, &cfg)?;
        let pattern = training.effect.is_some_and(|v| v > 0.0)
            && heldout.report.effect.is_some_and(|v| v < 0.0)
            && heldout.report.decision == Decision::NoShipRegression;
        Ok(
            json!({"required_pattern_observed":pattern,"actual_checkpoint":control.adapter_sha256,"training_diagnostic_only":training,"heldout_gate":heldout,"interpretation":if pattern{"Actual train gain and held-out loss; promotion blocked."}else{"The actual overfit intervention did not exhibit the required pattern; no substitute or search is performed."}}),
        )
    })();
    let seed = (|| -> Result<Value> {
        require_files(&[
            &paths.candidate_heldout,
            &paths.seed_heldout,
            &paths.candidate_training_config,
            &paths.seed_training_config,
        ])?;
        let candidate = control_manifest(
            &paths.candidate_heldout,
            root,
            TrainingPurpose::Candidate,
            &paths.candidate_training_config,
        )?;
        let control = control_manifest(
            &paths.seed_heldout,
            root,
            TrainingPurpose::SeedControl,
            &paths.seed_training_config,
        )?;
        let a = TrainConfig::load(&paths.candidate_training_config)?;
        let b = TrainConfig::load(&paths.seed_training_config)?;
        let mut left = serde_json::to_value(&a)?;
        let mut right = serde_json::to_value(&b)?;
        for value in [&mut left, &mut right] {
            value
                .as_object_mut()
                .context("training config schema")?
                .remove("seed");
            value
                .as_object_mut()
                .context("training config schema")?
                .remove("purpose");
        }
        ensure!(
            left == right && a.seed != b.seed,
            "seed control changes non-seed hyperparameters or reuses the seed"
        );
        ensure!(
            candidate.dataset_sha256 == control.dataset_sha256
                && candidate.dataset_manifest_sha256 == control.dataset_manifest_sha256
                && candidate.base_sha256 == control.base_sha256
                && candidate.tokenizer_sha256 == control.tokenizer_sha256
                && candidate.target_transform == control.target_transform
                && candidate.effective_examples == control.effective_examples,
            "seed adapters differ in data/base/tokenizer/targets"
        );
        ensure!(
            candidate.adapter_sha256 != control.adapter_sha256,
            "seed control reused the identical checkpoint artifact"
        );
        let report = evaluate::gate_files(
            &paths.candidate_heldout,
            &paths.seed_heldout,
            root,
            &gate_path,
        )?;
        let interval_null = |interval: Option<gate::Interval>| {
            interval.is_some_and(|i| i.lower <= 0.0 && i.upper >= 0.0)
        };
        let pattern = report.report.decision != Decision::Ship
            && report.report.decision != Decision::InvalidEvaluation
            && interval_null(report.report.t_interval)
            && interval_null(report.report.source_bootstrap_interval);
        Ok(
            json!({"required_pattern_observed":pattern,"predeclared_seeds":[a.seed,b.seed],"gate":report,"interpretation":if pattern{"No significant paired source difference is established by either interval; no win is claimed."}else{"The real seed pair did not support the required null pattern. Seed variation can produce real differences or false positives; retain the result and investigate fresh data."}}),
        )
    })();
    names
        .into_iter()
        .zip([worse, overfit, seed])
        .map(|(name, result)| match result {
            Ok(detail) if detail["required_pattern_observed"] == true => passed(
                name,
                "actual_checkpoint_evaluations_verified_against_tensor_and_raw_execution_evidence",
                detail,
            ),
            Ok(detail) => DrillCase {
                name: name.into(),
                status: "INCOMPLETE".into(),
                evidence_kind: "actual_control_pattern_not_observed".into(),
                detail,
            },
            Err(error) => blocked(name, format!("{error:#}")),
        })
        .collect()
}

fn paired_diagnostic(
    base: &evaluate::Evaluation,
    next: &evaluate::Evaluation,
    cfg: &GateConfig,
) -> Result<gate::GateReport> {
    let previous: BTreeMap<_, _> = base
        .outcomes
        .iter()
        .map(|o| ((o.task_id.as_str(), o.seed), o))
        .collect();
    let pairs = next
        .outcomes
        .iter()
        .map(|o| -> Result<Pair> {
            let b = previous
                .get(&(o.task_id.as_str(), o.seed))
                .context("unpaired diagnostic observation")?;
            Ok(Pair {
                source_id: o.source_id.clone(),
                task_id: o.task_id.clone(),
                seed: o.seed,
                incumbent_pass: b.success,
                candidate_pass: o.success,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        pairs.len() == previous.len(),
        "incomplete diagnostic pairing"
    );
    let report = gate::decide(
        &pairs,
        cfg,
        true,
        "actual_train_diagnostic_not_a_promotion_gate",
    );
    ensure!(
        report.decision != Decision::InvalidEvaluation,
        "invalid train diagnostic: {}",
        report.reason
    );
    Ok(report)
}

fn planted_dataset_drill(work: &Path) -> Result<Value> {
    let corpus_dir = work.join("corpus");
    tasks::write_corpus(&corpus_dir)?;
    let index = corpus_dir.join("grader_private/heldout-index.json");
    let corpus = tasks::corpus();
    let heldout = corpus
        .iter()
        .find(|t| t.split == Split::HeldOut)
        .context("no held-out task")?;
    let training = corpus
        .iter()
        .find(|t| t.split == Split::Train)
        .context("no training task")?;
    let mut observations = Vec::new();
    for (name, task_id, source, lineage, spec) in [
        (
            "exact_heldout_origin",
            heldout.id.clone(),
            heldout.source_id.clone(),
            heldout.lineage_id.clone(),
            heldout.specification.clone(),
        ),
        (
            "reworded_retained_lineage",
            training.id.clone(),
            training.source_id.clone(),
            heldout.lineage_id.clone(),
            "Reworded planted task retaining its actual held-out lineage.".into(),
        ),
        (
            "near_specification_hidden_metadata",
            training.id.clone(),
            training.source_id.clone(),
            training.lineage_id.clone(),
            format!("{} Please implement carefully.", heldout.specification),
        ),
    ] {
        let directory = work.join(name);
        std::fs::create_dir_all(&directory)?;
        let sample = curate::TrainingSample {
            id: format!("LABELLED-PLANTED-{name}"),
            task_id,
            source_id: source.clone(),
            lineage_id: lineage,
            prompt: "LABELLED CONTAMINATION FIXTURE; MUST NEVER TRAIN".into(),
            completion:
                "pub fn solve(_:serde_json::Value)->serde_json::Value {serde_json::Value::Null}"
                    .into(),
            specification: spec,
            run_id: "LABELLED-PLANTED-RUN".into(),
            attempt_id: "LABELLED-PLANTED-ATTEMPT".into(),
            request_key: "LABELLED-PLANTED-KEY".into(),
            provider_id: "LABELLED-PLANTED-PROVIDER".into(),
            permission_decision_id: "UNAPPROVED-FIXTURE".into(),
            specification_sha256: heldout.specification_sha256.clone(),
            code_sha256: "UNVERIFIED".into(),
            sandbox_evidence: "evidence/missing.sandbox.json".into(),
            sandbox_evidence_sha256: "UNVERIFIED".into(),
            provider_evidence: "evidence/missing.provider.json".into(),
            provider_evidence_sha256: "UNVERIFIED".into(),
            attempt_evidence: "evidence/missing.attempt.json".into(),
            attempt_evidence_sha256: "UNVERIFIED".into(),
        };
        let mut data = serde_json::to_vec(&sample)?;
        data.push(b'\n');
        let mut manifest = curate::DatasetManifest {
            schema_version: 1,
            data_sha256: artefact::hash(&data),
            heldout_index_sha256: artefact::file_hash(&index)?,
            sample_count: 1,
            source_ids: vec![source],
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = artefact::hash(&serde_json::to_vec(&manifest)?);
        artefact::atomic_write(&directory.join("dataset.jsonl"), &data)?;
        artefact::write_json(&directory.join("manifest.json"), &manifest)?;
        let output = directory.join("checkpoint-must-not-exist");
        let error = train::run(
            &directory,
            &index,
            &directory.join("base-must-not-be-loaded"),
            TrainConfig::default(),
            &output,
            false,
        )
        .err()
        .context("planted held-out row reached successful training")?;
        let error = format!("{error:#}");
        ensure!(
            error.contains("held-out") && !output.exists(),
            "planted data did not fail at the actual pre-optimiser origin check: {error}"
        );
        observations.push(json!({"variant":name,"rejected_by":"actual_train_entrypoint_before_model_or_optimiser","error":error,"checkpoint_directory_created":false,"dataset":directory}));
    }
    Ok(
        json!({"passed":true,"evidence_kind":"labelled_contamination_fault_injections_in_real_manifest_and_training_entrypoint","optimiser_steps":0,"canonical_dataset_modified":false,"observations":observations,"screening_unit_fixtures":curate::contamination_drill()?}),
    )
}

async fn infrastructure_failure_drill(work: &Path, root: &Path) -> Result<Value> {
    std::fs::create_dir_all(work.join("state"))?;
    let sentinel = work.join("state/routing.json");
    let incumbent =
        b"{\"incumbent\":\"unchanged-infrastructure-drill-sentinel\",\"canary_percent\":0}\n";
    artefact::atomic_write(&sentinel, incumbent)?;
    let actual_route = root.join("state/routing.json");
    let before = if actual_route.exists() {
        Some(std::fs::read(&actual_route)?)
    } else {
        None
    };
    let mut command = tokio::process::Command::new(std::env::current_exe()?);
    command
        .arg("__infrastructure-failure")
        .current_dir(work)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = tokio::time::timeout(Duration::from_secs(20), command.output())
        .await
        .context("infrastructure-failure child hung")??;
    let failure_path = work.join("reports/last-failure.json");
    let failure: Value = serde_json::from_slice(
        &std::fs::read(&failure_path).context("child omitted failure report")?,
    )?;
    let after = if actual_route.exists() {
        Some(std::fs::read(&actual_route)?)
    } else {
        None
    };
    ensure!(
        !child.status.success()
            && !child.stdout.is_empty()
            && failure["status"] == "FAILURE"
            && failure["promotion"] == "incumbent_retained",
        "unhandled infrastructure exception did not fail visibly"
    );
    ensure!(
        std::fs::read(&sentinel)? == incumbent
            && before == after
            && !work.join("checkpoints").exists(),
        "infrastructure failure changed a production pointer or created a checkpoint"
    );
    Ok(
        json!({"assertion_passed":true,"exit_code":child.status.code(),"failure_report":failure_path,"failure":failure,"isolated_routing_unchanged":true,"actual_production_routing_unchanged":true,"checkpoint_created":false,"stdout":String::from_utf8_lossy(&child.stdout)}),
    )
}

fn gate_fixtures() -> Result<Value> {
    let cfg = GateConfig::load(Path::new("config/gate.toml"))?;
    let pairs = |base: bool, next: bool| -> Vec<Pair> {
        (0..8)
            .flat_map(|s| {
                let seeds = cfg.evaluation_seeds.clone();
                (0..4).flat_map(move |t| {
                    seeds.clone().into_iter().map(move |seed| Pair {
                        source_id: format!("fixture-source-{s}"),
                        task_id: format!("fixture-source-{s}-task-{t}"),
                        seed,
                        incumbent_pass: base,
                        candidate_pass: next,
                    })
                })
            })
            .collect()
    };
    let worse = gate::decide(
        &pairs(true, false),
        &cfg,
        true,
        "synthetic_decision_fixture",
    );
    let null_report = gate::decide(&pairs(true, true), &cfg, true, "synthetic_decision_fixture");
    let overfit = gate::decide(
        &pairs(true, false),
        &cfg,
        true,
        "synthetic_heldout_loss_fixture_train_gain_is_irrelevant",
    );
    ensure!(
        worse.decision == Decision::NoShipRegression
            && overfit.decision == Decision::NoShipRegression
            && null_report.decision == Decision::NoShipInconclusive,
        "negative/null gate fixture assertion failed"
    );
    Ok(
        json!({"evidence_kind":"synthetic_decision_fixtures_not_checkpoint_evaluation","worse":worse,"train_gain_heldout_loss":overfit,"null":null_report}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_training_entrypoint_rejects_all_planted_variants_before_loading_assets() {
        let work = tempfile::tempdir().unwrap();
        let report = planted_dataset_drill(work.path()).unwrap();
        assert_eq!(report["optimiser_steps"], 0);
        assert_eq!(report["observations"].as_array().unwrap().len(), 3);
    }
    #[test]
    fn missing_actual_controls_are_not_score_fixture_successes() {
        let work = tempfile::tempdir().unwrap();
        let cases = actual_control_cases(work.path());
        assert_eq!(cases.len(), 3);
        assert!(cases.iter().all(|case| case.status == "INCOMPLETE"));
    }
}
