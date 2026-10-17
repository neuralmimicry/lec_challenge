//! Existing verifiers are reused; a report filename is never sufficient proof.
use super::*;
use crate::{
    curate, evaluate,
    fleet::{self, Recording},
    gate::Decision,
    model, train,
};
use std::collections::BTreeSet;

pub(super) fn original_evidence(i: &mut Inspector<'_>) {
    let design = [
        "docs/design.md",
        "docs/design.pdf",
        "docs/design-freeze.txt",
    ];
    let present = design.iter().all(|path| i.root.join(path).is_file());
    i.record("design","Design written first, at most three pages; eight requested topics",
        if present {EvidenceStatus::Observed} else {EvidenceStatus::Incomplete},
        "Frozen design artefacts are inspected separately from executable evidence; page count, chronology and prose quality require the retained design review.",
        "Review docs/assessment-audit.md and the frozen design/PDF; do not rewrite the original freeze.");

    let next = "Supply locally configured API credentials, applicable training rights, exact models/prices and an explicit dollar ceiling; collect all three businesses.";
    match provider_recordings(i) {
        Ok((businesses,completions))=>i.record("providers","Three real independent providers with reusable recordings",
            if businesses>=3 {EvidenceStatus::Verified}else{EvidenceStatus::Incomplete},
            format!("{businesses} real training-eligible provider businesses; {completions} usable checksum-verified completions; three businesses required"),next),
        Err(error)=>i.checked("providers","Three real independent providers with reusable recordings",Err(error),next),
    }

    let dataset_path = i.root.join("data/curated");
    let data = curate::validate_dataset(
        &dataset_path,
        &i.root.join("grader_private/heldout-index.json"),
    )
    .with_context(|| format!("verify provider-derived dataset {}", dataset_path.display()));
    let dataset_manifest = data.as_ref().ok().map(|d| d.manifest.clone());
    i.checked(
        "curation",
        "Real deduplicated/decontaminated run-traceable training set",
        data.map(|d| {
            format!(
                "{} independently verified provider-derived training rows",
                d.rows.len()
            )
        }),
        "Curate actual eligible passing provider runs; local-teacher bundles remain separate.",
    );

    let checkpoint = (|| -> Result<String> {
        let path = i.root.join("checkpoints/candidate");
        let manifest_path = path.join("manifest.json");
        let manifest: train::CheckpointManifest =
            serde_json::from_slice(&fs::read(&manifest_path).with_context(|| {
                format!("read assessment checkpoint {}", manifest_path.display())
            })?)?;
        let dataset = dataset_manifest
            .context("checkpoint lacks an independently validated provider dataset")?;
        ensure!(
            manifest.dataset_sha256 == dataset.data_sha256
                && manifest.dataset_manifest_sha256 == dataset.manifest_sha256,
            "checkpoint does not bind the verified training dataset"
        );
        let assets = model::validate_assets(&i.root.join("offline_bundle/base"))?;
        train::validate_checkpoint(&path, &manifest, &assets)?;
        Ok(format!(
            "Changed/reloadable provider-trained adapter {}; {} steps",
            manifest.adapter_sha256, manifest.steps
        ))
    })();
    i.checked("training","One genuine provider-derived training round and checkpoint",checkpoint,
        "Run the real trainer on the verified provider dataset; retain optimiser, assistant-mask, changed-tensor and reload evidence.");

    let heldout = (|| -> Result<String> {
        let pair = evaluate::gate_files(
            &i.root.join("reports/incumbent-evaluation/evaluation.json"),
            &i.root.join("reports/candidate-evaluation/evaluation.json"),
            i.root,
            &i.root.join("config/gate.toml"),
        ).context("verify reports/incumbent-evaluation/evaluation.json and reports/candidate-evaluation/evaluation.json as the original held-out checkpoint pair")?;
        ensure!(
            pair.report.decision != Decision::InvalidEvaluation,
            "invalid paired checkpoint evaluation"
        );
        ensure!(
            pair.report.t_interval.is_some() && pair.report.source_bootstrap_interval.is_some(),
            "missing required intervals"
        );
        Ok(format!(
            "Independently reconstructed {:?}; {} paired observations",
            pair.report.decision, pair.report.observations
        ))
    })();
    i.checked("gate","Two real checkpoints, held-out decision and interval",heldout,
        "Freeze one primary candidate/decoding plan; evaluate complete source-held-out pairs once. NO SHIP is an acceptable valid result.");

    let power = i.read(Path::new("reports/power.json"));
    match power {
        Ok(value)=>i.record("detectability","Smallest detectable effect under the actual gate",EvidenceStatus::Observed,
            format!("Power report present; evidence_kind={}; development calibration remains a separate requirement",value["evidence_kind"]),
            "Bind the simulation assumptions to complete development measurements and retain sensitivity/Monte Carlo uncertainty; no normal-approximation scalar substitution."),
        Err(error)=>i.checked("detectability","Smallest detectable effect under the actual gate",Err(error),"Produce the predeclared exact-gate power/calibration report before sealed access."),
    }
    drill_checks(i);
    costs(i);
    report_observation(
        i,
        "serving",
        "Adapted checkpoint served locally; canary/rollback evidence",
        "reports/candidate-endpoint.json",
        "actual_immutable_local_endpoint_smoke",
        "Serve the actual assessment adapter and retain matching health/generation identity and rollback evidence.",
    );

    match i.read(Path::new("reports/demo.json")) {
        Ok(value)=>i.record("offline_command","One clean-machine command replays the real loop offline",
            if value["status"]=="COMPLETE" {EvidenceStatus::Observed} else {EvidenceStatus::Incomplete},
            format!("Latest recorded assessment demo status: {}",value["status"]),
            "Complete upstream recordings/training/evaluation/drills, package exact dependencies and run the disconnected extracted archive."),
        Err(error)=>i.checked("offline_command","One clean-machine command replays the real loop offline",Err(error),"Run the complete documented offline command once its genuine inputs exist."),
    }
    match i.read(Path::new("reports/disconnected-qa.json")) {
        Ok(value)=>i.record("disconnected_delivery","Fresh credential-free delivery rehearsal",
            if value["full_assessment_complete"]==true {EvidenceStatus::Observed} else {EvidenceStatus::Incomplete},
            format!("Local QA passed={}; full assessment complete={}. A correctly failing rehearsal is not a complete assessment.",value["quality_assurance_passed"],value["full_assessment_complete"]),
            "Rebuild and independently verify the final exact archive; old inventories/binaries cannot certify current changes."),
        Err(error)=>i.checked("disconnected_delivery","Fresh credential-free delivery rehearsal",Err(error),"Run the supplied disconnected verifier against the final immutable archive."),
    }
    match i.read(Path::new("reports/evidence-walkthrough-manifest.json")) {
        Ok(value)=>{
            let valid=value["duration_seconds"].as_f64().is_some_and(|n|n>0.0 && n<=300.0)
                && value["video_sha256"].as_str().is_some_and(|sha|
                    artefact::file_hash(&i.root.join("reports/evidence-walkthrough.mp4")).is_ok_and(|actual|actual==sha));
            i.record("video","Repository/zip, README/design/reports and video up to five minutes",
                if valid {EvidenceStatus::Observed} else {EvidenceStatus::Invalid},
                format!("Video duration={} seconds; checksum verified={valid}; final snapshot currency remains unverified",value["duration_seconds"]),
                "Refresh delivery/video after final evidence stabilises. Do not publish or send without explicit instructions.");
        }
        Err(error)=>i.checked("video","Repository/zip, README/design/reports and video up to five minutes",Err(error),"Prepare a current truthful video, README/design/reports and archive; keep missing evidence explicit."),
    }
}

fn provider_recordings(i: &mut Inspector<'_>) -> Result<(usize, usize)> {
    let mut businesses = BTreeSet::new();
    let mut completions = 0usize;
    for (seen, entry) in fs::read_dir(i.root.join("recordings"))?.enumerate() {
        ensure!(seen < 100_000, "recording audit limit exceeded");
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let relative = Path::new("recordings").join(entry.file_name());
        let value = i.read(&relative)?;
        let record: Recording = serde_json::from_value(value)?;
        let artifact = i.inputs.last().context("recording identity absent")?;
        ensure!(
            entry.path().file_stem().and_then(|s| s.to_str()) == Some(artifact.sha256.as_str()),
            "recording filename/checksum mismatch"
        );
        record.validate_identity()?;
        ensure!(
            record.origin == "real_provider",
            "fixture/local transport cannot supply real-provider evidence"
        );
        // Failed real responses stay in the inventory, but cannot inflate the
        // count of independent businesses that actually supplied a completion.
        if fleet::parse_recording(&record).is_ok() && record.permission.approved_for_output_training
        {
            businesses.insert(record.business_id);
            completions += 1;
        }
    }
    Ok((businesses.len(), completions))
}

const DRILLS: [&str; 9] = [
    "kill_orchestrator_mid_round",
    "provider_error_then_rate_limit",
    "grader_gaming",
    "escape_attempts",
    "planted_heldout_task",
    "deliberately_worse_checkpoint",
    "train_gain_heldout_loss",
    "seed_only_checkpoints",
    "round_budget_exceeded",
];

fn drill_checks(i: &mut Inspector<'_>) {
    let report = i.read(Path::new("reports/drill.json"));
    if let Err(error) = &report {
        let missing = error.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        });
        for name in DRILLS {
            i.record(&format!("drill_{name}"),name,if missing {EvidenceStatus::Incomplete}else{EvidenceStatus::Invalid},
                format!("{error:#}"),"Restore the original valid drill evidence; missing/malformed reports cannot supply a passed case.");
        }
        return;
    }
    let cases = report
        .as_ref()
        .ok()
        .and_then(|value| value["cases"].as_array());
    for name in DRILLS {
        let matching = cases
            .map(|cases| {
                cases
                    .iter()
                    .filter(|case| case["name"] == name)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let (status, finding) = if matching.len() != 1 {
            (
                if matching.is_empty() {
                    EvidenceStatus::Incomplete
                } else {
                    EvidenceStatus::Invalid
                },
                format!("Expected exactly one case; found {}", matching.len()),
            )
        } else {
            let case = matching[0];
            let kind = case["evidence_kind"].as_str().unwrap_or("");
            let status = match case["status"].as_str() {
                Some("PASSED")
                    if !kind.is_empty()
                        && !kind.contains("synthetic")
                        && !kind.contains("missing_required") =>
                {
                    EvidenceStatus::Observed
                }
                Some("INCOMPLETE") => EvidenceStatus::Incomplete,
                _ => EvidenceStatus::Invalid,
            };
            (
                status,
                format!(
                    "Recorded status={}, evidence_kind={kind}; {}",
                    case["status"],
                    case.pointer("/detail/reason")
                        .and_then(Value::as_str)
                        .unwrap_or("historical assertion requires its original artefact chain")
                ),
            )
        };
        i.record(&format!("drill_{name}"),name,status,finding,
            "Retain actual assertions and full failure denominators. Missing real controls/recordings cannot be supplied by score/transport fixtures.");
    }
}

fn costs(i: &mut Inspector<'_>) {
    match i.read(Path::new("reports/cost-summary.json")) {
        Ok(value)=>{
            let complete=value["acquisition_round_complete"]==true && value["real_businesses"].as_u64().is_some_and(|n|n>=3)
                && value["round_total_usd"].as_f64().is_some_and(|n|n.is_finite()&&n>=0.0);
            i.record("costs","Real cost per task, model, round and improvement point",
                if complete {EvidenceStatus::Observed}else{EvidenceStatus::Incomplete},
                format!("Real businesses={}, acquisition complete={}, total USD={}, proven cost/point={}; nonpositive/inconclusive gains must remain N/A or descriptive",value["real_businesses"],value["acquisition_round_complete"],value["round_total_usd"],value["proven_cost_per_point"]),
                "Reconstruct ledger-derived task/model costs, cash/credit settlements and measured compute. Never invent a price, gain or scalar cost per point.");
        }
        Err(error)=>i.checked("costs","Real cost per task, model, round and improvement point",Err(error),"Produce ledger-derived costs and real compute allocations after genuine acquisition/evaluation."),
    }
}

fn report_observation(
    i: &mut Inspector<'_>,
    id: &str,
    criterion: &str,
    path: &str,
    kind: &str,
    next: &str,
) {
    match i.read(Path::new(path)) {
        Ok(value)=>i.record(id,criterion,if value["evidence_kind"]==kind {EvidenceStatus::Observed}else{EvidenceStatus::Invalid},
            format!("Reported evidence kind={}; historical report is not independently executed by this audit",value["evidence_kind"]),next),
        Err(error)=>i.checked(id,criterion,Err(error),next),
    }
}

pub(super) fn exploration_summary(value: &Value) -> Value {
    let stages=value["stages"].as_object().map(|stages|stages.iter().map(|(name,stage)|json!({"name":name,"status":stage["status"],"healthy":stage["details"]["healthy"]})).collect::<Vec<_>>()).unwrap_or_default();
    let comparisons=value.pointer("/details/baseline_comparisons").and_then(Value::as_array).map(|rows|rows.iter().map(|row|json!({
        "profile":row["profile"],"arm_id":row["arm_id"],"seed":row["training_seed"],"candidate_passed":row["candidate_passed"],
        "baseline_passed":row["baseline_passed"],"planned_tasks":row["planned_tasks"],"reported_quality_comparison_valid":row["quality_comparison_valid"],
        "reported_difference_percentage_points":row["observed_score_delta_percentage_points"],"held_out_used":row["held_out_used"],"assessment_promotion":false
    })).collect::<Vec<_>>()).unwrap_or_default();
    json!({"status":value["status"],"success":value["success"],"held_out_used":value["held_out_used"],
        "original_three_provider_assessment_satisfied":value["original_three_provider_assessment_satisfied"],"stages":stages,"reported_development_comparisons":comparisons,
        "interpretation":"Development observations only. One pass versus zero in sixteen development tasks is not a significant held-out gain. Stage report verification is not successful generation/training."})
}

/// Reconstruct the original failed child behind a parent's generic missing-file
/// error. Every traversed state/receipt/log remains bound to its declared hash.
pub(super) fn failed_children(i: &mut Inspector<'_>, report: &Value) -> Vec<Value> {
    let mut failures = vec![];
    let Some(stages) = report["stages"].as_object() else {
        return failures;
    };
    for (name, stage) in stages
        .iter()
        .filter(|(name, _)| name.starts_with("cohort-"))
        .take(64)
    {
        let Some(artifacts) = stage["artefacts"].as_array() else {
            continue;
        };
        for artifact in artifacts {
            let Ok(state) = serde_json::from_value::<Artefact>(artifact.clone()) else {
                continue;
            };
            if state.path.file_name().and_then(|s| s.to_str()) != Some("state.json") {
                continue;
            }
            let result = (|| -> Result<Option<Value>> {
                state.verify(i.root)?;
                let contents = i.read(&state.path)?;
                let Some(events) = contents["events"].as_array() else {
                    return Ok(None);
                };
                let Some(event) = events
                    .iter()
                    .rev()
                    .find(|e| e["stage"] == "rollout-evidence")
                else {
                    return Ok(None);
                };
                let attempt: Artefact = serde_json::from_value(event["attempt"].clone())?;
                attempt.verify(i.root)?;
                let receipt = i.read(&attempt.path)?;
                if receipt["success"] == true {
                    return Ok(None);
                }
                let expected = i
                    .root
                    .join(state.path.parent().context("cohort state has no parent")?)
                    .join("rollouts/rollout-report.json");
                let outcome = crate::explore::read_required_stage_report(
                    i.root,
                    &expected,
                    false,
                    Some(&attempt),
                );
                Ok(Some(
                    json!({"cohort":name,"attempt":attempt,"exit_code":receipt["exit_code"],
                    "recorded_process_success":false,"reason":outcome.err().map(|e|format!("{e:#}"))
                        .unwrap_or_else(||"Failed child supplied a report; inspect its current-batch failure counts.".into())}),
                ))
            })();
            match result {
                Ok(Some(value))=>failures.push(value),
                Ok(None)=>{},
                Err(error)=>failures.push(json!({"cohort":name,"status":"unverifiable_child_evidence","reason":format!("{error:#}")})),
            }
        }
    }
    failures
}
