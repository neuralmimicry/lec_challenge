//! Actual unadapted checkpoints under the same development quality protocol.
//! Resource differences remain visible in descriptive comparisons.
use super::*;
use experiments::{CompletedResult, StudentArchitecture, StudyPlan};

pub(super) async fn run(
    context: &mut ContextState,
    plan: &StudyPlan,
) -> Result<BTreeMap<StudentArchitecture, PathBuf>> {
    let mut baselines = BTreeMap::new();
    for student in &plan.inputs.students {
        let Some(arm) = plan
            .arms
            .iter()
            .find(|arm| arm.applicable && arm.student == student.architecture)
        else {
            continue;
        };
        let tag = serde_json::to_value(student.architecture)?
            .as_str()
            .context("student architecture tag missing")?
            .to_owned();
        let stage_name = format!("baseline-{tag}");
        let output = context.options.output.join("baselines").join(&tag);
        let assets = student
            .base_assets
            .as_ref()
            .context("baseline student assets missing")?;
        let base = assets
            .path
            .parent()
            .context("baseline student directory missing")?;
        let resources = experiments::resources_for(&plan.inputs, arm, experiments::Fidelity::Quick);
        let mut process_ok = true;
        if !context.cached(&stage_name)? {
            let mut args = vec![
                "study-evaluate".into(),
                "--base".into(),
                base.display().to_string(),
                "--student".into(),
                if student.architecture == StudentArchitecture::TernaryBitnet {
                    "bitnet"
                } else {
                    "qwen"
                }
                .into(),
                "--output".into(),
                output.display().to_string(),
                "--split".into(),
                "development".into(),
            ];
            args.extend(runtime_arguments(&plan.inputs.student_runtime));
            if context.options.offline {
                args.push("--offline".into());
            }
            let engine = context.protocol.stage_engine.clone();
            process_ok = process::stage_with_cpu(
                context,
                &stage_name,
                &engine,
                args,
                Some(resources.cpu_threads),
            )
            .await?;
        }
        let checked = output.clone();
        let report =
            tokio::task::spawn_blocking(move || crate::study_evaluate::verify_completed(&checked))
                .await??;
        let frozen = read(&output.join("plan.json"))?;
        ensure!(
            report["split"] == "development"
                && report["architecture"] == serde_json::to_value(student.architecture)?
                && frozen["checkpoint"]["manifest"] == serde_json::to_value(assets)?
                && frozen["checkpoint"]["base_assets"] == serde_json::to_value(assets)?
                && frozen["runtime"] == serde_json::to_value(&plan.inputs.student_runtime)?,
            "cached baseline is not this student's actual unadapted weights and matched runtime"
        );
        ensure!(
            process_ok || report["healthy_complete"] != true,
            "baseline process failed despite a healthy report; inspect retained attempt logs"
        );
        context.publish(&stage_name,"verified_complete",vec![capture(&context.root,&output.join("report.json"))?,capture(&context.root,&output.join("plan.json"))?],
            json!({"architecture":student.architecture,"checkpoint_id":report["checkpoint_id"],"passed":report["passed"],"planned":report["planned"],
                "healthy":report["healthy_complete"],"generation_or_infrastructure_errors":report["infrastructure_or_generation_errors"],
                "assigned_cpu_threads":resources.cpu_threads,"generation_ms":report["summed_generation_ms"],"production_eligible":false})).await?;
        baselines.insert(student.architecture, output);
    }
    ensure!(
        !baselines.is_empty(),
        "no applicable student has an actual baseline evaluation"
    );
    Ok(baselines)
}

pub(super) async fn compare_profile(
    context: &mut ContextState,
    profile: &str,
    study_directory: &Path,
    completed: &[CompletedResult],
    baselines: &BTreeMap<StudentArchitecture, PathBuf>,
) -> Result<Vec<Value>> {
    let state = read(&study_directory.join(profile).join("state.json"))?;
    let events = state["events"]
        .as_array()
        .context("study attempt events missing")?;
    let mut rows = Vec::new();
    for (architecture, baseline) in baselines {
        let arms = completed
            .iter()
            .filter(|arm| arm.arm.student == *architecture)
            .collect::<Vec<_>>();
        let candidates = arms
            .iter()
            .map(|arm| {
                arm.outcome
                    .development_evaluation
                    .as_ref()
                    .context("arm development evidence absent")?
                    .path
                    .parent()
                    .context("arm development directory absent")
                    .map(Path::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        if candidates.is_empty() {
            continue;
        }
        let baseline = baseline.clone();
        let compared = tokio::task::spawn_blocking(move || {
            crate::study_evaluate::compare_many(&baseline, &candidates)
        })
        .await??;
        for (arm, mut comparison) in arms.into_iter().zip(compared) {
            let times = events
                .iter()
                .filter(|event| {
                    event["kind"] == "worker_attempt_finished" && event["arm_id"] == arm.arm.id
                })
                .map(|event| {
                    event["elapsed_seconds"]
                        .as_f64()
                        .context("recorded worker attempt timing missing")
                })
                .collect::<Result<Vec<_>>>()?;
            comparison["arm_id"] = arm.arm.id.clone().into();
            comparison["profile"] = profile.into();
            comparison["training_seed"] = arm.arm.seed.into();
            comparison["reserved_cpu_threads"] = arm.request.resources.cpu_threads.into();
            comparison["reserved_ram_bytes"] = arm.request.resources.ram_bytes.into();
            comparison["reserved_vram_bytes"] = arm.request.resources.vram_bytes.into();
            comparison["realised_optimizer_steps"] = arm.outcome.completed_steps.into();
            comparison["latest_worker_attempt_wall_seconds"] = arm.worker_wall_seconds.into();
            comparison["summed_recorded_worker_attempt_seconds"] = if times.is_empty() {
                Value::Null
            } else {
                times.iter().sum::<f64>().into()
            };
            comparison["recorded_worker_attempts"] = times.len().into();
            comparison["declared_worker_attempts"] =
                state["trials"][&arm.arm.id]["attempts"].clone();
            comparison["actual_peak_rss_bytes"] = Value::Null;
            comparison["measured_energy_usd"] = Value::Null;
            comparison["cost_per_observed_percentage_point_usd"] = Value::Null;
            rows.push(comparison);
        }
    }
    let path = study_directory
        .join(profile)
        .join("baseline-comparisons.json");
    artefact::write_json(
        &path,
        &json!({"schema_version":1,"evidence_kind":"actual_descriptive_development_baseline_and_resource_table",
        "rows":rows,"held_out_used":false,"production_eligible":false,
        "limitations":["RAM/VRAM columns are declared reservations; peak resident memory and energy were not measured.",
            "Recorded worker attempt seconds sum retained completion/interruption events; a controller crash can leave unrecorded time.",
            "Observed score differences do not establish significance or authorise promotion."]}),
    )?;
    context.publish(&format!("{profile}-baseline-comparisons"),"verified_complete",vec![capture(&context.root,&path)?],
        json!({"rows":rows.len(),"quality_comparisons_valid":rows.iter().filter(|row|row["quality_comparison_valid"]==true).count(),
            "positive_observed_deltas":rows.iter().filter(|row|row["positive_observed_delta"]==true).count(),"table":path})).await?;
    Ok(rows)
}
