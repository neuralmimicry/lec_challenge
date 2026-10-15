//! Account for measured local inference work without inventing energy prices.
//! This descriptive report cannot establish learning or authorise promotion.
use anyhow::{Context, Result, ensure};
use lec_distil::{artefact, artefact::Artefact, comparative::Observation, sandbox::SandboxReport};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let output = PathBuf::from(
        arguments
            .next()
            .context("usage: comparative_resources OUTPUT_DIRECTORY COMPARISON_DIRECTORY...")?,
    );
    let directories = arguments.map(PathBuf::from).collect::<Vec<_>>();
    ensure!(!directories.is_empty(), "no comparison inputs supplied");
    ensure!(
        !output.join("resources.json").exists(),
        "preserve existing resource evidence; select a fresh directory"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    let mut inputs = Vec::new();
    let mut rows = Vec::new();
    let mut models = Vec::new();
    let mut seen = BTreeSet::new();
    for directory in directories {
        let plan_path = directory.join("plan.json");
        let plan: Value = serde_json::from_slice(&fs::read(&plan_path)?)?;
        let plan_hash = artefact::file_hash(&plan_path)?;
        let report_path = directory.join("report.json");
        let report: Value = serde_json::from_slice(&fs::read(&report_path)?)?;
        ensure!(
            plan["promotable"] == false
                && plan["split"] == "development"
                && report["coverage_complete"] == true,
            "resource summary requires a complete exploratory development run"
        );
        let tasks = plan["tasks"].as_array().context("missing planned tasks")?;
        for model in plan["models"].as_array().context("missing models")? {
            let id = model["model"]["id"].as_str().context("model ID")?;
            ensure!(
                seen.insert(id.to_owned()),
                "model appears in multiple plans"
            );
            let mut elapsed_ms = 0_u128;
            let mut input_tokens = 0_usize;
            let mut output_tokens = 0_usize;
            let mut passed = 0_usize;
            let mut generated = 0_usize;
            let mut complete = 0_usize;
            let mut sandbox_elapsed_ms = 0_u128;
            let mut sandbox_runs = 0_usize;
            for task in tasks {
                ensure!(task["split"] == "development", "unexpected task split");
                let task_id = task["id"].as_str().context("task ID")?;
                let path = directory.join(id).join(format!("{task_id}.json"));
                let observation: Observation = serde_json::from_slice(&fs::read(&path)?)?;
                ensure!(
                    observation.model_id == id
                        && observation.task_id == task_id
                        && observation.plan_sha256 == plan_hash
                        && observation.source_id
                            == task["source_id"].as_str().context("source ID")?,
                    "observation identity differs from plan"
                );
                observation.raw_generation.verify(&root)?;
                let sandbox_metrics = observation
                    .sandbox_report
                    .as_ref()
                    .map(|artifact| {
                        let report: SandboxReport =
                            serde_json::from_slice(&fs::read(artifact.verify(&root)?)?)?;
                        sandbox_elapsed_ms += u128::from(report.duration_ms);
                        sandbox_runs += 1;
                        Ok::<_, anyhow::Error>(json!({
                            "elapsed_ms":report.duration_ms, "passed":report.passed,
                            "quarantined":report.quarantined,
                            "cases_passed":report.cases_passed,
                            "cases_total":report.cases_total,
                            "trace_complete":report.trace_complete
                        }))
                    })
                    .transpose()?;
                passed += usize::from(observation.success);
                if let Some(generation) = &observation.generation {
                    generated += 1;
                    complete += usize::from(generation.complete);
                    elapsed_ms += generation.elapsed_ms;
                    input_tokens += generation.input_tokens;
                    output_tokens += generation.output_tokens;
                }
                rows.push(json!({
                    "model":id, "task_id":task_id,
                    "source_id":observation.source_id,
                    "success":observation.success,
                    "generation":observation.generation.as_ref().map(|g| json!({
                        "elapsed_ms":g.elapsed_ms, "input_tokens":g.input_tokens,
                        "output_tokens":g.output_tokens, "complete":g.complete,
                        "finish_reason":g.finish_reason
                    })),
                    "generation_error":observation.generation_error,
                    "grading_error":observation.grading_error,
                    "sandbox":sandbox_metrics,
                    "evidence":Artefact::capture(&root,path)?
                }));
            }
            let existing = report["models"]
                .as_array()
                .context("missing model summary")?
                .iter()
                .find(|row| row["id"] == id)
                .context("model summary absent")?;
            ensure!(
                existing["passed"] == passed && existing["observed"] == tasks.len(),
                "derived counts differ from original comparative report"
            );
            models.push(json!({
                "model":id, "planned_tasks":tasks.len(), "passed":passed,
                "generations":generated, "complete_generations":complete,
                "summed_generation_seconds":elapsed_ms as f64 / 1000.,
                "mean_generation_seconds":if generated == 0 {None} else {
                    Some(elapsed_ms as f64 / 1000. / generated as f64)
                },
                "input_tokens":input_tokens, "output_tokens":output_tokens,
                "sandbox_runs":sandbox_runs,
                "summed_sandbox_wall_seconds":sandbox_elapsed_ms as f64 / 1000.,
                "remote_inference_calls":0, "remote_inference_charge_usd":0,
                "total_monetary_cost_usd":Value::Null,
                "cost_per_point_of_learning_usd":Value::Null
            }));
        }
        inputs.push(json!({
            "plan":Artefact::capture(&root,plan_path)?,
            "report":Artefact::capture(&root,report_path)?
        }));
    }
    let report = json!({
        "schema_version":1,
        "evidence_kind":"derived_measured_local_comparative_resources",
        "observed_unix_ms":lec_distil::ledger::unix_millis(),
        "inputs":inputs, "models":models, "tasks":rows,
        "promotable":false,
        "limitations":[
            "Generation time is reported by the native engine; sums exclude asset checks, loading, grading and idle time, and are not CUDA kernel time.",
            "Sandbox durations include container compilation/execution/cleanup and can overlap inference and each other; summed times are not end-to-end elapsed time or CPU core time.",
            "Missing generations have unknown duration and remain failures in comparative denominators.",
            "All inference is local. Zero remote inference charges do not imply zero total cost: electricity, hardware depreciation, model downloads and labour were not priced.",
            "No training round or learning improvement occurred in this experiment, so cost per point of learning is not applicable.",
            "One decoding seed on four jointly authored development sources supports descriptive comparisons only."
        ]
    });
    fs::create_dir_all(&output)?;
    artefact::write_json(&output.join("resources.json"), &report)?;
    let mut markdown = String::from(
        "# Local comparative resources\n\nMeasured descriptive development results; no training or promotion claim.\n\n| Model | Passed / planned | Complete generations | Summed inference seconds | Mean seconds / generation |\n| --- | ---: | ---: | ---: | ---: |\n",
    );
    for model in report["models"]
        .as_array()
        .context("derived models missing")?
    {
        let mean = model["mean_generation_seconds"]
            .as_f64()
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "N/A".into());
        markdown.push_str(&format!(
            "| {} | {} / {} | {} | {:.3} | {mean} |\n",
            model["model"].as_str().context("derived model ID")?,
            model["passed"],
            model["planned_tasks"],
            model["complete_generations"],
            model["summed_generation_seconds"]
                .as_f64()
                .context("inference time")?
        ));
    }
    markdown.push_str("\nNo remote inference calls were made. Total monetary cost and cost per point of learning are N/A: energy/depreciation were not measured, and no training round or learning gain occurred. Times exclude model loading, asset verification and grading. Per-task timings, tokens, failures and source evidence hashes are retained in resources.json.\n");
    artefact::atomic_write(&output.join("resources.md"), markdown.as_bytes())?;
    println!("{}", output.join("resources.md").display());
    Ok(())
}
