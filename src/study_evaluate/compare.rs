//! Descriptive matched baseline comparisons; no interval or ship decision is
//! inferred from repeatedly selected local development outcomes.
use super::*;

pub(super) fn matched_protocol(baseline: &Plan, candidate: &Plan) -> Result<()> {
    ensure!(
        baseline.split == candidate.split
            && same(&baseline.tasks, &candidate.tasks)?
            && baseline.runtime == candidate.runtime
            && same(&baseline.engine, &candidate.engine)?
            && baseline.seed == candidate.seed
            && baseline.max_new_tokens == candidate.max_new_tokens
            && baseline.temperature == candidate.temperature
            && baseline.top_p == candidate.top_p
            && baseline.timeout_seconds == candidate.timeout_seconds
            && baseline.sandbox_image_id == candidate.sandbox_image_id
            && baseline.sandbox_policy_sha256 == candidate.sandbox_policy_sha256
            && baseline.checkpoint.architecture == candidate.checkpoint.architecture
            && same(
                &baseline.checkpoint.base_assets,
                &candidate.checkpoint.base_assets
            )?
            && same(
                &baseline.checkpoint.tokenizer,
                &candidate.checkpoint.tokenizer
            )?,
        "baseline/candidate differ in task, seed, decoding, runtime, engine, base, tokeniser or sandbox protocol"
    );
    Ok(())
}

fn macro_score(report: &serde_json::Value) -> Result<f64> {
    let sources = report["per_source"]
        .as_array()
        .context("study comparison source rates missing")?;
    ensure!(
        !sources.is_empty(),
        "study comparison has no source denominator"
    );
    Ok(sources
        .iter()
        .map(|source| {
            source["pass_rate"]
                .as_f64()
                .context("source pass rate missing")
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<f64>()
        / sources.len() as f64)
}
fn execution(output: &Path) -> Result<serde_json::Value> {
    let path = output.join("execution.json");
    if path.exists() {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    } else {
        Ok(serde_json::Value::Null)
    }
}
fn threads(metadata: &serde_json::Value) -> serde_json::Value {
    metadata
        .get("cpu_threads")
        .or_else(|| metadata.get("candle_cpu_threads"))
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

/// Verify a baseline once, then reconstruct every candidate independently. A
/// failed baseline is retained as descriptive evidence and cannot establish a
/// model-quality improvement; its failed outputs stay in the denominator.
pub fn compare_many(baseline: &Path, candidates: &[PathBuf]) -> Result<Vec<serde_json::Value>> {
    let root = std::env::current_dir()?.canonicalize()?;
    let baseline_directory = baseline.canonicalize()?;
    let baseline = baseline_directory.strip_prefix(&root)?;
    let baseline_report = verify_completed(baseline)?;
    let baseline_plan: Plan = serde_json::from_slice(&fs::read(baseline.join("plan.json"))?)?;
    ensure!(
        same(
            &baseline_plan.checkpoint.manifest,
            &baseline_plan.checkpoint.base_assets
        )?,
        "requested baseline is an adapted checkpoint rather than actual unadapted base weights"
    );
    let baseline_execution = execution(baseline)?;
    let baseline_numerical = verify_numerical_equivalence(baseline)?;
    let baseline_score = macro_score(&baseline_report)?;
    let mut results = Vec::new();
    for candidate in candidates {
        let candidate_directory = candidate.canonicalize()?;
        let candidate = candidate_directory.strip_prefix(&root)?;
        let report = verify_completed(candidate)?;
        let plan: Plan = serde_json::from_slice(&fs::read(candidate.join("plan.json"))?)?;
        matched_protocol(&baseline_plan, &plan)?;
        let loaded = execution(candidate)?;
        let numerical = verify_numerical_equivalence(candidate)?;
        let actual_runtime_matched = baseline_execution["engine"] == loaded["engine"]
            && baseline_execution["actual_device"] == loaded["actual_device"]
            && baseline_execution["actual_dtype"] == loaded["actual_dtype"]
            && baseline_execution["engine_manifest"] == loaded["engine_manifest"];
        let healthy = baseline_report["healthy_complete"] == true
            && report["healthy_complete"] == true
            && actual_runtime_matched
            && baseline_numerical.qualified
            && numerical.qualified;
        let score = macro_score(&report)?;
        let delta = (score - baseline_score) * 100.;
        let same_threads = !threads(&baseline_execution).is_null()
            && threads(&baseline_execution) == threads(&loaded);
        results.push(serde_json::json!({"schema_version":2,"evidence_kind":"descriptive_matched_local_student_baseline_comparison",
            "baseline":Artefact::capture(&root,baseline.join("report.json"))?,"candidate":Artefact::capture(&root,candidate.join("report.json"))?,
            "architecture":plan.checkpoint.architecture,"split":plan.split,"baseline_checkpoint_id":baseline_plan.checkpoint.checkpoint_id,"candidate_checkpoint_id":plan.checkpoint.checkpoint_id,
            "baseline_source_macro_pass_rate":baseline_score,"candidate_source_macro_pass_rate":score,
            "observed_score_delta_percentage_points":delta,"quality_comparison_valid":healthy,
            "candidate_minus_baseline_percentage_points":if healthy{Some(delta)}else{None},"positive_observed_delta":healthy && delta>0.,
            "baseline_passed":baseline_report["passed"],"candidate_passed":report["passed"],"planned_tasks":report["planned"],
            "baseline_errors":baseline_report["infrastructure_or_generation_errors"],"candidate_errors":report["infrastructure_or_generation_errors"],
            "actual_runtime_matched":actual_runtime_matched,"same_inference_thread_allocation":same_threads,
            "baseline_numerical_equivalence":baseline_numerical,"candidate_numerical_equivalence":numerical,
            "baseline_inference_threads":threads(&baseline_execution),"candidate_inference_threads":threads(&loaded),
            "baseline_generation_ms":baseline_report["summed_generation_ms"],"candidate_generation_ms":report["summed_generation_ms"],
            "baseline_output_tokens":baseline_report["output_tokens"],"candidate_output_tokens":report["output_tokens"],
            "timings_support_matched_resource_comparison":actual_runtime_matched && same_threads,
            "held_out_used":false,"production_eligible":false,
            "limitations":["Observed local train/development differences only; no confidence interval, significant win, learning guarantee or global optimum is inferred.",
                "Generation time excludes loading, training and grading; different thread allocations are disclosed rather than treated as a fair speed comparison.",
                "Any baseline/candidate generation or infrastructure failure prevents a model-quality improvement claim.",
                "BitNet quality comparisons require independently reconstructed checkpoint-bound Rust/C++ numerical equivalence; missing, stale and failed receipts preserve descriptive scores but cannot establish improvement."]}));
    }
    Ok(results)
}

pub fn compare_completed(baseline: &Path, candidate: &Path) -> Result<serde_json::Value> {
    compare_many(baseline, &[candidate.to_owned()])?
        .pop()
        .context("verified local comparison absent")
}
