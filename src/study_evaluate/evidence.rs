//! Reconstruct study results from their complete generation and sandbox evidence.
use super::*;

pub(super) fn checkpoint(identity: &Checkpoint, root: &Path) -> Result<()> {
    ensure!(
        identity.checkpoint_id == identity.tensor.sha256,
        "study checkpoint identifier differs from its tensor"
    );
    let assets_path = identity.base_assets.verify(root)?;
    let base = assets_path
        .parent()
        .context("base manifest directory missing")?;
    let assets = match identity.architecture {
        StudentArchitecture::ConventionalQwenLora => crate::model::validate_assets(base)?,
        StudentArchitecture::TernaryBitnet => crate::hub::validate_bitnet_student_assets(base)?,
    };
    identity.tokenizer.verify(root)?;
    ensure!(
        assets.files.get("tokenizer.json") == Some(&identity.tokenizer.sha256),
        "study tokeniser differs from its frozen base"
    );
    let tensor = identity.tensor.verify(root)?;
    let manifest = identity.manifest.verify(root)?;
    let loaded = candle_core::safetensors::load(tensor, &candle_core::Device::Cpu)?;
    ensure!(
        !loaded.is_empty(),
        "study checkpoint contains no real tensors"
    );
    if assets.files.get("model.safetensors") == Some(&identity.checkpoint_id) {
        ensure!(
            same(&identity.manifest, &identity.base_assets)?,
            "base comparison has inconsistent metadata"
        );
    } else {
        let proof = crate::distillation::checkpoint::validate_checkpoint(
            manifest
                .parent()
                .context("adapter manifest directory missing")?,
            &assets,
        )?;
        ensure!(
            proof.adapter_sha256 == identity.checkpoint_id
                && !proof.production_eligible
                && serde_json::to_value(proof.spec.student_kind)?
                    == serde_json::to_value(identity.architecture)?,
            "study adapter purpose, architecture or tensor identity differs"
        );
        crate::teacher_evidence::load_verified(&proof.evidence_bundle.path)?;
    }
    Ok(())
}

pub(super) fn plan(plan: &Plan, root: &Path) -> Result<()> {
    let expected = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == plan.split.corpus_split())
        .collect::<Vec<_>>();
    ensure!(
        plan.schema_version == 1
            && plan.evidence_kind == "frozen_local_study_checkpoint_evaluation"
            && plan.frozen_unix_ms > 0
            && !plan.production_eligible
            && same(&plan.tasks, &expected)?
            && !plan.tasks.is_empty()
            && plan.seed == 1729
            && plan.max_new_tokens == 512
            && plan.temperature == 0.2
            && plan.top_p == 0.95
            && plan.timeout_seconds == 120
            && (1..=256).contains(&plan.maximum_concurrent_graders)
            && !plan.sandbox_image_id.is_empty()
            && plan.sandbox.expected_image_id == plan.sandbox_image_id
            && sandbox::policy_sha256(&plan.sandbox)? == plan.sandbox_policy_sha256,
        "study evaluation plan changed task denominator, inference or sandbox protocol"
    );
    checkpoint(&plan.checkpoint, root)?;
    plan.engine.verify(root)?;
    backend::verify(plan, root)?;
    let probe: serde_json::Value =
        serde_json::from_slice(&fs::read(plan.sandbox_preflight.verify(root)?)?)?;
    ensure!(
        probe["probes_passed"] == true
            && probe["image_id"] == plan.sandbox_image_id
            && probe["policy_sha256"] == plan.sandbox_policy_sha256,
        "study evaluation lacks its effective sandbox controls"
    );
    Ok(())
}

pub(super) fn raw(record: &RawGeneration, task: &Task, plan: &Plan, plan_sha: &str) -> Result<()> {
    ensure!(
        record.schema_version == 1
            && record.plan_sha256 == plan_sha
            && record.task_id == task.id
            && record.source_id == task.source_id
            && same(&record.request, &plan.request(task))?,
        "cached study generation changed plan, task, seed or decoding request"
    );
    match (&record.generation, &record.model_error) {
        (Some(g), None) => ensure!(
            g.checkpoint_id == plan.checkpoint.checkpoint_id
                && g.input_tokens > 0
                && g.output_tokens <= plan.max_new_tokens
                && matches!(g.finish_reason.as_str(), "stop" | "length" | "timeout")
                && g.complete == (g.finish_reason == "stop"),
            "study generation has inconsistent checkpoint, token or completion metadata"
        ),
        (None, Some(error)) => ensure!(
            !error.trim().is_empty(),
            "empty model error cannot replace output"
        ),
        _ => anyhow::bail!("study generation requires exactly one result or explicit model error"),
    }
    Ok(())
}

pub(super) fn observation(
    row: &Observation,
    task: &Task,
    plan: &Plan,
    plan_sha: &str,
    root: &Path,
) -> Result<RawGeneration> {
    ensure!(
        row.schema_version == 1
            && row.plan_sha256 == plan_sha
            && row.task_id == task.id
            && row.source_id == task.source_id,
        "cached study observation identity changed"
    );
    let generated: RawGeneration =
        serde_json::from_slice(&fs::read(row.raw_generation.verify(root)?)?)?;
    raw(&generated, task, plan, plan_sha)?;
    let usable = generated
        .generation
        .as_ref()
        .is_some_and(|g| g.complete && !g.completion.trim().is_empty());
    if !usable {
        ensure!(
            !row.success && row.sandbox_report.is_none() && row.grading_error.is_none(),
            "failed/empty/truncated model output was accepted or supplied invented grading"
        );
    } else if let Some(error) = &row.grading_error {
        ensure!(
            !error.trim().is_empty() && !row.success && row.sandbox_report.is_none(),
            "grading error was accepted or mixed with a successful verdict"
        );
    } else {
        let report_artifact = row
            .sandbox_report
            .as_ref()
            .context("complete study generation lacks trusted grading evidence")?;
        let mut report: SandboxReport =
            serde_json::from_slice(&fs::read(report_artifact.verify(root)?)?)?;
        let audit = if report.audit_path.is_absolute() {
            report.audit_path.clone()
        } else {
            root.join(&report.audit_path)
        }
        .canonicalize()?;
        ensure!(
            audit.starts_with(root.canonicalize()?)
                && artefact::file_hash(&audit)? == report.audit_sha256,
            "study audit escaped its root or changed"
        );
        report.audit_path = audit;
        let code = crate::evaluate::extract_code(
            &generated
                .generation
                .as_ref()
                .context("checked generation missing")?
                .completion,
        );
        sandbox::verify_report(&report, &plan.sandbox, task, &code)?;
        let required = task.public_examples.len() + tasks::private_cases(task)?.len();
        let success = report.passed
            && !report.quarantined
            && report.trace_complete
            && report.cases_total == required
            && report.cases_passed == required;
        ensure!(
            report.policy_sha256 == plan.sandbox_policy_sha256
                && report.image_id == plan.sandbox_image_id
                && row.success == success,
            "study score differs from its complete trusted grading verdict"
        );
    }
    Ok(generated)
}

pub(super) fn summary(
    plan: &Plan,
    rows: &[Observation],
    raws: &[RawGeneration],
    plan_sha: &str,
) -> Result<serde_json::Value> {
    ensure!(
        rows.len() == raws.len(),
        "study observation/raw count differs"
    );
    let mut seen = BTreeSet::new();
    let mut sources: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut errors = 0;
    let mut generation_ms = 0u128;
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    for (row, generated) in rows.iter().zip(raws) {
        ensure!(
            seen.insert(&row.task_id) && plan.tasks.iter().any(|t| t.id == row.task_id),
            "duplicate or unplanned study observation"
        );
        let entry = sources.entry(row.source_id.clone()).or_default();
        entry.0 += 1;
        entry.1 += usize::from(row.success);
        errors += usize::from(
            row.grading_error.is_some()
                || generated.model_error.is_some()
                || generated
                    .generation
                    .as_ref()
                    .is_none_or(|g| !g.complete || g.completion.trim().is_empty()),
        );
        if let Some(g) = &generated.generation {
            generation_ms += g.elapsed_ms;
            input_tokens += g.input_tokens;
            output_tokens += g.output_tokens;
        }
    }
    let passed = rows.iter().filter(|r| r.success).count();
    let per_source=sources.into_iter().map(|(source,(observed,passed))|serde_json::json!({"source_id":source,"observed":observed,"passed":passed,"pass_rate":passed as f64/observed as f64})).collect::<Vec<_>>();
    Ok(
        serde_json::json!({"schema_version":1,"evidence_kind":"actual_local_study_checkpoint_evaluation",
        "split":plan.split,"checkpoint_id":plan.checkpoint.checkpoint_id,"architecture":plan.checkpoint.architecture,
        "status":if rows.len()!=plan.tasks.len() {"partial"} else if errors>0 {"complete_with_errors"} else {"complete"},
        "healthy_complete":rows.len()==plan.tasks.len() && errors==0,
        "plan_sha256":plan_sha,"planned":plan.tasks.len(),"observed":rows.len(),"passed":passed,
        "pass_rate":passed as f64/plan.tasks.len() as f64,"coverage_complete":rows.len()==plan.tasks.len(),
        "infrastructure_or_generation_errors":errors,"seed":plan.seed,"max_new_tokens":plan.max_new_tokens,
        "summed_generation_ms":generation_ms,"input_tokens":input_tokens,"output_tokens":output_tokens,
        "per_source":per_source,"held_out_used":false,"production_eligible":false,
        "limitations":["Local train/development measurement only; no learning or promotion claim.","One fixed evaluation seed and jointly authored source groups do not establish general superiority.","Generation time excludes loading, hashing and grading; overlapping work is not serial wall time.","Every failed, empty, truncated or missing output remains in the planned denominator."]}),
    )
}

pub fn verify_completed(output: &Path) -> Result<serde_json::Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    let output = output.canonicalize()?;
    ensure!(
        output.starts_with(&root),
        "study evaluation is outside its evidence root"
    );
    let frozen: Plan = serde_json::from_slice(&fs::read(output.join("plan.json"))?)?;
    plan(&frozen, &root)?;
    let plan_sha = artefact::file_hash(&output.join("plan.json"))?;
    let mut rows = Vec::new();
    let mut raws = Vec::new();
    for task in &frozen.tasks {
        let row: Observation = serde_json::from_slice(&fs::read(
            output
                .join("observations")
                .join(format!("{}.json", task.id)),
        )?)?;
        let raw = observation(&row, task, &frozen, &plan_sha, &root)?;
        rows.push(row);
        raws.push(raw);
    }
    executions(&output, &frozen, &root, &raws)?;
    let computed = summary(&frozen, &rows, &raws, &plan_sha)?;
    let recorded: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("report.json"))?)?;
    ensure!(
        computed["coverage_complete"] == true && computed == recorded,
        "study report differs from its complete raw/graded source evidence"
    );
    Ok(computed)
}

pub(super) fn executions(
    output: &Path,
    plan: &Plan,
    root: &Path,
    raws: &[RawGeneration],
) -> Result<()> {
    let execution_path = output.join("execution.json");
    ensure!(
        execution_path.exists() || raws.iter().all(|r| r.generation.is_none()),
        "study generations lack their loaded backend evidence"
    );
    if execution_path.exists() {
        let metadata = serde_json::from_slice(&fs::read(execution_path)?)?;
        backend::execution(&metadata, plan, root)?;
    }
    let attempts = output.join("execution-attempts");
    if attempts.exists() {
        for entry in fs::read_dir(attempts)? {
            let path = entry?.path();
            // An interrupted atomic write can leave its private .tmp file.
            // Only published JSON attempts constitute completed evidence.
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let metadata = serde_json::from_slice(&fs::read(path)?)?;
            backend::execution(&metadata, plan, root)?;
        }
    }
    Ok(())
}
