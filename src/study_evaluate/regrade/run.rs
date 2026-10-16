//! Bounded regrading runs no model factory and never rewrites original evidence.
use super::*;
use std::time::Duration;
use tokio::task::JoinSet;

pub(super) fn output_path(root: &Path, requested: &Path, originals: &[PathBuf]) -> Result<PathBuf> {
    ensure!(
        !requested.as_os_str().is_empty()
            && requested
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "regrade output must be relative without traversal"
    );
    let absolute = root.join(requested);
    let mut ancestor = absolute.clone();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(
            ancestor
                .file_name()
                .context("output ancestor missing")?
                .to_owned(),
        );
        ensure!(ancestor.pop(), "output ancestor absent");
    }
    let mut resolved = ancestor.canonicalize()?;
    for name in suffix.iter().rev() {
        resolved.push(name);
    }
    ensure!(
        resolved.starts_with(root)
            && originals
                .iter()
                .all(|p| !resolved.starts_with(p) && !p.starts_with(&resolved)),
        "regrade output overlaps original evidence"
    );
    Ok(resolved.strip_prefix(root)?.to_owned())
}

/// Recover only the exact pre-launch identity. A launch without its terminal
/// report is an infrastructure failure; it must never become another attempt.
pub(super) fn prepare_or_adopt(
    root: &Path,
    plan: &RegradePlan,
    plan_artifact: &Artefact,
    task: &Task,
    imported: &ImportedGeneration,
    output: &Path,
    code: &str,
) -> Result<(sandbox::PreparedGrade, Artefact)> {
    let receipt_path = output.join("prepared").join(format!("{}.json", task.id));
    if !receipt_path.try_exists()? {
        let attempt_key = artefact::hash(
            format!(
                "{}:{}:{}",
                plan_artifact.sha256, task.id, imported.original_raw.sha256
            )
            .as_bytes(),
        );
        let mut config = plan.grading.sandbox.clone();
        config.report_dir = output.join("sandbox");
        sandbox::prepare_grade(task, code, &config, &attempt_key, &receipt_path)?;
    }
    let receipt = capture(root, &receipt_path)?;
    let prepared = verify::prepared(
        root,
        &receipt,
        plan,
        plan_artifact,
        task,
        code,
        &imported.original_raw,
    )?;
    Ok((prepared, receipt))
}

pub(super) async fn grade(
    plan: RegradePlan,
    plan_artifact: Artefact,
    index: usize,
    output: PathBuf,
) -> Result<()> {
    let root = std::env::current_dir()?.canonicalize()?;
    let task = plan.original.generation_plan.tasks[index].clone();
    let imported = plan.original.rows[index].clone();
    let row_path = output
        .join("observations")
        .join(format!("{}.json", task.id));
    let initial = tokio::task::spawn_blocking({
        let root = root.clone();
        let plan = plan.clone();
        let plan_artifact = plan_artifact.clone();
        let task = task.clone();
        let imported = imported.clone();
        let row_path = row_path.clone();
        move || -> Result<Option<(RegradedObservation, Option<String>)>> {
            if row_path.try_exists()? {
                let row: RegradedObservation = read(&row_path)?;
                verify::observation(&root, &plan, &plan_artifact, &task, &imported, &row)?;
                return Ok(None);
            }
            let raw = original_raw(&root, &plan.original, &imported, &task)?;
            let code = if usable(&raw) {
                Some(crate::evaluate::extract_code(
                    &raw.generation
                        .as_ref()
                        .context("usable generation absent")?
                        .completion,
                ))
            } else {
                None
            };
            let row = RegradedObservation {
                schema_version: 1,
                regrade_plan_sha256: plan_artifact.sha256.clone(),
                task_id: task.id.clone(),
                source_id: task.source_id.clone(),
                original_plan: plan.original.plan.clone(),
                original_raw: imported.original_raw.clone(),
                outcome: Outcome::OriginalGenerationFailure,
                success: false,
            };
            Ok(Some((row, code)))
        }
    })
    .await??;
    let Some((mut row, code)) = initial else {
        return Ok(());
    };
    if let Some(code) = code {
        let prepared = tokio::task::spawn_blocking({
            let root = root.clone();
            let plan = plan.clone();
            let plan_artifact = plan_artifact.clone();
            let task = task.clone();
            let imported = imported.clone();
            let output = output.clone();
            move || {
                prepare_or_adopt(
                    &root,
                    &plan,
                    &plan_artifact,
                    &task,
                    &imported,
                    &output,
                    &code,
                )
            }
        })
        .await?;
        match prepared {
            Err(error) => {
                row.outcome = Outcome::InfrastructureFailure {
                    prepared: None,
                    error: format!("grade preparation failed: {error:#}"),
                }
            }
            Ok((prepared, identity)) => match sandbox::execute_prepared(&prepared).await {
                Err(error) => {
                    row.outcome = Outcome::InfrastructureFailure {
                        prepared: Some(identity),
                        error: format!("prepared grade failed: {error:#}"),
                    }
                }
                Ok(report) => {
                    let checked = prepared.clone();
                    let observed = report.clone();
                    let verify_root = root.clone();
                    let checked_task = task.clone();
                    let verified = tokio::task::spawn_blocking(
                        move || -> Result<(bool, Artefact, Artefact)> {
                            sandbox::verify_prepared_report(&checked, &observed)?;
                            Ok((
                                verify::passed(&checked_task, &observed)?,
                                capture(&verify_root, &checked.report_path)?,
                                capture(&verify_root, &observed.audit_path)?,
                            ))
                        },
                    )
                    .await?;
                    match verified {
                        Err(error) => {
                            row.outcome = Outcome::InfrastructureFailure {
                                prepared: Some(identity),
                                error: format!("prepared trusted grade rejected: {error:#}"),
                            }
                        }
                        Ok((success, report, audit)) => {
                            row.success = success;
                            row.outcome = Outcome::Graded {
                                prepared: identity,
                                report,
                                audit,
                            };
                        }
                    }
                }
            },
        }
    }
    let checked = row.clone();
    let verify_root = root.clone();
    let copied_plan = plan.clone();
    let copied_artifact = plan_artifact.clone();
    tokio::task::spawn_blocking(move || {
        verify::observation(
            &verify_root,
            &copied_plan,
            &copied_artifact,
            &task,
            &imported,
            &checked,
        )?;
        artefact::write_json(&row_path, &checked)
    })
    .await??;
    eprintln!(
        "study-regrade: durable task={} success={} original={} ",
        row.task_id,
        row.success,
        row.original_raw.path.display()
    );
    Ok(())
}

/// Freeze the entire requested original denominator before any new grade.
pub async fn run_batch(mut options: RegradeBatchOptions) -> Result<serde_json::Value> {
    ensure!(
        !options.originals.is_empty()
            && options.expected_generations > 0
            && (1..=256).contains(&options.maximum_concurrent_graders),
        "invalid regrade batch bounds"
    );
    ensure!(
        options.sandbox.controller_version == 3,
        "derived regrading requires explicit controller 3"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    let original_paths = options
        .originals
        .iter()
        .map(|p| p.canonicalize())
        .collect::<std::io::Result<Vec<_>>>()?;
    ensure!(
        original_paths.iter().all(|p| p.starts_with(&root)),
        "original evidence is outside the root"
    );
    options.output = output_path(&root, &options.output, &original_paths)?;
    fs::create_dir_all(&options.output)?;
    let _lease = crate::engine::RunLease::acquire(&options.output.join(".regrade-batch.lock"))?;
    let batch_path = options.output.join("batch-plan.json");
    let current_engine = capture(&root, &std::env::current_exe()?)?;
    let batch: BatchPlan = if batch_path.exists() {
        let old: BatchPlan = read(&batch_path)?;
        ensure!(
            old.expected_generations == options.expected_generations
                && old.maximum_concurrent_graders == options.maximum_concurrent_graders
                && old.grading.engine.sha256 == current_engine.sha256
                && old.grading.extractor_sha256 == extractor_hash(),
            "regrade launch protocol changed; use its original engine or a fresh output"
        );
        let expected = original_paths
            .iter()
            .map(|p| p.strip_prefix(&root).map(Path::to_owned))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            old.originals
                .iter()
                .map(|o| o.directory.clone())
                .collect::<Vec<_>>()
                == expected,
            "regrade original cohort changed"
        );
        validate_cohort(&old.originals, old.expected_generations)?;
        options.sandbox.expected_image_id = old.grading.image_id.clone();
        options.sandbox.report_dir = old.grading.sandbox.report_dir.clone();
        ensure!(
            same(&options.sandbox, &old.grading.sandbox)?,
            "saved new grading policy changed"
        );
        let verify_root = root.clone();
        let originals = old.originals.clone();
        tokio::task::spawn_blocking(move || {
            for original in originals {
                import::verify(&verify_root, &original)?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        old
    } else {
        let paths = original_paths.clone();
        let verify_root = root.clone();
        let originals = tokio::task::spawn_blocking(move || {
            paths
                .iter()
                .map(|p| import::collect(&verify_root, p))
                .collect::<Result<Vec<_>>>()
        })
        .await??;
        validate_cohort(&originals, options.expected_generations)?;
        let count = options.expected_generations;
        options.sandbox.report_dir = options.output.join("sandbox");
        let preflight = sandbox::preflight(&options.sandbox).await?;
        ensure!(
            preflight.probes_passed,
            "new regrade sandbox preflight failed"
        );
        options.sandbox.expected_image_id = preflight.image_id.clone();
        let preflight_path = options.output.join("sandbox-preflight.json");
        artefact::write_json(&preflight_path, &preflight)?;
        let batch = BatchPlan {
            schema_version: 1,
            evidence_kind: BATCH_KIND.into(),
            originals,
            grading: GradingProtocol {
                engine: current_engine,
                sandbox: options.sandbox.clone(),
                policy_sha256: sandbox::policy_sha256(&options.sandbox)?,
                image_id: preflight.image_id,
                preflight: capture(&root, &preflight_path)?,
                extractor_sha256: extractor_hash(),
            },
            expected_generations: count,
            maximum_concurrent_graders: options.maximum_concurrent_graders,
            production_eligible: false,
        };
        artefact::write_json(&batch_path, &batch)?;
        batch
    };
    let batch_artifact = capture(&root, &batch_path)?;
    let hardware = tokio::task::spawn_blocking(crate::platform::detect).await??;
    let per_grader = batch
        .grading
        .sandbox
        .memory_mb
        .saturating_mul(1024 * 1024)
        .saturating_add(64 * 1024 * 1024);
    ensure!(
        hardware.usable_ram_bytes >= per_grader,
        "guarded RAM cannot hold one bounded regrade container"
    );
    let memory_slots = (hardware.usable_ram_bytes / per_grader) as usize;
    let maximum = batch
        .maximum_concurrent_graders
        .min(hardware.available_cpu_threads)
        .min(memory_slots)
        .max(1);
    let mut workers = JoinSet::new();
    for (source_index, original) in batch.originals.iter().enumerate() {
        let output = options.output.join(child_name(source_index, original));
        fs::create_dir_all(output.join("observations"))?;
        let plan = RegradePlan {
            schema_version: 1,
            evidence_kind: PLAN_KIND.into(),
            batch_plan: batch_artifact.clone(),
            original: original.clone(),
            grading: batch.grading.clone(),
            maximum_concurrent_graders: batch.maximum_concurrent_graders,
            production_eligible: false,
        };
        let path = output.join("plan.json");
        if path.exists() {
            ensure!(
                same(&plan, &read::<RegradePlan>(&path)?)?,
                "derived regrade plan changed"
            );
        } else {
            artefact::write_json(&path, &plan)?;
        }
        let plan_artifact = capture(&root, &path)?;
        for index in 0..original.rows.len() {
            while workers.len() >= maximum {
                workers
                    .join_next()
                    .await
                    .context("regrade workers absent")???;
            }
            workers.spawn(grade(
                plan.clone(),
                plan_artifact.clone(),
                index,
                output.clone(),
            ));
        }
    }
    while !workers.is_empty() {
        tokio::select! {next=workers.join_next()=>{next.context("regrade worker set empty")???;},_=tokio::time::sleep(Duration::from_secs(10))=>eprintln!("study-regrade: heartbeat active={}",workers.len())}
    }
    let output = options.output.clone();
    tokio::task::spawn_blocking(move || {
        for (index, original) in batch.originals.iter().enumerate() {
            let directory = output.join(child_name(index, original));
            let view = verify::one(&directory)?;
            artefact::write_json(&directory.join("report.json"), &view.report)?;
        }
        let (report, _) = verify::compute_batch(&output)?;
        artefact::write_json(&output.join("batch-report.json"), &report)?;
        verify_batch(&output)
    })
    .await?
}
