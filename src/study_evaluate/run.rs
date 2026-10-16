//! Lazy generation overlaps a bounded number of independent trusted graders.
use super::*;
use std::time::Duration;
use tokio::task::JoinSet;

async fn grade(
    raw: RawGeneration,
    artifact: Artefact,
    task: Task,
    plan: Plan,
    plan_sha: String,
    output: PathBuf,
) -> Result<Observation> {
    let mut row = Observation {
        schema_version: 1,
        plan_sha256: plan_sha,
        task_id: task.id.clone(),
        source_id: task.source_id.clone(),
        success: false,
        raw_generation: artifact,
        sandbox_report: None,
        grading_error: None,
    };
    if let Some(generation) = &raw.generation
        && generation.complete
        && !generation.completion.trim().is_empty()
    {
        let code = crate::evaluate::extract_code(&generation.completion);
        match sandbox::grade(&task, &code, &plan.sandbox).await {
            Ok(report) => {
                let root = std::env::current_dir()?;
                let checked = report.clone();
                let config = plan.sandbox.clone();
                let copied_task = task.clone();
                let copied_code = code.clone();
                let verified = tokio::task::spawn_blocking(move || {
                    sandbox::verify_report(&checked, &config, &copied_task, &copied_code)
                })
                .await?;
                match verified {
                    Ok(()) => {
                        let required =
                            task.public_examples.len() + tasks::private_cases(&task)?.len();
                        row.success = report.passed
                            && !report.quarantined
                            && report.trace_complete
                            && report.cases_total == required
                            && report.cases_passed == required;
                        row.sandbox_report = Some(Artefact::capture(
                            &root,
                            plan.sandbox
                                .report_dir
                                .join(format!("{}.json", report.run_id)),
                        )?);
                    }
                    Err(error) => {
                        row.grading_error = Some(format!("trusted grade rejected: {error:#}"))
                    }
                }
            }
            Err(error) => row.grading_error = Some(format!("{error:#}")),
        }
    }
    let checked = row.clone();
    let copied_plan = plan.clone();
    let copied_task = task.clone();
    let copied_sha = row.plan_sha256.clone();
    tokio::task::spawn_blocking(move || {
        evidence::observation(
            &checked,
            &copied_task,
            &copied_plan,
            &copied_sha,
            &std::env::current_dir()?,
        )?;
        artefact::write_json(&output, &checked)?;
        Ok::<_, anyhow::Error>(checked)
    })
    .await?
}

async fn next_grade(
    workers: &mut JoinSet<Result<Observation>>,
    rows: &mut BTreeMap<String, Observation>,
) -> Result<()> {
    let row = loop {
        tokio::select! {
            result=workers.join_next()=>break result.context("study grader set unexpectedly empty")???,
            _=tokio::time::sleep(Duration::from_secs(10))=>eprintln!("study-evaluate: heartbeat awaiting trusted grading; active={} completed={}",workers.len(),rows.len()),
        }
    };
    ensure!(
        rows.insert(row.task_id.clone(), row).is_none(),
        "duplicate study grade result"
    );
    Ok(())
}

pub async fn run<F, G>(mut options: StudyEvalOptions, factory: F) -> Result<serde_json::Value>
where
    F: FnOnce() -> Result<G> + Send + 'static,
    G: StudyGenerator + 'static,
{
    ensure!(
        (1..=256).contains(&options.maximum_concurrent_graders),
        "invalid study grader ceiling"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    fs::create_dir_all(&options.output)?;
    options.output = options
        .output
        .canonicalize()?
        .strip_prefix(&root)?
        .to_owned();
    let _lease = crate::engine::RunLease::acquire(&options.output.join(".study-evaluation.lock"))?;
    fs::create_dir_all(options.output.join("generations"))?;
    fs::create_dir_all(options.output.join("observations"))?;
    options.sandbox.report_dir = options.output.join("sandbox");
    let plan_path = options.output.join("plan.json");
    let existing = if plan_path.exists() {
        Some(serde_json::from_slice::<Plan>(&fs::read(&plan_path)?)?)
    } else {
        None
    };
    if let Some(plan) = &existing {
        if options.sandbox.expected_image_id.is_empty() {
            options.sandbox.expected_image_id = plan.sandbox_image_id.clone();
        }
        ensure!(
            same(&plan.checkpoint, &options.checkpoint)?
                && plan.runtime == options.runtime
                && same(&plan.engine, &options.engine)?
                && same(&plan.backend_artefacts, &options.backend_artefacts)?
                && plan.split == options.split
                && same(&plan.sandbox, &options.sandbox)?,
            "study evaluation resume changed checkpoint/runtime/engine/backend/split/sandbox; use an explicit fresh output"
        );
        if options.output.join("report.json").exists() {
            let directory = options.output.clone();
            let report =
                tokio::task::spawn_blocking(move || super::verify_completed(&directory)).await??;
            eprintln!(
                "study-evaluate: reused complete independently verified evaluation; no model/container started"
            );
            return Ok(report);
        }
    }
    eprintln!("study-evaluate: verifying effective sandbox controls");
    let probe = sandbox::preflight(&options.sandbox).await?;
    ensure!(
        probe.probes_passed,
        "study evaluation sandbox probes failed"
    );
    if options.sandbox.expected_image_id.is_empty() {
        options.sandbox.expected_image_id = probe.image_id.clone();
    }
    ensure!(
        options.sandbox.expected_image_id == probe.image_id,
        "study evaluation sandbox image changed"
    );
    let plan = if let Some(plan) = existing {
        plan
    } else {
        let path = options.output.join("sandbox-preflight.json");
        artefact::write_json(&path, &probe)?;
        let plan = Plan {
            schema_version: 1,
            evidence_kind: "frozen_local_study_checkpoint_evaluation".into(),
            frozen_unix_ms: crate::ledger::unix_millis(),
            checkpoint: options.checkpoint,
            runtime: options.runtime,
            engine: options.engine,
            backend_artefacts: options.backend_artefacts,
            split: options.split,
            tasks: tasks::corpus()
                .into_iter()
                .filter(|t| t.split == options.split.corpus_split())
                .collect(),
            seed: 1729,
            max_new_tokens: 512,
            temperature: 0.2,
            top_p: 0.95,
            timeout_seconds: 120,
            maximum_concurrent_graders: options.maximum_concurrent_graders,
            sandbox: options.sandbox,
            sandbox_image_id: probe.image_id,
            sandbox_policy_sha256: probe.policy_sha256,
            sandbox_preflight: Artefact::capture(&root, path)?,
            production_eligible: false,
        };
        artefact::write_json(&plan_path, &plan)?;
        plan
    };
    let copied = plan.clone();
    let copied_root = root.clone();
    tokio::task::spawn_blocking(move || evidence::plan(&copied, &copied_root)).await??;
    let execution_path = options.output.join("execution.json");
    if execution_path.exists() {
        let metadata = serde_json::from_slice(&fs::read(&execution_path)?)?;
        backend::execution(&metadata, &plan, &root)?;
    }
    let plan_sha = artefact::file_hash(&plan_path)?;
    let hardware = tokio::task::spawn_blocking(crate::platform::detect).await??;
    let assigned = std::env::var("LEC_STUDY_CPU_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(hardware.available_cpu_threads);
    let model_threads = candle_core::utils::get_num_threads();
    let cpu_model = plan.runtime["device"] == "cpu";
    let overlap = !cpu_model || assigned > model_threads;
    let spare = if cpu_model {
        assigned.saturating_sub(model_threads).max(1)
    } else {
        assigned.saturating_sub(1).max(1)
    };
    let maximum = plan
        .maximum_concurrent_graders
        .min(options.maximum_concurrent_graders)
        .min(spare)
        .min((hardware.usable_ram_bytes / (768 << 20)).max(1) as usize)
        .max(1);
    eprintln!(
        "study-evaluate: frozen tasks={} seed1729 max512; graders={} inference_overlap={overlap}",
        plan.tasks.len(),
        maximum
    );
    let mut rows = BTreeMap::new();
    let mut pending = Vec::new();
    for task in &plan.tasks {
        let row_path = options
            .output
            .join("observations")
            .join(format!("{}.json", task.id));
        if row_path.exists() {
            let row: Observation = serde_json::from_slice(&fs::read(row_path)?)?;
            evidence::observation(&row, task, &plan, &plan_sha, &root)?;
            ensure!(
                rows.insert(task.id.clone(), row).is_none(),
                "duplicate cached study observation"
            );
        } else {
            let path = options
                .output
                .join("generations")
                .join(format!("{}.json", task.id));
            let raw = if path.exists() {
                let raw: RawGeneration = serde_json::from_slice(&fs::read(&path)?)?;
                evidence::raw(&raw, task, &plan, &plan_sha)?;
                Some(raw)
            } else {
                None
            };
            pending.push((task.clone(), path, raw));
        }
    }
    let mut factory = Some(factory);
    let mut generator: Option<G> = None;
    let mut loading_error: Option<String> = None;
    let mut workers = JoinSet::new();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
    for (task, path, cached) in pending {
        if !overlap {
            while !workers.is_empty() {
                next_grade(&mut workers, &mut rows).await?;
            }
        }
        let raw = if let Some(raw) = cached {
            raw
        } else {
            if generator.is_none() && loading_error.is_none() {
                let load = factory
                    .take()
                    .context("study generator factory was already consumed")?;
                let loaded_plan = plan.clone();
                let loaded_root = root.clone();
                let mut loading = tokio::task::spawn_blocking(move || {
                    let model = load()?;
                    let metadata = model.execution_metadata();
                    // A rejected native backend owns a supervised process and
                    // local runtime; destroy it on this blocking thread too.
                    backend::execution(&metadata, &loaded_plan, &loaded_root)?;
                    Ok::<_, anyhow::Error>((model, metadata))
                });
                let loaded = loop {
                    tokio::select! {
                        result=&mut loading=>break result?,
                        _=heartbeat.tick()=>eprintln!("study-evaluate: heartbeat model loading; completed={}/{}",rows.len(),plan.tasks.len()),
                    }
                };
                match loaded {
                    Ok((model, metadata)) => {
                        // Retain the first load and every subsequent attempt so
                        // restarting a server cannot erase its earlier log link.
                        if !execution_path.exists() {
                            artefact::write_json(&execution_path, &metadata)?;
                        }
                        artefact::write_json(
                            &options
                                .output
                                .join("execution-attempts")
                                .join(format!("{}.json", uuid::Uuid::new_v4())),
                            &metadata,
                        )?;
                        generator = Some(model);
                    }
                    Err(error) => {
                        loading_error = Some(format!("study model loading failed: {error:#}"));
                    }
                }
            }
            let (generation, model_error) = if let Some(error) = &loading_error {
                (None, Some(error.clone()))
            } else {
                let mut model = generator.take().context("loaded study generator missing")?;
                let request = plan.request(&task);
                let mut generating = tokio::task::spawn_blocking(move || {
                    let result = model.generate(request);
                    (model, result)
                });
                let (returned, result) = loop {
                    tokio::select! {
                        result=&mut generating=>break result?,
                        _=heartbeat.tick()=>eprintln!("study-evaluate: heartbeat generating {} completed={}/{}",task.id,rows.len(),plan.tasks.len()),
                    }
                };
                generator = Some(returned);
                match result {
                    Ok(g) => (Some(g), None),
                    Err(error) => (None, Some(format!("{error:#}"))),
                }
            };
            let raw = RawGeneration {
                schema_version: 1,
                plan_sha256: plan_sha.clone(),
                task_id: task.id.clone(),
                source_id: task.source_id.clone(),
                request: plan.request(&task),
                generation,
                model_error,
            };
            evidence::raw(&raw, &task, &plan, &plan_sha)?;
            artefact::write_json(&path, &raw)?;
            raw
        };
        while workers.len() >= maximum {
            next_grade(&mut workers, &mut rows).await?;
        }
        let artifact = Artefact::capture(&root, path)?;
        let row_path = options
            .output
            .join("observations")
            .join(format!("{}.json", task.id));
        workers.spawn(grade(
            raw,
            artifact,
            task,
            plan.clone(),
            plan_sha.clone(),
            row_path,
        ));
    }
    if let Some(model) = generator {
        tokio::task::spawn_blocking(move || {
            model.synchronize()?;
            drop(model);
            Ok::<_, anyhow::Error>(())
        })
        .await??;
    }
    while !workers.is_empty() {
        tokio::select! {
            result=workers.join_next()=>{let row=result.context("study graders unexpectedly empty")???;ensure!(rows.insert(row.task_id.clone(),row).is_none(),"duplicate study grade result");},
            _=heartbeat.tick()=>eprintln!("study-evaluate: heartbeat grading; completed={}/{}",rows.len(),plan.tasks.len()),
        }
    }
    let mut observations = Vec::new();
    let mut raws = Vec::new();
    for task in &plan.tasks {
        let row = rows
            .remove(&task.id)
            .context("missing planned study observation")?;
        raws.push(evidence::observation(&row, task, &plan, &plan_sha, &root)?);
        observations.push(row);
    }
    evidence::executions(&options.output, &plan, &root, &raws)?;
    let report = evidence::summary(&plan, &observations, &raws, &plan_sha)?;
    ensure!(
        report["coverage_complete"] == true,
        "study evaluation denominator incomplete"
    );
    artefact::write_json(&options.output.join("report.json"), &report)?;
    Ok(report)
}
