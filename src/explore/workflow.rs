//! Ordered durable stages; each completed artefact is independently rechecked.
use super::*;
use selection::{AssetLease, Selection};

pub(super) fn semantics(options: &ExploreOptions) -> Result<Value> {
    let mut value = serde_json::to_value(options)?;
    let object = value
        .as_object_mut()
        .context("exploration options must be an object")?;
    // Offline changes acquisition permission, and each invocation gets an
    // explicit wall budget. Neither alters frozen data or numerical methods.
    object.remove("offline");
    object.remove("maximum_run_seconds");
    Ok(value)
}

fn initialise(options: ExploreOptions, started: Instant) -> Result<(ContextState, RunLease)> {
    for path in [
        &options.search_config,
        &options.search_output,
        &options.study_config,
        &options.output,
        &options.base,
        &options.bitnet_base,
    ]
    .into_iter()
    .chain(&options.reuse_train_comparisons)
    {
        relative(path)?;
    }
    ensure!(
        options.output.starts_with("reports")
            && (1..=604_800).contains(&options.maximum_run_seconds)
            && (1..=48).contains(&options.content_maximum_tasks)
            && (1..=1024).contains(&options.content_max_tokens)
            && (1..=1024).contains(&options.collection_max_tokens)
            && (1..=8).contains(&options.top_groups)
            && options.maximum_parallel_jobs.is_none_or(|n| n > 0),
        "invalid bounded exploration protocol"
    );
    fs::create_dir_all(&options.output)?;
    let lease = RunLease::acquire(&options.output.join(".explore.lock"))?;
    let root = std::env::current_dir()?.canonicalize()?;
    let protocol_path = options.output.join("protocol.json");
    let protocol = if protocol_path.exists() {
        let previous: Protocol = serde_json::from_slice(&fs::read(&protocol_path)?)?;
        ensure!(
            previous.schema_version == 1
                && previous.evidence_kind == "frozen_resumable_local_exploration_protocol"
                && previous.options == semantics(&options)?
                && !previous.held_out_used
                && !previous.production_eligible,
            "exploration semantics changed; choose an explicit fresh output"
        );
        previous.search_config.verify(&root)?;
        previous.study_config.verify(&root)?;
        previous.stage_engine.verify()?;
        ensure!(
            previous.search_config.path == options.search_config
                && previous.study_config.path == options.study_config,
            "exploration config paths changed"
        );
        previous
    } else {
        let protocol = Protocol {
            schema_version: 1,
            evidence_kind: "frozen_resumable_local_exploration_protocol".into(),
            options: semantics(&options)?,
            search_config: capture(&root, &options.search_config)?,
            study_config: capture(&root, &options.study_config)?,
            stage_engine: engine::pin_executable(
                &std::env::current_exe()?,
                Path::new("state/engines"),
            )?,
            held_out_used: false,
            production_eligible: false,
        };
        artefact::write_json(&protocol_path, &protocol)?;
        protocol
    };
    let protocol_artifact = capture(&root, &protocol_path)?;
    let state_path = options.output.join("state.json");
    let state = if state_path.exists() {
        let state: State = serde_json::from_slice(&fs::read(&state_path)?)?;
        ensure!(
            state.schema_version == 1 && same(&state.protocol, &protocol_artifact)?,
            "exploration state belongs to another protocol"
        );
        state
    } else {
        State {
            schema_version: 1,
            protocol: protocol_artifact,
            stages: BTreeMap::new(),
            selection: None,
            events: vec![],
        }
    };
    let deadline = started + Duration::from_secs(options.maximum_run_seconds);
    Ok((
        ContextState {
            options,
            root,
            protocol,
            state,
            started,
            deadline,
        },
        lease,
    ))
}

fn search_engine(context: &ContextState) -> Result<PinnedEngine> {
    let freeze = context.options.search_output.join("freeze.json");
    if freeze.exists() {
        let hash = read(&freeze)?["engine_binary_sha256"]
            .as_str()
            .context("search freeze has no engine identity")?
            .to_owned();
        engine::resolve(Path::new("state/engines"), &hash)
    } else {
        Ok(context.protocol.stage_engine.clone())
    }
}

async fn selected(context: &mut ContextState) -> Result<Selection> {
    if let Some(artifact) = &context.state.selection {
        let selection: Selection =
            serde_json::from_slice(&fs::read(artifact.verify(&context.root)?)?)?;
        let remaining = context.remaining()?;
        tokio::time::timeout(
            remaining,
            selection::rehydrate(&selection, context.options.offline),
        )
        .await
        .context("wall budget stopped frozen teacher rehydration")??;
        return Ok(selection);
    }
    process::wait_for_search(context).await?;
    // The older pinned search cannot be patched while it runs. Preserve its
    // current failed-attempt state before any subsequent invocation can retry.
    let snapshot = context
        .options
        .output
        .join("search-history")
        .join(uuid::Uuid::new_v4().to_string());
    let mut before = Vec::new();
    for name in ["state.json", "report.json"] {
        let path = context.options.search_output.join(name);
        if path.exists() {
            artefact::atomic_write(&snapshot.join(name), &fs::read(path)?)?;
            before.push(capture(&context.root, &snapshot.join(name))?);
        }
    }
    if snapshot.join("state.json").exists() {
        let saved = read(&snapshot.join("state.json"))?;
        for trial in saved["trials"]
            .as_array()
            .context("saved search trials missing")?
        {
            let id = trial["id"]
                .as_str()
                .context("saved search trial identifier missing")?;
            ensure!(
                !id.is_empty()
                    && id
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte)),
                "unsafe saved trial identifier"
            );
            let resource = context
                .options
                .search_output
                .join("trials")
                .join(id)
                .join("resources.json");
            if resource.exists() {
                let target = snapshot.join("trials").join(id).join("resources.json");
                artefact::atomic_write(&target, &fs::read(resource)?)?;
                before.push(capture(&context.root, &target)?);
            }
        }
    }
    context.state.events.push(json!({"kind":"search_history_preserved_before_retry","snapshots":before,"unix_ms":crate::ledger::unix_millis()}));
    context.save().await?;
    let engine = search_engine(context)?;
    let mut args = vec![
        "select-teachers".into(),
        "--config".into(),
        context.options.search_config.display().to_string(),
        "--output".into(),
        context.options.search_output.display().to_string(),
    ];
    if context.options.offline {
        args.push("--offline".into());
    }
    let success = process::stage(context, "teacher-search", &engine, args).await?;
    let report = read(&context.options.search_output.join("report.json"))?;
    ensure!(
        success
            || matches!(
                report["status"].as_str(),
                Some("cached_only_shortfall" | "declared_roster_exhausted_shortfall")
            ),
        "teacher search failed before a complete bounded selection report; prior failure history is retained"
    );
    let selection = selection::freeze(context).await?;
    context.publish("teacher-search","verified_complete",vec![selection.search_freeze.clone(),selection.search_state_snapshot.clone(),selection.search_report_snapshot.clone()],
        json!({"qualified_target_met":selection.qualified_target_met,"models":selection.models.iter().map(|m|&m.id).collect::<Vec<_>>(),
            "research_fallback":!selection.qualified_target_met,"search_process_succeeded":success,
            "selection_policy":"All qualified distinct original sources, or the strongest measured fallback if none qualifies"})).await?;
    Ok(selection)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CollectionPolicy {
    schema_version: u32,
    registry: Artefact,
    comparisons: Vec<PathBuf>,
    reused_plan: Option<Artefact>,
    runtime: RuntimeOptions,
    max_tokens: usize,
    seed: u64,
    models: Vec<String>,
}

pub(super) fn reusable(
    plan: &Value,
    selection: &Selection,
    max_tokens: usize,
    root: &Path,
) -> Result<bool> {
    let expected = crate::tasks::corpus()
        .into_iter()
        .filter(|task| task.split == crate::tasks::Split::Train)
        .collect::<Vec<_>>();
    if plan["split"] != "train"
        || (!plan["pack"].is_null() && plan["pack"] != "original")
        || plan["tasks"] != serde_json::to_value(expected)?
        || plan["runtime"] != serde_json::to_value(&selection.runtime)?
        || plan["max_tokens"] != max_tokens
        || plan["seed"] != 1729
        || plan["temperature"] != 0.2
        || plan["top_p"] != 0.95
        || (!plan["generation_timeout_seconds"].is_null()
            && plan["generation_timeout_seconds"] != 120)
    {
        return Ok(false);
    }
    let models = plan["models"]
        .as_array()
        .context("reusable comparison model list absent")?;
    if models.len() != selection.models.len() {
        return Ok(false);
    }
    for (frozen, trial) in models.iter().zip(&selection.trials) {
        if frozen["model"] != serde_json::to_value(&trial.model)?
            || frozen["assets"] != serde_json::to_value(&trial.assets)?
        {
            return Ok(false);
        }
    }
    let registry: Artefact = serde_json::from_value(plan["registry"].clone())?;
    let original = Registry::load(&registry.verify(root)?)?;
    same(&original.models, &selection.models)
}

fn collection_policy(context: &ContextState, selection: &Selection) -> Result<CollectionPolicy> {
    let path = context.options.output.join("collection-policy.json");
    let expected_models = selection
        .models
        .iter()
        .map(|model| model.id.clone())
        .collect::<Vec<_>>();
    if path.exists() {
        let policy: CollectionPolicy = serde_json::from_slice(&fs::read(path)?)?;
        policy.registry.verify(&context.root)?;
        ensure!(
            policy.schema_version == 1
                && same(&policy.runtime, &selection.runtime)?
                && policy.models == expected_models
                && policy.max_tokens == context.options.collection_max_tokens
                && policy.seed == 1729,
            "frozen collection policy changed"
        );
        if let Some(plan) = &policy.reused_plan {
            plan.verify(&context.root)?;
            ensure!(
                reusable(
                    &read(&plan.path)?,
                    selection,
                    policy.max_tokens,
                    &context.root
                )?,
                "reused train comparison changed"
            );
        }
        return Ok(policy);
    }
    let mut policy = CollectionPolicy {
        schema_version: 1,
        registry: selection.registry.clone(),
        comparisons: vec![],
        reused_plan: None,
        runtime: selection.runtime.clone(),
        max_tokens: context.options.collection_max_tokens,
        seed: 1729,
        models: expected_models,
    };
    for directory in &context.options.reuse_train_comparisons {
        let path = directory.join("plan.json");
        if path.exists() {
            let plan = read(&path)?;
            if reusable(&plan, selection, policy.max_tokens, &context.root)? {
                policy.registry = serde_json::from_value(plan["registry"].clone())?;
                policy.comparisons = vec![directory.clone()];
                policy.reused_plan = Some(capture(&context.root, &path)?);
                break;
            }
        }
    }
    artefact::write_json(&path, &policy)?;
    Ok(policy)
}

async fn evidence(context: &mut ContextState, selection: &Selection) -> Result<PathBuf> {
    let policy = collection_policy(context, selection)?;
    if let Some(plan_artifact) = &policy.reused_plan {
        let comparison = policy
            .comparisons
            .first()
            .context("reused train comparison missing")?;
        if !comparison.join("report.json").exists() {
            let plan = read(&plan_artifact.path)?;
            let hash = plan["engine_binary_sha256"]
                .as_str()
                .context("reused train engine absent")?;
            let engine = engine::resolve(Path::new("state/engines"), hash)?;
            let mut args = vec![
                "compare-models".into(),
                "--config".into(),
                policy.registry.path.display().to_string(),
                "--output".into(),
                comparison.display().to_string(),
                "--split".into(),
                "train".into(),
                "--models".into(),
                policy.models.join(","),
                "--max-tokens".into(),
                policy.max_tokens.to_string(),
                "--seed".into(),
                "1729".into(),
            ];
            args.extend(runtime_arguments(&policy.runtime));
            let _completed =
                process::stage(context, "resume-existing-train", &engine, args).await?;
        }
        if context.state.stages.contains_key("resume-existing-train") {
            let checked = comparison.clone();
            let report = tokio::task::spawn_blocking(move || {
                comparative::verify_completed_comparison(&checked)
            })
            .await??;
            let errors = report["models"]
                .as_array()
                .context("reused train report models missing")?
                .iter()
                .map(|row| {
                    row["infrastructure_or_generation_errors"]
                        .as_u64()
                        .context("reused train error count missing")
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .sum::<u64>();
            context.publish("resume-existing-train","verified_complete",vec![capture(&context.root,&comparison.join("report.json"))?],
                json!({"coverage_complete":report["coverage_complete"],"generation_or_infrastructure_errors":errors,"healthy":errors==0,
                    "semantics":"Completed raw failures remain failures; only separately verified passing rows can enter training."})).await?;
        }
    }
    let output = context.options.output.join("evidence-v1");
    if !context.cached("training-evidence")? {
        let mut args = vec![
            "collect-teacher-evidence".into(),
            "--registry".into(),
            policy.registry.path.display().to_string(),
            "--output".into(),
            output.display().to_string(),
            "--models".into(),
            policy.models.join(","),
            "--max-tokens".into(),
            policy.max_tokens.to_string(),
        ];
        for path in &policy.comparisons {
            args.extend(["--comparison".into(), path.display().to_string()]);
        }
        args.extend(runtime_arguments(&policy.runtime));
        if context.options.offline {
            args.push("--offline".into());
        }
        let engine = context.protocol.stage_engine.clone();
        ensure!(
            process::stage(context, "training-evidence", &engine, args).await?,
            "teacher evidence collection failed; existing raw results retained"
        );
    }
    let path = output.clone();
    let bundle =
        tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&path)).await??;
    let mut generation_errors = 0u64;
    for file in bundle
        .input_comparisons
        .iter()
        .filter(|file| file.path.file_name().is_some_and(|n| n == "report.json"))
    {
        let report = read(&file.path)?;
        generation_errors += report["models"]
            .as_array()
            .context("training model scores missing")?
            .iter()
            .map(|row| {
                row["infrastructure_or_generation_errors"]
                    .as_u64()
                    .unwrap_or(0)
            })
            .sum::<u64>();
    }
    context.publish("training-evidence","verified_complete",vec![capture(&context.root,&output.join("bundle.json"))?,capture(&context.root,&context.options.output.join("collection-policy.json"))?],
        json!({"teachers":bundle.teachers.len(),"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"reused_comparisons":policy.comparisons,
            "generation_or_infrastructure_errors":generation_errors,"healthy":generation_errors==0})).await?;
    let content = context.options.output.join("content-v1");
    let mut content_process_ok = true;
    if !context.cached("content-evidence")? {
        let mut args = vec![
            "collect-teacher-content".into(),
            "--bundle".into(),
            output.join("bundle.json").display().to_string(),
            "--output".into(),
            content.display().to_string(),
            "--content".into(),
            "process,context,tool_trajectory".into(),
            "--maximum-tasks".into(),
            context.options.content_maximum_tasks.to_string(),
            "--max-tokens".into(),
            context.options.content_max_tokens.to_string(),
        ];
        args.extend(runtime_arguments(&selection.runtime));
        if context.options.offline {
            args.push("--offline".into());
        }
        let engine = context.protocol.stage_engine.clone();
        content_process_ok = process::stage(context, "content-evidence", &engine, args).await?;
    }
    let path = content.clone();
    let bundle =
        tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&path)).await??;
    let unavailable = read(&content.join("unavailable.json"))?;
    let content_report = read(&content.join("content-report.json"))?;
    ensure!(
        content_process_ok || content_report["healthy_complete"] != true,
        "content process failed despite healthy report; inspect retained attempt logs"
    );
    let failed_records = bundle
        .content_records
        .iter()
        .map(|file| read(&file.path))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .filter(|record| record["success"] != true)
        .count();
    context.publish("content-evidence",if content_report["healthy_complete"]==true {"verified_complete"} else {"bounded_evidence_gaps"},
        vec![capture(&context.root,&content.join("bundle.json"))?,capture(&context.root,&content.join("content-policy.json"))?,capture(&context.root,&content.join("content-report.json"))?],
        json!({"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"content_records":bundle.content_records.len(),"failed_content_records":failed_records,
            "unavailable":unavailable,"healthy":content_report["healthy_complete"],"current_batch":content_report})).await?;
    Ok(content.join("bundle.json"))
}

async fn pipeline(context: &mut ContextState) -> Result<Value> {
    let selection = selected(context).await?;
    let _assets = selection
        .models
        .iter()
        .map(|model| AssetLease::acquire(&model.directory))
        .collect::<Result<Vec<_>>>()?;
    let checked = selection.clone();
    let root = context.root.clone();
    tokio::task::spawn_blocking(move || selection::verify(&checked, &root)).await??;
    if !context.options.base.join("assets.json").is_file() {
        ensure!(
            !context.options.offline,
            "student base is missing offline; no HTTP attempted"
        );
        let engine = context.protocol.stage_engine.clone();
        ensure!(
            process::stage(
                context,
                "bootstrap-student",
                &engine,
                vec![
                    "bootstrap-online".into(),
                    "--base".into(),
                    context.options.base.display().to_string()
                ]
            )
            .await?,
            "student bootstrap failed"
        );
    }
    if context.state.stages.contains_key("bootstrap-student") {
        let base = context.options.base.clone();
        let assets = tokio::task::spawn_blocking(move || model::validate_assets(&base)).await??;
        context.publish("bootstrap-student","verified_complete",vec![capture(&context.root,&context.options.base.join("assets.json"))?],
            json!({"base_model_id":assets.model_id,"base_sha256":assets.files["model.safetensors"],"source":"Publisher-pinned real base assets; sandbox controls are checked independently before grading."})).await?;
    }
    let bundle = evidence(context, &selection).await?;
    let directory = context.options.output.join("study");
    if !context.cached("study-plan")? {
        let mut args = vec![
            "study-plan".into(),
            "--config".into(),
            context.options.study_config.display().to_string(),
            "--bundle".into(),
            bundle.display().to_string(),
            "--base".into(),
            context.options.base.display().to_string(),
            "--bitnet-base".into(),
            context.options.bitnet_base.display().to_string(),
            "--output".into(),
            directory.display().to_string(),
        ];
        if let Some(jobs) = context.options.maximum_parallel_jobs {
            args.extend(["--maximum-parallel-jobs".into(), jobs.to_string()]);
        }
        args.extend(runtime_arguments(&selection.runtime));
        if context.options.offline {
            args.push("--offline".into());
        }
        let engine = context.protocol.stage_engine.clone();
        ensure!(
            process::stage(context, "study-plan", &engine, args).await?,
            "study planning failed; evidence remains immutable"
        );
    }
    let path = directory.join("plan.json");
    let checked = path.clone();
    let plan = tokio::task::spawn_blocking(move || experiments::load_plan(&checked)).await??;
    ensure!(
        plan.inputs.evidence_bundle.sha256 == artefact::file_hash(&bundle)?,
        "study plan changed its frozen evidence bundle"
    );
    context.publish("study-plan","verified_complete",vec![capture(&context.root,&path)?],json!({"catalogue_arms":plan.arms.len(),"quick_arms":plan.quick_arms.len(),"quick_coverage_gaps":plan.quick_coverage_gaps})).await?;
    let (baselines, baseline_error) = match baselines::run(context, &plan).await {
        Ok(baselines) => (baselines, None),
        Err(error) => (BTreeMap::new(), Some(format!("{error:#}"))),
    };
    let cohort_catalogue = cohorts::prepare(
        context,
        &bundle,
        &plan,
        &selection
            .models
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
    )
    .await?;
    let mut baseline_comparisons = Vec::new();
    let offline_result=async {
    if let Some(error)=&baseline_error {anyhow::bail!("Offline baseline verification failed; retained as an independent family failure: {error}");}
    for profile in ["quick", "full"] {
        if profile == "full" {
            let selection_path = directory.join("selection.json");
            if !selection_path.exists() {
                let engine = context.protocol.stage_engine.clone();
                ensure!(
                    process::stage(
                        context,
                        "development-selection",
                        &engine,
                        vec![
                            "study-select".into(),
                            "--study".into(),
                            directory.display().to_string(),
                            "--top-groups".into(),
                            context.options.top_groups.to_string()
                        ]
                    )
                    .await?,
                    "no healthy complete development comparisons could be selected"
                );
            }
            let chosen = read(&selection_path)?;
            let ids: Vec<String> = serde_json::from_value(chosen["selected_arms"].clone())?;
            study::verify_selection(&selection_path, &path, &ids)?;
            context
                .publish(
                    "development-selection",
                    "verified_complete",
                    vec![capture(&context.root, &selection_path)?],
                    chosen,
                )
                .await?;
        }
        let stage_name = format!("{profile}-study");
        let report_path = directory.join(profile).join("report.json");
        if !context.cached(&stage_name)? {
            let profile_cap = if profile == "quick" {
                plan.quick.timeout_seconds
            } else {
                plan.full.timeout_seconds
            };
            let budget = context.remaining()?.as_secs().min(profile_cap).max(1);
            let mut args = vec![
                "study-run".into(),
                "--study".into(),
                directory.display().to_string(),
                "--profile".into(),
                profile.into(),
                "--maximum-run-seconds".into(),
                budget.to_string(),
            ];
            if profile == "full" {
                let selected = read(&directory.join("selection.json"))?;
                let ids: Vec<String> = serde_json::from_value(selected["selected_arms"].clone())?;
                args.extend([
                    "--selected-arms".into(),
                    ids.join(","),
                    "--selection-evidence".into(),
                    directory.join("selection.json").display().to_string(),
                ]);
            }
            if context.options.offline {
                args.push("--offline".into());
            }
            let engine = context.protocol.stage_engine.clone();
            let _success = process::stage(context, &stage_name, &engine, args).await?;
        }
        let report = read(&report_path)?;
        let output = directory.join(profile);
        let plan_path = path.clone();
        let completed = tokio::task::spawn_blocking(move || {
            experiments::completed_results(&output, &plan_path)
        })
        .await??;
        ensure!(
            report["completed_arms"] == completed.len(),
            "study aggregate differs from independently verified completed arms"
        );
        context
            .publish(
                &stage_name,
                if report["selected_profile_succeeded"] == true {
                    "verified_complete"
                } else {
                    "bounded_incomplete"
                },
                vec![capture(&context.root, &report_path)?],
                json!({"status":report["status"],"selected_profile_succeeded":report["selected_profile_succeeded"],
                    "catalogue_arms":report["catalogue_arms"],"applicable_arms":report["applicable_arms"],
                    "completed_arms":report["completed_arms"],"failed_arms":report["failed_arms"],
                    "queued_applicable_arms":report["queued_applicable_arms"],"reserved_training_steps":report["reserved_training_steps"],
                    "summed_worker_wall_seconds":report["summed_worker_wall_seconds"],"quick_coverage_gaps":report["quick_coverage_gaps"],
                    "full_report":capture(&context.root,&report_path)?}),
            )
            .await?;
        ensure!(
            !completed.is_empty(),
            "{profile} profile produced no healthy complete measured arm; progress retained"
        );
        baseline_comparisons.extend(
            baselines::compare_profile(context, profile, &directory, &completed, &baselines)
                .await?,
        );
    }
    Ok::<(),anyhow::Error>(())
    }.await;
    let offline_error = offline_result.err().map(|error| format!("{error:#}"));
    context.publish("offline-family",if offline_error.is_none() {"verified_complete"} else {"bounded_evidence_gaps"},vec![capture(&context.root,&path)?],
        json!({"healthy":offline_error.is_none(),"error":offline_error,"baseline_error":baseline_error})).await?;
    let cohort_results = cohorts::run(context, cohort_catalogue, &baselines).await?;
    Ok(
        json!({"teacher_qualification_target_met":selection.qualified_target_met,"selected_teachers":selection.models,"study":directory,
        "qualified_teacher_count":selection.trials.iter().filter(|trial|trial.qualified).count(),"teacher_target":selection.target_models,
        "quick_coverage_gaps":plan.quick_coverage_gaps,"baseline_comparisons":baseline_comparisons,"offline_error":offline_error,"cohorts":cohort_results}),
    )
}

pub async fn run(options: ExploreOptions) -> Result<Value> {
    let started = Instant::now();
    eprintln!(
        "explore: total wall-time budget={}s; offline={}; resume output={}; local research only",
        options.maximum_run_seconds,
        options.offline,
        options.output.display()
    );
    let (mut context, _lease) =
        tokio::task::spawn_blocking(move || initialise(options, started)).await??;
    context.save().await?;
    let result = pipeline(&mut context).await;
    let (details, error) = match result {
        Ok(value) => (value, None),
        Err(error) => (Value::Null, Some(format!("{error:#}"))),
    };
    let budget_stopped = context.deadline <= Instant::now();
    let healthy = context.state.stages.values().all(|stage| {
        stage.status == "verified_complete"
            && stage
                .details
                .get("healthy")
                .is_none_or(|value| *value == true)
    });
    let target_met = details["teacher_qualification_target_met"] == true;
    let success = error.is_none() && healthy && target_met;
    let status = if budget_stopped {
        "wall_budget_stopped_progress_saved"
    } else if error.is_some() {
        "incomplete_progress_saved"
    } else if !target_met {
        "completed_with_teacher_target_shortfall"
    } else if !healthy {
        "completed_with_evidence_gaps"
    } else {
        "completed_local_research"
    };
    let report = json!({"schema_version":1,"evidence_kind":"actual_resumable_local_exploration_pipeline","status":status,"success":success,
        "invocation_wall_seconds":context.started.elapsed().as_secs_f64(),"maximum_run_seconds":context.options.maximum_run_seconds,"offline":context.options.offline,
        "protocol":context.state.protocol,"stage_engine":context.protocol.stage_engine,"stages":context.state.stages,"events":context.state.events,
        "details":details,"error":error,"held_out_used":false,"production_eligible":false,"original_three_provider_assessment_satisfied":false,
        "costs":{"new_paid_provider_spend_usd":"0","local_electricity_usd":null,"hardware_amortisation_usd":null,"cost_per_point_of_improvement_usd":null},
        "limitations":["Local open-weight teachers are not three independent API provider businesses.","Development-only screening and selection do not authorise ship or establish learning/general superiority.","A fallback below the frozen qualification target remains a non-success, even if bounded research stages complete.","Failed, missing, empty and truncated outputs are retained; no failed generation is resampled on resume.","Energy, hardware cash cost and cost per improvement point remain unknown without real measurement."]});
    artefact::write_json(&context.options.output.join("report.json"), &report)?;
    context.save().await?;
    Ok(report)
}
