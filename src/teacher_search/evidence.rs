//! Independently revalidate frozen models, task packs and linked grading evidence.
use super::*;

pub(super) fn expected_trial_model(
    trial: &Trial,
    config: &SearchConfig,
    registry: &Registry,
    output: &Path,
) -> Result<RegistryModel> {
    let (model, source, comparison) = if trial.imported {
        let model = registry
            .models
            .iter()
            .find(|m| m.id == trial.id)
            .context("imported trial is absent from the original registry")?
            .clone();
        ensure!(
            trial
                .comparison
                .as_ref()
                .is_some_and(|p| config.initial_comparisons.contains(p)),
            "imported trial has an undeclared comparison"
        );
        (
            model.clone(),
            model.model_id.clone(),
            trial.comparison.clone(),
        )
    } else {
        let candidate = config
            .candidates
            .iter()
            .find(|c| c.id == trial.id)
            .context("saved trial is absent from the frozen candidate roster")?;
        (
            candidate.registry_model(),
            candidate.source_model_id.clone(),
            Some(output.join("trials").join(&candidate.id).join("comparison")),
        )
    };
    ensure!(
        trial.model_id == model.model_id
            && trial.source_model_id == source
            && trial.revision == model.revision
            && trial.directory == model.directory
            && trial.role == model.role
            && trial.planned == 16
            && trial.errors <= 16
            && trial.passed.is_none_or(|n| n <= 16)
            && trial
                .comparison
                .as_ref()
                .is_none_or(|p| Some(p) == comparison.as_ref())
            && trial
                .qualification_comparison
                .as_ref()
                .is_none_or(|p| *p == output.join("trials").join(&trial.id).join("qualification")),
        "saved trial identity or denominator differs from its frozen roster"
    );
    let qualified = trial.passed.is_some_and(|passed| {
        model.role != "student_baseline"
            && trial.errors == 0
            && passed as f64 / 16.0 >= config.minimum_pass_rate
    }) && trial
        .qualification_passed
        .is_some_and(|passed| passed <= 16 && passed as f64 / 16.0 >= config.minimum_pass_rate)
        && trial.qualification_errors == Some(0);
    ensure!(
        trial.qualified == qualified,
        "saved qualification differs from its verified score"
    );
    ensure!(
        trial.passed.is_some() || (trial.errors == 0 && !trial.pruned && trial.report.is_none()),
        "an unobserved trial cannot carry a score, report or prune decision"
    );
    ensure!(
        trial.passed.is_none() || (trial.comparison.is_some() && trial.report.is_some()),
        "a scored trial requires complete comparison evidence"
    );
    ensure!(
        trial.qualification_passed.is_none()
            || (trial.qualification_comparison
                == Some(output.join("trials").join(&trial.id).join("qualification"))
                && trial.qualification_report.is_some()
                && trial.qualification_errors.is_some_and(|n| n <= 16)
                && trial
                    .passed
                    .is_some_and(|n| n as f64 / 16.0 >= config.minimum_pass_rate)
                && trial.errors == 0),
        "fresh qualification requires a complete linked comparison after primary qualification"
    );
    ensure!(
        trial.qualification_passed.is_some()
            || (trial.qualification_report.is_none() && trial.qualification_errors.is_none()),
        "unobserved fresh qualification cannot carry aggregate results"
    );
    Ok(model)
}

pub(super) fn verify_plan_identity(
    plan: &serde_json::Value,
    model: &RegistryModel,
    config: &SearchConfig,
    imported: bool,
    fresh: bool,
    registry_path: &Path,
) -> Result<()> {
    verify_plan_identity_pinned(
        plan,
        model,
        config,
        imported,
        fresh,
        registry_path,
        &engine_sha256()?,
    )
}

fn verify_plan_identity_pinned(
    plan: &serde_json::Value,
    model: &RegistryModel,
    config: &SearchConfig,
    imported: bool,
    fresh: bool,
    registry_path: &Path,
    expected_engine: &str,
) -> Result<()> {
    let corpus = if fresh {
        qualification::corpus()
    } else {
        tasks::corpus()
            .into_iter()
            .filter(|t| t.split == Split::Development)
            .collect::<Vec<_>>()
    };
    let models = plan["models"]
        .as_array()
        .context("comparison plan models missing")?;
    let expected_model = serde_json::to_value(model)?;
    let mut expected_tasks = serde_json::to_value(corpus)?;
    let mut observed_tasks = plan["tasks"].clone();
    if imported {
        // Adding the isolated fresh-pack dispatch changes the whole tasks.rs
        // source checksum. Preserve the historical hash in its original plan;
        // require every contract, example, source and specification unchanged.
        for value in [&mut expected_tasks, &mut observed_tasks] {
            for task in value.as_array_mut().context("planned task list missing")? {
                task.as_object_mut()
                    .context("planned task object missing")?
                    .remove("oracle_sha256");
            }
        }
    }
    ensure!(
        plan["split"] == "development"
            && (if fresh {
                plan["pack"] == "qualification"
            } else {
                plan["pack"].is_null() || plan["pack"] == "original"
            })
            && observed_tasks == expected_tasks
            && plan["max_tokens"] == config.max_new_tokens
            && plan["seed"] == config.seed
            && plan["registry"]["path"] == serde_json::to_value(registry_path)?
            && plan["registry"]["sha256"] == artefact::file_hash(registry_path)?
            && models.iter().any(|m| m["model"] == expected_model)
            && (imported || (models.len() == 1 && plan["engine_binary_sha256"] == expected_engine)),
        "comparison plan differs from the frozen development protocol or model registry"
    );
    Ok(())
}

pub(super) fn frozen_runtime(
    plan_path: &Path,
    candidate: &Candidate,
    config: &SearchConfig,
    registry_path: &Path,
) -> Result<platform::RuntimeSelection> {
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(plan_path)?)?;
    verify_plan_identity(
        &plan,
        &candidate.registry_model(),
        config,
        false,
        false,
        registry_path,
    )?;
    let options: model::RuntimeOptions = serde_json::from_value(plan["runtime"].clone())?;
    ensure!(
        match config.device.as_str() {
            "auto" => true,
            "cpu" => options.device == model::ExecutionDevice::Cpu,
            "cuda" =>
                options.device
                    == (model::ExecutionDevice::Cuda {
                        ordinal: config.gpu_ordinal
                    }),
            _ => false,
        },
        "frozen runtime differs from explicit hardware policy"
    );
    Ok(platform::RuntimeSelection { options, reason: "Resuming the original immutable runtime; cached raw generations can be graded without allocating model/device memory".into() })
}

pub(super) async fn verify_saved_trials(
    state: &State,
    config: &SearchConfig,
    output: &Path,
) -> Result<()> {
    let registry = Registry::load(&config.initial_registry)?;
    let freeze: serde_json::Value = serde_json::from_slice(&fs::read(output.join("freeze.json"))?)?;
    let expected_engine = freeze["engine_binary_sha256"]
        .as_str()
        .context("search engine pin absent")?
        .to_owned();
    if expected_engine != engine_sha256()? {
        let hash = expected_engine.clone();
        tokio::task::spawn_blocking(move || {
            crate::engine::resolve(Path::new("state/engines"), &hash)
        })
        .await??;
    }
    let mut seen = BTreeSet::new();
    for trial in &state.trials {
        ensure!(
            seen.insert(&trial.id),
            "duplicate trial in saved search state"
        );
        let model = expected_trial_model(trial, config, &registry, output)?;
        if trial.pruned {
            let copy = trial.clone();
            ensure!(
                tokio::task::spawn_blocking(move || prune::verify_pruned_directory(
                    &copy.directory,
                    &copy.id,
                    &copy.model_id,
                    &copy.revision
                ))
                .await??,
                "saved prune decision lacks matching owned deletion evidence"
            );
        }
        if let (Some(directory), Some(artifact)) = (&trial.comparison, &trial.report) {
            let directory = directory.clone();
            let artifact = artifact.clone();
            let copy = trial.clone();
            let config = config.clone();
            let registry_path = if trial.imported {
                config.initial_registry.clone()
            } else {
                output.join("trials").join(&trial.id).join("registry.toml")
            };
            let expected_engine = expected_engine.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                ensure!(
                    artifact.path == directory.join("report.json"),
                    "saved report points outside its trial comparison"
                );
                artifact.verify(&std::env::current_dir()?)?;
                let report = comparative::verify_completed_comparison(&directory)?;
                let plan = serde_json::from_slice(&fs::read(directory.join("plan.json"))?)?;
                verify_plan_identity_pinned(
                    &plan,
                    &model,
                    &config,
                    copy.imported,
                    false,
                    &registry_path,
                    &expected_engine,
                )?;
                ensure!(
                    report["split"] == "development",
                    "teacher search cannot import another split"
                );
                let row = report["models"]
                    .as_array()
                    .context("model rows missing")?
                    .iter()
                    .find(|r| r["id"] == copy.id)
                    .context("saved model result missing")?;
                ensure!(
                    row["passed"].as_u64() == copy.passed
                        && row["model_id"] == copy.model_id
                        && row["role"] == copy.role
                        && row["observed"] == 16
                        && row["planned"] == copy.planned
                        && row["infrastructure_or_generation_errors"] == copy.errors,
                    "saved search score differs from trusted comparison"
                );
                Ok(())
            })
            .await??;
        } else {
            ensure!(
                trial.passed.is_none(),
                "unverified aggregate score in teacher search state"
            );
        }
        if let (Some(directory), Some(artifact)) =
            (&trial.qualification_comparison, &trial.qualification_report)
        {
            let directory = directory.clone();
            let artifact = artifact.clone();
            let copy = trial.clone();
            let config = config.clone();
            let model = expected_trial_model(trial, &config, &registry, output)?;
            let registry_path = output
                .join("trials")
                .join(&trial.id)
                .join("qualification-registry.toml");
            let expected_engine = expected_engine.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                ensure!(
                    artifact.path == directory.join("report.json"),
                    "fresh report points outside its declared comparison"
                );
                artifact.verify(&std::env::current_dir()?)?;
                let observed = comparative::verify_completed_comparison(&directory)?;
                let plan = serde_json::from_slice(&fs::read(directory.join("plan.json"))?)?;
                verify_plan_identity_pinned(
                    &plan,
                    &model,
                    &config,
                    false,
                    true,
                    &registry_path,
                    &expected_engine,
                )?;
                let row = &observed["models"][0];
                ensure!(
                    observed["pack"] == "qualification"
                        && row["id"] == copy.id
                        && row["model_id"] == copy.model_id
                        && row["role"] == copy.role
                        && row["observed"] == 16
                        && row["planned"] == 16
                        && row["passed"].as_u64() == copy.qualification_passed
                        && row["infrastructure_or_generation_errors"].as_u64()
                            == copy.qualification_errors,
                    "saved fresh qualification differs from trusted linked observations"
                );
                Ok(())
            })
            .await??;
        }
    }
    Ok(())
}

pub(super) async fn load_best_model(
    config_path: &Path,
    output: &Path,
) -> Result<Option<RegistryModel>> {
    Ok(load_selected_models(config_path, output)
        .await?
        .into_iter()
        .next())
}

pub(super) async fn load_selected_models(
    config_path: &Path,
    output: &Path,
) -> Result<Vec<RegistryModel>> {
    relative(config_path)?;
    relative(output)?;
    let config = SearchConfig::load(config_path)?;
    let freeze_path = output.join("freeze.json");
    let freeze: serde_json::Value = serde_json::from_slice(&fs::read(&freeze_path)?)?;
    let registry: Artefact = serde_json::from_value(freeze["initial_registry"].clone())?;
    let qualification: Artefact = serde_json::from_value(freeze["qualification_pack"].clone())?;
    ensure!(
        freeze["config_sha256"] == artefact::file_hash(config_path)?
            && registry.path == config.initial_registry
            && freeze["held_out_allowed"] == false
            && freeze["promotable"] == false,
        "selected model config differs from its search freeze"
    );
    registry.verify(&std::env::current_dir()?)?;
    qualification.verify(&std::env::current_dir()?)?;
    ensure!(
        qualification::freeze(&output.join("qualification-pack.json"))? == qualification.sha256,
        "fresh qualification pack differs from the search freeze"
    );
    let state: State = serde_json::from_slice(&fs::read(output.join("state.json"))?)?;
    ensure!(
        state.schema_version == 1 && state.freeze_sha256 == artefact::file_hash(&freeze_path)?,
        "selected model state differs from its search freeze"
    );
    verify_saved_trials(&state, &config, output).await?;
    let original = Registry::load(&config.initial_registry)?;
    let keep = retained(&state, &config);
    let mut ranked = state
        .trials
        .iter()
        .filter(|t| {
            t.passed.is_some()
                && t.errors == 0
                && !t.pruned
                && t.role != "student_baseline"
                && keep.contains(&t.id)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| selection_order(b, a));
    let mut sources = BTreeSet::new();
    let mut selected = ranked
        .iter()
        .copied()
        .filter(|t| t.qualified)
        .filter(|t| sources.insert(&t.source_model_id))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected.extend(ranked.first().copied());
    }
    selected
        .into_iter()
        .map(|t| expected_trial_model(t, &config, &original, output))
        .collect()
}
