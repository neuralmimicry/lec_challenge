//! Independent development qualification after a successful primary selection.
use super::evidence::{expected_trial_model, verify_plan_identity};
use super::*;

pub(super) async fn qualify_trial(
    output: &Path,
    state: &mut State,
    config: &SearchConfig,
    index: usize,
) -> Result<()> {
    let trial = state.trials[index].clone();
    if trial.role == "student_baseline"
        || trial.pruned
        || trial.qualification_passed.is_some()
        || trial.errors != 0
        || !trial
            .passed
            .is_some_and(|n| n as f64 / 16.0 >= config.minimum_pass_rate)
    {
        return Ok(());
    }
    let original = Registry::load(&config.initial_registry)?;
    let model = expected_trial_model(&trial, config, &original, output)?;
    let copied_model = model.clone();
    let context = config.context_tokens;
    let max_new_tokens = config.max_new_tokens;
    let context_check = tokio::task::spawn_blocking(move || -> Result<()> {
        let encoder = model::ComparativePromptEncoder::load(
            &copied_model.directory,
            copied_model.weight_format == WeightFormat::Gguf,
        )?;
        for task in qualification::corpus() {
            ensure!(
                encoder
                    .input_tokens(&task.prompt())?
                    .checked_add(max_new_tokens)
                    .is_some_and(|n| n <= context),
                "fresh qualification prompt plus output exceeds the frozen context cap"
            );
        }
        Ok(())
    })
    .await?;
    if let Err(error) = context_check {
        state.trials[index].status = "fresh_qualification_context_rejected".into();
        state.trials[index].reason = Some(format!("{error:#}"));
        return save(output, state).await;
    }
    let directory = output.join("trials").join(&trial.id);
    safe_directory(&directory)?;
    let registry_path = directory.join("qualification-registry.toml");
    let registry = Registry {
        schema_version: 1,
        models: vec![model.clone()],
    };
    let bytes = toml::to_string_pretty(&registry)?;
    if registry_path.exists() {
        ensure!(
            fs::read_to_string(&registry_path)? == bytes,
            "fresh qualification registry changed"
        );
    } else {
        artefact::atomic_write(&registry_path, bytes.as_bytes())?;
    }
    let primary = trial
        .comparison
        .as_ref()
        .context("primary comparison missing before qualification")?;
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(primary.join("plan.json"))?)?;
    let runtime: model::RuntimeOptions = serde_json::from_value(plan["runtime"].clone())?;
    let comparison = directory.join("qualification");
    state.trials[index].qualification_comparison = Some(comparison.clone());
    state.trials[index].status = "qualifying_on_fresh_development_pack".into();
    save(output, state).await?;
    eprintln!(
        "teacher-search: primary model={} passed={}/16; evaluating independent frozen 16-task qualification pack",
        trial.id,
        trial.passed.unwrap_or_default()
    );
    let run = comparative::run_qualification(CompareOptions {
        registry: registry_path.clone(),
        output: comparison.clone(),
        split: ComparisonSplit::Development,
        runtime,
        max_tokens: config.max_new_tokens,
        seed: config.seed,
        models: vec![trial.id.clone()],
    })
    .await;
    let result = match run {
        Ok(_) => {
            let checked = comparison.clone();
            let config = config.clone();
            tokio::task::spawn_blocking(move || -> Result<_> {
                let observed = comparative::verify_completed_comparison(&checked)?;
                let plan = serde_json::from_slice(&fs::read(checked.join("plan.json"))?)?;
                verify_plan_identity(&plan, &model, &config, false, true, &registry_path)?;
                let row = &observed["models"][0];
                ensure!(
                    observed["pack"] == "qualification"
                        && row["id"] == trial.id
                        && row["observed"] == 16
                        && row["planned"] == 16,
                    "incomplete fresh qualification denominator"
                );
                let passed = row["passed"].as_u64().context("fresh pass count missing")?;
                let errors = row["infrastructure_or_generation_errors"]
                    .as_u64()
                    .context("fresh error count missing")?;
                let artefact =
                    Artefact::capture(&std::env::current_dir()?, checked.join("report.json"))?;
                Ok((passed, errors, artefact))
            })
            .await?
        }
        Err(error) => Err(error),
    };
    match result {
        Ok((passed, errors, artefact)) => {
            let row = &mut state.trials[index];
            row.qualification_passed = Some(passed);
            row.qualification_errors = Some(errors);
            row.qualification_report = Some(artefact);
            row.qualified = errors == 0 && passed as f64 / 16.0 >= config.minimum_pass_rate;
            row.status = if row.qualified {
                "qualified_on_both_development_packs"
            } else {
                "fresh_qualification_below_threshold"
            }
            .into();
            eprintln!(
                "teacher-search: fresh qualification model={} passed={passed}/16 errors={errors} qualified={}",
                row.id, row.qualified
            );
        }
        Err(error) => {
            state.trials[index].status = "fresh_qualification_incomplete".into();
            state.trials[index].reason = Some(format!("{error:#}"));
        }
    }
    save(output, state).await
}
