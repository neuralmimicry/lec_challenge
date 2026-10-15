//! Search orchestration: each declared candidate is visited at most once per invocation.
use super::evidence::{frozen_runtime, verify_plan_identity, verify_saved_trials};
use super::selection::qualify_trial;
use super::*;

fn primary_result_status(passed: u64, errors: u64, minimum_pass_rate: f64) -> &'static str {
    if errors > 0 {
        // Complete coverage preserves failed outputs, but it cannot establish
        // model quality when generation or infrastructure did not complete.
        "complete_with_errors_inconclusive"
    } else if passed as f64 / 16.0 >= minimum_pass_rate {
        "primary_passed_waiting_for_fresh_qualification"
    } else {
        "verified_below_threshold"
    }
}

async fn prune_scored(output: &Path, state: &mut State, config: &SearchConfig) -> Result<()> {
    let keep = retained(state, config);
    let protected = state
        .trials
        .iter()
        .filter(|t| keep.contains(&t.id))
        .map(|t| t.directory.clone())
        .collect::<Vec<_>>();
    let eligible = state
        .trials
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            t.passed.is_some()
                && t.errors == 0
                && t.qualification_errors.unwrap_or(0) == 0
                && !t.pruned
                && !keep.contains(&t.id)
                && config
                    .managed_roots
                    .iter()
                    .any(|r| t.directory.starts_with(r) && t.directory != *r)
        })
        .map(|(i, t)| (i, t.clone()))
        .collect::<Vec<_>>();
    for (index, trial) in eligible {
        eprintln!(
            "teacher-search: pruning scored inactive model={} (metadata/reports retained)",
            trial.id
        );
        let guard = protected.clone();
        let directory = trial.directory.clone();
        let id = trial.id.clone();
        let result = tokio::task::spawn_blocking(move || {
            prune::prune_candidate(&directory, &guard, &id, false)
        })
        .await?;
        match result {
            Ok(measured)=>{
                state.trials[index].pruned=true;
                state.events.push(serde_json::json!({"kind":"pruned_scored_inactive_model","evidence":measured}));
            }
            Err(error)=>state.events.push(serde_json::json!({"kind":"prune_refused","model":trial.id,"reason":format!("{error:#}")})),
        }
        save(output, state).await?;
    }
    Ok(())
}

fn free_disk_bytes(path: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    let mut information = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: the C string and writable statvfs pointer live throughout this call.
    ensure!(
        unsafe { libc::statvfs(path.as_ptr(), information.as_mut_ptr()) } == 0,
        "free disk space could not be measured"
    );
    // SAFETY: successful statvfs initialises the complete structure.
    let information = unsafe { information.assume_init() };
    u64::try_from(u128::from(information.f_bavail) * u128::from(information.f_frsize))
        .context("free disk measurement overflow")
}

async fn import_initial(output: &Path, state: &mut State, config: &SearchConfig) -> Result<()> {
    if state.initialised {
        return Ok(());
    }
    let registry = Registry::load(&config.initial_registry)?;
    for comparison in &config.initial_comparisons {
        if !comparison.join("report.json").exists() {
            continue;
        }
        let directory = comparison.clone();
        let verified = tokio::task::spawn_blocking(move || {
            comparative::verify_completed_comparison(&directory)
        })
        .await?;
        let report = match verified {
            Ok(report) => report,
            Err(error) => {
                state.events.push(serde_json::json!({"kind":"initial_comparison_refused","directory":comparison,"reason":format!("{error:#}")}));
                continue;
            }
        };
        ensure!(
            report["split"] == "development" && report["source_count"] == 4,
            "initial comparison must use the original development sources"
        );
        let plan: serde_json::Value =
            serde_json::from_slice(&fs::read(comparison.join("plan.json"))?)?;
        let initial_hash = artefact::file_hash(&config.initial_registry)?;
        ensure!(
            plan["registry"]["path"] == serde_json::to_value(&config.initial_registry)?
                && plan["registry"]["sha256"] == initial_hash,
            "initial comparison does not bind the declared original registry"
        );
        for row in report["models"]
            .as_array()
            .context("initial model rows missing")?
        {
            let id = row["id"].as_str().context("initial model ID missing")?;
            if state.trials.iter().any(|t| t.id == id) {
                continue;
            }
            let model = registry
                .models
                .iter()
                .find(|m| m.id == id && row["model_id"] == m.model_id)
                .context("initial model differs from the original registry")?;
            verify_plan_identity(&plan, model, config, true, false, &config.initial_registry)?;
            ensure!(
                row["planned"] == 16 && row["observed"] == 16,
                "initial comparison denominator is incomplete"
            );
            let passed = row["passed"]
                .as_u64()
                .context("initial passed count missing")?;
            let errors = row["infrastructure_or_generation_errors"]
                .as_u64()
                .context("initial error count missing")?;
            if model.role != "student_baseline"
                && config
                    .managed_roots
                    .iter()
                    .any(|r| model.directory.starts_with(r) && model.directory != *r)
            {
                prune::claim_directory(
                    &model.directory,
                    &model.id,
                    &model.model_id,
                    &model.revision,
                )?;
            }
            state.trials.push(Trial {
                id: model.id.clone(),
                model_id: model.model_id.clone(),
                source_model_id: model.model_id.clone(),
                revision: model.revision.clone(),
                directory: model.directory.clone(),
                role: model.role.clone(),
                status: "verified_initial_comparison".into(),
                passed: Some(passed),
                planned: 16,
                errors,
                qualified: false,
                qualification_comparison: None,
                qualification_report: None,
                qualification_passed: None,
                qualification_errors: None,
                comparison: Some(comparison.clone()),
                report: Some(Artefact::capture(
                    &std::env::current_dir()?,
                    comparison.join("report.json"),
                )?),
                reason: None,
                elapsed_seconds: None,
                imported: true,
                pruned: false,
            });
        }
    }
    state.initialised = true;
    save(output, state).await
}

fn verify_candidate(proof: &hub::HubGgufProvenance, candidate: &Candidate) -> Result<()> {
    ensure!(
        proof.model_id == candidate.model_id
            && proof.revision == candidate.revision
            && proof.filename == candidate.filename
            && proof.file_sha256 == candidate.sha256
            && proof.file_bytes == candidate.bytes
            && proof.source_assets.manifest.model_id == candidate.source_model_id
            && proof.source_assets.manifest.revision == candidate.source_revision,
        "cached composite model differs from the frozen candidate pins"
    );
    Ok(())
}

async fn cached_candidate(candidate: &Candidate) -> Result<bool> {
    if !candidate.directory.join("gguf-assets.json").exists() {
        return Ok(false);
    }
    let copy = candidate.clone();
    tokio::task::spawn_blocking(move || {
        if hub::is_intentionally_pruned_gguf(&copy.directory)? {
            // Only an owned, manifest-bound deletion authorises rehydration.
            // A present corrupt weight remains an error, never a cache miss.
            return Ok(false);
        }
        let proof = hub::validate_gguf_assets(&copy.directory)?;
        verify_candidate(&proof, &copy)?;
        Ok(true)
    })
    .await?
}

pub(super) async fn run(
    config_path: &Path,
    output: &Path,
    offline: bool,
) -> Result<serde_json::Value> {
    relative(config_path)?;
    relative(output)?;
    safe_directory(output)?;
    let config = SearchConfig::load(config_path)?;
    let _output_lease = Lease::acquire(&output.join(".search.lock"))?;
    let mut pool_leases = Vec::new();
    for root in &config.managed_roots {
        safe_directory(root)?;
        pool_leases.push(Lease::acquire(&root.join(".teacher-search-pool.lock"))?);
    }
    let freeze_path = output.join("freeze.json");
    let corpus = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::Development)
        .collect::<Vec<_>>();
    ensure!(
        corpus.len() == 16,
        "teacher selection expects the complete 16-task development denominator"
    );
    let qualification_path = output.join("qualification-pack.json");
    let qualification_hash = qualification::freeze(&qualification_path)?;
    let qualification_artefact = Artefact::capture(&std::env::current_dir()?, qualification_path)?;
    ensure!(
        qualification_artefact.sha256 == qualification_hash,
        "fresh qualification publication checksum changed"
    );
    let initial_comparisons = config.initial_comparisons.iter().filter(|p| p.join("report.json").exists())
        .map(|p| -> Result<_> { Ok(serde_json::json!({"plan":Artefact::capture(&std::env::current_dir()?,p.join("plan.json"))?,"report":Artefact::capture(&std::env::current_dir()?,p.join("report.json"))?})) })
        .collect::<Result<Vec<_>>>()?;
    let freeze = serde_json::json!({"schema_version":1,"config_sha256":artefact::file_hash(config_path)?,"initial_registry":Artefact::capture(&std::env::current_dir()?, config.initial_registry.clone())?,"initial_comparisons":initial_comparisons,"engine_binary_sha256":engine_sha256()?,"development_tasks":corpus,"qualification_pack":qualification_artefact,"promotable":false,"held_out_allowed":false});
    if freeze_path.exists() {
        let previous: serde_json::Value = serde_json::from_slice(&fs::read(&freeze_path)?)?;
        ensure!(
            previous == freeze,
            "search protocol/binary/tasks changed; use a new explicitly versioned output directory"
        );
    } else {
        artefact::write_json(&freeze_path, &freeze)?;
    }
    let freeze_hash = artefact::file_hash(&freeze_path)?;
    let state_path = output.join("state.json");
    let mut state: State = if state_path.exists() {
        let prior: State = serde_json::from_slice(&fs::read(&state_path)?)?;
        ensure!(
            prior.schema_version == 1 && prior.freeze_sha256 == freeze_hash,
            "search state differs from its immutable freeze"
        );
        prior
    } else {
        State {
            schema_version: 1,
            freeze_sha256: freeze_hash,
            initialised: false,
            trials: vec![],
            download_bytes_reserved: 0,
            cached_only: offline,
            connectivity_reason: None,
            events: vec![],
        }
    };
    // Each invocation may explicitly re-enable public acquisition; a failure in
    // this invocation switches all remaining candidates to cached-only mode.
    state.cached_only = offline;
    if offline {
        state.connectivity_reason =
            Some("Explicit offline mode: cached/downloaded models only; zero HTTP requests".into());
    } else {
        state.connectivity_reason = None;
    }
    verify_saved_trials(&state, &config, output).await?;
    import_initial(output, &mut state, &config).await?;
    // Completed primary evaluations may have survived a crash immediately before
    // fresh qualification. Resume these before visiting any new candidate.
    for index in 0..state.trials.len() {
        qualify_trial(output, &mut state, &config, index).await?;
    }
    prune_scored(output, &mut state, &config).await?;
    if report(&state, &config, "checked_existing")["qualified_target_met"] == true {
        return finish(output, &state, &config, "qualified_target_met").await;
    }
    for (iteration, candidate) in config
        .candidates
        .iter()
        .take(config.maximum_iterations)
        .enumerate()
    {
        if state
            .trials
            .iter()
            .any(|t| t.id == candidate.id && t.passed.is_some())
        {
            continue;
        }
        let started = Instant::now();
        eprintln!(
            "teacher-search: iteration={}/{} candidate={} target={} threshold={:.2}% cost=unmeasured",
            iteration + 1,
            config.maximum_iterations,
            candidate.id,
            config.target_models,
            100.0 * config.minimum_pass_rate
        );
        let index = if let Some(index) = state.trials.iter().position(|t| t.id == candidate.id) {
            index
        } else {
            state.trials.push(Trial {
                id: candidate.id.clone(),
                model_id: candidate.model_id.clone(),
                source_model_id: candidate.source_model_id.clone(),
                revision: candidate.revision.clone(),
                directory: candidate.directory.clone(),
                role: "coding_teacher".into(),
                status: "pending".into(),
                passed: None,
                planned: 16,
                errors: 0,
                qualified: false,
                qualification_comparison: None,
                qualification_report: None,
                qualification_passed: None,
                qualification_errors: None,
                comparison: None,
                report: None,
                reason: None,
                elapsed_seconds: None,
                imported: false,
                pruned: false,
            });
            state.trials.len() - 1
        };
        if state.trials[index].reason.is_some() {
            // Preserve the complete previous attempt before a retry can replace
            // its mutable status/reason. Published generations remain immutable.
            let path = output
                .join("attempts")
                .join(&candidate.id)
                .join(format!("{}.json", uuid::Uuid::new_v4()));
            let comparison = output.join("trials").join(&candidate.id).join("comparison");
            let resources_path = output
                .join("trials")
                .join(&candidate.id)
                .join("resources.json");
            let resource_receipt = if resources_path.exists() {
                let preserved = path.with_extension("resources.json");
                artefact::atomic_write(&preserved, &fs::read(&resources_path)?)?;
                Some(Artefact::capture(&std::env::current_dir()?, preserved)?)
            } else {
                None
            };
            let plan_path = comparison.join("plan.json");
            let runtime = if plan_path.exists() {
                Some(
                    serde_json::from_slice::<serde_json::Value>(&fs::read(plan_path)?)?["runtime"]
                        .clone(),
                )
            } else if resources_path.exists() {
                Some(serde_json::from_slice::<serde_json::Value>(&fs::read(&resources_path)?)?["selection"]["options"].clone())
            } else {
                None
            };
            let kind = if ["pending", "downloading", "evaluating"]
                .contains(&state.trials[index].status.as_str())
            {
                "interrupted_attempt_preserved_before_retry"
            } else {
                "attempt_failed_before_retry"
            };
            artefact::write_json(
                &path,
                &serde_json::json!({
                    "schema_version":1,"kind":kind,
                    "recorded_unix_ms":crate::ledger::unix_millis(),
                    "trial":state.trials[index],"runtime":runtime,"comparison":comparison,"resources":resource_receipt,
                    "generation_policy":"Existing raw results, including failures, are never resampled by retrying infrastructure."
                }),
            )?;
            state.events.push(serde_json::json!({"kind":kind,
                "candidate":candidate.id,"evidence":Artefact::capture(&std::env::current_dir()?,path)?,
                "recorded_unix_ms":crate::ledger::unix_millis()}));
            save(output, &state).await?;
        }
        let trial_directory = output.join("trials").join(&candidate.id);
        safe_directory(&trial_directory)?;
        let registry_path = trial_directory.join("registry.toml");
        let registry = Registry {
            schema_version: 1,
            models: vec![candidate.registry_model()],
        };
        let registry_bytes = toml::to_string_pretty(&registry)?;
        if registry_path.exists() {
            ensure!(
                fs::read_to_string(&registry_path)? == registry_bytes,
                "per-trial registry changed"
            );
        } else {
            artefact::atomic_write(&registry_path, registry_bytes.as_bytes())?;
        }
        let plan_path = trial_directory.join("comparison/plan.json");
        // Runtime and original resource evidence are immutable across resumes.
        // A busy GPU must not prevent grading already durable raw generations.
        let selection = if plan_path.exists() {
            frozen_runtime(&plan_path, candidate, &config, &registry_path)?
        } else {
            let cached = match cached_candidate(candidate).await {
                Ok(cached) => cached,
                Err(error) => {
                    state.trials[index].status = "invalid_cache".into();
                    state.trials[index].reason = Some(format!("{error:#}"));
                    save(output, &state).await?;
                    continue;
                }
            };
            if !cached {
                if state.cached_only {
                    state.trials[index].status = "waiting_for_cached_assets".into();
                    state.trials[index].reason = Some(
                    "Cached/downloaded model unavailable; offline mode performs no HTTP requests"
                        .into(),
                );
                    save(output, &state).await?;
                    continue;
                }
                let charge = candidate
                    .bytes
                    .checked_add(64 << 20)
                    .context("asset reservation overflow")?;
                let already_reserved = state.events.iter().any(|e| {
                    e["kind"] == "asset_budget_reserved" && e["candidate"] == candidate.id
                });
                if !already_reserved
                    && state
                        .download_bytes_reserved
                        .checked_add(charge)
                        .is_none_or(|n| n > config.maximum_download_bytes)
                {
                    state.trials[index].status = "asset_budget_stopped".into();
                    state.trials[index].reason = Some(
                        "Declared maximum asset download bytes exceeded before acquisition".into(),
                    );
                    save(output, &state).await?;
                    continue;
                }
                safe_directory(&candidate.directory)?;
                if free_disk_bytes(&candidate.directory)?
                    < config.minimum_free_disk_bytes.saturating_add(charge)
                {
                    state.trials[index].status = "disk_floor_stopped".into();
                    state.trials[index].reason = Some(
                    "Insufficient measured free space after retaining the configured disk floor"
                        .into(),
                );
                    save(output, &state).await?;
                    continue;
                }
                prune::claim_directory(
                    &candidate.directory,
                    &candidate.id,
                    &candidate.model_id,
                    &candidate.revision,
                )?;
                if !already_reserved {
                    state.download_bytes_reserved += charge;
                    state.events.push(serde_json::json!({"kind":"asset_budget_reserved","candidate":candidate.id,"bytes":charge}));
                }
                state.trials[index].status = "downloading".into();
                save(output, &state).await?;
                eprintln!(
                    "teacher-search: downloading immutable model={} bytes={} (public Hub; no paid calls)",
                    candidate.model_id, candidate.bytes
                );
                let downloaded = hub::download_gguf(hub::HubGgufDownloadOptions {
                    model_id: candidate.model_id.clone(),
                    revision: candidate.revision.clone(),
                    filename: candidate.filename.clone(),
                    destination: candidate.directory.clone(),
                    token_stdin: false,
                    source: hub::HubGgufSource {
                        model_id: candidate.source_model_id.clone(),
                        revision: candidate.source_revision.clone(),
                    },
                    expected_sha256: Some(candidate.sha256.clone()),
                    expected_bytes: Some(candidate.bytes),
                })
                .await;
                match downloaded {
                    Ok(proof) => verify_candidate(&proof, candidate)?,
                    Err(error) => {
                        if hub::is_connectivity_error(&error) {
                            state.cached_only = true;
                            state.connectivity_reason=Some("HuggingFace unreachable; cached-only for the remainder of this invocation".into());
                            eprintln!(
                                "teacher-search: HuggingFace unreachable; continuing with cached/downloaded models only"
                            );
                        }
                        state.trials[index].status = if state.cached_only {
                            "waiting_for_cached_assets"
                        } else {
                            "download_rejected"
                        }
                        .into();
                        state.trials[index].reason = Some(format!("{error:#}"));
                        save(output, &state).await?;
                        continue;
                    }
                }
            } else {
                prune::claim_directory(
                    &candidate.directory,
                    &candidate.id,
                    &candidate.model_id,
                    &candidate.revision,
                )?;
                eprintln!(
                    "teacher-search: reusing checksum-verified cached model={} (zero Hub requests)",
                    candidate.id
                );
            }
            let candidate_copy = candidate.clone();
            let maximum = config.max_new_tokens;
            let context = config.context_tokens;
            let tasks = corpus.clone();
            let estimates = tokio::task::spawn_blocking(move || -> Result<_> {
                let encoder =
                    model::ComparativePromptEncoder::load(&candidate_copy.directory, true)?;
                let input = tasks
                    .iter()
                    .map(|t| encoder.input_tokens(&t.prompt()))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .max()
                    .context("no development prompts")?;
                let total = input.checked_add(maximum).context("context overflow")?;
                ensure!(
                    total <= context,
                    "actual maximum prompt plus output exceeds the declared context cap"
                );
                let source: model::QwenConfig = serde_json::from_slice(&fs::read(
                    candidate_copy.directory.join("source/config.json"),
                )?)?;
                ensure!(
                    source.model_type == candidate_copy.architecture,
                    "verified source architecture differs from candidate"
                );
                let gpu =
                    model::estimate_quantised_memory(&source, candidate_copy.bytes, context, true)?;
                let cpu = model::estimate_quantised_memory(
                    &source,
                    candidate_copy.bytes,
                    context,
                    false,
                )?;
                ensure!(
                    cpu.dense_embedding_bytes == candidate_copy.dense_embedding_bytes
                        && cpu.kv_cache_bytes / context as u64 == candidate_copy.kv_bytes_per_token,
                    "declared dimensions differ from verified original configuration"
                );
                Ok((total, gpu, cpu))
            })
            .await?;
            let (maximum_context, gpu, cpu) = match estimates {
                Ok(value) => value,
                Err(error) => {
                    state.trials[index].status = "context_or_memory_plan_rejected".into();
                    state.trials[index].reason = Some(format!("{error:#}"));
                    save(output, &state).await?;
                    continue;
                }
            };
            let profile = tokio::task::spawn_blocking(platform::detect).await??;
            let selection = match platform::select_runtime(
                &profile,
                &config.device,
                config.gpu_ordinal,
                true,
                gpu.conservative_total_bytes,
                cpu.conservative_total_bytes,
            ) {
                Ok(value) => value,
                Err(error) => {
                    state.trials[index].status = "resource_blocked".into();
                    state.trials[index].reason = Some(format!("{error:#}"));
                    save(output, &state).await?;
                    continue;
                }
            };
            let resource = serde_json::json!({"profile":profile,"selection":selection,"maximum_actual_context_tokens":maximum_context,"reserved_context_tokens":context,"gpu_estimate":gpu,"cpu_estimate":cpu});
            artefact::write_json(&trial_directory.join("resources.json"), &resource)?;
            selection
        };
        state.trials[index].status = "evaluating".into();
        state.trials[index].comparison = Some(trial_directory.join("comparison"));
        state.trials[index].reason = Some(selection.reason.clone());
        save(output, &state).await?;
        eprintln!(
            "teacher-search: evaluating model={} runtime={:?}; {}",
            candidate.id, selection.options, selection.reason
        );
        let result = comparative::run(CompareOptions {
            registry: registry_path,
            output: trial_directory.join("comparison"),
            split: ComparisonSplit::Development,
            runtime: selection.options,
            max_tokens: config.max_new_tokens,
            seed: config.seed,
            models: vec![candidate.id.clone()],
        })
        .await;
        match result {
            Ok(_) => {
                let directory = trial_directory.join("comparison");
                let checked = directory.clone();
                let observed = tokio::task::spawn_blocking(move || {
                    comparative::verify_completed_comparison(&checked)
                })
                .await??;
                let row = &observed["models"][0];
                ensure!(
                    row["id"] == candidate.id && row["planned"] == 16 && row["observed"] == 16,
                    "candidate evaluation has an incomplete denominator"
                );
                let passed = row["passed"]
                    .as_u64()
                    .context("candidate passed count missing")?;
                let errors = row["infrastructure_or_generation_errors"]
                    .as_u64()
                    .context("candidate error count missing")?;
                state.trials[index].passed = Some(passed);
                state.trials[index].errors = errors;
                state.trials[index].qualified = false;
                state.trials[index].status =
                    primary_result_status(passed, errors, config.minimum_pass_rate).into();
                state.trials[index].report = Some(Artefact::capture(
                    &std::env::current_dir()?,
                    directory.join("report.json"),
                )?);
                eprintln!(
                    "teacher-search: result model={} passed={}/16 generation/infrastructure_errors={} qualified={}",
                    candidate.id, passed, errors, state.trials[index].qualified
                );
            }
            Err(error) => {
                state.trials[index].status = "evaluation_incomplete".into();
                state.trials[index].reason = Some(format!("{error:#}"));
            }
        }
        state.trials[index].elapsed_seconds = Some(started.elapsed().as_secs_f64());
        save(output, &state).await?;
        qualify_trial(output, &mut state, &config, index).await?;
        prune_scored(output, &mut state, &config).await?;
        let _ = finish(output, &state, &config, "searching").await?;
        if report(&state, &config, "checked")["qualified_target_met"] == true {
            return finish(output, &state, &config, "qualified_target_met").await;
        }
    }
    prune_scored(output, &mut state, &config).await?;
    finish(
        output,
        &state,
        &config,
        if state.cached_only {
            "cached_only_shortfall"
        } else {
            "declared_roster_exhausted_shortfall"
        },
    )
    .await
}

#[cfg(test)]
mod result_status_tests {
    use super::primary_result_status;

    #[test]
    fn failed_generations_remain_inconclusive_even_when_the_observed_score_is_high() {
        for (passed, errors) in [(0, 16), (7, 6), (15, 1)] {
            assert_eq!(
                primary_result_status(passed, errors, 0.8),
                "complete_with_errors_inconclusive"
            );
        }
        assert_eq!(primary_result_status(7, 0, 0.8), "verified_below_threshold");
        assert_eq!(
            primary_result_status(13, 0, 0.8),
            "primary_passed_waiting_for_fresh_qualification"
        );
    }
}
