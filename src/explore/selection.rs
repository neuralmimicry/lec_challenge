//! Freeze selected source identities and their actual development evidence.
//! A later roster improvement cannot mutate an already training study.
use super::*;
use std::{
    fs::File,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SelectedTrial {
    pub model: RegistryModel,
    pub source_model_id: String,
    pub assets: Artefact,
    pub primary_plan: Artefact,
    pub primary_report: Artefact,
    pub qualification_plan: Option<Artefact>,
    pub qualification_report: Option<Artefact>,
    pub qualified: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Selection {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub registry: Artefact,
    pub search_freeze: Artefact,
    pub search_state_snapshot: Artefact,
    pub search_report_snapshot: Artefact,
    pub models: Vec<RegistryModel>,
    pub trials: Vec<SelectedTrial>,
    pub runtime: RuntimeOptions,
    pub target_models: usize,
    pub minimum_pass_rate: f64,
    pub qualified_target_met: bool,
    pub held_out_used: bool,
    pub production_eligible: bool,
}

pub(super) struct AssetLease(File);
impl AssetLease {
    pub fn acquire(directory: &Path) -> Result<Self> {
        relative(directory)?;
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join(".hub-download.lock"))?;
        // SAFETY: the owned descriptor remains open until the complete workflow ends.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) } == 0,
            "selected teacher is currently being changed; retry after its downloader/pruner completes"
        );
        Ok(Self(file))
    }
}
impl Drop for AssetLease {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

fn original_source(model: &RegistryModel) -> Result<String> {
    Ok(if model.weight_format == WeightFormat::Gguf {
        let proof: hub::HubGgufProvenance =
            serde_json::from_slice(&fs::read(model.directory.join("gguf-provenance.json"))?)?;
        proof.source_assets.manifest.model_id
    } else {
        model.model_id.clone()
    })
}

pub(super) async fn freeze(context: &mut ContextState) -> Result<Selection> {
    let models = teacher_search::load_selected_models(
        &context.options.search_config,
        &context.options.search_output,
    )
    .await?;
    ensure!(
        !models.is_empty(),
        "no independently verified retained teacher is available"
    );
    let search_state = read(&context.options.search_output.join("state.json"))?;
    let search_report = read(&context.options.search_output.join("report.json"))?;
    let search_config: toml::Value =
        toml::from_str(&fs::read_to_string(&context.options.search_config)?)?;
    let target_models = usize::try_from(
        search_config["target_models"]
            .as_integer()
            .context("search target missing")?,
    )?;
    let minimum_pass_rate = search_config["minimum_pass_rate"]
        .as_float()
        .context("search threshold missing")?;
    let directory = context.options.output.join("selection");
    fs::create_dir_all(&directory)?;
    artefact::write_json(&directory.join("search-state-snapshot.json"), &search_state)?;
    artefact::write_json(
        &directory.join("search-report-snapshot.json"),
        &search_report,
    )?;
    let registry_path = directory.join("registry.toml");
    let registry = Registry {
        schema_version: 1,
        models: models.clone(),
    };
    artefact::atomic_write(
        &registry_path,
        toml::to_string_pretty(&registry)?.as_bytes(),
    )?;
    let mut trials = Vec::new();
    for model in &models {
        let trial = search_state["trials"]
            .as_array()
            .context("search trials absent")?
            .iter()
            .find(|trial| trial["id"] == model.id)
            .context("selected teacher has no measured search trial")?;
        let primary = PathBuf::from(
            trial["comparison"]
                .as_str()
                .context("selected primary comparison absent")?,
        );
        let qualification = trial["qualification_comparison"]
            .as_str()
            .map(PathBuf::from);
        let qualified = trial["qualified"] == true;
        trials.push(SelectedTrial {
            model: model.clone(),
            source_model_id: original_source(model)?,
            assets: capture(
                &context.root,
                &model.directory.join(model.weight_format.manifest_name()),
            )?,
            primary_plan: capture(&context.root, &primary.join("plan.json"))?,
            primary_report: capture(&context.root, &primary.join("report.json"))?,
            qualification_plan: if qualified {
                Some(capture(
                    &context.root,
                    &qualification
                        .as_ref()
                        .context("qualified teacher lacks fresh comparison")?
                        .join("plan.json"),
                )?)
            } else {
                None
            },
            qualification_report: if qualified {
                Some(capture(
                    &context.root,
                    &qualification
                        .context("qualified teacher lacks fresh comparison")?
                        .join("report.json"),
                )?)
            } else {
                None
            },
            qualified,
        });
    }
    let runtime = if let Some(runtime) = &context.options.teacher_runtime {
        runtime.clone()
    } else {
        serde_json::from_value(read(&trials[0].primary_plan.path)?["runtime"].clone())?
    };
    let selection = Selection {
        schema_version: 1,
        evidence_kind: "frozen_local_research_teacher_selection".into(),
        registry: capture(&context.root, &registry_path)?,
        search_freeze: capture(
            &context.root,
            &context.options.search_output.join("freeze.json"),
        )?,
        search_state_snapshot: capture(
            &context.root,
            &directory.join("search-state-snapshot.json"),
        )?,
        search_report_snapshot: capture(
            &context.root,
            &directory.join("search-report-snapshot.json"),
        )?,
        qualified_target_met: trials.iter().filter(|trial| trial.qualified).count()
            >= target_models,
        models,
        trials,
        runtime,
        target_models,
        minimum_pass_rate,
        held_out_used: false,
        production_eligible: false,
    };
    verify(&selection, &context.root)?;
    let path = directory.join("selection.json");
    artefact::write_json(&path, &selection)?;
    context.state.selection = Some(capture(&context.root, &path)?);
    context.save().await?;
    Ok(selection)
}

pub(super) fn verify(selection: &Selection, root: &Path) -> Result<()> {
    ensure!(
        selection.schema_version == 1
            && selection.evidence_kind == "frozen_local_research_teacher_selection"
            && !selection.held_out_used
            && !selection.production_eligible
            && !selection.models.is_empty()
            && selection.models.len() == selection.trials.len()
            && (1..=12).contains(&selection.target_models)
            && selection.minimum_pass_rate.is_finite()
            && (0. ..=1.).contains(&selection.minimum_pass_rate),
        "invalid frozen exploration teacher selection"
    );
    selection.search_freeze.verify(root)?;
    selection.search_state_snapshot.verify(root)?;
    selection.search_report_snapshot.verify(root)?;
    let search_report = read(&selection.search_report_snapshot.path)?;
    ensure!(
        search_report["target_models"] == selection.target_models
            && search_report["minimum_pass_rate"] == selection.minimum_pass_rate,
        "frozen teacher target differs from its original search policy"
    );
    let registry = Registry::load(&selection.registry.verify(root)?)?;
    ensure!(
        same(&registry.models, &selection.models)?,
        "frozen teacher registry changed"
    );
    let mut sources = BTreeSet::new();
    for (model, trial) in selection.models.iter().zip(&selection.trials) {
        ensure!(
            same(model, &trial.model)?
                && original_source(model)? == trial.source_model_id
                && sources.insert(&trial.source_model_id),
            "frozen teacher source/model changed or duplicates an original source"
        );
        trial.assets.verify(root)?;
        match model.weight_format {
            WeightFormat::Gguf => {
                hub::validate_gguf_assets(&model.directory)?;
            }
            WeightFormat::Safetensors => {
                hub::validate_downloaded_assets(&model.directory)?;
            }
        }
        let primary_plan = read(&trial.primary_plan.verify(root)?)?;
        let primary = trial.primary_report.verify(root)?;
        let expected_model = serde_json::to_value(model)?;
        let expected_assets = serde_json::to_value(&trial.assets)?;
        ensure!(
            trial.primary_plan.path.parent() == trial.primary_report.path.parent()
                && primary_plan["models"]
                    .as_array()
                    .context("primary teacher plan models missing")?
                    .iter()
                    .any(|entry| entry["model"] == expected_model
                        && entry["assets"] == expected_assets),
            "frozen teacher primary plan names another model or asset identity"
        );
        let report = comparative::verify_completed_comparison(
            primary.parent().context("primary report parent absent")?,
        )?;
        let row = report["models"]
            .as_array()
            .context("primary model scores missing")?
            .iter()
            .find(|row| row["id"] == model.id)
            .context("primary teacher result missing")?;
        ensure!(
            report["split"] == "development"
                && row["planned"] == 16
                && row["observed"] == 16
                && row["infrastructure_or_generation_errors"] == 0,
            "selected teacher lacks healthy complete primary development evidence"
        );
        let mut qualified = false;
        if let (Some(plan), Some(report_file)) =
            (&trial.qualification_plan, &trial.qualification_report)
        {
            let planned = read(&plan.verify(root)?)?;
            let path = report_file.verify(root)?;
            let expected_model = serde_json::to_value(model)?;
            let expected_assets = serde_json::to_value(&trial.assets)?;
            ensure!(
                plan.path.parent() == report_file.path.parent()
                    && planned["models"]
                        .as_array()
                        .context("fresh teacher plan models missing")?
                        .iter()
                        .any(|entry| entry["model"] == expected_model
                            && entry["assets"] == expected_assets),
                "frozen fresh qualification plan names another model or assets"
            );
            let fresh = comparative::verify_completed_comparison(
                path.parent()
                    .context("qualification report parent absent")?,
            )?;
            let fresh_row = &fresh["models"][0];
            ensure!(
                fresh["pack"] == "qualification"
                    && fresh_row["id"] == model.id
                    && fresh_row["planned"] == 16
                    && fresh_row["observed"] == 16,
                "selected teacher fresh qualification denominator changed"
            );
            qualified = row["passed"]
                .as_u64()
                .context("primary pass count absent")? as f64
                / 16.
                >= selection.minimum_pass_rate
                && fresh_row["passed"]
                    .as_u64()
                    .context("fresh pass count absent")? as f64
                    / 16.
                    >= selection.minimum_pass_rate
                && fresh_row["infrastructure_or_generation_errors"] == 0;
        }
        ensure!(
            qualified == trial.qualified,
            "frozen teacher qualification differs from actual comparison evidence"
        );
    }
    let qualified = selection
        .trials
        .iter()
        .filter(|trial| trial.qualified)
        .count();
    ensure!(
        selection.qualified_target_met == (qualified >= selection.target_models)
            && (qualified > 0 || selection.models.len() == 1),
        "frozen research fallback or target claim is invalid"
    );
    Ok(())
}

/// A selected frozen identity may be rehydrated only from its owned deletion
/// receipt. Corruption and unclaimed absence are errors, never resampling.
pub(super) async fn rehydrate(selection: &Selection, offline: bool) -> Result<()> {
    for model in &selection.models {
        let pruned = if model.weight_format == WeightFormat::Gguf {
            hub::is_intentionally_pruned_gguf(&model.directory)?
        } else {
            let missing = teacher_search::verify_owned_pruned_model(model)?;
            if missing {
                let manifest: model::AssetManifest =
                    serde_json::from_slice(&fs::read(model.directory.join("assets.json"))?)?;
                let authority = read(&model.directory.join(".pruned-weights.json"))?;
                let removed = authority["removed_weights"]
                    .as_array()
                    .context("owned removed-weight list missing")?
                    .iter()
                    .map(|file| {
                        file["filename"]
                            .as_str()
                            .context("owned removed-weight filename missing")
                    })
                    .collect::<Result<BTreeSet<_>>>()?;
                for (name, hash) in &manifest.files {
                    relative(Path::new(name))?;
                    let path = model.directory.join(name);
                    if path.exists() {
                        ensure!(
                            artefact::file_hash(&path)? == *hash,
                            "present native teacher asset is corrupt; no implicit repair permitted"
                        );
                    } else {
                        ensure!(
                            removed.contains(name.as_str()),
                            "native teacher asset is missing without owned deletion authority"
                        );
                    }
                }
            }
            missing
        };
        if !pruned {
            continue;
        }
        ensure!(
            !offline,
            "frozen teacher {} was intentionally pruned and is unavailable offline; no HTTP attempted",
            model.id
        );
        eprintln!(
            "explore: rehydrating owned, frozen teacher {} without changing its identity",
            model.id
        );
        match model.weight_format {
            WeightFormat::Gguf => {
                let proof: hub::HubGgufProvenance = serde_json::from_slice(&fs::read(
                    model.directory.join("gguf-provenance.json"),
                )?)?;
                hub::download_gguf(hub::HubGgufDownloadOptions {
                    model_id: model.model_id.clone(),
                    revision: model.revision.clone(),
                    filename: proof.filename,
                    destination: model.directory.clone(),
                    token_stdin: false,
                    source: hub::HubGgufSource {
                        model_id: proof.source_assets.manifest.model_id,
                        revision: proof.source_assets.manifest.revision,
                    },
                    expected_sha256: Some(proof.file_sha256),
                    expected_bytes: Some(proof.file_bytes),
                })
                .await?;
            }
            WeightFormat::Safetensors => {
                hub::download_model(hub::HubDownloadOptions {
                    model_id: model.model_id.clone(),
                    revision: model.revision.clone(),
                    destination: model.directory.clone(),
                    concurrency: 2,
                    token_stdin: false,
                })
                .await?;
            }
        }
    }
    Ok(())
}
