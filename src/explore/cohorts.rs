//! Rollout studies are isolated by initial student and organiser. An external
//! teacher's source ID is never rewritten to hide heterogeneous student origins.
use super::*;
use experiments::{ContentMethod, DataPolicy, StudentArchitecture, StudyConfig, StudyPlan};
use teacher_evidence::{ContentKind, RolloutMode};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Cohort {
    key: String,
    student: StudentArchitecture,
    mode: RolloutMode,
    base: PathBuf,
    assets: Artefact,
    initial_weight_sha256: String,
    input_bundle: Artefact,
    teachers: Vec<String>,
    student_runtime: RuntimeOptions,
    teacher_runtime: RuntimeOptions,
    maximum_tasks: usize,
    max_tokens: usize,
    seed: u64,
    source_config: Artefact,
    engine: PinnedEngine,
    reuse_student_attempts_from: Option<PathBuf>,
    numerical_equivalence: Option<crate::study_evaluate::NumericalEquivalence>,
    available: bool,
    reason: String,
}

fn key(value: &impl Serialize) -> Result<String> {
    Ok(artefact::hash(&serde_json::to_vec(value)?))
}
fn directory(root: &Path, cohort: &Cohort) -> PathBuf {
    root.join("cohorts").join(&cohort.key)
}
fn write_frozen(path: &Path, value: &impl Serialize) -> Result<()> {
    let expected = serde_json::to_value(value)?;
    if path.exists() {
        ensure!(
            read(path)? == expected,
            "cohort identity/configuration changed; use a fresh exploration output"
        );
        Ok(())
    } else {
        artefact::write_json(path, &expected)
    }
}

/// Freeze availability and identity before any cohort generation. The original
/// offline study retains its existing plan/output and scheduler reservations.
pub(super) async fn prepare(
    context: &mut ContextState,
    bundle: &Path,
    offline_plan: &StudyPlan,
    teacher_ids: &[String],
) -> Result<Vec<Cohort>> {
    let input = capture(&context.root, bundle)?;
    let source_config = context.protocol.study_config.clone();
    let mut rows = Vec::<Cohort>::new();
    let mut unavailable = Vec::new();
    for student in &offline_plan.inputs.students {
        let Some(assets) = &student.base_assets else {
            unavailable.push(json!({"student":student.architecture,"status":"unavailable","reason":"No independently verified frozen base assets are cached for this student."}));
            continue;
        };
        let asset_value: model::AssetManifest =
            serde_json::from_slice(&fs::read(assets.verify(&context.root)?)?)?;
        let base = assets
            .path
            .parent()
            .context("cohort base directory absent")?
            .to_owned();
        let (numerical_equivalence, parity_reason) = if student.architecture
            == StudentArchitecture::TernaryBitnet
        {
            let baseline = context.options.output.join("baselines/ternary_bitnet");
            let checked = baseline.clone();
            match tokio::task::spawn_blocking(move || {
                crate::study_evaluate::verify_numerical_equivalence(&checked)
            })
            .await?
            {
                Ok(mut receipt) => {
                    let frozen = read(&baseline.join("plan.json"))?;
                    if frozen["checkpoint"]["checkpoint_id"]
                        != asset_value.files["model.safetensors"]
                        || frozen["checkpoint"]["base_assets"] != serde_json::to_value(assets)?
                    {
                        receipt.qualified = false;
                        receipt.status = "identity_mismatch".into();
                        receipt.reasons.push(
                            "Attached baseline belongs to another original student/base identity."
                                .into(),
                        );
                    }
                    let reason = format!(
                        "Strict exact-base numerical equivalence {}: {}",
                        receipt.status,
                        receipt.reasons.join("; ")
                    );
                    (Some(receipt), reason)
                }
                Err(error) => (
                    None,
                    format!(
                        "Strict exact-base baseline numerical equivalence is missing/invalid: {error:#}"
                    ),
                ),
            }
        } else {
            (None, String::new())
        };
        let resources_fit = offline_plan
            .inputs
            .student_resources
            .iter()
            .find(|profile| profile.student == student.architecture)
            .is_some_and(|profile| {
                [&profile.quick, &profile.full].iter().all(|request| {
                    request.ram_bytes <= offline_plan.inputs.resources.ram_bytes
                        && request.cpu_threads <= offline_plan.inputs.resources.cpu_threads
                })
            });
        for mode in [RolloutMode::OnPolicy, RolloutMode::SelfDistillation] {
            let conventional = student.architecture == StudentArchitecture::ConventionalQwenLora;
            let available = (conventional
                || numerical_equivalence
                    .as_ref()
                    .is_some_and(|receipt| receipt.qualified))
                && resources_fit
                && student.training.usable()
                && student.exported_checkpoint_reload.usable();
            let reason = if available {
                "Frozen unadapted student cohort with verified training/reload and resource capability; actual new corrections and independent regrades determine organiser availability.".into()
            } else if !conventional {
                format!(
                    "Automatic BitNet cohort unavailable. {parity_reason}; configured resource profile fits={resources_fit}; training/reload capability={}/{}. Micro-model QAT cannot waive strict base equivalence.",
                    student.training.usable(),
                    student.exported_checkpoint_reload.usable()
                )
            } else {
                format!(
                    "Conventional student lacks verified training/export/reload or configured resource capability; resources_fit={resources_fit}."
                )
            };
            let mut cohort = Cohort {
                key: String::new(),
                student: student.architecture,
                mode,
                base: base.clone(),
                assets: assets.clone(),
                initial_weight_sha256: asset_value.files["model.safetensors"].clone(),
                input_bundle: input.clone(),
                teachers: if mode == RolloutMode::OnPolicy {
                    teacher_ids.to_vec()
                } else {
                    vec![]
                },
                student_runtime: offline_plan.inputs.student_runtime.clone(),
                teacher_runtime: if !conventional && mode == RolloutMode::SelfDistillation {
                    RuntimeOptions::default()
                } else {
                    offline_plan.inputs.teacher_runtime.clone()
                },
                maximum_tasks: context.options.content_maximum_tasks,
                max_tokens: context.options.content_max_tokens,
                seed: 1729,
                source_config: source_config.clone(),
                engine: context.protocol.stage_engine.clone(),
                reuse_student_attempts_from: if mode == RolloutMode::SelfDistillation {
                    rows.iter()
                        .find(|row| {
                            row.student == student.architecture && row.mode == RolloutMode::OnPolicy
                        })
                        .map(|row| directory(&context.options.output, row).join("rollouts"))
                } else {
                    None
                },
                numerical_equivalence: numerical_equivalence.clone(),
                available,
                reason,
            };
            cohort.key = key(&cohort)?;
            relative(&directory(&context.options.output, &cohort))?;
            rows.push(cohort);
        }
    }
    let path = context.options.output.join("cohort-catalogue.json");
    write_frozen(
        &path,
        &json!({"schema_version":1,"evidence_kind":"frozen_isolated_student_rollout_cohort_catalogue",
        "cohorts":rows,"unavailable_students":unavailable,"held_out_used":false,"production_eligible":false,
        "combined_external_and_self_teacher":"queued: requires independently verified same-cohort record composition; different origins are never merged",
        "coverage":"Finite base-initial cohorts, separate mode studies. Adapted-initial cohorts and arbitrary tool selection are not inferred. Each cohort shares the original invocation wall budget."}),
    )?;
    context.publish("cohort-catalogue", "verified_complete", vec![capture(&context.root,&path)?],
        json!({"available":rows.iter().filter(|row|row.available).count(),"unavailable":rows.iter().filter(|row|!row.available).count()+unavailable.len(),"bounded_coverage":true,
            "healthy":unavailable.is_empty() && rows.iter().all(|row|row.available)})).await?;
    Ok(rows)
}

fn filtered_config(
    original: &StudyConfig,
    cohort: &Cohort,
    bundle: &teacher_evidence::EvidenceBundle,
) -> Result<StudyConfig> {
    let samples = bundle
        .samples
        .iter()
        .filter(|sample| {
            sample.student_rollout.as_ref().is_some_and(|origin| {
                origin.student_checkpoint_sha256 == cohort.initial_weight_sha256
            })
        })
        .collect::<Vec<_>>();
    let mut config = original.clone();
    config.students = vec![cohort.student];
    config
        .student_training
        .retain(|override_| override_.student == cohort.student);
    config.policies = vec![if cohort.mode == RolloutMode::OnPolicy {
        DataPolicy::OnPolicy
    } else {
        DataPolicy::Offline
    }];
    config.self_distillation = vec![cohort.mode == RolloutMode::SelfDistillation];
    config.contents.retain(|content| match content {
        ContentMethod::Process => samples
            .iter()
            .any(|sample| sample.content == ContentKind::Process),
        ContentMethod::Context => samples
            .iter()
            .any(|sample| sample.content == ContentKind::Context),
        ContentMethod::ToolTrajectory => samples
            .iter()
            .any(|sample| sample.content == ContentKind::ToolTrajectory),
        ContentMethod::Preference => bundle.preferences.iter().any(|pair| {
            teacher_evidence::negative_pair_is_eligible(pair)
                && samples
                    .iter()
                    .any(|sample| sample.id == pair.chosen_sample_id)
        }),
    });
    // Preserve preregistered seeds, loss families and both fidelity budgets.
    // Pair-specific tensor/teacher compatibility remains the catalogue's job.
    config.validate()?;
    Ok(config)
}

fn scoped(context: &ContextState, cohort: &Cohort) -> Result<ContextState> {
    let mut options = context.options.clone();
    options.output = directory(&context.options.output, cohort);
    let policy_path = options.output.join("cohort.json");
    write_frozen(&policy_path, cohort)?;
    let protocol = capture(&context.root, &policy_path)?;
    let state_path = options.output.join("state.json");
    let state = if state_path.exists() {
        let state: State = serde_json::from_slice(&fs::read(&state_path)?)?;
        ensure!(
            same(&state.protocol, &protocol)?,
            "cohort state belongs to another frozen student/mode"
        );
        state
    } else {
        State {
            schema_version: 1,
            protocol,
            stages: BTreeMap::new(),
            selection: None,
            events: vec![],
        }
    };
    Ok(ContextState {
        options,
        root: context.root.clone(),
        protocol: context.protocol.clone(),
        state,
        started: context.started,
        deadline: context.deadline,
    })
}

async fn collect(context: &mut ContextState, cohort: &Cohort) -> Result<(PathBuf, Value)> {
    let output = context.options.output.join("rollouts");
    let mut process_ok = true;
    if !context.cached("rollout-evidence")? {
        if cohort.student == StudentArchitecture::TernaryBitnet {
            let hardware = tokio::task::spawn_blocking(crate::platform::detect).await??;
            ensure!(
                hardware.usable_ram_bytes >= 20u64 << 30,
                "native BitNet rollout needs 18 GiB model headroom plus 2 GiB controller reserve; progress retained"
            );
        }
        let mut args = vec![
            "collect-student-rollouts".into(),
            "--bundle".into(),
            cohort.input_bundle.path.display().to_string(),
            "--output".into(),
            output.display().to_string(),
            "--student-base".into(),
            cohort.base.display().to_string(),
            "--student-kind".into(),
            serde_json::to_value(cohort.student)?
                .as_str()
                .context("cohort student tag absent")?
                .into(),
            "--mode".into(),
            serde_json::to_value(cohort.mode)?
                .as_str()
                .context("cohort mode tag absent")?
                .into(),
            "--maximum-tasks".into(),
            cohort.maximum_tasks.to_string(),
            "--max-tokens".into(),
            cohort.max_tokens.to_string(),
            "--seed".into(),
            cohort.seed.to_string(),
            "--student-device".into(),
            "cpu".into(),
        ];
        // Student inference/training is explicitly CPU/F32 in study inputs;
        // the external teacher can use its separate frozen CUDA runtime.
        ensure!(
            cohort.student_runtime == RuntimeOptions::default(),
            "cohort collector requires the current explicit CPU/F32 student contract"
        );
        if !cohort.teachers.is_empty() {
            args.extend(["--teachers".into(), cohort.teachers.join(",")]);
        }
        if let Some(source) = &cohort.reuse_student_attempts_from {
            args.extend([
                "--reuse-student-attempts-from".into(),
                source.display().to_string(),
            ]);
        }
        args.extend(runtime_arguments(&cohort.teacher_runtime));
        if context.options.offline {
            args.push("--offline".into());
        }
        process_ok = process::stage(context, "rollout-evidence", &cohort.engine, args).await?;
    }
    let receipt = context
        .state
        .events
        .iter()
        .rev()
        .find(|event| event["stage"] == "rollout-evidence")
        .and_then(|event| serde_json::from_value::<Artefact>(event["attempt"].clone()).ok());
    let root = context.root.clone();
    let report_path = output.join("rollout-report.json");
    let report = tokio::task::spawn_blocking(move || {
        process::required_report(&root, &report_path, process_ok, receipt.as_ref())
    })
    .await??;
    ensure!(
        report["schema_version"] == 2
            && report["student_checkpoint_sha256"] == cohort.initial_weight_sha256,
        "cohort collection lacks current-batch health or frozen initial identity"
    );
    ensure!(
        process_ok || report["healthy_complete"] != true,
        "cohort process failed despite a healthy collection report; inspect saved attempt logs"
    );
    let bundle_path = output.join("bundle.json");
    let checked = bundle_path.clone();
    let _ =
        tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&checked)).await??;
    context
        .publish(
            "rollout-evidence",
            if report["healthy_complete"] == true {
                "verified_complete"
            } else {
                "bounded_evidence_gaps"
            },
            vec![
                capture(&context.root, &bundle_path)?,
                capture(&context.root, &output.join("rollout-report.json"))?,
            ],
            json!({"healthy":report["healthy_complete"],"current_batch":report}),
        )
        .await?;
    Ok((bundle_path, report))
}

async fn study(
    context: &mut ContextState,
    cohort: &Cohort,
    bundle_path: &Path,
    baselines: &BTreeMap<StudentArchitecture, PathBuf>,
) -> Result<Value> {
    let original = StudyConfig::load(&cohort.source_config.verify(&context.root)?)?;
    let checked = bundle_path.to_owned();
    let bundle =
        tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&checked)).await??;
    let config = filtered_config(&original, cohort, &bundle)?;
    let config_path = context.options.output.join("study-config.toml");
    let config_text = toml::to_string_pretty(&config)?;
    if config_path.exists() {
        ensure!(
            fs::read_to_string(&config_path)? == config_text,
            "cohort derived config changed"
        );
    } else {
        artefact::atomic_write(&config_path, config_text.as_bytes())?;
    }
    let output = context.options.output.join("study");
    if !context.cached("study-plan")? {
        let mut args = vec![
            "study-plan".into(),
            "--config".into(),
            config_path.display().to_string(),
            "--bundle".into(),
            bundle_path.display().to_string(),
            "--base".into(),
            context.options.base.display().to_string(),
            "--bitnet-base".into(),
            context.options.bitnet_base.display().to_string(),
            "--output".into(),
            output.display().to_string(),
        ];
        if let Some(jobs) = context.options.maximum_parallel_jobs {
            args.extend(["--maximum-parallel-jobs".into(), jobs.to_string()]);
        }
        args.extend(runtime_arguments(&cohort.teacher_runtime));
        if context.options.offline {
            args.push("--offline".into());
        }
        ensure!(
            process::stage(context, "study-plan", &cohort.engine, args).await?,
            "cohort study planning failed"
        );
    }
    let path = output.join("plan.json");
    let checked = path.clone();
    let plan = tokio::task::spawn_blocking(move || experiments::load_plan(&checked)).await??;
    ensure!(
        plan.inputs.evidence_bundle.sha256 == artefact::file_hash(bundle_path)?
            && plan.arms.iter().all(|arm| arm.student == cohort.student),
        "cohort study changed its bundle or mixed student architectures"
    );
    context.publish("study-plan","verified_complete",vec![capture(&context.root,&path)?,capture(&context.root,&config_path)?],
        json!({"applicable_arms":plan.arms.iter().filter(|arm|arm.applicable).count(),"bounded_coverage":true,"quick_coverage_gaps":plan.quick_coverage_gaps})).await?;
    if !plan.arms.iter().any(|arm| arm.applicable) {
        return Ok(
            json!({"status":"unavailable_no_applicable_checkpoint_bound_arm","healthy":false,"study":output}),
        );
    }
    let mut comparisons = Vec::new();
    for profile in ["quick", "full"] {
        let selection_path = output.join("selection.json");
        if profile == "full" {
            if !selection_path.exists() {
                let args = vec![
                    "study-select".into(),
                    "--study".into(),
                    output.display().to_string(),
                    "--top-groups".into(),
                    context.options.top_groups.to_string(),
                ];
                if !process::stage(context, "development-selection", &cohort.engine, args).await? {
                    return Ok(
                        json!({"status":"bounded_no_healthy_development_selection","healthy":false,"study":output,"comparisons":comparisons}),
                    );
                }
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
        let name = format!("{profile}-study");
        if !context.cached(&name)? {
            let cap = if profile == "quick" {
                plan.quick.timeout_seconds
            } else {
                plan.full.timeout_seconds
            };
            let mut args = vec![
                "study-run".into(),
                "--study".into(),
                output.display().to_string(),
                "--profile".into(),
                profile.into(),
                "--maximum-run-seconds".into(),
                context.remaining()?.as_secs().min(cap).max(1).to_string(),
            ];
            if profile == "full" {
                let selected = read(&selection_path)?;
                let ids: Vec<String> = serde_json::from_value(selected["selected_arms"].clone())?;
                args.extend([
                    "--selected-arms".into(),
                    ids.join(","),
                    "--selection-evidence".into(),
                    selection_path.display().to_string(),
                ]);
            }
            if context.options.offline {
                args.push("--offline".into());
            }
            let _ = process::stage(context, &name, &cohort.engine, args).await?;
        }
        let report_path = output.join(profile).join("report.json");
        let report = read(&report_path)?;
        let checked = output.join(profile);
        let plan_path = path.clone();
        let completed = tokio::task::spawn_blocking(move || {
            experiments::completed_results(&checked, &plan_path)
        })
        .await??;
        ensure!(
            report["completed_arms"] == completed.len(),
            "cohort aggregate differs from independently verified arm results"
        );
        context
            .publish(
                &name,
                if report["selected_profile_succeeded"] == true {
                    "verified_complete"
                } else {
                    "bounded_incomplete"
                },
                vec![capture(&context.root, &report_path)?],
                json!({"healthy":report["selected_profile_succeeded"],"report":report}),
            )
            .await?;
        if completed.is_empty() {
            return Ok(
                json!({"status":"bounded_no_complete_measured_arm","healthy":false,"study":output,"profile":profile}),
            );
        }
        comparisons.extend(
            baselines::compare_profile(context, profile, &output, &completed, baselines).await?,
        );
    }
    let healthy = baselines.contains_key(&cohort.student)
        && context.state.stages.values().all(|stage| {
            stage.status == "verified_complete"
                && stage
                    .details
                    .get("healthy")
                    .is_none_or(|value| *value == true)
        });
    Ok(
        json!({"status":if healthy {"completed_isolated_cohort_study"} else {"completed_with_bounded_evidence_gaps"},
        "healthy":healthy,"study":output,"comparisons":comparisons}),
    )
}

pub(super) async fn run(
    context: &mut ContextState,
    cohorts: Vec<Cohort>,
    baselines: &BTreeMap<StudentArchitecture, PathBuf>,
) -> Result<Vec<Value>> {
    let mut results = Vec::new();
    for cohort in cohorts {
        let name = format!("cohort-{}", cohort.key);
        if !cohort.available {
            results.push(json!({"cohort":cohort,"status":"unavailable","healthy":false}));
            continue;
        }
        let mut child = scoped(context, &cohort)?;
        child.save().await?;
        let outcome=async {
            let (bundle,report)=collect(&mut child,&cohort).await?;
            if report["new_verified_rollout_samples"].as_u64().unwrap_or(0)==0 {
                return Ok(json!({"status":"unavailable_no_new_passing_corrections","healthy":false,"collection":report}));
            }
            let mut result=study(&mut child,&cohort,&bundle,baselines).await?;
            result["collection"]=report.clone();
            result["healthy"]=json!(result["healthy"]==true && report["healthy_complete"]==true);
            Ok::<Value,anyhow::Error>(result)
        }.await;
        let result = match outcome {
            Ok(value) => value,
            Err(error) => {
                json!({"status":"incomplete_progress_saved","healthy":false,"error":format!("{error:#}")})
            }
        };
        let report_path = child.options.output.join("report.json");
        artefact::write_json(
            &report_path,
            &json!({"schema_version":1,"cohort":cohort,"result":result,"held_out_used":false,"production_eligible":false}),
        )?;
        child.save().await?;
        context
            .publish(
                &name,
                if result["healthy"] == true {
                    "verified_complete"
                } else {
                    "bounded_evidence_gaps"
                },
                vec![
                    capture(&context.root, &report_path)?,
                    capture(&context.root, &child.options.output.join("state.json"))?,
                ],
                json!({"healthy":result["healthy"],"cohort_key":cohort.key,"result":result}),
            )
            .await?;
        results.push(json!({"cohort_key":cohort.key,"result":result}));
        context.remaining()?;
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Cohort {
        let artifact = Artefact {
            path: "labelled-test-only".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        Cohort {
            key: String::new(),
            student: StudentArchitecture::ConventionalQwenLora,
            mode: RolloutMode::OnPolicy,
            base: "offline_bundle/base".into(),
            assets: artifact.clone(),
            initial_weight_sha256: "b".repeat(64),
            input_bundle: artifact.clone(),
            teachers: vec!["teacher".into()],
            student_runtime: RuntimeOptions::default(),
            teacher_runtime: RuntimeOptions::default(),
            maximum_tasks: 4,
            max_tokens: 256,
            seed: 1729,
            source_config: artifact,
            engine: PinnedEngine {
                path: "labelled-engine-not-executed".into(),
                sha256: "c".repeat(64),
                bytes: 0,
                cpu_build: None,
            },
            reuse_student_attempts_from: None,
            numerical_equivalence: None,
            available: true,
            reason: "labelled identity fixture; no model executed".into(),
        }
    }
    #[test]
    fn identity_and_paths_separate_student_mode_and_initial_checkpoint() {
        let first = fixture();
        let id = key(&first).unwrap();
        assert_eq!(id, key(&first).unwrap());
        let mut architecture = first.clone();
        architecture.student = StudentArchitecture::TernaryBitnet;
        let mut initial = first.clone();
        initial.initial_weight_sha256 = "d".repeat(64);
        let mut organiser = first.clone();
        organiser.mode = RolloutMode::SelfDistillation;
        for other in [architecture, initial, organiser] {
            assert_ne!(id, key(&other).unwrap());
        }
        let mut first = first;
        first.key = id;
        let path = directory(Path::new("reports/test"), &first);
        relative(&path).unwrap();
        assert!(path.starts_with("reports/test/cohorts"));
    }
    #[test]
    fn frozen_cohort_configuration_rejects_changed_inputs() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("cohort.json");
        write_frozen(&path, &json!({"initial":"a","seed":1729})).unwrap();
        write_frozen(&path, &json!({"initial":"a","seed":1729})).unwrap();
        assert!(write_frozen(&path, &json!({"initial":"b","seed":1729})).is_err());
    }
    #[test]
    fn derived_config_preserves_fidelity_and_seeds_without_importing_foreign_student_rows() {
        let original: StudyConfig =
            toml::from_str(include_str!("../../config/distillation-study.toml")).unwrap();
        let cohort = fixture();
        let artifact = cohort.assets.clone();
        let sample = teacher_evidence::EvidenceSample {
            id: "labelled-sample".into(),
            task: crate::tasks::corpus().remove(0),
            teacher_id: "teacher".into(),
            prompt: "labelled prompt".into(),
            completion: "labelled completion".into(),
            content: ContentKind::ToolTrajectory,
            observation: artifact.clone(),
            raw_generation: artifact.clone(),
            sandbox_report: artifact.clone(),
            audit: artifact.clone(),
            sequence_sha256: "x".repeat(64),
            provenance_kind: "labelled fixture, no model executed".into(),
            student_rollout: Some(teacher_evidence::StudentRolloutProvenance {
                student_checkpoint_sha256: cohort.initial_weight_sha256.clone(),
                student_raw_rollout: artifact.clone(),
                teacher_feedback: artifact.clone(),
                regraded_sandbox_report: artifact,
            }),
        };
        let mut foreign = sample.clone();
        foreign.id = "foreign".into();
        foreign.content = ContentKind::Context;
        foreign
            .student_rollout
            .as_mut()
            .unwrap()
            .student_checkpoint_sha256 = "z".repeat(64);
        let bundle = teacher_evidence::EvidenceBundle {
            schema_version: 1,
            evidence_kind: "labelled fixture only".into(),
            created_unix_ms: 0,
            teachers: vec![],
            samples: vec![sample, foreign],
            candidates: vec![],
            preferences: vec![],
            input_comparisons: vec![],
            content_records: vec![],
            rollout_records: vec![],
            capabilities: BTreeMap::new(),
            train_corpus_sha256: String::new(),
            limitations: vec![],
        };
        let filtered = filtered_config(&original, &cohort, &bundle).unwrap();
        assert_eq!(
            filtered.students,
            vec![StudentArchitecture::ConventionalQwenLora]
        );
        assert_eq!(filtered.policies, vec![DataPolicy::OnPolicy]);
        assert_eq!(filtered.self_distillation, vec![false]);
        assert_eq!(filtered.contents, vec![ContentMethod::ToolTrajectory]);
        assert_eq!(filtered.seeds, original.seeds);
        assert_eq!(filtered.progressive, original.progressive);
        assert_eq!(
            serde_json::to_value(&filtered.quick).unwrap(),
            serde_json::to_value(&original.quick).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&filtered.full).unwrap(),
            serde_json::to_value(&original.full).unwrap()
        );
        assert_eq!(original.students.len(), 2);
        let mut self_cohort = cohort;
        self_cohort.mode = RolloutMode::SelfDistillation;
        let self_config = filtered_config(&original, &self_cohort, &bundle).unwrap();
        assert_eq!(self_config.policies, vec![DataPolicy::Offline]);
        assert_eq!(self_config.self_distillation, vec![true]);
    }
}
