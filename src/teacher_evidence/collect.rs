use super::*;

/// Reuse verified immutable train comparisons, or generate the real 48-task
/// collection first. The existing comparative runner supplies resumable raw
/// generations, bounded execution and independently checked container audits.
pub async fn collect(options: CollectOptions) -> Result<EvidenceBundle> {
    relative(&options.output)?;
    ensure!(
        options.output.starts_with("reports") || options.output.starts_with("data"),
        "local-study output must live under reports or data"
    );
    if options.output.join("bundle.json").exists() {
        return tokio::task::spawn_blocking(move || {
            let bundle=load_verified(&options.output)?;
            let expected=if options.comparisons.is_empty(){vec![options.output.join("collection")]}else{options.comparisons.clone()};
            let recorded=bundle.input_comparisons.iter().filter(|a|a.path.file_name().is_some_and(|n|n=="report.json")).map(|a|a.path.parent().unwrap_or(Path::new("")).to_owned()).collect::<Vec<_>>();
            ensure!(expected==recorded,"existing evidence bundle belongs to different comparison inputs; use a new output directory");
            if options.comparisons.is_empty() {
                let plan:serde_json::Value=serde_json::from_slice(&fs::read(expected[0].join("plan.json"))?)?;
                let root=std::env::current_dir()?.canonicalize()?;
                let registry=capture(&root,&options.registry)?;
                let selected=bundle.teachers.iter().map(|t|t.id.as_str()).collect::<BTreeSet<_>>();
                ensure!(plan["registry"]==serde_json::to_value(registry)? && plan["runtime"]==serde_json::to_value(options.runtime)? && plan["max_tokens"]==options.max_tokens && plan["seed"]==options.seed && (options.models.is_empty() || options.models.iter().map(String::as_str).collect::<BTreeSet<_>>()==selected),"existing evidence bundle has a different frozen collection policy");
            }
            Ok(bundle)
        }).await.context("cached teacher evidence verification failed")?;
    }
    let mut comparisons = options.comparisons;
    if comparisons.is_empty() {
        let output = options.output.join("collection");
        comparative::run(CompareOptions {
            registry: options.registry,
            output: output.clone(),
            split: ComparisonSplit::Train,
            runtime: options.runtime,
            max_tokens: options.max_tokens,
            seed: options.seed,
            models: options.models,
        })
        .await?;
        comparisons.push(output);
    }
    // Multi-gigabyte model verification and filesystem checks must not occupy
    // an asynchronous executor thread.
    let output = options.output;
    tokio::task::spawn_blocking(move|| {
        let root=std::env::current_dir()?.canonicalize()?;
        let bundle=build_existing(&root,&comparisons)?;
        ensure!(!bundle.samples.is_empty(),"no passing, rights-cleared teacher output; no training bundle published");
        fs::create_dir_all(&output)?;
        let path=output.join("bundle.json");
        artefact::write_json(&path,&bundle)?;
        let manifest=serde_json::json!({"schema_version":1,"evidence_kind":"local_study_teacher_bundle_manifest","bundle":capture(&root,&path)?,"train_corpus_sha256":bundle.train_corpus_sha256,"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"promotable":false});
        artefact::write_json(&output.join("manifest.json"),&manifest)?;
        Ok(bundle)
    }).await.context("teacher evidence verifier worker failed")?
}

pub(super) fn build_existing(root: &Path, comparisons: &[PathBuf]) -> Result<EvidenceBundle> {
    ensure!(
        !comparisons.is_empty(),
        "no actual training comparisons supplied"
    );
    let mut teachers = BTreeMap::<String, TeacherEvidence>::new();
    let mut samples = Vec::new();
    let mut candidates = Vec::new();
    let mut eligibility = BTreeMap::new();
    let mut inputs = Vec::new();
    let mut seen = BTreeSet::new();
    for directory in comparisons {
        relative(directory)?;
        let report_path = directory.join("report.json");
        ensure!(seen.insert(directory.clone()), "duplicate comparison input");
        let plan_path = directory.join("plan.json");
        let plan: EvidencePlan = serde_json::from_slice(&fs::read(root.join(&plan_path))?)?;
        // Reject development/held-out inputs before the comparison verifier can
        // inspect any corresponding private test responses.
        validate_train_plan(&plan)?;
        comparative::verify_completed_comparison(&root.join(directory))?;
        inputs.push(capture(root, &report_path)?);
        inputs.push(capture(root, &plan_path)?);
        for frozen in &plan.models {
            if let Some(old) = teachers.get(&frozen.model.id) {
                ensure!(
                    json_equal(&old.model, &frozen.model)?
                        && json_equal(&old.assets, &frozen.assets)?,
                    "teacher identity changed between comparisons"
                );
            } else {
                let teacher = validate::teacher(root, frozen)?;
                teachers.insert(frozen.model.id.clone(), teacher);
            }
            for task in &plan.tasks {
                let path = directory
                    .join(&frozen.model.id)
                    .join(format!("{}.json", task.id));
                let observation: Observation =
                    serde_json::from_slice(&fs::read(root.join(&path))?)?;
                let raw: serde_json::Value =
                    serde_json::from_slice(&fs::read(observation.raw_generation.verify(root)?)?)?;
                ensure!(
                    raw["evidence_kind"] == "actual_local_model_generation",
                    "fixtures and reference solutions cannot enter local-study training"
                );
                let Some(generation) = &observation.generation else {
                    continue;
                };
                let Some(sandbox) = &observation.sandbox_report else {
                    continue;
                };
                let report: SandboxReport =
                    serde_json::from_slice(&fs::read(sandbox.verify(root)?)?)?;
                let audit = capture(root, &report.audit_path)?;
                ensure!(
                    audit.sha256 == report.audit_sha256,
                    "trusted audit identity changed"
                );
                let candidate = EvidenceCandidate {
                    task_id: task.id.clone(),
                    teacher_id: frozen.model.id.clone(),
                    completion: generation.completion.clone(),
                    observation: capture(root, &path)?,
                    raw_generation: observation.raw_generation.clone(),
                    sandbox_report: sandbox.clone(),
                    audit: audit.clone(),
                    passed: observation.success,
                    quarantined: report.quarantined,
                };
                if observation.success {
                    ensure!(
                        generation.complete
                            && generation.finish_reason == "stop"
                            && !generation.completion.trim().is_empty(),
                        "positive example is incomplete"
                    );
                    let id = artefact::hash(
                        format!(
                            "{}:{}:{}",
                            candidate.observation.sha256, task.id, frozen.model.id
                        )
                        .as_bytes(),
                    );
                    samples.push(EvidenceSample {
                        id,
                        task: task.clone(),
                        teacher_id: frozen.model.id.clone(),
                        prompt: observation.request.prompt.clone(),
                        completion: generation.completion.clone(),
                        content: ContentKind::CompleteSequence,
                        observation: candidate.observation.clone(),
                        raw_generation: observation.raw_generation.clone(),
                        sandbox_report: sandbox.clone(),
                        audit,
                        sequence_sha256: artefact::hash(generation.completion.as_bytes()),
                        provenance_kind:
                            "actual_local_model_generation_with_verified_container_grade".into(),
                        student_rollout: None,
                    });
                }
                if !candidate.passed {
                    eligibility.insert(
                        candidate.observation.sha256.clone(),
                        candidate_eligibility(root, &candidate, task)?,
                    );
                }
                candidates.push(candidate);
            }
        }
    }
    // Deterministic, code-normalised deduplication does not let formatting alone
    // multiply a teacher's contribution. Different genuine solutions survive.
    samples.sort_by(|a, b| {
        (&a.task.id, &a.teacher_id, &a.id).cmp(&(&b.task.id, &b.teacher_id, &b.id))
    });
    let mut sequences = BTreeSet::new();
    samples.retain(|s| {
        sequences.insert((
            s.task.id.clone(),
            s.teacher_id.clone(),
            normalised_code_hash(&s.completion),
        ))
    });
    let mut preferences = Vec::new();
    for chosen in &samples {
        for rejected in candidates.iter().filter(|c| {
            c.task_id == chosen.task.id
                && !c.passed
                && !c.completion.trim().is_empty()
                && eligibility
                    .get(&c.observation.sha256)
                    .is_some_and(ContentEligibility::negative_allowed)
        }) {
            // An oracle-only failure remains quarantined. Its independently
            // reconstructed role permits a negative target, never a positive.
            let (preference_prompt, prompt_transform) =
                preference_conditioning(&chosen.task, &chosen.prompt, &chosen.task.prompt());
            preferences.push(PreferencePair {id:artefact::hash(format!("{}:{}",chosen.id,rejected.observation.sha256).as_bytes()),task_id:chosen.task.id.clone(),chosen_sample_id:chosen.id.clone(),rejected:rejected.clone(),rejected_eligibility:eligibility.get(&rejected.observation.sha256).cloned(),preference_prompt:Some(preference_prompt),prompt_transform:Some(prompt_transform),reason:"chosen passes every trusted case; rejected has a verified oracle-only negative role with its original quarantine unchanged".into()});
        }
    }
    preferences.sort_by(|a, b| a.id.cmp(&b.id));
    preferences.dedup_by(|a, b| a.id == b.id);
    for teacher in teachers.values_mut() {
        let own = samples
            .iter()
            .filter(|s| s.teacher_id == teacher.id)
            .collect::<Vec<_>>();
        if !own.is_empty() {
            teacher.capabilities.complete_sequences = SignalAvailability {
                supported: true,
                available: true,
                reason: "Complete, nonempty generations passed every trusted container case".into(),
                artefacts: own.iter().map(|s| s.raw_generation.clone()).collect(),
            };
        }
        let pairs = preferences
            .iter()
            .filter(|p| {
                samples
                    .iter()
                    .any(|s| s.id == p.chosen_sample_id && s.teacher_id == teacher.id)
            })
            .collect::<Vec<_>>();
        if !pairs.is_empty() {
            teacher.capabilities.preferences = SignalAvailability {
                supported: true,
                available: true,
                reason: "Real same-task pass/oracle-negative pairs; security failures, runtime errors, malformed output and compiler-only feedback excluded".into(),
                artefacts: pairs
                    .iter()
                    .map(|p| p.rejected.observation.clone())
                    .collect(),
            };
        }
    }
    let teachers = teachers.into_values().collect::<Vec<_>>();
    let capabilities = teachers
        .iter()
        .map(|t| (t.id.clone(), t.capabilities.clone()))
        .collect();
    Ok(EvidenceBundle {schema_version:1,evidence_kind:"actual_local_teacher_training_evidence_not_provider_assessment".into(),created_unix_ms:crate::ledger::unix_millis(),teachers,samples,candidates,preferences,input_comparisons:inputs,content_records:vec![],rollout_records:vec![],capabilities,train_corpus_sha256:train_hash()?,limitations:vec!["48 tasks from 12 locally authored train source groups; no held-out or development data enters training.".into(),"Local model developers do not count as independent paid API provider businesses.".into(),"No learning, promotion, cash saving or teacher trace is inferred from this bundle alone.".into(),"Process, contextual revision, teacher-feedback trajectories, on-policy and self-distillation require additional genuine generation evidence.".into()]})
}

/// Candidate convenience flags cannot establish a training role. Re-read its
/// original observation, completion and grader artifact, already bound to the
/// frozen comparative sandbox by `verify_completed_comparison`.
pub(super) fn candidate_eligibility(
    root: &Path,
    candidate: &EvidenceCandidate,
    task: &Task,
) -> Result<ContentEligibility> {
    let observation: Observation =
        serde_json::from_slice(&fs::read(candidate.observation.verify(root)?)?)?;
    let generation = observation
        .generation
        .as_ref()
        .context("candidate has no original generation")?;
    ensure!(
        generation.completion == candidate.completion
            && json_equal(&observation.raw_generation, &candidate.raw_generation)?
            && observation
                .sandbox_report
                .as_ref()
                .is_some_and(|a| a.sha256 == candidate.sandbox_report.sha256
                    && a.path == candidate.sandbox_report.path),
        "candidate differs from its original model/grade evidence"
    );
    let report: SandboxReport =
        serde_json::from_slice(&fs::read(candidate.sandbox_report.verify(root)?)?)?;
    classify_content(
        root,
        &candidate.sandbox_report,
        &report.sandbox_config,
        task,
        &candidate.completion,
        generation.complete,
    )
}

fn normalised_code_hash(text: &str) -> String {
    tasks::hash(
        crate::evaluate::extract_code(text)
            .split_whitespace()
            .collect::<String>(),
    )
}
