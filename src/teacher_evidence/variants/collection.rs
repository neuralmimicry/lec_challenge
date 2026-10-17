//! Ordered asynchronous collection with bounded durable publication units.
use super::*;

/// Prepare and publish complete dependency units on bounded blocking workers.
/// Async cancellation may finish its accepted unit but starts no later unit.
pub(super) async fn collect_content_inner(
    options: ContentCollectOptions,
    publication: Publication,
) -> Result<EvidenceBundle> {
    let preparation = options.clone();
    let (mut bundle, root, mut sandbox, saved_preflight, engine) = publication
        .run(move || prepare_collection(&preparation))
        .await?;
    let preflight = if let Some(saved) = saved_preflight {
        saved
    } else {
        let observed = sandbox::preflight(&sandbox).await?;
        ensure!(
            observed.probes_passed,
            "content sandbox controls unavailable"
        );
        sandbox.expected_image_id = observed.image_id.clone();
        let preflight_path = options.output.join("sandbox-preflight.json");
        let capture_root = root.clone();
        publication
            .run(move || {
                artefact::write_json(&preflight_path, &observed)?;
                capture(&capture_root, &preflight_path)
            })
            .await?
    };
    let mut unavailable = Vec::new();
    for teacher in bundle.teachers.clone() {
        let mut loaded: Option<RecordedModel> = None;
        for task in train_tasks().into_iter().take(options.maximum_tasks) {
            for content in &options.content {
                let tag = serde_json::to_value(content)?
                    .as_str()
                    .context("content tag absent")?
                    .to_owned();
                let directory = options.output.join(&teacher.id).join(&task.id).join(tag);
                let preparation = RequestPreparation {
                    root: root.clone(),
                    directory: directory.clone(),
                    teacher: teacher.clone(),
                    task: task.clone(),
                    content: *content,
                    options: options.clone(),
                    sandbox: sandbox.clone(),
                    preflight: preflight.clone(),
                    engine: engine.clone(),
                };
                let (returned_bundle, prepared) =
                    publication.run(move || preparation.prepare(bundle)).await?;
                bundle = returned_bundle;
                let (plan, plan_artifact) = match prepared {
                    PreparedRequest::Resumed => continue,
                    PreparedRequest::Unavailable(reason) => {
                        unavailable.push(reason);
                        continue;
                    }
                    PreparedRequest::Pending(plan, artifact) => (plan, artifact),
                };
                let raw_path = directory.join("raw-generation.json");
                let generation_path = raw_path.clone();
                let generation_plan = plan_artifact.clone();
                let teacher_spec = teacher.clone();
                let runtime = options.runtime.clone();
                let request = plan.request;
                let context_tokens = plan.context_tokens;
                let previous_model = loaded.take();
                // The model and its durable raw observation share one worker.
                // A detached generation must not lose completed work before
                // fsync or release directory ownership while still running.
                let (returned_model, raw) = publication
                    .run(move || {
                        raw_or_generate(
                            &generation_path,
                            &generation_plan,
                            previous_model,
                            move |loaded| {
                                generate_content(
                                    loaded,
                                    teacher_spec,
                                    runtime,
                                    context_tokens,
                                    request,
                                )
                            },
                        )
                    })
                    .await?;
                loaded = returned_model;
                let (grade, grading_error) = if let Some(generation) = raw
                    .generation
                    .as_ref()
                    .filter(|g| g.complete && !g.completion.trim().is_empty())
                {
                    let link = directory.join("grade-link.json");
                    let saved_link = link.clone();
                    let saved = publication
                        .run(move || {
                            if saved_link.exists() {
                                Ok(Some(serde_json::from_slice::<Artefact>(&fs::read(
                                    &saved_link,
                                )?)?))
                            } else {
                                Ok(None)
                            }
                        })
                        .await?;
                    if let Some(saved) = saved {
                        (Some(saved), None)
                    } else {
                        match sandbox::grade(
                            &task,
                            &crate::evaluate::extract_code(&generation.completion),
                            &sandbox,
                        )
                        .await
                        {
                            Ok(report) => {
                                let report_path =
                                    sandbox.report_dir.join(format!("{}.json", report.run_id));
                                let capture_root = root.clone();
                                let artifact = publication
                                    .run(move || {
                                        let artifact = capture(&capture_root, &report_path)?;
                                        artefact::write_json(&link, &artifact)?;
                                        Ok(artifact)
                                    })
                                    .await?;
                                (Some(artifact), None)
                            }
                            Err(error) => (None, Some(format!("{error:#}"))),
                        }
                    }
                } else {
                    (None, None)
                };
                let record_root = root.clone();
                let record_content = *content;
                let (returned_bundle, success) = publication
                    .run(move || {
                        let success = if let Some(artifact) = &grade {
                            let report: SandboxReport =
                                serde_json::from_slice(&fs::read(artifact.verify(&record_root)?)?)?;
                            report.passed
                                && !report.quarantined
                                && report.trace_complete
                                && (record_content != ContentKind::Process
                                    || raw.generation.as_ref().is_some_and(|generation| {
                                        generation
                                            .completion
                                            .lines()
                                            .any(|line| line.trim_start().starts_with("//"))
                                    }))
                        } else {
                            false
                        };
                        let record = ContentRecord {
                            schema_version: 1,
                            evidence_kind: "actual_local_teacher_content_observation".into(),
                            plan: plan_artifact,
                            raw_generation: capture(&record_root, &raw_path)?,
                            sandbox_report: grade,
                            grading_error,
                            success,
                        };
                        let record_path = directory.join("record.json");
                        artefact::write_json(&record_path, &record)?;
                        append_verified(
                            &record_root,
                            &mut bundle,
                            &capture(&record_root, &record_path)?,
                        )?;
                        Ok((bundle, success))
                    })
                    .await?;
                bundle = returned_bundle;
                eprintln!(
                    "teacher content progress teacher={} task={} content={:?} accepted={} total_samples={}",
                    teacher.id,
                    task.id,
                    content,
                    success,
                    bundle.samples.len()
                );
            }
        }
        // Model destruction may release substantial native allocations.
        publication
            .run(move || {
                drop(loaded);
                Ok(())
            })
            .await?;
    }
    publication
        .run(move || {
            artefact::write_json(&options.output.join("unavailable.json"), &unavailable)?;
            artefact::write_json(&options.output.join("bundle.json"), &bundle)?;
            artefact::write_json(
                &options.output.join("manifest.json"),
                &serde_json::json!({"schema_version":1,"bundle":capture(&root,&options.output.join("bundle.json"))?,"train_corpus_sha256":bundle.train_corpus_sha256,"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"promotable":false}),
            )?;
            write_content_report(&root, &options, &bundle, &unavailable)?;
            Ok(bundle)
        })
        .await
}

type CollectionPreparation = (
    EvidenceBundle,
    PathBuf,
    SandboxConfig,
    Option<Artefact>,
    String,
);

fn prepare_collection(options: &ContentCollectOptions) -> Result<CollectionPreparation> {
    let bundle = load_verified(&options.bundle)?;
    ensure!(
        bundle.rollout_records.is_empty(),
        "content variants after student/self rollouts require ordered dependency replay, which is not implemented; no base-only fallback is permitted"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    let input_path = if options.bundle.is_dir() {
        options.bundle.join("bundle.json")
    } else {
        options.bundle.clone()
    };
    ensure!(
        input_path.canonicalize()?.parent() != Some(options.output.canonicalize()?.as_path()),
        "content output must not overwrite its verified input bundle"
    );
    let policy_path = options.output.join("content-policy.json");
    let mut policy = serde_json::json!({"schema_version":1,"input":capture(&root,&input_path)?,"content":options.content,"runtime":options.runtime,"max_tokens":options.max_tokens,"seed":options.seed,"maximum_tasks":options.maximum_tasks,"teachers":bundle.teachers.iter().map(|t|&t.id).collect::<Vec<_>>()});
    if let Some(context_tokens) = options.context_tokens {
        policy["context_tokens"] = context_tokens.into();
    }
    // Historical policy bytes retain their original serialised shape; their
    // generation plans already bind the executable that produced the output.
    if !policy_path.exists()
        || serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path)?)?
            .get("engine_binary_sha256")
            .is_some()
    {
        policy["engine_binary_sha256"] = artefact::file_hash(&std::env::current_exe()?)?.into();
    }
    if policy_path.exists() {
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(&policy_path)?)? == policy,
            "saved policy changed; use a new output directory for a fresh study"
        );
    } else {
        artefact::write_json(&policy_path, &policy)?;
    }
    let mut sandbox = SandboxConfig {
        report_dir: options.output.join("sandbox"),
        ..Default::default()
    };
    let preflight_path = options.output.join("sandbox-preflight.json");
    let preflight = if preflight_path.exists() {
        let saved: sandbox::PreflightReport = serde_json::from_slice(&fs::read(&preflight_path)?)?;
        ensure!(saved.probes_passed, "saved sandbox preflight failed");
        sandbox.expected_image_id = saved.image_id;
        Some(capture(&root, &preflight_path)?)
    } else {
        None
    };
    let engine = artefact::file_hash(&std::env::current_exe()?)?;
    Ok((bundle, root, sandbox, preflight, engine))
}

enum PreparedRequest {
    Resumed,
    Unavailable(serde_json::Value),
    Pending(Box<ContentPlan>, Artefact),
}

struct RequestPreparation {
    root: PathBuf,
    directory: PathBuf,
    teacher: TeacherEvidence,
    task: Task,
    content: ContentKind,
    options: ContentCollectOptions,
    sandbox: SandboxConfig,
    preflight: Artefact,
    engine: String,
}

impl RequestPreparation {
    fn prepare(self, mut bundle: EvidenceBundle) -> Result<(EvidenceBundle, PreparedRequest)> {
        fs::create_dir_all(&self.directory)?;
        let record_path = self.directory.join("record.json");
        if record_path.exists() {
            append_verified(&self.root, &mut bundle, &capture(&self.root, &record_path)?)?;
            return Ok((bundle, PreparedRequest::Resumed));
        }
        let (text, supporting_sample_ids, feedback_candidate, feedback_summary) =
            match prompt(&self.root, &bundle, &self.teacher, &self.task, self.content) {
                Ok(value) => value,
                Err(error) => {
                    return Ok((
                        bundle,
                        PreparedRequest::Unavailable(serde_json::json!({
                            "teacher":self.teacher.id,"task":self.task.id,"content":self.content,
                            "reason":error.to_string()
                        })),
                    ));
                }
            };
        let plan = ContentPlan {
            schema_version: 1,
            task_id: self.task.id,
            teacher_id: self.teacher.id,
            content: self.content,
            request: GenerationRequest {
                prompt: text,
                max_new_tokens: self.options.max_tokens,
                seed: self.options.seed,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: 120,
            },
            runtime: self.options.runtime,
            context_tokens: self.options.context_tokens,
            teacher_assets: self.teacher.assets,
            teacher_checkpoint: self.teacher.adapter_checkpoint,
            supporting_sample_ids,
            feedback_candidate,
            feedback_summary,
            sandbox: self.sandbox,
            preflight: self.preflight,
            engine_binary_sha256: self.engine,
        };
        let plan_path = self.directory.join("plan.json");
        let raw_path = self.directory.join("raw-generation.json");
        let plan = if plan_path.exists() {
            let old: ContentPlan = serde_json::from_slice(&fs::read(&plan_path)?)?;
            let mut expected = plan.clone();
            if raw_path.exists() {
                // Saved output keeps the executable identity that generated it.
                expected.engine_binary_sha256 = old.engine_binary_sha256.clone();
            }
            ensure!(
                json_equal(&old, &expected)?,
                "saved content request changed; refusing implicit resampling"
            );
            old
        } else {
            artefact::write_json(&plan_path, &plan)?;
            plan
        };
        let artifact = capture(&self.root, &plan_path)?;
        Ok((bundle, PreparedRequest::Pending(Box::new(plan), artifact)))
    }
}

struct GenerationOutcome {
    generation: Option<Generation>,
    generation_error: Option<String>,
    execution: Option<serde_json::Value>,
}

/// Reuse the exact saved observation, or finish generation and durable raw
/// publication before returning. Call only inside an admitted blocking unit.
fn raw_or_generate<M>(
    path: &Path,
    plan: &Artefact,
    model: M,
    generate: impl FnOnce(M) -> (M, GenerationOutcome),
) -> Result<(M, ContentRaw)> {
    if path.exists() {
        let saved: ContentRaw = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            json_equal(&saved.plan, plan)?,
            "saved raw content belongs to another request"
        );
        return Ok((model, saved));
    }
    let (model, outcome) = generate(model);
    let raw = ContentRaw {
        schema_version: 1,
        evidence_kind: "actual_local_teacher_content_generation".into(),
        plan: plan.clone(),
        generation: outcome.generation,
        generation_error: outcome.generation_error,
        execution: outcome.execution,
    };
    artefact::write_json(path, &raw)?;
    Ok((model, raw))
}

fn generate_content(
    loaded: Option<RecordedModel>,
    teacher: TeacherEvidence,
    runtime: RuntimeOptions,
    context_tokens: Option<usize>,
    request: GenerationRequest,
) -> (Option<RecordedModel>, GenerationOutcome) {
    let result = (|| {
        rollouts::validate_teacher_request(&teacher, context_tokens, &request)?;
        let mut model = match loaded {
            Some(model) => model,
            None => RecordedModel::teacher_with_context(&teacher, runtime, context_tokens)?,
        };
        let execution = model.execution_metadata();
        let result = model.generate(request);
        Ok::<_, anyhow::Error>((model, execution, result))
    })();
    let (model, generation, generation_error, execution) = match result {
        Ok((model, execution, result)) => match result {
            Ok(generation) => (Some(model), Some(generation), None, Some(execution)),
            Err(error) => (
                Some(model),
                None,
                Some(format!("{error:#}")),
                Some(execution),
            ),
        },
        Err(error) => (None, None, Some(format!("{error:#}")), None),
    };
    (
        model,
        GenerationOutcome {
            generation,
            generation_error,
            execution,
        },
    )
}

#[cfg(test)]
mod tests;
