//! Exploratory local-model comparisons on development or training sources only.
//!
//! These observations cannot authorise promotion or substitute local model
//! developers for the assessment's independent API provider businesses.
use crate::{
    artefact::{self, Artefact},
    evaluate, model,
    model::{Generation, GenerationRequest, ModelRuntime, RuntimeOptions},
    sandbox::{self, SandboxConfig, SandboxReport},
    tasks::{self, Split, Task},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};
use tokio::task::JoinSet;

mod records;
use records::{GenerationRecord, load_generation, publish_generation, verify_generation};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RegistryModel {
    pub id: String,
    pub model_id: String,
    pub revision: String,
    pub directory: PathBuf,
    pub architecture: String,
    pub role: String,
    pub licence: String,
    pub developer_id: String,
}
#[derive(Deserialize)]
pub struct Registry {
    pub schema_version: u32,
    pub models: Vec<RegistryModel>,
}
impl Registry {
    pub fn load(path: &Path) -> Result<Self> {
        let registry: Self = toml::from_str(&fs::read_to_string(path)?)?;
        ensure!(registry.schema_version == 1, "unsupported model registry");
        let mut ids = BTreeSet::new();
        for model in &registry.models {
            ensure!(
                !model.id.is_empty()
                    && model
                        .id
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                    && ids.insert(model.id.clone()),
                "model IDs must be unique safe file names"
            );
            ensure!(
                model.revision.len() == 40 && model.revision.bytes().all(|b| b.is_ascii_hexdigit()),
                "a comparative model requires an immutable Hub commit"
            );
            ensure!(
                !model.directory.is_absolute()
                    && model
                        .directory
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                "model directory must remain inside this workspace"
            );
        }
        ensure!(!registry.models.is_empty(), "empty comparative roster");
        Ok(registry)
    }
}

/// Deliberately has no held-out variant: exploratory comparison cannot open it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonSplit {
    Train,
    Development,
}
impl ComparisonSplit {
    fn corpus_split(self) -> Split {
        match self {
            Self::Train => Split::Train,
            Self::Development => Split::Development,
        }
    }
}

pub struct CompareOptions {
    pub registry: PathBuf,
    pub output: PathBuf,
    pub split: ComparisonSplit,
    pub runtime: RuntimeOptions,
    pub max_tokens: usize,
    pub seed: u64,
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FrozenModel {
    model: RegistryModel,
    assets: Artefact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Plan {
    schema_version: u32,
    evidence_kind: String,
    frozen_unix_ms: i64,
    registry: Artefact,
    models: Vec<FrozenModel>,
    tasks: Vec<Task>,
    split: ComparisonSplit,
    runtime: serde_json::Value,
    max_tokens: usize,
    seed: u64,
    temperature: f64,
    top_p: f64,
    sandbox: SandboxConfig,
    sandbox_preflight: Artefact,
    sandbox_policy_sha256: String,
    engine_binary_sha256: String,
    promotable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub model_id: String,
    pub task_id: String,
    pub source_id: String,
    pub specification_sha256: String,
    pub seed: u64,
    pub plan_sha256: String,
    pub request: GenerationRequest,
    pub generation: Option<Generation>,
    pub generation_error: Option<String>,
    /// The fsynced original generation precedes grading and is never regenerated on resume.
    pub raw_generation: Artefact,
    pub sandbox_report: Option<Artefact>,
    pub grading_error: Option<String>,
    pub success: bool,
}

/// A process lease prevents concurrent writers from replacing cached evidence.
struct Lease(fs::File);
impl Lease {
    fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        // SAFETY: flock receives a live descriptor and does not access Rust memory.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another comparative run owns this output directory"
        );
        Ok(Self(file))
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        // SAFETY: this lease retains the live descriptor until after unlocking.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

fn request(task: &Task, plan: &Plan) -> GenerationRequest {
    GenerationRequest {
        prompt: task.prompt(),
        max_new_tokens: plan.max_tokens,
        seed: plan.seed,
        temperature: plan.temperature,
        top_p: plan.top_p,
        timeout_seconds: 120,
    }
}

fn verify_observation(
    observation: &Observation,
    frozen: &FrozenModel,
    task: &Task,
    plan: &Plan,
    plan_hash: &str,
    root: &Path,
) -> Result<()> {
    let raw: GenerationRecord =
        serde_json::from_slice(&fs::read(observation.raw_generation.verify(root)?)?)?;
    verify_generation(&raw, frozen, task, plan, plan_hash, root)?;
    ensure!(
        observation.model_id == frozen.model.id
            && observation.task_id == task.id
            && observation.source_id == task.source_id
            && observation.specification_sha256 == task.specification_sha256
            && observation.seed == plan.seed
            && observation.plan_sha256 == plan_hash
            && serde_json::to_value(&observation.request)?
                == serde_json::to_value(request(task, plan))?
            && serde_json::to_value(&observation.generation)?
                == serde_json::to_value(&raw.generation)?
            && observation.generation_error == raw.generation_error,
        "cached comparative observation differs from its frozen request"
    );
    let mut success = false;
    if let Some(generation) = &observation.generation {
        ensure!(
            observation.generation_error.is_none(),
            "generation has both output and an error"
        );
        if let Some(artifact) = &observation.sandbox_report {
            ensure!(generation.complete, "incomplete generation was graded");
            let report: SandboxReport = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
            sandbox::verify_report(
                &report,
                &plan.sandbox,
                task,
                &evaluate::extract_code(&generation.completion),
            )?;
            success = report.passed && !report.quarantined && report.trace_complete;
        }
        ensure!(
            !generation.complete
                || generation.completion.trim().is_empty()
                || observation.sandbox_report.is_some()
                || observation.grading_error.is_some(),
            "complete generation lacks both grading evidence and an explicit grading error"
        );
    } else {
        ensure!(
            observation.generation_error.is_some(),
            "missing generation without an error"
        );
        ensure!(
            observation.sandbox_report.is_none(),
            "absent generation has a grader result"
        );
    }
    ensure!(
        success == observation.success,
        "comparative success does not match trusted evidence"
    );
    ensure!(
        !success || observation.grading_error.is_none(),
        "grading error cannot be success"
    );
    Ok(())
}

/// The original single-file base keeps its established identity. Other models
/// bind all safetensors shards, including their file names, into one identity.
pub fn comparative_checkpoint_id(assets: &model::AssetManifest) -> Result<String> {
    let shards = assets
        .files
        .iter()
        .filter(|(name, _)| name.ends_with(".safetensors"))
        .collect::<BTreeMap<_, _>>();
    ensure!(!shards.is_empty(), "model has no safetensors weights");
    if shards.len() == 1 {
        return Ok((*shards
            .values()
            .next()
            .context("missing single weight file")?)
        .clone());
    }
    Ok(artefact::hash(&serde_json::to_vec(&shards)?))
}

fn validate_plan(plan: &Plan) -> Result<()> {
    ensure!(
        plan.schema_version == 1 && !plan.promotable,
        "invalid exploratory plan"
    );
    ensure!(
        !plan.models.is_empty() && !plan.tasks.is_empty(),
        "empty comparative denominator"
    );
    let tasks = plan.tasks.iter().map(|t| &t.id).collect::<BTreeSet<_>>();
    let models = plan
        .models
        .iter()
        .map(|m| &m.model.id)
        .collect::<BTreeSet<_>>();
    ensure!(
        tasks.len() == plan.tasks.len() && models.len() == plan.models.len(),
        "duplicate planned model/task"
    );
    ensure!(
        plan.tasks
            .iter()
            .all(|t| t.split == plan.split.corpus_split()),
        "plan contains a task outside its exploratory split"
    );
    Ok(())
}

async fn grade_generation(
    raw: GenerationRecord,
    artifact: Artefact,
    task: Task,
    config: SandboxConfig,
    output: PathBuf,
    root: PathBuf,
) -> Result<Observation> {
    let mut observation = Observation {
        model_id: raw.model_id,
        task_id: raw.task_id,
        source_id: raw.source_id,
        specification_sha256: raw.specification_sha256,
        seed: raw.seed,
        plan_sha256: raw.plan_sha256,
        request: raw.request,
        generation: raw.generation,
        generation_error: raw.generation_error,
        raw_generation: artifact,
        sandbox_report: None,
        grading_error: None,
        success: false,
    };
    if let Some(generation) = &observation.generation
        && generation.complete
        && !generation.completion.trim().is_empty()
    {
        let code = evaluate::extract_code(&generation.completion);
        match sandbox::grade(&task, &code, &config).await {
            Ok(report) => {
                let passed = report.passed && !report.quarantined && report.trace_complete;
                let verified = tokio::task::spawn_blocking(move || {
                    sandbox::verify_report(&report, &config, &task, &code)?;
                    Artefact::capture(
                        &root,
                        config.report_dir.join(format!("{}.json", report.run_id)),
                    )
                })
                .await;
                match verified {
                    Ok(Ok(artifact)) => {
                        observation.success = passed;
                        observation.sandbox_report = Some(artifact);
                    }
                    Ok(Err(error)) => {
                        observation.grading_error =
                            Some(format!("trusted grading evidence rejected: {error:#}"))
                    }
                    Err(error) => {
                        observation.grading_error =
                            Some(format!("trusted grading verifier failed: {error}"))
                    }
                }
            }
            Err(error) => observation.grading_error = Some(format!("{error:#}")),
        }
    }
    tokio::task::spawn_blocking(move || {
        artefact::write_json(&output, &observation)?;
        Ok::<_, anyhow::Error>(observation)
    })
    .await?
}

async fn gpu_observation() -> serde_json::Value {
    let fields = [
        (
            "gpu",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        ),
        (
            "compute_processes",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
        ),
    ];
    let mut result = serde_json::Map::new();
    for (name, query) in fields {
        let output = tokio::process::Command::new("nvidia-smi")
            .args([query, "--format=csv,noheader,nounits"])
            .output()
            .await;
        let value = match output {
            Ok(output) if output.status.success() => {
                serde_json::json!({"observed_unix_ms":crate::ledger::unix_millis(),"csv":String::from_utf8_lossy(&output.stdout)})
            }
            Ok(output) => {
                serde_json::json!({"error":"GPU query failed","exit_code":output.status.code()})
            }
            Err(error) => serde_json::json!({"error":error.to_string()}),
        };
        result.insert(name.into(), value);
    }
    result.into()
}

fn summary(plan: &Plan, observations: &[Observation]) -> Result<serde_json::Value> {
    validate_plan(plan)?;
    let mut seen = BTreeSet::new();
    for observation in observations {
        ensure!(
            plan.models
                .iter()
                .any(|m| m.model.id == observation.model_id)
                && plan.tasks.iter().any(|t| t.id == observation.task_id)
                && seen.insert((&observation.model_id, &observation.task_id)),
            "duplicate or unplanned observation cannot establish coverage"
        );
    }
    let models = plan.models.iter().map(|frozen| {
        let own = observations.iter().filter(|o| o.model_id == frozen.model.id).collect::<Vec<_>>();
        let passed = own.iter().filter(|o| o.success).count();
        let failures = own.iter().filter(|o| o.generation_error.is_some() || o.grading_error.is_some()
            || o.generation.as_ref().is_none_or(|g| !g.complete || g.finish_reason != "stop" || g.completion.trim().is_empty())).count();
        serde_json::json!({"id":frozen.model.id,"model_id":frozen.model.model_id,"role":frozen.model.role,"passed":passed,"planned":plan.tasks.len(),"observed":own.len(),"pass_rate":passed as f64 / plan.tasks.len() as f64,"infrastructure_or_generation_errors":failures})
    }).collect::<Vec<_>>();
    let mut sources = BTreeSet::new();
    for task in &plan.tasks {
        sources.insert(task.source_id.clone());
    }
    Ok(
        serde_json::json!({"schema_version":1,"evidence_kind":"actual_local_exploratory_comparative_generations","split":plan.split,"models":models,"source_count":sources.len(),"planned_observations":plan.models.len()*plan.tasks.len(),"observed_observations":observations.len(),"coverage_complete":observations.len()==plan.models.len()*plan.tasks.len(),"promotable":false,"limitations":["Development/training comparison only: no sealed held-out tasks were evaluated.","These are open-weight models hosted locally, not three independent real provider businesses.","One decoding seed and a small, jointly authored corpus give descriptive results, not proof of learning or general model superiority.","Every missing, failed or truncated generation remains a failure in the planned denominator.","GPU memory snapshots are observations, not continuous peak-memory measurements; energy and cash costs are unmeasured."]}),
    )
}

pub async fn run(mut options: CompareOptions) -> Result<serde_json::Value> {
    ensure!(
        (1..=1024).contains(&options.max_tokens),
        "invalid comparative output cap"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    fs::create_dir_all(&options.output)?;
    ensure!(
        options.output.canonicalize()?.starts_with(&root),
        "comparison output must stay inside the workspace"
    );
    options.output = options
        .output
        .canonicalize()?
        .strip_prefix(&root)?
        .to_owned();
    options.registry = options
        .registry
        .canonicalize()?
        .strip_prefix(&root)?
        .to_owned();
    let _lease = Lease::acquire(&options.output.join(".process.lock"))?;
    let registry = Registry::load(&options.registry)?;
    let selected = registry
        .models
        .into_iter()
        .filter(|m| options.models.is_empty() || options.models.contains(&m.id))
        .collect::<Vec<_>>();
    ensure!(
        !selected.is_empty()
            && options
                .models
                .iter()
                .all(|id| selected.iter().any(|m| &m.id == id)),
        "unknown or empty comparative model selection"
    );
    let mut config = SandboxConfig {
        report_dir: options.output.join("sandbox"),
        ..Default::default()
    };
    eprintln!("compare: verifying effective sandbox controls before local model evaluation");
    let preflight = sandbox::preflight(&config).await?;
    ensure!(
        preflight.probes_passed,
        "comparative sandbox controls failed"
    );
    config.expected_image_id = preflight.image_id.clone();
    let preflight_path = options.output.join("sandbox-preflight.json");
    // Preserve the original successful probe artifact on a resumed plan.
    if !preflight_path.exists() {
        artefact::write_json(&preflight_path, &preflight)?;
    }
    let mut frozen_models = Vec::new();
    for entry in selected {
        let directory = entry.directory.clone();
        let baseline = entry.role == "student_baseline";
        let assets = tokio::task::spawn_blocking(move || {
            if baseline {
                model::validate_assets(&directory)
            } else {
                crate::hub::validate_downloaded_assets(&directory)
            }
        })
        .await??;
        ensure!(
            assets.model_id == entry.model_id && assets.revision == entry.revision,
            "registry differs from downloaded model provenance"
        );
        frozen_models.push(FrozenModel {
            assets: Artefact::capture(&root, entry.directory.join("assets.json"))?,
            model: entry,
        });
    }
    let plan_path = options.output.join("plan.json");
    let sandbox_policy_sha256 = sandbox::policy_sha256(&config)?;
    let mut plan = Plan {
        schema_version: 1,
        evidence_kind: "preregistered_exploratory_local_comparison".into(),
        frozen_unix_ms: crate::ledger::unix_millis(),
        registry: Artefact::capture(&root, options.registry)?,
        models: frozen_models,
        tasks: tasks::corpus()
            .into_iter()
            .filter(|t| t.split == options.split.corpus_split())
            .collect(),
        split: options.split,
        runtime: serde_json::to_value(&options.runtime)?,
        max_tokens: options.max_tokens,
        seed: options.seed,
        temperature: 0.2,
        top_p: 0.95,
        sandbox: config,
        sandbox_preflight: Artefact::capture(
            &root,
            preflight_path
                .strip_prefix(&root)
                .unwrap_or(&preflight_path),
        )?,
        sandbox_policy_sha256,
        engine_binary_sha256: artefact::file_hash(&std::env::current_exe()?)?,
        promotable: false,
    };
    validate_plan(&plan)?;
    if plan_path.exists() {
        let existing: Plan = serde_json::from_slice(&fs::read(&plan_path)?)?;
        plan.frozen_unix_ms = existing.frozen_unix_ms;
        ensure!(
            serde_json::to_value(&existing)? == serde_json::to_value(&plan)?,
            "comparative plan is immutable; changed inputs require a new output directory"
        );
        plan = existing;
    } else {
        artefact::write_json(&plan_path, &plan)?;
    }
    let plan_hash = artefact::file_hash(&plan_path)?;
    let mut observations = Vec::new();
    let execution_path = options.output.join("execution.json");
    let mut execution: Vec<serde_json::Value> = if execution_path.exists() {
        serde_json::from_slice(&fs::read(&execution_path)?)?
    } else {
        Vec::new()
    };
    for frozen in &plan.models {
        let directory = options.output.join(&frozen.model.id);
        fs::create_dir_all(&directory)?;
        let mut pending = Vec::new();
        for task in &plan.tasks {
            let path = directory.join(format!("{}.json", task.id));
            if path.exists() {
                let observation: Observation = serde_json::from_slice(&fs::read(path)?)?;
                verify_observation(&observation, frozen, task, &plan, &plan_hash, &root)?;
                observations.push(observation);
            } else {
                let raw_path = directory.join(format!("{}.generation.json", task.id));
                let cached = load_generation(&raw_path, frozen, task, &plan, &plan_hash, &root)?;
                pending.push((task.clone(), raw_path, cached));
            }
        }
        if pending.is_empty() {
            continue;
        }
        // Completed raw inference is replayed even if the process died while grading.
        // When every generation exists, resumption never opens model tensors or CUDA.
        let needs_inference = pending.iter().any(|(_, _, raw)| raw.is_none());
        let mut runtime = None;
        let mut loading_error = None;
        let mut actual_execution = None;
        if needs_inference {
            let before = gpu_observation().await;
            let model_dir = frozen.model.directory.clone();
            let baseline = frozen.model.role == "student_baseline";
            let runtime_options = options.runtime.clone();
            eprintln!(
                "compare: loading {} on requested device",
                frozen.model.model_id
            );
            let loaded = tokio::task::spawn_blocking(move || {
                if baseline {
                    ModelRuntime::load_with_options(&model_dir, None, runtime_options)
                } else {
                    ModelRuntime::load_comparative_with_options(&model_dir, runtime_options)
                }
            })
            .await;
            match loaded {
                Ok(Ok(model)) => {
                    actual_execution = Some(model.execution_metadata());
                    runtime = Some(model);
                }
                Ok(Err(error)) => loading_error = Some(format!("model loading failed: {error:#}")),
                Err(error) => loading_error = Some(format!("model loading worker failed: {error}")),
            }
            let after_load = gpu_observation().await;
            execution.push(serde_json::json!({"event":"model_loading","model":frozen.model.id,"plan_sha256":plan_hash,"requested_runtime":options.runtime,"actual_execution":actual_execution,"loading_error":loading_error,"before_loading":before,"after_loading":after_load}));
            artefact::write_json(&execution_path, &execution)?;
        }
        let mut grades = JoinSet::new();
        for (task, raw_path, cached) in pending {
            if grades.len() >= 2 {
                observations.push(
                    grades
                        .join_next()
                        .await
                        .context("grading worker missing")???,
                );
            }
            let (raw, artifact) = if let Some(cached) = cached {
                eprintln!(
                    "compare: reusing original generation model={} task={}",
                    frozen.model.id, task.id
                );
                cached
            } else {
                eprintln!(
                    "compare: model={} task={} seed={}",
                    frozen.model.id, task.id, plan.seed
                );
                let result = if let Some(error) = &loading_error {
                    Err(anyhow::anyhow!(error.clone()))
                } else {
                    let mut worker = runtime
                        .take()
                        .context("model loading produced no runtime or error")?;
                    let generation_request = request(&task, &plan);
                    let generated = tokio::task::spawn_blocking(move || {
                        let result = worker.generate(generation_request);
                        (worker, result)
                    })
                    .await?;
                    runtime = Some(generated.0);
                    generated.1
                };
                let raw = GenerationRecord::new(
                    frozen,
                    &task,
                    &plan,
                    &plan_hash,
                    result,
                    actual_execution.clone(),
                );
                // Publication is fsynced before any grading worker starts. A crash
                // after this point can repeat grading, but never the original inference.
                let raw_copy = raw.clone();
                let task_copy = task.clone();
                let frozen_copy = frozen.clone();
                let plan_copy = plan.clone();
                let hash_copy = plan_hash.clone();
                let root_copy = root.clone();
                let artifact = tokio::task::spawn_blocking(move || {
                    publish_generation(
                        &raw_path,
                        &raw_copy,
                        &frozen_copy,
                        &task_copy,
                        &plan_copy,
                        &hash_copy,
                        &root_copy,
                    )
                })
                .await??;
                (raw, artifact)
            };
            grades.spawn(grade_generation(
                raw,
                artifact,
                task.clone(),
                plan.sandbox.clone(),
                directory.join(format!("{}.json", task.id)),
                root.clone(),
            ));
        }
        while let Some(result) = grades.join_next().await {
            observations.push(result??);
        }
        if needs_inference {
            let after_generation = gpu_observation().await;
            execution.push(serde_json::json!({"event":"model_generation_complete","model":frozen.model.id,"plan_sha256":plan_hash,"actual_execution":actual_execution,"after_generation":after_generation}));
            artefact::write_json(&execution_path, &execution)?;
        }
        let had_runtime = runtime.is_some();
        tokio::task::spawn_blocking(move || -> Result<()> {
            if let Some(runtime) = runtime {
                runtime.model.device().synchronize()?;
                drop(runtime);
            }
            Ok(())
        })
        .await??;
        if needs_inference {
            let after_release = gpu_observation().await;
            execution.push(serde_json::json!({"event":"model_released","model":frozen.model.id,"plan_sha256":plan_hash,"synchronized_before_release":had_runtime,"after_release":after_release}));
            artefact::write_json(&execution_path, &execution)?;
        }
        let report = summary(&plan, &observations)?;
        artefact::write_json(&options.output.join("report.json"), &report)?;
    }
    observations.sort_by(|a, b| (&a.model_id, &a.task_id).cmp(&(&b.model_id, &b.task_id)));
    for observation in &observations {
        let frozen = plan
            .models
            .iter()
            .find(|m| m.model.id == observation.model_id)
            .context("unknown observed model")?;
        let task = plan
            .tasks
            .iter()
            .find(|t| t.id == observation.task_id)
            .context("unknown observed task")?;
        verify_observation(observation, frozen, task, &plan, &plan_hash, &root)?;
    }
    let report = summary(&plan, &observations)?;
    artefact::write_json(&options.output.join("report.json"), &report)?;
    let mut markdown = String::from(
        "# Local comparative coding evaluation\n\nExploratory development/training observations; no promotion or learning claim.\n\n| Model | Passed / planned | Pass rate | Generation/infrastructure errors |\n| --- | ---: | ---: | ---: |\n",
    );
    for row in report["models"]
        .as_array()
        .context("comparison summary missing models")?
    {
        markdown.push_str(&format!(
            "| {} | {} / {} | {:.2}% | {} |\n",
            row["model_id"].as_str().unwrap_or_default(),
            row["passed"],
            row["planned"],
            100. * row["pass_rate"].as_f64().unwrap_or_default(),
            row["infrastructure_or_generation_errors"]
        ));
    }
    markdown.push_str("\nAll failures remain in the planned denominator. These locally hosted weights do not fulfil the three-real-provider requirement. No sealed held-out sources were evaluated. GPU observations and immutable per-task code/grader artifacts accompany this report.\n");
    artefact::atomic_write(&options.output.join("report.md"), markdown.as_bytes())?;
    Ok(report)
}

#[cfg(test)]
mod tests;
