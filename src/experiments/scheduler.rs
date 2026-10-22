//! Bounded subprocess execution; the controller never trains on an async worker.
use super::*;
use std::{
    collections::VecDeque,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::{process::Command, task::JoinSet};

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub plan: PathBuf,
    pub output: PathBuf,
    pub fidelity: Fidelity,
    /// Full fidelity requires an explicitly frozen development-selected subset.
    pub selected_arms: Vec<String>,
    pub selection_evidence: Option<Artefact>,
    pub worker_executable: PathBuf,
    pub worker_arguments: Vec<String>,
    pub offline: bool,
    pub maximum_run_seconds: u64,
}

#[derive(Clone, Debug)]
pub struct CompletedResult {
    pub arm: ArmSpec,
    pub request: WorkerRequest,
    pub outcome: WorkerOutcome,
    pub outcome_artefact: Artefact,
    pub worker_wall_seconds: f64,
}

/// Reconstruct completed arms for development-only ranking. Aggregated state
/// cannot replace the pinned request, actual tensors or trusted evaluation.
pub fn completed_results(output: &Path, plan_path: &Path) -> Result<Vec<CompletedResult>> {
    let state: State = serde_json::from_slice(&fs::read(output.join("state.json"))?)?;
    let plan = load_plan(plan_path)?;
    let root = std::env::current_dir()?;
    ensure!(
        state.schema_version == 1 && state.plan_sha256 == artefact::file_hash(plan_path)?,
        "completed study state differs from its frozen plan"
    );
    state.engine.verify()?;
    let mut completed = Vec::new();
    for (key, trial) in state
        .trials
        .iter()
        .filter(|(_, t)| t.status == "completed_real_training")
    {
        ensure!(
            key == &trial.id
                && state.selected_arms.contains(key)
                && trial.elapsed_seconds.is_finite()
                && trial.elapsed_seconds >= 0.,
            "completed trial identity/timing is invalid"
        );
        let request = trial
            .request
            .as_ref()
            .context("completed trial request missing")?;
        let request_path = request.verify(&root)?;
        let request = validate_worker_request(&request_path)?;
        ensure!(
            request.arm.id == trial.id
                && request.study_plan.sha256 == state.plan_sha256
                && plan
                    .arms
                    .iter()
                    .any(|a| a.id == request.arm.id && a.applicable),
            "completed request is not a configured applicable arm"
        );
        let artefact = trial
            .outcome
            .as_ref()
            .context("completed trial outcome missing")?;
        let path = artefact.verify(&root)?;
        let request_sha = artefact::file_hash(&request_path)?;
        let outcome = validate_outcome(&path, &request, &request_sha)?;
        completed.push(CompletedResult {
            arm: request.arm.clone(),
            request,
            outcome,
            outcome_artefact: artefact.clone(),
            worker_wall_seconds: trial.elapsed_seconds,
        });
    }
    Ok(completed)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Engine {
    executable: PathBuf,
    sha256: String,
    bytes: u64,
    arguments: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cpu_build: Option<crate::engine::cpu::CpuBuildReceipt>,
}
impl Engine {
    fn capture(options: &RunOptions) -> Result<Self> {
        let executable = options.worker_executable.canonicalize()?;
        ensure!(
            fs::metadata(&executable)?.is_file(),
            "worker executable is not a regular file"
        );
        let sha256 = artefact::file_hash(&executable)?;
        let bytes = fs::metadata(&executable)?.len();
        let pinned = crate::engine::verified_snapshot(&executable, &sha256, bytes)?;
        let engine = Self {
            sha256,
            bytes,
            executable,
            arguments: options.worker_arguments.clone(),
            cpu_build: pinned.cpu_build,
        };
        engine.verify()?;
        Ok(engine)
    }
    fn verify(&self) -> Result<()> {
        let pinned = crate::engine::verified_snapshot(&self.executable, &self.sha256, self.bytes)?;
        ensure!(
            pinned.cpu_build == self.cpu_build,
            "frozen worker CPU build requirements changed"
        );
        Ok(())
    }
}

type Lease = crate::engine::RunLease;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Trial {
    id: String,
    status: String,
    attempts: u64,
    request: Option<Artefact>,
    outcome: Option<Artefact>,
    reason: Option<String>,
    exit_code: Option<i32>,
    elapsed_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct State {
    schema_version: u32,
    plan_sha256: String,
    engine: Engine,
    fidelity: Fidelity,
    selected_arms: Vec<String>,
    selection_evidence: Option<Artefact>,
    offline: bool,
    trials: BTreeMap<String, Trial>,
    reserved_training_steps: u64,
    worker_wall_seconds: f64,
    events: Vec<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub(super) struct Resources {
    capacity: ResourceCapacity,
    free_cpu: u32,
    free_ram: u64,
    free_vram: u64,
    active: usize,
}
impl Resources {
    pub(super) fn new(capacity: ResourceCapacity) -> Self {
        Self {
            free_cpu: capacity.cpu_threads,
            free_ram: capacity.ram_bytes,
            free_vram: capacity.vram_bytes,
            active: 0,
            capacity,
        }
    }
    pub(super) fn reserve(&mut self, request: &ResourceRequest) -> bool {
        if self.active >= self.capacity.maximum_parallel_jobs
            || request.cpu_threads > self.free_cpu
            || request.ram_bytes > self.free_ram
            || request.vram_bytes > self.free_vram
        {
            return false;
        }
        // All dimensions change in one controller operation; separate semaphore
        // acquisitions could deadlock while holding different partial resources.
        self.free_cpu -= request.cpu_threads;
        self.free_ram -= request.ram_bytes;
        self.free_vram -= request.vram_bytes;
        self.active += 1;
        true
    }
    pub(super) fn release(&mut self, request: &ResourceRequest) -> Result<()> {
        self.free_cpu = self
            .free_cpu
            .checked_add(request.cpu_threads)
            .context("CPU accounting overflow")?;
        self.free_ram = self
            .free_ram
            .checked_add(request.ram_bytes)
            .context("RAM accounting overflow")?;
        self.free_vram = self
            .free_vram
            .checked_add(request.vram_bytes)
            .context("VRAM accounting overflow")?;
        self.active = self
            .active
            .checked_sub(1)
            .context("released an inactive study worker")?;
        ensure!(
            self.free_cpu <= self.capacity.cpu_threads
                && self.free_ram <= self.capacity.ram_bytes
                && self.free_vram <= self.capacity.vram_bytes,
            "resource release exceeds measured capacity"
        );
        Ok(())
    }
}

pub(super) fn validate_outcome(
    path: &Path,
    request: &WorkerRequest,
    request_sha: &str,
) -> Result<WorkerOutcome> {
    let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(path)?)?;
    let root = std::env::current_dir()?;
    ensure!(
        outcome.schema_version == 1
            && outcome.arm_id == request.arm.id
            && outcome.request_sha256 == request_sha
            && outcome.checkpoint_changed
            && outcome.checkpoint_reloaded
            && outcome.completed_steps == request.training.steps
            && outcome.finite_losses.len() == outcome.completed_steps
            && outcome.finite_gradient_norms.len() == outcome.completed_steps
            && outcome.finite_losses.iter().all(|v| v.is_finite())
            && outcome
                .finite_gradient_norms
                .iter()
                .all(|v| v.is_finite() && *v >= 0.)
            && outcome.finite_gradient_norms.iter().any(|v| *v > 0.)
            && !outcome.checkpoint_artifacts.is_empty()
            && outcome.development_evaluation.is_some(),
        "worker success lacks matching real numerical updates/export/reload evidence"
    );
    let training_path = outcome.training_artifact.verify(&root)?;
    let training: serde_json::Value = serde_json::from_slice(&fs::read(training_path)?)?;
    ensure!(
        training["arm_id"] == request.arm.id
            && training["changed_adapter"] == true
            && training["reloaded"] == true
            && training["steps"] == request.training.steps
            && training["production_eligible"] == false,
        "linked training artefact disagrees with worker completion"
    );
    let tensors = outcome
        .checkpoint_artifacts
        .iter()
        .filter(|a| {
            a.path.extension().is_some_and(|e| {
                e == "safetensors"
                    || (request.arm.student == StudentArchitecture::TernaryBitnet && e == "gguf")
            })
        })
        .collect::<Vec<_>>();
    ensure!(
        !tensors.is_empty(),
        "a JSON manifest cannot substitute for a real exported adapter tensor"
    );
    ensure!(
        training["evidence_bundle_sha256"] == request.inputs.evidence_bundle.sha256
            && training["losses"] == serde_json::to_value(&outcome.finite_losses)?
            && training["gradient_norms"] == serde_json::to_value(&outcome.finite_gradient_norms)?
            && tensors
                .iter()
                .any(|a| training["adapter_sha256"] == a.sha256),
        "training source, numerical evidence or exported tensor hash differs from the outcome envelope"
    );
    for artifact in outcome
        .checkpoint_artifacts
        .iter()
        .chain(outcome.train_evaluation.iter())
        .chain(outcome.development_evaluation.iter())
    {
        artifact.verify(&root)?;
    }
    if request.arm.student == StudentArchitecture::ConventionalQwenLora {
        ensure!(
            training["base_sha256"] == crate::model::MODEL_SHA256,
            "conventional arm changed its fixed student base"
        );
        for artifact in tensors {
            let loaded =
                candle_core::safetensors::load(artifact.verify(&root)?, &candle_core::Device::Cpu)?;
            ensure!(
                !loaded.is_empty(),
                "exported checkpoint contains no real tensors"
            );
        }
    }
    for (evaluation, split) in [
        (&outcome.train_evaluation, "train"),
        (&outcome.development_evaluation, "development"),
    ] {
        if let Some(artifact) = evaluation {
            let path = artifact.verify(&root)?;
            ensure!(
                path.file_name().is_some_and(|n| n == "report.json"),
                "study evaluation artefact must name its independently verifiable report"
            );
            super::evaluation_failure::verify(&path, split, &training["adapter_sha256"])?;
        }
    }
    Ok(outcome)
}

async fn save(path: &Path, state: &State) -> Result<()> {
    let path = path.to_owned();
    let state = state.clone();
    tokio::task::spawn_blocking(move || artefact::write_json(&path, &state)).await?
}

struct Completion {
    id: String,
    elapsed: f64,
    exit: Option<i32>,
    result: Result<Artefact>,
    resources: ResourceRequest,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "study worker was interrupted or reached its time budget; validated optimiser progress is retained"
)]
struct WorkerInterrupted;

/// Each worker owns an isolated Unix process group. The evaluation reload and
/// native inference server inherit it, so timeout or future cancellation must
/// terminate the group rather than only its leading optimiser process.
#[cfg(unix)]
struct ProcessGroupGuard(Option<i32>);
#[cfg(unix)]
impl ProcessGroupGuard {
    fn new(pid: u32) -> Result<Self> {
        let id = i32::try_from(pid).context("worker PID is outside process-group range")?;
        ensure!(id > 0, "worker process-group ID must be positive");
        Ok(Self(Some(id)))
    }
    fn terminate(&mut self) {
        if let Some(id) = self.0.take() {
            // SAFETY: a positive child PID was made its own process-group ID
            // before exec; the negative argument addresses only that group.
            unsafe { libc::kill(-id, libc::SIGKILL) };
        }
    }
}
#[cfg(unix)]
impl Drop for ProcessGroupGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[cfg(all(test, unix))]
mod process_tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    #[tokio::test]
    async fn cancelled_worker_group_terminates_every_owned_member() {
        // Trusted coreutils sleepers model multiple members of a worker group;
        // no generated candidate or shell is executed on the host.
        let mut leader = Command::new("/bin/sleep")
            .arg("60")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let id = leader.id().unwrap();
        let group = ProcessGroupGuard::new(id).unwrap();
        let mut member = Command::new("/bin/sleep")
            .arg("60")
            .process_group(i32::try_from(id).unwrap())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        drop(group);
        for child in [&mut leader, &mut member] {
            let status = tokio::time::timeout(Duration::from_secs(3), child.wait())
                .await
                .expect("cancelled process-group member survived")
                .unwrap();
            assert_eq!(status.signal(), Some(libc::SIGKILL));
        }
    }
}

async fn execute(
    engine: Engine,
    request: WorkerRequest,
    request_path: PathBuf,
    request_sha: String,
    deadline: Instant,
    invocation_offline: bool,
) -> Completion {
    let id = request.arm.id.clone();
    let resources = request.resources.clone();
    let started = Instant::now();
    let mut exit = None;
    let result = async {
        // Rehash before every launch. Cargo rebuilding another path cannot change
        // this worker's identity unnoticed between two training arms.
        let copied = engine.clone();
        tokio::task::spawn_blocking(move || copied.verify()).await??;
        let stdout = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(request.output.join("stdout.log"))?;
        let stderr = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(request.output.join("stderr.log"))?;
        let mut command = Command::new(&engine.executable);
        let model_threads = if matches!(
            request.inputs.student_runtime.device,
            crate::model::ExecutionDevice::Cpu
        ) {
            request.resources.cpu_threads.saturating_sub(1).max(1)
        } else {
            request.resources.cpu_threads
        };
        command
            .args(&engine.arguments)
            .arg("--request")
            .arg(&request_path)
            .arg("--output")
            .arg(&request.output)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .kill_on_drop(true)
            .env("RAYON_NUM_THREADS", model_threads.to_string())
            .env("OMP_NUM_THREADS", model_threads.to_string())
            .env("MKL_NUM_THREADS", model_threads.to_string())
            .env(
                "LEC_STUDY_CPU_THREADS",
                request.resources.cpu_threads.to_string(),
            )
            .env(
                "LEC_STUDY_OFFLINE",
                if request.offline || invocation_offline {
                    "1"
                } else {
                    "0"
                },
            );
        for key in [
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
        ] {
            command.env_remove(key);
        }
        #[cfg(unix)]
        command.process_group(0);
        #[cfg(target_os = "linux")]
        unsafe {
            let expected_parent = std::process::id() as libc::pid_t;
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != expected_parent {
                    return Err(std::io::Error::other(
                        "study controller exited before worker start",
                    ));
                }
                Ok(())
            });
        }
        let mut child = command
            .spawn()
            .context("spawn pinned native study worker")?;
        #[cfg(unix)]
        let mut process_group =
            ProcessGroupGuard::new(child.id().context("study worker has no process-group ID")?)?;
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(request.training.timeout_seconds));
        let status = match tokio::time::timeout(remaining, child.wait()).await {
            Ok(result) => result?,
            Err(_) => {
                #[cfg(unix)]
                process_group.terminate();
                let _ = child.kill().await;
                let _ = child.wait().await;
                return Err(anyhow::Error::new(WorkerInterrupted));
            }
        };
        // The group must be released immediately after reaping the leader,
        // before expensive evidence checks or any possible PID reuse.
        #[cfg(unix)]
        process_group.terminate();
        exit = status.code();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            if status
                .signal()
                .is_some_and(|s| [libc::SIGINT, libc::SIGTERM, libc::SIGKILL].contains(&s))
            {
                return Err(anyhow::Error::new(WorkerInterrupted));
            }
        }
        if exit.is_some_and(|c| [130, 137, 143].contains(&c)) {
            return Err(anyhow::Error::new(WorkerInterrupted));
        }
        ensure!(
            status.success(),
            "study worker failed with exit code {:?}; see retained stderr.log",
            exit
        );
        let outcome_path = request.output.join("outcome.json");
        let checked = request.clone();
        let hash = request_sha.clone();
        let path = outcome_path.clone();
        tokio::task::spawn_blocking(move || validate_outcome(&path, &checked, &hash)).await??;
        Artefact::capture(&std::env::current_dir()?, outcome_path)
    }
    .await;
    Completion {
        id,
        elapsed: started.elapsed().as_secs_f64(),
        exit,
        result,
        resources,
    }
}

fn report(plan: &StudyPlan, state: &State, status: &str) -> serde_json::Value {
    let arms=plan.arms.iter().map(|arm| {
        let trial=state.trials.get(&arm.id);
        let fidelity_reasons=fidelity_inapplicable_reasons(arm,&training_for(plan,arm,state.fidelity));
        serde_json::json!({"id":arm.id,"status":if !arm.applicable {"not_applicable"} else if !fidelity_reasons.is_empty() {"not_applicable_at_this_fidelity"} else {trial.map_or("queued",|t|t.status.as_str())},
            "inapplicable_reasons":arm.inapplicable_reasons,"fidelity_inapplicable_reasons":fidelity_reasons,"trial":trial})
    }).collect::<Vec<_>>();
    let completed = state
        .trials
        .values()
        .filter(|t| t.status == "completed_real_training")
        .count();
    let failed = state
        .trials
        .values()
        .filter(|t| t.status == "failed")
        .count();
    let applicable = plan
        .arms
        .iter()
        .filter(|a| {
            a.applicable
                && fidelity_inapplicable_reasons(a, &training_for(plan, a, state.fidelity))
                    .is_empty()
        })
        .count();
    let rejected = state
        .trials
        .values()
        .filter(|t| t.status == "not_applicable_actual_selection")
        .count();
    serde_json::json!({"schema_version":1,"evidence_kind":"bounded_local_distillation_study_execution","status":status,
        "selected_profile_succeeded":!state.selected_arms.is_empty() && completed==state.selected_arms.len() && failed==0,
        "fidelity":state.fidelity,"catalogue_arms":plan.arms.len(),"applicable_arms":applicable,"completed_arms":completed,
        "failed_arms":failed,"inapplicable_actual_selection_arms":rejected,"queued_applicable_arms":applicable.saturating_sub(completed+failed+rejected),
        "configured_catalogue_exhausted":completed+failed+rejected==applicable,"arms":arms,"quick_coverage_gaps":plan.quick_coverage_gaps,
        "reserved_training_steps":state.reserved_training_steps,"summed_worker_wall_seconds":state.worker_wall_seconds,
        "monetary_cost":"No paid provider calls; local energy/hardware cash cost remains unmeasured unless the worker supplies actual measurement.",
        "held_out_used":false,"production_eligible":false,
        "limitations":["Quick fidelity verifies executable method coverage; it is not proof of learning.","Queued arms have not run and cannot support a global optimum or exhaustive empirical comparison.","The controller bounds its own concurrent worker reservations; unrelated processes can still change machine availability.","Only one independently selected winner may enter the separately preregistered sealed promotion gate."]})
}

/// Run actual training workers; no HTTP is performed by this controller.
pub async fn run(options: RunOptions) -> Result<serde_json::Value> {
    ensure!(
        (1..=604_800).contains(&options.maximum_run_seconds),
        "invalid scheduler wall-time budget"
    );
    let plan = load_plan(&options.plan)?;
    let root = std::env::current_dir()?;
    plan.inputs.evidence_bundle.verify(&root)?;
    plan.inputs.student_base.verify(&root)?;
    fs::create_dir_all(&options.output)?;
    let _output = Lease::acquire(&options.output.join(".scheduler.lock"))?;
    fs::create_dir_all("reports")?;
    let _pool = Lease::acquire(Path::new("reports/.distillation-study-resources.lock"))?;
    let state_path = options.output.join("state.json");
    let prior = if state_path.exists() {
        Some(serde_json::from_slice::<State>(&fs::read(&state_path)?)?)
    } else {
        None
    };
    let engine = if let Some(prior) = &prior {
        ensure!(
            prior.engine.arguments == options.worker_arguments,
            "resumed worker arguments changed; use an explicit fresh output"
        );
        prior.engine.verify()?;
        if options.worker_executable.canonicalize().ok().as_ref() != Some(&prior.engine.executable)
        {
            eprintln!(
                "study: resuming the original checksum-verified worker {}",
                prior.engine.executable.display()
            );
        }
        prior.engine.clone()
    } else {
        Engine::capture(&options)?
    };
    let fidelity = match options.fidelity {
        Fidelity::Quick => &plan.quick,
        Fidelity::Full => &plan.full,
    };
    let selected = if options.fidelity == Fidelity::Quick {
        ensure!(
            options.selected_arms.is_empty() && options.selection_evidence.is_none(),
            "quick coverage selection is frozen in the plan"
        );
        plan.quick_arms.clone()
    } else {
        ensure!(
            !options.selected_arms.is_empty() && options.selection_evidence.is_some(),
            "full fidelity requires frozen development selection evidence"
        );
        let selection_path = options
            .selection_evidence
            .as_ref()
            .context("selection evidence missing")?
            .verify(&root)?;
        crate::study::verify_selection(&selection_path, &options.plan, &options.selected_arms)?;
        options.selected_arms.clone()
    };
    ensure!(
        selected.len() <= fidelity.maximum_trials
            && selected.iter().collect::<BTreeSet<_>>().len() == selected.len()
            && selected
                .iter()
                .all(|id| plan.arms.iter().any(|a| &a.id == id
                    && a.applicable
                    && fidelity_inapplicable_reasons(
                        a,
                        &training_for(&plan, a, options.fidelity)
                    )
                    .is_empty())),
        "selected arms are duplicate, unavailable or exceed the trial budget"
    );
    let plan_sha = artefact::file_hash(&options.plan)?;
    let mut state = if let Some(mut prior) = prior {
        ensure!(
            prior.schema_version == 1
                && prior.plan_sha256 == plan_sha
                && prior.fidelity == options.fidelity
                && prior.selected_arms == selected
                && serde_json::to_value(&prior.engine)? == serde_json::to_value(&engine)?
                && serde_json::to_value(&prior.selection_evidence)?
                    == serde_json::to_value(&options.selection_evidence)?,
            "study resumption differs from its frozen plan/worker/profile/selection"
        );
        for trial in prior.trials.values_mut() {
            if trial.status == "running" {
                let cached = options
                    .output
                    .join("arms")
                    .join(&trial.id)
                    .join("outcome.json");
                if cached.exists() {
                    let request = trial
                        .request
                        .as_ref()
                        .context("interrupted trial request missing")?;
                    let checked: WorkerRequest =
                        serde_json::from_slice(&fs::read(request.verify(&root)?)?)?;
                    validate_outcome(&cached, &checked, &request.sha256)?;
                    trial.outcome = Some(Artefact::capture(&root, cached)?);
                    trial.status = "completed_real_training".into();
                } else {
                    trial.status = "queued_after_controller_restart".into();
                }
            }
            if trial.status == "completed_real_training" {
                let request = trial
                    .request
                    .as_ref()
                    .context("completed trial request missing")?;
                let outcome = trial
                    .outcome
                    .as_ref()
                    .context("completed trial outcome missing")?;
                let request_path = request.verify(&root)?;
                let outcome_path = outcome.verify(&root)?;
                let checked: WorkerRequest = serde_json::from_slice(&fs::read(request_path)?)?;
                validate_outcome(&outcome_path, &checked, &request.sha256)?;
            }
        }
        prior.offline = options.offline;
        prior
    } else {
        State {
            schema_version: 1,
            plan_sha256: plan_sha,
            engine: engine.clone(),
            fidelity: options.fidelity,
            selected_arms: selected.clone(),
            selection_evidence: options.selection_evidence.clone(),
            offline: options.offline,
            trials: BTreeMap::new(),
            reserved_training_steps: 0,
            worker_wall_seconds: 0.,
            events: vec![],
        }
    };
    let mut pending = selected
        .into_iter()
        .filter(|id| {
            state.trials.get(id).is_none_or(|t| {
                t.status == "queued_after_controller_restart"
                    || t.status == "interrupted_progress_saved"
            })
        })
        .collect::<VecDeque<_>>();
    save(&state_path, &state).await?;
    let mut resources = Resources::new(plan.inputs.resources.clone());
    let mut workers = JoinSet::new();
    let started = Instant::now();
    let deadline = started + Duration::from_secs(options.maximum_run_seconds);
    let mut budget_stopped = false;
    let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
    loop {
        let mut considered = 0usize;
        while let Some(id) = pending.front().cloned() {
            if started.elapsed().as_secs() >= options.maximum_run_seconds {
                budget_stopped = true;
                break;
            }
            let already_reserved = state.trials.get(&id).is_some_and(|t| t.request.is_some());
            let additional_steps = if already_reserved {
                0
            } else {
                fidelity.steps as u64
            };
            if state
                .reserved_training_steps
                .checked_add(additional_steps)
                .is_none_or(|n| n > fidelity.maximum_total_steps)
            {
                pending.rotate_left(1);
                considered += 1;
                if considered >= pending.len() {
                    budget_stopped = true;
                    break;
                }
                continue;
            }
            let arm = plan
                .arms
                .iter()
                .find(|a| a.id == id)
                .context("selected arm disappeared")?
                .clone();
            let request_resources = resources_for(&plan.inputs, &arm, options.fidelity);
            let live = tokio::task::spawn_blocking(crate::platform::detect).await??;
            let active_cpu = resources.capacity.cpu_threads - resources.free_cpu;
            let gpu_fits = request_resources.vram_bytes == 0
                || plan.inputs.resources.gpu_ordinal.is_some_and(|ordinal| {
                    live.gpus.iter().any(|gpu| {
                        gpu.ordinal == ordinal
                            && request_resources.vram_bytes
                                <= gpu
                                    .free_memory_bytes
                                    .saturating_sub(gpu.free_memory_bytes / 10)
                                    .saturating_sub(128 << 20)
                    })
                });
            if request_resources.ram_bytes > live.usable_ram_bytes
                || active_cpu.saturating_add(request_resources.cpu_threads) as usize
                    > live.available_cpu_threads
                || !gpu_fits
            {
                pending.rotate_left(1);
                considered += 1;
                if considered >= pending.len() {
                    break;
                }
                continue;
            }
            pending.pop_front();
            considered = 0;
            let teacher_targets_cache_only = plan
                .inputs
                .student_resources
                .iter()
                .find(|p| p.student == arm.student)
                .is_some_and(|p| p.teacher_targets_cache_only);
            let directory = options.output.join("arms").join(&id);
            fs::create_dir_all(&directory)?;
            let training = training_for(&plan, &arm, options.fidelity);
            let mut request = WorkerRequest {
                schema_version: plan.schema_version,
                study_plan: Artefact::capture(&root, options.plan.clone())?,
                arm,
                fidelity: options.fidelity,
                training,
                inputs: plan.inputs.clone(),
                resources: request_resources,
                teacher_targets_cache_only,
                output: directory.clone(),
                offline: options.offline,
            };
            let request_path = directory.join("request.json");
            if plan.schema_version >= 2 && !request_path.exists() {
                let checked = request.clone();
                let eligibility =
                    tokio::task::spawn_blocking(move || crate::study::prepare_selection(&checked))
                        .await?;
                match eligibility {
                    Ok(receipt) => {
                        artefact::write_json(&directory.join("prepared-selection.json"), &receipt)?
                    }
                    Err(error) => {
                        let reason = format!("{error:#}");
                        artefact::write_json(
                            &directory.join("eligibility-rejection.json"),
                            &serde_json::json!({"schema_version":2,"arm_id":id,"reason":reason,"optimiser_created":false,"steps_reserved":0}),
                        )?;
                        state.trials.insert(
                            id.clone(),
                            Trial {
                                id,
                                status: "not_applicable_actual_selection".into(),
                                attempts: 0,
                                request: None,
                                outcome: None,
                                reason: Some(reason),
                                exit_code: None,
                                elapsed_seconds: 0.,
                            },
                        );
                        save(&state_path, &state).await?;
                        continue;
                    }
                }
            }
            if !resources.reserve(&request.resources) {
                pending.push_back(id);
                break;
            }
            if request_path.exists() {
                let previous: WorkerRequest = serde_json::from_slice(&fs::read(&request_path)?)?;
                request.offline = previous.offline;
            }
            let bytes = serde_json::to_vec_pretty(&request)?;
            if request_path.exists() {
                ensure!(
                    fs::read(&request_path)? == bytes,
                    "frozen worker request changed"
                );
            } else {
                artefact::atomic_write(&request_path, &bytes)?;
            }
            let artifact = Artefact::capture(&root, request_path.clone())?;
            let attempts = state.trials.get(&id).map_or(1, |t| t.attempts + 1);
            state.reserved_training_steps += additional_steps;
            state.trials.insert(
                id.clone(),
                Trial {
                    id: id.clone(),
                    status: "running".into(),
                    attempts,
                    request: Some(artifact.clone()),
                    outcome: None,
                    reason: None,
                    exit_code: None,
                    elapsed_seconds: 0.,
                },
            );
            state.events.push(serde_json::json!({"kind":"worker_resources_reserved_before_spawn","arm_id":id,"resources":request.resources,"steps":fidelity.steps}));
            save(&state_path, &state).await?;
            eprintln!(
                "study: starting arm={} fidelity={:?} threads={} RAM={} VRAM={}",
                id,
                options.fidelity,
                request.resources.cpu_threads,
                request.resources.ram_bytes,
                request.resources.vram_bytes
            );
            workers.spawn(execute(
                engine.clone(),
                request,
                request_path,
                artifact.sha256,
                deadline,
                options.offline,
            ));
        }
        if workers.is_empty() {
            if !pending.is_empty() && !budget_stopped {
                eprintln!(
                    "study: waiting for currently available resources; frozen plan and queued progress retained"
                );
                heartbeat.tick().await;
                continue;
            }
            break;
        }
        tokio::select! {
            completion=workers.join_next()=>{
                let completion=completion.context("worker set became empty")??;
                resources.release(&completion.resources)?;
                state.worker_wall_seconds+=completion.elapsed;
                let trial=state.trials.get_mut(&completion.id).context("worker result has unknown arm")?;
                trial.elapsed_seconds=completion.elapsed;trial.exit_code=completion.exit;
                match completion.result {
                    Ok(artifact)=>{trial.status="completed_real_training".into();trial.outcome=Some(artifact);},
                    Err(error)=>{trial.status=if error.is::<WorkerInterrupted>() {"interrupted_progress_saved"} else {"failed"}.into();trial.reason=Some(format!("{error:#}"));},
                }
                state.events.push(serde_json::json!({"kind":"worker_attempt_finished","arm_id":trial.id,"attempt":trial.attempts,"status":trial.status,
                    "elapsed_seconds":trial.elapsed_seconds,"exit_code":trial.exit_code,"reason":trial.reason,"outcome":trial.outcome}));
                eprintln!("study: arm={} status={} elapsed={:.2}s",trial.id,trial.status,trial.elapsed_seconds);
                save(&state_path,&state).await?;
            }
            _=heartbeat.tick()=>{eprintln!("study: heartbeat active={} selected_queued={} completed={} reserved_steps={}",resources.active,pending.len(),state.trials.values().filter(|t|t.status=="completed_real_training").count(),state.reserved_training_steps);}
        }
    }
    save(&state_path, &state).await?;
    budget_stopped |=
        started.elapsed().as_secs() >= options.maximum_run_seconds && !pending.is_empty();
    let status = if state.selected_arms.is_empty() {
        "no_executable_selected_arms"
    } else if budget_stopped {
        "budget_stopped_progress_saved"
    } else if state.trials.values().any(|t| t.status == "failed") {
        "completed_with_failures"
    } else if state
        .trials
        .values()
        .any(|t| t.status == "not_applicable_actual_selection")
    {
        "completed_with_inapplicable_data_selections"
    } else if state
        .trials
        .values()
        .any(|t| t.status == "interrupted_progress_saved")
    {
        "interrupted_progress_saved"
    } else {
        "selected_profile_complete_catalogue_may_remain_queued"
    };
    let value = report(&plan, &state, status);
    let path = options.output.join("report.json");
    let copy = value.clone();
    tokio::task::spawn_blocking(move || artefact::write_json(&path, &copy)).await??;
    Ok(value)
}
