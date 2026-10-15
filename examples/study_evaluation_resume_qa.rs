//! Actual owned-process interruption and immutable development-cache recovery.
//! This tests recovery, not learning or assessment promotion.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact::{self, Artefact},
    engine, sandbox, study_evaluate,
    tasks::{self, Split},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::process::{CommandExt, ExitStatusExt},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    /// Exact immutable application executable; this command never builds it.
    #[arg(long)]
    engine: Option<PathBuf>,
    #[arg(long, default_value = "offline_bundle/base")]
    base: PathBuf,
    #[arg(long, default_value = "reports/local-study/native-composed-operator")]
    checkpoint: PathBuf,
    #[arg(
        long,
        default_value = "reports/local-study/study-evaluation-resume-qa/round-001"
    )]
    output: PathBuf,
    #[arg(long, default_value="cuda", value_parser=["cpu","cuda"])]
    device: String,
    #[arg(long, default_value_t=4, value_parser=clap::value_parser!(u32).range(1..=256))]
    threads: u32,
    /// Includes both application attempts and the independent verifier process.
    #[arg(long, default_value_t=1800, value_parser=clap::value_parser!(u64).range(30..=14400))]
    wall_timeout_seconds: u64,
    #[arg(long, hide = true)]
    verify_only: Option<PathBuf>,
}

struct OwnedChild {
    child: Child,
    group: i32,
    reaped: bool,
}
impl Drop for OwnedChild {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.kill();
        }
    }
}
impl OwnedChild {
    fn spawn(mut command: Command) -> Result<Self> {
        command.process_group(0);
        let parent = std::process::id() as i32;
        // SAFETY: the hook performs only async-signal-safe libc calls before
        // exec. Exact-parent checking closes the parent-death registration race.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
        let child = command
            .spawn()
            .context("start owned evaluation QA process")?;
        let group = i32::try_from(child.id())?;
        Ok(Self {
            child,
            group,
            reaped: false,
        })
    }
    fn poll(&mut self) -> Result<Option<ExitStatus>> {
        let result = self.child.try_wait()?;
        if result.is_some() {
            self.reaped = true;
        }
        Ok(result)
    }
    fn kill(&mut self) -> Result<ExitStatus> {
        ensure!(
            !self.reaped,
            "owned process already exited before deliberate SIGKILL"
        );
        // SAFETY: this isolated process group belongs to our retained unreaped
        // child; no external or user-supplied PID is ever signalled.
        let signalled = unsafe { libc::kill(-self.group, libc::SIGKILL) };
        let error = std::io::Error::last_os_error();
        ensure!(
            signalled == 0 || error.raw_os_error() == Some(libc::ESRCH),
            "owned process SIGKILL failed: {error}"
        );
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }
    fn wait_until(&mut self, deadline: Instant, phase: &str) -> Result<ExitStatus> {
        let mut heartbeat = Instant::now();
        loop {
            if let Some(status) = self.poll()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                let status = self.kill()?;
                anyhow::bail!(
                    "{phase} exceeded whole-QA deadline; owned child reaped with {status}"
                );
            }
            if heartbeat.elapsed() >= Duration::from_secs(10) {
                eprintln!(
                    "evaluation-resume-qa: {phase}; owned child {} remains active",
                    self.child.id()
                );
                heartbeat = Instant::now();
            }
            thread::sleep(Duration::from_millis(100));
        }
    }
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(root, path.canonicalize()?.strip_prefix(root)?)
}
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn published(directory: &Path) -> Result<Vec<PathBuf>> {
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut result = fs::read_dir(directory)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    result.retain(|p| p.extension().is_some_and(|s| s == "json"));
    result.sort();
    Ok(result)
}

/// A model error or truncated output cannot stand in for a completed grade.
fn graded_observation(root: &Path, output: &Path) -> Result<Option<Value>> {
    let plan_path = output.join("plan.json");
    if !plan_path.exists() {
        return Ok(None);
    }
    let plan = read(&plan_path)?;
    ensure!(
        plan["split"] == "development" && plan["held_out_used"] != true,
        "QA cannot open held-out data"
    );
    for path in published(&output.join("observations"))? {
        let row = read(&path)?;
        if row["sandbox_report"].is_null() {
            continue;
        }
        let raw_file: Artefact = serde_json::from_value(row["raw_generation"].clone())?;
        let raw = read(&raw_file.verify(root)?)?;
        let completion = raw["generation"]["completion"]
            .as_str()
            .context("graded raw completion absent")?;
        ensure!(
            raw["generation"]["complete"] == true
                && !completion.trim().is_empty()
                && raw["model_error"].is_null()
                && row["grading_error"].is_null()
                && row["plan_sha256"] == artefact::file_hash(&plan_path)?
                && raw["plan_sha256"] == row["plan_sha256"]
                && raw["generation"]["checkpoint_id"] == plan["checkpoint"]["checkpoint_id"]
                && row["task_id"] == raw["task_id"]
                && row["source_id"] == raw["source_id"],
            "published grade lacks its complete checkpoint-bound raw output"
        );
        let task = tasks::corpus()
            .into_iter()
            .find(|task| {
                task.split == Split::Development
                    && row["task_id"] == task.id
                    && row["source_id"] == task.source_id
            })
            .context("grade is outside original development pack")?;
        let config: sandbox::SandboxConfig = serde_json::from_value(plan["sandbox"].clone())?;
        let grade_file: Artefact = serde_json::from_value(row["sandbox_report"].clone())?;
        let mut grade: sandbox::SandboxReport =
            serde_json::from_slice(&fs::read(grade_file.verify(root)?)?)?;
        let audit = if grade.audit_path.is_absolute() {
            grade.audit_path.clone()
        } else {
            root.join(&grade.audit_path)
        }
        .canonicalize()?;
        ensure!(
            audit.starts_with(root) && artefact::file_hash(&audit)? == grade.audit_sha256,
            "completed grade audit changed"
        );
        grade.audit_path = audit;
        sandbox::verify_report(
            &grade,
            &config,
            &task,
            &lec_distil::evaluate::extract_code(completion),
        )?;
        let anchor_kind = if grade.trace_complete {
            "executed_candidate_verdict"
        } else {
            let role = lec_distil::teacher_evidence::classify_content(
                root,
                &grade_file,
                &config,
                &task,
                completion,
                true,
            )?;
            if role.role != lec_distil::teacher_evidence::ContentRole::CompilationFeedbackOnly {
                continue;
            }
            // A verified compiler verdict is completed failed grading work.
            // Recovery may reuse it; no positive or DPO-negative role follows.
            "verified_compilation_failure_verdict"
        };
        return Ok(Some(
            json!({"task_id":task.id,"observation":capture(root,&path)?,
            "anchor_kind":anchor_kind,"training_eligibility_inferred":false,
            "raw_generation":raw_file,"sandbox_report":grade_file,"sandbox_audit":capture(root,&grade.audit_path)?,
            "candidate_passed":grade.passed,"quarantined":grade.quarantined}),
        ));
    }
    Ok(None)
}

fn snapshot(root: &Path, evaluation: &Path) -> Result<BTreeMap<PathBuf, Artefact>> {
    let mut paths = Vec::new();
    for name in [
        "plan.json",
        "launch-engine.json",
        "sandbox-preflight.json",
        "execution.json",
    ] {
        let path = evaluation.join(name);
        if path.exists() {
            paths.push(path);
        }
    }
    for name in ["generations", "observations", "execution-attempts"] {
        paths.extend(published(&evaluation.join(name))?);
    }
    for observation in published(&evaluation.join("observations"))? {
        let row = read(&observation)?;
        if !row["sandbox_report"].is_null() {
            let file: Artefact = serde_json::from_value(row["sandbox_report"].clone())?;
            let path = file.verify(root)?;
            let report: sandbox::SandboxReport = serde_json::from_slice(&fs::read(&path)?)?;
            paths.push(path);
            paths.push(if report.audit_path.is_absolute() {
                report.audit_path
            } else {
                root.join(report.audit_path)
            });
        }
    }
    paths
        .into_iter()
        .map(|path| {
            let file = capture(root, &path)?;
            Ok((file.path.clone(), file))
        })
        .collect()
}

fn environment(threads: u32) -> BTreeMap<String, String> {
    [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "LEC_STUDY_CPU_THREADS",
    ]
    .into_iter()
    .map(|name| (name.into(), threads.to_string()))
    .chain([
        ("TOKIO_WORKER_THREADS".into(), threads.min(4).to_string()),
        ("LEC_STUDY_OFFLINE".into(), "1".into()),
    ])
    .collect()
}
fn command(
    executable: &Path,
    arguments: &[String],
    env: &BTreeMap<String, String>,
    output: &Path,
    name: &str,
) -> Result<Command> {
    let mut command = Command::new(executable);
    command
        .args(arguments)
        .envs(env)
        .stdin(Stdio::null())
        .stdout(Stdio::from(fs::File::create(
            output.join(format!("{name}-stdout.log")),
        )?))
        .stderr(Stdio::from(fs::File::create(
            output.join(format!("{name}-stderr.log")),
        )?));
    Ok(command)
}

fn exercise(args: &Arguments, root: &Path) -> Result<Value> {
    let deadline = Instant::now() + Duration::from_secs(args.wall_timeout_seconds);
    let executable = args
        .engine
        .as_ref()
        .context("--engine must identify an immutable pinned application executable")?
        .canonicalize()?;
    ensure!(
        executable.starts_with(root),
        "application engine is outside evidence root"
    );
    let pinned = engine::verified_snapshot(
        &executable,
        &artefact::file_hash(&executable)?,
        fs::metadata(&executable)?.len(),
    )?;
    let evaluation = args.output.join("evaluation");
    ensure!(
        !evaluation.exists(),
        "existing evaluation retained; select a new QA output"
    );
    fs::create_dir_all(&evaluation)?;
    // The launch receipt is durable even if the application dies before plan.json.
    artefact::write_json(&evaluation.join("launch-engine.json"), &pinned)?;
    let adapter = capture(root, &args.checkpoint.join("adapter.safetensors"))?;
    let manifest = capture(root, &args.checkpoint.join("manifest.json"))?;
    let checkpoint = read(&manifest.verify(root)?)?;
    ensure!(
        checkpoint["reloaded"] == true
            && checkpoint["adapter_sha256"] == adapter.sha256
            && checkpoint["spec"]["student_kind"] == "conventional_qwen_lora",
        "QA requires an actual reloaded conventional checkpoint"
    );
    let arguments = vec![
        "study-evaluate".into(),
        "--base".into(),
        args.base.display().to_string(),
        "--checkpoint".into(),
        args.checkpoint.display().to_string(),
        "--output".into(),
        evaluation.display().to_string(),
        "--split".into(),
        "development".into(),
        "--device".into(),
        args.device.clone(),
        "--dtype".into(),
        "f32".into(),
        "--offline".into(),
    ];
    let env = environment(args.threads);
    artefact::write_json(
        &args.output.join("protocol.json"),
        &json!({"schema_version":1,
        "evidence_kind":"frozen_owned_study_evaluation_sigkill_resume_qa","engine":pinned,
        "qa_executable":capture(root,&std::env::current_exe()?)?,"adapter":adapter,"manifest":manifest,
        "base_assets":capture(root,&args.base.join("assets.json"))?,"arguments":arguments,"thread_environment":env,
        "wall_timeout_seconds":args.wall_timeout_seconds,"split":"development","held_out_used":false,
        "coding_quality_or_learning_claim":false,"production_eligible":false}),
    )?;
    let mut first = OwnedChild::spawn(command(
        &pinned.path,
        &arguments,
        &env,
        &args.output,
        "interrupted",
    )?)?;
    let mut heartbeat = Instant::now();
    let completed = loop {
        ensure!(
            first.poll()?.is_none(),
            "first evaluation exited before actual interruption"
        );
        ensure!(
            Instant::now() < deadline,
            "no completed raw+grade appeared within QA deadline"
        );
        if let Some(row) = graded_observation(root, &evaluation)? {
            ensure!(
                !evaluation.join("report.json").exists(),
                "evaluation completed before SIGKILL; no interruption proven"
            );
            break row;
        }
        if heartbeat.elapsed() >= Duration::from_secs(10) {
            eprintln!(
                "evaluation-resume-qa: waiting for real completed generation and container grade; owned PID {}",
                first.child.id()
            );
            heartbeat = Instant::now();
        }
        thread::sleep(Duration::from_millis(100));
    };
    let pid = first.child.id();
    let killed = first.kill()?;
    ensure!(
        killed.signal() == Some(libc::SIGKILL),
        "owned evaluator was not actually killed by SIGKILL"
    );
    ensure!(
        !evaluation.join("report.json").exists()
            && published(&evaluation.join("observations"))?.len() < 16,
        "evaluation finished all observations before interruption; no mid-round recovery proven"
    );
    for key in [
        "observation",
        "raw_generation",
        "sandbox_report",
        "sandbox_audit",
    ] {
        serde_json::from_value::<Artefact>(completed[key].clone())?.verify(root)?;
    }
    let cached = snapshot(root, &evaluation)?;
    artefact::write_json(
        &args.output.join("interrupted.json"),
        &json!({"schema_version":1,"owned_pid":pid,
        "exit_signal":killed.signal(),"completed_grade":completed,"cached_artefacts":cached,
        "complete_report_existed":evaluation.join("report.json").exists()}),
    )?;
    eprintln!(
        "evaluation-resume-qa: evaluator SIGKILLed after a verified grade; restarting exact engine and cached output"
    );
    let mut resumed = OwnedChild::spawn(command(
        &pinned.path,
        &arguments,
        &env,
        &args.output,
        "resumed",
    )?)?;
    let resumed_status = resumed.wait_until(deadline, "resumed evaluation")?;
    // This separate process reconstructs every generation and trusted grade.
    let verifier = engine::pin_executable(
        &std::env::current_exe()?,
        Path::new("state/engines/diagnostics"),
    )?;
    let verify_arguments = vec![
        "--verify-only".into(),
        evaluation.display().to_string(),
        "--output".into(),
        args.output.display().to_string(),
    ];
    let mut checking = OwnedChild::spawn(command(
        &verifier.path,
        &verify_arguments,
        &env,
        &args.output,
        "independent-verifier",
    )?)?;
    let verification_status = checking.wait_until(deadline, "independent verification")?;
    let mut changed = Vec::new();
    for (path, file) in &cached {
        if let Err(error) = file.verify(root) {
            changed.push(json!({"path":path,"error":format!("{error:#}")}));
        }
    }
    let verification = read(&args.output.join("independent-verification.json")).ok();
    let healthy = verification
        .as_ref()
        .is_some_and(|v| v["report"]["healthy_complete"] == true);
    let passed =
        resumed_status.success() && verification_status.success() && changed.is_empty() && healthy;
    let mut failure_reasons = Vec::new();
    if !resumed_status.success() {
        failure_reasons.push("Resumed evaluator exited unsuccessfully.");
    }
    if !verification_status.success() {
        failure_reasons.push("Independent raw/graded verification exited unsuccessfully.");
    }
    if !changed.is_empty() {
        failure_reasons.push("Published pre-interruption cache artefacts changed.");
    }
    if !healthy {
        failure_reasons.push("The complete original development evaluation is absent or retains generation/infrastructure failures.");
    }
    Ok(
        json!({"schema_version":1,"evidence_kind":"actual_owned_study_evaluation_sigkill_resume_qa","passed":passed,
        "failure_reasons":failure_reasons,
        "actual_sigkill":true,"resumed_exit_code":resumed_status.code(),"independent_verifier_exit_code":verification_status.code(),
        "cached_artefacts":cached,"changed_cached_artefacts":changed,"completed_before_kill":completed,
        "independent_verification":verification,"complete_healthy_evaluation":healthy,
        "evaluation_report":if evaluation.join("report.json").exists(){Some(capture(root,&evaluation.join("report.json"))?)}else{None},
        "held_out_used":false,"coding_quality_or_learning_claim":false,"production_eligible":false,
        "limitations":["A three-step composed operator checkpoint is not evidence of useful learning.",
            "Recovery preserves committed outputs; interrupted uncommitted inference may run again.",
            "Errors and truncated outputs remain failures in the full original development denominator."]}),
    )
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let current = std::env::current_exe()?.canonicalize()?;
    let pinned = engine::pin_executable(&current, Path::new("state/engines/diagnostics"))?;
    if current != pinned.path {
        // exec retains the caller's process and signals while ensuring all
        // observed QA work belongs to the immutable diagnostic executable.
        let error = Command::new(&pinned.path)
            .args(std::env::args_os().skip(1))
            .exec();
        return Err(error).context("exec pinned evaluation recovery diagnostic");
    }
    if let Some(directory) = &args.verify_only {
        let report = study_evaluate::verify_completed(directory)?;
        let root = std::env::current_dir()?.canonicalize()?;
        artefact::write_json(
            &args.output.join("independent-verification.json"),
            &json!({"schema_version":1,
            "evidence_kind":"independently_reconstructed_complete_study_evaluation","report":report,
            "report_artefact":capture(&root,&directory.join("report.json"))?}),
        )?;
        ensure!(
            report["healthy_complete"] == true,
            "complete development evaluation retains generation/infrastructure failures"
        );
        return Ok(());
    }
    ensure!(
        !args.output.exists(),
        "QA output already exists; retain it and choose a new output"
    );
    fs::create_dir_all(&args.output)?;
    let root = std::env::current_dir()?.canonicalize()?;
    ensure!(
        args.output.canonicalize()?.starts_with(&root),
        "QA output escapes evidence root"
    );
    let result = exercise(&args, &root);
    let report = match result {
        Ok(report) => report,
        Err(error) => {
            json!({"schema_version":1,"evidence_kind":"incomplete_actual_study_evaluation_sigkill_resume_qa",
            "passed":false,"error":format!("{error:#}"),
            "interrupted_evidence":if args.output.join("interrupted.json").exists(){Some(capture(&root,&args.output.join("interrupted.json"))?)}else{None},
            "held_out_used":false,"production_eligible":false})
        }
    };
    artefact::write_json(&args.output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        report["passed"] == true,
        "actual evaluation interruption/resume QA failed; evidence retained in {}",
        args.output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deadline_kills_and_reaps_only_the_owned_process_group() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        let mut owned = OwnedChild::spawn(command).unwrap();
        let started = Instant::now();
        assert!(
            owned
                .wait_until(
                    started + Duration::from_millis(150),
                    "labelled lifecycle test"
                )
                .is_err()
        );
        assert!(owned.reaped);
        assert!(started.elapsed() < Duration::from_secs(3));
    }
}
