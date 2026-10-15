//! Actual orchestrator SIGKILL, independent container ceiling and journal replay.
//! This supplements sandbox QA; it is not a provider-recording recovery drill.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact,
    sandbox::{self, SandboxConfig},
    tasks::{self, Split},
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::{
    process::{Child, Command},
    time::{sleep, timeout},
};

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    worker: Option<PathBuf>,
    #[arg(long, default_value = "reports/sandbox-crash-qa/round-001")]
    output: PathBuf,
    /// The declared ceiling covers container startup and execution; Podman's
    /// attached client may also retain its pipes during removal bookkeeping.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u64).range(1..=60))]
    wall_timeout_secs: u64,
}
struct OwnedWorker(Child);
impl Drop for OwnedWorker {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

async fn runtime(cfg: &SandboxConfig, arguments: &[&str]) -> Result<std::process::Output> {
    runtime_with_deadline(cfg, arguments, 10).await
}
async fn runtime_with_deadline(
    cfg: &SandboxConfig,
    arguments: &[&str],
    seconds: u64,
) -> Result<std::process::Output> {
    timeout(
        Duration::from_secs(seconds),
        Command::new(&cfg.runtime)
            .args(arguments)
            .kill_on_drop(true)
            .output(),
    )
    .await
    .context("crash QA runtime command deadline exceeded")?
    .context("start crash QA runtime command")
}
async fn inspect(cfg: &SandboxConfig, target: &str) -> Result<Option<Value>> {
    let result = runtime(cfg, &["inspect", target]).await?;
    if !result.status.success() {
        let error = String::from_utf8_lossy(&result.stderr).to_ascii_lowercase();
        ensure!(
            error.contains("no such container")
                || error.contains("no such object")
                || error.contains("no container with")
                || error.contains("does not exist"),
            "inspect failure does not establish absence: {error}"
        );
        return Ok(None);
    }
    let entries: Vec<Value> = serde_json::from_slice(&result.stdout)?;
    ensure!(entries.len() == 1, "ambiguous container inspection");
    Ok(entries.into_iter().next())
}
fn worker_journal(pid: u32) -> Result<Option<(PathBuf, Value)>> {
    let root = Path::new("state/sandbox-owned-runs");
    if !root.exists() {
        return Ok(None);
    }
    for scope in fs::read_dir(root)? {
        let scope = scope?;
        if !scope.file_type()?.is_dir() {
            continue;
        }
        for entry in fs::read_dir(scope.path())? {
            let entry = entry?;
            if entry.path().extension().is_none_or(|s| s != "json") {
                continue;
            }
            let value: Value = serde_json::from_slice(&fs::read(entry.path())?)?;
            if value["owner_pid"] == pid && value["completed_unix_ms"].is_null() {
                return Ok(Some((entry.path(), value)));
            }
        }
    }
    Ok(None)
}

async fn worker(path: &Path) -> Result<()> {
    let config: SandboxConfig = serde_json::from_slice(&fs::read(path)?)?;
    let task = tasks::corpus()
        .into_iter()
        .find(|task| task.split == Split::Train)
        .context("training fixture absent")?;
    let report=sandbox::grade(&task,"pub fn solve(_: serde_json::Value) -> serde_json::Value { loop { std::hint::spin_loop(); } }",&config).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    anyhow::bail!("controlled holder was not SIGKILLed while its candidate was running")
}

async fn exercise(
    output: &Path,
    cfg: &SandboxConfig,
    sentinel_id: &str,
    sentinel_nonce: &str,
) -> Result<Value> {
    let config_path = output.join("worker-config.json");
    artefact::write_json(&config_path, cfg)?;
    let executable = std::env::current_exe()?;
    let mut holder = OwnedWorker(
        Command::new(&executable)
            .arg("--worker")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::from(fs::File::create(
                output.join("worker-stdout.log"),
            )?))
            .stderr(Stdio::from(fs::File::create(
                output.join("worker-stderr.log"),
            )?))
            .kill_on_drop(true)
            .spawn()?,
    );
    let pid = holder.0.id().context("holder PID absent")?;
    let started = Instant::now();
    let (journal_path, journal, container, top) = loop {
        ensure!(
            holder.0.try_wait()?.is_none(),
            "holder ended before SIGKILL"
        );
        ensure!(
            started.elapsed() < Duration::from_secs(30),
            "controlled candidate did not start within bounded QA window"
        );
        if let Some((path, journal)) = worker_journal(pid)? {
            let name = journal["name"].as_str().context("journal name absent")?;
            if let Some(container) = inspect(cfg, name).await? {
                let id = container["Id"]
                    .as_str()
                    .context("owned container ID absent")?;
                let top = runtime(cfg, &["top", id, "args"]).await?;
                let processes = String::from_utf8_lossy(&top.stdout).to_string();
                if top.status.success()
                    && processes
                        .lines()
                        .any(|line| line.contains("/tmp/candidate"))
                {
                    break (path, journal, container, processes);
                }
            }
        }
        sleep(Duration::from_millis(100)).await;
    };
    let id = container["Id"]
        .as_str()
        .context("candidate ID absent")?
        .to_owned();
    let name = journal["name"].as_str().context("candidate name absent")?;
    ensure!(
        container["State"]["Running"] == true
            && container["Name"]
                .as_str()
                .context("container name absent")?
                .trim_start_matches('/')
                == name,
        "controlled candidate is not running under the recorded name"
    );
    ensure!(
        container["Config"]["Labels"]["org.lec.sandbox.nonce"] == journal["nonce"],
        "actual runtime nonce differs from durable journal"
    );
    artefact::write_json(
        &output.join("before-kill.json"),
        &json!({"holder_pid":pid,"journal_path":journal_path,"journal":journal,"container":container,"processes":top}),
    )?;
    let killed_at = Instant::now();
    // SAFETY: the retained child handle, actual pending journal and live child
    // check establish this PID belongs to the controlled QA holder we spawned.
    ensure!(
        unsafe { libc::kill(pid as i32, libc::SIGKILL) } == 0,
        "SIGKILL of controlled holder failed"
    );
    let status = timeout(Duration::from_secs(5), holder.0.wait()).await??;
    use std::os::unix::process::ExitStatusExt;
    ensure!(
        status.signal() == Some(libc::SIGKILL),
        "holder was not terminated by SIGKILL"
    );
    // No recovery command runs during this phase. The runtime's own timeout
    // must remove the untrusted workload despite abrupt host process death.
    while inspect(cfg, &id).await?.is_some() {
        ensure!(
            killed_at.elapsed() < Duration::from_secs(cfg.wall_timeout_secs + 12),
            "container survived the independent runtime ceiling"
        );
        sleep(Duration::from_millis(200)).await;
    }
    let ceiling_seconds = killed_at.elapsed().as_secs_f64();
    let still_pending: Value = serde_json::from_slice(&fs::read(&journal_path)?)?;
    ensure!(
        still_pending["completed_unix_ms"].is_null(),
        "journal was acknowledged without subsequent recovery"
    );
    let reference = tasks::corpus()
        .into_iter()
        .find(|task| task.split == Split::Train)
        .context("reference task absent")?;
    let grade=sandbox::grade(&reference,"pub fn solve(input: serde_json::Value) -> serde_json::Value { serde_json::json!(input.as_array().unwrap().iter().map(|value| value.as_i64().unwrap()).sum::<i64>()) }",cfg).await?;
    ensure!(
        grade.passed && grade.trace_complete && !grade.quarantined,
        "post-crash reference grading failed: {:?}",
        grade.failure
    );
    let recovered: Value = serde_json::from_slice(&fs::read(&journal_path)?)?;
    ensure!(
        recovered["completed_unix_ms"].as_i64().is_some()
            && recovered["completion_reason"] == "recovered_after_original_owner_died",
        "subsequent dispatch did not durably acknowledge the dead holder's ownership journal"
    );
    let sentinel = inspect(cfg, sentinel_id)
        .await?
        .context("unrelated sentinel was removed")?;
    ensure!(
        sentinel["State"]["Running"] == true
            && sentinel["Config"]["Labels"]["org.lec.crash-qa.unrelated"] == sentinel_nonce,
        "unrelated sentinel did not survive recovery unchanged"
    );
    Ok(
        json!({"schema_version":1,"evidence_kind":"actual_orchestrator_sigkill_sandbox_ceiling_and_owned_recovery_qa",
        "passed":true,"holder_pid":pid,"holder_exit_signal":status.signal(),"actual_infinite_candidate_observed":true,
        "runtime_hard_ceiling_seconds":cfg.wall_timeout_secs,"seconds_from_sigkill_to_absence":ceiling_seconds,
        "runtime_ceiling_observed_before_recovery":true,"recovered_journal_path":journal_path,"recovered_journal":recovered,
        "post_crash_reference_run_id":grade.run_id,"post_crash_reference_passed":grade.passed,"unrelated_sentinel_survived":true,
        "unrelated_sentinel_id":sentinel_id,"example_executable_sha256":artefact::file_hash(&executable)?,
        "limitation":"Actual local sandbox crash/ceiling/recovery only; no provider recordings, paid-call recovery, model quality, held-out result or assessment drill completion is claimed."}),
    )
}

async fn run_parent(args: Arguments) -> Result<()> {
    ensure!(
        !args.output.exists(),
        "preserve previous crash QA; select a fresh output directory"
    );
    fs::create_dir_all(&args.output)?;
    let mut cfg = SandboxConfig::portable();
    cfg.report_dir = args.output.join("runs");
    cfg.wall_timeout_secs = args.wall_timeout_secs;
    eprintln!("sandbox-crash-qa: verifying controls before actual holder SIGKILL");
    let preflight = sandbox::preflight(&cfg).await?;
    cfg.expected_image_id = preflight.image_id;
    let nonce = uuid::Uuid::new_v4().to_string();
    let sentinel_name = format!("lec-crash-unrelated-{nonce}");
    let label = format!("org.lec.crash-qa.unrelated={nonce}");
    let created = runtime(
        &cfg,
        &[
            "run",
            "-d",
            "--rm",
            "--name",
            &sentinel_name,
            "--read-only",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--user",
            "65532:65532",
            "--memory",
            "32m",
            "--memory-swap",
            "32m",
            "--pids-limit",
            "8",
            "--cpus",
            "0.1",
            "--timeout",
            "120",
            "--label",
            &label,
            "--entrypoint",
            "/bin/sleep",
            &cfg.expected_image_id,
            "110",
        ],
    )
    .await?;
    ensure!(
        created.status.success(),
        "unrelated sentinel failed to start: {}",
        String::from_utf8_lossy(&created.stderr)
    );
    let sentinel_id = String::from_utf8(created.stdout)?.trim().to_owned();
    ensure!(
        sentinel_id.len() == 64 && sentinel_id.bytes().all(|b| b.is_ascii_hexdigit()),
        "sentinel did not return immutable ID"
    );
    let result = exercise(&args.output, &cfg, &sentinel_id, &nonce).await;
    // Keep the measured assertions independently, including when the final
    // trusted-fixture cleanup fails. Overall success is published only later.
    if let Ok(value) = &result {
        artefact::write_json(&args.output.join("exercise-report.json"), value)?;
    }
    let cleanup = async {
        if let Some(actual) = inspect(&cfg, &sentinel_id).await? {
            ensure!(
                actual["Config"]["Labels"]["org.lec.crash-qa.unrelated"] == nonce,
                "refusing to remove a sentinel whose ownership changed"
            );
            if actual["State"]["Running"] == true {
                let killed = runtime(&cfg, &["kill", "--signal", "KILL", &sentinel_id]).await?;
                ensure!(
                    killed.status.success() || inspect(&cfg, &sentinel_id).await?.is_none(),
                    "trusted QA sentinel SIGKILL was not acknowledged"
                );
            }
            // Runtime removal housekeeping is separately bounded; it must not
            // consume the candidate's declared execution ceiling.
            let _ = runtime_with_deadline(&cfg, &["rm", "-f", &sentinel_id], 30).await;
        }
        ensure!(
            inspect(&cfg, &sentinel_id).await?.is_none(),
            "trusted QA sentinel cleanup incomplete"
        );
        Ok::<_, anyhow::Error>(())
    }
    .await;
    let complete = match (result, cleanup) {
        (Ok(mut value), Ok(())) => {
            value["sentinel_cleanup_verified"] = true.into();
            Ok(value)
        }
        (Err(error), _) => Err(error),
        (_, Err(error)) => Err(error.context("final trusted QA sentinel cleanup failed")),
    };
    let report = match &complete {
        Ok(value) => value.clone(),
        Err(error) => {
            json!({"schema_version":1,"passed":false,"evidence_kind":"incomplete_actual_sandbox_sigkill_qa","error":format!("{error:#}")})
        }
    };
    artefact::write_json(&args.output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    complete.map(|_| ())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arguments::parse();
    if let Some(path) = args.worker {
        return worker(&path).await;
    }
    let output = args.output.clone();
    let existed = output.exists();
    let result = run_parent(args).await;
    if !existed
        && output.exists()
        && !output.join("report.json").exists()
        && let Err(error) = &result
    {
        let failure = json!({"schema_version":1,"passed":false,"evidence_kind":"incomplete_actual_sandbox_sigkill_qa","error":format!("{error:#}")});
        artefact::write_json(&output.join("report.json"), &failure)?;
        println!("{}", serde_json::to_string_pretty(&failure)?);
    }
    result
}
