//! Host controller three: stdin-gated execution with cached kernel controls.
use super::*;
use std::{os::unix::process::ExitStatusExt, process::Output, sync::mpsc};

pub(super) const METADATA_SECS: u64 = 30;
const STARTUP_SECS: u64 = 60;
const CLEANUP_SECS: u64 = 30;
const AUDIT_CAP: usize = 1 << 20;
const STDERR_CAP: usize = 64 << 10;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleEvidence {
    pub receipt_path: PathBuf,
    pub receipt_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeCall {
    operation: String,
    arguments: Vec<String>,
    duration_ms: u64,
    exit_code: Option<i32>,
    signal: Option<i32>,
    deadline_exceeded: bool,
    stdout: stream::StreamEvidence,
    stderr: stream::StreamEvidence,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Receipt {
    schema_version: u32,
    run_id: String,
    container_name: String,
    container_id: Option<String>,
    nonce: String,
    scope: String,
    image_id: String,
    policy_sha256: String,
    source_sha256: String,
    code_sha256: String,
    native_timeout_secs: u64,
    candidate_timeout_secs: u64,
    pid1_watchdog_timeout_secs: u64,
    startup_budget_secs: u64,
    cleanup_budget_secs: u64,
    metadata_budget_secs: u64,
    attached_launch_unix_ms: Option<i64>,
    input_release_unix_ms: Option<i64>,
    input_release_after_attached_launch_ms: Option<u64>,
    native_timeout_starts_before_input_release: bool,
    full_post_release_allowance_guaranteed: bool,
    cgroup: Option<control::CgroupBinding>,
    kill: Option<control::CachedKillEvidence>,
    guard_stop_reason: Option<String>,
    terminal_inspection: Option<Value>,
    gated_inspection: Option<Value>,
    process_binding: Option<Value>,
    eof_timer_metadata: Option<Value>,
    pid1_watchdog: Option<Value>,
    container_removed: bool,
    classification: String,
    failure: Option<String>,
    calls: Vec<NativeCall>,
    audit: Option<stream::StreamEvidence>,
    stderr: Option<stream::StreamEvidence>,
    duration_ms: u64,
}
pub(super) struct Execution {
    pub output: Option<Output>,
    pub failure: Option<String>,
    pub timeout: Option<TimeoutEvidence>,
    pub evidence: LifecycleEvidence,
}

#[path = "v3/build.rs"]
mod build;
#[path = "v3/guard.rs"]
mod guard;
#[path = "v3/native.rs"]
mod native;
#[path = "v3/probe.rs"]
mod probe;
#[path = "v3/verification.rs"]
mod verification;
use guard::*;
use native::*;

pub(super) fn policy_sources() -> Value {
    json!({"controller_version":3,"metadata_seconds":METADATA_SECS,"startup_seconds":STARTUP_SECS,
        "cleanup_seconds":CLEANUP_SECS,"audit_cap":AUDIT_CAP,"stderr_cap":STDERR_CAP,
        "outer_startup_crash_ceiling_seconds":80,"pid1_watchdog_seconds":20,
        "host_source_sha256":tasks::hash(include_bytes!("../sandbox.rs")),
        "audit_v3_sha256":tasks::hash(include_bytes!("../../sandbox/policies/v3/audit.rs")),
        "syscalls_v3_sha256":tasks::hash(include_bytes!("../../sandbox/policies/v3/syscalls.rs")),
        "ptrace_v3_sha256":tasks::hash(include_bytes!("../../sandbox/policies/v3/ptrace.rs")),
        "seccomp_v3_sha256":tasks::hash(include_bytes!("../../sandbox/policies/v3/seccomp.json")),
        "containerfile_v3_sha256":tasks::hash(include_bytes!("../../sandbox/policies/v3/Containerfile")),
        "native_source_sha256":tasks::hash(include_bytes!("v3.rs")),
        "native_operations_sha256":tasks::hash(include_bytes!("v3/native.rs")),
        "deadline_guard_sha256":tasks::hash(include_bytes!("v3/guard.rs")),
        "preflight_probe_sha256":tasks::hash(include_bytes!("v3/probe.rs")),
        "verification_sha256":tasks::hash(include_bytes!("v3/verification.rs")),
        "cached_build_sha256":tasks::hash(include_bytes!("v3/build.rs")),
        "stream_source_sha256":tasks::hash(include_bytes!("stream.rs")),
        "control_source_sha256":tasks::hash(include_bytes!("control.rs")),
        "prepared_source_sha256":tasks::hash(include_bytes!("prepared.rs")),
        "recovery_source_sha256":tasks::hash(include_bytes!("recovery.rs")),
        "owned_source_sha256":tasks::hash(include_bytes!("owned.rs")),
        "lease_source_sha256":tasks::hash(include_bytes!("runtime_lease.rs")),
        "gate":"trusted_pid1_supervisor_stdin_eof_before_compilation",
        "deadline_control":"preopened_exact_owned_cgroup_kill_fd"})
}

pub(super) async fn execute(
    cfg: &SandboxConfig,
    name: &str,
    source: &Path,
    input: Vec<u8>,
    ownership: &mut recovery::OwnedRun,
    audit_path: &Path,
    code_sha256: &str,
) -> Result<Execution> {
    let start = Instant::now();
    let run_id = name
        .strip_prefix("lec-")
        .context("owned run name invalid")?;
    let prefix = cfg.report_dir.join(run_id);
    let mut receipt = Receipt {
        schema_version: 3,
        run_id: run_id.into(),
        container_name: name.into(),
        container_id: None,
        nonce: ownership.nonce().into(),
        scope: ownership.scope().into(),
        image_id: cfg.expected_image_id.clone(),
        policy_sha256: policy_sha256(cfg)?,
        source_sha256: tasks::hash(tokio::fs::read(source).await?),
        code_sha256: code_sha256.into(),
        native_timeout_secs: 80,
        candidate_timeout_secs: cfg.wall_timeout_secs,
        pid1_watchdog_timeout_secs: 20,
        startup_budget_secs: STARTUP_SECS,
        cleanup_budget_secs: CLEANUP_SECS,
        metadata_budget_secs: METADATA_SECS,
        attached_launch_unix_ms: None,
        input_release_unix_ms: None,
        input_release_after_attached_launch_ms: None,
        native_timeout_starts_before_input_release: true,
        full_post_release_allowance_guaranteed: true,
        cgroup: None,
        kill: None,
        guard_stop_reason: None,
        terminal_inspection: None,
        gated_inspection: None,
        process_binding: None,
        eof_timer_metadata: None,
        pid1_watchdog: None,
        container_removed: false,
        classification: "infrastructure_failure".into(),
        failure: None,
        calls: vec![],
        audit: None,
        stderr: None,
        duration_ms: 0,
    };
    let result = execute_inner(
        cfg,
        source,
        input,
        ownership,
        audit_path,
        &prefix,
        &mut receipt,
    )
    .await;
    let (output, timing) = match result {
        Ok(result) => result,
        Err(error) => {
            receipt.failure = Some(format!("{error:#}"));
            (None, None)
        }
    };
    receipt.duration_ms = start.elapsed().as_millis() as u64;
    let path = cfg.report_dir.join(format!("{run_id}.lifecycle.json"));
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    stream::publish_new(&path, &bytes).await?;
    Ok(Execution {
        output,
        failure: receipt.failure.clone(),
        timeout: timing,
        evidence: LifecycleEvidence {
            receipt_path: path,
            receipt_sha256: tasks::hash(bytes),
        },
    })
}

async fn execute_inner(
    cfg: &SandboxConfig,
    source: &Path,
    input: Vec<u8>,
    ownership: &mut recovery::OwnedRun,
    audit_path: &Path,
    prefix: &Path,
    receipt: &mut Receipt,
) -> Result<(Option<Output>, Option<TimeoutEvidence>)> {
    let container_name = receipt.container_name.clone();
    let name = container_name.as_str();
    ensure!(input.len() <= 131072, "trusted input cap exceeded");
    let startup = Instant::now();
    let mut create = super::base_command_kind(cfg, name, true)?;
    super::add_ownership_labels(&mut create, ownership);
    create.args([
        "--volume",
        &format!(
            "{}:/candidate:ro",
            source
                .parent()
                .context("candidate directory absent")?
                .display()
        ),
        "-i",
        image_reference(cfg),
        "/candidate/main.rs",
    ]);
    let create_args = create
        .as_std()
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let (created, create_call) = native(
        cfg,
        create,
        create_args,
        "create",
        prefix,
        remaining(startup, STARTUP_SECS)?,
    )
    .await?;
    let create_deadline = create_call.deadline_exceeded;
    receipt.calls.push(create_call);
    ensure!(
        !create_deadline && created.status.success(),
        "owned container creation failed: {}",
        String::from_utf8_lossy(&created.stderr)
            .chars()
            .take(512)
            .collect::<String>()
    );
    let id = String::from_utf8(created.stdout)?.trim().to_owned();
    ensure!(
        id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
        "created immutable container ID invalid"
    );
    receipt.container_id = Some(id.clone());
    let before = call(
        cfg,
        &["inspect", &id],
        "created-inspect",
        prefix,
        receipt,
        remaining(startup, STARTUP_SECS)?,
    )
    .await?;
    let before = inspected(&before)?;
    ensure!(
        ownership.verify_inspection(&before)? == id,
        "created ownership changed"
    );
    ownership.bind_container(&id).await?;
    ensure!(
        before["Config"]["Timeout"] == 80,
        "independent outer startup/crash ceiling differs from80s"
    );
    let stdout = stream::PreparedStream::new(audit_path.into(), AUDIT_CAP).await?;
    let stderr =
        stream::PreparedStream::new(prefix.with_extension("runtime.stderr"), STDERR_CAP).await?;
    let mut command = Command::new(&cfg.runtime);
    command.args(["start", "--attach", "--interactive", &id]);
    command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let attached = Instant::now();
    receipt.attached_launch_unix_ms = Some(crate::ledger::unix_millis());
    let mut child = command.spawn()?;
    let mut stdin = child.stdin.take().context("attached stdin absent")?;
    let mut stdout = stdout.attach(child.stdout.take().context("attached stdout absent")?);
    let mut stderr = stderr.attach(child.stderr.take().context("attached stderr absent")?);
    let setup:Result<control::PinnedCgroup>=async {
    let mut attempt = 0_u64;
    let running = loop {
        let observed = call(
            cfg,
            &["inspect", &id],
            &format!("gated-inspect-{attempt}"),
            prefix,
            receipt,
            remaining(startup, STARTUP_SECS)?,
        )
        .await?;
        let observed = inspected(&observed)?;
        ensure!(
            ownership.verify_inspection(&observed)? == id,
            "gated ownership changed"
        );
        if observed["State"]["Running"] == true {
            break observed;
        }
        ensure!(
            !matches!(
                observed["State"]["Status"].as_str(),
                Some("exited" | "stopped" | "dead")
            ),
            "native ceiling or startup terminated trusted supervisor before candidate input release"
        );
        attempt += 1;
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    let (path, process_binding) = cgroup_from_inspection(&running)?;
    receipt.gated_inspection = Some(running);
    receipt.process_binding = Some(process_binding);
    let control = tokio::task::spawn_blocking({
        let id = id.clone();
        move || control::PinnedCgroup::open(&id, &path)
    })
    .await??;
    Ok(control)
    }.await;
    let control = match setup {
        Ok(control) => control,
        Err(error) => {
            // Retain both the attached client and its open stdin while stopping
            // the verified container; dropping this gate would otherwise send
            // EOF and could begin compilation during failed startup cleanup.
            let stopped = super::stop_container(cfg, &id).await;
            let _ = timeout(Duration::from_secs(CLEANUP_SECS), child.wait()).await;
            receipt.audit = Some(stdout.finish(5).await?);
            receipt.stderr = Some(stderr.finish(5).await?);
            return Err(error.context(format!(
                "gated setup failed; cleanup acknowledged={}, removed={}",
                stopped.kill_acknowledged, stopped.container_removed
            )));
        }
    };
    receipt.cgroup = Some(control.binding.clone());
    // The trusted PID1 supervisor still waits for EOF: neither rustc nor
    // candidate code has run. Its independent 20s watchdog arms at EOF.
    let released = Instant::now();
    receipt.input_release_unix_ms = Some(crate::ledger::unix_millis());
    receipt.input_release_after_attached_launch_ms = Some(attached.elapsed().as_millis() as u64);
    let (cancel, mut deadline) = guard(control, released, cfg.wall_timeout_secs)?;
    stdin.write_all(&input).await?;
    drop(stdin);
    let mut cli_status = None;
    let (mut stdout_open, mut stderr_open) = (true, true);
    let mut guard_result = loop {
        tokio::select! {
            result=child.wait()=>{cli_status=Some(result?);let _=cancel.send(GuardSignal::Finished);break deadline.await.context("cached cgroup guard vanished")?;}
            result=&mut deadline=>{break result.context("cached cgroup deadline worker vanished")?;}
            changed=stdout.state.changed(),if stdout_open=>{
                let snapshot=stdout.state.borrow().clone();
                match stream::pipe_event(changed.is_err(),&snapshot) {
                    stream::PipeEvent::Failure=>{let _=cancel.send(GuardSignal::Stop("stdout_overflow_or_io_failure"));break deadline.await.context("cached stop worker vanished")?;}
                    stream::PipeEvent::HealthyEof=>stdout_open=false,
                    stream::PipeEvent::Progress=>{}
                }
            }
            changed=stderr.state.changed(),if stderr_open=>{
                let snapshot=stderr.state.borrow().clone();
                match stream::pipe_event(changed.is_err(),&snapshot) {
                    stream::PipeEvent::Failure=>{let _=cancel.send(GuardSignal::Stop("stderr_overflow_or_io_failure"));break deadline.await.context("cached stop worker vanished")?;}
                    stream::PipeEvent::HealthyEof=>stderr_open=false,
                    stream::PipeEvent::Progress=>{}
                }
            }
        }
    };
    receipt.kill = guard_result.kill.take();
    receipt.guard_stop_reason = Some(guard_result.reason.into());
    let cleanup = Instant::now();
    if cli_status.is_none() {
        cli_status = match timeout(Duration::from_secs(CLEANUP_SECS), child.wait()).await {
            Ok(status) => Some(status?),
            Err(_) => {
                let _ = child.start_kill();
                let _ = timeout(Duration::from_secs(5), child.wait()).await;
                receipt.failure = Some("runtime CLI cleanup deadline exceeded".into());
                None
            }
        };
    }
    let stdout = stdout.finish(5).await?;
    let stderr = stderr.finish(5).await?;
    receipt.audit = Some(stdout.clone());
    receipt.stderr = Some(stderr.clone());
    let terminal = call(
        cfg,
        &["inspect", &id],
        "terminal-inspect",
        prefix,
        receipt,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    let terminal = inspected(&terminal)?;
    ensure!(
        ownership.verify_inspection(&terminal)? == id,
        "terminal ownership changed"
    );
    receipt.terminal_inspection = Some(terminal.clone());
    if guard_result.control.populated()? {
        let killed =
            tokio::task::spawn_blocking(move || guard_result.control.kill(released)).await?;
        if receipt.kill.is_none() {
            receipt.kill = Some(killed);
        }
        receipt
            .failure
            .get_or_insert("runtime CLI ended while exact owned cgroup remained populated".into());
    }
    let removed = call(
        cfg,
        &["rm", "--force", &id],
        "remove",
        prefix,
        receipt,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    ensure!(
        removed.status.success(),
        "owned immutable container removal failed"
    );
    let absent = call(
        cfg,
        &["inspect", &id],
        "absence",
        prefix,
        receipt,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    let absence_text = String::from_utf8_lossy(&absent.stderr).to_ascii_lowercase();
    receipt.container_removed = !absent.status.success()
        && [
            "no such container",
            "no such object",
            "does not exist",
            "no container with",
        ]
        .iter()
        .any(|text| absence_text.contains(text));
    ensure!(
        receipt.container_removed,
        "owned removal was not independently established"
    );
    ownership.finish_verified_absence().await?;
    ensure!(
        !stdout.overflow
            && !stderr.overflow
            && stdout.eof
            && stderr.eof
            && stdout.error.is_none()
            && stderr.error.is_none(),
        "runtime audit pipe incomplete, oversized or errored"
    );
    let audit = tokio::fs::read(&stdout.path).await?;
    let runtime_stderr = tokio::fs::read(&stderr.path).await?;
    let marker = eof_marker(
        &runtime_stderr,
        cfg,
        receipt
            .input_release_unix_ms
            .context("input release absent")?,
    )?;
    receipt.eof_timer_metadata = Some(marker.clone());
    let actual_timeout = receipt.kill.as_ref().filter(|kill| {
        receipt.guard_stop_reason.as_deref() == Some("execution_wall_deadline")
            && kill.populated_before
            && kill.kill_written
            && kill.empty_observed
            && kill.failure.is_none()
            && kill.elapsed_from_release_ms >= cfg.wall_timeout_secs * 1000
    });
    if let Some(kill) = actual_timeout {
        receipt.classification = "candidate_wall_deadline".into();
        return Ok((
            None,
            Some(TimeoutEvidence {
                deadline_ms: cfg.wall_timeout_secs * 1000,
                wait_elapsed_ms: kill.elapsed_from_release_ms,
                kill_request_ms: kill.kill_write_ms,
                kill_acknowledged: true,
                cleanup_ms: cleanup.elapsed().as_millis() as u64,
                container_removed: true,
                kill_method: "preopened_exact_owned_cgroup_v2_kill".into(),
                kernel_proof: None,
            }),
        ));
    }
    let audit_value: Value =
        serde_json::from_slice(&audit).context("trusted terminal audit missing or partial")?;
    audit_timer_metadata(&audit_value, &marker, cfg)?;
    if audit_value["deadline_exceeded"] == true {
        let elapsed = audit_value["deadline_elapsed_ms"]
            .as_u64()
            .context("watchdog elapsed missing")?;
        ensure!(
            cfg.wall_timeout_secs == 20
                && elapsed >= 20000
                && terminal["State"]["Running"] == false
                && terminal["State"]["ExitCode"] == 124
                && audit_value["failure"] == "trusted EOF-armed candidate deadline exceeded",
            "independent watchdog envelope lacks matching namespace termination"
        );
        receipt.pid1_watchdog = Some(audit_value);
        receipt.classification = "trusted_pid1_candidate_deadline".into();
        return Ok((
            None,
            Some(TimeoutEvidence {
                deadline_ms: 20000,
                wait_elapsed_ms: elapsed,
                kill_request_ms: 0,
                kill_acknowledged: false,
                cleanup_ms: cleanup.elapsed().as_millis() as u64,
                container_removed: true,
                kill_method: "trusted_pid1_eof_armed_watchdog_exit124".into(),
                kernel_proof: None,
            }),
        ));
    }
    ensure!(
        audit_value["deadline_exceeded"] == false
            && audit_value["deadline_elapsed_ms"]
                .as_u64()
                .is_some_and(|elapsed| elapsed < 20000),
        "normal audit exceeded or omitted the candidate timer"
    );
    ensure!(
        receipt.failure.is_none(),
        "{}",
        receipt
            .failure
            .as_deref()
            .unwrap_or("unknown lifecycle failure")
    );
    ensure!(
        terminal["State"]["Running"] == false && terminal["State"]["ExitCode"] == 0,
        "trusted supervisor terminated without a normal complete audit"
    );
    let cli = cli_status.context("attached CLI status missing")?;
    ensure!(
        cli.success(),
        "attached runtime CLI failed after supervisor exit"
    );
    receipt.classification = "completed_supervisor_audit".into();
    Ok((
        Some(Output {
            status: cli,
            stdout: audit,
            stderr: runtime_stderr,
        }),
        None,
    ))
}

pub(super) async fn build_cached_image(root: &Path, cfg: &SandboxConfig) -> Result<String> {
    build::build_cached_image(root, cfg).await
}
pub(super) async fn cached_timeout_probe(
    cfg: &SandboxConfig,
    name: &str,
    command: Command,
    ownership: &mut recovery::OwnedRun,
) -> Result<TimeoutEvidence> {
    probe::cached_timeout_probe(cfg, name, command, ownership).await
}
pub(super) fn verify_prepared_binding(
    prepared: &PreparedGrade,
    report: &SandboxReport,
) -> Result<()> {
    verification::verify_prepared_binding(prepared, report)
}
pub(super) fn verify(report: &SandboxReport, cfg: &SandboxConfig) -> Result<bool> {
    verification::verify(report, cfg)
}
