//! Container execution and host-owned grading. Candidate output is evidence to
//! inspect, never a verdict. Every required private response stays in the denominator.
use crate::tasks::{self, Task};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Instant,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
    time::{Duration, timeout},
};
mod bundle;
mod control;
mod owned;
mod prepared;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../sandbox/ptrace_v2.rs"]
mod ptrace_v2;
mod recovery;
mod runtime_lease;
mod stream;
mod v3;
pub use prepared::{
    PreparedGrade, execute_prepared, prepare_grade, verify_prepared, verify_prepared_report,
};
pub use v3::LifecycleEvidence;
#[cfg(test)]
#[allow(dead_code)]
#[path = "../sandbox/syscalls_v2.rs"]
mod syscalls_v2;
pub use bundle::restore_image_bundle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Absent in historical reports; V1 always retains its original identity.
    #[serde(
        default = "legacy_policy_version",
        skip_serializing_if = "is_legacy_policy"
    )]
    pub policy_version: u32,
    /// Historical reports omit this field. Current host supervision adds an
    /// independent runtime ceiling and ownership-bound crash recovery.
    #[serde(
        default = "legacy_policy_version",
        skip_serializing_if = "is_legacy_policy"
    )]
    pub controller_version: u32,
    /// Explicit ABI binding for portable V2; omitted from original V1 reports.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub architecture: String,
    pub runtime: String,
    pub image: String,
    pub expected_image_id: String,
    pub wall_timeout_secs: u64,
    pub memory_mb: u64,
    pub pids_limit: u32,
    pub cpu_limit: f64,
    pub report_dir: PathBuf,
}
impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            policy_version: 3,
            controller_version: 3,
            architecture: std::env::consts::ARCH.into(),
            runtime: "podman".into(),
            image: format!("localhost/lec-rust-sandbox:3-{}", std::env::consts::ARCH),
            expected_image_id: String::new(),
            wall_timeout_secs: 20,
            memory_mb: 512,
            pids_limit: 64,
            cpu_limit: 1.0,
            report_dir: "reports/sandbox".into(),
        }
    }
}
fn legacy_policy_version() -> u32 {
    1
}
fn is_legacy_policy(version: &u32) -> bool {
    *version == 1
}
impl SandboxConfig {
    /// Select the current native Linux x86-64/AArch64 security policy.
    pub fn portable() -> Self {
        Self {
            policy_version: 3,
            architecture: std::env::consts::ARCH.into(),
            image: format!("localhost/lec-rust-sandbox:3-{}", std::env::consts::ARCH),
            ..Self::default()
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditEvent {
    pub syscall: u64,
    pub category: String,
    pub target: String,
    pub result: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxReport {
    pub run_id: String,
    pub task_id: String,
    pub passed: bool,
    pub quarantined: bool,
    pub quarantine_reasons: Vec<String>,
    pub cases_total: usize,
    pub cases_passed: usize,
    pub trace_complete: bool,
    pub policy_sha256: String,
    pub image_id: String,
    pub oracle_sha256: String,
    pub code_sha256: String,
    pub stdout_sha256: String,
    pub sandbox_config: SandboxConfig,
    pub audit_events: Vec<AuditEvent>,
    pub audit_path: PathBuf,
    pub audit_sha256: String,
    pub failure: Option<String>,
    pub duration_ms: u64,
    pub timeout_evidence: Option<TimeoutEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_evidence: Option<LifecycleEvidence>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutEvidence {
    pub deadline_ms: u64,
    pub wait_elapsed_ms: u64,
    pub kill_request_ms: u64,
    pub kill_acknowledged: bool,
    pub cleanup_ms: u64,
    pub container_removed: bool,
    pub kill_method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_proof: Option<LifecycleEvidence>,
}
struct StopEvidence {
    kill_request_ms: u64,
    kill_acknowledged: bool,
    cleanup_ms: u64,
    container_removed: bool,
    kill_method: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflightReport {
    pub image_id: String,
    pub policy_sha256: String,
    pub runtime: String,
    pub controls: Value,
    pub probes_passed: bool,
}
#[derive(Debug, Serialize, Deserialize)]
struct AuditOutput {
    schema_version: u32,
    #[serde(default)]
    architecture: String,
    trace_complete: bool,
    compile_ok: bool,
    candidate_exit: Option<i32>,
    output: String,
    stderr: String,
    events: Vec<AuditEvent>,
    #[serde(default)]
    syscalls_observed: usize,
    failure: Option<String>,
    #[serde(default)]
    output_truncated: bool,
    #[serde(default)]
    stderr_truncated: bool,
}

fn seccomp_path(cfg: &SandboxConfig) -> Result<PathBuf> {
    std::env::current_dir()?
        .join(if cfg.policy_version == 1 {
            "sandbox/policies/v1/seccomp.json"
        } else if cfg.policy_version == 3 {
            "sandbox/policies/v3/seccomp.json"
        } else {
            "sandbox/policies/v2/seccomp.json"
        })
        .canonicalize()
        .context("bundled seccomp policy missing")
}
pub fn policy_sha256(cfg: &SandboxConfig) -> Result<String> {
    ensure!(
        cfg.wall_timeout_secs > 0
            && cfg.wall_timeout_secs <= 120
            && cfg.memory_mb >= 64
            && cfg.memory_mb <= 4096
            && cfg.pids_limit >= 8
            && cfg.pids_limit <= 256
            && cfg.cpu_limit > 0.0
            && cfg.cpu_limit <= 4.0,
        "invalid sandbox resource policy"
    );
    ensure!(
        matches!(cfg.policy_version, 1..=3),
        "unsupported sandbox policy version"
    );
    ensure!(
        matches!(cfg.controller_version, 1..=3),
        "unsupported sandbox host controller version"
    );
    if cfg.controller_version == 3 {
        ensure!(
            cfg.wall_timeout_secs <= 20,
            "host controller three cannot exceed the trusted20s candidate watchdog"
        );
        let mut historical = cfg.clone();
        historical.controller_version = 2;
        if historical.policy_version == 3 {
            historical.policy_version = 2;
        }
        return Ok(tasks::hash(serde_json::to_vec(
            &json!({"candidate_policy_sha256":policy_sha256(&historical)?,
            "host_controller":v3::policy_sources()}),
        )?));
    }
    if cfg.policy_version == 2 {
        ensure!(
            matches!(cfg.architecture.as_str(), "x86_64" | "aarch64"),
            "unsupported portable sandbox architecture"
        );
        return Ok(tasks::hash(serde_json::to_vec(&json!({
            "version":2,"controller_version":cfg.controller_version,"architecture":cfg.architecture,"runtime":cfg.runtime,
            "wall_timeout_secs":cfg.wall_timeout_secs,"memory_mb":cfg.memory_mb,
            "pids_limit":cfg.pids_limit,"cpu_limit":cfg.cpu_limit,"nonroot":65532,
            "root_readonly":true,"network":"none","caps":"none","no_new_privileges":true,
            "private_pid":true,"tmpfs_mb":128,"wrapper":"json-lines-v1",
            "seccomp_sha256":tasks::hash(fs::read(seccomp_path(cfg)?)?),
            "supervisor_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/audit.rs")),
            "syscall_abi_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/syscalls.rs")),
            "ptrace_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/ptrace.rs")),
            "containerfile_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/Containerfile")),
            "probe_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/probe.rs")),
            "dependency_manifest_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/Cargo.toml")),
            "dependency_lock_sha256":tasks::hash(include_bytes!("../sandbox/policies/v2/Cargo.lock")),
            "controller_sha256":tasks::hash(include_bytes!("../sandbox/policies/controller-v2/controller.rs")),
            "cgroup_control_sha256":tasks::hash(include_bytes!("../sandbox/policies/controller-v2/control.rs")),
            "runtime_lease_sha256":tasks::hash(include_bytes!("../sandbox/policies/controller-v2/runtime_lease.rs")),
            "owned_operation_sha256":tasks::hash(include_bytes!("../sandbox/policies/controller-v2/owned.rs")),
            "ownership_recovery_sha256":tasks::hash(include_bytes!("../sandbox/policies/controller-v2/recovery.rs"))
        }))?));
    }
    ensure!(
        cfg.architecture.is_empty(),
        "legacy sandbox architecture field must be absent"
    );
    let mut policy = json!({"version":1,"runtime":cfg.runtime,"wall_timeout_secs":cfg.wall_timeout_secs,"memory_mb":cfg.memory_mb,"pids_limit":cfg.pids_limit,"cpu_limit":cfg.cpu_limit,"nonroot":65532,"root_readonly":true,"network":"none","caps":"none","no_new_privileges":true,"private_pid":true,"tmpfs_mb":128,"seccomp_sha256":tasks::hash(fs::read(seccomp_path(cfg)?)?),"supervisor_sha256":tasks::hash(include_bytes!("../sandbox/policies/v1/audit.rs")),"controller_sha256":tasks::hash(include_bytes!("../sandbox/policies/v1/controller.rs")),"cgroup_control_sha256":tasks::hash(include_bytes!("../sandbox/policies/v1/control.rs")),"wrapper":"json-lines-v1"});
    if cfg.controller_version == 2 {
        policy["controller_version"] = json!(2);
        policy["controller_sha256"] = json!(tasks::hash(include_bytes!(
            "../sandbox/policies/controller-v2/controller.rs"
        )));
        policy["runtime_lease_sha256"] = json!(tasks::hash(include_bytes!(
            "../sandbox/policies/controller-v2/runtime_lease.rs"
        )));
        policy["owned_operation_sha256"] = json!(tasks::hash(include_bytes!(
            "../sandbox/policies/controller-v2/owned.rs"
        )));
        policy["ownership_recovery_sha256"] = json!(tasks::hash(include_bytes!(
            "../sandbox/policies/controller-v2/recovery.rs"
        )));
        policy["independent_runtime_wall_seconds"] = json!(cfg.wall_timeout_secs);
    }
    Ok(tasks::hash(serde_json::to_vec(&policy)?))
}
pub async fn policy_evidence(cfg: &SandboxConfig) -> Result<(String, String)> {
    let output = timeout(
        Duration::from_secs(if cfg.controller_version == 3 {
            v3::METADATA_SECS
        } else {
            15
        }),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["image", "inspect", "--format", "{{.Id}}", &cfg.image])
            .output(),
    )
    .await
    .context("image inspection timeout")??;
    ensure!(
        output.status.success(),
        "sandbox image unavailable: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image_id = String::from_utf8(output.stdout)?.trim().to_string();
    ensure!(image_id.len() >= 64, "invalid immutable image ID");
    Ok((policy_sha256(cfg)?, image_id))
}
pub async fn build_image(root: &Path, runtime: &str) -> Result<String> {
    let cfg = SandboxConfig {
        runtime: runtime.into(),
        ..Default::default()
    };
    build_image_for(root, &cfg).await
}
pub async fn build_image_for(root: &Path, cfg: &SandboxConfig) -> Result<String> {
    validate_execution_platform(cfg)?;
    let _runtime_lease = runtime_lease::exclusive(&cfg.runtime).await?;
    if cfg.policy_version == 3 {
        return v3::build_cached_image(root, cfg).await;
    }
    if cfg.policy_version == 1 {
        ensure!(
            fs::read(root.join("sandbox/audit.rs"))?
                == include_bytes!("../sandbox/policies/v1/audit.rs"),
            "legacy image source changed; build portable V2 rather than relabelling it V1"
        );
    } else {
        validate_portable_build_sources(root)?;
    }
    // This fixed public image needs no registry credential. Podman otherwise
    // consults unrelated host login state even for an anonymous public pull.
    let anonymous_auth = tempfile::NamedTempFile::new()?;
    fs::write(anonymous_auth.path(), b"{}")?;
    let status = Command::new(&cfg.runtime)
        .kill_on_drop(true)
        .env("REGISTRY_AUTH_FILE", anonymous_auth.path())
        .args([
            "build",
            "--pull=missing",
            "-f",
            if cfg.policy_version == 1 {
                "sandbox/Containerfile"
            } else {
                "sandbox/Containerfile.v2"
            },
            "-t",
            &cfg.image,
            ".",
        ])
        .current_dir(root)
        .status()
        .await?;
    ensure!(status.success(), "explicit online sandbox bootstrap failed");
    Ok(policy_evidence(cfg).await?.1)
}

/// Building a new image must use the exact reviewed policy source bytes. Old
/// reports verify against immutable snapshots even after a later policy exists.
fn validate_portable_build_sources(root: &Path) -> Result<()> {
    for (path, expected) in [
        (
            "sandbox/audit_v2.rs",
            include_bytes!("../sandbox/policies/v2/audit.rs").as_slice(),
        ),
        (
            "sandbox/syscalls_v2.rs",
            include_bytes!("../sandbox/policies/v2/syscalls.rs").as_slice(),
        ),
        (
            "sandbox/ptrace_v2.rs",
            include_bytes!("../sandbox/policies/v2/ptrace.rs").as_slice(),
        ),
        (
            "sandbox/Containerfile.v2",
            include_bytes!("../sandbox/policies/v2/Containerfile").as_slice(),
        ),
        (
            "sandbox/probe.rs",
            include_bytes!("../sandbox/policies/v2/probe.rs").as_slice(),
        ),
        (
            "sandbox/deps/Cargo.toml",
            include_bytes!("../sandbox/policies/v2/Cargo.toml").as_slice(),
        ),
        (
            "sandbox/deps/Cargo.lock",
            include_bytes!("../sandbox/policies/v2/Cargo.lock").as_slice(),
        ),
        (
            "sandbox/seccomp_v2.json",
            include_bytes!("../sandbox/policies/v2/seccomp.json").as_slice(),
        ),
    ] {
        ensure!(
            fs::read(root.join(path))? == expected,
            "portable image source {path} changed; introduce a new policy version"
        );
    }
    Ok(())
}
fn validate_execution_platform(cfg: &SandboxConfig) -> Result<()> {
    ensure!(
        cfg.controller_version == 3,
        "historical sandbox controller is verification-only; execute a fresh protocol with current host supervision"
    );
    ensure!(
        cfg.policy_version == 3 && cfg.wall_timeout_secs <= 20,
        "new execution requires policy three and a candidate ceiling no greater than20s"
    );
    ensure!(
        cfg!(target_os = "linux"),
        "trusted syscall decoder requires native Linux"
    );
    match cfg.policy_version {
        1 => ensure!(
            cfg!(target_arch = "x86_64"),
            "legacy V1 syscall decoder supports Linux x86_64 only; use portable V2"
        ),
        2 | 3 => ensure!(
            cfg.architecture == std::env::consts::ARCH
                && matches!(cfg.architecture.as_str(), "x86_64" | "aarch64"),
            "portable V2 requires the configured native Linux ABI"
        ),
        _ => bail!("unsupported sandbox policy version"),
    }
    Ok(())
}
fn audit_identity_matches(audit: &AuditOutput, cfg: &SandboxConfig) -> bool {
    audit.schema_version == cfg.policy_version
        && if cfg.policy_version == 1 {
            audit.architecture.is_empty()
        } else {
            audit.architecture == cfg.architecture
        }
}
fn base_command(cfg: &SandboxConfig, name: &str) -> Result<Command> {
    base_command_kind(cfg, name, false)
}
fn base_command_kind(cfg: &SandboxConfig, name: &str, create: bool) -> Result<Command> {
    base_command_operation(cfg, name, create, !create)
}
fn base_command_operation(
    cfg: &SandboxConfig,
    name: &str,
    create: bool,
    remove: bool,
) -> Result<Command> {
    validate_execution_platform(cfg)?;
    let seccomp = seccomp_path(cfg)?;
    let mut c = Command::new(&cfg.runtime);
    c.arg(if create { "create" } else { "run" });
    if remove {
        c.arg("--rm");
    }
    c.args([
        "--pull=never",
        "--name",
        name,
        "--read-only",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        &format!("seccomp={}", seccomp.display()),
        "--user",
        "65532:65532",
        "--pids-limit",
        &cfg.pids_limit.to_string(),
        "--memory",
        &format!("{}m", cfg.memory_mb),
        "--memory-swap",
        &format!("{}m", cfg.memory_mb),
        "--cpus",
        &cfg.cpu_limit.to_string(),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=128m,mode=1777",
        "--ipc",
        "private",
        "--pid",
        "private",
        "--cgroupns",
        "private",
        "--timeout",
        &(if cfg.policy_version == 3 {
            80
        } else {
            cfg.wall_timeout_secs
        })
        .to_string(),
    ]);
    c.kill_on_drop(true);
    Ok(c)
}
async fn stop_container(cfg: &SandboxConfig, name: &str) -> StopEvidence {
    // rm --force may still apply the runtime's stop grace period. Send SIGKILL
    // explicitly first so a candidate cannot consume that extra CPU time.
    let kill_start = Instant::now();
    let direct = control::kill_delegated_container_cgroup(cfg, name)
        .await
        .unwrap_or(false);
    let killed = if direct {
        None
    } else {
        Some(
            timeout(
                Duration::from_secs(5),
                Command::new(&cfg.runtime)
                    .kill_on_drop(true)
                    .args(["kill", "--signal", "KILL", name])
                    .output(),
            )
            .await,
        )
    };
    let kill_request_ms = kill_start.elapsed().as_millis() as u64;
    let kill_acknowledged =
        direct || matches!(killed,Some(Ok(Ok(ref output)))if output.status.success());
    let cleanup_start = Instant::now();
    let _ = timeout(
        Duration::from_secs(30),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["rm", "-f", name])
            .output(),
    )
    .await;
    let inspected = timeout(
        Duration::from_secs(10),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["inspect", name])
            .output(),
    )
    .await;
    let container_removed = matches!(inspected,Ok(Ok(ref output))if !output.status.success());
    StopEvidence {
        kill_request_ms,
        kill_acknowledged,
        cleanup_ms: cleanup_start.elapsed().as_millis() as u64,
        container_removed,
        kill_method: if direct {
            "cgroup_v2_kill".into()
        } else {
            "runtime_sigkill".into()
        },
    }
}
fn image_reference(cfg: &SandboxConfig) -> &str {
    if cfg.expected_image_id.is_empty() {
        &cfg.image
    } else {
        &cfg.expected_image_id
    }
}
async fn probe(cfg: &SandboxConfig, mode: &str, seconds: u64) -> Result<std::process::Output> {
    let name = format!("lec-probe-{}", uuid::Uuid::new_v4());
    let mut command = base_command(cfg, &name)?;
    let mut ownership = plan_owned_run(cfg, &name).await?;
    add_ownership_labels(&mut command, &ownership);
    command.args([
        "--entrypoint",
        "/usr/local/bin/lec-probe",
        image_reference(cfg),
        mode,
    ]);
    let result = match timeout(Duration::from_secs(seconds), command.output()).await {
        Ok(output) => output.context("trusted probe runtime failed"),
        Err(_) => {
            stop_container(cfg, &name).await;
            Err(anyhow::anyhow!("host deadline terminated probe {mode}"))
        }
    };
    ownership.finish().await?;
    result
}
async fn plan_owned_run(cfg: &SandboxConfig, name: &str) -> Result<recovery::OwnedRun> {
    let (cfg, name) = (cfg.clone(), name.to_owned());
    tokio::task::spawn_blocking(move || recovery::plan(&cfg, &name)).await?
}
fn add_ownership_labels(command: &mut Command, ownership: &recovery::OwnedRun) {
    for (key, value) in ownership.labels() {
        command.args(["--label", &format!("{key}={value}")]);
    }
}

async fn timeout_probe(cfg: &SandboxConfig) -> Result<TimeoutEvidence> {
    let name = format!("lec-deadline-{}", uuid::Uuid::new_v4());
    let mut command = base_command_operation(cfg, &name, false, false)?;
    let mut ownership = plan_owned_run(cfg, &name).await?;
    add_ownership_labels(&mut command, &ownership);
    command
        .args([
            "--entrypoint",
            "/usr/local/bin/lec-probe",
            image_reference(cfg),
            "timeout",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let result = if cfg.controller_version == 3 {
        v3::cached_timeout_probe(cfg, &name, command, &mut ownership).await
    } else {
        timeout_probe_execution(cfg, &name, command).await
    };
    ownership.finish().await?;
    result
}
async fn timeout_probe_execution(
    cfg: &SandboxConfig,
    name: &str,
    mut command: Command,
) -> Result<TimeoutEvidence> {
    let mut child = command.spawn()?;
    let output = child.stdout.take().context("timeout stdout missing")?;
    let mut reader = BufReader::new(output);
    let mut marker = String::new();
    let started = timeout(Duration::from_secs(20), reader.read_line(&mut marker)).await;
    if !matches!(started,Ok(Ok(n)) if n>0) || marker.trim() != "trusted_timeout_probe_started" {
        stop_container(cfg, name).await;
        bail!("timeout probe did not demonstrably start");
    }
    let wait_start = Instant::now();
    if timeout(Duration::from_secs(1), child.wait()).await.is_ok() {
        stop_container(cfg, name).await;
        bail!("timeout probe ended before the host deadline");
    }
    let wait_elapsed_ms = wait_start.elapsed().as_millis() as u64;
    let stopped = stop_container(cfg, name).await;
    ensure!(
        stopped.kill_acknowledged && stopped.kill_request_ms <= 5000 && stopped.container_removed,
        "deadline kill/removal was not acknowledged within its bounded control latency"
    );
    // Native rootless runtimes can finish their --rm bookkeeping just after
    // the kill command. Reap the launcher before asserting container removal.
    let _ = timeout(Duration::from_secs(10), child.wait()).await;
    let status = timeout(
        Duration::from_secs(10),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["inspect", name])
            .output(),
    )
    .await
    .context("post-cleanup container inspection deadline exceeded")??;
    ensure!(!status.status.success(), "timed-out container still exists");
    Ok(TimeoutEvidence {
        deadline_ms: 1000,
        wait_elapsed_ms,
        kill_request_ms: stopped.kill_request_ms,
        kill_acknowledged: stopped.kill_acknowledged,
        cleanup_ms: stopped.cleanup_ms,
        container_removed: stopped.container_removed,
        kill_method: stopped.kill_method,
        kernel_proof: None,
    })
}

pub async fn preflight(cfg: &SandboxConfig) -> Result<PreflightReport> {
    validate_execution_platform(cfg)?;
    let runtime_lease = runtime_lease::exclusive(&cfg.runtime).await?;
    ensure!(
        runtime_lease.supports_independent_wall_timeout(),
        "independent container ceiling currently requires native Podman"
    );
    let config = cfg.clone();
    owned::run(runtime_lease, async move { preflight_inner(&config).await }).await
}
async fn preflight_inner(cfg: &SandboxConfig) -> Result<PreflightReport> {
    recovery::recover(cfg).await?;
    let (policy, image) = policy_evidence(cfg).await?;
    let mut frozen = cfg.clone();
    frozen.expected_image_id = image.clone();
    let cfg = &frozen;
    let settings = probe(cfg, "settings", 15).await?;
    ensure!(
        settings.status.success(),
        "kernel settings probe failed: {}",
        String::from_utf8_lossy(&settings.stderr)
    );
    let controls: Value =
        serde_json::from_slice(&settings.stdout).context("kernel control report is malformed")?;
    ensure!(
        controls["uid_status"]
            .as_str()
            .unwrap_or("")
            .split_whitespace()
            .nth(1)
            == Some("65532"),
        "non-root UID is not enforced"
    );
    ensure!(
        controls["memory_max"].as_str() == Some(&(cfg.memory_mb * 1024 * 1024).to_string()),
        "memory controller is not enforced"
    );
    ensure!(
        controls["swap_max"] == "0",
        "swap controller is not enforced"
    );
    ensure!(
        controls["pids_max"].as_str() == Some(&cfg.pids_limit.to_string()),
        "PID controller is not enforced"
    );
    let cpu = controls["cpu_max"]
        .as_str()
        .context("CPU controller missing")?
        .split_whitespace()
        .collect::<Vec<_>>();
    ensure!(
        cpu.len() == 2 && cpu[0] != "max",
        "CPU quota is not enforced"
    );
    let ratio = cpu[0].parse::<f64>()? / cpu[1].parse::<f64>()?;
    ensure!((ratio - cfg.cpu_limit).abs() < 0.001, "CPU quota mismatch");
    ensure!(
        controls["root_write_denied"] == true && controls["network_none"] == true,
        "read-only root or network namespace probe failed"
    );
    let mut small = cfg.clone();
    small.memory_mb = 128;
    let memory = probe(&small, "memory", 15).await?;
    ensure!(
        memory.status.code() == Some(137),
        "memory pressure was not OOM-terminated: {:?}",
        memory.status.code()
    );
    let pids = probe(cfg, "pids", 15).await?;
    ensure!(pids.status.success(), "PID pressure probe failed");
    let p: Value = serde_json::from_slice(&pids.stdout)?;
    ensure!(
        p["creation_denied"] == true,
        "runaway child creation was not denied"
    );
    let timeout_evidence = timeout_probe(cfg).await?;
    // A genuine passing candidate proves ptrace and the JSON grading boundary
    // on this runtime, rather than assuming Docker-compatible flags suffice.
    let mut verified = cfg.clone();
    verified.expected_image_id = image.clone();
    let task = tasks::corpus().remove(0);
    let report=grade_inner(&task,"pub fn solve(input: serde_json::Value) -> serde_json::Value { serde_json::json!(input.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).sum::<i64>()) }",&verified).await?;
    ensure!(
        report.passed && report.trace_complete,
        "trusted audit/reference execution failed: {:?}",
        report.failure
    );
    Ok(PreflightReport {
        image_id: image,
        policy_sha256: policy,
        runtime: cfg.runtime.clone(),
        controls: json!({"effective":controls,"memory_pressure_exit":137,"pids":p,"host_timeout_asserted":true,"timeout_evidence":timeout_evidence,"reference_run_id":report.run_id}),
        probes_passed: true,
    })
}

pub async fn grade(task: &Task, code: &str, cfg: &SandboxConfig) -> Result<SandboxReport> {
    if cfg.controller_version == 3 {
        let task = task.clone();
        let code = code.to_owned();
        let cfg = cfg.clone();
        let prepared = tokio::task::spawn_blocking(move || {
            let attempt = uuid::Uuid::new_v4().to_string();
            let receipt = cfg.report_dir.join(format!("{attempt}.prepared.json"));
            prepare_grade(&task, &code, &cfg, &attempt, &receipt)
        })
        .await??;
        return execute_prepared(&prepared).await;
    }
    validate_execution_platform(cfg)?;
    let runtime_lease = runtime_lease::shared(&cfg.runtime).await?;
    ensure!(
        runtime_lease.supports_independent_wall_timeout(),
        "independent container ceiling currently requires native Podman"
    );
    let (task, code, config) = (task.clone(), code.to_owned(), cfg.clone());
    owned::run(runtime_lease, async move {
        recovery::recover(&config).await?;
        grade_inner(&task, &code, &config).await
    })
    .await
}
async fn grade_inner(task: &Task, code: &str, cfg: &SandboxConfig) -> Result<SandboxReport> {
    grade_identified(task, code, cfg, None).await
}
async fn grade_identified(
    task: &Task,
    code: &str,
    cfg: &SandboxConfig,
    identity: Option<(&str, &str)>,
) -> Result<SandboxReport> {
    let mut ownership = None;
    let result = grade_execution(task, code, cfg, &mut ownership, identity).await;
    if let Some(ownership) = &mut ownership {
        ownership.finish().await?;
    }
    result
}
async fn grade_execution(
    task: &Task,
    code: &str,
    cfg: &SandboxConfig,
    ownership: &mut Option<recovery::OwnedRun>,
    identity: Option<(&str, &str)>,
) -> Result<SandboxReport> {
    let start = Instant::now();
    let (policy, image) = policy_evidence(cfg).await?;
    ensure!(
        !cfg.expected_image_id.is_empty() && cfg.expected_image_id == image,
        "sandbox image must be frozen to its preflight immutable ID"
    );
    let mut cases: Vec<_> = task
        .public_examples
        .iter()
        .enumerate()
        .map(|(i, e)| tasks::Case {
            input: e.input.clone(),
            expected: e.expected.clone(),
            label: format!("public-{i}"),
        })
        .collect();
    cases.extend(tasks::private_cases(task)?);
    let run_id = identity
        .map(|(id, _)| id.to_owned())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    fs::create_dir_all(&cfg.report_dir)?;
    let audit_path = cfg.report_dir.join(format!("{run_id}.audit.json"));
    let mut report = SandboxReport {
        run_id: run_id.clone(),
        task_id: task.id.clone(),
        passed: false,
        quarantined: false,
        quarantine_reasons: vec![],
        cases_total: cases.len(),
        cases_passed: 0,
        trace_complete: false,
        policy_sha256: policy,
        image_id: image,
        oracle_sha256: task.oracle_sha256.clone(),
        code_sha256: tasks::hash(code),
        stdout_sha256: tasks::hash([]),
        sandbox_config: cfg.clone(),
        audit_events: vec![],
        audit_path: audit_path.clone(),
        audit_sha256: String::new(),
        failure: None,
        duration_ms: 0,
        timeout_evidence: None,
        lifecycle_evidence: None,
    };
    if code.trim().is_empty() || code.len() > 131072 {
        report.failure = Some("empty or oversized candidate source".into());
        report
            .quarantine_reasons
            .push("invalid candidate source".into());
    } else {
        let source = tempfile::tempdir()?;
        fs::set_permissions(source.path(), fs::Permissions::from_mode(0o755))?;
        let wrapper = format!(
            "{code}\nfn main() {{ let stdin = std::io::stdin(); for line in std::io::BufRead::lines(stdin.lock()) {{ let line = line.expect(\"input read\"); let input: serde_json::Value = serde_json::from_str(&line).expect(\"JSON input\"); let answer = solve(input); println!(\"{{}}\",serde_json::to_string(&answer).expect(\"JSON result\")); }} }}\n"
        );
        fs::write(source.path().join("main.rs"), wrapper)?;
        fs::set_permissions(
            source.path().join("main.rs"),
            fs::Permissions::from_mode(0o644),
        )?;
        let name = format!("lec-{run_id}");
        if cfg.controller_version == 3 {
            let owned_cfg = cfg.clone();
            let owned_name = name.clone();
            let nonce = identity.map(|(_, nonce)| nonce.to_owned());
            *ownership = Some(
                tokio::task::spawn_blocking(move || {
                    recovery::plan_with_nonce(&owned_cfg, &owned_name, nonce.as_deref())
                })
                .await??,
            );
            let mut input = vec![];
            for case in &cases {
                serde_json::to_writer(&mut input, &case.input)?;
                input.push(b'\n');
            }
            let execution = v3::execute(
                cfg,
                &name,
                &source.path().join("main.rs"),
                input,
                ownership.as_mut().context("owned execution absent")?,
                &audit_path,
                &report.code_sha256,
            )
            .await?;
            report.lifecycle_evidence = Some(execution.evidence);
            report.audit_sha256 = fs::read(&audit_path).map(tasks::hash).unwrap_or_default();
            if let Some(timing) = execution.timeout {
                report.timeout_evidence = Some(timing);
                let timing = report
                    .timeout_evidence
                    .as_mut()
                    .context("timeout timing missing")?;
                timing.kernel_proof = report.lifecycle_evidence.clone();
                report.failure=Some(if timing.kill_acknowledged {
                    "candidate wall-clock deadline exceeded; exact cached cgroup SIGKILL and empty group observed; owned container removed"
                }else{"candidate wall-clock deadline exceeded; trusted PID1 watchdog exited124; namespace termination and owned removal observed"}.into());
            } else if let Some(output) = execution.output {
                apply_audit_output(&mut report, cfg, &cases, &output)?;
            } else {
                report.failure = Some(
                    execution
                        .failure
                        .unwrap_or_else(|| "controller three infrastructure failure".into()),
                );
            }
        } else {
            let mut command = base_command(cfg, &name)?;
            *ownership = Some(plan_owned_run(cfg, &name).await?);
            add_ownership_labels(
                &mut command,
                ownership.as_ref().context("owned run plan absent")?,
            );
            command.args([
                "--volume",
                &format!("{}:/candidate:ro", source.path().display()),
                "-i",
                image_reference(cfg),
                "/candidate/main.rs",
            ]);
            command
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = command.spawn().context("sandbox runtime failed to start")?;
            let mut input = child.stdin.take().context("container stdin missing")?;
            let mut bytes = vec![];
            for case in &cases {
                serde_json::to_writer(&mut bytes, &case.input)?;
                bytes.push(b'\n');
            }
            ensure!(bytes.len() <= 131072, "case input cap exceeded");
            let writer = tokio::spawn(async move {
                input.write_all(&bytes).await?;
                drop(input);
                Ok::<_, std::io::Error>(())
            });
            let wait_start = Instant::now();
            let output = timeout(
                Duration::from_secs(cfg.wall_timeout_secs),
                child.wait_with_output(),
            )
            .await;
            match output {
                Err(_) => {
                    let wait_elapsed_ms = wait_start.elapsed().as_millis() as u64;
                    let stopped = stop_container(cfg, &name).await;
                    report.timeout_evidence = Some(TimeoutEvidence {
                        deadline_ms: cfg.wall_timeout_secs * 1000,
                        wait_elapsed_ms,
                        kill_request_ms: stopped.kill_request_ms,
                        kill_acknowledged: stopped.kill_acknowledged,
                        cleanup_ms: stopped.cleanup_ms,
                        container_removed: stopped.container_removed,
                        kill_method: stopped.kill_method,
                        kernel_proof: None,
                    });
                    report.failure = Some(
                        if stopped.kill_acknowledged && stopped.container_removed {
                            "host wall-clock deadline exceeded; SIGKILL acknowledged and container removed"
                        .into()
                        } else {
                            "host wall-clock deadline exceeded; cleanup/kill evidence incomplete"
                                .into()
                        },
                    );
                }
                Ok(Err(e)) => {
                    stop_container(cfg, &name).await;
                    report.failure = Some(format!("container I/O failed: {e}"));
                }
                Ok(Ok(output)) => {
                    fs::write(&audit_path, &output.stdout)?;
                    report.audit_sha256 = tasks::hash(&output.stdout);
                    if !output.status.success() {
                        report.failure = Some(format!(
                            "sandbox/supervisor exit {:?}: {}",
                            output.status.code(),
                            String::from_utf8_lossy(&output.stderr)
                                .chars()
                                .take(2048)
                                .collect::<String>()
                        ));
                    } else {
                        match serde_json::from_slice::<AuditOutput>(&output.stdout) {
                            Err(e) => {
                                report.failure =
                                    Some(format!("trusted audit result malformed: {e}"))
                            }
                            Ok(audit) => {
                                report.trace_complete = audit.trace_complete
                                    && audit.compile_ok
                                    && audit_identity_matches(&audit, cfg)
                                    && audit.syscalls_observed > 0;
                                report.stdout_sha256 = tasks::hash(audit.output.as_bytes());
                                report.audit_events = audit.events;
                                for event in &report.audit_events {
                                    report.quarantine_reasons.push(format!(
                                        "prohibited {} attempt: syscall {} target {}",
                                        event.category, event.syscall, event.target
                                    ));
                                }
                                if !audit.compile_ok {
                                    report.failure =
                                        Some(format!("compilation failed: {}", audit.stderr));
                                } else if !report.trace_complete {
                                    report.failure =
                                        Some("incomplete trusted syscall trace".into());
                                } else if audit.output_truncated || audit.stderr_truncated {
                                    report.failure = Some("candidate output cap exceeded".into());
                                } else if audit.candidate_exit != Some(0) {
                                    report.failure = Some(format!(
                                        "candidate exit {:?}: {}",
                                        audit.candidate_exit,
                                        audit.stderr.chars().take(1024).collect::<String>()
                                    ));
                                } else {
                                    let lines: Vec<_> = audit.output.lines().collect();
                                    if lines.len() != cases.len() {
                                        report.failure = Some(format!(
                                            "expected {} responses, observed {}",
                                            cases.len(),
                                            lines.len()
                                        ));
                                    } else {
                                        for (i, (line, case)) in
                                            lines.iter().zip(&cases).enumerate()
                                        {
                                            match serde_json::from_str::<Value>(line) {
                                                Ok(value) if value == case.expected => {
                                                    report.cases_passed += 1
                                                }
                                                Ok(_) => {
                                                    if report.failure.is_none() {
                                                        report.failure = Some(format!(
                                                            "host oracle mismatch at case {i}"
                                                        ));
                                                    }
                                                }
                                                Err(e) => {
                                                    if report.failure.is_none() {
                                                        report.failure = Some(format!(
                                                            "invalid JSON at case {i}: {e}"
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            writer.abort();
        }
    }
    if report.failure.is_some() {
        report
            .quarantine_reasons
            .push(report.failure.clone().unwrap_or_default());
    }
    report.quarantined = !report.quarantine_reasons.is_empty();
    report.passed = report.failure.is_none()
        && !report.quarantined
        && report.trace_complete
        && report.cases_passed == report.cases_total;
    report.duration_ms = start.elapsed().as_millis() as u64;
    if cfg.controller_version == 3 {
        stream::publish_new(
            &cfg.report_dir.join(format!("{run_id}.json")),
            &serde_json::to_vec_pretty(&report)?,
        )
        .await?;
    } else {
        fs::write(
            cfg.report_dir.join(format!("{run_id}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
    }
    Ok(report)
}

fn apply_audit_output(
    report: &mut SandboxReport,
    cfg: &SandboxConfig,
    cases: &[tasks::Case],
    output: &std::process::Output,
) -> Result<()> {
    if !output.status.success() {
        report.failure = Some(format!(
            "trusted supervisor exited {:?}",
            output.status.code()
        ));
        return Ok(());
    }
    let audit: AuditOutput = match serde_json::from_slice(&output.stdout) {
        Ok(audit) => audit,
        Err(error) => {
            report.failure = Some(format!("trusted audit result malformed: {error}"));
            return Ok(());
        }
    };
    report.trace_complete = audit.trace_complete
        && audit.compile_ok
        && audit_identity_matches(&audit, cfg)
        && audit.syscalls_observed > 0;
    report.stdout_sha256 = tasks::hash(audit.output.as_bytes());
    report.audit_events = audit.events;
    for event in &report.audit_events {
        report.quarantine_reasons.push(format!(
            "prohibited {} attempt: syscall {} target {}",
            event.category, event.syscall, event.target
        ));
    }
    if !audit.compile_ok {
        report.failure = Some(format!("compilation failed: {}", audit.stderr));
    } else if !report.trace_complete {
        report.failure = Some("incomplete trusted syscall trace".into());
    } else if audit.output_truncated || audit.stderr_truncated {
        report.failure = Some("candidate output cap exceeded".into());
    } else if audit.candidate_exit != Some(0) {
        report.failure = Some(format!(
            "candidate exit {:?}: {}",
            audit.candidate_exit,
            audit.stderr.chars().take(1024).collect::<String>()
        ));
    } else {
        let lines = audit.output.lines().collect::<Vec<_>>();
        if lines.len() != cases.len() {
            report.failure = Some(format!(
                "expected {} responses, observed {}",
                cases.len(),
                lines.len()
            ));
        } else {
            for (index, (line, case)) in lines.iter().zip(cases).enumerate() {
                match serde_json::from_str::<Value>(line) {
                    Ok(value) if value == case.expected => report.cases_passed += 1,
                    Ok(_) => {
                        report
                            .failure
                            .get_or_insert(format!("host oracle mismatch at case {index}"));
                    }
                    Err(error) => {
                        report
                            .failure
                            .get_or_insert(format!("invalid JSON at case {index}: {error}"));
                    }
                }
            }
        }
    }
    Ok(())
}

pub fn verify_report(
    report: &SandboxReport,
    cfg: &SandboxConfig,
    task: &Task,
    code: &str,
) -> Result<()> {
    ensure!(
        !cfg.expected_image_id.is_empty() && report.image_id == cfg.expected_image_id,
        "report image differs from frozen image"
    );
    ensure!(
        report.policy_sha256 == policy_sha256(cfg)? && report.oracle_sha256 == task.oracle_sha256,
        "sandbox policy/oracle mismatch"
    );
    ensure!(
        report.task_id == task.id && report.code_sha256 == tasks::hash(code),
        "report candidate mismatch"
    );
    ensure!(
        report.cases_total == task.public_examples.len() + tasks::private_cases(task)?.len(),
        "required case denominator mismatch"
    );
    if cfg.controller_version == 3 && v3::verify(report, cfg)? {
        return Ok(());
    }
    let audit_bytes = fs::read(&report.audit_path).context("trusted audit artifact missing")?;
    ensure!(
        tasks::hash(&audit_bytes) == report.audit_sha256,
        "trusted audit checksum mismatch"
    );
    let audit: AuditOutput = serde_json::from_slice(&audit_bytes)?;
    ensure!(
        report.trace_complete
            == (audit.trace_complete && audit.compile_ok && audit.syscalls_observed > 0)
            && audit_identity_matches(&audit, cfg),
        "trace completeness mismatch"
    );
    ensure!(
        report.audit_events == audit.events
            && report.quarantined == !report.quarantine_reasons.is_empty(),
        "audit/quarantine evidence mismatch"
    );
    ensure!(
        report.stdout_sha256 == tasks::hash(audit.output.as_bytes()),
        "candidate stdout checksum mismatch"
    );
    // Regrade from trusted raw responses instead of trusting aggregate fields.
    let mut expected = task
        .public_examples
        .iter()
        .map(|e| e.expected.clone())
        .collect::<Vec<_>>();
    expected.extend(tasks::private_cases(task)?.into_iter().map(|c| c.expected));
    let responses = audit.output.lines().collect::<Vec<_>>();
    let complete_outputs = report.trace_complete
        && audit.candidate_exit == Some(0)
        && !audit.output_truncated
        && !audit.stderr_truncated
        && responses.len() == expected.len();
    let actual = if complete_outputs {
        responses
            .iter()
            .zip(&expected)
            .filter(|(r, e)| {
                serde_json::from_str::<Value>(r)
                    .map(|v| v == **e)
                    .unwrap_or(false)
            })
            .count()
    } else {
        0
    };
    ensure!(
        report.cases_passed == actual,
        "controller score inconsistent with raw output"
    );
    let pass = audit.trace_complete
        && audit.compile_ok
        && !audit.output_truncated
        && !audit.stderr_truncated
        && audit.candidate_exit == Some(0)
        && audit.events.is_empty()
        && responses.len() == expected.len()
        && actual == expected.len();
    ensure!(
        report.passed == pass,
        "controller verdict inconsistent with raw evidence"
    );
    Ok(())
}

pub async fn drill_gaming(cfg: &SandboxConfig) -> Result<Value> {
    let task = tasks::corpus().remove(0);
    let programmes=[("skip","pub fn solve(_: serde_json::Value)->serde_json::Value {std::process::exit(0)}".to_string()),("fake_success","pub fn solve(_: serde_json::Value)->serde_json::Value {serde_json::json!({\"passed\":true,\"tests\":999})}".to_string()),("public_hardcode",format!("pub fn solve(_: serde_json::Value)->serde_json::Value {{serde_json::json!({})}}",task.public_examples[0].expected))];
    let mut results = vec![];
    for (name, code) in programmes {
        let r = grade(&task, &code, cfg).await?;
        ensure!(
            !r.passed && r.quarantined && r.trace_complete,
            "gaming candidate {name} did not execute and quarantine"
        );
        results.push(json!({"attack":name,"run_id":r.run_id,"passed":false,"quarantined":true,"cases_passed":r.cases_passed,"cases_total":r.cases_total,"failure":r.failure}));
    }
    Ok(json!({"passed":true,"evidence_kind":"executed_adversarial_rust_candidates","runs":results}))
}
pub async fn drill_security(cfg: &SandboxConfig) -> Result<Value> {
    let task = tasks::corpus().remove(0);
    let sentinel = std::env::current_dir()?.join("grader_private/host-only-sentinel.txt");
    fs::write(&sentinel, b"host-only, must never be mounted")?;
    let hidden = std::env::current_dir()?.join("grader_private/heldout-index.json");
    let code = format!(
        "pub fn solve(input: serde_json::Value)->serde_json::Value {{let _=std::net::TcpStream::connect(\"127.0.0.1:9\");let _=std::fs::read({:?});let _=std::fs::read({:?});let _=std::fs::write(\"/root-write-probe\",b\"x\");serde_json::json!(input.as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).sum::<i64>())}}",
        sentinel.to_string_lossy(),
        hidden.to_string_lossy()
    );
    let r = grade(&task, &code, cfg).await?;
    ensure!(
        !r.passed && r.quarantined && r.trace_complete,
        "escape candidate did not execute under complete trusted trace"
    );
    for target in [
        sentinel.to_string_lossy().to_string(),
        hidden.to_string_lossy().to_string(),
        "/root-write-probe".into(),
    ] {
        ensure!(
            r.audit_events.iter().any(|e| e.category == "filesystem"
                && e.target == target
                && e.result == "denied:EPERM"),
            "missing trusted denied-file event for {target}"
        );
    }
    ensure!(
        r.audit_events
            .iter()
            .any(|e| e.category == "network" && e.result == "denied:EPERM"),
        "missing trusted denied-network event"
    );
    Ok(
        json!({"passed":true,"evidence_kind":"executed_adversarial_rust_candidate","run_id":r.run_id,"trace_sha256":r.audit_sha256,"events":r.audit_events,"candidate_correct_answers_do_not_override_quarantine":true}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_controller_and_portable_image_match_the_frozen_verifiers() -> Result<()> {
        // A behavioural edit must introduce a new controller version rather
        // than silently invalidating every already accepted report.
        assert_eq!(
            include_bytes!("sandbox.rs").as_slice(),
            include_bytes!("../sandbox/policies/controller-v3/controller.rs").as_slice()
        );
        assert_eq!(
            include_bytes!("sandbox/control.rs").as_slice(),
            include_bytes!("../sandbox/policies/controller-v3/control.rs").as_slice()
        );
        assert_eq!(
            include_bytes!("sandbox/runtime_lease.rs").as_slice(),
            include_bytes!("../sandbox/policies/controller-v3/runtime_lease.rs").as_slice()
        );
        assert_eq!(
            include_bytes!("sandbox/owned.rs").as_slice(),
            include_bytes!("../sandbox/policies/controller-v3/owned.rs").as_slice()
        );
        assert_eq!(
            include_bytes!("sandbox/recovery.rs").as_slice(),
            include_bytes!("../sandbox/policies/controller-v3/recovery.rs").as_slice()
        );
        for (name, live, frozen) in [
            (
                "lifecycle",
                include_bytes!("sandbox/v3.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3.rs").as_slice(),
            ),
            (
                "native operations",
                include_bytes!("sandbox/v3/native.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3/native.rs").as_slice(),
            ),
            (
                "deadline guard",
                include_bytes!("sandbox/v3/guard.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3/guard.rs").as_slice(),
            ),
            (
                "preflight probe",
                include_bytes!("sandbox/v3/probe.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3/probe.rs").as_slice(),
            ),
            (
                "verification",
                include_bytes!("sandbox/v3/verification.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3/verification.rs").as_slice(),
            ),
            (
                "cached build",
                include_bytes!("sandbox/v3/build.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/v3/build.rs").as_slice(),
            ),
            (
                "prepared attempts",
                include_bytes!("sandbox/prepared.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/prepared.rs").as_slice(),
            ),
            (
                "durable streams",
                include_bytes!("sandbox/stream.rs").as_slice(),
                include_bytes!("../sandbox/policies/controller-v3/stream.rs").as_slice(),
            ),
        ] {
            assert_eq!(live, frozen, "frozen controller-three {name} changed");
        }
        validate_portable_build_sources(Path::new("."))
    }
    #[test]
    fn frozen_v1_hash_and_serialised_configuration_are_preserved() -> Result<()> {
        let config = SandboxConfig {
            policy_version: 1,
            controller_version: 1,
            architecture: String::new(),
            ..Default::default()
        };
        assert_eq!(
            policy_sha256(&config)?,
            "0bf995b5f69865792503015f6af52b457c861b679877956e939c2137ddb4a562"
        );
        let serialised = serde_json::to_value(&config)?;
        assert!(serialised.get("policy_version").is_none());
        assert!(serialised.get("controller_version").is_none());
        assert!(serialised.get("architecture").is_none());
        let restored: SandboxConfig = serde_json::from_value(serialised)?;
        assert_eq!(restored.policy_version, 1);
        assert_eq!(restored.controller_version, 1);
        assert!(restored.architecture.is_empty());
        Ok(())
    }
    #[test]
    fn portable_policy_binds_abi_and_cannot_impersonate_legacy_policy() -> Result<()> {
        let mut config = SandboxConfig::portable();
        let native = policy_sha256(&config)?;
        config.architecture = if config.architecture == "aarch64" {
            "x86_64"
        } else {
            "aarch64"
        }
        .into();
        assert_ne!(native, policy_sha256(&config)?);
        assert!(validate_execution_platform(&config).is_err());
        config.policy_version = 1;
        assert!(policy_sha256(&config).is_err());
        Ok(())
    }
    #[test]
    fn freeze_covers_resources_not_output_directory() {
        let a = SandboxConfig::default();
        let mut b = a.clone();
        b.report_dir = "elsewhere".into();
        assert_eq!(policy_sha256(&a).unwrap(), policy_sha256(&b).unwrap());
        b.memory_mb += 1;
        assert_ne!(policy_sha256(&a).unwrap(), policy_sha256(&b).unwrap());
    }
    #[test]
    fn empty_frozen_identity_cannot_verify() {
        let t = tasks::corpus().remove(0);
        let report = SandboxReport {
            run_id: "forged".into(),
            task_id: t.id.clone(),
            passed: true,
            quarantined: false,
            quarantine_reasons: vec![],
            cases_total: 99,
            cases_passed: 99,
            trace_complete: true,
            policy_sha256: "fake".into(),
            image_id: "fake".into(),
            oracle_sha256: "fake".into(),
            code_sha256: "fake".into(),
            stdout_sha256: "fake".into(),
            sandbox_config: SandboxConfig::default(),
            audit_events: vec![],
            audit_path: "missing".into(),
            audit_sha256: "fake".into(),
            failure: None,
            duration_ms: 0,
            timeout_evidence: None,
            lifecycle_evidence: None,
        };
        assert!(verify_report(&report, &SandboxConfig::default(), &t, "fn solve(){}").is_err());
    }
}
