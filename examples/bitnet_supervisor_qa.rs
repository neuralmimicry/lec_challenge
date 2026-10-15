//! Actual Linux SIGKILL/drop/moving-worker probes using a controlled Rust leaf.
//! This exercises the native engine supervisor; it is not an assessment drill.
use anyhow::{Context, Result, ensure};
use lec_distil::{artefact, bitnet::cpp};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[cfg(target_os = "linux")]
fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|a| a == "--leaf") {
        loop {
            std::thread::park_timeout(Duration::from_secs(60));
        }
    }
    let executable = std::env::current_exe()?;
    if arguments.first().is_some_and(|a| a == "--holder") {
        let marker = PathBuf::from(arguments.get(1).context("holder marker absent")?);
        let probe = cpp::spawn_supervision_probe(&executable)?;
        artefact::write_json(
            &marker,
            &serde_json::json!({"holder_pid":std::process::id(),"leaf_pid":probe.pid()}),
        )?;
        loop {
            std::thread::park_timeout(Duration::from_secs(60));
        }
    }
    let output = PathBuf::from("reports/comparative/bitnet/supervisor-qa.json");
    if output.exists() {
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(output)?)?;
        ensure!(
            saved["passed"] == true
                && saved["evidence_kind"] == "actual_native_supervisor_linux_lifetime_qa",
            "saved supervisor probe failed"
        );
        println!("{}", serde_json::to_string_pretty(&saved)?);
        return Ok(());
    }
    // SAFETY: this process opts into adopting its own controlled descendants so
    // the killed holder's leaf can be reaped rather than left as a PID1 zombie.
    ensure!(
        unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1) } == 0,
        "cannot install isolated QA subreaper"
    );
    let worker_executable = executable.clone();
    let probe = std::thread::spawn(move || cpp::spawn_supervision_probe(&worker_executable))
        .join()
        .map_err(|_| anyhow::anyhow!("probe worker panicked"))??;
    let normal_pid = probe.pid();
    std::thread::sleep(Duration::from_millis(150));
    let survives_creator_worker = unsafe { libc::kill(normal_pid as i32, 0) } == 0;
    drop(probe);
    let normal_reaped = unsafe { libc::kill(normal_pid as i32, 0) } < 0
        && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
    let marker = PathBuf::from(format!(
        "reports/comparative/bitnet/supervisor-holder-{}.json",
        uuid::Uuid::new_v4()
    ));
    let mut holder = std::process::Command::new(&executable)
        .arg("--holder")
        .arg(&marker)
        .spawn()?;
    let started = Instant::now();
    while !marker.exists() {
        ensure!(
            started.elapsed() < Duration::from_secs(10),
            "holder readiness deadline exceeded"
        );
        if let Some(status) = holder.try_wait()? {
            anyhow::bail!("controlled holder exited before readiness: {status}");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let ids: serde_json::Value = serde_json::from_slice(&std::fs::read(&marker)?)?;
    let leaf = ids["leaf_pid"].as_u64().context("leaf PID absent")? as i32;
    holder.kill()?;
    let holder_status = holder.wait()?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut status = 0_i32;
    let mut reaped = false;
    while Instant::now() < deadline {
        let result = unsafe { libc::waitpid(leaf, &mut status, libc::WNOHANG) };
        if result == leaf {
            reaped = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let parent_death_killed =
        reaped && libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGKILL;
    if !reaped {
        unsafe { libc::kill(leaf, libc::SIGKILL) };
        unsafe { libc::waitpid(leaf, &mut status, 0) };
    }
    let passed = survives_creator_worker && normal_reaped && parent_death_killed;
    let root = std::env::current_dir()?.canonicalize()?;
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"actual_native_supervisor_linux_lifetime_qa","assessment_drill":false,"controlled_rust_leaf_not_model_output":true,"survives_creator_worker_exit":survives_creator_worker,"normal_drop_killed_and_reaped":normal_reaped,"holder_sigkill_status":holder_status.to_string(),"parent_sigkill_kills_leaf":parent_death_killed,"adopted_leaf_reaped":reaped,"holder_marker":artefact::Artefact::capture(&root,&marker)?,"probe_binary_sha256":artefact::file_hash(&executable)?,"passed":passed});
    artefact::write_json(&output, &report)?;
    ensure!(passed, "native supervisor lifetime invariant failed");
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn main() -> Result<()> {
    anyhow::bail!("native parent-death QA requires Linux")
}
