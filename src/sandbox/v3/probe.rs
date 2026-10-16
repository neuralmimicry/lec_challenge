//! Actual cached-FD preflight with an explicit one-second QA deadline.
use super::*;
pub(super) async fn cached_timeout_probe(
    cfg: &SandboxConfig,
    name: &str,
    mut command: Command,
    ownership: &mut recovery::OwnedRun,
) -> Result<TimeoutEvidence> {
    let prefix = cfg.report_dir.join(name);
    tokio::fs::create_dir_all(&cfg.report_dir).await?;
    let stderr =
        stream::PreparedStream::new(prefix.with_extension("probe.stderr"), STDERR_CAP).await?;
    let mut child = command.spawn()?;
    let stderr = stderr.attach(child.stderr.take().context("probe stderr absent")?);
    let mut stdout = BufReader::new(child.stdout.take().context("probe stdout absent")?);
    let mut marker = String::new();
    timeout(
        Duration::from_secs(STARTUP_SECS),
        stdout.read_line(&mut marker),
    )
    .await
    .context("trusted probe startup deadline")??;
    ensure!(
        marker == "trusted_timeout_probe_started\n",
        "trusted deadline probe did not start"
    );
    stream::write_new(&prefix.with_extension("probe.stdout"), marker.as_bytes()).await?;
    let mut inspect_command = Command::new(&cfg.runtime);
    inspect_command.args(["inspect", name]);
    let (inspect_output, inspection_call) = native(
        cfg,
        inspect_command,
        vec!["inspect".into(), name.into()],
        "probe-inspect",
        &prefix,
        METADATA_SECS,
    )
    .await?;
    ensure!(
        !inspection_call.deadline_exceeded,
        "trusted probe inspection deadline"
    );
    let inspection = inspected(&inspect_output)?;
    let id = ownership.verify_inspection(&inspection)?;
    ensure!(
        inspection["State"]["Running"] == true,
        "trusted timeout probe ended before cached control"
    );
    ownership.bind_container(&id).await?;
    let (path, process) = cgroup_from_inspection(&inspection)?;
    let control = tokio::task::spawn_blocking({
        let id = id.clone();
        move || control::PinnedCgroup::open(&id, &path)
    })
    .await??;
    let binding = control.binding.clone();
    let released = Instant::now();
    let (cancel, done) = guard(control, released, 1)?;
    let mut done = done;
    let controlled = tokio::select! {
        result=&mut done=>result.context("trusted deadline guard failed")?,
        status=child.wait()=>{let _=cancel.send(GuardSignal::Finished);let _=done.await;bail!("trusted probe ended before host deadline: {:?}",status?.code());}
    };
    let kill = controlled
        .kill
        .context("trusted probe supplied no cached kill")?;
    ensure!(
        controlled.reason == "execution_wall_deadline"
            && kill.populated_before
            && kill.kill_written
            && kill.empty_observed
            && kill.failure.is_none()
            && kill.elapsed_from_release_ms >= 1000
            && kill.elapsed_from_release_ms <= 2000
            && kill.kill_write_ms <= 5000,
        "bounded cached SIGKILL/empty-group evidence incomplete"
    );
    let cleanup = Instant::now();
    let exit = timeout(Duration::from_secs(CLEANUP_SECS), child.wait())
        .await
        .context("trusted probe CLI cleanup deadline")??;
    let stderr = stderr.finish(5).await?;
    let mut terminal_command = Command::new(&cfg.runtime);
    terminal_command.args(["inspect", &id]);
    let (terminal_output, terminal_call) = native(
        cfg,
        terminal_command,
        vec!["inspect".into(), id.clone()],
        "probe-terminal",
        &prefix,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    let terminal = inspected(&terminal_output)?;
    ensure!(
        ownership.verify_inspection(&terminal)? == id && terminal["State"]["Running"] == false,
        "trusted probe terminal state differs"
    );
    let mut remove_command = Command::new(&cfg.runtime);
    remove_command.args(["rm", "--force", "--time", "0", &id]);
    let (removed, remove_call) = native(
        cfg,
        remove_command,
        vec![
            "rm".into(),
            "--force".into(),
            "--time".into(),
            "0".into(),
            id.clone(),
        ],
        "probe-remove",
        &prefix,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    ensure!(
        removed.status.success(),
        "trusted probe exact removal failed"
    );
    let mut absent_command = Command::new(&cfg.runtime);
    absent_command.args(["container", "exists", &id]);
    let (absent, absence_call) = native(
        cfg,
        absent_command,
        vec!["container".into(), "exists".into(), id.clone()],
        "probe-absence",
        &prefix,
        remaining(cleanup, CLEANUP_SECS)?,
    )
    .await?;
    ensure!(
        absent.status.code() == Some(1) && !absence_call.deadline_exceeded,
        "trusted probe exact absence not proved"
    );
    ownership.finish_verified_absence().await?;
    let proof = json!({"schema_version":3,"evidence_kind":"actual_trusted_namespace_cached_fd_deadline_probe",
        "container_id":id,"name":name,"image_id":cfg.expected_image_id,"nonce":ownership.nonce(),"scope":ownership.scope(),
        "host_test_deadline_ms":1000,"outer_startup_crash_ceiling_seconds":80,"binding":binding,"process":process,"kill":kill,
        "inspection":inspection_call,"terminal":terminal_call,"terminal_state":terminal,"removal":remove_call,"absence":absence_call,
        "container_removed":true,"cli_exit":exit.code(),"stderr":stderr,"cleanup_ms":cleanup.elapsed().as_millis()});
    let path = prefix.with_extension("probe-proof.json");
    let bytes = serde_json::to_vec_pretty(&proof)?;
    stream::publish_new(&path, &bytes).await?;
    Ok(TimeoutEvidence {
        deadline_ms: 1000,
        wait_elapsed_ms: kill.elapsed_from_release_ms,
        kill_request_ms: kill.kill_write_ms,
        kill_acknowledged: true,
        cleanup_ms: cleanup.elapsed().as_millis() as u64,
        container_removed: true,
        kill_method: "preopened_exact_owned_cgroup_v2_kill".into(),
        kernel_proof: Some(LifecycleEvidence {
            receipt_path: path,
            receipt_sha256: tasks::hash(bytes),
        }),
    })
}
