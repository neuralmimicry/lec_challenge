//! Enforce a total wall clock across pinned stage subprocesses. Native model
//! work must not outlive a cancelled orchestration future.
use super::*;
use std::{os::fd::AsRawFd, process::Stdio};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    process::Command,
};

/// A missing downstream report must not conceal the child's original failure.
/// Call on a blocking worker: receipt verification may hash a substantial log.
pub(crate) fn required_report(
    root: &Path,
    path: &Path,
    process_ok: bool,
    attempt: Option<&Artefact>,
) -> Result<Value> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid required stage report {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let Some(attempt) = attempt else {
                anyhow::bail!(
                    "required report {} is absent; process_success={process_ok}; no durable attempt receipt",
                    path.display()
                );
            };
            let receipt: Value = serde_json::from_slice(&fs::read(attempt.verify(root)?)?)?;
            let stderr: Artefact = serde_json::from_value(receipt["stderr"].clone())?;
            let stderr_path = stderr.verify(root)?;
            use std::io::{Read, Seek, SeekFrom};
            let mut file = fs::File::open(&stderr_path)?;
            file.seek(SeekFrom::End(-(stderr.bytes.min(8192) as i64)))?;
            let mut bytes = Vec::new();
            file.take(8192).read_to_end(&mut bytes)?;
            let tail = String::from_utf8_lossy(&bytes);
            let cause = tail
                .lines()
                .rev()
                .find_map(|line| line.strip_prefix("FAILURE: "))
                .unwrap_or("child emitted no structured failure line")
                .chars()
                .take(1024)
                .collect::<String>();
            anyhow::bail!(
                "required report {} is absent; process_success={process_ok}; exit_code={}; child failure: {cause}; attempt={}; stderr={}",
                path.display(),
                receipt["exit_code"],
                attempt.path.display(),
                stderr.path.display()
            );
        }
        Err(error) => {
            Err(error).with_context(|| format!("read required stage report {}", path.display()))
        }
    }
}

/// Fixed-size chunks also bound a child's unbroken or very long log lines.
/// Persist bytes before forwarding them, then sync before publishing a receipt.
async fn tee<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    mut source: R,
    mut file: tokio::fs::File,
    mut visible: W,
) -> Result<()> {
    let mut buffer = [0u8; 8192];
    loop {
        let read = source.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read]).await?;
        visible.write_all(&buffer[..read]).await?;
        visible.flush().await?;
    }
    file.flush().await?;
    file.sync_all().await?;
    visible.flush().await?;
    Ok(())
}

struct ProcessGroup(Option<i32>);
impl ProcessGroup {
    fn terminate(&mut self) {
        if let Some(group) = self.0.take() {
            // SAFETY: the positive PID was assigned its own group before exec.
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.terminate();
    }
}

/// Wait for an existing search rather than launching a competing writer. The
/// child acquires its own authoritative lease once this observational wait ends.
pub(super) async fn wait_for_search(context: &ContextState) -> Result<()> {
    let path = context.options.search_output.join(".search.lock");
    if !path.exists() {
        return Ok(());
    }
    loop {
        context.remaining()?;
        let file = fs::OpenOptions::new().read(true).write(true).open(&path)?;
        // SAFETY: the descriptor remains live; this probes, then releases ownership.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
            return Ok(());
        }
        ensure!(
            std::io::Error::last_os_error()
                .raw_os_error()
                .is_some_and(|code| [libc::EAGAIN, libc::EWOULDBLOCK].contains(&code)),
            "cannot inspect existing search ownership"
        );
        eprintln!(
            "explore: existing teacher search is active; waiting without launching a duplicate; {:.0}s remain",
            context.remaining()?.as_secs_f64()
        );
        tokio::time::sleep(Duration::from_secs(10).min(context.remaining()?)).await;
    }
}

pub(super) async fn stage(
    context: &mut ContextState,
    name: &str,
    engine: &PinnedEngine,
    args: Vec<String>,
) -> Result<bool> {
    stage_with_cpu(context, name, engine, args, None).await
}

pub(super) async fn stage_with_cpu(
    context: &mut ContextState,
    name: &str,
    engine: &PinnedEngine,
    args: Vec<String>,
    assigned_cpu_threads: Option<u32>,
) -> Result<bool> {
    ensure!(
        assigned_cpu_threads.is_none_or(|threads| threads > 0),
        "assigned stage CPU threads must be positive"
    );
    context.remaining()?;
    let checked = engine.clone();
    tokio::task::spawn_blocking(move || checked.verify()).await??;
    let directory = context
        .options
        .output
        .join("attempts")
        .join(name)
        .join(uuid::Uuid::new_v4().to_string());
    fs::create_dir_all(&directory)?;
    let stdout = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join("stdout.log"))?;
    let stdout_sync = stdout.try_clone()?;
    let stderr = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join("stderr.log"))?;
    let request = directory.join("request.json");
    artefact::write_json(
        &request,
        &json!({"schema_version":1,"stage":name,"engine":engine,"arguments":args,
        "started_unix_ms":crate::ledger::unix_millis(),"offline":context.options.offline,
        "remaining_wall_seconds":context.remaining()?.as_secs_f64(),"assigned_cpu_threads":assigned_cpu_threads,
        "model_cpu_threads":assigned_cpu_threads.map(|threads|threads.saturating_sub(1).max(1))}),
    )?;
    let entry = context.state.stages.entry(name.into()).or_insert(Stage {
        status: "pending".into(),
        attempts: vec![],
        artefacts: vec![],
        details: Value::Null,
    });
    entry.status = "running".into();
    entry.attempts.push(capture(&context.root, &request)?);
    context.save().await?;
    let mut command = Command::new(&engine.path);
    command
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .process_group(0);
    if let Some(threads) = assigned_cpu_threads {
        let model_threads = threads.saturating_sub(1).max(1).to_string();
        for name in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
            command.env(name, &model_threads);
        }
        command.env("LEC_STUDY_CPU_THREADS", threads.to_string());
    }
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
    #[cfg(target_os = "linux")]
    unsafe {
        let parent = std::process::id() as libc::pid_t;
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
    let mut child = command
        .spawn()
        .with_context(|| format!("start exploration stage {name}"))?;
    let mut group = ProcessGroup(Some(i32::try_from(
        child.id().context("stage PID absent")?,
    )?));
    let pipe = child.stderr.take().context("stage stderr pipe missing")?;
    let mut drain = tokio::spawn(tee(
        pipe,
        tokio::fs::File::from_std(stderr),
        tokio::io::stderr(),
    ));
    let started = Instant::now();
    let mut heartbeat = tokio::time::interval(Duration::from_secs(10));
    let mut tee_finished: Option<Result<()>> = None;
    let result = loop {
        tokio::select! {
            result=child.wait()=>break Some(result?),
            result=&mut drain,if tee_finished.is_none()=>{
                let result=result.map_err(anyhow::Error::from).and_then(|result|result);
                let failed=result.is_err();tee_finished=Some(result);
                if failed {group.terminate();break Some(child.wait().await?);}
            },
            _=tokio::time::sleep(context.deadline.saturating_duration_since(Instant::now()))=>break None,
            _=heartbeat.tick()=>eprintln!("explore: stage={name} elapsed={:.0}s remaining={:.0}s log={}",started.elapsed().as_secs_f64(),context.deadline.saturating_duration_since(Instant::now()).as_secs_f64(),directory.join("stderr.log").display()),
        }
    };
    group.terminate();
    if result.is_none() {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }
    let tee_error = if let Some(result) = tee_finished {
        result
            .err()
            .map(|error| format!("stderr tee failed: {error:#}"))
    } else {
        match tokio::time::timeout(Duration::from_secs(5), &mut drain).await {
            Ok(Ok(Ok(()))) => None,
            Ok(Ok(Err(error))) => Some(format!("stderr tee failed: {error:#}")),
            Ok(Err(error)) => Some(format!("stderr tee task failed: {error}")),
            Err(_) => {
                drain.abort();
                let _ = drain.await;
                Some("stderr drain exceeded bounded cleanup deadline".into())
            }
        }
    };
    let stderr_path = directory.join("stderr.log");
    tokio::task::spawn_blocking(move || -> Result<()> {
        stdout_sync.sync_all()?;
        fs::File::open(stderr_path)?.sync_all()?;
        Ok(())
    })
    .await??;
    let successful = result.as_ref().is_some_and(|status| status.success()) && tee_error.is_none();
    let receipt = directory.join("result.json");
    artefact::write_json(
        &receipt,
        &json!({"schema_version":1,"stage":name,"request":capture(&context.root,&request)?,
        "success":successful,"status":if result.is_none(){"wall_budget_stopped_progress_saved"}else if successful{"process_completed"}else{"process_failed"},
        "exit_code":result.and_then(|status|status.code()),"wall_seconds":started.elapsed().as_secs_f64(),"stderr_tee_error":tee_error,
        "stdout":capture(&context.root,&directory.join("stdout.log"))?,"stderr":capture(&context.root,&directory.join("stderr.log"))?}),
    )?;
    context.state.events.push(json!({"stage":name,"attempt":capture(&context.root,&receipt)?,"unix_ms":crate::ledger::unix_millis()}));
    context
        .state
        .stages
        .get_mut(name)
        .context("stage state missing")?
        .status = if successful {
        "process_completed"
    } else {
        "incomplete_progress_saved"
    }
    .into();
    context.save().await?;
    context.remaining()?;
    Ok(successful)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_report_retains_verified_child_failure_and_rejects_changed_logs() {
        let root = tempfile::tempdir().unwrap();
        let mut log = "é".repeat(5000);
        log.push_str("\nFAILURE: host deadline terminated probe pids\n");
        artefact::atomic_write(&root.path().join("stderr.log"), log.as_bytes()).unwrap();
        let stderr = Artefact::capture(root.path(), "stderr.log").unwrap();
        artefact::write_json(
            &root.path().join("result.json"),
            &json!({"success":false,"exit_code":1,"stderr":stderr}),
        )
        .unwrap();
        let receipt = Artefact::capture(root.path(), "result.json").unwrap();
        let path = root.path().join("missing-report.json");
        let error = required_report(root.path(), &path, false, Some(&receipt)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("host deadline terminated probe pids")
        );
        assert!(error.to_string().contains("exit_code=1"));
        assert!(error.to_string().contains("result.json"));
        artefact::atomic_write(&root.path().join("stderr.log"), b"forged replacement log").unwrap();
        assert!(
            !required_report(root.path(), &path, false, Some(&receipt))
                .unwrap_err()
                .to_string()
                .contains("forged replacement")
        );
    }

    #[test]
    fn successful_process_with_no_output_is_still_an_error() {
        let root = tempfile::tempdir().unwrap();
        let error =
            required_report(root.path(), &root.path().join("absent.json"), true, None).unwrap_err();
        assert!(error.to_string().contains("process_success=true"));
        assert!(!root.path().join("absent.json").exists());
    }

    #[tokio::test]
    async fn long_unbroken_stderr_is_preserved_in_bounded_chunks() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("labelled-fixture.log");
        let input = vec![b'x'; 128 * 1024 + 7];
        let file = tokio::fs::File::create(&path).await.unwrap();
        tee(input.as_slice(), file, tokio::io::sink())
            .await
            .unwrap();
        assert_eq!(tokio::fs::read(path).await.unwrap(), input);
    }
}
