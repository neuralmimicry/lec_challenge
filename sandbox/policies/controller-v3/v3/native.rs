//! Bounded native commands and independently bound process/timer metadata.
use super::*;
pub(super) async fn native(
    cfg: &SandboxConfig,
    mut command: Command,
    args: Vec<String>,
    operation: &str,
    prefix: &Path,
    seconds: u64,
) -> Result<(Output, NativeCall)> {
    let stdout = stream::PreparedStream::new(
        prefix.with_extension(format!("{operation}.stdout")),
        1 << 20,
    )
    .await?;
    let stderr = stream::PreparedStream::new(
        prefix.with_extension(format!("{operation}.stderr")),
        STDERR_CAP,
    )
    .await?;
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let start = Instant::now();
    let mut child = command.spawn()?;
    let stdout = stdout.attach(child.stdout.take().context("native stdout absent")?);
    let stderr = stderr.attach(child.stderr.take().context("native stderr absent")?);
    let result = timeout(Duration::from_secs(seconds.max(1)), child.wait()).await;
    let deadline_exceeded = result.is_err();
    let exit = match result {
        Ok(value) => value?,
        Err(_) => {
            let _ = child.start_kill();
            timeout(Duration::from_secs(5), child.wait())
                .await
                .context("native CLI could not be reaped")??
        }
    };
    let stdout = stdout.finish(5).await?;
    let stderr = stderr.finish(5).await?;
    let output = Output {
        status: exit,
        stdout: tokio::fs::read(&stdout.path).await?,
        stderr: tokio::fs::read(&stderr.path).await?,
    };
    let call = NativeCall {
        operation: operation.into(),
        arguments: args,
        duration_ms: start.elapsed().as_millis() as u64,
        exit_code: exit.code(),
        signal: exit.signal(),
        deadline_exceeded,
        stdout,
        stderr,
    };
    ensure!(
        cfg.controller_version == 3,
        "native capture requires controller three"
    );
    Ok((output, call))
}
pub(super) async fn call(
    cfg: &SandboxConfig,
    args: &[&str],
    operation: &str,
    prefix: &Path,
    receipt: &mut Receipt,
    seconds: u64,
) -> Result<Output> {
    let mut command = Command::new(&cfg.runtime);
    command.args(args);
    let (output, observed) = native(
        cfg,
        command,
        args.iter().map(|s| (*s).into()).collect(),
        operation,
        prefix,
        seconds,
    )
    .await?;
    let failed = observed.deadline_exceeded
        || observed.stdout.overflow
        || observed.stderr.overflow
        || observed.stdout.error.is_some()
        || observed.stderr.error.is_some()
        || !observed.stdout.eof
        || !observed.stderr.eof;
    receipt.calls.push(observed);
    ensure!(
        !failed,
        "bounded native {operation} failed or exceeded its metadata deadline"
    );
    Ok(output)
}
pub(super) fn inspected(output: &Output) -> Result<Value> {
    ensure!(
        output.status.success(),
        "owned container inspection failed: {}",
        String::from_utf8_lossy(&output.stderr)
            .chars()
            .take(512)
            .collect::<String>()
    );
    let mut values: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    ensure!(values.len() == 1, "ambiguous owned container inspection");
    Ok(values.remove(0))
}
pub(super) fn remaining(start: Instant, budget: u64) -> Result<u64> {
    let elapsed = start.elapsed();
    ensure!(
        elapsed < Duration::from_secs(budget),
        "native lifecycle phase deadline exceeded"
    );
    Ok((Duration::from_secs(budget) - elapsed)
        .as_secs()
        .clamp(1, METADATA_SECS))
}
pub(super) fn cgroup_from_inspection(inspected: &Value) -> Result<(String, Value)> {
    let pid = inspected["State"]["Pid"]
        .as_u64()
        .context("owned container PID absent")?;
    ensure!(
        pid > 1 && pid <= i32::MAX as u64,
        "owned container PID invalid"
    );
    let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let cgroup = fs::read_to_string(format!("/proc/{pid}/cgroup"))?;
    let process_status = fs::read_to_string(format!("/proc/{pid}/status"))?;
    ensure!(
        process_status
            .lines()
            .find_map(|line| line.strip_prefix("NSpid:"))
            .and_then(|line| line.split_whitespace().last())
            == Some("1"),
        "trusted supervisor is not exact namespace PID1"
    );
    ensure!(
        stat == fs::read_to_string(format!("/proc/{pid}/stat"))? || {
            // CPU accounting may change, but Linux start ticks must not.
            let ticks = |s: &str| {
                s.rsplit_once(')')
                    .and_then(|(_, t)| t.split_whitespace().nth(19))
                    .map(str::to_owned)
            };
            ticks(&stat) == ticks(&fs::read_to_string(format!("/proc/{pid}/stat"))?)
        },
        "owned container PID changed during cgroup binding"
    );
    let path = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("owned unified cgroup absent")?;
    if let Some(native) = inspected["State"]["CgroupPath"]
        .as_str()
        .filter(|s| !s.is_empty())
    {
        ensure!(native == path, "native and process cgroup paths disagree");
    }
    Ok((
        path.into(),
        json!({"host_pid":pid,"stat":stat,"cgroup":cgroup,"status":process_status,"namespace_pid1_verified":true}),
    ))
}

pub(super) fn eof_marker(stderr: &[u8], cfg: &SandboxConfig, release: i64) -> Result<Value> {
    let text = std::str::from_utf8(stderr)?;
    let markers = text
        .lines()
        .filter_map(|line| line.strip_prefix("LEC_TRUSTED_EOF_START "))
        .collect::<Vec<_>>();
    ensure!(
        markers.len() == 1,
        "trusted EOF-start marker missing or ambiguous"
    );
    let value: Value = serde_json::from_str(markers[0])?;
    ensure!(
        value["schema_version"] == 3
            && value["architecture"] == cfg.architecture
            && value["deadline_ms"] == 20000
            && value["started_at_eof"] == true
            && value["deadline_started_unix_ms"]
                .as_i64()
                .is_some_and(|start| start >= release),
        "trusted EOF-start marker differs from released protocol"
    );
    Ok(value)
}
pub(super) fn audit_timer_metadata(
    audit: &Value,
    marker: &Value,
    cfg: &SandboxConfig,
) -> Result<()> {
    ensure!(
        audit["schema_version"] == 3
            && audit["architecture"] == cfg.architecture
            && audit["deadline_ms"] == 20000
            && audit["started_at_eof"] == true
            && audit["deadline_started_unix_ms"] == marker["deadline_started_unix_ms"]
            && audit["deadline_elapsed_ms"].as_u64().is_some(),
        "trusted audit timer differs from EOF marker"
    );
    Ok(())
}
