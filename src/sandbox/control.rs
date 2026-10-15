//! Use a delegated cgroup-v2 kill file when available. Unlike a runtime CLI's
//! graceful stop, this atomically kills every process in the candidate group.
use super::*;

fn confined_cgroup(container_id: &str, cgroup: &str) -> Result<PathBuf> {
    ensure!(
        container_id.len() == 64 && container_id.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid container ID from runtime"
    );
    let path = Path::new(cgroup);
    ensure!(
        path.is_absolute()
            && path.components().all(|c| matches!(
                c,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )),
        "invalid container cgroup path"
    );
    let leaf = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("container cgroup leaf missing")?;
    ensure!(
        leaf == container_id
            || leaf == format!("libpod-{container_id}.scope")
            || leaf == format!("docker-{container_id}.scope"),
        "cgroup does not bind the exact inspected container ID"
    );
    let root = Path::new("/sys/fs/cgroup").canonicalize()?;
    let directory = root.join(cgroup.trim_start_matches('/')).canonicalize()?;
    ensure!(
        directory.starts_with(&root) && directory != root,
        "container cgroup escapes delegated hierarchy"
    );
    Ok(directory)
}

pub(super) async fn kill_delegated_container_cgroup(
    cfg: &SandboxConfig,
    name: &str,
) -> Result<bool> {
    let output = timeout(
        Duration::from_secs(2),
        Command::new(&cfg.runtime)
            .args(["inspect", "--format", "{{.Id}} {{.State.CgroupPath}}", name])
            .output(),
    )
    .await??;
    if !output.status.success() {
        return Ok(false);
    }
    let text = String::from_utf8(output.stdout)?;
    let mut fields = text.split_whitespace();
    let id = fields.next().context("container identity missing")?;
    let cgroup = fields.next().context("cgroup path missing")?;
    ensure!(fields.next().is_none(), "unexpected cgroup identity fields");
    let directory = confined_cgroup(id, cgroup)?;
    if tokio::fs::write(directory.join("cgroup.kill"), b"1")
        .await
        .is_err()
    {
        return Ok(false);
    }
    let started = Instant::now();
    loop {
        match tokio::fs::read_to_string(directory.join("cgroup.events")).await {
            Ok(events) if events.lines().any(|l| l == "populated 0") => return Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(true),
            Err(e) => return Err(e.into()),
            _ => {}
        }
        if started.elapsed() > Duration::from_secs(1) {
            return Ok(false);
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cgroup_control_cannot_target_root_ancestors_or_another_container() {
        let id = "a".repeat(64);
        for path in [
            "/",
            "/user.slice",
            "/user.slice/../",
            "/user.slice/not-this-container",
        ] {
            assert!(confined_cgroup(&id, path).is_err(), "{path}");
        }
    }
}
