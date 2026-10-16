//! Use a delegated cgroup-v2 kill file when available. Unlike a runtime CLI's
//! graceful stop, this atomically kills every process in the candidate group.
use super::*;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CgroupBinding {
    pub container_id: String,
    pub directory: PathBuf,
    pub device: u64,
    pub inode: u64,
    pub kill_device: u64,
    pub kill_inode: u64,
    pub events_device: u64,
    pub events_inode: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedKillEvidence {
    pub elapsed_from_release_ms: u64,
    pub populated_before: bool,
    pub kill_written: bool,
    pub kill_write_ms: u64,
    pub empty_observed: bool,
    pub empty_observation: Option<String>,
    pub empty_wait_ms: u64,
    pub failure: Option<String>,
}
pub(super) struct PinnedCgroup {
    pub binding: CgroupBinding,
    kill: File,
    events: File,
    population_evidence: String,
}
impl PinnedCgroup {
    pub fn open(container_id: &str, cgroup: &str) -> Result<Self> {
        let directory = confined_cgroup(container_id, cgroup)?;
        let metadata = fs::metadata(&directory)?;
        let options = || {
            let mut options = fs::OpenOptions::new();
            options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
            options
        };
        let kill = options()
            .write(true)
            .open(directory.join("cgroup.kill"))
            .context("exact owned cgroup.kill must be writable before candidate release")?;
        let events = options().read(true).open(directory.join("cgroup.events"))?;
        let after = fs::metadata(&directory)?;
        ensure!(
            after.dev() == metadata.dev()
                && after.ino() == metadata.ino()
                && kill.metadata()?.dev() == metadata.dev()
                && events.metadata()?.dev() == metadata.dev(),
            "exact owned cgroup changed while controls were opened"
        );
        let mut pinned = Self {
            binding: CgroupBinding {
                container_id: container_id.into(),
                directory,
                device: metadata.dev(),
                inode: metadata.ino(),
                kill_device: kill.metadata()?.dev(),
                kill_inode: kill.metadata()?.ino(),
                events_device: events.metadata()?.dev(),
                events_inode: events.metadata()?.ino(),
            },
            kill,
            events,
            population_evidence: String::new(),
        };
        ensure!(
            pinned.populated()?,
            "owned candidate cgroup is empty before input release"
        );
        Ok(pinned)
    }
    pub fn populated(&mut self) -> Result<bool> {
        let mut events = String::new();
        let read = (|| -> std::io::Result<()> {
            self.events.seek(SeekFrom::Start(0))?;
            self.events.read_to_string(&mut events)?;
            Ok(())
        })();
        if let Err(error) = read {
            if error.raw_os_error() == Some(libc::ENODEV)
                && (self.events.metadata()?.nlink() == 0 || !self.binding.directory.try_exists()?)
            {
                self.population_evidence = "exact_opened_cgroup_inode_removed_enodev".into();
                return Ok(false);
            }
            return Err(error.into());
        }
        if events.lines().any(|line| line == "populated 1") {
            self.population_evidence = "cached_cgroup_events_populated_1".into();
            return Ok(true);
        }
        ensure!(
            events.lines().any(|line| line == "populated 0"),
            "cached cgroup population state absent"
        );
        self.population_evidence = "cached_cgroup_events_populated_0".into();
        Ok(false)
    }
    /// Called only on a dedicated deadline thread with already opened controls.
    /// An already empty group never becomes an acknowledged kill.
    pub fn kill(&mut self, released: Instant) -> CachedKillEvidence {
        let elapsed_from_release_ms = released.elapsed().as_millis() as u64;
        let mut evidence = CachedKillEvidence {
            elapsed_from_release_ms,
            populated_before: false,
            kill_written: false,
            kill_write_ms: 0,
            empty_observed: false,
            empty_observation: None,
            empty_wait_ms: 0,
            failure: None,
        };
        let result: Result<()> = (|| {
            evidence.populated_before = self.populated()?;
            if !evidence.populated_before {
                evidence.empty_observed = true;
                evidence.empty_observation = Some(self.population_evidence.clone());
                return Ok(());
            }
            let start = Instant::now();
            self.kill.write_all(b"1")?;
            evidence.kill_write_ms = start.elapsed().as_millis() as u64;
            evidence.kill_written = true;
            let wait = Instant::now();
            loop {
                if !self.populated()? {
                    evidence.empty_observed = true;
                    evidence.empty_observation = Some(self.population_evidence.clone());
                    break;
                }
                if wait.elapsed() >= Duration::from_secs(1) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            evidence.empty_wait_ms = wait.elapsed().as_millis() as u64;
            Ok(())
        })();
        if let Err(error) = result {
            evidence.failure = Some(format!("{error:#}"));
        }
        evidence
    }
}

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
