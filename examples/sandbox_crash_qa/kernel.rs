//! Observe exact candidate-group death without supplying a competing host kill.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{
    io::SeekFrom,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};
use tokio::io::{AsyncReadExt, AsyncSeekExt};

pub(super) struct ObservedCgroup {
    events: tokio::fs::File,
    pub binding: Value,
    pub last_events: String,
}
impl ObservedCgroup {
    pub async fn open(container: &Value, id: &str) -> Result<Self> {
        ensure!(
            id.len() == 64 && id.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid exact container ID"
        );
        let pid = container["State"]["Pid"]
            .as_u64()
            .context("namespace PID absent")?;
        ensure!(
            pid > 1 && pid <= i32::MAX as u64,
            "invalid live namespace PID"
        );
        let status = tokio::fs::read_to_string(format!("/proc/{pid}/status")).await?;
        ensure!(
            status
                .lines()
                .find_map(|line| line.strip_prefix("NSpid:"))
                .and_then(|line| line.split_whitespace().last())
                == Some("1"),
            "observed process is not namespace PID1"
        );
        let cgroup = tokio::fs::read_to_string(format!("/proc/{pid}/cgroup")).await?;
        let relative = cgroup
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .context("unified group absent")?;
        ensure!(
            Path::new(relative).components().all(|part| matches!(
                part,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )),
            "unsafe cgroup path"
        );
        let directory = tokio::fs::canonicalize(
            Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/')),
        )
        .await?;
        let leaf = directory
            .file_name()
            .and_then(|s| s.to_str())
            .context("group leaf absent")?;
        ensure!(
            directory.starts_with("/sys/fs/cgroup")
                && (leaf == id
                    || leaf == format!("libpod-{id}.scope")
                    || leaf == format!("docker-{id}.scope")),
            "group does not bind exact container"
        );
        let metadata = tokio::fs::metadata(&directory).await?;
        let path = directory.join("cgroup.events");
        let file = tokio::task::spawn_blocking(move || {
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(path)
        })
        .await??;
        let mut observed = Self {
            events: tokio::fs::File::from_std(file),
            binding: json!({"container_id":id,"host_pid":pid,
            "directory":directory,"device":metadata.dev(),"inode":metadata.ino(),"namespace_pid1_status":status}),
            last_events: String::new(),
        };
        ensure!(
            !observed.empty().await?,
            "candidate group was already empty before SIGKILL"
        );
        Ok(observed)
    }
    pub async fn empty(&mut self) -> Result<bool> {
        self.events.seek(SeekFrom::Start(0)).await?;
        let mut bytes = vec![0; 4097];
        let count = self.events.read(&mut bytes).await?;
        ensure!(
            count > 0 && count <= 4096,
            "invalid bounded kernel population record"
        );
        self.last_events = String::from_utf8(bytes[..count].to_vec())?;
        if self.last_events.lines().any(|line| line == "populated 0") {
            return Ok(true);
        }
        ensure!(
            self.last_events.lines().any(|line| line == "populated 1"),
            "kernel population field absent"
        );
        Ok(false)
    }
}
