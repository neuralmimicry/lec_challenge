//! Isolate pressure/deadline probes across processes using one native runtime.
//! Ordinary grading retains shared leases and may run in parallel. A writer
//! turnstile, once acquired, keeps later grades behind a waiting preflight.
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub(super) struct RuntimeLease {
    _gate: File,
    _turnstile: Option<File>,
    native_podman: bool,
}
impl RuntimeLease {
    pub fn supports_independent_wall_timeout(&self) -> bool {
        self.native_podman
    }
}

pub(super) async fn shared(runtime: &str) -> Result<RuntimeLease> {
    acquire(runtime, false).await
}
pub(super) async fn exclusive(runtime: &str) -> Result<RuntimeLease> {
    acquire(runtime, true).await
}

async fn acquire(runtime: &str, exclusive: bool) -> Result<RuntimeLease> {
    let runtime = runtime.to_owned();
    // Executable resolution and filesystem operations belong on a blocking
    // worker. All contended lock waits below use non-blocking flock and Tokio.
    let files = tokio::task::spawn_blocking(move || {
        let executable = executable(&runtime)?;
        // SAFETY: geteuid has no pointer arguments or process-state mutation.
        let uid = unsafe { libc::geteuid() };
        let identity = crate::tasks::hash(executable.as_os_str().as_encoded_bytes());
        Ok::<_, anyhow::Error>((
            open_files(&PathBuf::from(format!(
                "/tmp/lec-sandbox-runtime-{uid}-{identity}"
            )))?,
            executable.file_name().is_some_and(|name| name == "podman"),
        ))
    })
    .await??;
    let mut lease = acquire_files(files.0, exclusive).await?;
    lease.native_podman = files.1;
    Ok(lease)
}

fn executable(runtime: &str) -> Result<PathBuf> {
    let configured = Path::new(runtime);
    if configured.components().count() > 1 {
        return configured
            .canonicalize()
            .context("resolve configured sandbox runtime");
    }
    for directory in
        std::env::split_paths(&std::env::var_os("PATH").context("runtime PATH absent")?)
    {
        let candidate = directory.join(configured);
        if fs::metadata(&candidate).is_ok_and(|m| m.is_file() && m.mode() & 0o111 != 0) {
            return candidate
                .canonicalize()
                .context("resolve sandbox runtime executable");
        }
    }
    anyhow::bail!("configured sandbox runtime executable is unavailable")
}

fn open_files(prefix: &Path) -> Result<(File, File)> {
    let open = |extension: &str| -> Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(prefix.with_extension(extension))?;
        let metadata = file.metadata()?;
        // SAFETY: geteuid has no pointer arguments or process-state mutation.
        let uid = unsafe { libc::geteuid() };
        ensure!(
            metadata.is_file()
                && metadata.nlink() == 1
                && metadata.uid() == uid
                && metadata.mode() & 0o077 == 0,
            "runtime lease must be an owned private regular file"
        );
        Ok(file)
    };
    Ok((open("turnstile.lock")?, open("gate.lock")?))
}

async fn acquire_files((turnstile, gate): (File, File), exclusive: bool) -> Result<RuntimeLease> {
    let started = Instant::now();
    lock(&turnstile, libc::LOCK_EX, started).await?;
    lock(
        &gate,
        if exclusive {
            libc::LOCK_EX
        } else {
            libc::LOCK_SH
        },
        started,
    )
    .await?;
    Ok(RuntimeLease {
        _gate: gate,
        _turnstile: if exclusive { Some(turnstile) } else { None },
        native_podman: false,
    })
}

async fn lock(file: &File, operation: i32, started: Instant) -> Result<()> {
    let mut heartbeat = Instant::now();
    loop {
        // SAFETY: LOCK_NB cannot block the executor; the descriptor stays live.
        if unsafe { libc::flock(file.as_raw_fd(), operation | libc::LOCK_NB) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EAGAIN) {
            return Err(error).context("acquire sandbox runtime coordination lease");
        }
        ensure!(
            started.elapsed() < Duration::from_secs(600),
            "sandbox runtime lease exceeded ten-minute coordination deadline"
        );
        if heartbeat.elapsed() >= Duration::from_secs(15) {
            eprintln!(
                "Sandbox runtime waiting for isolated preflight/active grades: {:.0}s",
                started.elapsed().as_secs_f64()
            );
            heartbeat = Instant::now();
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancelled_waiter_releases_turnstile_without_blocking_later_grades() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let prefix = directory.path().join("cancelled");
        let first = acquire_files(open_files(&prefix)?, false).await?;
        let files = open_files(&prefix)?;
        let waiter = tokio::spawn(async move { acquire_files(files, true).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        waiter.abort();
        assert!(matches!(waiter.await, Err(error) if error.is_cancelled()));
        let later = tokio::time::timeout(
            Duration::from_secs(3),
            acquire_files(open_files(&prefix)?, false),
        )
        .await??;
        drop((first, later));
        Ok(())
    }
    #[tokio::test]
    async fn grades_share_but_a_waiting_preflight_excludes_later_grades() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let prefix = directory.path().join("runtime");
        let first = acquire_files(open_files(&prefix)?, false).await?;
        let second = acquire_files(open_files(&prefix)?, false).await?;
        let files = open_files(&prefix)?;
        let mut writer = tokio::spawn(async move { acquire_files(files, true).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let files = open_files(&prefix)?;
        let mut later_reader = tokio::spawn(async move { acquire_files(files, false).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut writer)
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut later_reader)
                .await
                .is_err()
        );
        drop((first, second));
        let exclusive = tokio::time::timeout(Duration::from_secs(3), writer).await???;
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut later_reader)
                .await
                .is_err()
        );
        drop(exclusive);
        drop(tokio::time::timeout(Duration::from_secs(3), later_reader).await???);
        Ok(())
    }
    #[test]
    fn executable_aliases_resolve_to_the_same_runtime_identity() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let alias = directory.path().join("runtime-alias");
        std::os::unix::fs::symlink("/bin/true", &alias)?;
        assert_eq!(
            executable(alias.to_str().context("test path not UTF-8")?)?,
            Path::new("/bin/true").canonicalize()?
        );
        Ok(())
    }
}
