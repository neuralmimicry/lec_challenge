//! Serialise shared native checkout/build publication without blocking Tokio.
use super::*;
use std::os::{
    fd::AsRawFd,
    unix::fs::{MetadataExt, OpenOptionsExt},
};

pub(super) struct BuildLease(fs::File);
impl BuildLease {
    pub async fn acquire(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        ensure!(
            file.metadata()?.is_file() && file.metadata()?.nlink() == 1,
            "native bootstrap lease must be a private regular file"
        );
        let started = Instant::now();
        let mut heartbeat = Instant::now();
        loop {
            // SAFETY: flock borrows a retained live descriptor and no pointers.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(Self(file));
            }
            let error = std::io::Error::last_os_error();
            if !matches!(error.raw_os_error(), Some(libc::EAGAIN)) {
                return Err(error).context("acquire shared native bootstrap lease");
            }
            ensure!(
                started.elapsed() < Duration::from_secs(3600),
                "native bootstrap lease deadline exceeded"
            );
            if heartbeat.elapsed() >= Duration::from_secs(15) {
                eprintln!(
                    "BitNet native bootstrap waiting for existing build: {:.0}s",
                    started.elapsed().as_secs_f64()
                );
                heartbeat = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }
}
impl Drop for BuildLease {
    fn drop(&mut self) {
        // SAFETY: the descriptor remains live until the lock is released.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn competing_bootstraps_wait_then_reuse_the_same_publication_boundary() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("build.lock");
        let first = BuildLease::acquire(&path).await?;
        let mut pending = tokio::spawn(async move { BuildLease::acquire(&path).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut pending)
                .await
                .is_err()
        );
        drop(first);
        let second = tokio::time::timeout(Duration::from_secs(3), pending).await???;
        drop(second);
        Ok(())
    }
}
