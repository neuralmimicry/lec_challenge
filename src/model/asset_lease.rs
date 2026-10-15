//! A model-lifetime shared lease interlocks inference with managed downloads
//! and pruning. Closing the descriptor releases it, including after a crash.
use anyhow::{Context, Result, ensure};
use candle_core::Device;
use std::{fs::File, path::Path};

pub(super) struct AssetLease {
    _file: File,
    device: Option<Device>,
}

impl AssetLease {
    #[cfg(unix)]
    pub(super) fn acquire(directory: &Path) -> Result<Self> {
        use std::{fs::OpenOptions, os::fd::AsRawFd, os::unix::fs::OpenOptionsExt};
        let metadata = std::fs::symlink_metadata(directory)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "comparative model lease requires a real directory"
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(directory.join(".hub-download.lock"))
            .context("open comparative model asset lease")?;
        ensure!(file.metadata()?.is_file(), "asset lease is not a file");
        // SAFETY: flock borrows this valid descriptor; File owns its lifetime.
        let locked = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
        if locked != 0 {
            return Err(std::io::Error::last_os_error())
                .context("comparative model assets are being downloaded or pruned");
        }
        Ok(Self {
            _file: file,
            device: None,
        })
    }

    #[cfg(not(unix))]
    pub(super) fn acquire(_directory: &Path) -> Result<Self> {
        anyhow::bail!("comparative inference requires the supported Unix asset-lease backend")
    }

    pub(super) fn with_device(mut self, device: &Device) -> Self {
        self.device = Some(device.clone());
        self
    }
}

impl Drop for AssetLease {
    fn drop(&mut self) {
        // QwenModel declares the lease after all tensor fields. Those tensors
        // have dropped before this final device synchronisation/unlock.
        if let Some(device) = &self.device
            && let Err(error) = device.synchronize()
        {
            eprintln!(
                "comparative model device synchronisation during lease release failed: {error}"
            );
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::{fs::OpenOptions, os::fd::AsRawFd};

    fn exclusive(directory: &Path) -> Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(directory.join(".hub-download.lock"))?;
        // SAFETY: File keeps this descriptor alive through the synchronous call.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "an active shared or exclusive lease prevented access"
        );
        Ok(file)
    }

    #[test]
    fn concurrent_readers_block_pruning_until_the_final_runtime_lease_drops() {
        let directory = tempfile::tempdir().unwrap();
        let first = AssetLease::acquire(directory.path()).unwrap();
        let second = AssetLease::acquire(directory.path())
            .unwrap()
            .with_device(&Device::Cpu);
        assert!(exclusive(directory.path()).is_err());
        drop(first);
        assert!(exclusive(directory.path()).is_err());
        drop(second);
        let writer = exclusive(directory.path()).unwrap();
        assert!(AssetLease::acquire(directory.path()).is_err());
        drop(writer);
        AssetLease::acquire(directory.path()).unwrap();
    }

    #[test]
    fn an_unwound_load_releases_its_shared_lease() {
        let directory = tempfile::tempdir().unwrap();
        let result = (|| -> Result<()> {
            let _lease = AssetLease::acquire(directory.path())?;
            anyhow::bail!("labelled model-load failure fixture")
        })();
        assert!(result.is_err());
        exclusive(directory.path()).unwrap();
    }

    #[test]
    fn a_symlink_cannot_redirect_the_lock_descriptor() {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        symlink(outside.path(), directory.path().join(".hub-download.lock")).unwrap();
        assert!(AssetLease::acquire(directory.path()).is_err());
    }
}
