//! Bounded blocking units retain directory ownership through durable writes.
use super::*;
use std::sync::{Arc, OnceLock};
use tokio::sync::Semaphore;

const MAXIMUM_BLOCKING_UNITS: usize = 2;
fn slots() -> Arc<Semaphore> {
    static SLOTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
    SLOTS
        .get_or_init(|| Arc::new(Semaphore::new(MAXIMUM_BLOCKING_UNITS)))
        .clone()
}
#[derive(Clone)]
pub(super) struct Publication {
    lease: Arc<Lease>,
    slots: Arc<Semaphore>,
}
async fn admitted<T: Send + 'static>(
    slots: Arc<Semaphore>,
    lease: Option<Arc<Lease>>,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let permit = slots
        .acquire_owned()
        .await
        .context("content publication admission closed")?;
    tokio::task::spawn_blocking(move || {
        // Both guards belong to the worker, so cancelling its awaiting future
        // cannot admit another writer or unlock a directory during fsync.
        let _permit = permit;
        let _lease = lease;
        work()
    })
    .await
    .context("content publication worker failed")?
}
impl Publication {
    pub async fn acquire(output: PathBuf) -> Result<Self> {
        Self::acquire_with_slots(output, slots()).await
    }
    async fn acquire_with_slots(output: PathBuf, slots: Arc<Semaphore>) -> Result<Self> {
        let lease = admitted(slots.clone(), None, move || {
            fs::create_dir_all(&output)?;
            Ok(Arc::new(Lease::acquire(
                &output.join(".content-collection.lock"),
            )?))
        })
        .await?;
        Ok(Self { lease, slots })
    }
    pub async fn run<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<T> {
        admitted(self.slots.clone(), Some(self.lease.clone()), work).await
    }
}

#[cfg(test)]
mod tests;
