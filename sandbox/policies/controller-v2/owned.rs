//! Retain a runtime lease through bounded execution and cleanup even when the
//! requesting future is cancelled. Process death needs the independent ceiling.
use super::runtime_lease::RuntimeLease;
use anyhow::{Context, Result};
use std::future::Future;

pub(super) async fn run<T: Send + 'static>(
    lease: RuntimeLease,
    operation: impl Future<Output = Result<T>> + Send + 'static,
) -> Result<T> {
    tokio::spawn(async move {
        let _lease = lease;
        operation.await
    })
    .await
    .context("owned sandbox operation worker failed")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[tokio::test]
    async fn cancelling_a_request_retains_lease_until_owned_cleanup_finishes() -> Result<()> {
        let lease = super::super::runtime_lease::shared("/bin/true").await?;
        let (started, ready) = tokio::sync::oneshot::channel();
        let (finish, cleanup) = tokio::sync::oneshot::channel();
        let request = tokio::spawn(run(lease, async move {
            let _ = started.send(());
            cleanup.await.context("controlled cleanup cancelled")?;
            Ok(())
        }));
        ready.await?;
        request.abort();
        assert!(request.await.unwrap_err().is_cancelled());
        let mut next = tokio::spawn(super::super::runtime_lease::exclusive("/bin/true"));
        assert!(
            tokio::time::timeout(Duration::from_millis(100), &mut next)
                .await
                .is_err()
        );
        finish
            .send(())
            .map_err(|_| anyhow::anyhow!("owned cleanup did not survive request cancellation"))?;
        drop(tokio::time::timeout(Duration::from_secs(3), next).await???);
        Ok(())
    }
}
