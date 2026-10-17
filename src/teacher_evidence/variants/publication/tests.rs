//! Inject slow native writes rather than loading a model or running a container.
use super::*;
use std::{sync::mpsc, time::Duration};
use tokio::sync::oneshot;

async fn isolated(output: PathBuf, capacity: usize) -> Result<Publication> {
    Publication::acquire_with_slots(output, Arc::new(Semaphore::new(capacity))).await
}

#[tokio::test(flavor = "current_thread")]
async fn slow_publication_keeps_the_single_thread_executor_responsive() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let output = temporary.path().to_owned();
    let publication = isolated(output.clone(), 1).await?;
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let work = tokio::spawn(async move {
        publication
            .run(move || {
                let _ = started_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5))?;
                artefact::write_json(
                    &output.join("published.json"),
                    &serde_json::json!({"fixture":true}),
                )
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), started_rx).await??;
    // A timer and an independent ready task must run while the blocking writer
    // is still waiting for our release. Running fsync on this executor would
    // force the writer's five-second timeout before either could progress.
    tokio::time::timeout(Duration::from_secs(2), async {
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(tokio::spawn(async { 41 + 1 }).await.unwrap(), 42);
    })
    .await?;
    assert!(!work.is_finished());
    release_tx.send(())?;
    work.await??;
    assert!(temporary.path().join("published.json").is_file());
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn cancelling_an_awaiter_keeps_the_directory_owned_until_durable_completion() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let output = temporary.path().to_owned();
    let publication = isolated(output.clone(), 1).await?;
    let retained = publication.clone();
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = oneshot::channel();
    let write_path = output.join("published.json");
    let work = tokio::spawn(async move {
        retained
            .run(move || {
                let _ = started_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5))?;
                artefact::write_json(&write_path, &serde_json::json!({"fixture":"durable"}))?;
                let _ = finished_tx.send(());
                Ok(())
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), started_rx).await??;
    work.abort();
    assert!(work.await.unwrap_err().is_cancelled());
    drop(publication);
    let lock = output.join(".content-collection.lock");
    assert!(Lease::acquire(&lock).is_err());
    release_tx.send(())?;
    tokio::time::timeout(Duration::from_secs(2), finished_rx).await??;
    let recovered = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(lease) = Lease::acquire(&lock) {
                break lease;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&fs::read(output.join("published.json"))?)?,
        serde_json::json!({"fixture":"durable"})
    );
    drop(recovered);
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_blocking_workers_keep_their_capacity_reservation() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let publication = isolated(temporary.path().to_owned(), 1).await?;
    let held = publication.clone();
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = tokio::spawn(async move {
        held.run(move || {
            let _ = started_tx.send(());
            release_rx.recv_timeout(Duration::from_secs(5))?;
            Ok(())
        })
        .await
    });
    tokio::time::timeout(Duration::from_secs(2), started_rx).await??;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    let (next_tx, mut next_rx) = oneshot::channel();
    let second = tokio::spawn(async move {
        publication
            .run(move || {
                let _ = next_tx.send(());
                Ok(())
            })
            .await
    });
    // Cancelling the first awaiter cannot create a second blocking slot while
    // its already admitted native work remains active.
    assert!(
        tokio::time::timeout(Duration::from_millis(25), &mut next_rx)
            .await
            .is_err()
    );
    release_tx.send(())?;
    tokio::time::timeout(Duration::from_secs(2), next_rx).await??;
    second.await??;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn a_symlink_cannot_replace_the_ownership_lock() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let output = temporary.path().join("output");
    fs::create_dir(&output)?;
    let sentinel = temporary.path().join("sentinel");
    fs::write(&sentinel, b"unchanged")?;
    std::os::unix::fs::symlink(&sentinel, output.join(".content-collection.lock"))?;
    assert!(isolated(output, 1).await.is_err());
    assert_eq!(fs::read(sentinel)?, b"unchanged");
    Ok(())
}
