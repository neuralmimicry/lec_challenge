//! Exercise the real raw-publication/reuse boundary without any model weights.
use super::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};
use tokio::sync::oneshot;

#[tokio::test(flavor = "current_thread")]
async fn a_completed_cancelled_generation_is_durable_and_never_resampled() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let root = temporary.path().to_owned();
    let publication = Publication::acquire(root.clone()).await?;
    let plan_path = root.join("plan.json");
    artefact::write_json(
        &plan_path,
        &serde_json::json!({"evidence_kind":"synthetic_publication_fixture"}),
    )?;
    let plan = Artefact::capture(&root, "plan.json")?;
    let raw_path = root.join("raw-generation.json");
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let work_path = raw_path.clone();
    let work_plan = plan.clone();
    let (started_tx, started_rx) = oneshot::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = oneshot::channel();
    let work = tokio::spawn(async move {
        publication
            .run(move || {
                let result = raw_or_generate(&work_path, &work_plan, (), |_| {
                    count.fetch_add(1, Ordering::SeqCst);
                    let _ = started_tx.send(());
                    release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                    (
                        (),
                        GenerationOutcome {
                            generation: None,
                            generation_error: Some(
                                "controlled synthetic generation failure".into(),
                            ),
                            execution: None,
                        },
                    )
                });
                let _ = finished_tx.send(());
                result.map(|_| ())
            })
            .await
    });
    tokio::time::timeout(Duration::from_secs(2), started_rx).await??;
    work.abort();
    assert!(work.await.unwrap_err().is_cancelled());
    release_tx.send(())?;
    tokio::time::timeout(Duration::from_secs(2), finished_rx).await??;
    let publication = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(publication) = Publication::acquire(root.clone()).await {
                break publication;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await?;
    let original_bytes = fs::read(&raw_path)?;
    let saved_path = raw_path.clone();
    let saved_plan = plan.clone();
    let unexpected = calls.clone();
    let raw = publication
        .run(move || {
            raw_or_generate(&saved_path, &saved_plan, (), |_| {
                unexpected.fetch_add(1, Ordering::SeqCst);
                panic!("a durable generation must never be requested again")
            })
            .map(|(_, raw)| raw)
        })
        .await?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(fs::read(&raw_path)?, original_bytes);
    assert!(raw.generation.is_none());
    assert!(raw.generation_error.is_some());

    // A different frozen request must fail without replacing either the raw
    // failure or the plan identity, even when a new generator is available.
    let mut other_plan = plan;
    other_plan.sha256 = "b".repeat(64);
    let count = calls.clone();
    let wrong_path = raw_path.clone();
    assert!(
        publication
            .run(move || {
                raw_or_generate(&wrong_path, &other_plan, (), |_| {
                    count.fetch_add(1, Ordering::SeqCst);
                    panic!("a request mismatch cannot authorise resampling")
                })
                .map(|_| ())
            })
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(fs::read(raw_path)?, original_bytes);
    Ok(())
}
