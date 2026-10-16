//! Persist bounded runtime pipes independently of a cancellable CLI wait.
use super::*;
use tokio::{io::AsyncRead, sync::watch, task::JoinHandle};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct StreamState {
    pub bytes_seen: u64,
    pub bytes_retained: usize,
    pub overflow: bool,
    pub eof: bool,
    pub error: Option<String>,
}
#[derive(Debug, PartialEq, Eq)]
pub(super) enum PipeEvent {
    Progress,
    HealthyEof,
    Failure,
}
pub(super) fn pipe_event(channel_closed: bool, state: &StreamState) -> PipeEvent {
    if state.overflow || state.error.is_some() || (channel_closed && !state.eof) {
        PipeEvent::Failure
    } else if state.eof {
        PipeEvent::HealthyEof
    } else {
        PipeEvent::Progress
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamEvidence {
    pub path: PathBuf,
    pub sha256: String,
    pub cap: usize,
    pub bytes_seen: u64,
    pub bytes_retained: usize,
    pub overflow: bool,
    pub eof: bool,
    pub error: Option<String>,
}
pub(super) struct PreparedStream {
    file: tokio::fs::File,
    path: PathBuf,
    cap: usize,
}
pub(super) struct Drain {
    worker: JoinHandle<()>,
    pub state: watch::Receiver<StreamState>,
    path: PathBuf,
    cap: usize,
}

pub(super) async fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    file.write_all(bytes).await?;
    file.sync_all().await?;
    tokio::fs::File::open(path.parent().context("durable output parent absent")?)
        .await?
        .sync_all()
        .await?;
    Ok(())
}
pub(super) async fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable output parent absent")?;
    let temporary = parent.join(format!(".{}.pending", uuid::Uuid::new_v4()));
    write_new(&temporary, bytes).await?;
    tokio::fs::hard_link(&temporary, path).await?;
    tokio::fs::remove_file(&temporary).await?;
    tokio::fs::File::open(parent).await?.sync_all().await?;
    Ok(())
}
impl PreparedStream {
    pub async fn new(path: PathBuf, cap: usize) -> Result<Self> {
        ensure!(
            (64..=4_194_304).contains(&cap),
            "invalid runtime pipe bound"
        );
        write_new(&path, &[]).await?;
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .await?;
        Ok(Self { file, path, cap })
    }
    pub fn attach(self, mut reader: impl AsyncRead + Unpin + Send + 'static) -> Drain {
        let Self {
            mut file,
            path,
            cap,
        } = self;
        let (sender, state) = watch::channel(StreamState::default());
        let worker = tokio::spawn(async move {
            let mut state = StreamState::default();
            let mut buffer = [0_u8; 8192];
            let result: std::io::Result<()> = async {
                loop {
                    let count = tokio::io::AsyncReadExt::read(&mut reader, &mut buffer).await?;
                    if count == 0 {
                        file.sync_all().await?;
                        state.eof = true;
                        sender.send_replace(state.clone());
                        return Ok(());
                    }
                    state.bytes_seen = state.bytes_seen.saturating_add(count as u64);
                    let keep = count.min(cap.saturating_sub(state.bytes_retained));
                    if keep > 0 {
                        file.write_all(&buffer[..keep]).await?;
                        file.sync_data().await?;
                        state.bytes_retained += keep;
                    }
                    state.overflow |= keep != count;
                    sender.send_replace(state.clone());
                }
            }
            .await;
            if let Err(error) = result {
                state.error = Some(error.to_string());
                sender.send_replace(state);
            }
        });
        Drain {
            worker,
            state,
            path,
            cap,
        }
    }
}
impl Drain {
    pub async fn finish(mut self, seconds: u64) -> Result<StreamEvidence> {
        let problem = match timeout(Duration::from_secs(seconds), &mut self.worker).await {
            Err(_) => {
                self.worker.abort();
                Some("runtime pipe drainage deadline exceeded".to_owned())
            }
            Ok(Err(error)) => Some(format!("runtime pipe worker failed: {error}")),
            Ok(Ok(())) => None,
        };
        let mut state = self.state.borrow().clone();
        if problem.is_some() {
            state.error = problem;
        }
        let metadata = tokio::fs::metadata(&self.path).await?;
        ensure!(
            metadata.len() <= self.cap as u64,
            "retained runtime pipe exceeded cap"
        );
        let bytes = tokio::fs::read(&self.path).await?;
        Ok(StreamEvidence {
            path: self.path,
            sha256: tasks::hash(&bytes),
            cap: self.cap,
            bytes_seen: state.bytes_seen,
            bytes_retained: bytes.len(),
            overflow: state.overflow,
            eof: state.eof,
            error: state.error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn healthy_pipe_eof_before_pending_cli_wait_does_not_request_a_kill() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let prepared = PreparedStream::new(directory.path().join("audit.part"), 4096).await?;
        let (mut writer, reader) = tokio::io::duplex(128);
        let mut drain = prepared.attach(reader);
        let (wait_finish, wait_ready) = tokio::sync::oneshot::channel::<()>();
        let wait = tokio::spawn(wait_ready);
        writer.write_all(b"{\"trusted\":true}\n").await?;
        writer.shutdown().await?;
        drop(writer);
        timeout(Duration::from_secs(15), async {
            while !drain.state.borrow().eof {
                drain.state.changed().await.expect("drain must report EOF");
            }
        })
        .await?;
        assert!(!wait.is_finished());
        assert_eq!(
            pipe_event(false, &drain.state.borrow()),
            PipeEvent::HealthyEof
        );
        assert!(drain.state.changed().await.is_err());
        assert_eq!(
            pipe_event(true, &drain.state.borrow()),
            PipeEvent::HealthyEof
        );
        wait_finish.send(()).expect("pending wait receiver");
        wait.await??;
        let evidence = drain.finish(5).await?;
        assert!(evidence.eof && !evidence.overflow);
        assert_eq!(fs::read(evidence.path)?, b"{\"trusted\":true}\n");
        Ok(())
    }
    #[tokio::test]
    async fn excess_pipe_bytes_are_drained_but_never_exceed_the_retained_cap() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let prepared = PreparedStream::new(directory.path().join("audit.part"), 64).await?;
        let (mut writer, reader) = tokio::io::duplex(128);
        let drain = prepared.attach(reader);
        let writing = tokio::spawn(async move {
            writer.write_all(&vec![b'x'; 16384]).await?;
            writer.shutdown().await
        });
        let evidence = drain.finish(10).await?;
        writing.await??;
        assert!(
            evidence.eof
                && evidence.overflow
                && evidence.bytes_seen == 16384
                && evidence.bytes_retained == 64
        );
        assert_eq!(fs::metadata(evidence.path)?.len(), 64);
        Ok(())
    }
}
