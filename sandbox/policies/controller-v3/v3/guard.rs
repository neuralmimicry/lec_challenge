//! Dedicated deadline thread with preopened exact kernel controls.
use super::*;
pub(super) enum GuardSignal {
    Finished,
    Stop(&'static str),
}
pub(super) struct GuardResult {
    pub(super) control: control::PinnedCgroup,
    pub(super) kill: Option<control::CachedKillEvidence>,
    pub(super) reason: &'static str,
}
pub(super) fn guard(
    mut control: control::PinnedCgroup,
    released: Instant,
    seconds: u64,
) -> Result<(
    mpsc::Sender<GuardSignal>,
    tokio::sync::oneshot::Receiver<GuardResult>,
)> {
    let (sender, receiver) = mpsc::channel();
    let (done, result) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("lec-cgroup-deadline".into())
        .spawn(move || {
            let deadline = released + Duration::from_secs(seconds);
            let (kill, reason) =
                match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                    Ok(GuardSignal::Finished) => (None, "runtime_cli_finished"),
                    Ok(GuardSignal::Stop(reason)) => (Some(control.kill(released)), reason),
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        (Some(control.kill(released)), "execution_wall_deadline")
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => (
                        Some(control.kill(released)),
                        "controller_signal_channel_closed",
                    ),
                };
            let _ = done.send(GuardResult {
                control,
                kill,
                reason,
            });
        })?;
    Ok((sender, result))
}
