//! The spawning OS thread remains alive for the server's entire lifetime.
//! Linux parent-death signalling therefore covers both process SIGKILL and
//! accidental loss of this dedicated supervisor, without pool-thread races.
use super::*;
use std::{
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
};

pub(super) struct Supervisor {
    control: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<Option<String>>>,
    pub pid: u32,
}
struct KillOnDrop(std::process::Child);
impl KillOnDrop {
    fn stop(&mut self) {
        let _ = self.0.kill();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.0.try_wait() {
                Ok(Some(_)) => return,
                Err(_) => return,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        eprintln!(
                            "BitNet.cpp child{} did not acknowledge SIGKILL before cleanup deadline",
                            self.0.id()
                        );
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }
    }
}
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        self.stop();
    }
}
impl Supervisor {
    #[cfg(not(target_os = "linux"))]
    pub fn spawn(_command: Command) -> Result<Self> {
        anyhow::bail!(
            "crash-safe BitNet.cpp process supervision currently requires Linux parent-death signalling; use the native Rust backend on this platform"
        )
    }
    #[cfg(target_os = "linux")]
    pub fn spawn(mut command: Command) -> Result<Self> {
        let (control, receive) = mpsc::channel();
        let (ready, readiness) = mpsc::sync_channel(1);
        let status = Arc::new(Mutex::new(None));
        let child_status = status.clone();
        let parent = std::process::id() as libc::pid_t;
        let thread = std::thread::Builder::new()
            .name("bitnet-native-supervisor".into())
            .spawn(move || {
                #[cfg(target_os = "linux")]
                {
                    use std::os::unix::process::CommandExt;
                    // SAFETY: pre_exec calls only async-signal-safe libc operations,
                    // without allocation or locks. The parent PID check closes the
                    // window where the spawning process dies before prctl is set.
                    unsafe {
                        command.pre_exec(move || {
                            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                                return Err(std::io::Error::last_os_error());
                            }
                            if libc::getppid() != parent {
                                return Err(std::io::Error::from_raw_os_error(libc::ECHILD));
                            }
                            Ok(())
                        });
                    }
                }
                let child = match command.spawn() {
                    Ok(child) => child,
                    Err(error) => {
                        let _ = ready.send(Err(error.to_string()));
                        return;
                    }
                };
                let mut child = KillOnDrop(child);
                let _ = ready.send(Ok(child.0.id()));
                loop {
                    match receive.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            return;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    match child.0.try_wait() {
                        Ok(Some(exit)) => {
                            if let Ok(mut status) = child_status.lock() {
                                *status = Some(exit.to_string());
                            }
                            return;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            if let Ok(mut status) = child_status.lock() {
                                *status = Some(format!("child status failed: {error}"));
                            }
                            return;
                        }
                    }
                }
            })?;
        let pid = match readiness.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(pid)) => pid,
            Ok(Err(error)) => {
                let _ = thread.join();
                anyhow::bail!("cannot spawn pinned native server: {error}");
            }
            Err(error) => {
                drop(control);
                let _ = thread.join();
                return Err(error.into());
            }
        };
        Ok(Self {
            control: Some(control),
            thread: Some(thread),
            status,
            pid,
        })
    }
    pub fn exited(&self) -> Option<String> {
        self.status.lock().ok().and_then(|status| status.clone())
    }
}
impl Drop for Supervisor {
    fn drop(&mut self) {
        if let Some(control) = self.control.take() {
            let _ = control.send(());
        }
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            eprintln!(
                "BitNet native supervisor panicked; parent-death signalling remains enforced"
            );
        }
    }
}
