//! Trusted, finite kernel-enforcement probes; no generated programme runs here.
use serde_json::json;
use std::{
    fs,
    process::{Command, Stdio},
    time::Duration,
};
fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "settings" => {
            let read = |p: &str| fs::read_to_string(p).unwrap_or_else(|e| format!("ERROR:{e}"));
            // The image grants everyone write permission to this existing file.
            // EROFS therefore proves the mount policy, rather than merely UID DAC.
            let root_write_errno = fs::OpenOptions::new()
                .write(true)
                .open("/opt/rootfs-probe")
                .err()
                .and_then(|e| e.raw_os_error());
            println!(
                "{}",
                json!({"uid_status":read("/proc/self/status").lines().find(|l|l.starts_with("Uid:")).unwrap_or("missing"),"memory_max":read("/sys/fs/cgroup/memory.max").trim(),"swap_max":read("/sys/fs/cgroup/memory.swap.max").trim(),"cpu_max":read("/sys/fs/cgroup/cpu.max").trim(),"pids_max":read("/sys/fs/cgroup/pids.max").trim(),"root_write_denied":root_write_errno==Some(30),"root_write_errno":root_write_errno,"network_none":read("/proc/net/route").lines().count()<=1})
            );
        }
        "memory" => {
            let mut held = Vec::new();
            for i in 0..1024 {
                let mut v = vec![0u8; 2 * 1024 * 1024];
                for j in (0..v.len()).step_by(4096) {
                    v[j] = (i % 251 + 1) as u8;
                }
                held.push(v);
                std::hint::black_box(&held);
            }
            println!("memory limit NOT enforced");
            std::process::exit(72);
        }
        "pids" => {
            let mut children = vec![];
            let mut denied = false;
            for _ in 0..256 {
                match Command::new("/bin/sleep")
                    .arg("5")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                {
                    Ok(c) => children.push(c),
                    Err(e) => {
                        denied = e.raw_os_error() == Some(11);
                        break;
                    }
                }
            }
            let count = children.len();
            for mut c in children {
                let _ = c.kill();
                let _ = c.wait();
            }
            println!(
                "{}",
                json!({"creation_denied":denied,"children_created":count})
            );
            if !denied {
                std::process::exit(73);
            }
        }
        "timeout" => {
            println!("trusted_timeout_probe_started");
            use std::io::Write;
            std::io::stdout().flush().expect("flush start evidence");
            std::thread::sleep(Duration::from_secs(30));
            println!("deadline NOT enforced");
            std::process::exit(74);
        }
        _ => {
            eprintln!("unknown trusted probe");
            std::process::exit(64);
        }
    }
}
