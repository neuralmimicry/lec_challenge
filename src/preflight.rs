//! Host measurements. Advertised runtime support is not enforcement evidence.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path, process::Command};

#[derive(Debug, Serialize, Deserialize)]
pub struct HostPreflight {
    pub observed_unix_ms: i64,
    pub architecture: String,
    pub logical_cpus: usize,
    pub memory_total_kib: Option<u64>,
    pub memory_available_kib: Option<u64>,
    pub cgroup_controllers: Vec<String>,
    pub network_namespace: String,
    pub ipv4_route_table: String,
    pub commands: BTreeMap<String, CommandObservation>,
    pub credential_presence: BTreeMap<String, bool>,
    pub assessment_ready: bool,
    pub blockers: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct CommandObservation {
    pub success: bool,
    pub output: String,
}
fn observe(program: &str, args: &[&str]) -> CommandObservation {
    match Command::new(program).args(args).output() {
        Ok(o) => {
            let mut text = String::from_utf8_lossy(&o.stdout).into_owned();
            text.push_str(&String::from_utf8_lossy(&o.stderr));
            text.truncate(text.floor_char_boundary(16000));
            CommandObservation {
                success: o.status.success(),
                output: text,
            }
        }
        Err(e) => CommandObservation {
            success: false,
            output: e.to_string(),
        },
    }
}
pub fn inspect() -> Result<HostPreflight> {
    let mem = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| {
        mem.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
    };
    let mut commands = BTreeMap::new();
    for (name, args) in [
        ("uname", vec!["-a"]),
        ("rustc", vec!["--version"]),
        ("cargo", vec!["--version"]),
        ("docker", vec!["version"]),
        ("podman", vec!["info", "--format", "json"]),
        (
            "nvidia-smi",
            vec![
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
        ),
        ("df", vec!["-Pk", "."]),
    ] {
        commands.insert(name.to_owned(), observe(name, &args));
    }
    let credential_presence = ["DEEPSEEK_API_KEY", "TOGETHER_API_KEY", "MISTRAL_API_KEY"]
        .into_iter()
        .map(|key| {
            (
                key.into(),
                std::env::var_os(key).is_some_and(|v| !v.is_empty()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut blockers=vec!["Sandbox resource controls and trusted syscall tracing require direct probes; host inventory alone cannot pass them.".into(),"Three approved provider rights/model/price decisions and an explicit live spending cap are required.".into()];
    if credential_presence.values().any(|p| !*p) {
        blockers.push("At least one required provider credential is absent; values are never included in this report.".into());
    }
    if !Path::new("recordings").exists()
        || crate::fleet::ReplayTransport::load("recordings").is_err()
    {
        blockers.push("No complete valid real recording index is present.".into());
    }
    Ok(HostPreflight {
        observed_unix_ms: crate::ledger::unix_millis(),
        architecture: std::env::consts::ARCH.into(),
        logical_cpus: std::thread::available_parallelism()?.get(),
        memory_total_kib: field("MemTotal:"),
        memory_available_kib: field("MemAvailable:"),
        cgroup_controllers: std::fs::read_to_string("/sys/fs/cgroup/cgroup.controllers")
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_owned)
            .collect(),
        network_namespace: std::fs::read_link("/proc/self/ns/net")
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ipv4_route_table: std::fs::read_to_string("/proc/net/route").unwrap_or_default(),
        commands,
        credential_presence,
        assessment_ready: false,
        blockers,
    })
}
