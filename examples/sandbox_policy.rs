//! Print the exact reviewed policy identity without starting a container.
use anyhow::{Result, ensure};
use lec_distil::sandbox::{self, SandboxConfig};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.iter()
            .all(|arg| matches!(arg.as_str(), "--portable" | "--historical-controller")),
        "usage: sandbox_policy [--portable] [--historical-controller]"
    );
    let mut config = if args.iter().any(|arg| arg == "--portable") {
        SandboxConfig::portable()
    } else {
        SandboxConfig::default()
    };
    if args.iter().any(|arg| arg == "--historical-controller") {
        config.controller_version = 1;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"config":config,"policy_sha256":sandbox::policy_sha256(&config)?})
        )?
    );
    Ok(())
}
