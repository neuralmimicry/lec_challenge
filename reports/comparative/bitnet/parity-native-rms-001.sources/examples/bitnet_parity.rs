//! Thin command wrapper for the reusable frozen numerical evidence producer.
use anyhow::{Result, ensure};
use clap::Parser;
use lec_distil::bitnet::parity_run::{self, ParityRunOptions};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
    base: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value = "reports/comparative/bitnet/parity.json")]
    output: PathBuf,
    #[arg(long)]
    engine_manifest: Option<PathBuf>,
    #[arg(long)]
    gguf: Option<PathBuf>,
    #[arg(long)]
    quantisation_v2: bool,
    #[arg(long)]
    evaluation: Vec<PathBuf>,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let root = std::env::current_dir()?.canonicalize()?;
    let current = std::env::current_exe()?.canonicalize()?;
    let pinned = lec_distil::engine::pin_executable(&current, &root.join("state/engines"))?;
    if current != pinned.path {
        let status = std::process::Command::new(&pinned.path)
            .args(std::env::args_os().skip(1))
            .status()?;
        ensure!(
            status.success(),
            "pinned numerical diagnostic failed with {status}"
        );
        return Ok(());
    }
    let report = parity_run::run(ParityRunOptions {
        base: args.base,
        checkpoint: args.checkpoint,
        output: args.output,
        engine_manifest: args.engine_manifest,
        gguf: args.gguf,
        quantisation_v2: args.quantisation_v2,
        evaluation: args.evaluation,
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        report["passed"] == true,
        "native I2_S parity failed; full raw numerical evidence retained"
    );
    Ok(())
}
