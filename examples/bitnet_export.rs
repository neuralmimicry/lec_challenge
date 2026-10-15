//! Native GGUF export; the original model remains a baseline until real QAT.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
    base: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let report = lec_distil::bitnet::export::export(&a.base, a.checkpoint.as_deref(), &a.output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
