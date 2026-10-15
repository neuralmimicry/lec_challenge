//! Technical QA only: synthetic token tensors through the actual pinned base.
//! This publishes no adapter and is not the required provider-output training round.
use anyhow::Result;
use std::path::PathBuf;
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let base = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("offline_bundle/base"));
    let tokens = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(64);
    let result = lec_distil::train::native_gradient_smoke(&base, tokens)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
