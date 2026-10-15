//! Execute labelled micro-model QAT numerical checks, never assessment training.
use anyhow::Result;
use std::path::Path;
fn main() -> Result<()> {
    let report = lec_distil::bitnet::qa::run_operator_qa(Path::new("reports/comparative/bitnet"))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
