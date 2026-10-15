//! Verify actual public base inference without provider credentials.
//! This example does not train an assessment adapter or claim an improvement.
use anyhow::Result;
use lec_distil::model::{GenerationRequest, ModelRuntime};
use std::path::PathBuf;

fn main() -> Result<()> {
    let base = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("offline_bundle/base"));
    let started = std::time::Instant::now();
    let mut runtime = ModelRuntime::load(&base, None)?;
    eprintln!(
        "loaded actual pinned base in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    let result=runtime.generate(GenerationRequest{prompt:"Write a Rust function add(a: i64, b: i64) -> i64. Return only the function source code.".into(),max_new_tokens:96,seed:1701,temperature:0.,top_p:1.,timeout_seconds:600})?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
