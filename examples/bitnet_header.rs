//! Inspect a bounded public safetensors header without loading model tensors.
use anyhow::{Result, ensure};
use std::{fs::File, io::Read, path::PathBuf};
fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/bitnet-header.range".into());
    let mut file = File::open(path)?;
    let mut len = [0; 8];
    file.read_exact(&mut len)?;
    let bytes = usize::try_from(u64::from_le_bytes(len))?;
    ensure!(bytes <= 8 << 20, "safetensors header oversized");
    let mut header = vec![0; bytes];
    file.read_exact(&mut header)?;
    let value: serde_json::Value = serde_json::from_slice(&header)?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}
