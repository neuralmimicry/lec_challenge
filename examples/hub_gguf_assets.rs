//! Explicit selected GGUF acquisition with independently pinned original data.
use anyhow::Result;
use clap::Parser;
use lec_distil::hub::{HubGgufDownloadOptions, HubGgufSource, download_gguf};
use std::path::PathBuf;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    model: String,
    #[arg(long)]
    revision: String,
    #[arg(long)]
    filename: String,
    #[arg(long)]
    source_model: String,
    #[arg(long)]
    source_revision: String,
    #[arg(long)]
    destination: PathBuf,
    #[arg(long)]
    sha256: Option<String>,
    #[arg(long)]
    bytes: Option<u64>,
    #[arg(long)]
    token_stdin: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arguments::parse();
    let proof = download_gguf(HubGgufDownloadOptions {
        model_id: args.model,
        revision: args.revision,
        filename: args.filename,
        destination: args.destination,
        token_stdin: args.token_stdin,
        source: HubGgufSource {
            model_id: args.source_model,
            revision: args.source_revision,
        },
        expected_sha256: args.sha256,
        expected_bytes: args.bytes,
    })
    .await?;
    println!("{}", serde_json::to_string_pretty(&proof)?);
    Ok(())
}
