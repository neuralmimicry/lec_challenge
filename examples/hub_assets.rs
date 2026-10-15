//! Explicit native Rust Hub download; secrets are read from stdin or HF_TOKEN.
use anyhow::Result;
use clap::Parser;
use lec_distil::hub::{HubDownloadOptions, download_model};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Download hash-verified model data at an immutable Hub commit")]
struct Arguments {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "main")]
    revision: String,
    #[arg(long)]
    destination: PathBuf,
    #[arg(long, default_value_t = 2)]
    concurrency: usize,
    #[arg(long)]
    token_stdin: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Arguments::parse();
    let provenance = download_model(HubDownloadOptions {
        model_id: args.model,
        revision: args.revision,
        destination: args.destination,
        concurrency: args.concurrency,
        token_stdin: args.token_stdin,
    })
    .await?;
    println!("{}", serde_json::to_string_pretty(&provenance)?);
    Ok(())
}
