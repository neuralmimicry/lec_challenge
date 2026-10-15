//! Public pinned BitNet student retrieval; no executable model code is loaded.
use anyhow::Result;
use lec_distil::hub::{HubDownloadOptions, download_bitnet_student};
#[tokio::main]
async fn main() -> Result<()> {
    let provenance = download_bitnet_student(HubDownloadOptions {
        model_id: "microsoft/bitnet-b1.58-2B-4T".into(),
        revision: "04c3b9ad9361b824064a1f25ea60a8be9599b127".into(),
        destination: "offline_bundle/students/bitnet-2b-4t".into(),
        concurrency: 4,
        token_stdin: false,
    })
    .await?;
    println!("{}", serde_json::to_string_pretty(&provenance)?);
    Ok(())
}
