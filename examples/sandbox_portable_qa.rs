//! Native portable container evidence. x86-64 execution never verifies ARM hardware.
use anyhow::{Result, ensure};
use clap::Parser;
use lec_distil::sandbox::{self, SandboxConfig};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    build: bool,
    #[arg(long, default_value = "reports/sandbox-portable-qa")]
    output: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let output = args.output;
    ensure!(
        !output.join("report.json").exists(),
        "preserve the previous QA report; select a fresh --output"
    );
    fs::create_dir_all(&output)?;
    let mut config = SandboxConfig::portable();
    config.report_dir = output.join("runs");
    if args.build {
        eprintln!("Building explicitly requested native portable sandbox image");
        config.expected_image_id = sandbox::build_image_for(Path::new("."), &config).await?;
    }
    let result = async {
        let preflight = sandbox::preflight(&config).await?;
        config.expected_image_id = preflight.image_id.clone();
        let security = sandbox::drill_security(&config).await?;
        let gaming = sandbox::drill_gaming(&config).await?;
        Ok::<_, anyhow::Error>(json!({"passed":true,"evidence_kind":"actual_native_portable_sandbox_qa_not_model_quality_evidence", "architecture":config.architecture,"arm_hardware_verified":config.architecture=="aarch64", "policy_version":config.policy_version,"controller_version":config.controller_version,"policy_sha256":sandbox::policy_sha256(&config)?,"preflight":preflight,"security":security,"gaming":gaming}))
    }.await;
    let report = match &result {
        Ok(report) => report.clone(),
        Err(error) => {
            json!({"passed":false,"evidence_kind":"incomplete_native_portable_sandbox_qa", "architecture":config.architecture,"policy_version":config.policy_version,"controller_version":config.controller_version,"arm_hardware_verified":false,"failure":format!("{error:#}")})
        }
    };
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    result.map(|_| ())
}
