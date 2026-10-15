//! Real pinned native-engine baseline/adapted generation; no quality claim.
use anyhow::Result;
use clap::Parser;
use lec_distil::{
    bitnet::{self, CppRuntimeOptions},
    model::GenerationRequest,
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
    base: PathBuf,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(
        long,
        default_value = "offline_bundle/students/bitnet-2b-4t/model.i2_s.gguf"
    )]
    gguf: PathBuf,
    #[arg(long, default_value = "reports/comparative/bitnet/cpp-smoke.json")]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    bitnet::cpp::record_engine(
        std::path::Path::new("state/engines-src/bitnet"),
        std::path::Path::new("state/engines/bitnet-build"),
        std::path::Path::new("state/engines/bitnet-build/engine-manifest.json"),
    )?;
    let mut runtime = bitnet::BitNetCppRuntime::load_with_options(
        &a.base,
        a.checkpoint.as_deref(),
        CppRuntimeOptions {
            gguf: a.gguf,
            ..Default::default()
        },
    )?;
    let generated = runtime.generate(GenerationRequest {
        prompt: "Say hello in one short sentence.".into(),
        max_new_tokens: 32,
        seed: 1729,
        temperature: 0.,
        top_p: 1.,
        timeout_seconds: 120,
    })?;
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"actual_pinned_bitnet_cpp_smoke","generation":generated,"execution":runtime.execution_metadata(),"quality_evaluation":false,"production_eligible":false});
    lec_distil::artefact::write_json(&a.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
