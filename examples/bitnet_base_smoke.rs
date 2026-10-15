//! Actual official-base CPU resource/gradient smoke on labelled synthetic text.
//! It is never a teacher-derived training round or evidence of task learning.
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use clap::Parser;
use lec_distil::{
    artefact,
    bitnet::{self, BitNetStudent},
    model::AdapterConfig,
};
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "offline_bundle/students/bitnet-2b-4t")]
    base: PathBuf,
    #[arg(
        long,
        default_value = "reports/comparative/bitnet/official-base-smoke.json"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = 16)]
    tokens: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (4..=32).contains(&args.tokens),
        "official base smoke requires4..32tokens"
    );
    ensure!(
        !args.output.exists(),
        "existing official-base smoke report retained; use a different output for a new measurement"
    );
    let started = Instant::now();
    let mut model =
        BitNetStudent::load(&args.base, Some(AdapterConfig { rank: 2, alpha: 4. }), 1729)?;
    let loaded_seconds = started.elapsed().as_secs_f64();
    let tokenizer = tokenizers::Tokenizer::from_file(args.base.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let text = bitnet::formatted_prompt(
        "Say hello in one short sentence. This synthetic prompt exercises the original BitNet architecture.",
    );
    let mut ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    ensure!(
        ids.len() >= args.tokens,
        "smoke text has insufficient tokens"
    );
    ids.truncate(args.tokens);
    let input = Tensor::from_vec(ids, (1, args.tokens), &Device::Cpu)?;
    let logits = model.forward(&input)?;
    let finite = logits
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .all(|x| x.is_finite());
    ensure!(finite, "official BitNet forward non-finite");
    let forward_seconds = started.elapsed().as_secs_f64() - loaded_seconds;
    let loss = logits.sqr()?.mean_all()?;
    let loss_value = loss.to_scalar::<f32>()?;
    let gradients = loss.backward()?;
    let mut square = 0_f64;
    for v in model.adapter_vars.values() {
        if let Some(g) = gradients.get(v.as_tensor()) {
            square += f64::from(g.sqr()?.sum_all()?.to_scalar::<f32>()?);
        }
    }
    let gradient_norm = square.sqrt();
    ensure!(
        gradient_norm.is_finite() && gradient_norm > 0.,
        "official BitNet QAT gradient absent"
    );
    let assets = model
        .assets
        .as_ref()
        .context("actual model source identity absent")?;
    #[cfg(unix)]
    let peak_rss_bytes = {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        ensure!(
            unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0,
            "getrusage failed"
        );
        let usage = unsafe { usage.assume_init() };
        #[cfg(target_os = "macos")]
        let bytes = usage.ru_maxrss as u64;
        #[cfg(not(target_os = "macos"))]
        let bytes = (usage.ru_maxrss as u64) * 1024;
        bytes
    };
    #[cfg(not(unix))]
    let peak_rss_bytes = 0_u64;
    let report = serde_json::json!({"schema_version":1,"evidence_kind":"actual_official_bitnet_base_synthetic_operator_smoke","official_full_base":true,"training_data":"labelled_synthetic_text_not_teacher_output","assessment_training":false,"optimiser_steps":0,"source_model":assets,"tokens":args.tokens,"dtype":"F32","device":"Cpu","forward_finite":finite,"loss":loss_value,"gradient_norm":gradient_norm,"load_seconds":loaded_seconds,"forward_seconds":forward_seconds,"total_seconds":started.elapsed().as_secs_f64(),"peak_rss_bytes":peak_rss_bytes,"resource_estimate":bitnet::resource_profile(model.configuration(),true)?,"production_eligible":false,"limitations":["Actual full-size pretrained model forward/backward and memory measurement; no optimiser update, task learning, adapter quality or promotion claim.","Sequence activations and optimiser state may increase memory for longer real training arms."]});
    artefact::write_json(&args.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
