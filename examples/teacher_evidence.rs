//! Collect or verify the separate local-study train-only teacher evidence.
use anyhow::Result;
use clap::{Parser, Subcommand};
use lec_distil::{
    model::{ExecutionDevice, InferenceDtype, RuntimeOptions},
    teacher_evidence::{
        self, CollectOptions, ContentCollectOptions, ContentKind, RolloutCollectOptions,
        RolloutMode,
    },
};
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Rollout {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "offline_bundle/base")]
        student_base: PathBuf,
        #[arg(long)]
        student_checkpoint: Option<PathBuf>,
        #[arg(long)]
        reuse_student_attempts_from: Option<PathBuf>,
        #[arg(long)]
        bitnet: bool,
        #[arg(long)]
        self_distillation: bool,
        #[arg(long)]
        teacher: Vec<String>,
        #[arg(long, default_value_t = 4)]
        maximum_tasks: usize,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long)]
        teacher_context_tokens: Option<usize>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
    },
    Collect {
        #[arg(long)]
        registry: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        comparison: Vec<PathBuf>,
        #[arg(long)]
        model: Vec<String>,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long)]
        cuda: bool,
        #[arg(long,default_value="f32",value_parser=["f32","f16"])]
        dtype: String,
    },
    Verify {
        path: PathBuf,
    },
    Content {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "process,context,tool_trajectory"
        )]
        content: Vec<String>,
        #[arg(long, default_value_t = 48)]
        maximum_tasks: usize,
        #[arg(long, default_value_t = 768)]
        max_tokens: usize,
        #[arg(long)]
        context_tokens: Option<usize>,
        #[arg(long, default_value_t = 1729)]
        seed: u64,
        #[arg(long)]
        cuda: bool,
        #[arg(long,default_value="f32",value_parser=["f32","f16"])]
        dtype: String,
    },
}
#[tokio::main]
async fn main() -> Result<()> {
    let bundle = match Args::parse().command {
        Command::Rollout {
            bundle,
            output,
            student_base,
            student_checkpoint,
            reuse_student_attempts_from,
            bitnet,
            self_distillation,
            teacher,
            maximum_tasks,
            max_tokens,
            teacher_context_tokens,
            seed,
        } => {
            teacher_evidence::collect_rollouts(RolloutCollectOptions {
                bundle,
                output,
                student_base,
                student_checkpoint,
                student_kind: if bitnet {
                    lec_distil::distillation::DistillationStudent::TernaryBitnet
                } else {
                    lec_distil::distillation::DistillationStudent::ConventionalQwenLora
                },
                mode: if self_distillation {
                    RolloutMode::SelfDistillation
                } else {
                    RolloutMode::OnPolicy
                },
                teacher_ids: teacher,
                teacher_runtime: RuntimeOptions::default(),
                teacher_context_tokens,
                student_runtime: RuntimeOptions::default(),
                max_tokens,
                seed,
                maximum_tasks,
                reuse_student_attempts_from,
            })
            .await?
        }
        Command::Collect {
            registry,
            output,
            comparison,
            model,
            max_tokens,
            seed,
            cuda,
            dtype,
        } => {
            teacher_evidence::collect(CollectOptions {
                registry,
                comparisons: comparison,
                output,
                runtime: runtime(cuda, &dtype),
                max_tokens,
                seed,
                models: model,
            })
            .await?
        }
        Command::Verify { path } => {
            tokio::task::spawn_blocking(move || teacher_evidence::load_verified(&path)).await??
        }
        Command::Content {
            bundle,
            output,
            content,
            maximum_tasks,
            max_tokens,
            context_tokens,
            seed,
            cuda,
            dtype,
        } => {
            let content = content
                .iter()
                .map(|c| match c.as_str() {
                    "process" => Ok(ContentKind::Process),
                    "context" => Ok(ContentKind::Context),
                    "tool_trajectory" => Ok(ContentKind::ToolTrajectory),
                    _ => anyhow::bail!("unknown generated content {c}"),
                })
                .collect::<Result<Vec<_>>>()?;
            let report_path = output.join("content-report.json");
            let collected = teacher_evidence::collect_content(ContentCollectOptions {
                bundle,
                output,
                content,
                maximum_tasks,
                max_tokens,
                context_tokens,
                seed,
                runtime: runtime(cuda, &dtype),
            })
            .await?;
            let report: serde_json::Value = serde_json::from_slice(&std::fs::read(report_path)?)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            anyhow::ensure!(
                report["healthy_complete"] == true,
                "content collection was incomplete or unhealthy; durable report and partial evidence retained"
            );
            collected
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"evidence_kind":bundle.evidence_kind,"teachers":bundle.teachers.len(),"samples":bundle.samples.len(),"preferences":bundle.preferences.len(),"train_corpus_sha256":bundle.train_corpus_sha256,"promotable":false})
        )?
    );
    Ok(())
}

fn runtime(cuda: bool, dtype: &str) -> RuntimeOptions {
    RuntimeOptions {
        device: if cuda {
            ExecutionDevice::Cuda { ordinal: 0 }
        } else {
            ExecutionDevice::Cpu
        },
        dtype: if dtype == "f16" {
            InferenceDtype::F16
        } else {
            InferenceDtype::F32
        },
    }
}
