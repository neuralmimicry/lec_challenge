//! Reconstruct descriptive checkpoint comparisons without model inference.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact::{self, Artefact},
    study_evaluate,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
#[path = "study_comparison/diagnose.rs"]
mod diagnose;

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long = "candidate", required = true)]
    candidates: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    /// Optional separate report about retained grading infrastructure failures.
    #[arg(long)]
    diagnosis_output: Option<PathBuf>,
    #[arg(long)]
    build_receipt: Option<PathBuf>,
    /// Previously retained JSONL runtime events; no runtime command is issued.
    #[arg(long = "runtime-events")]
    runtime_events: Vec<PathBuf>,
}

fn capture(path: &Path) -> Result<Artefact> {
    let root = std::env::current_dir()?.canonicalize()?;
    Artefact::capture(&root, path.canonicalize()?.strip_prefix(&root)?)
}

/// A direct rustc build records its actual executable and explicit input plan.
/// Cargo users do not need this optional standalone-compilation entry point.
fn record_build(arguments: &[String]) -> Result<()> {
    ensure!(
        arguments.len() == 3,
        "usage: --record-build COMMAND_JSON RECEIPT_JSON"
    );
    let command: Vec<String> = serde_json::from_slice(&fs::read(&arguments[1])?)?;
    ensure!(
        command.first().is_some_and(|s| s == "rustc"),
        "standalone build command must name rustc"
    );
    let output = Path::new(&arguments[2]);
    ensure!(!output.exists(), "build receipt already exists");
    let mut dependencies = Vec::new();
    for pair in command.windows(2).filter(|pair| pair[0] == "--extern") {
        dependencies.push(capture(Path::new(
            pair[1]
                .split_once('=')
                .context("invalid --extern argument")?
                .1,
        ))?);
    }
    let version = Command::new("rustc").arg("-vV").output()?;
    ensure!(
        version.status.success(),
        "cannot inspect Rust compiler identity"
    );
    let source = Path::new("examples/study_comparison.rs");
    let diagnostic = Path::new("examples/study_comparison/diagnose.rs");
    ensure!(
        artefact::file_hash(source)? == artefact::hash(include_bytes!("study_comparison.rs"))
            && artefact::file_hash(diagnostic)?
                == artefact::hash(include_bytes!("study_comparison/diagnose.rs")),
        "comparison sources changed since standalone compilation"
    );
    let pinned = lec_distil::engine::pin_executable(
        &std::env::current_exe()?,
        Path::new("state/engines/diagnostics"),
    )?;
    artefact::write_json(
        output,
        &json!({"schema_version":1,"evidence_kind":"actual_standalone_rust_read_only_comparison_build",
        "source":capture(source)?,"diagnosis_source":capture(diagnostic)?,"dependencies":dependencies,
        "compiler_arguments":command,"compiler_command_file":capture(Path::new(&arguments[1]))?,
        "rustc_version_verbose":String::from_utf8(version.stdout)?,"pinned_executable":pinned,
        "cargo_lock":capture(Path::new("Cargo.lock"))?,"cargo_invoked":false,
        "prebuilt_library_used":true,"library_rebuilt_from_current_source":false,
        "inference_performed":false,"containers_started":false}),
    )?;
    Ok(())
}

fn markdown(comparisons: &[Value]) -> String {
    let mut text = String::from(
        "# Descriptive development comparison\n\n| Candidate | Baseline passed | Candidate passed | Baseline errors | Candidate errors | Valid quality comparison | Quality delta (points) |\n| --- | ---: | ---: | ---: | ---: | --- | --- |\n",
    );
    for row in comparisons {
        let candidate = row["candidate"]["path"]
            .as_str()
            .unwrap_or("unknown")
            .replace('|', "\\|");
        let delta = if row["quality_comparison_valid"] == true {
            row["candidate_minus_baseline_percentage_points"]
                .as_f64()
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "Unavailable".into())
        } else {
            "Unavailable".into()
        };
        text.push_str(&format!(
            "| {candidate} | {}/{} | {}/{} | {} | {} | {} | {delta} |\n",
            row["baseline_passed"],
            row["planned_tasks"],
            row["candidate_passed"],
            row["planned_tasks"],
            row["baseline_errors"],
            row["candidate_errors"],
            row["quality_comparison_valid"]
        ));
    }
    text.push_str("\nFailed observations remain in the original denominator. An invalid quality comparison supplies no improvement estimate or learning claim. Raw observed score differences remain descriptive in JSON. No confidence interval, ship decision or significant win is inferred.\n");
    text
}

fn main() -> Result<()> {
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.first().is_some_and(|s| s == "--record-build") {
        return record_build(&raw);
    }
    let args = Arguments::parse();
    let markdown_path = args.output.with_extension("md");
    ensure!(
        !args.output.exists() && !markdown_path.exists(),
        "comparison output already exists; preserve it and choose a new output"
    );
    if let Some(path) = &args.diagnosis_output {
        ensure!(!path.exists(), "diagnosis output already exists");
    }
    let build = args
        .build_receipt
        .as_ref()
        .map(|path| -> Result<_> {
            let file = capture(path)?;
            file.verify(&std::env::current_dir()?)?;
            Ok(file)
        })
        .transpose()?;
    let comparisons = study_evaluate::compare_many(&args.baseline, &args.candidates)?;
    ensure!(!comparisons.is_empty(), "comparison has no candidates");
    let valid = comparisons
        .iter()
        .all(|row| row["quality_comparison_valid"] == true);
    if let Some(path) = &args.diagnosis_output {
        artefact::write_json(
            path,
            &diagnose::collect(&args.candidates, &args.runtime_events)?,
        )?;
    }
    let report = json!({"schema_version":1,"evidence_kind":"independently_verified_descriptive_study_comparisons",
        "status":if valid{"descriptive_comparison_complete"}else{"invalid_quality_comparison"},
        "all_quality_comparisons_valid":valid,"build_receipt":build,"comparisons":comparisons,
        "inference_performed":false,"containers_started":false,"held_out_used":false,"production_eligible":false});
    artefact::write_json(&args.output, &report)?;
    artefact::atomic_write(&markdown_path, markdown(&comparisons).as_bytes())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        valid,
        "quality comparison is invalid; descriptive table and raw evidence were retained"
    );
    Ok(())
}
