//! Preserve original packaged notices and explicitly pinned upstream supplements.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::artefact;
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
#[path = "dependency_notices/upstream.rs"]
mod upstream;

fn source_evidence(root: &Path, destination: &Path) -> Result<Vec<artefact::Artefact>> {
    let mut evidence = Vec::new();
    for filename in [
        "SOURCE-CARGO-VCS-INFO.json",
        "SOURCE-CARGO-TOML.orig",
        "SOURCE-UPSTREAM-TREE.json",
        "SOURCE-WORKSPACE-CARGO.toml",
        "SOURCE-CURRENT-REPOSITORY.json",
        "SOURCE-CURRENT-UPSTREAM-TREE.json",
        "SOURCE-LICENCE-HISTORY.json",
        "SOURCE-RESEARCH.json",
    ] {
        let path = destination.join(filename);
        if path.is_file() {
            evidence.push(artefact::Artefact::capture(root, path.strip_prefix(root)?)?);
        }
    }
    Ok(evidence)
}
#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    supplemental: Option<PathBuf>,
    /// Explicitly fetch original notices from exact packaged upstream commits.
    #[arg(long)]
    fetch_missing: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let metadata = Command::new("cargo")
        .args([
            "metadata",
            "--locked",
            "--offline",
            "--all-features",
            "--format-version",
            "1",
        ])
        .output()?;
    ensure!(metadata.status.success(), "offline Cargo metadata failed");
    let value: Value = serde_json::from_slice(&metadata.stdout)?;
    let root = std::env::current_dir()?;
    let output = root.join("offline_bundle/notices");
    fs::create_dir_all(&output)?;
    let lock_hash = artefact::file_hash(Path::new("Cargo.lock"))?;
    let supplemental = arguments
        .supplemental
        .unwrap_or_else(|| output.join("supplemental.json"));
    let mut supplements = upstream::Supplements::load(&root, &supplemental, &lock_hash)?;
    let mut packages = Vec::new();
    let mut gaps = Vec::new();
    let mut acquisition_errors = Vec::new();
    for package in value["packages"]
        .as_array()
        .context("metadata packages missing")?
    {
        let name = package["name"]
            .as_str()
            .context("dependency name missing")?;
        let version = package["version"]
            .as_str()
            .context("dependency version missing")?;
        let manifest = PathBuf::from(
            package["manifest_path"]
                .as_str()
                .context("manifest path missing")?,
        );
        let source = manifest.parent().context("dependency directory missing")?;
        let dest = output.join(format!("{name}-{version}"));
        fs::create_dir_all(&dest)?;
        let mut notices = Vec::new();
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let filename = entry.file_name();
            let upper = filename.to_string_lossy().to_ascii_uppercase();
            if entry.file_type()?.is_file()
                && ["LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT"]
                    .iter()
                    .any(|p| upper.starts_with(p))
            {
                let target = dest.join(filename);
                fs::copy(entry.path(), &target)?;
                notices.push(artefact::Artefact::capture(
                    &root,
                    target.strip_prefix(&root)?,
                )?);
            }
        }
        if let Some(file) = package["license_file"].as_str() {
            let from = source.join(file);
            if from.is_file() {
                let target = dest.join("DECLARED-LICENCE");
                fs::copy(from, &target)?;
                notices.push(artefact::Artefact::capture(
                    &root,
                    target.strip_prefix(&root)?,
                )?);
            }
        }
        let supplement = match supplements
            .resolve(
                package,
                source,
                arguments.fetch_missing && notices.is_empty(),
            )
            .await
        {
            Ok(value) => value,
            Err(error) => {
                let evidence = source_evidence(&root, &dest)?;
                eprintln!("notices: {name}-{version} remains incomplete: {error:#}");
                acquisition_errors.push(json!({"name":name,"version":version,"error":format!("{error:#}"),"retained_original_source_evidence":evidence}));
                None
            }
        };
        if let Some(entry) = &supplement {
            notices.extend(entry.notices.iter().map(|notice| notice.artifact.clone()));
        }
        notices.sort_by(|a, b| a.path.cmp(&b.path));
        notices.dedup_by(|a, b| a.path == b.path);
        if notices.is_empty() {
            gaps.push(format!("{name}-{version}"));
        }
        let source_evidence = source_evidence(&root, &dest)?;
        packages.push(json!({"name":name,"version":version,"licence_expression":package["license"],"repository":package["repository"],"original_notices":notices,"supplemental_upstream_provenance":supplement,"retained_original_source_evidence":source_evidence,"notice_gap":notices.is_empty()}));
    }
    artefact::write_json(
        Path::new("offline_bundle/notices/manifest.json"),
        &json!({"schema_version":2,"cargo_lock_sha256":lock_hash,"packages":packages,"remaining_notice_gaps":gaps,"notice_acquisition_errors":acquisition_errors,"scope":"Original packaged notices plus exact-commit original upstream notices where workspaces omit packaged files. Includes all-feature and target-specific dependencies. A licence identifier alone is never substituted for an original notice."}),
    )?;
    println!(
        "Inventoried {} resolved packages across all features; {} notice gaps and {} acquisition/verification errors.",
        packages.len(),
        gaps.len(),
        acquisition_errors.len()
    );
    ensure!(
        gaps.is_empty() && acquisition_errors.is_empty(),
        "dependency notice inventory is incomplete; retained explicit gaps and original evidence in offline_bundle/notices/manifest.json"
    );
    Ok(())
}
