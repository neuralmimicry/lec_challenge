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
        .args(["metadata", "--locked", "--offline", "--format-version", "1"])
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
        let supplement = supplements
            .resolve(
                package,
                source,
                arguments.fetch_missing && notices.is_empty(),
            )
            .await?;
        if let Some(entry) = &supplement {
            notices.extend(entry.notices.iter().map(|notice| notice.artifact.clone()));
        }
        notices.sort_by(|a, b| a.path.cmp(&b.path));
        notices.dedup_by(|a, b| a.path == b.path);
        if notices.is_empty() {
            gaps.push(format!("{name}-{version}"));
        }
        packages.push(json!({"name":name,"version":version,"licence_expression":package["license"],"repository":package["repository"],"original_notices":notices,"supplemental_upstream_provenance":supplement,"notice_gap":notices.is_empty()}));
    }
    artefact::write_json(
        Path::new("offline_bundle/notices/manifest.json"),
        &json!({"schema_version":2,"cargo_lock_sha256":lock_hash,"packages":packages,"remaining_notice_gaps":gaps,"scope":"Original packaged notices plus exact-commit original upstream notices where workspaces omit packaged files. Includes target-specific dependencies. A licence identifier alone is never substituted for an original notice."}),
    )?;
    println!(
        "Preserved original licence notices for {} resolved packages; {} remaining gaps.",
        packages.len(),
        gaps.len()
    );
    Ok(())
}
