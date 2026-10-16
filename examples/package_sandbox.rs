//! Package an exported OCI image with its actual build and inherited notices.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{artefact, sandbox::SandboxConfig};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Arguments {
    archive: PathBuf,
    image_id: String,
    #[arg(long)]
    portable: bool,
    #[arg(long, default_value_t = 3)]
    policy: u32,
    #[arg(long)]
    build_receipt: Option<PathBuf>,
}
fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(
        matches!(arguments.policy, 1..=3),
        "unsupported sandbox image policy"
    );
    ensure!(
        arguments.image_id.trim_start_matches("sha256:").len() == 64
            && arguments
                .image_id
                .trim_start_matches("sha256:")
                .bytes()
                .all(|b| b.is_ascii_hexdigit()),
        "immutable image ID required"
    );
    let config = SandboxConfig::portable();
    ensure!(
        !arguments.portable || arguments.policy >= 2,
        "portable requires policy two or three"
    );
    let architecture = if arguments.policy == 1 {
        "x86_64"
    } else {
        config.architecture.as_str()
    };
    let containerfile = match arguments.policy {
        1 => "sandbox/Containerfile",
        2 => "sandbox/Containerfile.v2",
        _ => "sandbox/Containerfile.v3",
    };
    let mut manifest = json!({"schema_version":1,"asset_kind":"oci_archive","policy_version":arguments.policy,"architecture":architecture,
        "archive":arguments.archive.file_name().context("archive filename missing")?.to_string_lossy(),
        "bytes":fs::metadata(&arguments.archive)?.len(),"sha256":artefact::file_hash(&arguments.archive)?,"image_id":arguments.image_id,
        "containerfile_sha256":artefact::file_hash(Path::new(containerfile))?,"dependency_lock_sha256":artefact::file_hash(Path::new("sandbox/deps/Cargo.lock"))?,
        "licences":{"local_supervisor_and_probe":"Apache-2.0","rust_toolchain":"Apache-2.0 OR MIT; upstream notices retained in the image",
            "serde_json":"MIT OR Apache-2.0","serde_core":"MIT OR Apache-2.0","itoa":"MIT OR Apache-2.0","memchr":"Unlicense OR MIT","zmij":"MIT",
            "additional_locked_crates":"See LICENCES.md and upstream Cargo package notices preserved in the image",
            "debian_packages":"Multiple licences; /usr/share/doc/*/copyright retained in the OCI root filesystem"},
        "offline_load":"Select the exact policy/architecture manifest, verify the complete archive hash, then podman load without network",
        "verification":"Require immutable image identity and actual native enforcement preflight"});
    if arguments.policy == 3 {
        let receipt_path = arguments.build_receipt.context(
            "policy-three packaging requires --build-receipt from the actual cached image build",
        )?;
        let receipt: Value = serde_json::from_slice(&fs::read(&receipt_path)?)?;
        ensure!(
            receipt["policy_version"] == 3
                && receipt["architecture"] == architecture
                && receipt["image_id"] == arguments.image_id
                && receipt["network"] == "none"
                && receipt["pull"] == "never"
                && receipt["containerfile_sha256"] == manifest["containerfile_sha256"],
            "build receipt differs from exported image/protocol"
        );
        manifest["cached_parent_image_id"] = receipt["cached_parent_image_id"].clone();
        manifest["base_image"] = receipt["parent_provenance"]["base_image"].clone();
        manifest["parent_provenance"] = receipt["parent_provenance"].clone();
        manifest["build_receipt"] =
            json!({"path":receipt_path,"sha256":artefact::file_hash(&receipt_path)?});
        manifest["acquisition"] = json!(
            "Explicit local cached build with --pull=never --network=none; inherited toolchain/dependency notices retained"
        );
        manifest["candidate_watchdog_seconds"] = json!(20);
        manifest["outer_startup_crash_seconds"] = json!(80);
    } else {
        manifest["base_image"] = json!(if arguments.policy == 2 {
            "docker.io/library/rust@sha256:114c7a4425406451c2866b6aafe69fe29b1b298832db1277d411ac73c82d04d6"
        } else {
            "docker.io/library/rust@sha256:b36c246742b4d323472588f789601be34eb4af053255e2af53969078ee8dcec7"
        });
        manifest["acquisition"] =
            json!("Previously exported historical image; no new acquisition implied");
    }
    let destination = arguments
        .archive
        .parent()
        .context("archive parent missing")?
        .join("manifest.json");
    ensure!(
        !destination.try_exists()?,
        "manifest already exists; preserve it and use a new policy/architecture directory"
    );
    artefact::write_json(&destination, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
