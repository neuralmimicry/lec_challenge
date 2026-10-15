//! Package a previously exported immutable OCI image without downloading assets.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact;
use serde_json::json;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn main() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    ensure!(
        args.len() == 3,
        "usage: cargo run --example package_sandbox -- ARCHIVE IMMUTABLE_IMAGE_ID"
    );
    let archive = PathBuf::from(&args[1]);
    let image = args[2].to_str().context("image ID is not UTF-8")?;
    ensure!(
        image.trim_start_matches("sha256:").len() == 64,
        "immutable image ID required"
    );
    let manifest = json!({"schema_version":1,"asset_kind":"oci_archive","archive":archive.file_name().context("archive filename missing")?.to_string_lossy(),"bytes":fs::metadata(&archive)?.len(),"sha256":artefact::file_hash(&archive)?,"image_id":image,"base_image":"docker.io/library/rust@sha256:b36c246742b4d323472588f789601be34eb4af053255e2af53969078ee8dcec7","containerfile_sha256":artefact::file_hash(Path::new("sandbox/Containerfile"))?,"dependency_lock_sha256":artefact::file_hash(Path::new("sandbox/deps/Cargo.lock"))?,"licences":{"local_supervisor_and_probe":"Apache-2.0","rust_toolchain":"Apache-2.0 OR MIT, plus upstream third-party notices retained in image","serde_json":"MIT OR Apache-2.0","serde_core":"MIT OR Apache-2.0","itoa":"MIT OR Apache-2.0","memchr":"Unlicense OR MIT","zmij":"MIT","additional_locked_crates":"See LICENCES.md and upstream Cargo package notices preserved in the image","debian_packages":"Multiple package licences; /usr/share/doc/*/copyright retained in the OCI root filesystem"},"acquisition":"Explicit local online bootstrap; no provider calls","offline_load":"podman load --input offline_bundle/images/lec-rust-sandbox.tar","verification":"Check archive SHA-256 before load, then require exact image ID and execute enforcement preflight"});
    let destination = archive
        .parent()
        .context("archive parent missing")?
        .join("manifest.json");
    artefact::write_json(&destination, &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
