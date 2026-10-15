//! Prepare local CUDA 12.9 headers for glibc 2.43 without changing the toolkit.
//! Imported NVIDIA headers remain in excluded application state, never delivery.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact;
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn copy_headers(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if fs::metadata(entry.path())?.is_dir() {
            copy_headers(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        arguments.len() == 2,
        "usage: cuda_compat /usr/local/cuda-12.9 state/cuda-compat"
    );
    let toolkit = PathBuf::from(&arguments[0]).canonicalize()?;
    let destination = PathBuf::from(&arguments[1]);
    fs::create_dir_all(&destination)?;
    let destination = destination.canonicalize()?;
    ensure!(
        !destination.starts_with(&toolkit),
        "compatibility overlay must be outside the installed toolkit"
    );
    let original = toolkit.join("include/crt/math_functions.h");
    let header = fs::read_to_string(&original)?;
    let original_sha256 = artefact::file_hash(&original)?;
    let mut patched = header;
    let rules = [
        "rsqrt(double x)",
        "rsqrtf(float x)",
        "sinpi(double x)",
        "sinpif(float x)",
        "cospi(double x)",
        "cospif(float x)",
    ];
    for prototype in rules {
        let before = format!("{prototype};");
        ensure!(
            patched.matches(&before).count() == 1,
            "unexpected installed CUDA math prototype: {prototype}"
        );
        patched = patched.replace(&before, &format!("{prototype} noexcept (true);"));
    }
    copy_headers(&toolkit.join("include"), &destination.join("include"))?;
    artefact::atomic_write(
        &destination.join("include/crt/math_functions.h"),
        patched.as_bytes(),
    )?;
    // Only headers are copied. Libraries and tools retain their local installation.
    for name in ["bin", "lib64"] {
        let path = destination.join(name);
        if fs::symlink_metadata(&path).is_ok() {
            fs::remove_file(&path)?;
        }
        std::os::unix::fs::symlink(toolkit.join(name), path)?;
    }
    let qa = destination.join("compatibility-check.cu");
    fs::write(
        &qa,
        "// Header compilation QA only; no model or application logic.\n#include <cuda_runtime.h>\n",
    )?;
    let nvcc = toolkit.join("bin/nvcc");
    let include_flag = format!("-I{}", destination.join("include").display());
    let capability = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "75".into());
    ensure!(
        !capability.is_empty() && capability.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid CUDA compute capability"
    );
    let compiler = std::env::var_os("NVCC_CCBIN").unwrap_or_else(|| "g++".into());
    let compiled = Command::new(&nvcc)
        .args(["--ptx", "-std=c++17", "-ccbin"])
        .arg(compiler)
        .arg(format!("--gpu-architecture=sm_{capability}"))
        .arg(&include_flag)
        .arg(&qa)
        .arg("-o")
        .arg(destination.join("compatibility-check.ptx"))
        .env("NVCC_PREPEND_FLAGS", &include_flag)
        .output()
        .context("compile original CUDA headers against the observed glibc")?;
    ensure!(
        compiled.status.success(),
        "local CUDA compatibility check failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let version = Command::new(&nvcc).arg("--version").output()?;
    ensure!(
        version.status.success(),
        "cannot observe installed nvcc version"
    );
    let report = json!({"evidence_kind":"local_cuda_header_compatibility_overlay_and_actual_compile_check","toolkit":toolkit,"overlay":destination,"original_header_sha256":original_sha256,"patched_header_sha256":artefact::file_hash(&destination.join("include/crt/math_functions.h"))?,"rules":rules,"modification":"Add matching noexcept(true) to six original CUDA host math declarations; numerical kernels are unchanged.","nvcc_version":String::from_utf8(version.stdout)?,"ptx_sha256":artefact::file_hash(&destination.join("compatibility-check.ptx"))?,"system_headers_modified":false,"redistribution":"Copied proprietary headers remain in excluded local state; the deliverable contains only this Rust preparation tool and patch rules."});
    artefact::write_json(&destination.join("manifest.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
