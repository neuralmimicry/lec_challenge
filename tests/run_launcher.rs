//! Exercise only operational dispatch, using isolated command fixtures.
//! No model, compiler build, container or network operation is performed.
#![cfg(unix)]
use anyhow::Result;
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Command, Output},
};

fn script(path: &Path, bytes: &str) -> Result<()> {
    fs::write(path, bytes)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))?;
    Ok(())
}

fn exercise(marker: Option<bool>, bundled: Option<bool>, source: bool) -> Result<(Output, String)> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::write(root.join("run.sh"), include_bytes!("../run.sh"))?;
    fs::create_dir(root.join("commands"))?;
    fs::create_dir_all(root.join("offline_bundle/bin"))?;
    // These shell fixtures stand in for native command launchers only. All
    // assertions and setup are Rust; the shell does no assessment computation.
    script(
        &root.join("commands/cargo"),
        "#!/bin/sh\nprintf '%s\\n' cargo \"$@\" > \"$LEC_LAUNCH_QA_LOG\"\nexit 23\n",
    )?;
    if let Some(valid) = bundled {
        let version = if valid { "0" } else { "29" };
        script(
            &root.join("offline_bundle/bin/lec-distil"),
            &format!(
                "#!/bin/sh\nif [ \"${{1:-}}\" = --version ]; then exit {version}; fi\nprintf '%s\\n' bundled \"$@\" > \"$LEC_LAUNCH_QA_LOG\"\nexit 17\n"
            ),
        )?;
    }
    if let Some(pointer_file) = marker {
        if pointer_file {
            fs::write(root.join(".git"), "gitdir: /unused/test-pointer\n")?;
        } else {
            fs::create_dir(root.join(".git"))?;
        }
    }
    let log = root.join("dispatch.txt");
    let paths = std::env::join_paths([root.join("commands"), "/usr/bin".into(), "/bin".into()])?;
    let output = Command::new("/bin/bash")
        .arg(root.join("run.sh"))
        .args([
            "gate",
            "--candidate",
            "path with spaces",
            "--incumbent",
            "base",
        ])
        .current_dir(std::env::temp_dir())
        .env("PATH", paths)
        .env("LEC_LAUNCH_QA_LOG", &log)
        .env("LEC_BUILD_FROM_SOURCE", if source { "1" } else { "0" })
        .output()?;
    Ok((output, fs::read_to_string(log)?))
}

#[test]
fn git_checkouts_use_current_source_and_preserve_arguments_and_exit() -> Result<()> {
    for pointer_file in [false, true] {
        let (output, log) = exercise(Some(pointer_file), Some(true), false)?;
        assert_eq!(output.status.code(), Some(23));
        assert_eq!(
            log,
            "cargo\nrun\n--locked\n--offline\n--release\n--\ngate\n--candidate\npath with spaces\n--incumbent\nbase\n"
        );
    }
    Ok(())
}

#[test]
fn extracted_delivery_uses_its_compatible_prebuilt_binary() -> Result<()> {
    let (output, log) = exercise(None, Some(true), false)?;
    assert_eq!(output.status.code(), Some(17));
    assert!(log.starts_with("bundled\ngate\n"));
    Ok(())
}

#[test]
fn source_override_and_missing_or_incompatible_delivery_use_cargo() -> Result<()> {
    for (bundled, source) in [(Some(true), true), (Some(false), false), (None, false)] {
        let (output, log) = exercise(None, bundled, source)?;
        assert_eq!(output.status.code(), Some(23));
        assert!(log.starts_with("cargo\nrun\n--locked\n--offline\n--release\n"));
    }
    Ok(())
}
