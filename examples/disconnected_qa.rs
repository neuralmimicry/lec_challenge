//! Rehearse a delivery in a fresh directory and a real empty network namespace.
//! Missing provider evidence must still fail the assessment, while proving that
//! local offline components execute without keys or the development directory.
use anyhow::{Context, Result, ensure};
use lec_distil::{artefact, package};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

fn main() -> Result<()> {
    let args = std::env::args_os().collect::<Vec<_>>();
    if args.get(1).is_some_and(|a| a == "--inside") {
        ensure!(
            args.len() == 4,
            "internal namespace launcher arguments missing"
        );
        ensure!(
            Command::new("ip")
                .args(["link", "set", "lo", "up"])
                .status()?
                .success(),
            "cannot enable isolated loopback for local HTTP"
        );
        let original = Path::new(&args[3]);
        let hidden = tempfile::tempdir_in(std::env::current_dir()?)?;
        ensure!(
            Command::new("mount")
                .arg("--bind")
                .arg(hidden.path())
                .arg(original)
                .status()?
                .success(),
            "cannot hide original workspace in private mount namespace"
        );
        ensure!(
            !original.join("Cargo.toml").exists(),
            "original workspace remained visible"
        );
        artefact::write_json(
            Path::new("reports/namespace-isolation.json"),
            &json!({"network_and_mount_namespaces":"unshare -Urnm","loopback_enabled_for_local_endpoint":true,"original_workspace_hidden":true,"original_workspace":original}),
        )?;
        use std::os::unix::process::CommandExt;
        return Err(Command::new(&args[2])
            .args(["demo", "--offline"])
            .exec()
            .into());
    }
    ensure!(args.len() == 2, "usage: disconnected_qa DELIVERY.tar.gz");
    let archive = PathBuf::from(&args[1]).canonicalize()?;
    let original = std::env::current_dir()?;
    let fresh = tempfile::tempdir()?;
    ensure!(
        Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(fresh.path())
            .status()?
            .success(),
        "delivery extraction failed"
    );
    ensure!(
        !fresh.path().join("state").exists(),
        "delivery leaked existing application state"
    );
    let binary = fresh.path().join("offline_bundle/bin/lec-distil");
    ensure!(binary.exists(), "bundled binary missing");
    let before: package::BundleManifest = serde_json::from_slice(&fs::read(
        fresh.path().join("reports/bundle-manifest.json"),
    )?)?;
    // Hash the extracted inputs before starting any bundled application code.
    // The digest excludes reports/state, so it has no archive self-reference.
    let verified_inventory = package::verify_extracted_assets(fresh.path(), &before)?;
    // SAFETY: getuid reads the kernel identity and does not access Rust memory.
    let uid = unsafe { libc::getuid() };
    let mut command = Command::new("unshare");
    command
        .args(["-Urnm", "--"])
        .arg(std::env::current_exe()?)
        .arg("--inside")
        .arg(&binary)
        .arg(&original)
        .current_dir(fresh.path())
        .env(
            "CONTAINER_HOST",
            format!("unix:///run/user/{uid}/podman/podman.sock"),
        )
        .env("XDG_CACHE_HOME", fresh.path().join("empty-cache"))
        .stdout(Stdio::from(fs::File::create(
            original.join("reports/disconnected-qa-command.json"),
        )?))
        .stderr(Stdio::inherit());
    for key in [
        "DEEPSEEK_API_KEY",
        "TOGETHER_API_KEY",
        "MISTRAL_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ] {
        command.env_remove(key);
    }
    let started = Instant::now();
    let status = command
        .status()
        .context("start real network-namespace rehearsal")?;
    let host: Value =
        serde_json::from_slice(&fs::read(fresh.path().join("reports/preflight.json"))?)?;
    let demo: Value = serde_json::from_slice(&fs::read(fresh.path().join("reports/demo.json"))?)?;
    let drill: Value = serde_json::from_slice(&fs::read(fresh.path().join("reports/drill.json"))?)?;
    let namespace: Value = serde_json::from_slice(&fs::read(
        fresh.path().join("reports/namespace-isolation.json"),
    )?)?;
    let no_routes = host["ipv4_route_table"]
        .as_str()
        .is_some_and(|s| s.lines().count() <= 1);
    let no_keys = host["credential_presence"]
        .as_object()
        .is_some_and(|m| m.values().all(|v| v == false));
    ensure!(
        no_routes && no_keys && namespace["original_workspace_hidden"] == true,
        "network/credential/workspace isolation was not observed"
    );
    let phases = demo["phases"]
        .as_array()
        .context("offline demo phases missing")?;
    for name in ["corpus", "base_assets", "base_serving_qa"] {
        ensure!(
            phases
                .iter()
                .any(|p| p["phase"] == name && p["status"] == "complete"),
            "offline local component {name} failed"
        );
    }
    ensure!(
        phases
            .iter()
            .any(|p| p["phase"] == "offline_container_image"
                && (p["status"] == "verified_existing"
                    || p["status"] == "restored_verified_bundle")),
        "offline OCI image was not verified"
    );
    let cases = drill["cases"]
        .as_array()
        .context("offline drill cases missing")?;
    for name in [
        "grader_gaming",
        "escape_attempts",
        "planted_heldout_task",
        "round_budget_exceeded",
    ] {
        ensure!(
            cases
                .iter()
                .any(|c| c["name"] == name && c["status"] == "PASSED"),
            "offline required local drill {name} failed"
        );
    }
    let complete = status.success() && demo["status"] == "COMPLETE" && drill["complete"] == true;
    ensure!(
        complete || (!status.success() && demo["status"] == "INCOMPLETE"),
        "failed offline run was misreported as success"
    );
    let evidence = original.join("reports/disconnected-evidence");
    fs::create_dir_all(&evidence)?;
    let mut copied = Vec::new();
    for name in [
        "preflight.json",
        "namespace-isolation.json",
        "demo.json",
        "drill.json",
        "eval.json",
        "cost-summary.json",
    ] {
        let from = fresh.path().join("reports").join(name);
        if from.exists() {
            let to = evidence.join(name);
            fs::copy(from, &to)?;
            copied.push(artefact::Artefact::capture(
                &original,
                to.strip_prefix(&original)?,
            )?);
        }
    }
    let report = json!({"schema_version":1,"evidence_kind":"actual_fresh_archive_network_namespace_execution","quality_assurance_passed":true,"full_assessment_complete":complete,"archive_sha256":artefact::file_hash(&archive)?,"binary_sha256":before.binary_sha256,"source_commit":before.source_commit,"verified_input_asset_inventory_sha256":verified_inventory,"incoming_asset_inventory_verified_before_execution":true,"fresh_application_state":true,"empty_application_cache":true,"provider_credentials_absent":no_keys,"ipv4_routes_absent":no_routes,"network_namespace":host["network_namespace"],"runtime_connection":"Host Podman service through an existing Unix socket; candidate containers retain no-network policy. Runtime image cache was preloaded, not cleared.","exit_code":status.code(),"elapsed_seconds":started.elapsed().as_secs_f64(),"evidence":copied,"limitation":"A correctly failing incomplete assessment is local offline QA only, never a completed provider-trained rehearsal."});
    artefact::write_json(Path::new("reports/disconnected-qa.json"), &report)?;
    if complete {
        artefact::write_json(Path::new("reports/disconnected-rehearsal.json"), &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
