//! Cached, network-free image construction and actual provenance.
use super::*;
pub(super) async fn build_cached_image(root: &Path, cfg: &SandboxConfig) -> Result<String> {
    for (path, expected) in [
        (
            "sandbox/audit_v3.rs",
            include_bytes!("../../../sandbox/policies/v3/audit.rs").as_slice(),
        ),
        (
            "sandbox/Containerfile.v3",
            include_bytes!("../../../sandbox/policies/v3/Containerfile").as_slice(),
        ),
        (
            "sandbox/seccomp_v3.json",
            include_bytes!("../../../sandbox/policies/v3/seccomp.json").as_slice(),
        ),
    ] {
        ensure!(
            tokio::fs::read(root.join(path)).await? == expected,
            "policy-three build input {path} differs from frozen source"
        );
    }
    let portable_manifest = root.join(format!(
        "offline_bundle/images/policy-2/{}/manifest.json",
        cfg.architecture
    ));
    let base_manifest = if portable_manifest.try_exists()? {
        portable_manifest
    } else {
        root.join("offline_bundle/images/manifest.json")
    };
    let provenance: Value = serde_json::from_slice(&tokio::fs::read(&base_manifest).await?)?;
    let base = provenance["image_id"]
        .as_str()
        .context("cached parent sandbox image ID absent")?;
    let architecture = provenance["architecture"].as_str().unwrap_or("x86_64");
    ensure!(
        architecture == cfg.architecture && base.trim_start_matches("sha256:").len() == 64,
        "cached parent sandbox architecture/ID differs"
    );
    let inspected = timeout(
        Duration::from_secs(METADATA_SECS),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(["image", "inspect", "--format", "{{.Id}}", base])
            .output(),
    )
    .await
    .context("cached parent image inspection deadline")??;
    ensure!(
        inspected.status.success()
            && String::from_utf8_lossy(&inspected.stdout)
                .trim()
                .trim_start_matches("sha256:")
                == base.trim_start_matches("sha256:"),
        "required verified parent sandbox image is absent; explicitly restore its offline archive first"
    );
    let start = Instant::now();
    let build_arg = format!("LEC_SANDBOX_BASE={base}");
    let arguments = [
        "build",
        "--pull=never",
        "--network=none",
        "--build-arg",
        &build_arg,
        "-f",
        "sandbox/Containerfile.v3",
        "-t",
        &cfg.image,
        ".",
    ];
    let output = timeout(
        Duration::from_secs(300),
        Command::new(&cfg.runtime)
            .kill_on_drop(true)
            .args(arguments)
            .current_dir(root)
            .output(),
    )
    .await
    .context("cached policy-three image build exceeded five-minute deadline")??;
    let directory = root
        .join("reports/sandbox-build")
        .join(uuid::Uuid::new_v4().to_string());
    tokio::fs::create_dir_all(&directory).await?;
    stream::write_new(&directory.join("stdout.log"), &output.stdout).await?;
    stream::write_new(&directory.join("stderr.log"), &output.stderr).await?;
    let image = if output.status.success() {
        Some(policy_evidence(cfg).await?.1)
    } else {
        None
    };
    let report = json!({"schema_version":3,"policy_version":3,"controller_version":3,"architecture":cfg.architecture,"image_id":image,
        "cached_parent_image_id":base,"cached_parent_manifest":base_manifest,"parent_provenance":provenance,
        "containerfile_sha256":tasks::hash(include_bytes!("../../../sandbox/policies/v3/Containerfile")),
        "arguments":arguments,"network":"none","pull":"never","duration_ms":start.elapsed().as_millis(),
        "exit_code":output.status.code(),"source":"explicit cached image build; no registry or package fallback"});
    stream::write_new(
        &directory.join("report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )
    .await?;
    ensure!(
        output.status.success(),
        "cached policy-three image build failed; logs retained at {}",
        directory.display()
    );
    tracing::info!(receipt=%directory.join("report.json").display(),"verified cached policy-three image built");
    image.context("built image identity absent")
}
