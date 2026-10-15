//! Local delivery packaging. Completeness is tied to evidence, never filenames.
use crate::artefact::{self, Artefact};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Component, Path},
    process::Command,
};

// Reports and application state are deliberately outside this immutable input
// set: rehearsal creates new reports/state, and the manifest cannot hash itself.
const ASSET_ROOTS: [&str; 14] = [
    "offline_bundle/base",
    "offline_bundle/comparative",
    "offline_bundle/images",
    "offline_bundle/bin",
    "offline_bundle/notices",
    "offline_bundle/evidence",
    "checkpoints",
    "recordings",
    "data",
    "config",
    "tasks",
    "grader_private",
    "sandbox",
    "run.sh",
];

#[derive(Serialize, Deserialize)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub complete: bool,
    pub assessment_status: String,
    pub architecture: String,
    pub binary_sha256: String,
    pub asset_inventory_sha256: String,
    pub assets: Vec<Artefact>,
    pub source_commit: String,
    pub limitations: String,
}

/// Canonicalise by relative path, including size and checksum for every input.
/// Directory enumeration order must never affect a rehearsal identity.
pub fn asset_inventory_sha256(assets: &[Artefact]) -> Result<String> {
    let mut sorted = assets.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| left.path.cmp(&right.path));
    let mut previous: Option<&Path> = None;
    for item in &sorted {
        ensure!(
            !item.path.as_os_str().is_empty()
                && item
                    .path
                    .components()
                    .all(|part| matches!(part, Component::Normal(_)))
                && ASSET_ROOTS.iter().any(|root| item.path.starts_with(root)),
            "inventory path is outside the immutable runtime input set"
        );
        ensure!(
            previous != Some(&item.path),
            "duplicate runtime inventory path"
        );
        ensure!(
            item.sha256.len() == 64 && item.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid runtime asset checksum"
        );
        previous = Some(&item.path);
    }
    Ok(artefact::hash(&serde_json::to_vec(&sorted)?))
}

fn capture_asset_inventory(root: &Path) -> Result<Vec<Artefact>> {
    let mut assets = Vec::new();
    for relative in ASSET_ROOTS {
        let path = root.join(relative);
        // symlink_metadata also sees dangling links, which must not evade walk.
        match fs::symlink_metadata(&path) {
            Ok(_) => walk(root, &path, &mut assets)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    assets.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(assets)
}

/// Verify every extracted immutable input before executing its bundled binary.
/// Capturing the actual tree also rejects unlisted extra files and symlinks.
pub fn verify_extracted_assets(root: &Path, manifest: &BundleManifest) -> Result<String> {
    ensure!(
        manifest.schema_version == 2,
        "bundle inventory schema is unsupported"
    );
    let expected = asset_inventory_sha256(&manifest.assets)?;
    ensure!(
        expected == manifest.asset_inventory_sha256,
        "bundle manifest inventory checksum mismatch"
    );
    let actual = capture_asset_inventory(root)?;
    ensure!(
        asset_inventory_sha256(&actual)? == expected,
        "extracted runtime inventory differs from the incoming bundle manifest"
    );
    ensure!(
        actual.iter().any(|asset| {
            asset.path == Path::new("offline_bundle/bin/lec-distil")
                && asset.sha256 == manifest.binary_sha256
        }),
        "bundled binary identity is missing from the verified inventory"
    );
    Ok(expected)
}

fn rehearsal_identity_matches(
    report: &serde_json::Value,
    binary_sha256: &str,
    source_commit: &str,
    inventory_sha256: &str,
) -> bool {
    report["schema_version"] == 1
        && report["evidence_kind"] == "actual_fresh_archive_network_namespace_execution"
        && report["full_assessment_complete"] == true
        && report["quality_assurance_passed"] == true
        && report["exit_code"] == 0
        && report["binary_sha256"] == binary_sha256
        && report["source_commit"] == source_commit
        && report["verified_input_asset_inventory_sha256"] == inventory_sha256
        && report["incoming_asset_inventory_verified_before_execution"] == true
        && report["fresh_application_state"] == true
        && report["empty_application_cache"] == true
        && report["provider_credentials_absent"] == true
        && report["ipv4_routes_absent"] == true
}
fn walk(root: &Path, path: &Path, assets: &mut Vec<Artefact>) -> Result<()> {
    ensure!(
        !fs::symlink_metadata(path)?.file_type().is_symlink(),
        "symlinks are not bundled: {}",
        path.display()
    );
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            walk(root, &entry?.path(), assets)?;
        }
    } else {
        assets.push(Artefact::capture(root, path.strip_prefix(root)?)?);
    }
    Ok(())
}
pub fn build() -> Result<serde_json::Value> {
    let root = std::env::current_dir()?;
    let binary = root.join("target/release/lec-distil");
    ensure!(binary.exists(), "build release binary before packaging");
    fs::create_dir_all(root.join("offline_bundle/bin"))?;
    fs::copy(&binary, root.join("offline_bundle/bin/lec-distil"))?;
    if root.join("state/ledger.sqlite").exists() {
        let evidence = root.join("offline_bundle/evidence");
        fs::create_dir_all(&evidence)?;
        let snapshot = evidence.join("acquisition-ledger.sqlite");
        if snapshot.exists() {
            fs::remove_file(&snapshot)?;
        }
        let database = rusqlite::Connection::open_with_flags(
            root.join("state/ledger.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?;
        database.busy_timeout(std::time::Duration::from_secs(15))?;
        database.execute(
            "VACUUM INTO ?1",
            [snapshot.to_str().context("snapshot path is not UTF-8")?],
        )?;
        fs::File::open(&snapshot)?.sync_all()?;
    }
    let assets = capture_asset_inventory(&root)?;
    let inventory_sha256 = asset_inventory_sha256(&assets)?;
    let demo: serde_json::Value = fs::read(root.join("reports/demo.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    let commit = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    ensure!(commit.status.success(), "cannot record source commit");
    let source_commit = String::from_utf8(commit.stdout)?.trim().to_owned();
    let binary_sha256 = artefact::file_hash(&binary)?;
    let rehearsal_path = root.join("reports/disconnected-rehearsal.json");
    let mut rehearsal_valid = false;
    if rehearsal_path.exists() {
        let r: serde_json::Value = serde_json::from_slice(&fs::read(&rehearsal_path)?)?;
        rehearsal_valid =
            rehearsal_identity_matches(&r, &binary_sha256, &source_commit, &inventory_sha256);
        let evidence: Vec<Artefact> = serde_json::from_value(r["evidence"].clone())?;
        rehearsal_valid &= evidence.len() >= 4;
        for item in &evidence {
            item.verify(&root)?;
        }
        for (name, field, expected) in [
            ("demo.json", "status", serde_json::json!("COMPLETE")),
            ("drill.json", "complete", serde_json::json!(true)),
        ] {
            let Some(item) = evidence
                .iter()
                .find(|a| a.path.file_name().is_some_and(|n| n == name))
            else {
                rehearsal_valid = false;
                continue;
            };
            let value: serde_json::Value = serde_json::from_slice(&fs::read(item.verify(&root)?)?)?;
            rehearsal_valid &= value[field] == expected;
        }
    }
    let complete = demo["status"] == "COMPLETE" && rehearsal_valid;
    let manifest=BundleManifest{schema_version:2,complete,assessment_status:if complete{"completed_with_verified_rehearsal"}else{"incomplete_real_provider_and_checkpoint_evidence"}.into(),architecture:std::env::consts::ARCH.into(),binary_sha256,asset_inventory_sha256:inventory_sha256,assets,source_commit,limitations:"docs/limitations.md; assets include original licences. A local QA replay is not a complete provider-trained disconnected rehearsal.".into()};
    artefact::write_json(&root.join("reports/bundle-manifest.json"), &manifest)?;
    fs::create_dir_all(root.join("deliverables"))?;
    let temp = tempfile::tempdir()?;
    let archive = temp.path().join("lec-distil-delivery.tar.gz");
    let status = Command::new("tar")
        .args([
            "--exclude=./.git",
            "--exclude=./.idea",
            "--exclude=./.env*",
            "--exclude=./target",
            "--exclude=./state",
            "--exclude=./deliverables",
            "--exclude=./reports/video-frames",
            "--exclude=./Principal_AI_Engineer_Prebuild_Pack.zip",
            "--exclude=./design.pdf",
            "-I",
            "gzip -1",
            "-cf",
        ])
        .arg(&archive)
        .arg(".")
        .current_dir(&root)
        .status()
        .context("invoke standard tar/gzip encoder")?;
    ensure!(status.success(), "delivery archive failed");
    let destination = root.join("deliverables/lec-distil-delivery.tar.gz");
    fs::copy(&archive, &destination)?;
    let hash = artefact::file_hash(&destination)?;
    eprintln!("package: encoding ZIP delivery with the same source and runtime assets");
    let zip_archive = temp.path().join("lec-distil-delivery.zip");
    let zip_status = Command::new("zip")
        .args(["-q", "-r", "-1"])
        .arg(&zip_archive)
        .arg(".")
        .args([
            "-x",
            ".git/*",
            ".idea/*",
            ".env*",
            "target/*",
            "state/*",
            "deliverables/*",
            "reports/video-frames/*",
            "Principal_AI_Engineer_Prebuild_Pack.zip",
            "design.pdf",
        ])
        .current_dir(&root)
        .status()
        .context("invoke standard ZIP encoder")?;
    ensure!(zip_status.success(), "ZIP delivery archive failed");
    let zip_destination = root.join("deliverables/lec-distil-delivery.zip");
    fs::copy(&zip_archive, &zip_destination)?;
    let zip_hash = artefact::file_hash(&zip_destination)?;
    artefact::atomic_write(
        &root.join("deliverables/SHA256SUMS"),
        format!("{hash}  lec-distil-delivery.tar.gz\n{zip_hash}  lec-distil-delivery.zip\n")
            .as_bytes(),
    )?;
    Ok(
        serde_json::json!({"archive":destination,"bytes":fs::metadata(destination)?.len(),"sha256":hash,"zip_archive":zip_destination,"zip_bytes":fs::metadata(zip_destination)?.len(),"zip_sha256":zip_hash,"assessment_complete":complete,"manifest":"reports/bundle-manifest.json"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_manifest(root: &Path) -> BundleManifest {
        fs::create_dir_all(root.join("offline_bundle/bin")).unwrap();
        fs::create_dir_all(root.join("offline_bundle/base")).unwrap();
        fs::create_dir_all(root.join("config")).unwrap();
        fs::write(
            root.join("offline_bundle/bin/lec-distil"),
            b"unit fixture binary",
        )
        .unwrap();
        fs::write(
            root.join("offline_bundle/base/model.safetensors"),
            b"unit fixture tensors",
        )
        .unwrap();
        fs::write(
            root.join("config/training.toml"),
            b"unit fixture configuration",
        )
        .unwrap();
        let assets = capture_asset_inventory(root).unwrap();
        BundleManifest {
            schema_version: 2,
            complete: false,
            assessment_status: "unit_fixture_incomplete".into(),
            architecture: "test".into(),
            binary_sha256: artefact::file_hash(&root.join("offline_bundle/bin/lec-distil"))
                .unwrap(),
            asset_inventory_sha256: asset_inventory_sha256(&assets).unwrap(),
            assets,
            source_commit: "unit-fixture-source".into(),
            limitations: "Unit fixtures are not runnable binaries or assessment weights.".into(),
        }
    }

    #[test]
    fn inventory_identity_is_order_independent_and_rejects_duplicate_paths() {
        let root = tempfile::tempdir().unwrap();
        let mut manifest = fixture_manifest(root.path());
        manifest.assets.reverse();
        assert_eq!(
            asset_inventory_sha256(&manifest.assets).unwrap(),
            manifest.asset_inventory_sha256
        );
        let moved = manifest.assets[0].path.clone();
        manifest.assets[0].path = Path::new("config/another.toml").into();
        assert_ne!(
            asset_inventory_sha256(&manifest.assets).unwrap(),
            manifest.asset_inventory_sha256
        );
        manifest.assets[0].path = moved;
        manifest.assets.push(manifest.assets[0].clone());
        assert!(asset_inventory_sha256(&manifest.assets).is_err());
    }

    #[test]
    fn extracted_inputs_reject_tampering_missing_and_unlisted_assets_before_execution() {
        let root = tempfile::tempdir().unwrap();
        let manifest = fixture_manifest(root.path());
        assert_eq!(
            verify_extracted_assets(root.path(), &manifest).unwrap(),
            manifest.asset_inventory_sha256
        );
        let weights = root.path().join("offline_bundle/base/model.safetensors");
        fs::write(&weights, b"unit changed tensors").unwrap();
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
        fs::write(&weights, b"unit fixture tensors").unwrap();
        fs::remove_file(&weights).unwrap();
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
        fs::write(&weights, b"unit fixture tensors").unwrap();
        let extra = root.path().join("offline_bundle/base/unlisted.json");
        fs::write(&extra, b"unlisted runtime input").unwrap();
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
        fs::remove_file(extra).unwrap();
        std::os::unix::fs::symlink(
            "model.safetensors",
            root.path().join("offline_bundle/base/alias"),
        )
        .unwrap();
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
    }

    #[test]
    fn manifest_digest_and_binary_identity_cannot_override_extracted_input_checks() {
        let root = tempfile::tempdir().unwrap();
        let mut manifest = fixture_manifest(root.path());
        manifest.asset_inventory_sha256 = artefact::hash(b"wrong inventory");
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
        manifest.asset_inventory_sha256 = asset_inventory_sha256(&manifest.assets).unwrap();
        manifest.binary_sha256 = artefact::hash(b"wrong binary");
        assert!(verify_extracted_assets(root.path(), &manifest).is_err());
        manifest.assets[0].path = "../outside".into();
        assert!(asset_inventory_sha256(&manifest.assets).is_err());
    }

    #[test]
    fn changed_runtime_assets_invalidate_rehearsal_even_when_binary_and_source_match() {
        let root = tempfile::tempdir().unwrap();
        let manifest = fixture_manifest(root.path());
        // This fixture tests completeness eligibility only, not real rehearsal.
        let mut report = serde_json::json!({
            "schema_version":1,
            "evidence_kind":"actual_fresh_archive_network_namespace_execution",
            "full_assessment_complete":true,
            "quality_assurance_passed":true,
            "exit_code":0,
            "binary_sha256":manifest.binary_sha256,
            "source_commit":manifest.source_commit,
            "verified_input_asset_inventory_sha256":manifest.asset_inventory_sha256,
            "incoming_asset_inventory_verified_before_execution":true,
            "fresh_application_state":true,
            "empty_application_cache":true,
            "provider_credentials_absent":true,
            "ipv4_routes_absent":true,
        });
        let check = |report: &serde_json::Value, inventory: &str| {
            rehearsal_identity_matches(
                report,
                &manifest.binary_sha256,
                &manifest.source_commit,
                inventory,
            )
        };
        assert!(check(&report, &manifest.asset_inventory_sha256));
        fs::write(
            root.path().join("config/training.toml"),
            b"different training configuration",
        )
        .unwrap();
        let changed =
            asset_inventory_sha256(&capture_asset_inventory(root.path()).unwrap()).unwrap();
        assert!(!check(&report, &changed));
        assert_eq!(
            artefact::file_hash(&root.path().join("offline_bundle/bin/lec-distil")).unwrap(),
            manifest.binary_sha256
        );
        report["full_assessment_complete"] = serde_json::json!(false);
        assert!(!check(&report, &manifest.asset_inventory_sha256));
        report["full_assessment_complete"] = serde_json::json!(true);
        report
            .as_object_mut()
            .unwrap()
            .remove("verified_input_asset_inventory_sha256");
        assert!(
            !check(&report, &manifest.asset_inventory_sha256),
            "older unbound rehearsal must not certify a new bundle"
        );
    }
}
