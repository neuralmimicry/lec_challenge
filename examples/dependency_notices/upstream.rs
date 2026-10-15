//! Original upstream notice provenance, bound to the published crate's VCS record.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact::{self, Artefact};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Notice {
    pub url: String,
    pub source_path: String,
    pub upstream_blob_sha1: String,
    pub artifact: Artefact,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub version: String,
    pub repository: String,
    pub commit: String,
    pub path_in_vcs: String,
    pub crate_vcs_info: Artefact,
    pub crate_manifest: Artefact,
    pub upstream_tree: Artefact,
    pub upstream_tree_url: String,
    pub workspace_manifest: Notice,
    pub notices: Vec<Notice>,
}
#[derive(Serialize, Deserialize)]
struct Manifest {
    schema_version: u32,
    cargo_lock_sha256: String,
    acquired_unix_ms: i64,
    entries: Vec<Entry>,
}
pub struct Supplements {
    root: PathBuf,
    path: PathBuf,
    manifest: Manifest,
    client: reqwest::Client,
}
fn relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path)
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)))
}
fn slug(repository: &str) -> Result<&str> {
    let value = repository
        .strip_prefix("https://github.com/")
        .context("supplement requires an explicit GitHub repository")?
        .trim_end_matches('/')
        .trim_end_matches(".git");
    ensure!(
        value.split('/').count() == 2
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_/.".contains(&b)),
        "invalid upstream repository"
    );
    Ok(value)
}
fn notice_name(name: &str) -> bool {
    let name = name.to_ascii_uppercase();
    [
        "LICENSE",
        "LICENCE",
        "COPYING",
        "NOTICE",
        "COPYRIGHT",
        "AUTHORS",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}
fn original_licence_grant(path: &str, bytes: &[u8]) -> bool {
    let upper = path.to_ascii_uppercase();
    let name = Path::new(&upper)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    let conventional = ["LICENSE", "LICENCE", "COPYING"]
        .iter()
        .any(|p| name.starts_with(p))
        || upper.starts_with("LICENSES/")
        || upper.starts_with("LICENCES/");
    // r-efi retains its complete original MIT grant and copyright in AUTHORS,
    // rather than a conventionally named LICENSE file. Preserve the actual file.
    conventional
        || (name == "AUTHORS"
            && std::str::from_utf8(bytes).is_ok_and(|text| {
                text.contains("AUTHORS-MIT:")
                    && text.contains("Permission is hereby granted, free of charge")
                    && text.contains("COPYRIGHT:")
            }))
}
fn vcs(source: &Path) -> Result<(Value, Vec<u8>)> {
    let bytes = fs::read(source.join(".cargo_vcs_info.json")).context("crate VCS proof missing")?;
    Ok((serde_json::from_slice(&bytes)?, bytes))
}
fn store(root: &Path, path: &Path, bytes: &[u8]) -> Result<Artefact> {
    artefact::atomic_write(path, bytes)?;
    Artefact::capture(root, path.strip_prefix(root)?)
}
async fn get(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let response = client.get(url).send().await?.error_for_status()?;
    let bytes = response.bytes().await?;
    ensure!(
        !bytes.is_empty() && bytes.len() <= 5 * 1024 * 1024,
        "original evidence empty or oversized"
    );
    Ok(bytes.to_vec())
}

impl Supplements {
    pub fn load(root: &Path, path: &Path, lock_hash: &str) -> Result<Self> {
        let manifest = if path.exists() {
            let saved: Manifest = serde_json::from_slice(&fs::read(path)?)?;
            ensure!(
                saved.schema_version == 1 && saved.cargo_lock_sha256 == lock_hash,
                "supplement Cargo.lock identity changed"
            );
            saved
        } else {
            Manifest {
                schema_version: 1,
                cargo_lock_sha256: lock_hash.into(),
                acquired_unix_ms: lec_distil::ledger::unix_millis(),
                entries: vec![],
            }
        };
        let client = reqwest::Client::builder()
            .user_agent("lec-distil-original-notices/0.1")
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(90))
            .build()?;
        Ok(Self {
            root: root.into(),
            path: path.into(),
            manifest,
            client,
        })
    }
    pub async fn resolve(
        &mut self,
        package: &Value,
        source: &Path,
        fetch: bool,
    ) -> Result<Option<Entry>> {
        let name = package["name"].as_str().context("package name")?;
        let version = package["version"].as_str().context("package version")?;
        if !self
            .manifest
            .entries
            .iter()
            .any(|e| e.name == name && e.version == version)
            && fetch
        {
            eprintln!("notices: retrieving original pinned workspace notices for {name}-{version}");
            let entry = self.fetch(package, source).await?;
            self.manifest.entries.push(entry);
            artefact::write_json(&self.path, &self.manifest)?;
        }
        let entry = self
            .manifest
            .entries
            .iter()
            .find(|e| e.name == name && e.version == version)
            .cloned();
        if let Some(entry) = &entry {
            self.verify(entry, package, source)?;
        }
        Ok(entry)
    }
    async fn fetch(&self, package: &Value, source: &Path) -> Result<Entry> {
        let name = package["name"].as_str().context("name")?;
        let version = package["version"].as_str().context("version")?;
        let repository = package["repository"].as_str().context("repository")?;
        let repository_slug = slug(repository)?;
        let (proof, proof_bytes) = vcs(source)?;
        let commit = proof["git"]["sha1"]
            .as_str()
            .context("crate commit missing")?;
        let path_in_vcs = proof["path_in_vcs"]
            .as_str()
            .context("crate path missing")?;
        ensure!(
            commit.len() == 40
                && commit.bytes().all(|b| b.is_ascii_hexdigit())
                && (path_in_vcs.is_empty() || relative(path_in_vcs)),
            "invalid crate VCS record"
        );
        let destination = self
            .root
            .join("offline_bundle/notices")
            .join(format!("{name}-{version}"));
        let crate_vcs_info = store(
            &self.root,
            &destination.join("SOURCE-CARGO-VCS-INFO.json"),
            &proof_bytes,
        )?;
        let crate_manifest = store(
            &self.root,
            &destination.join("SOURCE-CARGO-TOML.orig"),
            &fs::read(source.join("Cargo.toml.orig"))?,
        )?;
        let upstream_tree_url = format!(
            "https://api.github.com/repos/{repository_slug}/git/trees/{commit}?recursive=1"
        );
        let tree_bytes = get(&self.client, &upstream_tree_url).await?;
        let tree: Value = serde_json::from_slice(&tree_bytes)?;
        ensure!(
            tree["sha"] == commit && tree["truncated"] == false,
            "upstream tree is not the complete pinned commit"
        );
        let upstream_tree = store(
            &self.root,
            &destination.join("SOURCE-UPSTREAM-TREE.json"),
            &tree_bytes,
        )?;
        let blobs = tree["tree"].as_array().context("upstream blobs missing")?;
        let cargo = blobs
            .iter()
            .find(|blob| blob["path"] == "Cargo.toml" && blob["type"] == "blob")
            .context("original workspace manifest missing")?;
        let url =
            format!("https://raw.githubusercontent.com/{repository_slug}/{commit}/Cargo.toml");
        let workspace_manifest = Notice {
            url: url.clone(),
            source_path: "Cargo.toml".into(),
            upstream_blob_sha1: cargo["sha"].as_str().context("manifest blob hash")?.into(),
            artifact: store(
                &self.root,
                &destination.join("SOURCE-WORKSPACE-CARGO.toml"),
                &get(&self.client, &url).await?,
            )?,
        };
        let mut notices = Vec::new();
        let mut licence_files = 0;
        for blob in blobs {
            if blob["type"] != "blob" {
                continue;
            }
            let path = blob["path"].as_str().context("upstream path")?;
            ensure!(relative(path), "unsafe upstream path");
            let file = Path::new(path)
                .file_name()
                .context("upstream filename")?
                .to_string_lossy();
            let parent = Path::new(path).parent().unwrap_or(Path::new(""));
            let ancestor = parent.as_os_str().is_empty()
                || (!path_in_vcs.is_empty() && Path::new(path_in_vcs).starts_with(parent));
            let licence_directory=parent.components().next().is_some_and(|c|matches!(c,std::path::Component::Normal(n) if ["LICENSES","LICENCES"].contains(&n.to_string_lossy().to_ascii_uppercase().as_str())));
            if !(ancestor && notice_name(&file) || licence_directory) {
                continue;
            }
            let url =
                format!("https://raw.githubusercontent.com/{repository_slug}/{commit}/{path}");
            let bytes = get(&self.client, &url).await?;
            if original_licence_grant(path, &bytes) {
                licence_files += 1;
            }
            notices.push(Notice {
                url,
                source_path: path.into(),
                upstream_blob_sha1: blob["sha"].as_str().context("upstream blob hash")?.into(),
                artifact: store(&self.root, &destination.join("upstream").join(path), &bytes)?,
            });
        }
        ensure!(
            licence_files > 0,
            "only attribution metadata found; original licence grants still missing"
        );
        notices.sort_by(|a, b| a.source_path.cmp(&b.source_path));
        Ok(Entry {
            name: name.into(),
            version: version.into(),
            repository: repository.into(),
            commit: commit.into(),
            path_in_vcs: path_in_vcs.into(),
            crate_vcs_info,
            crate_manifest,
            upstream_tree,
            upstream_tree_url,
            workspace_manifest,
            notices,
        })
    }
    fn verify(&self, entry: &Entry, package: &Value, source: &Path) -> Result<()> {
        ensure!(
            entry.name == package["name"]
                && entry.version == package["version"]
                && entry.repository == package["repository"],
            "supplement package identity mismatch"
        );
        let repository_slug = slug(&entry.repository)?;
        let (proof, bytes) = vcs(source)?;
        ensure!(
            proof["git"]["sha1"] == entry.commit
                && proof["path_in_vcs"] == entry.path_in_vcs
                && artefact::hash(&bytes) == entry.crate_vcs_info.sha256
                && artefact::file_hash(&source.join("Cargo.toml.orig"))?
                    == entry.crate_manifest.sha256,
            "supplement differs from original packaged source provenance"
        );
        entry.crate_vcs_info.verify(&self.root)?;
        entry.crate_manifest.verify(&self.root)?;
        ensure!(
            entry.upstream_tree_url
                == format!(
                    "https://api.github.com/repos/{repository_slug}/git/trees/{}?recursive=1",
                    entry.commit
                ),
            "upstream tree URL is not pinned"
        );
        let tree: Value =
            serde_json::from_slice(&fs::read(entry.upstream_tree.verify(&self.root)?)?)?;
        ensure!(
            tree["sha"] == entry.commit && tree["truncated"] == false,
            "upstream tree identity mismatch"
        );
        let blobs = tree["tree"]
            .as_array()
            .context("upstream tree blobs missing")?;
        let mut licence_files = 0;
        for notice in std::iter::once(&entry.workspace_manifest).chain(&entry.notices) {
            ensure!(
                relative(&notice.source_path)
                    && notice.url
                        == format!(
                            "https://raw.githubusercontent.com/{repository_slug}/{}/{}",
                            entry.commit, notice.source_path
                        ),
                "notice source URL is not exact-commit original upstream"
            );
            ensure!(
                blobs.iter().any(|b| b["path"] == notice.source_path
                    && b["sha"] == notice.upstream_blob_sha1
                    && b["type"] == "blob"),
                "notice blob absent from original upstream tree"
            );
            let verified = notice.artifact.verify(&self.root)?;
            if original_licence_grant(&notice.source_path, &fs::read(verified)?) {
                licence_files += 1;
            }
        }
        ensure!(
            !entry.notices.is_empty() && licence_files > 0,
            "supplement has no original licence grant"
        );
        Ok(())
    }
}
