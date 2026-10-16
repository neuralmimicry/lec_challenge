//! Durable ownership survives orchestrator SIGKILL. A pending journal permits
//! cleanup only after the original process has died and the runtime confirms
//! nonce, scope, name and immutable image. Cleanup targets an immutable ID.
use super::SandboxConfig;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    os::{
        fd::AsRawFd,
        unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::{process::Command, time::timeout};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    schema_version: u32,
    controller_version: u32,
    nonce: String,
    scope: String,
    workspace: PathBuf,
    uid: u32,
    runtime: String,
    name: String,
    image_id: String,
    owner_pid: u32,
    owner_start_ticks: u64,
    owner_boot_id: String,
    planned_unix_ms: i64,
    completed_unix_ms: Option<i64>,
    completion_reason: Option<String>,
    verified_container_id: Option<String>,
}
impl Record {
    fn labels(&self) -> Vec<(String, String)> {
        vec![
            ("org.lec.sandbox.nonce".into(), self.nonce.clone()),
            ("org.lec.sandbox.scope".into(), self.scope.clone()),
            ("org.lec.sandbox.uid".into(), self.uid.to_string()),
            (
                "org.lec.sandbox.controller".into(),
                self.controller_version.to_string(),
            ),
        ]
    }
}

pub(super) struct OwnedRun {
    path: PathBuf,
    record: Record,
}
impl OwnedRun {
    pub fn labels(&self) -> Vec<(String, String)> {
        self.record.labels()
    }
    pub fn verify_inspection(&self, inspected: &Value) -> Result<String> {
        verified_id(&self.record, inspected)
    }
    pub fn nonce(&self) -> &str {
        &self.record.nonce
    }
    pub fn scope(&self) -> &str {
        &self.record.scope
    }
    pub async fn bind_container(&mut self, id: &str) -> Result<()> {
        let id = image_id(id)?.to_owned();
        ensure!(
            self.record
                .verified_container_id
                .as_ref()
                .is_none_or(|old| old == &id),
            "owned immutable container ID changed"
        );
        self.record.verified_container_id = Some(id);
        let path = self.path.clone();
        let record = self.record.clone();
        tokio::task::spawn_blocking(move || publish(&path, &record)).await??;
        Ok(())
    }
    pub async fn finish_verified_absence(&mut self) -> Result<()> {
        ensure!(
            self.record.controller_version == 3 && self.record.verified_container_id.is_some(),
            "verified version-three absence lacks bound container identity"
        );
        self.record.completed_unix_ms = Some(crate::ledger::unix_millis());
        self.record.completion_reason = Some("controller_three_verified_exact_id_removal".into());
        let path = self.path.clone();
        let record = self.record.clone();
        tokio::task::spawn_blocking(move || publish(&path, &record)).await??;
        Ok(())
    }

    /// The controller retains its runtime lease until verified cleanup and this
    /// durable acknowledgement finish, even if the requesting future is dropped.
    pub async fn finish(&mut self) -> Result<()> {
        if self.record.completed_unix_ms.is_some() {
            return Ok(());
        }
        let config = SandboxConfig {
            runtime: self.record.runtime.clone(),
            ..Default::default()
        };
        let mut finished = self.record.clone();
        clean(&config, &mut finished, "controller_finished").await?;
        let path = self.path.clone();
        let record = finished.clone();
        tokio::task::spawn_blocking(move || publish(&path, &record)).await??;
        self.record = finished;
        Ok(())
    }
}

fn uid() -> u32 {
    // SAFETY: geteuid accepts no pointers and does not mutate process state.
    unsafe { libc::geteuid() }
}
fn boot_id() -> Result<String> {
    Ok(fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_owned())
}
fn process_start(text: &str) -> Result<(char, u64)> {
    let (_, tail) = text
        .rsplit_once(')')
        .context("Linux process identity is malformed")?;
    let fields = tail.split_whitespace().collect::<Vec<_>>();
    let state = fields
        .first()
        .and_then(|s| s.chars().next())
        .context("Linux process state absent")?;
    let ticks = fields
        .get(19)
        .context("Linux process start time absent")?
        .parse()?;
    Ok((state, ticks))
}
fn owner_dead(record: &Record) -> Result<bool> {
    if record.owner_boot_id != boot_id()? {
        return Ok(true);
    }
    match fs::read_to_string(format!("/proc/{}/stat", record.owner_pid)) {
        Ok(text) => {
            let (state, ticks) = process_start(&text)?;
            Ok(ticks != record.owner_start_ticks || matches!(state, 'Z' | 'X'))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error).context("cannot prove original sandbox owner has died"),
    }
}

fn private_directory(path: &Path) -> Result<()> {
    if !path.try_exists()? {
        fs::create_dir_all(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        File::open(path)?.sync_all()?;
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_dir() && metadata.uid() == uid() && metadata.mode() & 0o077 == 0,
        "sandbox ownership journal must be an owned private real directory"
    );
    Ok(())
}
fn runtime_executable(runtime: &str) -> Result<PathBuf> {
    let configured = Path::new(runtime);
    if configured.components().count() > 1 {
        return Ok(configured.canonicalize()?);
    }
    for directory in
        std::env::split_paths(&std::env::var_os("PATH").context("sandbox runtime PATH absent")?)
    {
        let candidate = directory.join(configured);
        if fs::metadata(&candidate).is_ok_and(|m| m.is_file() && m.mode() & 0o111 != 0) {
            return Ok(candidate.canonicalize()?);
        }
    }
    anyhow::bail!("sandbox runtime executable is unavailable")
}
fn root(cfg: &SandboxConfig) -> Result<(PathBuf, String, PathBuf, String)> {
    let workspace = std::env::current_dir()?.canonicalize()?;
    let runtime = runtime_executable(&cfg.runtime)?
        .to_str()
        .context("sandbox runtime path is not UTF-8")?
        .to_owned();
    let scope = crate::artefact::hash(&serde_json::to_vec(&(&workspace, uid(), &runtime))?);
    let directory = Path::new("state/sandbox-owned-runs").join(&scope);
    private_directory(Path::new("state/sandbox-owned-runs"))?;
    private_directory(&directory)?;
    Ok((directory, scope, workspace, runtime))
}
fn publish(path: &Path, record: &Record) -> Result<()> {
    let parent = path.parent().context("journal directory absent")?;
    let temporary = parent.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(&serde_json::to_vec_pretty(record)?)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
fn image_id(value: &str) -> Result<&str> {
    let id = value.trim().strip_prefix("sha256:").unwrap_or(value.trim());
    ensure!(
        id.len() == 64 && id.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "immutable image/container ID absent"
    );
    Ok(id)
}

/// Publish before spawning a runtime client. A crash between publication and
/// creation leaves a harmless pending record, recovered as an absent container.
pub(super) fn plan(cfg: &SandboxConfig, name: &str) -> Result<OwnedRun> {
    plan_with_nonce(cfg, name, None)
}
pub(super) fn plan_with_nonce(
    cfg: &SandboxConfig,
    name: &str,
    nonce: Option<&str>,
) -> Result<OwnedRun> {
    ensure!(
        matches!(cfg.controller_version, 2 | 3),
        "durable sandbox execution requires controller version two"
    );
    ensure!(
        name.starts_with("lec-")
            && name.len() <= 100
            && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'),
        "invalid owned sandbox name"
    );
    let (directory, scope, workspace, runtime) = root(cfg)?;
    let path = directory.join(format!("{name}.json"));
    ensure!(
        !path.try_exists()?,
        "sandbox run name already has durable ownership evidence"
    );
    let (_, owner_start_ticks) = process_start(&fs::read_to_string("/proc/self/stat")?)?;
    let record = Record {
        schema_version: 1,
        controller_version: cfg.controller_version,
        nonce: nonce
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        scope,
        workspace,
        uid: uid(),
        runtime,
        name: name.into(),
        image_id: image_id(&cfg.expected_image_id)?.into(),
        owner_pid: std::process::id(),
        owner_start_ticks,
        owner_boot_id: boot_id()?,
        planned_unix_ms: crate::ledger::unix_millis(),
        completed_unix_ms: None,
        completion_reason: None,
        verified_container_id: None,
    };
    publish(&path, &record)?;
    Ok(OwnedRun { path, record })
}

async fn inspect(cfg: &SandboxConfig, target: &str) -> Result<Option<Value>> {
    let output = timeout(
        Duration::from_secs(if cfg.controller_version == 3 { 30 } else { 10 }),
        Command::new(&cfg.runtime)
            .args(["inspect", target])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .context("owned sandbox inspection deadline exceeded")??;
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr).to_ascii_lowercase();
        ensure!(
            error.contains("no such container")
                || error.contains("no such object")
                || error.contains("does not exist")
                || error.contains("no container with"),
            "runtime inspection failed without proving container absence: {}",
            error.chars().take(512).collect::<String>()
        );
        return Ok(None);
    }
    let mut entries: Vec<Value> = serde_json::from_slice(&output.stdout)?;
    ensure!(
        entries.len() == 1,
        "owned sandbox inspection returned an ambiguous container"
    );
    Ok(entries.pop())
}
fn verified_id(record: &Record, inspected: &Value) -> Result<String> {
    let name = inspected["Name"]
        .as_str()
        .context("inspected sandbox name absent")?
        .trim_start_matches('/');
    ensure!(
        name == record.name,
        "sandbox ownership name changed; refusing cleanup"
    );
    ensure!(
        image_id(
            inspected["Image"]
                .as_str()
                .context("inspected image ID absent")?
        )? == record.image_id,
        "sandbox ownership image changed; refusing cleanup"
    );
    for (key, expected) in record.labels() {
        ensure!(
            inspected["Config"]["Labels"][&key].as_str() == Some(expected.as_str()),
            "sandbox ownership label {key} differs; refusing unrelated container cleanup"
        );
    }
    Ok(image_id(
        inspected["Id"]
            .as_str()
            .context("inspected container ID absent")?,
    )?
    .into())
}
async fn clean(cfg: &SandboxConfig, record: &mut Record, reason: &str) -> Result<()> {
    if let Some(inspected) = inspect(cfg, &record.name).await? {
        let id = verified_id(record, &inspected)?;
        super::stop_container(cfg, &id).await;
        ensure!(
            inspect(cfg, &id).await?.is_none(),
            "verified owned sandbox remains after cleanup"
        );
        record.verified_container_id = Some(id);
    }
    record.completed_unix_ms = Some(crate::ledger::unix_millis());
    record.completion_reason = Some(reason.into());
    Ok(())
}

/// Each stale journal has its own kernel-released claim. Normal grades may
/// retain shared runtime leases without racing recovery of the same dead run.
pub(super) async fn recover(cfg: &SandboxConfig) -> Result<()> {
    let owned_cfg = cfg.clone();
    let (directory, scope, workspace, runtime) =
        tokio::task::spawn_blocking(move || root(&owned_cfg)).await??;
    let paths = tokio::task::spawn_blocking(move || -> Result<Vec<PathBuf>> {
        fs::read_dir(directory)?
            .filter_map(|entry| match entry {
                Ok(entry) if entry.path().extension().is_some_and(|s| s == "json") => {
                    Some(Ok(entry.path()))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error.into())),
            })
            .collect()
    })
    .await??;
    for path in paths {
        let load_path = path.clone();
        let expected_scope = scope.clone();
        let expected_workspace = workspace.clone();
        let expected_runtime = runtime.clone();
        let loaded = tokio::task::spawn_blocking(move || -> Result<Option<(File, Record)>> {
            let claim = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(load_path.with_extension("claim"))?;
            // SAFETY: the live descriptor owns this non-blocking advisory claim.
            if unsafe { libc::flock(claim.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() == Some(libc::EAGAIN) {
                    return Ok(None);
                }
                return Err(error.into());
            }
            let metadata = fs::symlink_metadata(&load_path)?;
            ensure!(
                metadata.is_file()
                    && metadata.uid() == uid()
                    && metadata.nlink() == 1
                    && metadata.mode() & 0o077 == 0,
                "sandbox ownership journal is not an owned private regular file"
            );
            let record: Record = serde_json::from_slice(&fs::read(&load_path)?)?;
            ensure!(
                record.schema_version == 1
                    && matches!(record.controller_version, 2 | 3)
                    && record.scope == expected_scope
                    && record.workspace == expected_workspace
                    && record.uid == uid()
                    && record.runtime == expected_runtime,
                "sandbox ownership journal scope changed"
            );
            ensure!(
                load_path.file_stem().and_then(|s| s.to_str()) == Some(record.name.as_str()),
                "sandbox ownership journal name changed"
            );
            if record.completed_unix_ms.is_some() || !owner_dead(&record)? {
                return Ok(None);
            }
            Ok(Some((claim, record)))
        })
        .await??;
        let Some((_claim, mut record)) = loaded else {
            continue;
        };
        clean(cfg, &mut record, "recovered_after_original_owner_died").await?;
        eprintln!(
            "Sandbox recovered owned run {} after verified original-owner death",
            record.name
        );
        tokio::task::spawn_blocking(move || publish(&path, &record)).await??;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Record {
        let (_, ticks) = process_start(&fs::read_to_string("/proc/self/stat").unwrap()).unwrap();
        Record {
            schema_version: 1,
            controller_version: 2,
            nonce: "nonce-fixture".into(),
            scope: "scope-fixture".into(),
            workspace: PathBuf::from("/labelled-fixture"),
            uid: uid(),
            runtime: "/bin/true".into(),
            name: "lec-fixture".into(),
            image_id: "a".repeat(64),
            owner_pid: std::process::id(),
            owner_start_ticks: ticks,
            owner_boot_id: boot_id().unwrap(),
            planned_unix_ms: 1,
            completed_unix_ms: None,
            completion_reason: None,
            verified_container_id: None,
        }
    }
    #[test]
    fn pid_reuse_boot_changes_and_zombies_cannot_impersonate_the_original_owner() {
        let mut record = fixture();
        assert!(!owner_dead(&record).unwrap());
        record.owner_start_ticks += 1;
        assert!(owner_dead(&record).unwrap());
        record.owner_start_ticks -= 1;
        record.owner_boot_id = "another-boot".into();
        assert!(owner_dead(&record).unwrap());
        let text = format!(
            "123 (worker with ) spaces) Z {}",
            (1..=25)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        assert_eq!(process_start(&text).unwrap(), ('Z', 19));
    }
    #[test]
    fn cleanup_requires_every_label_name_image_and_an_immutable_container_id() {
        let record = fixture();
        let labels = record
            .labels()
            .into_iter()
            .map(|(key, value)| (key, Value::String(value)))
            .collect::<serde_json::Map<String, Value>>();
        let mut inspected = serde_json::json!({"Id":"b".repeat(64),"Name":"/lec-fixture","Image":format!("sha256:{}",record.image_id),"Config":{"Labels":labels}});
        assert_eq!(verified_id(&record, &inspected).unwrap(), "b".repeat(64));
        for key in [
            "org.lec.sandbox.nonce",
            "org.lec.sandbox.scope",
            "org.lec.sandbox.uid",
            "org.lec.sandbox.controller",
        ] {
            let mut altered = inspected.clone();
            altered["Config"]["Labels"][key] = "unrelated".into();
            assert!(verified_id(&record, &altered).is_err());
        }
        inspected["Name"] = "somebody-elses-container".into();
        assert!(verified_id(&record, &inspected).is_err());
        inspected["Name"] = "lec-fixture".into();
        inspected["Image"] = "c".repeat(64).into();
        assert!(verified_id(&record, &inspected).is_err());
        inspected["Image"] = record.image_id.clone().into();
        inspected["Id"] = "../../unrelated".into();
        assert!(verified_id(&record, &inspected).is_err());
    }
}
