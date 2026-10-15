//! Authenticated, commit-pinned model assets, without executable repository code.
//!
//! Authentication belongs only to the original HTTPS Hub request. Redirects
//! permanently lose the bearer header, including redirects back to the Hub.
//! Downloaded bytes must match the publisher's Git/LFS identity before an atomic
//! asset manifest can make them available to native model loading.

use anyhow::{Context, Result, bail, ensure};
use futures_util::{StreamExt, stream};
use reqwest::{Client, Response, StatusCode, Url, header};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::io::AsyncWriteExt;
use zeroize::Zeroizing;

use crate::model::AssetManifest;

mod gguf;
mod historical;
mod ranges;
pub use gguf::{
    HubGgufDownloadOptions, HubGgufProvenance, HubGgufSource, HubGgufSourceAssets, download_gguf,
    is_intentionally_pruned_gguf, validate_gguf_assets,
};
pub(crate) use gguf::{PrunedWeight, PrunedWeightsProof};
pub use historical::{validate_downloaded_asset_metadata, validate_gguf_asset_metadata};

const MAX_METADATA_BYTES: usize = 32 * 1024 * 1024;
const MAX_TOKEN_BYTES: usize = 8 * 1024;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const PROVENANCE_FILE: &str = "hub-provenance.json";
const PLAN_FILE: &str = "hub-plan.json";
const AUTHENTICATION_POLICY: &str = "Optional bearer applied to initial HTTPS huggingface.co request only; all redirects strip authentication; no token persisted";

/// Sanitised classification for switching a search into its cached-only mode.
#[derive(Debug, thiserror::Error)]
pub enum HubRequestFailure {
    #[error("Hub connectivity failed during {stage} (URL and credentials redacted)")]
    Connectivity { stage: &'static str },
    #[error("Hub HTTP {status} (URL and credentials redacted)")]
    Http { status: u16 },
}

/// Permanent model/authentication errors such as 404/401 are never offline signals.
pub fn is_connectivity_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| match cause.downcast_ref::<HubRequestFailure>() {
            Some(HubRequestFailure::Connectivity { .. }) => true,
            Some(HubRequestFailure::Http { status }) => {
                matches!(*status, 408 | 425 | 429 | 500..=599)
            }
            None => false,
        })
}

#[derive(Clone, Debug)]
pub struct HubDownloadOptions {
    pub model_id: String,
    /// A ref is resolved before downloading; only the resulting commit is used.
    pub revision: String,
    pub destination: PathBuf,
    /// Concurrent file transfers, bounded between one and eight.
    pub concurrency: usize,
    /// Read one bounded token line from stdin instead of HF_TOKEN.
    pub token_stdin: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "digest", rename_all = "snake_case")]
pub enum PublisherIdentity {
    LfsSha256(String),
    /// SHA-1 covers the Git blob header and file, not merely the file bytes.
    GitBlobSha1(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HubAsset {
    pub filename: String,
    pub bytes: u64,
    pub publisher_identity: PublisherIdentity,
    /// Canonical pinned URL; never a signed CDN URL or credential-bearing URL.
    pub canonical_url: String,
    /// Computed from verified complete bytes, regardless of publisher hash type.
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubProvenance {
    pub schema_version: u32,
    pub model_id: String,
    pub requested_revision: String,
    pub revision: String,
    pub licence: String,
    pub licence_files: Vec<String>,
    pub licence_evidence: String,
    pub recorded_unix_seconds: u64,
    pub metadata_url: String,
    pub metadata_sha256: String,
    /// Original commit-pinned metadata preserves model-card/licence information.
    pub metadata: serde_json::Value,
    pub files: BTreeMap<String, HubAsset>,
    pub authentication_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ApiModel {
    id: String,
    sha: String,
    siblings: Vec<ApiFile>,
    #[serde(default, rename = "cardData")]
    card_data: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ApiFile {
    rfilename: String,
    #[serde(rename = "blobId")]
    blob_id: String,
    size: u64,
    #[serde(default)]
    lfs: Option<ApiLfs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ApiLfs {
    sha256: String,
    size: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct DownloadPlan {
    model_id: String,
    revision: String,
    files: BTreeMap<String, PlannedFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct PlannedFile {
    filename: String,
    bytes: u64,
    publisher_identity: PublisherIdentity,
}

/// Intentionally has no Debug/Serialize implementation and no secret getter.
struct Bearer(header::HeaderValue);

impl Bearer {
    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_TOKEN_BYTES,
            "HF token is empty or exceeds the supported length"
        );
        ensure!(
            bytes.iter().all(|b| b.is_ascii_graphic()),
            "HF token contains whitespace or invalid characters"
        );
        let mut value = Zeroizing::new(Vec::with_capacity(bytes.len() + 7));
        value.extend_from_slice(b"Bearer ");
        value.extend_from_slice(bytes);
        let mut header = header::HeaderValue::from_bytes(&value)
            .map_err(|_| anyhow::anyhow!("HF token cannot form an HTTP bearer header"))?;
        header.set_sensitive(true);
        Ok(Self(header))
    }
}

fn load_bearer(token_stdin: bool) -> Result<Option<Bearer>> {
    if token_stdin {
        #[cfg(unix)]
        let _echo = {
            use std::os::fd::AsRawFd;
            TerminalEchoGuard::new(std::io::stdin().as_raw_fd())?
        };
        let mut bytes = Zeroizing::new(Vec::new());
        // Read only the token line, not unrelated later input or terminal data.
        for byte in std::io::stdin().lock().bytes().take(MAX_TOKEN_BYTES + 2) {
            let byte = byte.context("read HF token from stdin")?;
            if byte == b'\n' {
                break;
            }
            bytes.push(byte);
        }
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        return Bearer::from_bytes(&bytes).map(Some);
    }
    match std::env::var("HF_TOKEN") {
        Ok(token) => {
            let token = Zeroizing::new(token);
            Bearer::from_bytes(token.as_bytes()).map(Some)
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => bail!("HF_TOKEN is not valid Unicode"),
    }
}

/// Disable terminal echo only while reading the secret, including error paths.
#[cfg(unix)]
struct TerminalEchoGuard {
    descriptor: std::os::fd::RawFd,
    original: Option<libc::termios>,
}

#[cfg(unix)]
impl TerminalEchoGuard {
    fn new(descriptor: std::os::fd::RawFd) -> Result<Self> {
        // SAFETY: isatty inspects the borrowed descriptor without taking ownership.
        if unsafe { libc::isatty(descriptor) } == 0 {
            return Ok(Self {
                descriptor,
                original: None,
            });
        }
        let mut settings = std::mem::MaybeUninit::<libc::termios>::uninit();
        // SAFETY: tcgetattr initialises the supplied termios object on success.
        ensure!(
            unsafe { libc::tcgetattr(descriptor, settings.as_mut_ptr()) } == 0,
            "cannot inspect terminal before secret input"
        );
        // SAFETY: the preceding successful tcgetattr initialised every field.
        let original = unsafe { settings.assume_init() };
        let mut silent = original;
        silent.c_lflag &= !(libc::ECHO | libc::ECHONL);
        // SAFETY: tcsetattr borrows a fully initialised termios object.
        ensure!(
            unsafe { libc::tcsetattr(descriptor, libc::TCSANOW, &silent) } == 0,
            "cannot disable terminal echo for secret input"
        );
        Ok(Self {
            descriptor,
            original: Some(original),
        })
    }
}

#[cfg(unix)]
impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        if let Some(original) = &self.original {
            // SAFETY: the original terminal settings and borrowed fd are valid.
            let _ = unsafe { libc::tcsetattr(self.descriptor, libc::TCSANOW, original) };
        }
    }
}

fn client() -> Result<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(3600))
        .user_agent("lec-distil/0.1 pinned-native-assets")
        .build()
        .context("construct Hub HTTP client")
}

fn valid_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn validate_model_id(value: &str) -> Result<()> {
    let pieces: Vec<_> = value.split('/').collect();
    ensure!(pieces.len() == 2, "model ID must be namespace/name");
    ensure!(
        pieces.iter().all(|p| !p.is_empty()
            && *p != "."
            && *p != ".."
            && p.len() <= 128
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))),
        "unsafe model ID"
    );
    Ok(())
}

fn validate_ref(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty()
            && value.len() <= 128
            && value != "."
            && value != ".."
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
        "unsupported or unsafe model revision"
    );
    Ok(())
}

fn licence_filename(name: &str) -> bool {
    matches!(
        name,
        "LICENSE"
            | "LICENSE.txt"
            | "LICENSE.md"
            | "LICENCE"
            | "LICENCE.txt"
            | "LICENCE.md"
            | "COPYING"
            | "COPYING.txt"
    )
}

/// An exact data/document allowlist excludes Python, pickle and repository code.
fn allowed_filename(name: &str) -> bool {
    if licence_filename(name)
        || matches!(
            name,
            "README.md"
                | "NOTICE"
                | "NOTICE.txt"
                | "config.json"
                | "generation_config.json"
                | "tokenizer.json"
                | "tokenizer_config.json"
                | "special_tokens_map.json"
                | "added_tokens.json"
                | "tokenizer.model"
                | "vocab.json"
                | "merges.txt"
                | "model.safetensors"
                | "model.safetensors.index.json"
        )
    {
        return true;
    }
    let Some(shard) = name
        .strip_prefix("model-")
        .and_then(|s| s.strip_suffix(".safetensors"))
    else {
        return false;
    };
    let Some((number, total)) = shard.split_once("-of-") else {
        return false;
    };
    number.len() == 5
        && total.len() == 5
        && number.bytes().all(|b| b.is_ascii_digit())
        && total.bytes().all(|b| b.is_ascii_digit())
        && number != "00000"
        && total != "00000"
        && number <= total
}

fn plan_from_metadata(model_id: &str, metadata: &ApiModel) -> Result<DownloadPlan> {
    validate_model_id(model_id)?;
    ensure!(
        metadata.id == model_id,
        "Hub model identity differs from the requested repository"
    );
    ensure!(
        valid_hex(&metadata.sha, 40),
        "Hub returned a non-commit revision"
    );
    let mut files = BTreeMap::new();
    let mut total = 0_u64;
    for file in &metadata.siblings {
        if !allowed_filename(&file.rfilename) {
            continue;
        }
        ensure!(
            file.size > 0 && file.size <= MAX_FILE_BYTES,
            "unsupported asset size: {}",
            file.rfilename
        );
        let identity = match &file.lfs {
            Some(lfs) => {
                ensure!(
                    valid_hex(&lfs.sha256, 64) && lfs.size == file.size,
                    "invalid LFS metadata: {}",
                    file.rfilename
                );
                PublisherIdentity::LfsSha256(lfs.sha256.clone())
            }
            None => {
                ensure!(
                    valid_hex(&file.blob_id, 40),
                    "invalid Git blob metadata: {}",
                    file.rfilename
                );
                PublisherIdentity::GitBlobSha1(file.blob_id.clone())
            }
        };
        total = total
            .checked_add(file.size)
            .context("asset size overflow")?;
        ensure!(
            total <= MAX_TOTAL_BYTES,
            "model exceeds the bounded asset size"
        );
        ensure!(
            files
                .insert(
                    file.rfilename.clone(),
                    PlannedFile {
                        filename: file.rfilename.clone(),
                        bytes: file.size,
                        publisher_identity: identity
                    }
                )
                .is_none(),
            "duplicate Hub asset metadata"
        );
    }
    ensure!(
        files.contains_key("config.json") && files.contains_key("tokenizer_config.json"),
        "model config or original chat template missing"
    );
    ensure!(
        files.contains_key("tokenizer.json") || files.contains_key("tokenizer.model"),
        "original tokeniser missing"
    );
    ensure!(
        files.keys().any(|p| p.ends_with(".safetensors")),
        "native safetensors weights missing"
    );
    ensure!(
        files.keys().any(|p| licence_filename(p))
            || (files.contains_key("README.md")
                && recognised_spdx_licence(&licence_from_metadata(metadata))),
        "licence document missing and no recognised SPDX model-card declaration"
    );
    Ok(DownloadPlan {
        model_id: model_id.to_owned(),
        revision: metadata.sha.clone(),
        files,
    })
}

fn licence_from_metadata(metadata: &ApiModel) -> String {
    metadata
        .card_data
        .get("license")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("unspecified; inspect bundled licence document")
        .to_owned()
}

fn recognised_spdx_licence(licence: &str) -> bool {
    matches!(
        licence,
        "apache-2.0" | "mit" | "bsd-2-clause" | "bsd-3-clause" | "cc0-1.0"
    )
}

fn licence_evidence(has_document: bool) -> String {
    if has_document {
        "Original pinned repository licence document(s) and model-card metadata retained".into()
    } else {
        "Original pinned model card and API metadata declare a recognised SPDX licence; repository has no separate licence file; retain canonical full text separately for distribution".into()
    }
}

fn canonical_file_url(model_id: &str, commit: &str, filename: &str) -> String {
    format!("https://huggingface.co/{model_id}/resolve/{commit}/{filename}")
}

/// Reqwest errors may contain signed URLs, so none are attached to public errors.
async fn request_with_redirects(
    client: &Client,
    url: Url,
    bearer: Option<&Bearer>,
    range_start: Option<u64>,
    test_http: bool,
) -> Result<Response> {
    request_with_range(
        client,
        url,
        bearer,
        range_start.map(|start| HttpRange {
            start,
            end: None,
            if_range: None,
        }),
        test_http,
    )
    .await
}

#[derive(Clone)]
struct HttpRange {
    start: u64,
    end: Option<u64>,
    if_range: Option<String>,
}

async fn request_with_range(
    client: &Client,
    mut url: Url,
    bearer: Option<&Bearer>,
    range: Option<HttpRange>,
    test_http: bool,
) -> Result<Response> {
    for redirect in 0..=8 {
        ensure!(
            url.username().is_empty() && url.password().is_none(),
            "credential-bearing asset URL rejected"
        );
        ensure!(
            url.scheme() == "https" || (test_http && url.scheme() == "http"),
            "insecure asset URL rejected"
        );
        let mut request = client
            .get(url.clone())
            .header(header::ACCEPT_ENCODING, "identity");
        if redirect == 0
            && let Some(bearer) = bearer
        {
            ensure!(
                test_http || url.host_str() == Some("huggingface.co"),
                "authentication is restricted to the Hub origin"
            );
            request = request.header(header::AUTHORIZATION, bearer.0.clone());
        }
        if let Some(range) = &range {
            request = request.header(
                header::RANGE,
                format!(
                    "bytes={}-{}",
                    range.start,
                    range.end.map(|end| end.to_string()).unwrap_or_default()
                ),
            );
            if let Some(etag) = &range.if_range {
                request = request.header(header::IF_RANGE, etag);
            }
        }
        let pending = request.send();
        tokio::pin!(pending);
        let mut heartbeat = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(15),
            Duration::from_secs(15),
        );
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let response = loop {
            tokio::select! {
                response = &mut pending => break response.map_err(|_| HubRequestFailure::Connectivity { stage:"response headers" })?,
                _ = heartbeat.tick() => eprintln!("Hub waiting for response headers; transfer progress resumes when the server responds"),
            }
        };
        if response.status().is_redirection() {
            ensure!(redirect < 8, "Hub redirect limit exceeded");
            let location = response
                .headers()
                .get(header::LOCATION)
                .context("Hub redirect lacks Location")?
                .to_str()
                .map_err(|_| anyhow::anyhow!("invalid Hub redirect Location"))?;
            url = url
                .join(location)
                .map_err(|_| anyhow::anyhow!("invalid Hub redirect URL"))?;
            continue;
        }
        return Ok(response);
    }
    bail!("Hub redirect limit exceeded")
}

async fn bounded_body(mut response: Response, maximum: usize) -> Result<Vec<u8>> {
    if let Some(size) = response.content_length() {
        ensure!(
            size <= maximum as u64,
            "Hub metadata exceeds the supported size"
        );
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| HubRequestFailure::Connectivity {
            stage: "metadata stream",
        })?
    {
        ensure!(
            chunk.len() <= maximum.saturating_sub(bytes.len()),
            "Hub metadata exceeds the supported size"
        );
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn metadata(
    client: &Client,
    model_id: &str,
    revision: &str,
    bearer: Option<&Bearer>,
) -> Result<(ApiModel, Vec<u8>)> {
    let url =
        format!("https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true");
    let mut bytes = None;
    for attempt in 0..3 {
        let result = async {
            let response =
                request_with_redirects(client, Url::parse(&url)?, bearer, None, false).await?;
            if !response.status().is_success() {
                return Err(HubRequestFailure::Http {
                    status: response.status().as_u16(),
                }
                .into());
            }
            bounded_body(response, MAX_METADATA_BYTES).await
        }
        .await;
        match result {
            Ok(body) => {
                bytes = Some(body);
                break;
            }
            Err(error) if attempt < 2 && is_connectivity_error(&error) => {
                tokio::time::sleep(Duration::from_secs(1 << attempt)).await
            }
            Err(error) => return Err(error).context("Hub metadata bounded acquisition failed"),
        }
    }
    let bytes = bytes.context("Hub metadata acquisition produced no body")?;
    let value: ApiModel = serde_json::from_slice(&bytes).context("invalid Hub model metadata")?;
    Ok((value, bytes))
}

fn ensure_safe_path(path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "empty model directory");
    let mut prefix = PathBuf::new();
    for component in path.components() {
        ensure!(
            !matches!(component, Component::ParentDir),
            "parent traversal in model path"
        );
        prefix.push(component.as_os_str());
        match fs::symlink_metadata(&prefix) {
            Ok(meta) => ensure!(
                !meta.file_type().is_symlink(),
                "symlink in model asset path"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("inspect model asset path"),
        }
    }
    Ok(())
}

/// Unique temporary names allow a crashed publication to restart immediately.
fn atomic_publish(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure_safe_path(path)?;
    let parent = path.parent().context("asset manifest lacks parent")?;
    let name = path
        .file_name()
        .context("asset manifest lacks filename")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags_no_follow()
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

struct DestinationLock(File);
impl DestinationLock {
    fn acquire(dir: &Path) -> Result<Self> {
        ensure_safe_path(dir)?;
        fs::create_dir_all(dir)?;
        ensure_safe_path(dir)?;
        let path = dir.join(".hub-download.lock");
        ensure_safe_path(&path)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .custom_flags_no_follow()
            .open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: flock borrows a valid live descriptor and does not own it.
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            ensure!(
                result == 0,
                "another model download holds this destination lock"
            );
        }
        #[cfg(not(unix))]
        bail!("safe restartable downloader requires the supported Unix file-lock backend");
        Ok(Self(file))
    }
}
impl Drop for DestinationLock {
    fn drop(&mut self) {
        // File close releases the advisory lock, including process crashes.
        let _ = self.0.sync_all();
    }
}

trait NoFollowOptions {
    fn custom_flags_no_follow(&mut self) -> &mut Self;
}
impl NoFollowOptions for OpenOptions {
    fn custom_flags_no_follow(&mut self) -> &mut Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            self.custom_flags(libc::O_NOFOLLOW);
        }
        self
    }
}

struct FileProgress<'a> {
    filename: &'a str,
    stage: &'a str,
    total: u64,
    initial: u64,
    started: Instant,
}

impl<'a> FileProgress<'a> {
    fn new(filename: &'a str, stage: &'a str, total: u64, initial: u64) -> Self {
        Self {
            filename,
            stage,
            total,
            initial,
            started: Instant::now(),
        }
    }

    fn report(&self, processed: u64) {
        let present = self.initial.saturating_add(processed).min(self.total);
        let speed = processed as f64 / self.started.elapsed().as_secs_f64().max(0.001);
        let eta = if speed > 0. {
            let seconds = ((self.total - present) as f64 / speed).ceil() as u64;
            format!(
                "{:02}:{:02}:{:02}",
                seconds / 3600,
                (seconds / 60) % 60,
                seconds % 60
            )
        } else {
            "unknown".into()
        };
        eprintln!(
            "Hub {} {}: {present}/{} bytes, {:.2} MiB/s, ETA {eta} (publisher checksum pending)",
            self.stage,
            self.filename,
            self.total,
            speed / (1024. * 1024.)
        );
    }
}

fn verify_file(path: &Path, planned: &PlannedFile) -> Result<String> {
    verify_file_tracked(path, planned, None)
}

fn verify_file_tracked(
    path: &Path,
    planned: &PlannedFile,
    progress: Option<Arc<std::sync::atomic::AtomicU64>>,
) -> Result<String> {
    ensure_safe_path(path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags_no_follow()
        .open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() == planned.bytes,
        "asset size mismatch: {}",
        planned.filename
    );
    let mut sha256 = Sha256::new();
    // Large LFS weights need only their publisher SHA-256; avoid an unnecessary
    // second multi-gigabyte digest. Ordinary Git files still verify both hashes.
    let mut git = if matches!(
        planned.publisher_identity,
        PublisherIdentity::GitBlobSha1(_)
    ) {
        let mut hash = Sha1::new();
        hash.update(format!("blob {}\0", planned.bytes).as_bytes());
        Some(hash)
    } else {
        None
    };
    let mut buffer = vec![0_u8; 1024 * 1024];
    let meter = FileProgress::new(&planned.filename, "hashing", planned.bytes, 0);
    let large = planned.bytes >= 64 * 1024 * 1024;
    let mut last_report = Instant::now();
    let mut processed = 0_u64;
    if large && progress.is_none() {
        meter.report(0);
    }
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        sha256.update(&buffer[..n]);
        if let Some(hash) = &mut git {
            hash.update(&buffer[..n]);
        }
        processed += n as u64;
        if let Some(progress) = &progress {
            progress.store(processed, std::sync::atomic::Ordering::Relaxed);
        }
        if large && progress.is_none() && last_report.elapsed() >= Duration::from_secs(15) {
            meter.report(processed);
            last_report = Instant::now();
        }
    }
    let sha256 = hex::encode(sha256.finalize());
    let identity_matches = match &planned.publisher_identity {
        PublisherIdentity::LfsSha256(expected) => expected == &sha256,
        PublisherIdentity::GitBlobSha1(expected) => {
            expected == &hex::encode(git.context("Git identity verifier missing")?.finalize())
        }
    };
    ensure!(
        identity_matches,
        "publisher checksum mismatch: {}",
        planned.filename
    );
    Ok(sha256)
}

async fn verify_file_async(path: PathBuf, planned: PlannedFile) -> Result<String> {
    let progress = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let worker_progress = progress.clone();
    let worker_plan = planned.clone();
    let mut worker = tokio::task::spawn_blocking(move || {
        verify_file_tracked(&path, &worker_plan, Some(worker_progress))
    });
    let meter = FileProgress::new(&planned.filename, "hashing", planned.bytes, 0);
    let large = planned.bytes >= 64 * 1024 * 1024;
    if large {
        meter.report(0);
    }
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            result = &mut worker => return result?,
            _ = heartbeat.tick(), if large => meter.report(progress.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}

fn validate_content_range(value: &str, start: u64, total: u64) -> Result<()> {
    let parts = value
        .strip_prefix("bytes ")
        .and_then(|p| p.split_once('/'))
        .context("invalid resumed asset Content-Range")?;
    let (first, last) = parts
        .0
        .split_once('-')
        .context("invalid resumed asset range")?;
    let first: u64 = first.parse()?;
    let last: u64 = last.parse()?;
    let returned_total: u64 = parts.1.parse()?;
    ensure!(
        first == start && returned_total == total && last == total - 1,
        "resumed asset Content-Range mismatch"
    );
    Ok(())
}

async fn download_file(
    client: &Client,
    bearer: Option<&Bearer>,
    destination: &Path,
    model_id: &str,
    commit: &str,
    planned: PlannedFile,
) -> Result<HubAsset> {
    let final_path = destination.join(&planned.filename);
    let partial = destination.join(format!(".{}.partial", planned.filename));
    let state_path = destination.join(format!(".{}.partial.json", planned.filename));
    for path in [&final_path, &partial, &state_path] {
        ensure_safe_path(path)?;
    }
    let canonical_url = canonical_file_url(model_id, commit, &planned.filename);
    if final_path.exists() {
        let sha256 = verify_file_async(final_path, planned.clone()).await?;
        ranges::cleanup(destination, &planned)?;
        return Ok(HubAsset {
            filename: planned.filename,
            bytes: planned.bytes,
            publisher_identity: planned.publisher_identity,
            canonical_url,
            sha256,
        });
    }
    let inspect_state = state_path.clone();
    let inspect_partial = partial.clone();
    let inspect_plan = planned.clone();
    tokio::task::spawn_blocking(move || {
        if inspect_state.exists() {
            let state: PlannedFile = serde_json::from_slice(&fs::read(&inspect_state)?)?;
            ensure!(
                state == inspect_plan,
                "partial asset metadata changed; use a new destination"
            );
        } else {
            ensure!(
                !inspect_partial.exists(),
                "untracked partial asset rejected"
            );
            atomic_publish(&inspect_state, &serde_json::to_vec_pretty(&inspect_plan)?)?;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    if planned.bytes >= 64 * 1024 * 1024
        && tokio::fs::metadata(&partial)
            .await
            .map(|m| m.len() < planned.bytes)
            .unwrap_or(true)
    {
        ranges::download(ranges::RangeDownload {
            client,
            bearer,
            destination,
            url: &canonical_url,
            partial: &partial,
            planned: &planned,
            test_http: false,
        })
        .await?;
    }
    let mut completed = false;
    for attempt in 0..3 {
        let offset = match tokio::fs::metadata(&partial).await {
            Ok(meta) => {
                ensure!(
                    meta.is_file() && meta.len() <= planned.bytes,
                    "invalid partial asset size"
                );
                meta.len()
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(error) => return Err(error).context("read partial asset metadata"),
        };
        if offset == planned.bytes {
            completed = true;
            break;
        }
        let transfer = async {
            let response = request_with_redirects(
                client,
                Url::parse(&canonical_url)?,
                bearer,
                (offset > 0).then_some(offset),
                false,
            )
            .await?;
            write_asset_response(response, &partial, offset, planned.bytes, &planned.filename).await
        }
        .await;
        match transfer {
            Ok(()) => {
                completed = true;
                break;
            }
            Err(error) if attempt == 2 => {
                return Err(error).with_context(|| {
                    format!("download asset {} after bounded retries", planned.filename)
                });
            }
            Err(_) => tokio::time::sleep(Duration::from_secs(1 << attempt)).await,
        }
    }
    ensure!(completed, "asset transfer incomplete");
    let sha256 = match verify_file_async(partial.clone(), planned.clone()).await {
        Ok(hash) => hash,
        Err(error) => {
            // Keep failed bytes for diagnosis, outside the published asset set.
            let rejected = destination.join(format!(
                ".{}.rejected-{}",
                planned.filename,
                uuid::Uuid::new_v4()
            ));
            tokio::fs::rename(&partial, rejected).await?;
            tokio::fs::remove_file(&state_path).await?;
            return Err(error);
        }
    };
    tokio::fs::rename(&partial, &final_path).await?;
    tokio::fs::remove_file(&state_path).await?;
    ranges::cleanup(destination, &planned)?;
    let sync_dir = destination.to_owned();
    tokio::task::spawn_blocking(move || File::open(sync_dir)?.sync_all()).await??;
    eprintln!(
        "verified Hub asset {} ({} bytes)",
        planned.filename, planned.bytes
    );
    Ok(HubAsset {
        filename: planned.filename,
        bytes: planned.bytes,
        publisher_identity: planned.publisher_identity,
        canonical_url,
        sha256,
    })
}

async fn write_asset_response(
    mut response: Response,
    partial: &Path,
    offset: u64,
    total: u64,
    filename: &str,
) -> Result<()> {
    ensure!(total > 0 && offset < total, "invalid partial byte offset");
    let append = offset > 0 && response.status() == StatusCode::PARTIAL_CONTENT;
    if append {
        let range = response
            .headers()
            .get(header::CONTENT_RANGE)
            .context("resumed asset lacks Content-Range")?
            .to_str()?;
        validate_content_range(range, offset, total)?;
    } else {
        if response.status() != StatusCode::OK {
            return Err(HubRequestFailure::Http {
                status: response.status().as_u16(),
            }
            .into());
        }
    }
    let expected_remaining = total - if append { offset } else { 0 };
    if let Some(length) = response.content_length() {
        ensure!(
            length == expected_remaining,
            "asset Content-Length mismatch"
        );
    }
    let mut options = tokio::fs::OpenOptions::new();
    options
        .write(true)
        .create(true)
        .append(append)
        .truncate(!append);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    let mut file = options.open(partial).await?;
    let mut received = 0_u64;
    let meter = FileProgress::new(
        filename,
        "receiving",
        total,
        if append { offset } else { 0 },
    );
    meter.report(0);
    let mut heartbeat = tokio::time::interval_at(
        tokio::time::Instant::now() + Duration::from_secs(15),
        Duration::from_secs(15),
    );
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            chunk = response.chunk() => {
                let chunk = chunk.map_err(|_| HubRequestFailure::Connectivity { stage:"asset stream" })?;
                let Some(chunk) = chunk else { break; };
                received = received.checked_add(chunk.len() as u64).context("asset byte counter overflow")?;
                ensure!(received <= expected_remaining, "asset stream exceeds pinned size");
                file.write_all(&chunk).await?;
            }
            _ = heartbeat.tick() => {
                // Partial bytes have passed size bounds, but no complete hash yet.
                meter.report(received);
            }
        }
    }
    file.sync_all().await?;
    ensure!(received == expected_remaining, "asset stream incomplete");
    meter.report(received);
    Ok(())
}

/// Explicit online operation. Normal model loading calls only the offline validator.
pub async fn download_model(options: HubDownloadOptions) -> Result<HubProvenance> {
    download_model_for(options, false).await
}

/// Retrieve the official packed BitNet student as data only. Its published
/// `auto_map` is retained but never imported or executed by this Rust build.
/// The assessment model loader and its stricter validator remain unchanged.
pub async fn download_bitnet_student(options: HubDownloadOptions) -> Result<HubProvenance> {
    ensure!(
        options.model_id == "microsoft/bitnet-b1.58-2B-4T"
            && options.revision == "04c3b9ad9361b824064a1f25ea60a8be9599b127",
        "BitNet student acquisition requires the reviewed official immutable pin"
    );
    ensure!(
        options.destination.starts_with("offline_bundle/students")
            && options
                .destination
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "BitNet student must remain under the protected student asset root"
    );
    download_model_for(options, true).await
}

async fn download_model_for(options: HubDownloadOptions, bitnet: bool) -> Result<HubProvenance> {
    validate_model_id(&options.model_id)?;
    validate_ref(&options.revision)?;
    ensure!(
        (1..=8).contains(&options.concurrency),
        "Hub download concurrency must be between one and eight"
    );
    let token_stdin = options.token_stdin;
    let bearer = tokio::task::spawn_blocking(move || load_bearer(token_stdin)).await??;
    let client = client()?;
    let (resolved, _) = metadata(
        &client,
        &options.model_id,
        &options.revision,
        bearer.as_ref(),
    )
    .await?;
    ensure!(
        valid_hex(&resolved.sha, 40),
        "Hub did not resolve an immutable commit"
    );
    let (pinned, metadata_bytes) =
        metadata(&client, &options.model_id, &resolved.sha, bearer.as_ref()).await?;
    ensure!(
        pinned.sha == resolved.sha,
        "Hub pinned metadata revision mismatch"
    );
    let plan = plan_from_metadata(&options.model_id, &pinned)?;
    let destination = options.destination.clone();
    let _lock =
        tokio::task::spawn_blocking(move || DestinationLock::acquire(&destination)).await??;
    let plan_path = options.destination.join(PLAN_FILE);
    let persist_plan = plan.clone();
    tokio::task::spawn_blocking(move || {
        ensure_safe_path(&plan_path)?;
        if plan_path.exists() {
            let existing: DownloadPlan = serde_json::from_slice(&fs::read(&plan_path)?)?;
            ensure!(
                existing == persist_plan,
                "destination contains a different pinned model; use a new destination"
            );
        } else {
            atomic_publish(&plan_path, &serde_json::to_vec_pretty(&persist_plan)?)?;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    eprintln!(
        "resolved {} at immutable commit {} ({} assets; no executable repository files)",
        plan.model_id,
        plan.revision,
        plan.files.len()
    );
    let client = Arc::new(client);
    let bearer = bearer.map(Arc::new);
    let mut jobs = stream::iter(plan.files.values().cloned())
        .map(|file| {
            let client = client.clone();
            let bearer = bearer.clone();
            let destination = options.destination.clone();
            let model_id = plan.model_id.clone();
            let revision = plan.revision.clone();
            async move {
                download_file(
                    &client,
                    bearer.as_deref(),
                    &destination,
                    &model_id,
                    &revision,
                    file,
                )
                .await
            }
        })
        .buffer_unordered(options.concurrency);
    let mut files = BTreeMap::new();
    while let Some(result) = jobs.next().await {
        let asset = result?;
        files.insert(asset.filename.clone(), asset);
    }
    drop(jobs);
    let licence = licence_from_metadata(&pinned);
    let licence_files: Vec<String> = files
        .keys()
        .filter(|name| licence_filename(name))
        .cloned()
        .collect();
    let licence_evidence = licence_evidence(!licence_files.is_empty());
    let provenance = HubProvenance {
        schema_version: 1,
        model_id: plan.model_id.clone(),
        requested_revision: options.revision,
        revision: plan.revision.clone(),
        licence: licence.clone(),
        licence_files,
        licence_evidence,
        recorded_unix_seconds: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        metadata_url: format!(
            "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
            plan.model_id, plan.revision
        ),
        metadata_sha256: hex::encode(Sha256::digest(&metadata_bytes)),
        metadata: serde_json::from_slice(&metadata_bytes)?,
        files,
        authentication_policy: AUTHENTICATION_POLICY.into(),
    };
    let manifest = AssetManifest {
        model_id: plan.model_id,
        revision: plan.revision,
        licence,
        files: provenance
            .files
            .iter()
            .map(|(name, asset)| (name.clone(), asset.sha256.clone()))
            .collect(),
    };
    let dir = options.destination;
    let persist_provenance = provenance.clone();
    tokio::task::spawn_blocking(move || {
        // Validate repository data before the final assets.json publication.
        validate_weight_index(&dir, &manifest)?;
        validate_config_for(&dir, bitnet)?;
        // Original bytes make the metadata digest independent of JSON formatting.
        atomic_publish(&dir.join("hub-metadata.json"), &metadata_bytes)?;
        atomic_publish(
            &dir.join(PROVENANCE_FILE),
            &serde_json::to_vec_pretty(&persist_provenance)?,
        )?;
        atomic_publish(
            &dir.join("assets.json"),
            &serde_json::to_vec_pretty(&manifest)?,
        )?;
        validate_downloaded_assets_for(&dir, bitnet)
    })
    .await??;
    Ok(provenance)
}

/// Verify comparator assets offline, without changing the assessment base pin.
pub fn validate_downloaded_assets(base_dir: &Path) -> Result<AssetManifest> {
    validate_downloaded_assets_for(base_dir, false)
}

/// Validate packed official BitNet data without permitting it into Qwen loaders.
pub fn validate_bitnet_student_assets(base_dir: &Path) -> Result<AssetManifest> {
    let manifest = validate_downloaded_assets_for(base_dir, true)?;
    ensure!(
        manifest.model_id == "microsoft/bitnet-b1.58-2B-4T"
            && manifest.revision == "04c3b9ad9361b824064a1f25ea60a8be9599b127"
            && manifest
                .files
                .get("model.safetensors")
                .is_some_and(|hash| hash
                    == "8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b"),
        "BitNet student differs from the reviewed official tensor identity"
    );
    Ok(manifest)
}

fn validate_downloaded_assets_for(base_dir: &Path, bitnet: bool) -> Result<AssetManifest> {
    validate_downloaded_assets_with_policy(base_dir, bitnet, false)
}

fn validate_downloaded_assets_with_policy(
    base_dir: &Path,
    bitnet: bool,
    historical: bool,
) -> Result<AssetManifest> {
    ensure_safe_path(base_dir)?;
    for name in [
        "assets.json",
        PROVENANCE_FILE,
        "hub-metadata.json",
        PLAN_FILE,
    ] {
        ensure_safe_path(&base_dir.join(name))?;
    }
    let manifest: AssetManifest = serde_json::from_slice(&fs::read(base_dir.join("assets.json"))?)?;
    let provenance: HubProvenance =
        serde_json::from_slice(&fs::read(base_dir.join(PROVENANCE_FILE))?)?;
    let bytes = fs::read(base_dir.join("hub-metadata.json"))?;
    ensure!(
        bytes.len() <= MAX_METADATA_BYTES,
        "Hub metadata artifact oversized"
    );
    ensure!(
        hex::encode(Sha256::digest(&bytes)) == provenance.metadata_sha256,
        "Hub metadata artifact checksum mismatch"
    );
    let metadata: ApiModel = serde_json::from_slice(&bytes)?;
    let plan = plan_from_metadata(&manifest.model_id, &metadata)?;
    let saved_plan: DownloadPlan = serde_json::from_slice(&fs::read(base_dir.join(PLAN_FILE))?)?;
    ensure!(plan == saved_plan, "Hub plan differs from pinned metadata");
    ensure!(
        provenance.schema_version == 1
            && manifest.model_id == provenance.model_id
            && manifest.revision == provenance.revision
            && manifest.revision == plan.revision
            && manifest.licence == provenance.licence,
        "Hub asset provenance mismatch"
    );
    ensure!(
        manifest.licence == licence_from_metadata(&metadata),
        "Hub licence differs from original model metadata"
    );
    ensure!(
        provenance.licence_files
            == plan
                .files
                .keys()
                .filter(|name| licence_filename(name))
                .cloned()
                .collect::<Vec<_>>(),
        "Hub licence evidence files changed"
    );
    ensure!(
        provenance.licence_evidence == licence_evidence(!provenance.licence_files.is_empty()),
        "Hub licence evidence declaration changed"
    );
    ensure!(
        provenance.metadata_url
            == format!(
                "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
                plan.model_id, plan.revision
            ),
        "Hub metadata provenance URL changed"
    );
    ensure!(
        provenance.authentication_policy == AUTHENTICATION_POLICY,
        "Hub authentication policy provenance changed"
    );
    validate_ref(&provenance.requested_revision)?;
    ensure!(
        provenance.metadata == serde_json::from_slice::<serde_json::Value>(&bytes)?,
        "Hub provenance metadata changed"
    );
    ensure!(
        manifest.files.len() == plan.files.len() && provenance.files.len() == plan.files.len(),
        "Hub assets missing or unexpected"
    );
    for (name, planned) in &plan.files {
        let asset = provenance
            .files
            .get(name)
            .context("Hub provenance asset missing")?;
        ensure!(
            asset.filename == *name
                && asset.bytes == planned.bytes
                && asset.publisher_identity == planned.publisher_identity
                && asset.canonical_url == canonical_file_url(&plan.model_id, &plan.revision, name),
            "Hub publisher identity changed"
        );
        let sha256 = if historical && name.ends_with(".safetensors") {
            historical::verify_weight_metadata(base_dir, "assets.json", planned, &asset.sha256)?;
            asset.sha256.clone()
        } else {
            verify_file(&base_dir.join(name), planned)?
        };
        ensure!(
            manifest.files.get(name) == Some(&sha256) && asset.sha256 == sha256,
            "Hub asset SHA-256 manifest mismatch: {name}"
        );
    }
    validate_weight_index(base_dir, &manifest)?;
    validate_config_for(base_dir, bitnet)?;
    Ok(manifest)
}

fn validate_config(base_dir: &Path) -> Result<()> {
    validate_config_for(base_dir, false)
}

fn validate_config_for(base_dir: &Path, bitnet: bool) -> Result<()> {
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
    if bitnet {
        ensure!(
            config["model_type"] == "bitnet"
                && config["architectures"] == serde_json::json!(["BitNetForCausalLM"])
                && config["auto_map"]
                    == serde_json::json!({"AutoConfig":"configuration_bitnet.BitNetConfig","AutoModelForCausalLM":"modeling_bitnet.BitNetForCausalLM"})
                && config["custom_pipelines"].is_null()
                && config["hidden_size"] == 2560
                && config["num_hidden_layers"] == 30
                && config["num_attention_heads"] == 20
                && config["num_key_value_heads"] == 5
                && config["intermediate_size"] == 6912
                && config["vocab_size"] == 128256
                && config["hidden_act"] == "relu2"
                && config["tie_word_embeddings"] == true
                && config["quantization_config"]
                    == serde_json::json!({"quant_method":"bitnet","linear_class":"autobitlinear","quantization_mode":"offline"}),
            "unsupported or modified official BitNet architecture data"
        );
        return Ok(());
    }
    ensure!(
        config.get("auto_map").is_none() && config.get("custom_pipelines").is_none(),
        "remote executable model code is unsupported"
    );
    Ok(())
}

fn validate_weight_index(base_dir: &Path, manifest: &AssetManifest) -> Result<()> {
    if manifest.files.contains_key("model.safetensors.index.json") {
        let index: serde_json::Value =
            serde_json::from_slice(&fs::read(base_dir.join("model.safetensors.index.json"))?)?;
        let weights = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .context("safetensors index lacks weight_map")?;
        ensure!(!weights.is_empty(), "empty safetensors weight_map");
        for shard in weights.values() {
            let shard = shard.as_str().context("invalid safetensors shard name")?;
            ensure!(
                allowed_filename(shard)
                    && shard.ends_with(".safetensors")
                    && manifest.files.contains_key(shard),
                "unverified safetensors shard in index"
            );
        }
    } else {
        ensure!(
            manifest.files.contains_key("model.safetensors"),
            "sharded weights require an original index"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn allowlist_rejects_executable_traversal_and_unrecognised_weights() {
        for name in [
            "model.py",
            "model.bin",
            "adapter_model.bin",
            "../config.json",
            "/LICENSE",
            "sub/model.safetensors",
            "model-00000-of-00002.safetensors",
            "model-00003-of-00002.safetensors",
        ] {
            assert!(!allowed_filename(name), "{name}");
        }
        assert!(allowed_filename("model-00001-of-00002.safetensors"));
        assert!(allowed_filename("tokenizer.model"));
    }

    fn file_metadata(contents: &[u8], name: &str) -> ApiFile {
        let mut git = Sha1::new();
        git.update(format!("blob {}\0", contents.len()));
        git.update(contents);
        ApiFile {
            rfilename: name.into(),
            blob_id: hex::encode(git.finalize()),
            size: contents.len() as u64,
            lfs: None,
        }
    }

    #[test]
    fn publisher_checksums_distinguish_git_blob_and_lfs_and_reject_corruption() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("README.md");
        fs::write(&path, b"licenced bytes")?;
        let file = file_metadata(b"licenced bytes", "README.md");
        let planned = PlannedFile {
            filename: file.rfilename,
            bytes: file.size,
            publisher_identity: PublisherIdentity::GitBlobSha1(file.blob_id),
        };
        let expected = hex::encode(Sha256::digest(b"licenced bytes"));
        assert_eq!(verify_file(&path, &planned)?, expected);
        let mut lfs = planned.clone();
        lfs.publisher_identity = PublisherIdentity::LfsSha256(expected);
        verify_file(&path, &lfs)?;
        fs::write(&path, b"corruptd bytes")?;
        assert!(verify_file(&path, &planned).is_err());
        assert!(verify_file(&path, &lfs).is_err());
        Ok(())
    }

    #[test]
    fn metadata_requires_commit_and_consistent_publisher_sizes() -> Result<()> {
        let mut model = ApiModel {
            id: "organisation/model".into(),
            sha: "a".repeat(40),
            card_data: serde_json::json!({"license":"apache-2.0"}),
            siblings: [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "LICENSE",
                "model.safetensors",
            ]
            .into_iter()
            .map(|name| file_metadata(b"{}", name))
            .collect(),
        };
        assert_eq!(
            plan_from_metadata("organisation/model", &model)?.revision,
            "a".repeat(40)
        );
        model.sha = "main".into();
        assert!(plan_from_metadata("organisation/model", &model).is_err());
        model.sha = "a".repeat(40);
        model.siblings[4].lfs = Some(ApiLfs {
            sha256: "b".repeat(64),
            size: 3,
        });
        assert!(plan_from_metadata("organisation/model", &model).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn bearer_is_sensitive_and_never_survives_redirects_even_to_same_origin() -> Result<()> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for index in 0..3 {
                let (mut socket, _) = listener.accept().await?;
                let mut bytes = Vec::new();
                loop {
                    let mut buffer = [0_u8; 2048];
                    let n = socket.read(&mut buffer).await?;
                    ensure!(n > 0, "test HTTP request ended early");
                    bytes.extend_from_slice(&buffer[..n]);
                    if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                requests.push(String::from_utf8(bytes)?);
                let response = if index < 2 {
                    format!(
                        "HTTP/1.1 302 Found\r\nLocation: http://{address}/redirect-{}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        index + 1
                    )
                } else {
                    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok".into()
                };
                socket.write_all(response.as_bytes()).await?;
            }
            Ok::<_, anyhow::Error>(requests)
        });
        let token = Bearer::from_bytes(b"public-test-placeholder-no-credentials")?;
        assert!(token.0.is_sensitive());
        let response = request_with_redirects(
            &client()?,
            Url::parse(&format!("http://{address}/initial"))?,
            Some(&token),
            None,
            true,
        )
        .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let requests = server.await??;
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer public-test-placeholder-no-credentials")
        );
        assert!(!requests[1].to_ascii_lowercase().contains("authorization:"));
        assert!(!requests[2].to_ascii_lowercase().contains("authorization:"));
        Ok(())
    }

    #[test]
    fn resumed_download_requires_exact_range_and_rejects_symlinks() -> Result<()> {
        validate_content_range("bytes 4-9/10", 4, 10)?;
        assert!(validate_content_range("bytes 0-9/10", 4, 10).is_err());
        assert!(validate_content_range("bytes 4-8/10", 4, 10).is_err());
        assert!(validate_content_range("bytes 4-9/11", 4, 10).is_err());
        #[cfg(unix)]
        {
            let dir = tempfile::tempdir()?;
            std::os::unix::fs::symlink("/tmp", dir.path().join("link"))?;
            assert!(ensure_safe_path(&dir.path().join("link/model.safetensors")).is_err());
        }
        Ok(())
    }

    #[test]
    fn model_card_spdx_declaration_is_explicit_and_unknown_licence_fails_closed() -> Result<()> {
        let mut model = ApiModel {
            id: "organisation/model".into(),
            sha: "a".repeat(40),
            card_data: serde_json::json!({"license":"apache-2.0"}),
            siblings: [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "README.md",
                "model.safetensors",
            ]
            .into_iter()
            .map(|name| file_metadata(b"{}", name))
            .collect(),
        };
        plan_from_metadata("organisation/model", &model)?;
        model.card_data = serde_json::json!({"license":"unspecified"});
        assert!(plan_from_metadata("organisation/model", &model).is_err());
        // A notice alone is not a grant of licence rights.
        model.siblings.push(file_metadata(b"notice", "NOTICE"));
        assert!(plan_from_metadata("organisation/model", &model).is_err());
        model
            .siblings
            .push(file_metadata(b"licence document", "LICENSE"));
        plan_from_metadata("organisation/model", &model)?;
        model.id = "another/repository".into();
        assert!(plan_from_metadata("organisation/model", &model).is_err());
        Ok(())
    }

    fn fixture_bundle(dir: &Path) -> Result<()> {
        let contents = b"{}";
        let metadata = ApiModel {
            id: "organisation/model".into(),
            sha: "a".repeat(40),
            card_data: serde_json::json!({"license":"apache-2.0"}),
            siblings: [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "LICENSE",
                "model.safetensors",
            ]
            .into_iter()
            .map(|name| {
                let mut file = file_metadata(contents, name);
                if name.ends_with(".safetensors") {
                    file.lfs = Some(ApiLfs {
                        sha256: hex::encode(Sha256::digest(contents)),
                        size: contents.len() as u64,
                    });
                }
                file
            })
            .collect(),
        };
        let metadata_bytes = serde_json::to_vec(&metadata)?;
        let plan = plan_from_metadata(&metadata.id, &metadata)?;
        let mut files = BTreeMap::new();
        for (name, planned) in &plan.files {
            fs::write(dir.join(name), contents)?;
            files.insert(
                name.clone(),
                HubAsset {
                    filename: name.clone(),
                    bytes: planned.bytes,
                    publisher_identity: planned.publisher_identity.clone(),
                    canonical_url: canonical_file_url(&plan.model_id, &plan.revision, name),
                    sha256: hex::encode(Sha256::digest(contents)),
                },
            );
        }
        let manifest = AssetManifest {
            model_id: plan.model_id.clone(),
            revision: plan.revision.clone(),
            licence: "apache-2.0".into(),
            files: files
                .iter()
                .map(|(name, asset)| (name.clone(), asset.sha256.clone()))
                .collect(),
        };
        let provenance = HubProvenance {
            schema_version: 1,
            model_id: plan.model_id.clone(),
            requested_revision: "main".into(),
            revision: plan.revision.clone(),
            licence: "apache-2.0".into(),
            licence_files: vec!["LICENSE".into()],
            licence_evidence: licence_evidence(true),
            recorded_unix_seconds: 1,
            metadata_url: format!(
                "https://huggingface.co/api/models/{}/revision/{}?blobs=true",
                plan.model_id, plan.revision
            ),
            metadata_sha256: hex::encode(Sha256::digest(&metadata_bytes)),
            metadata: serde_json::from_slice(&metadata_bytes)?,
            files,
            authentication_policy: AUTHENTICATION_POLICY.into(),
        };
        fs::write(dir.join("assets.json"), serde_json::to_vec(&manifest)?)?;
        fs::write(dir.join(PROVENANCE_FILE), serde_json::to_vec(&provenance)?)?;
        fs::write(dir.join("hub-metadata.json"), metadata_bytes)?;
        fs::write(dir.join(PLAN_FILE), serde_json::to_vec(&plan)?)?;
        Ok(())
    }

    #[test]
    fn historical_metadata_is_bound_but_never_authorises_changed_weights() -> Result<()> {
        let dir = tempfile::tempdir()?;
        fixture_bundle(dir.path())?;
        validate_downloaded_asset_metadata(dir.path())?;
        // Same-size damage is intentionally outside historical proof. A new
        // model load must still catch it through the unchanged full validator.
        fs::write(dir.path().join("model.safetensors"), b"[]")?;
        validate_downloaded_asset_metadata(dir.path())?;
        assert!(validate_downloaded_assets(dir.path()).is_err());
        fs::write(dir.path().join("tokenizer.json"), b"[]")?;
        assert!(validate_downloaded_asset_metadata(dir.path()).is_err());
        fixture_bundle(dir.path())?;
        fs::remove_file(dir.path().join("model.safetensors"))?;
        assert!(validate_downloaded_asset_metadata(dir.path()).is_err());
        Ok(())
    }

    #[test]
    fn offline_validator_rejects_changed_commit_licence_bytes_and_index_paths() -> Result<()> {
        let dir = tempfile::tempdir()?;
        fixture_bundle(dir.path())?;
        validate_downloaded_assets(dir.path())?;
        let path = dir.path().join(PROVENANCE_FILE);
        let mut provenance: HubProvenance = serde_json::from_slice(&fs::read(&path)?)?;
        provenance.revision = "b".repeat(40);
        fs::write(&path, serde_json::to_vec(&provenance)?)?;
        assert!(validate_downloaded_assets(dir.path()).is_err());
        fixture_bundle(dir.path())?;
        let mut manifest: AssetManifest =
            serde_json::from_slice(&fs::read(dir.path().join("assets.json"))?)?;
        manifest.licence = "mit".into();
        fs::write(
            dir.path().join("assets.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        assert!(validate_downloaded_assets(dir.path()).is_err());
        fixture_bundle(dir.path())?;
        fs::write(dir.path().join("model.safetensors"), b"[]")?;
        assert!(validate_downloaded_assets(dir.path()).is_err());
        let manifest = AssetManifest {
            model_id: "organisation/model".into(),
            revision: "a".repeat(40),
            licence: "apache-2.0".into(),
            files: BTreeMap::from([("model.safetensors.index.json".into(), "placeholder".into())]),
        };
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"layer":"../outside.safetensors"}}"#,
        )?;
        assert!(validate_weight_index(dir.path(), &manifest).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn bearer_does_not_cross_origin_on_redirect() -> Result<()> {
        let origin = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let target = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let origin_address = origin.local_addr()?;
        let target_address = target.local_addr()?;
        let first = tokio::spawn(async move {
            let (mut socket, _) = origin.accept().await?;
            let request = read_test_request(&mut socket).await?;
            socket.write_all(format!("HTTP/1.1 302 Found\r\nLocation: http://{target_address}/asset\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await?;
            Ok::<_, anyhow::Error>(request)
        });
        let second = tokio::spawn(async move {
            let (mut socket, _) = target.accept().await?;
            let request = read_test_request(&mut socket).await?;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                .await?;
            Ok::<_, anyhow::Error>(request)
        });
        let token = Bearer::from_bytes(b"public-test-placeholder-no-credentials")?;
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            request_with_redirects(
                &client()?,
                Url::parse(&format!("http://{origin_address}/start"))?,
                Some(&token),
                None,
                true,
            ),
        )
        .await??;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            first
                .await??
                .to_ascii_lowercase()
                .contains("authorization:")
        );
        assert!(
            !second
                .await??
                .to_ascii_lowercase()
                .contains("authorization:")
        );
        Ok(())
    }

    async fn read_test_request(socket: &mut tokio::net::TcpStream) -> Result<String> {
        let mut bytes = Vec::new();
        loop {
            let mut buffer = [0_u8; 2048];
            let n = socket.read(&mut buffer).await?;
            ensure!(n > 0, "test HTTP request ended early");
            bytes.extend_from_slice(&buffer[..n]);
            ensure!(bytes.len() <= 8192, "test HTTP request too large");
            if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
        }
        Ok(String::from_utf8(bytes)?)
    }

    async fn test_response(raw: &'static [u8], range: u64) -> Result<Response> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let request = read_test_request(&mut socket).await?;
            ensure!(
                request
                    .to_ascii_lowercase()
                    .contains(&format!("range: bytes={range}-")),
                "resume test lacks Range header"
            );
            socket.write_all(raw).await?;
            Ok::<_, anyhow::Error>(())
        });
        tokio::time::timeout(
            Duration::from_secs(5),
            request_with_redirects(
                &client()?,
                Url::parse(&format!("http://{address}/asset"))?,
                None,
                Some(range),
                true,
            ),
        )
        .await?
    }

    #[tokio::test]
    async fn real_http_resume_appends_206_replaces_200_and_rejects_wrong_range() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let partial = dir.path().join(".model.safetensors.partial");
        fs::write(&partial, b"abc")?;
        let response = test_response(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef", 3).await?;
        write_asset_response(response, &partial, 3, 6, "model.safetensors").await?;
        assert_eq!(fs::read(&partial)?, b"abcdef");
        fs::write(&partial, b"abc")?;
        let response = test_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nuvwxyz",
            3,
        )
        .await?;
        write_asset_response(response, &partial, 3, 6, "model.safetensors").await?;
        assert_eq!(fs::read(&partial)?, b"uvwxyz");
        fs::write(&partial, b"abc")?;
        let response = test_response(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 2-4/6\r\nConnection: close\r\n\r\ndef", 3).await?;
        assert!(
            write_asset_response(response, &partial, 3, 6, "model.safetensors")
                .await
                .is_err()
        );
        assert_eq!(fs::read(&partial)?, b"abc");
        let response = test_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nshort",
            3,
        )
        .await?;
        assert!(
            write_asset_response(response, &partial, 3, 6, "model.safetensors")
                .await
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn destination_lock_rejects_parallel_writers_and_releases_on_drop() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let first = DestinationLock::acquire(dir.path())?;
        assert!(DestinationLock::acquire(dir.path()).is_err());
        drop(first);
        DestinationLock::acquire(dir.path())?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn secret_terminal_input_disables_echo_and_restores_original_settings() -> Result<()> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
        let mut master = -1;
        let mut slave = -1;
        // SAFETY: openpty initialises both descriptors; optional arguments are null.
        ensure!(
            unsafe {
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                )
            } == 0,
            "open test pseudoterminal"
        );
        // SAFETY: these newly returned descriptors have exactly one owner each.
        let master = unsafe { OwnedFd::from_raw_fd(master) };
        // SAFETY: slave is the distinct second descriptor returned by openpty.
        let slave = unsafe { OwnedFd::from_raw_fd(slave) };
        let terminal_settings = |fd| -> Result<libc::termios> {
            let mut value = std::mem::MaybeUninit::uninit();
            // SAFETY: tcgetattr writes a termios into the provided storage.
            ensure!(
                unsafe { libc::tcgetattr(fd, value.as_mut_ptr()) } == 0,
                "read pseudoterminal settings"
            );
            // SAFETY: tcgetattr succeeded and fully initialised the object.
            Ok(unsafe { value.assume_init() })
        };
        let original = terminal_settings(slave.as_raw_fd())?;
        assert_ne!(original.c_lflag & libc::ECHO, 0);
        {
            let _guard = TerminalEchoGuard::new(slave.as_raw_fd())?;
            let silent = terminal_settings(slave.as_raw_fd())?;
            assert_eq!(silent.c_lflag & (libc::ECHO | libc::ECHONL), 0);
            let placeholder = b"public-test-placeholder-no-credentials\n";
            // SAFETY: the borrowed byte slice and live descriptor remain valid.
            assert_eq!(
                unsafe {
                    libc::write(
                        master.as_raw_fd(),
                        placeholder.as_ptr().cast(),
                        placeholder.len(),
                    )
                },
                placeholder.len() as isize
            );
            let mut event = libc::pollfd {
                fd: master.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: poll borrows one initialised pollfd for the stated duration.
            assert_eq!(
                unsafe { libc::poll(&mut event, 1, 50) },
                0,
                "terminal must not echo secret input"
            );
        }
        assert_eq!(
            terminal_settings(slave.as_raw_fd())?.c_lflag,
            original.c_lflag
        );
        let mut descriptors = [-1; 2];
        // SAFETY: pipe initialises two descriptors in the supplied array.
        ensure!(
            unsafe { libc::pipe(descriptors.as_mut_ptr()) } == 0,
            "create non-terminal test pipe"
        );
        // SAFETY: each new pipe descriptor acquires exactly one owner.
        let read = unsafe { OwnedFd::from_raw_fd(descriptors[0]) };
        // SAFETY: this is the distinct new pipe write descriptor.
        let _write = unsafe { OwnedFd::from_raw_fd(descriptors[1]) };
        assert!(TerminalEchoGuard::new(read.as_raw_fd())?.original.is_none());
        Ok(())
    }

    #[test]
    fn cached_only_signal_keeps_permanent_http_and_corruption_distinct() {
        for status in [408, 425, 429, 500, 502, 503, 504] {
            let error = anyhow::Error::new(HubRequestFailure::Http { status })
                .context("bounded retries exhausted");
            assert!(is_connectivity_error(&error));
            assert!(format!("{error:#}").contains(&status.to_string()));
        }
        for status in [400, 401, 403, 404, 410, 422] {
            assert!(!is_connectivity_error(
                &HubRequestFailure::Http { status }.into()
            ));
        }
        assert!(is_connectivity_error(
            &HubRequestFailure::Connectivity {
                stage: "response headers"
            }
            .into()
        ));
        assert!(!is_connectivity_error(&anyhow::anyhow!(
            "publisher checksum mismatch"
        )));
    }

    #[test]
    fn official_bitnet_metadata_exception_never_widens_the_assessment_loader() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let mut config = serde_json::json!({"model_type":"bitnet","architectures":["BitNetForCausalLM"],"auto_map":{"AutoConfig":"configuration_bitnet.BitNetConfig","AutoModelForCausalLM":"modeling_bitnet.BitNetForCausalLM"},"hidden_size":2560,"num_hidden_layers":30,"num_attention_heads":20,"num_key_value_heads":5,"intermediate_size":6912,"vocab_size":128256,"hidden_act":"relu2","tie_word_embeddings":true,"quantization_config":{"quant_method":"bitnet","linear_class":"autobitlinear","quantization_mode":"offline"}});
        fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config)?,
        )?;
        assert!(validate_config(directory.path()).is_err());
        validate_config_for(directory.path(), true)?;
        config["auto_map"]["AutoConfig"] = serde_json::json!("unreviewed.RemoteCode");
        fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&config)?,
        )?;
        assert!(validate_config_for(directory.path(), true).is_err());
        Ok(())
    }
}
