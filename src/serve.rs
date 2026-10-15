//! Bounded localhost inference and atomic routing between immutable model processes.

use crate::model::{
    Generation, GenerationRequest, ModelRuntime, RuntimeOptions, atomic_write, sha256_file,
};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeConfig {
    pub address: String,
    pub queue_capacity: usize,
    pub maximum_request_bytes: usize,
    pub maximum_output_tokens: usize,
    pub timeout_seconds: u64,
}
impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:8080".into(),
            queue_capacity: 4,
            maximum_request_bytes: 32768,
            maximum_output_tokens: 256,
            timeout_seconds: 120,
        }
    }
}
impl ServeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let value: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<SocketAddr> {
        let addr: SocketAddr = self.address.parse()?;
        ensure!(
            addr.ip().is_loopback(),
            "inference must bind to a loopback address"
        );
        ensure!(
            (1..=32).contains(&self.queue_capacity)
                && (1024..=262144).contains(&self.maximum_request_bytes)
                && (1..=1024).contains(&self.maximum_output_tokens)
                && (1..=1800).contains(&self.timeout_seconds),
            "invalid serving limits"
        );
        Ok(addr)
    }
}

struct Job {
    request: GenerationRequest,
    reply: oneshot::Sender<std::result::Result<Generation, String>>,
    cancel: Arc<AtomicBool>,
    expires: Instant,
}
#[derive(Clone)]
struct ServiceState {
    queue: mpsc::Sender<Job>,
    metadata: Arc<Health>,
    config: ServeConfig,
    requests: Arc<AtomicU64>,
    errors: Arc<AtomicU64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Health {
    #[serde(default)]
    pub process_id: u32,
    pub healthy: bool,
    pub checkpoint_id: String,
    pub model_id: String,
    pub base_revision: String,
    pub tokenizer_sha256: String,
    pub checksum_validated: bool,
    pub inference_smoke_output: String,
    pub smoke_elapsed_ms: u128,
    pub immutable: bool,
    pub production_eligible: bool,
    #[serde(default)]
    pub execution: serde_json::Value,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
    checkpoint_id: String,
}
type HttpError = (StatusCode, Json<ErrorBody>);
fn http_error(state: &ServiceState, status: StatusCode, error: impl Into<String>) -> HttpError {
    state.errors.fetch_add(1, Ordering::Relaxed);
    (
        status,
        Json(ErrorBody {
            error: error.into(),
            checkpoint_id: state.metadata.checkpoint_id.clone(),
        }),
    )
}

async fn health(State(state): State<ServiceState>) -> Json<Health> {
    Json((*state.metadata).clone())
}
async fn metadata(State(state): State<ServiceState>) -> Json<serde_json::Value> {
    Json(
        serde_json::json!({"model":*state.metadata,"limits":state.config,"requests":state.requests.load(Ordering::Relaxed),"errors":state.errors.load(Ordering::Relaxed)}),
    )
}
async fn generate(
    State(state): State<ServiceState>,
    Json(mut request): Json<GenerationRequest>,
) -> std::result::Result<Json<Generation>, HttpError> {
    state.requests.fetch_add(1, Ordering::Relaxed);
    if request.max_new_tokens > state.config.maximum_output_tokens {
        return Err(http_error(
            &state,
            StatusCode::BAD_REQUEST,
            "output cap exceeds this endpoint's limit",
        ));
    }
    request.timeout_seconds = request.timeout_seconds.min(state.config.timeout_seconds);
    if request.timeout_seconds == 0 {
        return Err(http_error(
            &state,
            StatusCode::BAD_REQUEST,
            "inference deadline must be positive",
        ));
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let (reply, rx) = oneshot::channel();
    let timeout = Duration::from_secs(request.timeout_seconds);
    let job = Job {
        request,
        reply,
        cancel: cancel.clone(),
        expires: Instant::now() + timeout,
    };
    state.queue.try_send(job).map_err(|e| {
        http_error(
            &state,
            StatusCode::SERVICE_UNAVAILABLE,
            format!("bounded inference queue unavailable: {e}"),
        )
    })?;
    match tokio::time::timeout(timeout, rx).await {
        Ok(Ok(Ok(result))) if result.complete => Ok(Json(result)),
        Ok(Ok(Ok(result))) => Err(http_error(
            &state,
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("incomplete generation ({})", result.finish_reason),
        )),
        Ok(Ok(Err(message))) => Err(http_error(
            &state,
            StatusCode::UNPROCESSABLE_ENTITY,
            message,
        )),
        Ok(Err(_)) => Err(http_error(
            &state,
            StatusCode::SERVICE_UNAVAILABLE,
            "model worker stopped",
        )),
        Err(_) => {
            cancel.store(true, Ordering::Relaxed);
            Err(http_error(
                &state,
                StatusCode::GATEWAY_TIMEOUT,
                "inference deadline exceeded; worker cancellation requested",
            ))
        }
    }
}

/// One immutable checkpoint per process. CPU work runs on one dedicated worker;
/// queue pressure and timeouts cannot create an unbounded spawn_blocking backlog.
pub async fn run(
    base_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    config: ServeConfig,
) -> Result<()> {
    run_with_options(base_dir, checkpoint, config, RuntimeOptions::default()).await
}

pub async fn run_with_options(
    base_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    config: ServeConfig,
    options: RuntimeOptions,
) -> Result<()> {
    run_inner(base_dir, checkpoint, config, options, false).await
}

/// Comparative teachers are evaluation-only; they cannot authorise production
/// student routing, even if their local inference health checks pass.
pub async fn run_comparative(
    base_dir: PathBuf,
    config: ServeConfig,
    options: RuntimeOptions,
) -> Result<()> {
    run_inner(base_dir, None, config, options, true).await
}

fn assessment_execution_eligible(checkpoint_present: bool, options: &RuntimeOptions) -> bool {
    !checkpoint_present || *options == RuntimeOptions::default()
}

async fn run_inner(
    base_dir: PathBuf,
    checkpoint: Option<PathBuf>,
    config: ServeConfig,
    options: RuntimeOptions,
    comparative: bool,
) -> Result<()> {
    let addr = config.validate()?;
    let (tx, mut rx) = mpsc::channel::<Job>(config.queue_capacity);
    let (ready_tx, ready_rx) = oneshot::channel::<std::result::Result<Health, String>>();
    std::thread::Builder::new()
        .name("qwen-inference".into())
        .spawn(move || {
            let runtime = (|| -> Result<(ModelRuntime, Health)> {
                // Existing assessment evidence fixes CPU float32 execution.
                // Tensor identity alone cannot authorise another precision/device.
                let execution_eligible =
                    assessment_execution_eligible(checkpoint.is_some(), &options);
                let mut model = if comparative {
                    ModelRuntime::load_comparative_with_options(&base_dir, options)?
                } else {
                    ModelRuntime::load_with_options(&base_dir, checkpoint.as_deref(), options)?
                };
                let production_eligible = !comparative
                    && execution_eligible
                    && match &checkpoint {
                        None => true,
                        Some(path) => {
                            serde_json::from_slice::<crate::train::CheckpointManifest>(
                                &std::fs::read(path.join("manifest.json"))?,
                            )?
                            .training_purpose
                                == crate::train::TrainingPurpose::Candidate
                        }
                    };
                let smoke = model.generate(GenerationRequest {
                    prompt: "Write a Rust expression for adding two integers.".into(),
                    max_new_tokens: 1,
                    seed: 1701,
                    temperature: 0.,
                    top_p: 1.,
                    timeout_seconds: 1800,
                })?;
                let health = Health {
                    process_id: std::process::id(),
                    healthy: true,
                    checkpoint_id: model.checkpoint_id.clone(),
                    model_id: model.assets.model_id.clone(),
                    base_revision: model.assets.revision.clone(),
                    tokenizer_sha256: model
                        .assets
                        .files
                        .get("tokenizer.json")
                        .or_else(|| model.assets.files.get("source/tokenizer.json"))
                        .context("validated model is missing its original tokeniser hash")?
                        .clone(),
                    checksum_validated: true,
                    inference_smoke_output: smoke.completion,
                    smoke_elapsed_ms: smoke.elapsed_ms,
                    immutable: true,
                    production_eligible,
                    execution: model.execution_metadata(),
                };
                Ok((model, health))
            })();
            let (mut model, health) = match runtime {
                Ok(v) => v,
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            if ready_tx.send(Ok(health)).is_err() {
                return;
            }
            while let Some(job) = rx.blocking_recv() {
                if job.reply.is_closed()
                    || job.cancel.load(Ordering::Relaxed)
                    || Instant::now() >= job.expires
                {
                    continue;
                }
                let result = model
                    .generate_cancellable(job.request, job.cancel)
                    .map_err(|e| format!("{e:#}"));
                let _ = job.reply.send(result);
            }
        })?;
    let ready_health = ready_rx
        .await
        .context("model worker failed before health handshake")?
        .map_err(anyhow::Error::msg)?;
    let state = ServiceState {
        queue: tx,
        metadata: Arc::new(ready_health),
        config: config.clone(),
        requests: Arc::new(AtomicU64::new(0)),
        errors: Arc::new(AtomicU64::new(0)),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/metadata", get(metadata))
        .route("/v1/generate", post(generate))
        .layer(DefaultBodyLimit::max(config.maximum_request_bytes))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("serving: immutable model listening at http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VersionedEndpoint {
    pub url: String,
    pub checkpoint_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingManifest {
    pub schema_version: u32,
    pub generation: u64,
    pub incumbent: VersionedEndpoint,
    pub candidate: Option<VersionedEndpoint>,
    pub canary_percent: u8,
    pub gate_report_sha256: Option<String>,
    pub rollback_reason: Option<String>,
}

fn publish_route(
    path: &Path,
    expected_generation: Option<u64>,
    manifest: &RoutingManifest,
) -> Result<()> {
    let parent = path.parent().context("routing manifest parent missing")?;
    std::fs::create_dir_all(parent)?;
    let lock = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path.with_extension("lock"))?;
    use std::os::fd::AsRawFd;
    // SAFETY: flock operates on a live file descriptor, without accessing memory.
    ensure!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another routing update is in progress"
    );
    match expected_generation {
        None => ensure!(!path.exists(), "routing manifest was concurrently created"),
        Some(expected) => {
            let current: RoutingManifest = serde_json::from_slice(&std::fs::read(path)?)?;
            ensure!(
                current.generation == expected,
                "routing policy changed during staging; retry from the current policy"
            );
        }
    }
    atomic_write(path, &serde_json::to_vec_pretty(manifest)?)?;
    Ok(())
}

fn endpoint_url(endpoint: &VersionedEndpoint) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(&endpoint.url)?;
    ensure!(
        url.scheme() == "http"
            && url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "invalid local endpoint URL"
    );
    let host = url.host_str().context("endpoint host missing")?;
    // Literal loopback addresses prevent DNS rebinding or external redirects.
    ensure!(
        host.parse::<IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false),
        "routing requires a literal loopback address"
    );
    Ok(url)
}
fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(120))
        .build()?)
}
async fn check_health(endpoint: &VersionedEndpoint) -> Result<Health> {
    let url = endpoint_url(endpoint)?.join("health")?;
    let health: Health = client()?
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        health.healthy
            && health.checksum_validated
            && health.immutable
            && !health.inference_smoke_output.trim().is_empty()
            && health.checkpoint_id == endpoint.checkpoint_id,
        "immutable endpoint health or checkpoint identity mismatch"
    );
    Ok(health)
}

/// The integration gate writes checkpoint-bound evidence. Statistics fixtures cannot
/// authorise production traffic, even if their numerical decision happens to be SHIP.
pub async fn initialise_routing(
    path: &Path,
    incumbent: VersionedEndpoint,
) -> Result<RoutingManifest> {
    ensure!(!path.exists(), "routing manifest already exists");
    ensure!(
        check_health(&incumbent).await?.production_eligible,
        "a control checkpoint cannot become the production incumbent"
    );
    let manifest = RoutingManifest {
        schema_version: 1,
        generation: 0,
        incumbent,
        candidate: None,
        canary_percent: 0,
        gate_report_sha256: None,
        rollback_reason: None,
    };
    publish_route(path, None, &manifest)?;
    Ok(manifest)
}

/// Make a NO SHIP or control checkpoint available only via explicit evaluation
/// routing. Production still selects the incumbent for every request.
pub async fn stage_evaluation(
    path: &Path,
    candidate: VersionedEndpoint,
) -> Result<RoutingManifest> {
    let mut manifest: RoutingManifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected = manifest.generation;
    check_health(&candidate).await?;
    manifest.candidate = Some(candidate);
    manifest.canary_percent = 0;
    manifest.gate_report_sha256 = None;
    manifest.generation += 1;
    manifest.rollback_reason =
        Some("evaluation-only staging; production incumbent retained".into());
    publish_route(path, Some(expected), &manifest)?;
    Ok(manifest)
}
pub async fn stage_canary(
    path: &Path,
    candidate: VersionedEndpoint,
    incumbent_evaluation: &Path,
    candidate_evaluation: &Path,
    root: &Path,
    gate_config: &Path,
) -> Result<RoutingManifest> {
    let mut manifest: RoutingManifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected = manifest.generation;
    let (incumbent_path, candidate_path, root_path, config_path) = (
        incumbent_evaluation.to_owned(),
        candidate_evaluation.to_owned(),
        root.to_owned(),
        gate_config.to_owned(),
    );
    // Recompute the gate from complete verified evaluations. A caller-authored SHIP
    // aggregate, score fixture or renamed tensor cannot authorise production.
    let auth = tokio::task::spawn_blocking(move || {
        crate::evaluate::gate_files(&incumbent_path, &candidate_path, &root_path, &config_path)
    })
    .await??;
    ensure!(
        auth.report.decision == crate::gate::Decision::Ship
            && auth.report.evidence_kind == "verified_checkpoint_evaluation"
            && auth.incumbent_checkpoint_id == manifest.incumbent.checkpoint_id
            && auth.candidate_checkpoint_id == candidate.checkpoint_id,
        "NO SHIP, fixture, or unbound gate cannot authorise canary traffic"
    );
    check_health(&manifest.incumbent).await?;
    ensure!(
        check_health(&candidate).await?.production_eligible,
        "control-purpose adapters cannot receive production traffic"
    );
    ensure!(
        candidate.checkpoint_id != manifest.incumbent.checkpoint_id,
        "candidate and incumbent must be distinct immutable checkpoints"
    );
    let auth_bytes = serde_json::to_vec_pretty(&auth)?;
    let auth_hash = format!("{:x}", Sha256::digest(&auth_bytes));
    let auth_path = path.with_file_name(format!("canary-gate-{auth_hash}.json"));
    atomic_write(&auth_path, &auth_bytes)?;
    manifest.candidate = Some(candidate);
    manifest.canary_percent = 5;
    manifest.generation += 1;
    manifest.gate_report_sha256 = Some(sha256_file(&auth_path)?);
    manifest.rollback_reason = None;
    publish_route(path, Some(expected), &manifest)?;
    Ok(manifest)
}
pub fn rollback(path: &Path, reason: &str) -> Result<RoutingManifest> {
    ensure!(!reason.trim().is_empty(), "rollback reason is required");
    let mut manifest: RoutingManifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let expected = manifest.generation;
    manifest.generation += 1;
    manifest.canary_percent = 0;
    manifest.rollback_reason = Some(reason.into());
    // Keep candidate metadata for audit. New requests select the incumbent while
    // existing requests continue on their originally selected immutable process.
    publish_route(path, Some(expected), &manifest)?;
    Ok(manifest)
}

pub fn choose_endpoint(
    manifest: &RoutingManifest,
    request_id: &str,
    evaluation: bool,
) -> Result<VersionedEndpoint> {
    ensure!(
        manifest.schema_version == 1 && manifest.canary_percent <= 5,
        "invalid routing policy"
    );
    ensure!(
        manifest.canary_percent == 0
            || manifest
                .gate_report_sha256
                .as_ref()
                .is_some_and(|h| h.len() == 64 && h.bytes().all(|c| c.is_ascii_hexdigit())),
        "canary routing lacks gate provenance"
    );
    if evaluation {
        return manifest
            .candidate
            .clone()
            .context("no staged evaluation checkpoint");
    }
    let hash = Sha256::digest(request_id.as_bytes());
    let slot = u64::from_le_bytes(hash[..8].try_into().expect("SHA-256 prefix length")) % 100;
    if slot < manifest.canary_percent as u64 {
        return manifest
            .candidate
            .clone()
            .context("canary enabled without candidate");
    }
    Ok(manifest.incumbent.clone())
}

/// Snapshot the atomic routing manifest once per request. The caller controls a
/// stable request ID; an operational candidate failure rolls back new traffic.
pub async fn route_generate(
    path: &Path,
    request_id: &str,
    evaluation: bool,
    request: GenerationRequest,
) -> Result<Generation> {
    ensure!(
        !request_id.is_empty() && request_id.len() <= 256,
        "bounded routing request ID required"
    );
    let manifest: RoutingManifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let endpoint = choose_endpoint(&manifest, request_id, evaluation)?;
    let result = (|| async {
        let response = client()?
            .post(endpoint_url(&endpoint)?.join("v1/generate")?)
            .json(&request)
            .send()
            .await?
            .error_for_status()?;
        let generated: Generation = response.json().await?;
        ensure!(
            generated.checkpoint_id == endpoint.checkpoint_id,
            "response checkpoint identity mismatch"
        );
        ensure!(
            !generated.completion.trim().is_empty(),
            "empty inference response"
        );
        ensure!(
            generated.complete,
            "incomplete candidate response ({})",
            generated.finish_reason
        );
        Ok::<_, anyhow::Error>(generated)
    })()
    .await;
    if result.is_err() && !evaluation && endpoint.checkpoint_id != manifest.incumbent.checkpoint_id
    {
        rollback(
            path,
            "candidate operational failure: predeclared one-error rollback threshold",
        )?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn untested_device_or_precision_cannot_promote_an_assessment_adapter() {
        assert!(assessment_execution_eligible(
            true,
            &RuntimeOptions::default()
        ));
        assert!(!assessment_execution_eligible(
            true,
            &RuntimeOptions::cuda(0)
        ));
        assert!(!assessment_execution_eligible(
            true,
            &RuntimeOptions {
                device: crate::model::ExecutionDevice::Cuda { ordinal: 0 },
                dtype: crate::model::InferenceDtype::F32
            }
        ));
        assert!(assessment_execution_eligible(
            false,
            &RuntimeOptions::cuda(0)
        ));
    }
    #[test]
    fn rollback_retains_incumbent_and_candidate_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routing.json");
        let original = RoutingManifest {
            schema_version: 1,
            generation: 3,
            incumbent: VersionedEndpoint {
                url: "http://127.0.0.1:8080/".into(),
                checkpoint_id: "base".into(),
            },
            candidate: Some(VersionedEndpoint {
                url: "http://127.0.0.1:8081/".into(),
                checkpoint_id: "candidate".into(),
            }),
            canary_percent: 5,
            gate_report_sha256: Some("gate".into()),
            rollback_reason: None,
        };
        atomic_write(&path, &serde_json::to_vec(&original).unwrap()).unwrap();
        let rolled = rollback(&path, "injected operational error threshold").unwrap();
        assert_eq!(rolled.generation, 4);
        assert!(rolled.candidate.is_some());
        for i in 0..1000 {
            assert_eq!(
                choose_endpoint(&rolled, &i.to_string(), false)
                    .unwrap()
                    .checkpoint_id,
                "base"
            );
        }
    }
    #[test]
    fn endpoints_are_strictly_local() {
        for url in [
            "https://127.0.0.1:1",
            "http://example.org",
            "http://user:secret@127.0.0.1/",
        ] {
            assert!(
                endpoint_url(&VersionedEndpoint {
                    url: url.into(),
                    checkpoint_id: "x".into()
                })
                .is_err()
            );
        }
    }
}
