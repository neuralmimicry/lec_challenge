//! Sanitised live HTTP acquisition and exact immutable offline replay.
use super::*;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    /// Exact UTF-8 response bytes as text; never an invented normalised API reply.
    pub raw_body: String,
    pub elapsed_millis: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    pub schema_version: u32,
    pub origin: String,
    pub acquired_unix_millis: i64,
    pub request_key: String,
    pub attempt_ordinal: u32,
    pub provider_id: String,
    pub business_id: String,
    pub model: String,
    pub model_revision: String,
    pub endpoint: String,
    pub output_cap_parameter: String,
    pub request: FleetRequest,
    pub request_body: Value,
    pub price: PriceSnapshot,
    pub permission: PermissionDecision,
    pub response: TransportResponse,
}

impl Recording {
    pub(super) fn from_response(
        prepared: &PreparedRequest,
        response: TransportResponse,
        origin: &str,
    ) -> Result<Self> {
        Ok(Self {
            schema_version: 2,
            origin: origin.to_owned(),
            acquired_unix_millis: unix_millis(),
            request_key: prepared.request_key.clone(),
            attempt_ordinal: prepared.attempt_ordinal,
            provider_id: prepared.provider.id.clone(),
            business_id: prepared.provider.business_id.clone(),
            model: prepared.provider.model.clone(),
            model_revision: prepared.provider.model_revision.clone(),
            endpoint: prepared.provider.endpoint.clone(),
            output_cap_parameter: prepared.provider.output_cap_parameter.clone(),
            request: prepared.request.clone(),
            request_body: prepared.body.clone(),
            price: prepared.provider.price.clone().context("missing price")?,
            permission: prepared.provider.permission.clone(),
            response,
        })
    }
    pub fn validate_identity(&self) -> Result<()> {
        ensure!(self.schema_version == 2, "unsupported recording version");
        ensure!(
            matches!(
                self.origin.as_str(),
                "real_provider" | "synthetic_fixture" | "deterministic_fault_injection"
            ),
            "unknown evidence origin"
        );
        ensure!(
            self.attempt_ordinal > 0 && self.acquired_unix_millis > 0,
            "recording attempt/time missing"
        );
        ensure!(
            matches!(
                self.output_cap_parameter.as_str(),
                "max_tokens" | "max_completion_tokens"
            ),
            "unknown billable output cap"
        );
        ensure!(
            canonical_json(&self.request_body)?
                == canonical_json(&chat_body(
                    &self.model,
                    &self.output_cap_parameter,
                    &self.request
                ))?,
            "recorded prompt/parameters do not match canonical request body"
        );
        ensure!(
            self.request_key
                == request_identity(
                    &self.request,
                    &self.provider_id,
                    &self.business_id,
                    &self.endpoint,
                    &self.model_revision,
                    &self.request_body
                )?,
            "recording canonical request hash mismatch"
        );
        self.price.validate()
    }
    pub fn validate_for(&self, request: &PreparedRequest) -> Result<()> {
        self.validate_identity()?;
        ensure!(
            self.origin == "real_provider"
                || self.origin == "synthetic_fixture"
                || self.origin == "deterministic_fault_injection",
            "unknown evidence origin"
        );
        ensure!(
            self.request_key == request.request_key
                && self.attempt_ordinal == request.attempt_ordinal,
            "exact recording key/ordinal mismatch"
        );
        ensure!(
            self.provider_id == request.provider.id
                && self.business_id == request.provider.business_id
                && self.model == request.provider.model
                && self.model_revision == request.provider.model_revision
                && self.endpoint == request.provider.endpoint
                && self.output_cap_parameter == request.provider.output_cap_parameter,
            "recording provider identity mismatch"
        );
        ensure!(
            canonical_json(&self.request_body)? == canonical_json(&request.body)?,
            "recording request body mismatch"
        );
        self.price.validate()?;
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransportFailure {
    #[error("remote acceptance/billing unknown: {0}")]
    Unknown(String),
    #[error("local transport refused before dispatch: {0}")]
    BeforeDispatch(String),
    #[error("offline recording missing: {0}")]
    MissingRecording(String),
}

#[async_trait]
pub trait Transport: Send + Sync {
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure>;
    fn is_replay(&self) -> bool;
    /// Validate credentials or recording availability before reserving or dispatching.
    fn preflight(&self, _request: &PreparedRequest) -> Result<()> {
        Ok(())
    }
}

pub struct LiveTransport {
    client: reqwest::Client,
}
impl LiveTransport {
    pub fn new(spending_authorised: bool) -> Result<Self> {
        ensure!(
            spending_authorised,
            "live calls require explicit spending authority"
        );
        Ok(Self {
            client: reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
        })
    }
}

#[async_trait]
impl Transport for LiveTransport {
    fn is_replay(&self) -> bool {
        false
    }
    fn preflight(&self, request: &PreparedRequest) -> Result<()> {
        request.provider.validate_enabled()?;
        ensure!(
            std::env::var(&request.provider.api_key_environment)
                .is_ok_and(|v| !v.trim().is_empty()),
            "credential {} is absent (value redacted)",
            request.provider.api_key_environment
        );
        Ok(())
    }
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        let key = std::env::var(&request.provider.api_key_environment).map_err(|_| {
            TransportFailure::BeforeDispatch("credential absent; value redacted".into())
        })?;
        if key.trim().is_empty() {
            return Err(TransportFailure::BeforeDispatch(
                "empty credential; value redacted".into(),
            ));
        }
        let start = std::time::Instant::now();
        // Transport errors may occur after remote acceptance. Their display text can
        // contain URLs, so ledger diagnostics deliberately use a fixed redacted reason.
        let response = self
            .client
            .post(&request.provider.endpoint)
            .bearer_auth(key)
            .timeout(Duration::from_secs(request.provider.timeout_seconds))
            .json(&request.body)
            .send()
            .await
            .map_err(|_| {
                TransportFailure::Unknown(
                    "HTTP send failed or timed out; reconcile with provider".into(),
                )
            })?;
        let status = response.status().as_u16();
        let mut headers = BTreeMap::new();
        for name in [
            "x-request-id",
            "request-id",
            "retry-after",
            "date",
            "content-type",
        ] {
            if let Some(value) = response.headers().get(name).and_then(|v| v.to_str().ok()) {
                headers.insert(name.to_owned(), value.to_owned());
            }
        }
        let mut bytes = Vec::new();
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = chunk.map_err(|_| {
                TransportFailure::Unknown(
                    "response stream interrupted after remote acceptance".into(),
                )
            })?;
            if bytes.len().saturating_add(chunk.len()) > request.provider.max_response_bytes {
                return Err(TransportFailure::Unknown(
                    "response exceeded bounded recording size after acceptance".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let raw_body = String::from_utf8(bytes).map_err(|_| {
            TransportFailure::Unknown("non-UTF-8 response cannot be durably interpreted".into())
        })?;
        Recording::from_response(
            request,
            TransportResponse {
                status,
                headers,
                raw_body,
                elapsed_millis: start.elapsed().as_millis().min(u64::MAX as u128) as u64,
            },
            "real_provider",
        )
        .map_err(|_| TransportFailure::Unknown("cannot construct response record".into()))
    }
}

/// Immutable in-memory index. No network client exists in this transport.
pub struct ReplayTransport {
    recordings: HashMap<(String, u32), Recording>,
    calls: AtomicUsize,
    allow_fixtures: bool,
}
impl ReplayTransport {
    pub fn load(directory: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_policy(directory, false)
    }
    pub fn load_with_policy(
        directory: impl AsRef<Path>,
        allow_labelled_fixtures: bool,
    ) -> Result<Self> {
        let mut recordings = HashMap::new();
        for entry in fs::read_dir(directory)
            .context("recording directory missing; offline replay cannot continue")?
        {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let bytes = fs::read(&path)?;
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .context("invalid recording filename")?;
            ensure!(
                valid_hash(stem) && sha256(&bytes) == stem,
                "recording filename/checksum mismatch: {}",
                path.display()
            );
            let record: Recording = serde_json::from_slice(&bytes)
                .with_context(|| format!("invalid recording {}", path.display()))?;
            record.validate_identity()?;
            ensure!(
                record.origin == "real_provider" || allow_labelled_fixtures,
                "fixture recording is prohibited in evidence replay"
            );
            ensure!(
                recordings
                    .insert((record.request_key.clone(), record.attempt_ordinal), record)
                    .is_none(),
                "duplicate recording key/ordinal"
            );
        }
        ensure!(
            !recordings.is_empty(),
            "no recordings; offline replay is incomplete"
        );
        Ok(Self {
            recordings,
            calls: AtomicUsize::new(0),
            allow_fixtures: allow_labelled_fixtures,
        })
    }
    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
    pub fn len(&self) -> usize {
        self.recordings.len()
    }
    pub fn is_empty(&self) -> bool {
        self.recordings.is_empty()
    }
}
#[async_trait]
impl Transport for ReplayTransport {
    fn is_replay(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        let record = self
            .recordings
            .get(&(request.request_key.clone(), request.attempt_ordinal))
            .ok_or_else(|| {
                TransportFailure::MissingRecording(format!(
                    "{} attempt {}",
                    request.request_key, request.attempt_ordinal
                ))
            })?;
        record
            .validate_for(request)
            .map_err(|e| TransportFailure::BeforeDispatch(e.to_string()))?;
        if !self.allow_fixtures && record.origin != "real_provider" {
            return Err(TransportFailure::BeforeDispatch(
                "non-real recording rejected".into(),
            ));
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(record.clone())
    }
}

/// Explicit deterministic wrapper for drill fault injection; never real evidence.
pub struct FaultInjectionTransport {
    inner: Arc<dyn Transport>,
    provider_id: String,
    statuses: Mutex<VecDeque<u16>>,
}
impl FaultInjectionTransport {
    pub fn new(inner: Arc<dyn Transport>, provider_id: String, statuses: Vec<u16>) -> Self {
        Self {
            inner,
            provider_id,
            statuses: Mutex::new(statuses.into()),
        }
    }
}
#[async_trait]
impl Transport for FaultInjectionTransport {
    fn is_replay(&self) -> bool {
        self.inner.is_replay()
    }
    fn preflight(&self, request: &PreparedRequest) -> Result<()> {
        self.inner.preflight(request)
    }
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        let status = if request.provider.id == self.provider_id {
            self.statuses.lock().await.pop_front()
        } else {
            None
        };
        if let Some(status) = status {
            let mut headers = BTreeMap::new();
            if status == 429 {
                headers.insert("retry-after".into(), "0".into());
            }
            return Recording::from_response(request,TransportResponse{status,headers,raw_body:json!({"error":{"message":"DETERMINISTIC DRILL INJECTION; not a provider response"}}).to_string(),elapsed_millis:0},"deterministic_fault_injection").map_err(|e|TransportFailure::BeforeDispatch(e.to_string()));
        }
        self.inner.execute(request).await
    }
}
