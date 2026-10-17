//! Bounded orchestration, retries, failover and durable reservation settlement.
use super::*;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetCompletion {
    pub task_id: String,
    pub source_id: String,
    pub specification_sha256: String,
    pub request_key: String,
    pub attempt_id: String,
    pub provider_id: String,
    pub business_id: String,
    pub model: String,
    pub model_revision: String,
    pub response_artifact_sha256: String,
    pub code: String,
    pub usage: TokenUsage,
    pub historical_cost_nanos: i64,
    pub training_permission_decision_id: String,
    pub training_permission_approved: bool,
    pub real_provider_recording: bool,
    pub replayed: bool,
    pub cached: bool,
}

pub(super) struct ProviderRuntime {
    semaphore: Arc<Semaphore>,
    window: Mutex<VecDeque<(tokio::time::Instant, u64)>>,
    pub(super) circuit: Mutex<(u32, Option<tokio::time::Instant>)>,
    unavailable_until_reconciled: AtomicBool,
}
impl ProviderRuntime {
    fn new(provider: &ProviderConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(provider.concurrency.max(1))),
            window: Mutex::new(VecDeque::new()),
            circuit: Mutex::new((0, None)),
            unavailable_until_reconciled: AtomicBool::new(false),
        }
    }
    async fn rate_limit(&self, provider: &ProviderConfig, tokens: u64) -> Result<()> {
        ensure!(
            tokens <= provider.tokens_per_minute,
            "single request exceeds configured TPM allowance"
        );
        loop {
            let mut window = self.window.lock().await;
            let now = tokio::time::Instant::now();
            while window
                .front()
                .is_some_and(|(time, _)| now.duration_since(*time) >= Duration::from_secs(60))
            {
                window.pop_front();
            }
            let used = window
                .iter()
                .try_fold(0u64, |sum, (_, n)| sum.checked_add(*n))
                .context("rate token overflow")?;
            if window.len() < provider.requests_per_minute
                && used
                    .checked_add(tokens)
                    .is_some_and(|n| n <= provider.tokens_per_minute)
            {
                window.push_back((now, tokens));
                return Ok(());
            }
            let until =
                window.front().context("empty over-limit rate window")?.0 + Duration::from_secs(60);
            drop(window);
            tokio::time::sleep_until(until).await;
        }
    }
    async fn available(&self) -> bool {
        if self.unavailable_until_reconciled.load(Ordering::Relaxed) {
            return false;
        }
        let mut state = self.circuit.lock().await;
        if state
            .1
            .is_some_and(|until| until <= tokio::time::Instant::now())
        {
            *state = (0, None);
        }
        state.1.is_none()
    }
    async fn defer(&self, provider: &ProviderConfig, delay: Duration) {
        if let Some(until) = tokio::time::Instant::now().checked_add(delay) {
            *self.circuit.lock().await = (provider.circuit_breaker_failures, Some(until));
        } else {
            self.unavailable_until_reconciled
                .store(true, Ordering::Relaxed);
        }
    }
    async fn outcome(&self, provider: &ProviderConfig, success: bool) {
        let mut state = self.circuit.lock().await;
        if success {
            *state = (0, None);
        } else {
            state.0 = state.0.saturating_add(1);
            if state.0 >= provider.circuit_breaker_failures {
                state.1 = Some(
                    tokio::time::Instant::now()
                        + Duration::from_secs(provider.circuit_breaker_seconds),
                );
            }
        }
    }
}

pub struct FleetRunner {
    pub config: FleetConfig,
    pub ledger: Ledger,
    transport: Arc<dyn Transport>,
    pub(super) runtimes: HashMap<String, Arc<ProviderRuntime>>,
}
impl FleetRunner {
    pub fn new(config: FleetConfig, ledger: Ledger, transport: Arc<dyn Transport>) -> Result<Self> {
        config.validate()?;
        let runtimes = config
            .providers
            .iter()
            .map(|p| (p.id.clone(), Arc::new(ProviderRuntime::new(p))))
            .collect();
        Ok(Self {
            config,
            ledger,
            transport,
            runtimes,
        })
    }

    pub async fn complete(&self, round: &str, request: FleetRequest) -> Result<FleetCompletion> {
        let first = self
            .config
            .providers
            .iter()
            .find(|p| p.enabled)
            .context("no enabled, priced, rights-cleared provider")?;
        self.complete_on(round, &first.id, request).await
    }

    pub async fn run_many(
        &self,
        round: &str,
        requests: Vec<FleetRequest>,
    ) -> Vec<Result<FleetCompletion>> {
        stream::iter(
            requests
                .into_iter()
                .map(|request| self.complete(round, request)),
        )
        .buffer_unordered(self.config.max_parallel_tasks)
        .collect()
        .await
    }

    pub async fn complete_on(
        &self,
        round: &str,
        preferred: &str,
        request: FleetRequest,
    ) -> Result<FleetCompletion> {
        let first = self
            .config
            .providers
            .iter()
            .find(|p| p.id == preferred)
            .context("preferred provider is absent")?;
        let mut order = vec![first.id.as_str()];
        for id in &first.failover {
            if !order.contains(&id.as_str()) {
                order.push(id);
            }
        }
        let mut failures = Vec::new();
        for id in order {
            let provider = self
                .config
                .providers
                .iter()
                .find(|p| p.id == id)
                .context("failover provider absent")?;
            if !provider.enabled {
                continue;
            }
            let mut prepared = match prepare(provider, &request) {
                Ok(prepared) => prepared,
                Err(error) => {
                    let detail = format!("provider={id}; request unsupported: {error}");
                    self.log(round, None, "provider_request_refused", &detail)
                        .await?;
                    failures.push(detail);
                    continue;
                }
            };
            let ledger = self.ledger.clone();
            let key = prepared.request_key.clone();
            match tokio::task::spawn_blocking(move || ledger.lookup_request(&key)).await?? {
                Some(Claim::Cached {
                    attempt_id,
                    artifact_sha256,
                }) => {
                    return self.cached(&prepared, attempt_id, artifact_sha256).await;
                }
                Some(Claim::Unknown) => bail!(
                    "request {} has UNKNOWN remote acceptance; reconciliation required before another purchase",
                    prepared.request_key
                ),
                Some(Claim::InFlight) => bail!(
                    "request {} is already in flight; concurrent duplicate was not purchased",
                    prepared.request_key
                ),
                _ => {}
            }
            let runtime = self.runtimes.get(id).context("provider runtime absent")?;
            if !runtime.available().await {
                self.log(round, None, "circuit_open", &format!("provider={id}"))
                    .await?;
                continue;
            }
            // No secret value crosses into the request identity, recording or ledger.
            if let Err(error) = self.transport.preflight(&prepared) {
                let detail = format!("provider={id}; preflight refused: {error}");
                self.log(
                    round,
                    Some(&prepared.request_key),
                    "provider_unavailable",
                    &detail,
                )
                .await?;
                failures.push(detail);
                continue;
            }
            for retry in 0..=provider.retries {
                let _permit = runtime.semaphore.acquire().await?;
                if !runtime.available().await {
                    break;
                }
                let tokens = prepared
                    .input_token_upper_bound
                    .checked_add(request.generation.max_output_tokens)
                    .context("rate token overflow")?;
                if tokens > provider.tokens_per_minute {
                    let detail =
                        format!("provider={id}; single request exceeds configured TPM allowance");
                    self.log(
                        round,
                        Some(&prepared.request_key),
                        "provider_request_refused",
                        &detail,
                    )
                    .await?;
                    failures.push(detail);
                    break;
                }
                runtime.rate_limit(provider, tokens).await?;
                // A sibling may have received Retry-After while this worker waited
                // for the rate window. Do not dispatch using stale circuit state.
                if !runtime.available().await {
                    break;
                }
                let acquisition = Acquisition {
                    round_id: round.into(),
                    task_id: request.task_id.clone(),
                    source_id: request.source_id.clone(),
                    request_key: prepared.request_key.clone(),
                    provider_id: provider.id.clone(),
                    business_id: provider.business_id.clone(),
                    model: provider.model.clone(),
                    model_revision: provider.model_revision.clone(),
                    request_json: String::from_utf8(canonical_json(&prepared.body)?)?,
                    price_json: serde_json::to_string(
                        provider.price.as_ref().context("missing price")?,
                    )?,
                    reserved_nanos: prepared.reservation_nanos,
                    reserved_tokens: Some(tokens),
                    replay: self.transport.is_replay(),
                };
                let ledger = self.ledger.clone();
                let limits = self.config.budgets;
                let claim = tokio::task::spawn_blocking(move || {
                    ledger.claim_and_reserve(&acquisition, limits)
                })
                .await??;
                let attempt = match claim {
                    Claim::Cached {
                        attempt_id,
                        artifact_sha256,
                    } => {
                        return self.cached(&prepared, attempt_id, artifact_sha256).await;
                    }
                    Claim::Reserved(attempt) => attempt,
                    Claim::InFlight => bail!(
                        "request {} is already in flight; concurrent duplicate was not purchased",
                        prepared.request_key
                    ),
                    Claim::Unknown => bail!(
                        "request {} has UNKNOWN remote acceptance; reconciliation required before another purchase",
                        prepared.request_key
                    ),
                    Claim::BudgetStopped {
                        scope,
                        used_nanos,
                        requested_nanos,
                        limit_nanos,
                    } => bail!(
                        "BUDGET STOP {scope}: {used_nanos} + {requested_nanos} > {limit_nanos}; progress is durable"
                    ),
                    Claim::TokenBudgetStopped {
                        scope,
                        used,
                        requested,
                        limit,
                    } => bail!(
                        "TOKEN BUDGET STOP {scope}: {used} + {requested} > {limit}; progress is durable"
                    ),
                };
                prepared.attempt_ordinal = attempt.ordinal;
                self.log(
                    round,
                    Some(&prepared.request_key),
                    "dispatch",
                    &format!(
                        "provider={id}; attempt={}; retry={retry}; replay={}",
                        attempt.attempt_id,
                        self.transport.is_replay()
                    ),
                )
                .await?;
                let recording = match self.transport.execute(&prepared).await {
                    Ok(recording) => recording,
                    Err(TransportFailure::Unknown(reason)) => {
                        let ledger = self.ledger.clone();
                        let attempt_id = attempt.attempt_id.clone();
                        let detail = reason.clone();
                        tokio::task::spawn_blocking(move || {
                            ledger.mark_unknown(&attempt_id, &detail)
                        })
                        .await??;
                        bail!(
                            "UNKNOWN: {reason}; reservation retained; automatic retry prohibited"
                        );
                    }
                    Err(TransportFailure::BeforeDispatch(reason)) => {
                        let ledger = self.ledger.clone();
                        let attempt_id = attempt.attempt_id.clone();
                        let detail = reason.clone();
                        tokio::task::spawn_blocking(move || {
                            ledger.finish(
                                &attempt_id,
                                None,
                                false,
                                0,
                                0,
                                Some((0, 0, 0, 0)),
                                Some(&detail),
                            )
                        })
                        .await??;
                        self.log(
                            round,
                            Some(&prepared.request_key),
                            "connection_unavailable",
                            &format!("provider={id}; {reason}"),
                        )
                        .await?;
                        failures.push(format!("provider={id}; {reason}"));
                        runtime.outcome(provider, false).await;
                        break;
                    }
                    Err(error) => {
                        let ledger = self.ledger.clone();
                        let attempt_id = attempt.attempt_id.clone();
                        let detail = error.to_string();
                        tokio::task::spawn_blocking(move || {
                            ledger.finish(&attempt_id, None, false, 0, 0, None, Some(&detail))
                        })
                        .await??;
                        return Err(error.into());
                    }
                };
                recording.validate_for(&prepared)?;
                let bytes = serde_json::to_vec_pretty(&recording)?;
                let ledger = self.ledger.clone();
                let hash =
                    tokio::task::spawn_blocking(move || ledger.publish_artifact(&bytes)).await??;
                let mut parsed = parse_recording(&recording);
                let usage = extract_usage(&recording.response).ok();
                let invalid_usage = usage.is_none()
                    && serde_json::from_str::<Value>(&recording.response.raw_body).is_ok_and(
                        |body| body.get("usage").is_some() || body.get("usageMetadata").is_some(),
                    );
                let exceeded_bounds = usage.as_ref().is_some_and(|usage| {
                    usage.input_tokens > prepared.input_token_upper_bound
                        || usage.output_tokens > request.generation.max_output_tokens
                        || (matches!(
                            provider.api,
                            ProviderApi::GeminiGenerateContent | ProviderApi::VertexGenerateContent
                        ) && usage.reasoning_tokens != 0)
                });
                if exceeded_bounds || invalid_usage {
                    parsed = Err(anyhow::anyhow!(
                        "reported usage is invalid or exceeded approved token bounds"
                    ));
                }
                let cost = usage
                    .as_ref()
                    .map(|usage| recording.price.cost(usage))
                    .transpose()?;
                let lower = cost.unwrap_or(0);
                let upper = cost.unwrap_or(prepared.reservation_nanos);
                let (success, reason) = match &parsed {
                    Ok(_) => (true, None),
                    Err(error) => (false, Some(error.to_string())),
                };
                let ledger = self.ledger.clone();
                let attempt_id = attempt.attempt_id.clone();
                let artifact = hash.clone();
                let settled_usage = usage.as_ref().map(|u| {
                    (
                        u.input_tokens,
                        u.cached_input_tokens,
                        u.output_tokens,
                        u.reasoning_tokens,
                    )
                });
                tokio::task::spawn_blocking(move || {
                    ledger.finish(
                        &attempt_id,
                        Some(&artifact),
                        success,
                        lower,
                        upper,
                        settled_usage,
                        reason.as_deref(),
                    )
                })
                .await??;
                if exceeded_bounds || invalid_usage {
                    let ledger = self.ledger.clone();
                    let round = round.to_owned();
                    tokio::task::spawn_blocking(move || {
                        ledger.block_round(
                            &round,
                            "reported usage invalid or exceeded approved token bounds",
                        )
                    })
                    .await??;
                    bail!(
                        "usage invalid or token bounds exceeded; round blocked for reconciliation"
                    );
                }
                if let Ok((code, usage)) = parsed {
                    ensure!(
                        usage.input_tokens <= prepared.input_token_upper_bound
                            && usage.output_tokens <= request.generation.max_output_tokens,
                        "reported tokens exceeded approved bounds; stop for reconciliation"
                    );
                    runtime.outcome(provider, true).await;
                    return make_completion(
                        &prepared,
                        &recording,
                        CompletionEvidence {
                            attempt_id: attempt.attempt_id,
                            artifact_sha256: hash,
                            code,
                            usage,
                            replayed: self.transport.is_replay(),
                            cached: false,
                        },
                    );
                }
                let status = recording.response.status;
                let detail = format!(
                    "provider={id}; HTTP={status}; reason={}",
                    parsed.err().map(|e| e.to_string()).unwrap_or_default()
                );
                self.log(
                    round,
                    Some(&prepared.request_key),
                    "failed_attempt",
                    &detail,
                )
                .await?;
                failures.push(detail);
                let permanent = matches!(status, 400 | 401 | 402 | 403 | 404 | 405 | 410 | 422)
                    || serde_json::from_str::<Value>(&recording.response.raw_body)
                        .ok()
                        .and_then(|v| {
                            v.pointer("/error/code")
                                .and_then(Value::as_str)
                                .map(str::to_owned)
                        })
                        .is_some_and(|c| {
                            matches!(
                                c.as_str(),
                                "insufficient_quota" | "billing_hard_limit_reached"
                            )
                        });
                if permanent {
                    runtime
                        .unavailable_until_reconciled
                        .store(true, Ordering::Relaxed);
                }
                let transient = !permanent && (status == 429 || status == 408 || status >= 500);
                runtime.outcome(provider, false).await;
                if transient
                    && (status == 429 || recording.response.headers.contains_key("retry-after"))
                {
                    let delay = retry_delay(&recording.response, retry, &prepared.request_key);
                    runtime.defer(provider, delay).await;
                }
                if !transient || retry == provider.retries {
                    break;
                }
                let delay = retry_delay(&recording.response, retry, &prepared.request_key);
                self.log(
                    round,
                    Some(&prepared.request_key),
                    "retry_wait",
                    &format!("provider={id}; milliseconds={}", delay.as_millis()),
                )
                .await?;
                if delay > Duration::from_secs(60) {
                    self.log(
                        round,
                        Some(&prepared.request_key),
                        "retry_after_failover",
                        "provider requested a delay beyond the bounded retry wait",
                    )
                    .await?;
                    break;
                }
                drop(_permit);
                tokio::time::sleep(delay).await;
            }
            self.log(
                round,
                None,
                "failover",
                &format!("leaving_provider={id}; requested_provider={preferred}"),
            )
            .await?;
        }
        bail!(
            "fleet exhausted without a valid completion: {}",
            failures.join("; ")
        )
    }

    async fn log(&self, round: &str, key: Option<&str>, kind: &str, detail: &str) -> Result<()> {
        let ledger = self.ledger.clone();
        let round = round.to_owned();
        let key = key.map(str::to_owned);
        let kind = kind.to_owned();
        let detail = detail.to_owned();
        tokio::task::spawn_blocking(move || ledger.event(&round, key.as_deref(), &kind, &detail))
            .await?
    }
    async fn cached(
        &self,
        prepared: &PreparedRequest,
        attempt_id: String,
        hash: String,
    ) -> Result<FleetCompletion> {
        let ledger = self.ledger.clone();
        let artifact = hash.clone();
        let bytes = tokio::task::spawn_blocking(move || ledger.read_artifact(&artifact)).await??;
        let recording: Recording = serde_json::from_slice(&bytes)?;
        let mut matching = prepared.clone();
        matching.attempt_ordinal = recording.attempt_ordinal;
        recording.validate_for(&matching)?;
        let (code, usage) = parse_recording(&recording)?;
        make_completion(
            prepared,
            &recording,
            CompletionEvidence {
                attempt_id,
                artifact_sha256: hash,
                code,
                usage,
                replayed: self.transport.is_replay(),
                cached: true,
            },
        )
    }
}

struct CompletionEvidence {
    attempt_id: String,
    artifact_sha256: String,
    code: String,
    usage: TokenUsage,
    replayed: bool,
    cached: bool,
}
fn make_completion(
    prepared: &PreparedRequest,
    record: &Recording,
    evidence: CompletionEvidence,
) -> Result<FleetCompletion> {
    let request = &prepared.request;
    Ok(FleetCompletion {
        task_id: request.task_id.clone(),
        source_id: request.source_id.clone(),
        specification_sha256: request.specification_sha256.clone(),
        request_key: prepared.request_key.clone(),
        attempt_id: evidence.attempt_id,
        provider_id: record.provider_id.clone(),
        business_id: record.business_id.clone(),
        model: record.model.clone(),
        model_revision: record.model_revision.clone(),
        response_artifact_sha256: evidence.artifact_sha256,
        code: evidence.code,
        historical_cost_nanos: record.price.cost(&evidence.usage)?,
        usage: evidence.usage,
        training_permission_decision_id: record.permission.decision_id.clone(),
        training_permission_approved: record.permission.approved_for_output_training
            && prepared.provider.permission.approved_for_output_training
            && !prepared.provider.inference_only,
        real_provider_recording: record.origin == "real_provider",
        replayed: evidence.replayed,
        cached: evidence.cached,
    })
}

fn retry_delay(response: &TransportResponse, retry: u32, key: &str) -> Duration {
    let server = response
        .headers
        .get("retry-after")
        .and_then(|value| {
            value
                .parse::<u64>()
                .ok()
                .map(Duration::from_secs)
                .or_else(|| {
                    httpdate::parse_http_date(value)
                        .ok()
                        .map(|date| date.duration_since(SystemTime::now()).unwrap_or_default())
                })
        })
        .unwrap_or_default();
    let jitter = u64::from_str_radix(&key[..key.len().min(4)], 16).unwrap_or(0) % 101;
    let backoff = Duration::from_millis(250u64.saturating_mul(1u64 << retry.min(7)) + jitter);
    // Never shorten a provider's Retry-After. Excessive waits fail over rather
    // than retrying early; the caller's bounded retry count limits total work.
    server.max(backoff)
}
