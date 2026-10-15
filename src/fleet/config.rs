//! Prices, rights, request identity and immutable roster configuration.
use super::*;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceSnapshot {
    pub snapshot_id: String,
    pub source_url: String,
    pub effective_date: String,
    /// Nanodollars per million tokens preserve sub-nanodollar token prices.
    pub input_nanos_per_million: i64,
    pub cached_input_nanos_per_million: i64,
    pub output_nanos_per_million: i64,
}

impl PriceSnapshot {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.snapshot_id.is_empty()
                && !self.source_url.is_empty()
                && !self.effective_date.is_empty(),
            "incomplete price evidence"
        );
        ensure!(
            [
                self.input_nanos_per_million,
                self.cached_input_nanos_per_million,
                self.output_nanos_per_million
            ]
            .iter()
            .all(|x| *x >= 0),
            "negative token price"
        );
        Ok(())
    }
    pub(super) fn charge(tokens: u64, price: i64) -> Result<i64> {
        ensure!(price >= 0, "negative price");
        let product = i128::from(tokens)
            .checked_mul(i128::from(price))
            .context("token price overflow")?;
        i64::try_from((product + 999_999) / 1_000_000).context("charge exceeds integer money range")
    }
    pub fn cost(&self, usage: &TokenUsage) -> Result<i64> {
        ensure!(
            usage.cached_input_tokens <= usage.input_tokens,
            "cached input exceeds total input"
        );
        ensure!(
            usage.reasoning_tokens <= usage.output_tokens,
            "reasoning exceeds billable output"
        );
        let uncached = Self::charge(
            usage.input_tokens - usage.cached_input_tokens,
            self.input_nanos_per_million,
        )?;
        let cached = Self::charge(
            usage.cached_input_tokens,
            self.cached_input_nanos_per_million,
        )?;
        let output = Self::charge(usage.output_tokens, self.output_nanos_per_million)?;
        uncached
            .checked_add(cached)
            .and_then(|x| x.checked_add(output))
            .context("total charge overflow")
    }
    pub fn reserve(&self, input: u64, output: u64) -> Result<i64> {
        let input_price = self
            .input_nanos_per_million
            .max(self.cached_input_nanos_per_million);
        Self::charge(input, input_price)?
            .checked_add(Self::charge(output, self.output_nanos_per_million)?)
            .context("reservation overflow")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionDecision {
    pub approved_for_output_training: bool,
    pub decision_id: String,
    pub reviewed_at: String,
    pub evidence_sha256: String,
    pub model_licence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: String,
    pub business_id: String,
    pub enabled: bool,
    pub endpoint: String,
    pub api_key_environment: String,
    pub model: String,
    pub model_revision: String,
    pub timeout_seconds: u64,
    pub requests_per_minute: usize,
    pub tokens_per_minute: u64,
    pub concurrency: usize,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
    pub output_cap_parameter: String,
    pub output_cap_includes_reasoning: bool,
    pub max_response_bytes: usize,
    pub retries: u32,
    pub circuit_breaker_failures: u32,
    pub circuit_breaker_seconds: u64,
    #[serde(default)]
    pub failover: Vec<String>,
    pub permission: PermissionDecision,
    pub price: Option<PriceSnapshot>,
}

impl ProviderConfig {
    pub fn validate_enabled(&self) -> Result<()> {
        ensure!(self.enabled, "provider {} is disabled", self.id);
        ensure!(
            !self.id.is_empty()
                && !self.business_id.is_empty()
                && !self.model.is_empty()
                && !self.model_revision.is_empty(),
            "provider/model identity is incomplete"
        );
        let url = reqwest::Url::parse(&self.endpoint)?;
        ensure!(
            url.scheme() == "https"
                && url.username().is_empty()
                && url.password().is_none()
                && url.query().is_none(),
            "provider endpoint must be HTTPS without credentials or query"
        );
        ensure!(
            !self.api_key_environment.is_empty(),
            "API key environment name missing"
        );
        ensure!(
            self.timeout_seconds > 0
                && self.requests_per_minute > 0
                && self.tokens_per_minute > 0
                && self.concurrency > 0,
            "invalid provider limits"
        );
        ensure!(
            self.max_input_tokens > 0 && self.max_output_tokens > 0 && self.max_response_bytes > 0,
            "uncapped request"
        );
        ensure!(
            matches!(
                self.output_cap_parameter.as_str(),
                "max_tokens" | "max_completion_tokens"
            ) && self.output_cap_includes_reasoning,
            "billable output is not demonstrably capped"
        );
        ensure!(
            self.circuit_breaker_failures > 0 && self.circuit_breaker_seconds > 0,
            "invalid circuit breaker"
        );
        ensure!(
            self.circuit_breaker_seconds <= 3600
                && self.timeout_seconds <= 600
                && self.retries <= 5,
            "unbounded retry/circuit/timeout configuration"
        );
        ensure!(
            self.permission.approved_for_output_training,
            "output training permission unresolved for {}",
            self.id
        );
        ensure!(
            !self.permission.decision_id.is_empty()
                && !self.permission.reviewed_at.is_empty()
                && !self.permission.model_licence.is_empty(),
            "incomplete permission decision"
        );
        ensure!(
            valid_hash(&self.permission.evidence_sha256),
            "permission evidence hash missing"
        );
        self.price
            .as_ref()
            .context("missing price snapshot")?
            .validate()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetConfig {
    pub max_parallel_tasks: usize,
    pub budgets: BudgetLimits,
    pub providers: Vec<ProviderConfig>,
}

impl FleetConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.max_parallel_tasks > 0, "zero fleet concurrency");
        ensure!(
            [
                self.budgets.round_nanos,
                self.budgets.task_nanos,
                self.budgets.provider_nanos,
                self.budgets.model_nanos
            ]
            .iter()
            .all(|x| *x >= 0),
            "negative budget"
        );
        let mut ids = std::collections::HashSet::new();
        for provider in &self.providers {
            ensure!(ids.insert(provider.id.clone()), "duplicate provider ID");
            if provider.enabled {
                provider.validate_enabled()?;
            }
        }
        for provider in &self.providers {
            for fallback in &provider.failover {
                ensure!(
                    ids.contains(fallback),
                    "unknown failover provider {fallback}"
                );
            }
        }
        Ok(())
    }
    pub fn require_three_businesses(&self) -> Result<()> {
        let businesses: std::collections::HashSet<_> = self
            .providers
            .iter()
            .filter(|p| p.enabled)
            .map(|p| p.business_id.as_str())
            .collect();
        ensure!(
            businesses.len() >= 3,
            "three independently owned, enabled and rights-cleared providers are required; found {}",
            businesses.len()
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParameters {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    pub max_output_tokens: u64,
}

impl Default for GenerationParameters {
    fn default() -> Self {
        Self {
            temperature: 0.2,
            top_p: 1.0,
            seed: Some(17),
            max_output_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetRequest {
    pub task_id: String,
    pub source_id: String,
    pub specification_sha256: String,
    pub system_prompt: String,
    pub prompt: String,
    pub sample_slot: u32,
    pub generation: GenerationParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedRequest {
    pub request_key: String,
    pub request: FleetRequest,
    pub provider: ProviderConfig,
    pub body: Value,
    pub input_token_upper_bound: u64,
    pub reservation_nanos: i64,
    pub attempt_ordinal: u32,
}

pub fn prepare(provider: &ProviderConfig, request: &FleetRequest) -> Result<PreparedRequest> {
    provider.validate_enabled()?;
    ensure!(
        !request.task_id.is_empty()
            && !request.source_id.is_empty()
            && valid_hash(&request.specification_sha256),
        "task provenance missing"
    );
    ensure!(!request.prompt.trim().is_empty(), "empty prompt");
    let generation = &request.generation;
    ensure!(
        generation.temperature.is_finite() && (0.0..=2.0).contains(&generation.temperature),
        "invalid temperature"
    );
    ensure!(
        generation.top_p.is_finite()
            && (0.0..=1.0).contains(&generation.top_p)
            && generation.top_p > 0.0,
        "invalid top_p"
    );
    ensure!(
        generation.max_output_tokens > 0
            && generation.max_output_tokens <= provider.max_output_tokens,
        "output exceeds approved billable cap"
    );
    let body = chat_body(&provider.model, &provider.output_cap_parameter, request);
    // Byte fallback deliberately over-reserves where an exact provider tokenizer is
    // unavailable. The fixed overhead covers ordinary chat framing, not arbitrary tools.
    let input_bound = u64::try_from(serde_json::to_vec(&body)?.len())?
        .checked_add(1024)
        .context("input bound overflow")?;
    ensure!(
        input_bound <= provider.max_input_tokens,
        "conservative input bound exceeds provider input cap"
    );
    let key = request_identity(
        request,
        &provider.id,
        &provider.business_id,
        &provider.endpoint,
        &provider.model_revision,
        &body,
    )?;
    let reservation = provider
        .price
        .as_ref()
        .context("missing price")?
        .reserve(input_bound, generation.max_output_tokens)?;
    Ok(PreparedRequest {
        request_key: key,
        request: request.clone(),
        provider: provider.clone(),
        body,
        input_token_upper_bound: input_bound,
        reservation_nanos: reservation,
        attempt_ordinal: 0,
    })
}

pub(super) fn chat_body(model: &str, cap_parameter: &str, request: &FleetRequest) -> Value {
    let generation = &request.generation;
    let mut body = json!({"model":model,"messages":[{"role":"system","content":request.system_prompt},{"role":"user","content":request.prompt}],"temperature":generation.temperature,"top_p":generation.top_p,"stream":false});
    body[cap_parameter] = json!(generation.max_output_tokens);
    if let Some(seed) = generation.seed {
        body["seed"] = json!(seed);
    }
    body
}

pub(super) fn request_identity(
    request: &FleetRequest,
    provider: &str,
    business: &str,
    endpoint: &str,
    revision: &str,
    body: &Value,
) -> Result<String> {
    let identity = json!({"schema":1,"task_id":request.task_id,"source_id":request.source_id,"specification_sha256":request.specification_sha256,"sample_slot":request.sample_slot,"provider_id":provider,"business_id":business,"endpoint":endpoint,"model_revision":revision,"body":body});
    Ok(sha256(&canonical_json(&identity)?))
}

pub fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    fn ordered(value: &Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), ordered(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.iter().map(ordered).collect()),
            other => other.clone(),
        }
    }
    Ok(serde_json::to_vec(&ordered(value))?)
}

pub fn build_round_plan(
    round: &str,
    config: &FleetConfig,
    jobs: &[(String, FleetRequest)],
) -> Result<crate::ledger::RoundPlan> {
    config.validate()?;
    let mut planned = Vec::new();
    for (preferred, request) in jobs {
        let primary = config
            .providers
            .iter()
            .find(|p| p.id == *preferred)
            .context("planned preferred provider missing")?;
        primary.validate_enabled()?;
        let mut providers = vec![preferred.as_str()];
        for fallback in &primary.failover {
            if !providers.contains(&fallback.as_str()) {
                providers.push(fallback);
            }
        }
        let mut keys = Vec::new();
        for id in providers {
            let provider = config
                .providers
                .iter()
                .find(|p| p.id == id)
                .context("planned failover provider missing")?;
            if provider.enabled {
                keys.push(prepare(provider, request)?.request_key);
            }
        }
        planned.push(crate::ledger::PlannedCompletion {
            task_id: request.task_id.clone(),
            source_id: request.source_id.clone(),
            specification_sha256: request.specification_sha256.clone(),
            sample_slot: request.sample_slot,
            preferred_provider_id: preferred.clone(),
            allowed_request_keys: keys,
        });
    }
    let plan = crate::ledger::RoundPlan {
        schema_version: 1,
        round_id: round.into(),
        configuration_sha256: sha256(&serde_json::to_vec(config)?),
        jobs: planned,
    };
    plan.validate()?;
    Ok(plan)
}
