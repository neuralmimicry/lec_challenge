//! Invariant tests use labelled fixtures; they are not acquisition evidence.
use super::*;
pub(super) fn provider(id: &str) -> ProviderConfig {
    ProviderConfig {
        id: id.into(),
        business_id: format!("{id}-business"),
        enabled: true,
        endpoint: format!("https://{id}.invalid/chat/completions"),
        api_key_environment: "UNUSED_TEST_KEY".into(),
        model: "fixture-model".into(),
        model_revision: "fixture-revision".into(),
        timeout_seconds: 5,
        requests_per_minute: 100,
        tokens_per_minute: 1_000_000,
        concurrency: 4,
        max_input_tokens: 8192,
        max_output_tokens: 2048,
        output_cap_parameter: "max_tokens".into(),
        output_cap_includes_reasoning: true,
        max_response_bytes: 1_000_000,
        retries: 0,
        circuit_breaker_failures: 5,
        circuit_breaker_seconds: 1,
        failover: vec![],
        permission: PermissionDecision {
            approved_for_output_training: true,
            decision_id: "TEST FIXTURE ONLY".into(),
            reviewed_at: "test".into(),
            evidence_sha256: "a".repeat(64),
            model_licence: "TEST FIXTURE ONLY".into(),
        },
        price: Some(PriceSnapshot {
            snapshot_id: "TEST ASSUMPTION".into(),
            source_url: "https://invalid.example".into(),
            effective_date: "test".into(),
            input_nanos_per_million: 1_000_000_000,
            cached_input_nanos_per_million: 500_000_000,
            output_nanos_per_million: 2_000_000_000,
        }),
    }
}
fn request() -> FleetRequest {
    FleetRequest {
        task_id: "task".into(),
        source_id: "source".into(),
        specification_sha256: "b".repeat(64),
        system_prompt: "Return a Rust function.".into(),
        prompt: "Add one.".into(),
        sample_slot: 0,
        generation: GenerationParameters::default(),
    }
}
fn response() -> TransportResponse {
    TransportResponse{status:200,headers:BTreeMap::new(),raw_body:json!({"choices":[{"finish_reason":"stop","message":{"content":"```rust\nfn solve(x:i64)->i64{x+1}\n```"}}],"usage":{"prompt_tokens":50,"completion_tokens":20}}).to_string(),elapsed_millis:1}
}
struct FixtureTransport {
    calls: AtomicUsize,
    unknown: bool,
    statuses: Mutex<VecDeque<u16>>,
}
#[async_trait]
impl Transport for FixtureTransport {
    fn is_replay(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        prepared: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.unknown {
            return Err(TransportFailure::Unknown(
                "injected acceptance-before-persistence window".into(),
            ));
        }
        let mut response = response();
        if let Some(status) = self.statuses.lock().await.pop_front() {
            response.status = status;
        }
        Ok(Recording::from_response(prepared, response, "synthetic_fixture").unwrap())
    }
}
fn fixture_runner(
    dir: &Path,
    mut providers: Vec<ProviderConfig>,
    transport: Arc<dyn Transport>,
    budget: i64,
) -> FleetRunner {
    if providers.len() > 1 {
        providers[0].failover = vec![providers[1].id.clone()];
    }
    FleetRunner::new(
        FleetConfig {
            max_parallel_tasks: 4,
            budgets: BudgetLimits {
                round_nanos: budget,
                task_nanos: budget,
                provider_nanos: budget,
                model_nanos: budget,
            },
            providers,
        },
        Ledger::open(dir.join("ledger.db"), dir.join("artifacts")).unwrap(),
        transport,
    )
    .unwrap()
}
#[test]
fn canonical_key_keeps_sample_slot_and_full_prompt() {
    let provider = provider("a");
    let a = request();
    let first = prepare(&provider, &a).unwrap();
    let mut b = a.clone();
    b.sample_slot = 1;
    assert_ne!(
        first.request_key,
        prepare(&provider, &b).unwrap().request_key
    );
    b = a;
    b.prompt.push('!');
    assert_ne!(
        first.request_key,
        prepare(&provider, &b).unwrap().request_key
    );
    let mut new_price = provider.clone();
    new_price.price.as_mut().unwrap().output_nanos_per_million += 1;
    assert_eq!(
        first.request_key,
        prepare(&new_price, &request()).unwrap().request_key
    );
}
#[test]
fn integer_pricing_rounds_up_and_counts_cached_input_once() {
    assert_eq!(PriceSnapshot::charge(1, 1).unwrap(), 1);
    let price = provider("a").price.unwrap();
    assert_eq!(
        price
            .cost(&TokenUsage {
                input_tokens: 100,
                cached_input_tokens: 20,
                output_tokens: 10,
                reasoning_tokens: 5
            })
            .unwrap(),
        110_000
    );
}
#[test]
fn empty_truncated_missing_usage_are_failures() {
    let mut response = response();
    response.raw_body=json!({"choices":[{"finish_reason":"stop","message":{"content":""}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}).to_string();
    assert!(parse_response(&response).is_err());
    response.raw_body=json!({"choices":[{"finish_reason":"length","message":{"content":"code"}}],"usage":{"prompt_tokens":1,"completion_tokens":1}}).to_string();
    assert!(parse_response(&response).is_err());
    response.raw_body =
        json!({"choices":[{"finish_reason":"stop","message":{"content":"code"}}]}).to_string();
    assert!(parse_response(&response).is_err());
}
#[tokio::test]
async fn restart_cache_does_not_execute_transport_again() {
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(FixtureTransport {
        calls: AtomicUsize::new(0),
        unknown: false,
        statuses: Mutex::new(VecDeque::new()),
    });
    let runner = fixture_runner(
        dir.path(),
        vec![provider("a")],
        transport.clone(),
        100_000_000,
    );
    let first = runner.complete("round", request()).await.unwrap();
    assert!(!first.real_provider_recording);
    drop(runner);
    let restarted = fixture_runner(
        dir.path(),
        vec![provider("a")],
        transport.clone(),
        100_000_000,
    );
    assert!(restarted.complete("round", request()).await.unwrap().cached);
    assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
}
#[tokio::test]
async fn unknown_acceptance_never_retries_or_fails_over() {
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(FixtureTransport {
        calls: AtomicUsize::new(0),
        unknown: true,
        statuses: Mutex::new(VecDeque::new()),
    });
    let runner = fixture_runner(
        dir.path(),
        vec![provider("a"), provider("b")],
        transport.clone(),
        100_000_000,
    );
    assert!(runner.complete("round", request()).await.is_err());
    assert!(runner.complete("round", request()).await.is_err());
    assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
    assert_eq!(runner.ledger.costs(None).unwrap()[0].state, "unknown");
}
#[tokio::test]
async fn budget_stop_precedes_transport() {
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(FixtureTransport {
        calls: AtomicUsize::new(0),
        unknown: false,
        statuses: Mutex::new(VecDeque::new()),
    });
    let runner = fixture_runner(dir.path(), vec![provider("a")], transport.clone(), 1);
    assert!(
        runner
            .complete("round", request())
            .await
            .unwrap_err()
            .to_string()
            .contains("BUDGET STOP")
    );
    assert_eq!(transport.calls.load(Ordering::Relaxed), 0);
    assert!(runner.ledger.costs(None).unwrap().is_empty());
}
#[tokio::test]
async fn provider_failure_is_attributed_and_fails_over() {
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(FixtureTransport {
        calls: AtomicUsize::new(0),
        unknown: false,
        statuses: Mutex::new(VecDeque::from([500])),
    });
    let runner = fixture_runner(
        dir.path(),
        vec![provider("a"), provider("b")],
        transport.clone(),
        100_000_000,
    );
    let completion = runner.complete("round", request()).await.unwrap();
    assert_eq!(completion.provider_id, "b");
    assert_eq!(runner.ledger.costs(None).unwrap().len(), 2);
    assert_eq!(transport.calls.load(Ordering::Relaxed), 2);
}
#[test]
fn offline_empty_directory_never_invents_recordings() {
    let dir = tempfile::tempdir().unwrap();
    assert!(ReplayTransport::load(dir.path()).is_err());
}

struct RefusingTransport;
#[async_trait]
impl Transport for RefusingTransport {
    fn is_replay(&self) -> bool {
        true
    }
    fn preflight(&self, _: &PreparedRequest) -> Result<()> {
        bail!("credentials absent; cache must be checked first")
    }
    async fn execute(
        &self,
        _: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        panic!("cached work reached transport")
    }
}
#[tokio::test]
async fn cache_precedes_credentials_and_open_circuit() {
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(FixtureTransport {
        calls: AtomicUsize::new(0),
        unknown: false,
        statuses: Mutex::new(VecDeque::new()),
    });
    let runner = fixture_runner(dir.path(), vec![provider("a")], transport, 100_000_000);
    runner.complete("round", request()).await.unwrap();
    let resumed = fixture_runner(
        dir.path(),
        vec![provider("a")],
        Arc::new(RefusingTransport),
        100_000_000,
    );
    *resumed.runtimes["a"].circuit.lock().await = (
        99,
        Some(tokio::time::Instant::now() + Duration::from_secs(3600)),
    );
    assert!(resumed.complete("round", request()).await.unwrap().cached);
}
#[tokio::test]
async fn exact_replay_rejects_missing_sample_and_corrupt_hash() {
    let dir = tempfile::tempdir().unwrap();
    let recordings = dir.path().join("recordings");
    let capture = Ledger::open(dir.path().join("capture.db"), &recordings).unwrap();
    let mut prepared = prepare(&provider("a"), &request()).unwrap();
    prepared.attempt_ordinal = 1;
    let recording = Recording::from_response(&prepared, response(), "synthetic_fixture").unwrap();
    let hash = capture
        .publish_artifact(&serde_json::to_vec_pretty(&recording).unwrap())
        .unwrap();
    assert!(ReplayTransport::load(&recordings).is_err());
    let transport = Arc::new(ReplayTransport::load_with_policy(&recordings, true).unwrap());
    let runner = fixture_runner(
        &dir.path().join("replay"),
        vec![provider("a")],
        transport.clone(),
        100_000_000,
    );
    assert!(
        !runner
            .complete("round", request())
            .await
            .unwrap()
            .real_provider_recording
    );
    let mut other = request();
    other.sample_slot = 1;
    assert!(
        runner
            .complete("round", other)
            .await
            .unwrap_err()
            .to_string()
            .contains("recording missing")
    );
    assert_eq!(transport.call_count(), 1);
    fs::write(recordings.join(format!("{hash}.json")), b"tampered").unwrap();
    assert!(ReplayTransport::load_with_policy(&recordings, true).is_err());
}
