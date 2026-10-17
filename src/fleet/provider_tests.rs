//! Labelled protocol/failure fixtures; no external API is contacted.
use super::tests::{fixture_runner, provider};
use super::*;

fn request() -> FleetRequest {
    FleetRequest {
        task_id: "fixture-task".into(),
        source_id: "fixture-source".into(),
        specification_sha256: "b".repeat(64),
        system_prompt: "Return text.".into(),
        prompt: "Say hello.".into(),
        sample_slot: 0,
        generation: GenerationParameters {
            temperature: 1.0,
            top_p: 1.0,
            seed: None,
            max_output_tokens: 128,
        },
    }
}
fn native_provider(api: ProviderApi) -> ProviderConfig {
    let mut p = provider("native-fixture");
    p.api = api;
    p.inference_only = true;
    p.permission.approved_for_output_training = false;
    p.output_cap_parameter = match api {
        ProviderApi::OpenaiResponses => "max_output_tokens",
        ProviderApi::GeminiGenerateContent | ProviderApi::VertexGenerateContent => {
            "maxOutputTokens"
        }
        _ => "max_tokens",
    }
    .into();
    p.endpoint = match api {
        ProviderApi::OpenaiResponses => "https://fixture.invalid/v1/responses".into(),
        ProviderApi::AnthropicMessages => "https://fixture.invalid/v1/messages".into(),
        ProviderApi::GeminiGenerateContent | ProviderApi::VertexGenerateContent => format!(
            "https://fixture.invalid/v1/models/{}:generateContent",
            p.model
        ),
        _ => p.endpoint,
    };
    p
}
fn response(api: ProviderApi) -> TransportResponse {
    let body = match api {
        ProviderApi::OpenaiResponses => json!({"object":"response","status":"completed","output":[
            {"type":"reasoning","summary":[]}, {"type":"message","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":"hello"}]}],
            "usage":{"input_tokens":50,"output_tokens":20,"output_tokens_details":{"reasoning_tokens":5}}}),
        ProviderApi::AnthropicMessages => {
            json!({"type":"message","role":"assistant","stop_reason":"end_turn",
            "content":[{"type":"text","text":"hello"}],
            "usage":{"input_tokens":40,"cache_read_input_tokens":10,"output_tokens":20}})
        }
        ProviderApi::GeminiGenerateContent | ProviderApi::VertexGenerateContent => json!({
            "candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"hello"}]}}],
            "usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":20}}),
        _ => json!({"choices":[{"finish_reason":"stop","message":{"content":"hello"}}],
            "usage":{"prompt_tokens":50,"completion_tokens":20}}),
    };
    TransportResponse {
        status: 200,
        headers: BTreeMap::new(),
        raw_body: body.to_string(),
        elapsed_millis: 0,
    }
}

#[tokio::test]
async fn native_protocols_keep_credentials_out_of_identity_and_replay_without_keys() {
    for api in [
        ProviderApi::OpenaiResponses,
        ProviderApi::GeminiGenerateContent,
        ProviderApi::VertexGenerateContent,
        ProviderApi::AnthropicMessages,
    ] {
        let p = native_provider(api);
        let mut prepared = prepare(&p, &request()).unwrap();
        prepared.attempt_ordinal = 1;
        let wire = LiveTransport::new(true)
            .unwrap()
            .authenticated_request(&prepared, "LABELLED_FIXTURE_SECRET")
            .unwrap()
            .json(&prepared.body)
            .build()
            .unwrap();
        assert!(wire.url().query().is_none());
        assert!(
            !prepared
                .body
                .to_string()
                .contains("LABELLED_FIXTURE_SECRET")
        );
        let header = match api {
            ProviderApi::AnthropicMessages => "x-api-key",
            ProviderApi::GeminiGenerateContent => "x-goog-api-key",
            _ => "authorization",
        };
        assert!(wire.headers()[header].is_sensitive());
        if api == ProviderApi::AnthropicMessages {
            assert_eq!(wire.headers()["anthropic-version"], "2023-06-01");
        }
        let record =
            Recording::from_response(&prepared, response(api), "synthetic_fixture").unwrap();
        record.validate_for(&prepared).unwrap();
        let (text, usage) = parse_recording(&record).unwrap();
        assert_eq!(text, "hello");
        assert_eq!((usage.input_tokens, usage.output_tokens), (50, 20));
        assert!(!record.permission.approved_for_output_training);
        let bytes = serde_json::to_vec_pretty(&record).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("LABELLED_FIXTURE_SECRET"));
        let dir = tempfile::tempdir().unwrap();
        let capture =
            Ledger::open(dir.path().join("capture.db"), dir.path().join("recordings")).unwrap();
        capture.publish_artifact(&bytes).unwrap();
        let replay = Arc::new(
            ReplayTransport::load_with_policy(dir.path().join("recordings"), true).unwrap(),
        );
        let runner = fixture_runner(
            &dir.path().join("replay"),
            vec![p],
            replay.clone(),
            100_000_000,
        );
        let first = runner.complete("fixture-round", request()).await.unwrap();
        assert!(!first.real_provider_recording && !first.training_permission_approved);
        assert!(
            runner
                .complete("fixture-round", request())
                .await
                .unwrap()
                .cached
        );
        assert_eq!(replay.call_count(), 1);
        let mut forged = record;
        forged.api = ProviderApi::ChatCompletions;
        assert!(forged.validate_identity().is_err());
    }
}

#[test]
fn native_adversarial_responses_reject_truncation_tools_refusals_and_bad_usage() {
    for (api, pointer, value) in [
        (ProviderApi::OpenaiResponses, "/status", json!("incomplete")),
        (
            ProviderApi::OpenaiResponses,
            "/output/1/content/0/type",
            json!("refusal"),
        ),
        (
            ProviderApi::OpenaiResponses,
            "/output/1/type",
            json!("function_call"),
        ),
        (
            ProviderApi::OpenaiResponses,
            "/usage/output_tokens_details/reasoning_tokens",
            json!(21),
        ),
        (
            ProviderApi::OpenaiResponses,
            "/usage/output_tokens",
            Value::Null,
        ),
        (
            ProviderApi::AnthropicMessages,
            "/stop_reason",
            json!("max_tokens"),
        ),
        (
            ProviderApi::AnthropicMessages,
            "/content/0/type",
            json!("tool_use"),
        ),
        (
            ProviderApi::AnthropicMessages,
            "/usage/cache_read_input_tokens",
            json!(-1),
        ),
        (
            ProviderApi::GeminiGenerateContent,
            "/candidates/0/finishReason",
            json!("SAFETY"),
        ),
        (
            ProviderApi::GeminiGenerateContent,
            "/candidates/0/content/parts/0/text",
            json!(""),
        ),
        (
            ProviderApi::GeminiGenerateContent,
            "/usageMetadata/candidatesTokenCount",
            Value::Null,
        ),
    ] {
        let mut r = response(api);
        let mut body: Value = serde_json::from_str(&r.raw_body).unwrap();
        *body.pointer_mut(pointer).unwrap() = value;
        r.raw_body = body.to_string();
        assert!(
            parse_response(&r).is_err(),
            "accepted {api:?} mutation {pointer}"
        );
    }
    let mut r = response(ProviderApi::GeminiGenerateContent);
    let mut body: Value = serde_json::from_str(&r.raw_body).unwrap();
    body["usageMetadata"]["thoughtsTokenCount"] = json!(7);
    r.raw_body = body.to_string();
    assert_eq!(extract_usage(&r).unwrap().output_tokens, 27);
    assert_eq!(extract_usage(&r).unwrap().reasoning_tokens, 7);
}

struct FailureFixture {
    calls: Mutex<Vec<String>>,
    status: u16,
    before_dispatch: bool,
}
#[async_trait]
impl Transport for FailureFixture {
    fn is_replay(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        p: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        self.calls.lock().await.push(p.provider.id.clone());
        if p.provider.id == "primary" && self.before_dispatch {
            return Err(TransportFailure::BeforeDispatch(
                "LABELLED connection failure".into(),
            ));
        }
        let mut r = response(ProviderApi::ChatCompletions);
        if p.provider.id == "primary" {
            r.status = self.status;
            if self.status == 503 {
                r.headers.insert("retry-after".into(), "70".into());
            }
        }
        Ok(Recording::from_response(p, r, "synthetic_fixture").unwrap())
    }
}

#[tokio::test]
async fn unavailable_service_honours_long_retry_after_across_requests() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Arc::new(FailureFixture {
        calls: Mutex::new(vec![]),
        status: 503,
        before_dispatch: false,
    });
    let mut primary = provider("primary");
    primary.retries = 1;
    let runner = fixture_runner(
        dir.path(),
        vec![primary, provider("fallback")],
        fixture.clone(),
        100_000_000,
    );
    assert_eq!(
        runner
            .complete("fixture-round", request())
            .await
            .unwrap()
            .provider_id,
        "fallback"
    );
    assert!(
        runner.runtimes["primary"]
            .circuit
            .lock()
            .await
            .1
            .is_some_and(|until| until > tokio::time::Instant::now() + Duration::from_secs(60))
    );
    let mut next = request();
    next.sample_slot = 1;
    assert_eq!(
        runner
            .complete("fixture-round", next)
            .await
            .unwrap()
            .provider_id,
        "fallback"
    );
    assert_eq!(
        fixture
            .calls
            .lock()
            .await
            .iter()
            .filter(|id| *id == "primary")
            .count(),
        1
    );
}

struct ReplyFixture {
    reply: TransportResponse,
    calls: AtomicUsize,
}
#[async_trait]
impl Transport for ReplyFixture {
    fn is_replay(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        p: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(Recording::from_response(p, self.reply.clone(), "synthetic_fixture").unwrap())
    }
}

#[tokio::test]
async fn malformed_billing_and_unexpected_google_thinking_block_further_purchases() {
    for api in [
        ProviderApi::ChatCompletions,
        ProviderApi::GeminiGenerateContent,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut reply = response(api);
        let mut body: Value = serde_json::from_str(&reply.raw_body).unwrap();
        if api == ProviderApi::ChatCompletions {
            body["usage"]["prompt_tokens"] = json!("malformed");
        } else {
            body["usageMetadata"]["thoughtsTokenCount"] = json!(7);
        }
        reply.raw_body = body.to_string();
        let fixture = Arc::new(ReplyFixture {
            reply,
            calls: AtomicUsize::new(0),
        });
        let runner = fixture_runner(
            dir.path(),
            vec![native_provider(api)],
            fixture.clone(),
            100_000_000,
        );
        assert!(
            runner
                .complete("fixture-round", request())
                .await
                .unwrap_err()
                .to_string()
                .contains("round blocked")
        );
        let mut next = request();
        next.sample_slot = 1;
        assert!(runner.complete("fixture-round", next).await.is_err());
        assert_eq!(fixture.calls.load(Ordering::Relaxed), 1);
        assert_eq!(runner.ledger.costs(None).unwrap()[0].state, "failed");
    }
}

#[tokio::test]
async fn absent_credentials_fail_over_without_dispatch_or_reservation() {
    let dir = tempfile::tempdir().unwrap();
    let mut first = provider("primary");
    first.api_key_environment = format!("LEC_ABSENT_{}", uuid::Uuid::new_v4().simple());
    let mut second = provider("fallback");
    second.api_key_environment = format!("LEC_ABSENT_{}", uuid::Uuid::new_v4().simple());
    let runner = fixture_runner(
        dir.path(),
        vec![first, second],
        Arc::new(LiveTransport::new(true).unwrap()),
        100_000_000,
    );
    let error = runner
        .complete("fixture-round", request())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("fleet exhausted"));
    assert!(runner.ledger.costs(None).unwrap().is_empty());
    assert_eq!(
        runner
            .ledger
            .events(None)
            .unwrap()
            .iter()
            .filter(|e| e.kind == "provider_unavailable")
            .count(),
        2
    );
}

#[tokio::test]
async fn refused_connections_and_permanent_auth_errors_use_configured_fallback() {
    for (status, before_dispatch) in [(200, true), (401, false), (403, false), (404, false)] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = Arc::new(FailureFixture {
            calls: Mutex::new(vec![]),
            status,
            before_dispatch,
        });
        let runner = fixture_runner(
            dir.path(),
            vec![provider("primary"), provider("fallback")],
            fixture.clone(),
            100_000_000,
        );
        assert_eq!(
            runner
                .complete("fixture-round", request())
                .await
                .unwrap()
                .provider_id,
            "fallback"
        );
        let costs = runner.ledger.costs(None).unwrap();
        assert_eq!(costs.len(), 2);
        assert_eq!(costs[0].state, "failed");
        if before_dispatch {
            assert_eq!(costs[0].token_upper_bound, Some(0));
        }
        if !before_dispatch {
            let mut next = request();
            next.sample_slot = 1;
            runner.complete("fixture-round", next).await.unwrap();
            assert_eq!(
                fixture
                    .calls
                    .lock()
                    .await
                    .iter()
                    .filter(|id| *id == "primary")
                    .count(),
                1
            );
        }
    }
}

#[tokio::test]
async fn cumulative_token_stop_prevents_dispatch_even_with_money_remaining() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Arc::new(FailureFixture {
        calls: Mutex::new(vec![]),
        status: 200,
        before_dispatch: false,
    });
    let mut runner = fixture_runner(
        dir.path(),
        vec![provider("primary")],
        fixture.clone(),
        100_000_000,
    );
    runner.config.budgets.tokens.round = Some(1);
    assert!(
        runner
            .complete("fixture-round", request())
            .await
            .unwrap_err()
            .to_string()
            .contains("TOKEN BUDGET STOP")
    );
    assert!(fixture.calls.lock().await.is_empty());
    assert!(runner.ledger.costs(None).unwrap().is_empty());
}

#[test]
fn inference_configuration_does_not_clear_training_and_disabled_template_is_valid() {
    let config = FleetConfig::load("config/api-providers.toml").unwrap();
    assert_eq!(config.providers.len(), 4);
    assert!(config.require_three_businesses().is_err());
    let mut p = native_provider(ProviderApi::OpenaiResponses);
    p.validate_enabled().unwrap();
    p.inference_only = false;
    assert!(p.validate_enabled().is_err());
    assert!(LiveTransport::new(false).is_err());
}

#[tokio::test]
async fn retries_count_failed_usage_towards_the_token_budget() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Arc::new(FailureFixture {
        calls: Mutex::new(vec![]),
        status: 500,
        before_dispatch: false,
    });
    let mut p = provider("primary");
    p.retries = 1;
    let prepared = prepare(&p, &request()).unwrap();
    let reserved = prepared.input_token_upper_bound + request().generation.max_output_tokens;
    let mut runner = fixture_runner(dir.path(), vec![p], fixture.clone(), 100_000_000);
    runner.config.budgets.tokens.round = Some(reserved + 69);
    let error = runner
        .complete("fixture-round", request())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("TOKEN BUDGET STOP"));
    assert_eq!(fixture.calls.lock().await.len(), 1);
    let costs = runner.ledger.costs(None).unwrap();
    assert_eq!(costs.len(), 1);
    assert_eq!(costs[0].state, "failed");
    assert_eq!(costs[0].token_upper_bound, Some(70));
}

#[test]
fn native_durable_publish_window_recovers_without_another_purchase() {
    for api in [
        ProviderApi::OpenaiResponses,
        ProviderApi::GeminiGenerateContent,
        ProviderApi::VertexGenerateContent,
        ProviderApi::AnthropicMessages,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let ledger =
            Ledger::open(dir.path().join("ledger.db"), dir.path().join("artifacts")).unwrap();
        let p = native_provider(api);
        let mut prepared = prepare(&p, &request()).unwrap();
        let acquisition = Acquisition {
            round_id: "recovery-fixture".into(),
            task_id: prepared.request.task_id.clone(),
            source_id: prepared.request.source_id.clone(),
            request_key: prepared.request_key.clone(),
            provider_id: p.id.clone(),
            business_id: p.business_id.clone(),
            model: p.model.clone(),
            model_revision: p.model_revision.clone(),
            request_json: String::from_utf8(canonical_json(&prepared.body).unwrap()).unwrap(),
            price_json: serde_json::to_string(p.price.as_ref().unwrap()).unwrap(),
            reserved_nanos: prepared.reservation_nanos,
            reserved_tokens: Some(prepared.input_token_upper_bound + 128),
            replay: true,
        };
        let limits = BudgetLimits {
            round_nanos: 100_000_000,
            task_nanos: 100_000_000,
            provider_nanos: 100_000_000,
            model_nanos: 100_000_000,
            tokens: Default::default(),
        };
        let Claim::Reserved(claim) = ledger.claim_and_reserve(&acquisition, limits).unwrap() else {
            panic!()
        };
        prepared.attempt_ordinal = claim.ordinal;
        let record =
            Recording::from_response(&prepared, response(api), "synthetic_fixture").unwrap();
        ledger
            .publish_artifact(&serde_json::to_vec_pretty(&record).unwrap())
            .unwrap();
        assert_eq!(ledger.recover_interrupted().unwrap(), 0);
        assert!(matches!(
            ledger.lookup_request(&prepared.request_key).unwrap(),
            Some(Claim::Cached { .. })
        ));
        assert_eq!(ledger.costs(None).unwrap()[0].token_upper_bound, Some(70));
    }
}
