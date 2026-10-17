//! Labelled local fixtures prove recovery mechanics, never provider acquisition.
use super::*;
use crate::fleet::{
    FleetRequest, GenerationParameters, PermissionDecision, PriceSnapshot, ProviderConfig,
    TransportResponse, prepare,
};
use serde_json::json;

struct Fixture {
    directory: tempfile::TempDir,
    ledger: Ledger,
    acquisition: Acquisition,
    claim: AttemptClaim,
    record: Recording,
    limits: BudgetLimits,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(
            directory.path().join("ledger.db"),
            directory.path().join("artifacts"),
        )
        .unwrap();
        let provider = ProviderConfig {
            api: Default::default(),
            inference_only: false,
            id: "fixture-provider".into(),
            business_id: "fixture-business".into(),
            enabled: true,
            endpoint: "https://fixture.invalid/chat/completions".into(),
            api_key_environment: "UNUSED_FIXTURE_KEY".into(),
            model: "fixture-model".into(),
            model_revision: "fixture-revision".into(),
            timeout_seconds: 1,
            requests_per_minute: 1,
            tokens_per_minute: 10_000,
            concurrency: 1,
            max_input_tokens: 10_000,
            max_output_tokens: 100,
            output_cap_parameter: "max_tokens".into(),
            output_cap_includes_reasoning: true,
            max_response_bytes: 10_000,
            retries: 0,
            circuit_breaker_failures: 1,
            circuit_breaker_seconds: 1,
            failover: vec![],
            permission: PermissionDecision {
                approved_for_output_training: true,
                decision_id: "fixture-only-permission".into(),
                reviewed_at: "fixture-date".into(),
                evidence_sha256: "a".repeat(64),
                model_licence: "fixture-only".into(),
            },
            price: Some(PriceSnapshot {
                snapshot_id: "fixture-price".into(),
                source_url: "https://fixture.invalid/price".into(),
                effective_date: "fixture-date".into(),
                input_nanos_per_million: 1_000_000,
                cached_input_nanos_per_million: 500_000,
                output_nanos_per_million: 2_000_000,
            }),
        };
        let request = FleetRequest {
            task_id: "fixture-task".into(),
            source_id: "fixture-source".into(),
            specification_sha256: "b".repeat(64),
            system_prompt: "Produce a Rust function.".into(),
            prompt: "Increment an integer.".into(),
            sample_slot: 0,
            generation: GenerationParameters {
                max_output_tokens: 100,
                ..Default::default()
            },
        };
        let prepared = prepare(&provider, &request).unwrap();
        let acquisition = Acquisition {
            round_id: "fixture-round".into(),
            task_id: request.task_id.clone(),
            source_id: request.source_id.clone(),
            request_key: prepared.request_key.clone(),
            provider_id: provider.id.clone(),
            business_id: provider.business_id.clone(),
            model: provider.model.clone(),
            model_revision: provider.model_revision.clone(),
            request_json: String::from_utf8(canonical_json(&prepared.body).unwrap()).unwrap(),
            price_json: serde_json::to_string(provider.price.as_ref().unwrap()).unwrap(),
            reserved_nanos: prepared.reservation_nanos,
            reserved_tokens: Some(
                prepared.input_token_upper_bound + request.generation.max_output_tokens,
            ),
            replay: true,
        };
        let limits = BudgetLimits {
            round_nanos: 100_000,
            task_nanos: 100_000,
            provider_nanos: 100_000,
            model_nanos: 100_000,
            tokens: Default::default(),
        };
        let Claim::Reserved(claim) = ledger.claim_and_reserve(&acquisition, limits).unwrap() else {
            panic!("fixture reservation failed")
        };
        let record = Recording {
            api: Default::default(),
            schema_version: 2,
            origin: "synthetic_fixture".into(),
            acquired_unix_millis: unix_millis(),
            request_key: prepared.request_key,
            attempt_ordinal: claim.ordinal,
            provider_id: provider.id,
            business_id: provider.business_id,
            model: provider.model,
            model_revision: provider.model_revision,
            endpoint: provider.endpoint,
            output_cap_parameter: provider.output_cap_parameter,
            request,
            request_body: prepared.body,
            price: provider.price.unwrap(),
            permission: provider.permission,
            response: TransportResponse {
                status: 200,
                headers: Default::default(),
                raw_body: json!({"choices":[{"finish_reason":"stop","message":{"content":"fn solve(x:i64)->i64{x+1}"}}],"usage":{"prompt_tokens":50,"completion_tokens":20}}).to_string(),
                elapsed_millis: 1,
            },
        };
        Self {
            directory,
            ledger,
            acquisition,
            claim,
            record,
            limits,
        }
    }

    fn publish(&self, record: &Recording) -> String {
        self.ledger
            .publish_artifact(&serde_json::to_vec_pretty(record).unwrap())
            .unwrap()
    }

    fn reopen(&self) -> Ledger {
        Ledger::open(
            self.directory.path().join("ledger.db"),
            self.directory.path().join("artifacts"),
        )
        .unwrap()
    }

    fn assert_unknown(&self) {
        let reopened = self.reopen();
        assert_eq!(reopened.recover_interrupted().unwrap(), 1);
        assert!(matches!(
            reopened
                .claim_and_reserve(&self.acquisition, self.limits)
                .unwrap(),
            Claim::Unknown
        ));
        let rows = reopened.costs(None).unwrap();
        assert_eq!(rows.len(), 1, "recovery must never create another attempt");
        assert_eq!(rows[0].state, "unknown");
        assert_eq!(rows[0].historical_lower_nanos, 0);
        assert_eq!(
            rows[0].historical_upper_nanos,
            self.acquisition.reserved_nanos
        );
        assert_eq!(
            reopened
                .attempt_proof(&self.claim.attempt_id)
                .unwrap()
                .reserved_nanos,
            self.acquisition.reserved_nanos
        );
        let mut next = self.acquisition.clone();
        next.request_key = "c".repeat(64);
        assert!(
            reopened.claim_and_reserve(&next, self.limits).is_err(),
            "the round must remain blocked"
        );
        assert_eq!(reopened.recover_interrupted().unwrap(), 0);
        assert_eq!(
            reopened
                .events(None)
                .unwrap()
                .iter()
                .filter(|e| e.kind == "recovery_unknown")
                .count(),
            1
        );
    }
}

#[test]
fn published_blob_recovers_completed_without_a_new_attempt() {
    let fixture = Fixture::new();
    let hash = fixture.publish(&fixture.record);
    let reopened = fixture.reopen();
    assert_eq!(reopened.recover_interrupted().unwrap(), 0);
    let Claim::Cached {
        attempt_id,
        artifact_sha256,
    } = reopened
        .claim_and_reserve(&fixture.acquisition, fixture.limits)
        .unwrap()
    else {
        panic!("published work was not recovered")
    };
    assert_eq!(attempt_id, fixture.claim.attempt_id);
    assert_eq!(artifact_sha256, hash);
    let proof = reopened.attempt_proof(&attempt_id).unwrap();
    assert_eq!(proof.state, "completed");
    assert_eq!(proof.historical_lower_nanos, 90);
    assert_eq!(proof.historical_upper_nanos, 90);
    assert_eq!(proof.input_tokens, Some(50));
    assert_eq!(proof.output_tokens, Some(20));
    assert_eq!(reopened.costs(None).unwrap().len(), 1);
    assert_eq!(reopened.recover_interrupted().unwrap(), 0);
    let events = reopened.events(None).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == "recovery_reconciled")
            .count(),
        1
    );
    assert!(events.last().unwrap().detail.contains("synthetic_fixture"));
    let original: Recording =
        serde_json::from_slice(&reopened.read_artifact(&hash).unwrap()).unwrap();
    assert_eq!(original.permission.decision_id, "fixture-only-permission");
}

#[test]
fn durable_http_failure_recovers_actual_or_uncertain_billing() {
    for with_usage in [true, false] {
        let mut fixture = Fixture::new();
        fixture.record.response.status = 429;
        fixture.record.response.raw_body = if with_usage {
            json!({"error":{"message":"labelled fixture"},"usage":{"prompt_tokens":50,"completion_tokens":20}}).to_string()
        } else {
            json!({"error":{"message":"labelled fixture"}}).to_string()
        };
        let hash = fixture.publish(&fixture.record);
        let reopened = fixture.reopen();
        assert_eq!(reopened.recover_interrupted().unwrap(), 0);
        let proof = reopened.attempt_proof(&fixture.claim.attempt_id).unwrap();
        assert_eq!(proof.state, "failed");
        assert_eq!(proof.response_artifact_sha256, Some(hash));
        assert_eq!(
            proof.historical_lower_nanos,
            if with_usage { 90 } else { 0 }
        );
        assert_eq!(
            proof.historical_upper_nanos,
            if with_usage {
                90
            } else {
                fixture.acquisition.reserved_nanos
            }
        );
        assert_eq!(proof.input_tokens, with_usage.then_some(50));
        assert_eq!(reopened.costs(None).unwrap().len(), 1);
    }
}

#[test]
fn missing_corrupt_temporary_and_ambiguous_evidence_remain_unknown() {
    for case in [
        "missing",
        "corrupt",
        "malformed",
        "temporary",
        "ambiguous",
        "symlink",
    ] {
        let fixture = Fixture::new();
        let bytes = serde_json::to_vec_pretty(&fixture.record).unwrap();
        match case {
            "missing" => {}
            "corrupt" => {
                fs::write(
                    fixture
                        .ledger
                        .artifacts
                        .join(format!("{}.json", "d".repeat(64))),
                    bytes,
                )
                .unwrap();
            }
            "malformed" => {
                fixture.ledger.publish_artifact(b"not valid JSON").unwrap();
            }
            "temporary" => {
                fs::write(fixture.ledger.artifacts.join(".unpublished.tmp"), bytes).unwrap();
            }
            "ambiguous" => {
                fixture.publish(&fixture.record);
                let mut other = fixture.record.clone();
                other
                    .response
                    .headers
                    .insert("x-request-id".into(), "different-fixture-request".into());
                fixture.publish(&other);
            }
            "symlink" => {
                let target = fixture.directory.path().join("outside.json");
                fs::write(&target, &bytes).unwrap();
                std::os::unix::fs::symlink(
                    target,
                    fixture
                        .ledger
                        .artifacts
                        .join(format!("{}.json", sha256(&bytes))),
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        fixture.assert_unknown();
    }
}

#[test]
fn mismatched_identity_price_and_invalid_usage_remain_unknown() {
    for case in [
        "ordinal",
        "provider",
        "business",
        "model",
        "revision",
        "task",
        "source",
        "specification",
        "body",
        "price",
        "input_bound",
        "output_bound",
        "inconsistent_usage",
        "malformed_usage",
        "malformed_optional_usage",
        "conflicting_optional_usage",
    ] {
        let mut fixture = Fixture::new();
        match case {
            "ordinal" => fixture.record.attempt_ordinal += 1,
            "provider" => fixture.record.provider_id.push('x'),
            "business" => fixture.record.business_id.push('x'),
            "model" => fixture.record.model.push('x'),
            "revision" => fixture.record.model_revision.push('x'),
            "task" => fixture.record.request.task_id.push('x'),
            "source" => fixture.record.request.source_id.push('x'),
            "specification" => fixture.record.request.specification_sha256 = "c".repeat(64),
            "body" => fixture.record.request_body["temperature"] = json!(1.0),
            "price" => fixture.record.price.output_nanos_per_million += 1,
            "input_bound" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":100_000,"completion_tokens":20}}).to_string(),
            "output_bound" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":50,"completion_tokens":101}}).to_string(),
            "inconsistent_usage" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":50,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":51}}}).to_string(),
            "malformed_usage" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":"fifty","completion_tokens":20}}).to_string(),
            "malformed_optional_usage" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":50,"completion_tokens":20,"completion_tokens_details":{"reasoning_tokens":"five"}}}).to_string(),
            "conflicting_optional_usage" => fixture.record.response.raw_body = json!({"usage":{"prompt_tokens":50,"completion_tokens":20,"prompt_tokens_details":{"cached_tokens":10},"prompt_cache_hit_tokens":20}}).to_string(),
            _ => unreachable!(),
        }
        fixture.publish(&fixture.record);
        fixture.assert_unknown();
    }
}

#[test]
fn valid_blob_cannot_override_conflicting_invalid_evidence_or_a_prior_block() {
    let fixture = Fixture::new();
    fixture.publish(&fixture.record);
    let mut conflict = fixture.record.clone();
    conflict.price.snapshot_id.push('x');
    fixture.publish(&conflict);
    fixture.assert_unknown();

    let fixture = Fixture::new();
    fixture.publish(&fixture.record);
    fixture
        .ledger
        .block_round(&fixture.acquisition.round_id, "unrelated integrity failure")
        .unwrap();
    assert_eq!(fixture.ledger.recover_interrupted().unwrap(), 0);
    let mut next = fixture.acquisition.clone();
    next.request_key = "c".repeat(64);
    assert!(
        fixture
            .ledger
            .claim_and_reserve(&next, fixture.limits)
            .is_err()
    );
}

#[test]
fn prior_retry_blob_is_not_ambiguous_with_exact_current_ordinal() {
    let mut fixture = Fixture::new();
    let mut previous = fixture.record.clone();
    previous.response.status = 500;
    let previous_hash = fixture.publish(&previous);
    fixture
        .ledger
        .finish(
            &fixture.claim.attempt_id,
            Some(&previous_hash),
            false,
            90,
            90,
            Some((50, 0, 20, 0)),
            Some("labelled fixture failure"),
        )
        .unwrap();
    let Claim::Reserved(current) = fixture
        .ledger
        .claim_and_reserve(&fixture.acquisition, fixture.limits)
        .unwrap()
    else {
        panic!("labelled retry could not reserve")
    };
    assert_eq!(current.ordinal, 2);
    fixture.record.attempt_ordinal = current.ordinal;
    let current_hash = fixture.publish(&fixture.record);
    let reopened = fixture.reopen();
    assert_eq!(reopened.recover_interrupted().unwrap(), 0);
    assert!(
        matches!(reopened.lookup_request(&fixture.acquisition.request_key).unwrap(), Some(Claim::Cached { attempt_id, artifact_sha256 }) if attempt_id == current.attempt_id && artifact_sha256 == current_hash)
    );
    let rows = reopened.costs(None).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().any(|r| r.ordinal == 1 && r.state == "failed"));
    assert!(
        rows.iter()
            .any(|r| r.ordinal == 2 && r.state == "completed")
    );
}

#[test]
fn conflicting_database_state_refuses_partial_reconciliation() {
    let fixture = Fixture::new();
    fixture.publish(&fixture.record);
    fixture
        .ledger
        .connection()
        .unwrap()
        .execute(
            "UPDATE requests SET state='unknown' WHERE request_key=?1",
            [&fixture.acquisition.request_key],
        )
        .unwrap();
    assert!(fixture.ledger.recover_interrupted().is_err());
    assert_eq!(
        fixture
            .ledger
            .attempt_proof(&fixture.claim.attempt_id)
            .unwrap()
            .state,
        "in_flight"
    );
    assert!(
        !fixture
            .ledger
            .events(None)
            .unwrap()
            .iter()
            .any(|e| e.kind == "recovery_reconciled")
    );
}
