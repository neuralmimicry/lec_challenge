//! Operator setup invariants. All model identities and prices here are fixtures.
use super::*;

fn options(path: &Path) -> ProviderInitOptions {
    ProviderInitOptions {
        output: path.to_owned(),
        provider: "openai".into(),
        model: "fixture-coding-model".into(),
        model_revision: "fixture-pinned-revision".into(),
        endpoint: None,
        price: PriceSnapshot {
            snapshot_id: "LABELLED_SETUP_FIXTURE".into(),
            source_url: "https://fixture.invalid/pricing".into(),
            effective_date: "2026-10-16".into(),
            input_nanos_per_million: 2_000_000_000,
            cached_input_nanos_per_million: 2_000_000_000,
            output_nanos_per_million: 10_000_000_000,
        },
        budget_nanos: 250_000_000,
        token_budget: 20000,
        max_output_tokens: 4096,
    }
}

#[test]
fn setup_produces_bounded_native_inference_routes_without_training_clearance() {
    for id in ["openai", "gemini", "vertex", "claude"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/local.toml");
        let mut o = options(&path);
        o.provider = id.into();
        if id == "vertex" {
            o.endpoint = Some("https://aiplatform.googleapis.com/v1/projects/fixture/locations/global/publishers/google/models/fixture-coding-model:generateContent".into());
        }
        initialise_provider(o).unwrap();
        let config = FleetConfig::load(&path).unwrap();
        assert!(config.require_three_businesses().is_err());
        assert_eq!(config.providers.iter().filter(|p| p.enabled).count(), 1);
        let p = config.providers.iter().find(|p| p.id == id).unwrap();
        assert!(p.inference_only && !p.permission.approved_for_output_training);
        let request = FleetRequest {
            task_id: "setup-fixture".into(),
            source_id: "fixture".into(),
            specification_sha256: sha256(b"fixture"),
            system_prompt: "Return text.".into(),
            prompt: "Say hello.".into(),
            sample_slot: 0,
            generation: GenerationParameters {
                temperature: 1.0,
                top_p: 1.0,
                seed: None,
                max_output_tokens: 2048,
            },
        };
        let prepared = prepare(p, &request).unwrap();
        assert!(prepared.reservation_nanos <= config.budgets.task_nanos);
        assert!(prepared.input_token_upper_bound + 2048 <= config.budgets.tokens.task.unwrap());
        let ready = access::provider_status_fixture(&path, true).unwrap();
        assert_eq!(ready.config_path, path);
        assert!(ready.locally_ready_for_live_request && !ready.network_contacted);
        assert_eq!(
            ready
                .providers
                .iter()
                .filter(|p| p.locally_ready_for_live_request)
                .count(),
            1
        );
        assert!(
            ready
                .providers
                .iter()
                .all(|p| !p.training_eligible && !p.network_contacted)
        );
        let absent = access::provider_status_fixture(&path, false).unwrap();
        assert!(!absent.locally_ready_for_live_request);
        assert!(
            absent
                .providers
                .iter()
                .find(|p| p.id == id)
                .unwrap()
                .readiness_blockers
                .iter()
                .any(|b| b.contains(&p.api_key_environment))
        );
    }
}

#[test]
fn setup_never_replaces_a_file_or_symlink() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("local.toml");
    fs::write(&path, b"OPERATOR_EXISTING_CONFIGURATION").unwrap();
    let error = initialise_provider(options(&path)).unwrap_err();
    assert!(error.to_string().contains("never overwritten"));
    assert_eq!(fs::read(&path).unwrap(), b"OPERATOR_EXISTING_CONFIGURATION");
    let link = dir.path().join("link.toml");
    symlink(&path, &link).unwrap();
    assert!(initialise_provider(options(&link)).is_err());
    assert!(link.is_symlink());
    assert_eq!(fs::read(&path).unwrap(), b"OPERATOR_EXISTING_CONFIGURATION");
}

#[test]
fn invalid_setup_is_rejected_before_a_configuration_is_published() {
    for case in 0..12 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("local.toml");
        let mut o = options(&path);
        match case {
            0 => o.model = "UNRESOLVED_MODEL".into(),
            1 => o.model_revision.clear(),
            2 => o.budget_nanos = 0,
            3 => o.budget_nanos = 1,
            4 => o.token_budget = 100,
            5 => o.price.output_nanos_per_million = -1,
            6 => o.provider = "vertex".into(),
            7 => o.endpoint = Some("https://user:password@fixture.invalid/v1/responses".into()),
            8 => o.max_output_tokens = u64::MAX,
            9 => o.price.source_url = "https://user:password@fixture.invalid/pricing".into(),
            10 => o.price.effective_date = "not-a-date".into(),
            _ => {
                o.price.source_url = "https://fixture.invalid/pricing?token=LABELLED_FIXTURE".into()
            }
        }
        assert!(initialise_provider(o).is_err(), "case {case}");
        assert!(
            !path.exists(),
            "published invalid configuration in case {case}"
        );
    }
}

#[test]
fn local_checks_distinguish_template_key_presence_and_zero_allowances() {
    let status =
        access::provider_status_fixture(Path::new("config/api-providers.toml"), true).unwrap();
    assert!(!status.locally_ready_for_live_request && !status.network_contacted);
    assert!(status.configuration_error.is_none());
    for p in &status.providers {
        assert!(p.credential_present && !p.locally_ready_for_live_request);
        for expected in [
            "disabled",
            "unresolved",
            "price snapshot",
            "budgets.round_nanos",
            "budgets.tokens.round",
        ] {
            assert!(
                p.readiness_blockers.iter().any(|b| b.contains(expected)),
                "missing {expected}"
            );
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("valid.toml");
    initialise_provider(options(&path)).unwrap();
    let mut config = FleetConfig::load(&path).unwrap();
    config.budgets.tokens.model = Some(0);
    fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    assert!(
        !access::provider_status_fixture(&path, true)
            .unwrap()
            .locally_ready_for_live_request
    );
    config.budgets.tokens.model = Some(20000);
    config.providers[0].price = None;
    fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    assert!(
        !access::provider_status_fixture(&path, true)
            .unwrap()
            .locally_ready_for_live_request
    );
}

#[test]
fn missing_and_malformed_configuration_errors_name_the_path_and_setup_action() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing-local.toml");
    for error in [
        FleetConfig::load(&path).unwrap_err(),
        provider_status(&path).unwrap_err(),
    ] {
        let message = error.to_string();
        assert!(message.contains(path.to_str().unwrap()));
        assert!(message.contains("provider-init --help") && message.contains("same --config path"));
    }
    fs::write(&path, "[invalid TOML").unwrap();
    assert!(
        FleetConfig::load(&path)
            .unwrap_err()
            .to_string()
            .contains(path.to_str().unwrap())
    );
}
