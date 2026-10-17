//! Operator checks and single inference requests, separate from training collection.
use super::*;

#[derive(Debug, Serialize)]
pub struct ProviderStatus {
    pub id: String,
    pub api: ProviderApi,
    pub enabled: bool,
    pub model: String,
    pub model_revision: String,
    pub api_key_environment: String,
    pub credential_present: bool,
    pub configuration_error: Option<String>,
    pub locally_ready_for_live_request: bool,
    pub readiness_blockers: Vec<String>,
    pub training_eligible: bool,
    pub network_contacted: bool,
}

#[derive(Debug, Serialize)]
pub struct ProviderChecks {
    pub config_path: std::path::PathBuf,
    pub providers: Vec<ProviderStatus>,
    pub budgets: BudgetLimits,
    pub configuration_error: Option<String>,
    pub locally_ready_for_live_request: bool,
    pub network_contacted: bool,
}

/// Local validation only: neither a credential nor a successful check proves access.
pub fn provider_status(path: &Path) -> Result<ProviderChecks> {
    provider_status_with_credentials(path, |name| {
        std::env::var(name).is_ok_and(|key| {
            !key.trim().is_empty() && reqwest::header::HeaderValue::from_str(&key).is_ok()
        })
    })
}

fn provider_status_with_credentials(
    path: &Path,
    credential_present: impl Fn(&str) -> bool,
) -> Result<ProviderChecks> {
    let config = FleetConfig::read(path)?;
    let configuration_error = config.validate().err().map(|e| e.to_string());
    let budget_blockers = budget_blockers(&config.budgets);
    let providers = config
        .providers
        .iter()
        .map(|provider| {
            let mut enabled = provider.clone();
            enabled.enabled = true;
            let error = enabled.validate_enabled().err().map(|e| e.to_string());
            let credential_present = credential_present(&provider.api_key_environment);
            let mut readiness_blockers = budget_blockers.clone();
            if !provider.enabled {
                readiness_blockers.push(
                    "provider is disabled; enable only after reviewing model, prices and limits"
                        .into(),
                );
            }
            if let Some(error) = &error {
                readiness_blockers.push(error.clone());
            }
            if provider.price.is_none() {
                readiness_blockers.push("missing reviewed price snapshot".into());
            }
            if !credential_present {
                readiness_blockers.push(format!(
                    "export a valid {} in the same shell",
                    provider.api_key_environment
                ));
            }
            if let Some(error) = &configuration_error {
                readiness_blockers.push(format!("configuration is invalid: {error}"));
            }
            ProviderStatus {
                id: provider.id.clone(),
                api: provider.api,
                enabled: provider.enabled,
                model: provider.model.clone(),
                model_revision: provider.model_revision.clone(),
                api_key_environment: provider.api_key_environment.clone(),
                credential_present,
                configuration_error: error,
                locally_ready_for_live_request: readiness_blockers.is_empty(),
                readiness_blockers,
                training_eligible: provider.enabled
                    && !provider.inference_only
                    && provider.permission.approved_for_output_training,
                network_contacted: false,
            }
        })
        .collect::<Vec<_>>();
    Ok(ProviderChecks {
        config_path: path.to_owned(),
        locally_ready_for_live_request: providers.iter().any(|p| p.locally_ready_for_live_request),
        providers,
        budgets: config.budgets,
        configuration_error,
        network_contacted: false,
    })
}

fn budget_blockers(budgets: &BudgetLimits) -> Vec<String> {
    let mut blockers = Vec::new();
    for (scope, value) in [
        ("round", budgets.round_nanos),
        ("task", budgets.task_nanos),
        ("provider", budgets.provider_nanos),
        ("model", budgets.model_nanos),
    ] {
        if value <= 0 {
            blockers.push(format!(
                "budgets.{scope}_nanos must be positive for live requests"
            ));
        }
    }
    for (scope, value) in [
        ("round", budgets.tokens.round),
        ("task", budgets.tokens.task),
        ("provider", budgets.tokens.provider),
        ("model", budgets.tokens.model),
    ] {
        if value == Some(0) {
            blockers.push(format!(
                "budgets.tokens.{scope} is zero; no tokens are allowed"
            ));
        }
    }
    blockers
}

#[cfg(test)]
pub(super) fn provider_status_fixture(path: &Path, present: bool) -> Result<ProviderChecks> {
    provider_status_with_credentials(path, |_| present)
}

pub struct ProviderRequestOptions {
    pub config: std::path::PathBuf,
    pub provider: String,
    pub prompt: String,
    pub max_output_tokens: u64,
    pub round: String,
    pub authorised_nanos: Option<i64>,
    pub offline: bool,
    pub state_directory: std::path::PathBuf,
    pub recordings: std::path::PathBuf,
    pub output: std::path::PathBuf,
}

/// Buys at most one successful text response; never executes or curates its output.
pub async fn provider_request(options: ProviderRequestOptions) -> Result<FleetCompletion> {
    let mut config = FleetConfig::load(&options.config)?;
    for provider in &mut config.providers {
        provider.inference_only = true;
    }
    let transport: Arc<dyn Transport> = if options.offline {
        ensure!(
            options.authorised_nanos.is_none(),
            "offline replay does not accept spending authority"
        );
        let directory = options.recordings.clone();
        Arc::new(tokio::task::spawn_blocking(move || ReplayTransport::load(directory)).await??)
    } else {
        let cap = options
            .authorised_nanos
            .context("live request requires explicit --authorised-usd")?;
        ensure!(cap > 0, "live spending cap must be positive");
        config.budgets.round_nanos = config.budgets.round_nanos.min(cap);
        Arc::new(LiveTransport::new(true)?)
    };
    let directory = options.state_directory;
    let (ledger, _lease) = tokio::task::spawn_blocking(move || -> Result<_> {
        fs::create_dir_all(&directory)?;
        let lease = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join("orchestrator.lock"))?;
        use std::os::fd::AsRawFd;
        // SAFETY: the live file descriptor is owned until this request finishes.
        ensure!(
            unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another provider orchestrator owns this state directory"
        );
        let ledger = Ledger::open(directory.join("ledger.db"), directory.join("artifacts"))?;
        ledger.recover_interrupted()?;
        Ok((ledger, lease))
    })
    .await??;
    let specification = sha256(options.prompt.as_bytes());
    let request = FleetRequest {
        task_id: format!("operator-{specification}"),
        source_id: "operator-inference".into(),
        specification_sha256: specification,
        system_prompt: "Return a plain text answer. Do not use tools.".into(),
        prompt: options.prompt,
        sample_slot: 0,
        generation: GenerationParameters {
            temperature: 1.0,
            top_p: 1.0,
            seed: None,
            max_output_tokens: options.max_output_tokens,
        },
    };
    let runner = FleetRunner::new(config, ledger.clone(), transport)?;
    let result = runner
        .complete_on(&options.round, &options.provider, request)
        .await;
    let output = options.output;
    let round = options.round;
    let outcome = match &result {
        Ok(completion) => json!({"complete":true,"completion":completion}),
        Err(error) => json!({"complete":false,"reason":format!("{error:#}")}),
    };
    tokio::task::spawn_blocking(move || -> Result<()> {
        ledger.export_costs(Some(&round), &output)?;
        crate::artefact::write_json(&output.join("request.json"), &outcome)?;
        Ok(())
    })
    .await??;
    result
}
