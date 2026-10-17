//! Explicit operator choices for a secret-free local provider configuration.
use super::*;

#[derive(clap::Args)]
pub struct ProviderInitArguments {
    #[arg(long, default_value = "state/api-providers.local.toml")]
    config: PathBuf,
    #[arg(long, value_parser = ["openai", "gemini", "vertex", "claude"])]
    provider: String,
    /// Exact API model available to your account; no model is selected automatically.
    #[arg(long)]
    model: String,
    /// Pinned revision or an honest, documented alias revision policy.
    #[arg(long)]
    model_revision: String,
    /// Required for Vertex; otherwise use the provider's standard public endpoint.
    #[arg(long)]
    endpoint: Option<String>,
    /// Reviewed conservative USD per million input tokens, as a decimal.
    #[arg(long)]
    input_usd_per_million: String,
    #[arg(long)]
    cached_input_usd_per_million: String,
    #[arg(long)]
    output_usd_per_million: String,
    #[arg(long)]
    price_source: String,
    #[arg(long)]
    price_date: String,
    /// Local limit for all money scopes; does not authorise a live purchase.
    #[arg(long)]
    budget_usd: String,
    /// Cumulative input plus billable output limit for each scope.
    #[arg(long, default_value_t = 20000)]
    token_budget: u64,
    /// Per-call cap including reasoning, not a target response length.
    #[arg(long, default_value_t = 4096)]
    max_output_tokens: u64,
}

impl ProviderInitArguments {
    pub fn initialise(self) -> Result<fleet::ProviderChecks> {
        let input = usd_nanos(&self.input_usd_per_million).context("invalid input token price")?;
        let cached = usd_nanos(&self.cached_input_usd_per_million)
            .context("invalid cached input token price")?;
        let output =
            usd_nanos(&self.output_usd_per_million).context("invalid output token price")?;
        let identity = serde_json::to_vec(&serde_json::json!([
            self.model,
            self.model_revision,
            self.price_source,
            self.price_date,
            input,
            cached,
            output
        ]))?;
        fleet::initialise_provider(fleet::ProviderInitOptions {
            output: self.config,
            provider: self.provider,
            model: self.model,
            model_revision: self.model_revision,
            endpoint: self.endpoint,
            price: fleet::PriceSnapshot {
                snapshot_id: format!(
                    "operator-reviewed-{}",
                    lec_distil::ledger::sha256(&identity)
                ),
                source_url: self.price_source,
                effective_date: self.price_date,
                input_nanos_per_million: input,
                cached_input_nanos_per_million: cached,
                output_nanos_per_million: output,
            },
            budget_nanos: usd_nanos(&self.budget_usd)?,
            token_budget: self.token_budget,
            max_output_tokens: self.max_output_tokens,
        })
    }
}
