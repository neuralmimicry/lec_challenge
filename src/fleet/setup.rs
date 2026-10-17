//! Secret-free inference setup; model identity and reviewed rates remain explicit.
use super::*;
use std::io::Write;

pub struct ProviderInitOptions {
    pub output: std::path::PathBuf,
    pub provider: String,
    pub model: String,
    pub model_revision: String,
    pub endpoint: Option<String>,
    pub price: PriceSnapshot,
    pub budget_nanos: i64,
    pub token_budget: u64,
    pub max_output_tokens: u64,
}

/// Creates one enabled inference route. Never stores a key, calls HTTP or grants spend.
pub fn initialise_provider(options: ProviderInitOptions) -> Result<ProviderChecks> {
    let source = reqwest::Url::parse(&options.price.source_url)
        .context("price source must be a public HTTPS documentation URL")?;
    ensure!(
        source.scheme() == "https"
            && source.host_str().is_some()
            && source.username().is_empty()
            && source.password().is_none()
            && source.query().is_none(),
        "price source must be HTTPS without credentials or query parameters"
    );
    ensure!(
        options.price.effective_date.len() == 10
            && options
                .price
                .effective_date
                .bytes()
                .enumerate()
                .all(|(i, b)| {
                    if i == 4 || i == 7 {
                        b == b'-'
                    } else {
                        b.is_ascii_digit()
                    }
                }),
        "price date must use YYYY-MM-DD"
    );
    ensure!(
        options.budget_nanos > 0,
        "setup dollar ceiling must be positive"
    );
    ensure!(
        options.token_budget > 0 && options.token_budget <= i64::MAX as u64,
        "setup token ceiling must be positive and fit the database integer range"
    );
    ensure!(
        !options.model.is_empty()
            && options
                .model
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b)),
        "setup requires an exact model ID using letters, digits, hyphens, dots or underscores"
    );
    let mut config: FleetConfig = toml::from_str(include_str!("../../config/api-providers.toml"))?;
    let selected = config
        .providers
        .iter_mut()
        .find(|p| p.id == options.provider)
        .context("unknown setup provider; choose openai, gemini, vertex or claude")?;
    if selected.api == ProviderApi::VertexGenerateContent {
        ensure!(
            options.endpoint.is_some(),
            "Vertex setup requires --endpoint with your project, location and model ID; see docs/api-providers.md"
        );
    }
    if let Some(endpoint) = options.endpoint {
        selected.endpoint = endpoint;
    } else if selected.api == ProviderApi::GeminiGenerateContent {
        selected.endpoint = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            options.model
        );
    }
    selected.model = options.model;
    selected.model_revision = options.model_revision;
    selected.price = Some(options.price);
    selected.max_output_tokens = options.max_output_tokens;
    selected.enabled = true;
    // A first smoke request makes no retry/fallback purchase. Operators can later
    // add explicitly reviewed routes and bounded retry settings to the local file.
    selected.retries = 0;
    selected.failover.clear();
    selected.validate_enabled()?;
    let worst_tokens = selected
        .max_input_tokens
        .checked_add(selected.max_output_tokens)
        .context("setup token reservation overflow")?;
    ensure!(
        worst_tokens <= options.token_budget && worst_tokens <= selected.tokens_per_minute,
        "token ceiling/rate must cover max_input_tokens plus max_output_tokens ({worst_tokens})"
    );
    let worst_cost = selected
        .price
        .as_ref()
        .unwrap()
        .reserve(selected.max_input_tokens, selected.max_output_tokens)?;
    ensure!(
        worst_cost <= options.budget_nanos,
        "setup dollar ceiling is below the worst-case per-call reservation ({worst_cost} nanodollars); reduce token caps or increase the explicitly chosen ceiling"
    );
    config.max_parallel_tasks = 1;
    config.budgets = BudgetLimits {
        round_nanos: options.budget_nanos,
        task_nanos: options.budget_nanos,
        provider_nanos: options.budget_nanos,
        model_nanos: options.budget_nanos,
        tokens: crate::ledger::TokenBudgetLimits {
            round: Some(options.token_budget),
            task: Some(options.token_budget),
            provider: Some(options.token_budget),
            model: Some(options.token_budget),
        },
    };
    config.validate()?;
    let bytes = format!(
        "# Local inference configuration. No credentials; no output-training clearance.\n# Creating this file does not authorise a paid call. See docs/api-providers.md.\n{}",
        toml::to_string_pretty(&config)?
    );
    let path = &options.output;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    // Same-directory atomic no-clobber publication also protects existing symlinks.
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes.as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary.persist_noclobber(path).with_context(|| format!(
        "cannot create {}; existing files are never overwritten; inspect that file or choose another --config path", path.display()
    ))?;
    fs::File::open(parent)?.sync_all()?;
    provider_status(path)
}
