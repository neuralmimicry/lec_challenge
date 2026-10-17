# OpenAI Codex, Google Gemini and Anthropic Claude

The native Rust fleet supports OpenAI Responses, Google Gemini `generateContent`
(Developer API and GCP Vertex AI), Anthropic Messages and the existing Chat
Completions providers. These are text-only requests: no tools, web search,
automatic continuations or candidate execution. API access does not grant
output-training rights. The new providers default to `inference_only = true`;
their outputs cannot become assessment training rows. See [rights](rights.md).

No live connectivity, account access, usage or prices are claimed by the fixture
tests. These instructions do not authorise spending. A ChatGPT/Codex, Gemini or
Claude consumer subscription is not an API credential or an API spending budget.
Use the selected API product's credentials, rather than a GitHub/Hugging Face
personal access token. Missing credentials are supported and produce explicit
unavailability; they do not trigger a sign-in, an invented response or a charge.

## Configure locally

`state/api-providers.local.toml` is a local file and is absent from a clean
checkout. Passing it to `provider-request` before creating it fails. The default
`provider-check` examines `config/api-providers.toml`, whose four providers are
deliberately disabled, unpriced and zero-budget. It does not automatically find
your local file. A present API key does not change those settings.

Use the complete [README setup example](../README.md#api-credentials-and-token-budgets)
to create a priced, bounded OpenAI Responses profile. `provider-init --help`
lists every option. It requires explicit model/revision choices, reviewed rates,
price source/date and a positive monetary ceiling. It enables only the selected
provider and keeps training permissions false. The four money scopes receive
`--budget-usd`; the four cumulative token scopes receive `--token-budget`
(default 20,000). Its per-call input cap is 8,192 and output cap is
`--max-output-tokens` (default 4,096). The full per-call reservation must fit the
chosen money/token ceilings and token rate. Creating a configuration gives no
spending authority: the separate request command still needs `--live` and
`--authorised-usd`.
Other disabled providers retain unresolved placeholders in check output. Only
enabled routes must be configured; those placeholders do not block the selected
valid route.

The file is published atomically without overwriting existing files or symlinks.
On a second invocation, inspect/edit the existing local TOML or choose another
`--config` path; initialisation never silently resets models, prices or limits.
It makes no HTTP request, needs no API key, and embeds the shipped template so
it can also work from a packaged binary. The first source build may take several
minutes. Cargo reuses that build on subsequent commands.

For Gemini or Claude, use the same initialisation command with `--provider gemini`
or `--provider claude`, the exact supported model/revision, and **that provider's**
reviewed rates/source/date. The command selects the matching API and environment
name; Gemini's endpoint incorporates the model ID. For Vertex, use
`--provider vertex` and supply `--endpoint` containing your project, location and
matching model ID as described below. Check each model's supported request
contract before use; a syntactically valid model ID cannot prove account access.

For manual configuration of several providers, copy the disabled template to a
new local file and edit it deliberately. The copy below preserves an existing
local file:

```sh
mkdir -p state
cp --no-clobber config/api-providers.toml state/api-providers.local.toml
./run.sh provider-check --config state/api-providers.local.toml
```

`provider-check` reads configuration and credential presence only. It prints
`config_path`, model/revision, environment variable names, enabled state,
`readiness_blockers` and `locally_ready_for_live_request`, never credential values.
`network_contacted` remains false. Its normal successful exit means the status
was produced. Add `--require-ready` to exit nonzero when no provider passes local
checks. Readiness requires an enabled valid/priced provider, credential presence,
valid fleet configuration and nonzero allowances; it does not prove provider
availability, entitlement or sufficient remaining ledger budget for your actual
prompt. It works without credentials or an internet connection. A null top-level
`configuration_error` means the template is structurally valid; inspect each
route's blockers before requesting a live call.

Edit the local TOML. Enable only the providers you intend to use and set an
available exact `model`, an honest `model_revision` or documented provider
revision policy, prices and budgets. `UNRESOLVED...` identities are refused when
enabled. For both Google routes the endpoint must contain the same model ID as
`model`. Keep `inference_only = true` and
`approved_for_output_training = false`. Each enabled provider must be valid,
priced and bounded; a disabled fallback is simply skipped. The supplied failover
lists are explicit, editable routes. The two Google endpoints share the same
business identity and cannot count as two independent providers.

## OpenAI Codex through Responses

Create an API project key in the [OpenAI platform](https://platform.openai.com/api-keys)
with access to the selected Codex API model. Consult the official
[model catalogue](https://developers.openai.com/api/docs/models)
and your account's available models. The
[GPT-5.3-Codex page](https://developers.openai.com/api/docs/models/gpt-5.3-codex)
is marked deprecated in the documentation checked on 16 October 2026; its
existence does not establish account availability. The README's explicit
`gpt-6.1-sol` example uses the same Responses adapter and supports low reasoning
effort; it is a named setup example, not an automatic model migration.
No default model is silently selected or upgraded. Use `api = "openai_responses"`, the
`https://api.openai.com/v1/responses` endpoint and
`api_key_environment = "OPENAI_API_KEY"`.

In Bash, enter the key without placing it in shell history or command arguments:

```sh
read -r -s -p 'OpenAI API key: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
```

The adapter sends Bearer authentication, `store: false`, low reasoning effort
and `max_output_tokens`. The output cap includes reasoning and visible text.
Seed, temperature and top-p controls are omitted for Codex; the internal request
requires neutral sampling fields to prevent silently ignoring a requested setting.
The CLI default is 2,048 output tokens; the initialised per-call ceiling is
4,096. A short greeting can still need reasoning tokens: 256 is a small ceiling
and may yield an incomplete response with no visible text. Specify
`--max-output-tokens 2048` for the documented smoke request. Increase the request
cap only within the configured cap and approved budget; raising it changes the
request identity and can buy another call. An exhausted reasoning allowance or
incomplete response is a failure even if some text was returned. Its usage and
cost remain in the ledger. See the official [Responses API contract](https://developers.openai.com/api/reference/resources/responses/methods/create).
This integration does not read the Codex application's login files or use its
subscription OAuth credentials.

## Google Gemini Developer API

Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/api-keys),
enable billing where required and check the selected model's availability and
limits. Set `api = "gemini_generate_content"`,
`api_key_environment = "GEMINI_API_KEY"` and
`endpoint = "https://generativelanguage.googleapis.com/v1beta/models/MODEL_ID:generateContent"`.
Replace `MODEL_ID` and the matching `model` entry.

```sh
read -r -s -p 'Gemini API key: ' GEMINI_API_KEY
export GEMINI_API_KEY
printf '\n'
```

The key is sent in `x-goog-api-key`, never in a URL query. This adapter requests
one candidate with `maxOutputTokens` and `thinkingBudget = 0`. Select a model
whose documented API accepts disabled thinking, such as a compatible Gemini
Flash revision. Models requiring automatic thinking, a different thinking
configuration or another API version are unavailable through this bounded
adapter until their billing contract is implemented. A rejection follows the
configured failover route; the adapter never silently enables thinking. Returned
thought tokens are included in usage accounting and unexpected thinking blocks
the round for reconciliation. Grounding, tool use and blocked/truncated outputs
cannot be accepted. See Google's [generateContent contract](https://ai.google.dev/api/generate-content)
and [thinking controls](https://ai.google.dev/gemini-api/docs/thinking).

## Gemini on GCP Vertex AI

Enable the Vertex AI API in your Google Cloud project, configure billing and give
the calling identity the required Vertex prediction permissions. Choose a
supported location/model and set `api = "vertex_generate_content"`,
`api_key_environment = "GOOGLE_CLOUD_ACCESS_TOKEN"` and:

```text
https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/LOCATION/publishers/google/models/MODEL_ID:generateContent
```

For a model using the global endpoint, use `aiplatform.googleapis.com` with
`locations/global`. Keep `model = "MODEL_ID"` consistent with the URL. Vertex
uses a Google OAuth access token, rather than a Gemini Developer API key or a
personal access token. After authenticating the intended identity with the Google
Cloud CLI (`gcloud auth login`, or your approved service-account impersonation
procedure), obtain a short-lived token in your local shell:

```sh
export GOOGLE_CLOUD_ACCESS_TOKEN="$(gcloud auth print-access-token)"
```

The token travels as a Bearer header. Token refresh is an operator action; an
expired token produces a recorded authentication failure and configured failover.
Restart after exporting a fresh token. The application does not read service
account private-key files or invoke `gcloud` itself. Use your organisation's
approved identity/impersonation procedure; access, IAM, regional availability
and project quotas must all permit the request. The same text-only, disabled
thinking contract as the Developer API applies. See the official
[Vertex Gemini REST reference](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference).

## Anthropic Claude through Messages

Create an Anthropic API key in the [Claude Console](https://console.anthropic.com/)
with billing/access for your exact model. Set `api = "anthropic_messages"`,
`endpoint = "https://api.anthropic.com/v1/messages"` and
`api_key_environment = "ANTHROPIC_API_KEY"`.

```sh
read -r -s -p 'Anthropic API key: ' ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY
printf '\n'
```

The adapter sends `x-api-key`, `anthropic-version: 2023-06-01` and `max_tokens`.
It uses temperature alone, with no seed, tools, extended thinking or cache-control
blocks. Select a model accepting this Messages contract. Only an assistant
`end_turn` response containing text is accepted; `max_tokens`, refusal and tool
stops fail. Cached reads are counted once in total input; unexpected cache
creation is rejected because it requires separate pricing. See the official
[Messages reference](https://docs.anthropic.com/en/api/messages).

## Set both token and money budgets

The shipped inference configuration has **zero** monetary and token budgets and
no price snapshots. Set all four monetary limits and token limits deliberately.
The following is an illustrative small ceiling, not permission to spend:

```toml
[budgets]
round_nanos = 250000000    # USD 0.25 for this round, including retries/failover
task_nanos = 100000000     # USD 0.10 for one logical task
provider_nanos = 250000000
model_nanos = 250000000

[budgets.tokens]
round = 20000
task = 10000
provider = 20000
model = 20000
```

One dollar is `1_000_000_000` nanodollars. Token caps count **input plus billable
output**, including cached input once and reasoning output once. Per-provider
`max_input_tokens` and `max_output_tokens` bound each request;
`requests_per_minute`, `tokens_per_minute`, `concurrency` and
`max_parallel_tasks` control dispatch. The request's `--max-output-tokens` must
fit every intended fallback's output cap. TPM is a rate limit, not a cumulative
spending limit. Omitting an optional cumulative token scope in an old
configuration leaves that scope uncapped; `0` means no allowance. Keep explicit
scopes in new configurations. The limits cover this application's ledger, not
other applications using the account.

Add a `[providers.price]` table inside each enabled provider's entry, before the
next `[[providers]]`. Record a real dated source and reviewed rates. For example,
an **illustrative** USD 1 per million input tokens would be represented as
`input_nanos_per_million = 1000000000`; replace it with the applicable live rate.
Required fields are:

```toml
[providers.price]
snapshot_id = "YOUR_REVIEWED_PRICE_SNAPSHOT"
source_url = "https://YOUR_OFFICIAL_PRICE_SOURCE"
effective_date = "YYYY-MM-DD"
input_nanos_per_million = 1000000000          # illustrative; replace
cached_input_nanos_per_million = 1000000000   # illustrative; replace
output_nanos_per_million = 2000000000         # illustrative; replace
```

Use the full input rate for cached input when no lower rate has been verified.
Use a conservative rate covering the selected model, region, service tier and
the configured input/output range; tiered/long-context prices must not exceed
the reservation. A free-tier assumption cannot authorise an otherwise paid call.
Also set account/project spending controls where supported; provider budget
alerts may be advisory and do not replace local reservation checks.

The README example explicitly uses a conservative USD 2.50/million for both
input categories and USD 10/million for output, from the
[GPT-6.1 Sol documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
reviewed on 16 October 2026. That page lists USD 2 input, USD 0.10 cached reads,
USD 2.50 cache writes and USD 10 output for the documented short-context tier.
The profile covers cache-write liability through the higher input rate; it does
not model cache writes as a separate tariff. Its ledger costs are conservative
configured-rate estimates, not exact invoiced charges. The maximum per-call
reservation is USD 0.06144 (8,192 input at 2.50 plus 4,096 output at 10, per
million), within USD 0.25 and 20,000 tokens. A 2,048-output greeting reserves
less; the exact input reservation depends on serialised request bytes. Recheck
prices, tiers and account terms before use. An alias may change behaviour;
`model_revision` records that limitation without inventing an immutable snapshot.

Every attempted call atomically reserves worst-case money and input/output
tokens in SQLite before dispatch. The conservative input bound is serialised
UTF-8 request bytes plus 1,024 framing tokens, not a reported token count. Enable
only model/tokenisation contracts for which this bound and the chosen prices
are conservative. Actual usage settles the liability; a failed response without
usage retains the full reservation. Reported tokens exceeding approved bounds
block the round. Historical attempts with neither usage nor token reservations
cannot be silently treated as zero when a token cap is introduced.

## Run, fail over and resume

After configuring an enabled, priced provider and exporting its credential:

```sh
./run.sh provider-check --config state/api-providers.local.toml
# Example only; run after explicitly authorising this ceiling.
./run.sh provider-request --live --authorised-usd 0.25 \
  --config state/api-providers.local.toml --provider openai \
  --prompt 'Reply with a short greeting.' --max-output-tokens 2048 --round api-001
```

`--live` and an explicit positive `--authorised-usd` are both required for new
calls. The effective dollar cap is the smaller of the configured round limit and
that CLI cap. Reusing the same state directory, prompt, parameters and provider
reuses a verified completed result before checking credentials or circuit state.
Changing a prompt or model is a different request and can incur a charge.
New round names have separate limits; keep the same round to resume its budget.
State defaults to `state/provider-api/`; reports default to
`reports/provider-api/`. `costs.json/csv` expose reported input/output/reasoning,
the reserved token count, token liability upper bound and cost intervals.
`request.json` records completion or failure once the runner has been constructed;
costs are exported alongside it. Provider output is displayed as text and saved
as evidence; it is never executed, graded or automatically trained on.
Earlier configuration/replay-loading failures produce `last-failure.json` in
the output directory with a nonzero exit; they have no fabricated request or
cost report. An empty recording directory is refused before a ledger is opened.

| Condition | Behaviour |
| --- | --- |
| Disabled provider | Skip it; only explicitly configured fallbacks are considered. |
| Missing/empty/invalid credential | Record unavailable before any reservation or HTTP, then consider configured fallbacks. |
| DNS/TCP/TLS connection cannot be established | Settle the undispatched attempt at zero, then consider configured fallbacks. |
| 401/403, unavailable model, invalid request or exhausted paid quota | Preserve the failed observation; disable that provider for this runner and consider configured fallbacks. |
| 429, 408 or 5xx | Bounded retries with rate limits, jitter, Retry-After and circuit breaking; then configured failover. A Retry-After over 60 seconds defers the provider and fails over. |
| Timeout/stream loss after possible acceptance | Mark UNKNOWN, retain money/token reservation and stop the round; no automatic retry or fallback purchase. Reconcile with the provider first. |
| Money/token limit reached | Stop before transport; retain completed work and budget-stop evidence. |
| All routes unavailable | Nonzero exit with explicit failure; no synthetic success or silent local-model substitution. |
| `--offline` | Exact verified replay only; no HTTP, credential requirement or network fallback. Missing/corrupt recordings fail. |

For offline replay, copy the original checksum-named **real** recording files from
`state/provider-api/artifacts/` into a dedicated recording directory; keep their
bytes and filenames unchanged. Then use:

```sh
./run.sh provider-request --offline --config state/api-providers.local.toml \
  --provider openai --prompt 'Reply with a short greeting.' --max-output-tokens 2048 \
  --recordings recordings --state-dir state/provider-replay --round api-replay-001
```

Replay creates no new API charge, but historical usage/cost and conservative
budget checks remain visible. General inference recordings remain ineligible for
distillation. Do not clear a ledger or change a round name to bypass UNKNOWN
billing. Durable unique-response recovery runs under an exclusive process lease;
unresolved acceptance needs provider reconciliation. Configure a later explicit
run after fixing credentials/quota. Secret values are never put in configuration,
recordings, reports or request identities. Avoid shell tracing (`set -x`) when
exporting credentials; remove them from the shell with `unset` when finished.

Offline execution and loss of detectable internet connectivity preserve caches
and model weights. Teacher-search pruning requires confirmed Hub connectivity
before each deletion. No failed API connection is a reason to free cached models.

## Troubleshooting

| Observation | Meaning and remedy |
| --- | --- |
| Missing `state/api-providers.local.toml` / `No such file or directory` | The local file was never created, or the path differs. Run the README's complete `provider-init` command, then pass the same `--config` path to both commands. Current errors name the missing path and setup action. |
| Key present, provider disabled, model unresolved, budgets zero | You checked the shipped template or an unedited copy. Key export alone cannot enable a route. Initialise a complete local profile, or edit each field and add its reviewed price table. |
| Top-level `configuration_error: null`, readiness false | Structurally valid configuration can still forbid spending. Read `readiness_blockers`, including all zero money/token scopes. |
| `provider-init` refuses an existing file | The file was preserved. Check/edit it or choose a new path; do not delete a ledger or cached models to fix setup. |
| `credential_present: false` | Export the exact variable in the same shell that launches `run.sh`. Interactive shell exports may be absent from an IDE, service or this agent's environment. Never paste a key into a command argument or report. |
| 401/403/404 or unavailable model | Check the API product/key permissions, billing, model entitlement, region and exact model ID. Consumer subscription access does not grant API access. Review recorded failures and remaining liabilities before a new request. |
| Incomplete response / no greeting at 256 tokens | The cap includes reasoning. Use the documented 2,048-token request within the configured cap. The previous failed attempt remains billable if the provider reported usage. |
| Money/token `BUDGET STOP` | Reservations include input plus maximum output, previous failures and unresolved attempts. Inspect `costs.json` and limits; do not reset state or round names to evade liabilities. |
| UNKNOWN acceptance / timeout | Preserve the reservation and reconcile remotely. A response might have been purchased even when no text arrived locally. Do not retry automatically. |
| Offline replay has no recording | There is no exact real recording for that model/prompt/parameters. Run no network fallback; retain the failure and supply the original verified recording. Fixture tests do not create live evidence. |
