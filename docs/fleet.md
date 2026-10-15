# Durable teacher acquisition

The runner and ledger are native Rust. SQLite transactions and filesystem synchronisation run on Tokio blocking workers; HTTP, bounded task concurrency, rate windows and backoff use asynchronous Tokio operations. The three provider businesses are configured in `config/fleet.toml`. All remain disabled because no live model/account rights decision, current price snapshot, credential or authorised spending instruction has been supplied. No provider call, usage or acquisition cost has been fabricated.

`FleetRunner::complete_on(round_id, provider_id, request)` buys one intended sample, with configured failover. `complete` chooses the first enabled provider. `run_many` bounds parallel task work. Collecting best of three should call `complete_on` once for each independently owned business and send the returned candidates to the trusted grader; failover identity is retained. The host must call `require_three_businesses` when claiming assessment collection.

A semantic request hash covers the complete messages and generation parameters, task/specification/source identity, intended sample slot, endpoint, provider business, exact model and model revision policy. Price is attached to each attempt but excluded from the semantic key: a changed price cannot cause finished work to be purchased again. Different intended sample slots remain different requests.

Before every transport attempt, an IMMEDIATE SQLite transaction checks round, task, provider and model limits against the sum of all settled upper costs and unresolved reservations. Nanodollars are integers; prices are nanodollars per million tokens and multiplication uses checked 128-bit arithmetic with upward rounding. Cached input is counted once, and reasoning is included in the billable output cap. Without an exact tokenizer, the ordinary chat request's UTF-8 byte count plus 1,024 tokens is deliberately conservative. This fallback assumes a byte-based tokenizer and bounded ordinary chat framing; a provider with different semantics requires its own verified bound before enabling it.

A completed response is published to a SHA-256 addressed file, fsynced, and its directory fsynced before the transaction marks completion. Reading a cached result verifies the file hash and request identity. On explicit `recover_interrupted`, a unique valid durable response published before the interruption restores the observed completed or failed result without another purchase. Missing, invalid or ambiguous evidence leaves the attempt UNKNOWN, retaining its worst-case cost and blocking automatic repurchase. Recovery must run only with exclusive ownership of the orchestration database. The ledger prevents duplicate purchases among concurrent workers, but remote exactly-once billing cannot be guaranteed. Manual reconciliation support requires provider evidence and is deliberately not replaced by an unsafe 'release reservation' command.

An HTTP error with no usage retains a historical cost interval from zero to the entire reservation. That error response is durable evidence of failure and can be retried or failed over; a timeout/disconnect after dispatch has ambiguous remote acceptance and stops automatically. Authentication/configuration errors are not retried. 429 respects numeric and HTTP-date Retry-After, all retries traverse the rate and budget gates, and a shared provider circuit breaker bounds failures. A delay above 60 seconds causes failover, without retrying the throttled provider early. Empty content, malformed JSON, absent usage, tool calls and non-stop/truncated output never become successful completions.

The live transport only sends ordinary OpenAI-compatible text chat requests over HTTPS, rejects redirects, caps response bytes and stores only an allowlist of non-secret response headers. It contains no web search or tools. `LiveTransport::new(true)` requires the caller to have explicit spending authority; the default roster also fails closed on missing prices, rights evidence and output-cap semantics. Environment variable names are logged, never their values.

Offline replay uses `ReplayTransport::load(recording_directory)`. Files must be `<sha256>.json`, match their checksum and contain an exact request key and attempt ordinal. There is no HTTP client or network fallback in this transport. Recorded pricing supplies historical acquisition figures; replay's new API spend is zero. Labelled fixtures require an explicit alternative policy and cannot acquire the real-recording flag required by curation. Replay is not evidence that an API call occurred: the original acquisition and applicable invoice/request evidence remain necessary.

`Ledger::export_costs` writes CSV, Markdown and JSON at attempt/task/provider/model granularity. Missing measurements remain absent; an empty ledger explicitly says no acquisition evidence exists. Historical uncertainty and new API spending are separate columns. CPU/GPU hours, electricity, storage and serving require independent resource measurements and are not invented by this module. A cost-per-point calculation is meaningful only with the gate's measured improvement and the complete round's measured cost.

The fleet invariant tests use conspicuously labelled synthetic responses to test transactional claims, restart reuse, UNKNOWN retention, pre-dispatch budget stops, failover, canonical keys, integer pricing and response rejection. They do not satisfy the three-real-provider requirement. The subprocess drills likewise distinguish software assurance from missing real acquisition evidence.

To enable a provider, preserve dated applicable terms and selected model licence, hash the evidence, add the reviewed permission decision, pin the actual model/revision policy and price schedule, verify that its output cap includes all billable reasoning, supply the named credential outside candidate containers, and obtain an explicit maximum spend instruction. An ordinary price table has these fields:

```toml
[providers.price]
snapshot_id = "dated-reviewed-price-document-id"
source_url = "https://provider.example/pricing"
effective_date = "YYYY-MM-DD"
input_nanos_per_million = 1000000000
cached_input_nanos_per_million = 500000000
output_nanos_per_million = 4000000000
```

The numbers in this syntax example are illustrative, not a quote or measured cost.

`Ledger::attempt_proof(attempt_id)` exports the actual trusted transaction as an immutable `AttemptProof`: round/task/source/request identities, exact provider business/model/revision, ordinal/state, original request and price JSON, response hash, usage/cost, replay flag and timestamps. Collection writes and hashes this proof for each chosen run. Curation checks the proof against the real response and controller run, copies it into the dataset, and checks it again immediately before training. An arbitrary nonempty UUID therefore cannot substitute for a traceable completed attempt. Proof export reads the database and verifies any referenced response blob; it does not author a replacement attempt or claim a remote call occurred.
