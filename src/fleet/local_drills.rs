//! Explicitly labelled process-crash and provider-fault software-assurance drills.
use super::*;
fn drill_provider(id: &str) -> ProviderConfig {
    ProviderConfig {
        api: Default::default(),
        inference_only: false,
        id: id.into(),
        business_id: format!("{id}-fixture-business"),
        enabled: true,
        endpoint: format!("https://{id}.invalid/chat/completions"),
        api_key_environment: "UNUSED_LABELLED_FIXTURE".into(),
        model: "LABELLED_FIXTURE_MODEL".into(),
        model_revision: "LABELLED_FIXTURE_REVISION".into(),
        timeout_seconds: 5,
        requests_per_minute: 100,
        tokens_per_minute: 1_000_000,
        concurrency: 2,
        max_input_tokens: 8192,
        max_output_tokens: 2048,
        output_cap_parameter: "max_tokens".into(),
        output_cap_includes_reasoning: true,
        max_response_bytes: 65536,
        retries: 1,
        circuit_breaker_failures: 4,
        circuit_breaker_seconds: 1,
        failover: vec![],
        permission: PermissionDecision {
            approved_for_output_training: true,
            decision_id: "LABELLED FIXTURE ONLY; NOT RIGHTS EVIDENCE".into(),
            reviewed_at: "fixture".into(),
            evidence_sha256: "f".repeat(64),
            model_licence: "FIXTURE ONLY".into(),
        },
        price: Some(PriceSnapshot {
            snapshot_id: "FIXTURE ASSUMPTION; NOT A QUOTE".into(),
            source_url: "https://fixture.invalid/pricing".into(),
            effective_date: "fixture".into(),
            input_nanos_per_million: 1_000_000_000,
            cached_input_nanos_per_million: 1_000_000_000,
            output_nanos_per_million: 2_000_000_000,
        }),
    }
}

fn drill_request() -> FleetRequest {
    FleetRequest {
        task_id: "labelled-fixture-task".into(),
        source_id: "labelled-fixture-source".into(),
        specification_sha256: sha256(b"labelled fixture specification"),
        system_prompt: "LABELLED SOFTWARE-ASSURANCE FIXTURE ONLY".into(),
        prompt: "Return a Rust integer increment function.".into(),
        sample_slot: 0,
        generation: GenerationParameters {
            max_output_tokens: 128,
            ..Default::default()
        },
    }
}

struct DrillFixtureTransport {
    directory: std::path::PathBuf,
    statuses: Mutex<VecDeque<u16>>,
    unknown_window: bool,
}
impl DrillFixtureTransport {
    fn new(directory: impl AsRef<Path>, statuses: Vec<u16>, unknown_window: bool) -> Self {
        Self {
            directory: directory.as_ref().to_owned(),
            statuses: Mutex::new(statuses.into()),
            unknown_window,
        }
    }
}
#[async_trait]
impl Transport for DrillFixtureTransport {
    fn is_replay(&self) -> bool {
        true
    }
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        let directory = self.directory.clone();
        let provider = request.provider.id.clone();
        let key = request.request_key.clone();
        let ordinal = request.attempt_ordinal;
        tokio::task::spawn_blocking(move||->Result<()>{use std::io::Write;fs::create_dir_all(&directory)?;let mut file=std::fs::OpenOptions::new().create(true).append(true).open(directory.join("transport-calls.jsonl"))?;writeln!(file,"{}",json!({"origin":"synthetic_fixture","provider_id":provider,"request_key":key,"attempt_ordinal":ordinal}))?;file.sync_all()?;Ok(())}).await.map_err(|e|TransportFailure::BeforeDispatch(e.to_string()))?.map_err(|e|TransportFailure::BeforeDispatch(e.to_string()))?;
        if self.unknown_window {
            let directory = self.directory.clone();
            tokio::task::spawn_blocking(move || write_ready(&directory))
                .await
                .map_err(|e| TransportFailure::Unknown(e.to_string()))?
                .map_err(|e| TransportFailure::Unknown(e.to_string()))?;
            std::future::pending::<()>().await;
            unreachable!();
        }
        let status = self.statuses.lock().await.pop_front().unwrap_or(200);
        let mut headers = BTreeMap::new();
        if status == 429 {
            headers.insert("retry-after".into(), "0".into());
        }
        let raw_body = if status == 200 {
            json!({"choices":[{"finish_reason":"stop","message":{"content":"fn solve(x: i64) -> i64 { x + 1 }"}}],"usage":{"prompt_tokens":40,"completion_tokens":15}}).to_string()
        } else {
            json!({"error":{"message":"LABELLED DETERMINISTIC FIXTURE; NOT A PROVIDER ERROR"}})
                .to_string()
        };
        Recording::from_response(
            request,
            TransportResponse {
                status,
                headers,
                raw_body,
                elapsed_millis: 0,
            },
            "synthetic_fixture",
        )
        .map_err(|e| TransportFailure::BeforeDispatch(e.to_string()))
    }
}

fn write_ready(directory: &Path) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(directory.join("ready"))?;
    file.write_all(b"LABELLED FIXTURE READY\n")?;
    file.sync_all()?;
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
}
fn drill_runner(
    directory: &Path,
    transport: Arc<dyn Transport>,
    budget: i64,
    two_providers: bool,
) -> Result<FleetRunner> {
    let mut providers = vec![drill_provider("fixture-a")];
    if two_providers {
        providers[0].failover = vec!["fixture-b".into()];
        providers.push(drill_provider("fixture-b"));
    }
    FleetRunner::new(
        FleetConfig {
            max_parallel_tasks: 2,
            budgets: BudgetLimits {
                round_nanos: budget,
                task_nanos: budget,
                provider_nanos: budget,
                model_nanos: budget,
                tokens: Default::default(),
            },
            providers,
        },
        Ledger::open(directory.join("ledger.db"), directory.join("artifacts"))?,
        transport,
    )
}
fn persisted_call_count(directory: &Path) -> Result<usize> {
    let path = directory.join("transport-calls.jsonl");
    if !path.exists() {
        return Ok(0);
    }
    Ok(fs::read_to_string(path)?.lines().count())
}

/// Hidden CLI child used solely by the real-SIGKILL software-assurance drill.
pub async fn drill_child(directory: &Path, unknown_window: bool) -> Result<()> {
    if directory.join("real-replay-plan.json").exists() {
        return recorded_child(directory, unknown_window).await;
    }
    let transport = Arc::new(DrillFixtureTransport::new(
        directory,
        vec![],
        unknown_window,
    ));
    let runner = drill_runner(directory, transport, 100_000_000, false)?;
    runner.complete("fixture-round", drill_request()).await?;
    let directory = directory.to_owned();
    tokio::task::spawn_blocking(move || write_ready(&directory)).await??;
    std::future::pending::<()>().await;
    Ok(())
}

#[derive(Clone, Serialize, Deserialize)]
struct RecordedDrillPlan {
    config: FleetConfig,
    recording_directory: std::path::PathBuf,
    request: FleetRequest,
    preferred_provider: String,
}

struct CountingReplay {
    inner: Arc<dyn Transport>,
    directory: std::path::PathBuf,
    unknown_window: bool,
}
#[async_trait]
impl Transport for CountingReplay {
    fn is_replay(&self) -> bool {
        true
    }
    fn preflight(&self, request: &PreparedRequest) -> Result<()> {
        self.inner.preflight(request)
    }
    async fn execute(
        &self,
        request: &PreparedRequest,
    ) -> std::result::Result<Recording, TransportFailure> {
        let record = self.inner.execute(request).await?;
        let directory = self.directory.clone();
        let key = request.request_key.clone();
        let provider = request.provider.id.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::io::Write;
            fs::create_dir_all(&directory)?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(directory.join("transport-calls.jsonl"))?;
            writeln!(
                file,
                "{}",
                json!({"origin":"replayed_real_provider","request_key":key,"provider":provider})
            )?;
            file.sync_all()?;
            Ok(())
        })
        .await
        .map_err(|e| TransportFailure::BeforeDispatch(e.to_string()))?
        .map_err(|e| TransportFailure::BeforeDispatch(e.to_string()))?;
        if self.unknown_window {
            let directory = self.directory.clone();
            tokio::task::spawn_blocking(move || write_ready(&directory))
                .await
                .map_err(|e| TransportFailure::Unknown(e.to_string()))?
                .map_err(|e| TransportFailure::Unknown(e.to_string()))?;
            std::future::pending::<()>().await;
        }
        Ok(record)
    }
}

async fn recorded_child(directory: &Path, unknown_window: bool) -> Result<()> {
    let plan: RecordedDrillPlan =
        serde_json::from_slice(&fs::read(directory.join("real-replay-plan.json"))?)?;
    let transport = Arc::new(CountingReplay {
        inner: Arc::new(ReplayTransport::load(&plan.recording_directory)?),
        directory: directory.to_owned(),
        unknown_window,
    });
    let runner = FleetRunner::new(
        plan.config,
        Ledger::open(directory.join("ledger.db"), directory.join("artifacts"))?,
        transport,
    )?;
    runner
        .complete_on(
            "recorded-drill-round",
            &plan.preferred_provider,
            plan.request,
        )
        .await?;
    let directory = directory.to_owned();
    tokio::task::spawn_blocking(move || write_ready(&directory)).await??;
    std::future::pending::<()>().await;
    Ok(())
}

/// Repeat the crash/failover drill against original real acquisition recordings.
/// Faults are injected and clearly labelled; successful fallback is a real replay.
pub async fn run_recorded_drills(
    work_dir: &Path,
    config: FleetConfig,
    recording_directory: &Path,
) -> Result<Value> {
    config.validate()?;
    config.require_three_businesses()?;
    let replay = ReplayTransport::load(recording_directory)?;
    let mut businesses = std::collections::HashSet::new();
    let mut successful = Vec::new();
    for entry in fs::read_dir(recording_directory)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let record: Recording = serde_json::from_slice(&fs::read(path)?)?;
        record.validate_identity()?;
        businesses.insert(record.business_id.clone());
        if record.attempt_ordinal == 1 && parse_response(&record.response).is_ok() {
            successful.push(record);
        }
    }
    ensure!(
        businesses.len() >= 3,
        "real recordings from three independent provider businesses are required"
    );
    successful.sort_by(|a, b| a.request_key.cmp(&b.request_key));
    let recording = successful
        .into_iter()
        .find(|r| {
            config.providers.iter().any(|p| {
                p.enabled
                    && p.id == r.provider_id
                    && p.business_id == r.business_id
                    && prepare(p, &r.request).is_ok_and(|p| p.request_key == r.request_key)
            })
        })
        .context(
            "no successful first-attempt real recording matches the frozen configured roster",
        )?;
    let actual_provider = config
        .providers
        .iter()
        .find(|p| p.id == recording.provider_id)
        .context("recorded provider is not configured")?;
    let error_provider = config
        .providers
        .iter()
        .find(|p| p.enabled && p.business_id != actual_provider.business_id)
        .context("no independently owned eligible failure-injection provider")?
        .id
        .clone();
    let run_dir = work_dir.join(format!("real-replay-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&run_dir)?;
    let plan = RecordedDrillPlan {
        config: config.clone(),
        recording_directory: recording_directory.canonicalize()?,
        request: recording.request.clone(),
        preferred_provider: actual_provider.id.clone(),
    };
    let durable = run_dir.join("sigkill-durable");
    fs::create_dir_all(&durable)?;
    crate::artefact::write_json(&durable.join("real-replay-plan.json"), &plan)?;
    kill_fixture_child(&durable, false).await?;
    let before = persisted_call_count(&durable)?;
    ensure!(
        before == 1,
        "real replay child dispatched an unexpected number of acquisitions"
    );
    let counted = Arc::new(CountingReplay {
        inner: Arc::new(ReplayTransport::load(recording_directory)?),
        directory: durable.clone(),
        unknown_window: false,
    });
    let runner = FleetRunner::new(
        config.clone(),
        Ledger::open(durable.join("ledger.db"), durable.join("artifacts"))?,
        counted,
    )?;
    ensure!(
        runner.ledger.recover_interrupted()? == 0,
        "finished real replay was recovered as unknown"
    );
    let resumed = runner
        .complete_on(
            "recorded-drill-round",
            &plan.preferred_provider,
            plan.request.clone(),
        )
        .await?;
    ensure!(
        resumed.cached
            && resumed.real_provider_recording
            && persisted_call_count(&durable)? == before,
        "finished real recording was repurchased after restart"
    );
    let unknown = run_dir.join("sigkill-response-window");
    fs::create_dir_all(&unknown)?;
    crate::artefact::write_json(&unknown.join("real-replay-plan.json"), &plan)?;
    kill_fixture_child(&unknown, true).await?;
    let counted = Arc::new(CountingReplay {
        inner: Arc::new(ReplayTransport::load(recording_directory)?),
        directory: unknown.clone(),
        unknown_window: false,
    });
    let runner = FleetRunner::new(
        config.clone(),
        Ledger::open(unknown.join("ledger.db"), unknown.join("artifacts"))?,
        counted,
    )?;
    ensure!(
        runner.ledger.recover_interrupted()? == 1,
        "response window was not recovered as UNKNOWN"
    );
    ensure!(
        runner
            .complete_on(
                "recorded-drill-round",
                &plan.preferred_provider,
                plan.request.clone()
            )
            .await
            .is_err()
            && persisted_call_count(&unknown)? == 1,
        "UNKNOWN response window was retried"
    );
    let reservation_retained = runner.ledger.costs(None)?[0].historical_upper_nanos > 0;
    ensure!(reservation_retained, "UNKNOWN liability was released");
    let faults = run_dir.join("error-rate-limit-failover");
    fs::create_dir_all(&faults)?;
    let mut fault_config = config.clone();
    let injected = fault_config
        .providers
        .iter_mut()
        .find(|p| p.id == error_provider)
        .context("fault provider disappeared")?;
    injected.retries = 1;
    injected.circuit_breaker_failures = 3;
    injected.failover = vec![actual_provider.id.clone()];
    let replay = Arc::new(replay);
    let transport = Arc::new(FaultInjectionTransport::new(
        replay.clone(),
        error_provider.clone(),
        vec![500, 429],
    ));
    let runner = FleetRunner::new(
        fault_config,
        Ledger::open(faults.join("ledger.db"), faults.join("artifacts"))?,
        transport,
    )?;
    let completion = runner
        .complete_on(
            "recorded-drill-round",
            &error_provider,
            recording.request.clone(),
        )
        .await?;
    let attempts = runner.ledger.costs(None)?;
    ensure!(
        completion.real_provider_recording
            && completion.provider_id == actual_provider.id
            && attempts.len() == 3
            && replay.call_count() == 1,
        "500/429 failover did not complete through exactly one real recorded response"
    );
    ensure!(
        attempts.iter().all(|a| a.new_api_upper_nanos == 0),
        "offline drill created new provider spending"
    );
    Ok(
        json!({"evidence_kind":"real_recorded_replay_with_labelled_fault_injections","recorded_businesses":businesses,"artifact_directory":run_dir,"sigkill_resume":{"passed":true,"signal":"SIGKILL","transport_calls_before":before,"transport_calls_after":persisted_call_count(&durable)?,"cached":resumed.cached,"response_artifact_sha256":resumed.response_artifact_sha256},"unknown_window":{"passed":true,"window":"injected_after_real_replay_acceptance_before_durable_response","reservation_retained":reservation_retained},"error_then_rate_limit":{"passed":true,"injected_statuses":[500,429],"injected_provider":error_provider,"completed_provider":completion.provider_id,"completed_business":completion.business_id,"real_response_artifact_sha256":completion.response_artifact_sha256,"attempts":attempts}}),
    )
}

async fn kill_fixture_child(directory: &Path, unknown_window: bool) -> Result<()> {
    fs::create_dir_all(directory)?;
    let mut command = tokio::process::Command::new(std::env::current_exe()?);
    command
        .arg("__fleet-drill-child")
        .arg(directory)
        .kill_on_drop(true)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    if unknown_window {
        command.arg("--unknown-window");
    }
    let mut child = command
        .spawn()
        .context("start real fixture child orchestrator")?;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        if directory.join("ready").exists() {
            break;
        }
        if let Some(status) = child.try_wait()? {
            bail!(
                "fixture child exited before durable readiness: {status}; hidden CLI subcommand may be unavailable"
            );
        }
        ensure!(
            tokio::time::Instant::now() < deadline,
            "fixture child did not reach durable readiness"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    child.start_kill()?;
    let status = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .context("SIGKILL child did not terminate within the bounded wait")??;
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        ensure!(
            status.signal() == Some(9),
            "child was not actually terminated by SIGKILL"
        );
    }
    #[cfg(not(unix))]
    {
        ensure!(!status.success(), "child did not terminate unsuccessfully");
    }
    Ok(())
}

/// Exercises production orchestration with explicitly labelled transport fixtures.
/// This is never represented as the missing three-provider acquisition evidence.
pub async fn run_local_drills(work_dir: &Path) -> Result<Value> {
    let run_dir = work_dir.join(format!("fleet-fixtures-{}", uuid::Uuid::new_v4()));
    fs::create_dir_all(&run_dir)?;
    let durable = run_dir.join("sigkill-durable");
    kill_fixture_child(&durable, false).await?;
    let before = persisted_call_count(&durable)?;
    ensure!(
        before == 1,
        "child did not persist exactly one fixture transport call"
    );
    let transport = Arc::new(DrillFixtureTransport::new(&durable, vec![], false));
    let runner = drill_runner(&durable, transport, 100_000_000, false)?;
    ensure!(
        runner.ledger.recover_interrupted()? == 0,
        "completed work became unknown"
    );
    let resumed = runner.complete("fixture-round", drill_request()).await?;
    ensure!(
        resumed.cached && persisted_call_count(&durable)? == before,
        "restart repurchased finished fixture work"
    );
    let unknown = run_dir.join("sigkill-acceptance-window");
    kill_fixture_child(&unknown, true).await?;
    let transport = Arc::new(DrillFixtureTransport::new(&unknown, vec![], false));
    let runner = drill_runner(&unknown, transport, 100_000_000, false)?;
    ensure!(
        runner.ledger.recover_interrupted()? == 1,
        "interrupted acceptance window was not recovered as UNKNOWN"
    );
    let error = runner
        .complete("fixture-round", drill_request())
        .await
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("UNKNOWN") && persisted_call_count(&unknown)? == 1,
        "unknown acceptance was automatically retried"
    );
    ensure!(
        runner.ledger.costs(None)?[0].historical_upper_nanos > 0,
        "unknown reservation was released"
    );
    let faults = run_dir.join("error-rate-limit-failover");
    fs::create_dir_all(&faults)?;
    let transport = Arc::new(DrillFixtureTransport::new(&faults, vec![500, 429], false));
    let runner = drill_runner(&faults, transport, 100_000_000, true)?;
    let result = runner.complete("fixture-round", drill_request()).await?;
    let rows = runner.ledger.costs(None)?;
    ensure!(
        result.provider_id == "fixture-b" && rows.len() == 3 && persisted_call_count(&faults)? == 3,
        "500/429 sequence did not finish through independent configured failover"
    );
    ensure!(
        rows.iter().filter(|r| r.state == "failed").count() == 2,
        "failed attempts missing from durable costs"
    );
    let stopped = run_dir.join("budget-stop");
    fs::create_dir_all(&stopped)?;
    let transport = Arc::new(DrillFixtureTransport::new(&stopped, vec![], false));
    let runner = drill_runner(&stopped, transport, 1, false)?;
    let error = runner
        .complete("fixture-round", drill_request())
        .await
        .unwrap_err()
        .to_string();
    ensure!(
        error.contains("BUDGET STOP")
            && persisted_call_count(&stopped)? == 0
            && runner.ledger.costs(None)?.is_empty(),
        "budget stop dispatched or spent"
    );
    let restarted = Ledger::open(stopped.join("ledger.db"), stopped.join("artifacts"))?;
    let budget_events = restarted.events(Some("fixture-round"))?;
    ensure!(
        budget_events
            .iter()
            .any(|event| event.kind == "budget_stop"),
        "budget-stop progress was not durable after reopening the ledger"
    );
    Ok(
        json!({"evidence_kind":"labelled_software_assurance_fixtures","assessment_real_recording_drill_complete":false,"missing":"three independently owned real provider recordings are absent","artifact_directory":run_dir,"sigkill_resume":{"passed":true,"signal":"SIGKILL","transport_calls_before":before,"transport_calls_after":before,"cached":resumed.cached},"unknown_window":{"passed":true,"reservation_retained":true,"transport_calls":1},"error_then_rate_limit":{"passed":true,"injected_statuses":[500,429],"completed_provider":"fixture-b","attempts":rows},"budget_stop":{"passed":true,"transport_calls":0,"progress_saved":true,"reopened_ledger_events":budget_events}}),
    )
}
