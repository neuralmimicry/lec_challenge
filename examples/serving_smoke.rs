//! Actual local endpoint QA, using the pinned base only. This is neither a trained
//! candidate nor proof of statistical improvement or a completed production canary.
use anyhow::{Context, Result, ensure};
use lec_distil::{
    model::{Generation, GenerationRequest},
    serve::{self, Health, ServeConfig, VersionedEndpoint},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

struct ChildGuard(std::process::Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Serialize)]
struct SmokeReport {
    evidence_kind: String,
    health: Health,
    generation: Generation,
    oversized_body_status: u16,
    excessive_output_status: u16,
    pressure_status_counts: BTreeMap<u16, usize>,
    bounded_queue_rejected: bool,
    deadline_enforced: bool,
    initial_routing: serve::RoutingManifest,
    evaluation_routing: serve::RoutingManifest,
    evaluation_generation: Generation,
    rollback_routing: serve::RoutingManifest,
    elapsed_seconds: f64,
    limitations: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let started = Instant::now();
    let base = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("offline_bundle/base"))
        .canonicalize()?;
    let binary = std::env::current_exe()?
        .parent()
        .context("example directory")?
        .parent()
        .context("target profile directory")?
        .join("lec-distil");
    ensure!(
        binary.is_file(),
        "build the lec-distil binary alongside this example"
    );
    let reserved = std::net::TcpListener::bind("127.0.0.1:0")?;
    let addr = reserved.local_addr()?;
    drop(reserved);
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("serving.toml");
    let config = ServeConfig {
        address: addr.to_string(),
        queue_capacity: 1,
        maximum_request_bytes: 1024,
        maximum_output_tokens: 96,
        timeout_seconds: 10,
    };
    std::fs::write(&config_path, toml::to_string(&config)?)?;
    std::fs::create_dir_all("reports")?;
    let stderr = std::fs::File::create("reports/serving-smoke-process.log")?;
    let mut child = ChildGuard(
        Command::new(binary)
            .args(["serve", "--base"])
            .arg(&base)
            .arg("--config")
            .arg(&config_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr)
            .spawn()?,
    );
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(15))
        .build()?;
    let url = format!("http://{addr}/");
    let startup = Instant::now() + Duration::from_secs(90);
    let health: Health = loop {
        if let Some(status) = child.0.try_wait()? {
            anyhow::bail!("real model server exited before health: {status}");
        }
        if let Ok(response) = client.get(format!("{url}health")).send().await
            && response.status().is_success()
        {
            break response.json().await?;
        }
        ensure!(Instant::now() < startup, "model server startup timed out");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    ensure!(
        health.healthy
            && health.checksum_validated
            && health.production_eligible
            && health.immutable,
        "real health invariant failed"
    );
    let request=GenerationRequest{prompt:"Write a Rust function add(a: i64, b: i64) -> i64. Return only the function source code.".into(),max_new_tokens:96,seed:1701,temperature:0.,top_p:1.,timeout_seconds:10};
    let generation: Generation = client
        .post(format!("{url}v1/generate"))
        .json(&request)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    ensure!(
        generation.complete
            && !generation.completion.trim().is_empty()
            && generation.checkpoint_id == health.checkpoint_id,
        "actual endpoint generation failed"
    );
    let oversized_body_status = client
        .post(format!("{url}v1/generate"))
        .header("Content-Type", "application/json")
        .body("x".repeat(2048))
        .send()
        .await?
        .status()
        .as_u16();
    ensure!(
        oversized_body_status == 413,
        "request body cap did not reject oversized input"
    );
    let mut excessive = request.clone();
    excessive.max_new_tokens = 97;
    let excessive_output_status = client
        .post(format!("{url}v1/generate"))
        .json(&excessive)
        .send()
        .await?
        .status()
        .as_u16();
    ensure!(
        excessive_output_status == 400,
        "output token cap did not reject excessive request"
    );
    let mut slow = request.clone();
    slow.prompt="Write a complete Rust module with a binary search tree, insert, remove, lookup, iteration, and comments explaining every method. Include every function and no abbreviations.".into();
    slow.timeout_seconds = 1;
    let requests = (0..12).map(|_| {
        let client = client.clone();
        let url = format!("{url}v1/generate");
        let payload = slow.clone();
        async move {
            Ok::<_, anyhow::Error>(
                client
                    .post(url)
                    .json(&payload)
                    .send()
                    .await?
                    .status()
                    .as_u16(),
            )
        }
    });
    let mut pressure_status_counts = BTreeMap::new();
    for result in futures_util::future::join_all(requests).await {
        *pressure_status_counts.entry(result?).or_insert(0) += 1;
    }
    let bounded_queue_rejected = pressure_status_counts.get(&503).copied().unwrap_or(0) > 0;
    let deadline_enforced = pressure_status_counts.get(&504).copied().unwrap_or(0) > 0
        || pressure_status_counts.get(&422).copied().unwrap_or(0) > 0;
    ensure!(
        bounded_queue_rejected && deadline_enforced,
        "real endpoint queue/deadline pressure assertions failed: {pressure_status_counts:?}"
    );
    // Let the cooperative token-boundary cancellation drain before routing checks.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let endpoint = VersionedEndpoint {
        url: url.clone(),
        checkpoint_id: health.checkpoint_id.clone(),
    };
    let routing = temp.path().join("routing.json");
    let initial_routing = serve::initialise_routing(&routing, endpoint.clone()).await?;
    let evaluation_routing = serve::stage_evaluation(&routing, endpoint).await?;
    ensure!(
        evaluation_routing.canary_percent == 0,
        "evaluation staging changed production traffic"
    );
    let evaluation_generation =
        serve::route_generate(&routing, "measured-base-evaluation-only", true, request).await?;
    let rollback_routing =
        serve::rollback(&routing, "controlled local base evaluation-alias rollback")?;
    ensure!(
        rollback_routing.canary_percent == 0
            && rollback_routing.incumbent.checkpoint_id == health.checkpoint_id
            && rollback_routing.generation > evaluation_routing.generation,
        "atomic rollback invariant failed"
    );
    let report=SmokeReport{evidence_kind:"actual_pinned_base_endpoint_qa_not_trained_candidate_canary".into(),health,generation,oversized_body_status,excessive_output_status,pressure_status_counts,bounded_queue_rejected,deadline_enforced,initial_routing,evaluation_routing,evaluation_generation,rollback_routing,elapsed_seconds:started.elapsed().as_secs_f64(),limitations:vec!["The evaluation endpoint intentionally aliases the same immutable base tensor; no trained candidate or quality gain is claimed.".into(),"No SHIP gate or 5% candidate canary was exercised because rights-cleared provider training/evaluation artifacts are absent.".into(),"Inference cancellation is cooperative at token boundaries; a matrix multiplication already executing is not pre-empted.".into(),"Generated Rust output was not executed outside the required sandbox.".into()]};
    let bytes = serde_json::to_vec_pretty(&report)?;
    std::fs::write("reports/serving-smoke.json", &bytes)?;
    println!("{}", String::from_utf8(bytes)?);
    Ok(())
}
