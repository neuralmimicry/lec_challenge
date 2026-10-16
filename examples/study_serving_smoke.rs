//! Real exploratory-student HTTP inference and a selected training example.
//! This is operational QA, never a held-out quality or promotion result.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact::{self, Artefact},
    model::{Generation, GenerationRequest},
    sandbox::{self, SandboxConfig},
    serve::{self, Health, ServeConfig, VersionedEndpoint},
    tasks::{self, Split},
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    engine: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value = "offline_bundle/base")]
    base: PathBuf,
    #[arg(long, default_value = "ledger.prefix")]
    task: String,
    #[arg(long)]
    output: PathBuf,
}
struct OwnedServer(Child);
impl Drop for OwnedServer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn capture(path: &Path) -> Result<Artefact> {
    let root = std::env::current_dir()?.canonicalize()?;
    Artefact::capture(&root, path.canonicalize()?.strip_prefix(&root)?.to_owned())
}

async fn run(args: &Args) -> Result<Value> {
    let started = Instant::now();
    let task = tasks::corpus()
        .into_iter()
        .find(|task| task.id == args.task)
        .context("unknown example task")?;
    ensure!(
        task.split == Split::Train,
        "student endpoint example is training-only"
    );
    let manifest_path = args.checkpoint.join("manifest.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let expected = manifest["adapter_sha256"]
        .as_str()
        .context("real adapter identity absent")?;
    let checkpoint = capture(&args.checkpoint.join("adapter.safetensors"))?;
    ensure!(
        checkpoint.sha256 == expected,
        "endpoint checkpoint hash changed"
    );
    let engine = capture(&args.engine)?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);
    let config = ServeConfig {
        address: address.to_string(),
        queue_capacity: 1,
        maximum_request_bytes: 8192,
        maximum_output_tokens: 512,
        timeout_seconds: 120,
    };
    let config_path = args.output.join("serving.toml");
    fs::write(&config_path, toml::to_string(&config)?)?;
    let stdout = fs::File::create(args.output.join("server.stdout.log"))?;
    let stderr = fs::File::create(args.output.join("server.stderr.log"))?;
    let mut command = Command::new(&args.engine);
    command
        .args([
            "serve",
            "--local-study",
            "--device",
            "cpu",
            "--dtype",
            "f32",
            "--checkpoint",
        ])
        .arg(&args.checkpoint)
        .arg("--base")
        .arg(&args.base)
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: only async-signal-safe functions run in the child before exec.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::other("endpoint supervisor exited"));
                }
                Ok(())
            });
        }
    }
    let mut child = OwnedServer(command.spawn()?);
    eprintln!("study-serving-qa: loading exact student checkpoint on CPU/F32");
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(150))
        .build()?;
    let url = format!("http://{address}");
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut heartbeat = Instant::now();
    let health: Health = loop {
        ensure!(
            child.0.try_wait()?.is_none(),
            "student endpoint exited during startup; see retained process logs"
        );
        if let Ok(response) = client.get(format!("{url}/health")).send().await
            && response.status().is_success()
        {
            break response.json().await?;
        }
        ensure!(
            Instant::now() < deadline,
            "student endpoint startup deadline exceeded"
        );
        if heartbeat.elapsed() >= Duration::from_secs(15) {
            eprintln!("study-serving-qa: validating/loading student; startup remains pending");
            heartbeat = Instant::now();
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    ensure!(
        health.healthy
            && health.checksum_validated
            && health.immutable
            && !health.production_eligible
            && health.checkpoint_id == expected
            && health.execution["actual_device"] == "Cpu"
            && health.execution["actual_dtype"] == "F32"
            && health.execution["cpu_fallback"] == false,
        "student endpoint identity/runtime/evaluation-only health differs"
    );
    artefact::write_json(&args.output.join("health.json"), &health)?;
    let request = GenerationRequest {
        prompt: task.prompt(),
        max_new_tokens: 512,
        seed: 1729,
        temperature: 0.2,
        top_p: 0.95,
        timeout_seconds: 120,
    };
    artefact::write_json(&args.output.join("request.json"), &request)?;
    eprintln!(
        "study-serving-qa: requesting selected training example {}",
        task.id
    );
    let response = client
        .post(format!("{url}/v1/generate"))
        .json(&request)
        .send()
        .await?;
    let status = response.status();
    let bytes = response.bytes().await?;
    artefact::atomic_write(&args.output.join("http-response.json"), &bytes)?;
    ensure!(
        status.is_success(),
        "student HTTP inference failed with{status}; response retained"
    );
    let generation: Generation = serde_json::from_slice(&bytes)?;
    ensure!(
        generation.complete
            && !generation.completion.trim().is_empty()
            && generation.checkpoint_id == expected,
        "student endpoint returned incomplete/empty/substituted output"
    );
    let preflight = sandbox::preflight(&SandboxConfig {
        report_dir: args.output.join("sandbox"),
        ..Default::default()
    })
    .await?;
    let sandbox = SandboxConfig {
        report_dir: args.output.join("sandbox"),
        expected_image_id: preflight.image_id.clone(),
        ..Default::default()
    };
    artefact::write_json(&args.output.join("sandbox-preflight.json"), &preflight)?;
    let code = lec_distil::evaluate::extract_code(&generation.completion);
    let grade = sandbox::grade(&task, &code, &sandbox).await?;
    sandbox::verify_report(&grade, &sandbox, &task, &code)?;
    let denied = serve::initialise_routing(
        &args.output.join("forbidden-production-route.json"),
        VersionedEndpoint {
            url: url.clone(),
            checkpoint_id: expected.into(),
        },
    )
    .await;
    ensure!(
        denied.is_err() && !args.output.join("forbidden-production-route.json").exists(),
        "evaluation-only student was accepted as production incumbent"
    );
    checkpoint.verify(&std::env::current_dir()?)?;
    engine.verify(&std::env::current_dir()?)?;
    let result = json!({"schema_version":1,"evidence_kind":"actual_local_study_student_http_and_selected_training_example",
        "passed":grade.passed && grade.trace_complete && !grade.quarantined,"engine":engine,"checkpoint":checkpoint,
        "manifest":capture(&manifest_path)?,"health":health,"request":capture(&args.output.join("request.json"))?,
        "response":capture(&args.output.join("http-response.json"))?,"task_id":task.id,"task_split":"train",
        "selected_example":true,"sandbox_report":capture(&sandbox.report_dir.join(format!("{}.json",grade.run_id)))?,
        "production_routing_rejected":true,"held_out_used":false,"quality_evaluation":false,"production_eligible":false,
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "limitations":["One deliberately selected training example is an operational demonstration, not an unbiased quality estimate or learning result.",
            "The student remains evaluation-only; no assessment promotion or production canary is authorised."]});
    Ok(result)
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "preserve existing student endpoint QA; select a new output"
    );
    fs::create_dir_all(&args.output)?;
    let observed = run(&args).await;
    let report = match &observed {
        Ok(report) => report.clone(),
        Err(error) => {
            json!({"passed":false,"evidence_kind":"incomplete_local_study_student_endpoint_qa","error":format!("{error:#}"),"production_eligible":false})
        }
    };
    artefact::write_json(&args.output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    observed?;
    ensure!(
        report["passed"] == true,
        "selected endpoint example failed its real grader"
    );
    Ok(())
}
