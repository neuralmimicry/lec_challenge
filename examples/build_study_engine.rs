//! Build and pin exact executables for target capture, studies and their QA.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{artefact, engine};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Child, Command},
    thread,
    time::{Duration, Instant},
};
#[path = "support/source_inputs.rs"]
mod source_inputs;
use source_inputs::{SourceInput, capture as sources};

#[derive(Parser)]
struct Args {
    /// Use the caller's already verified CUDA toolkit environment.
    #[arg(long)]
    cuda: bool,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 7200)]
    wall_timeout_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Request {
    schema_version: u32,
    build: engine::cpu::CpuBuildConfig,
    sources: Vec<SourceInput>,
    cuda_environment: BTreeMap<String, String>,
    cargo_arguments: Vec<String>,
    wall_timeout_seconds: u64,
}

#[derive(Serialize, Deserialize)]
struct Executable {
    purpose: String,
    built_path: PathBuf,
    pinned: engine::PinnedEngine,
}

#[derive(Serialize, Deserialize)]
struct BuildResult {
    schema_version: u32,
    request_sha256: String,
    executables: Vec<Executable>,
}

struct OwnedBuild {
    child: Child,
    reaped: bool,
}
impl Drop for OwnedBuild {
    fn drop(&mut self) {
        if !self.reaped {
            // SAFETY: the unreaped child owns this isolated process group.
            unsafe { libc::kill(-(self.child.id() as i32), libc::SIGKILL) };
            let _ = self.child.wait();
        }
    }
}

const PURPOSES: [&str; 4] = [
    "application",
    "training_and_target_capture",
    "evaluation_recovery_qa",
    "numerical_prefix_qa",
];

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.wall_timeout_seconds > 0,
        "build wall deadline must be positive"
    );
    fs::create_dir_all(&args.output)?;
    let _lease = engine::RunLease::acquire(&args.output.join(".build.lock"))?;
    let profile = engine::cpu::select_profile(
        &std::env::var("LEC_CPU_PROFILE").unwrap_or_else(|_| "auto".into()),
        std::env::consts::ARCH,
        &engine::cpu::host_features(),
    )?;
    let mut build = engine::cpu::CpuBuildConfig::inspect(profile)?;
    if args.cuda {
        build.cargo_features = vec!["cuda".into()];
    }
    build.verify_host()?;
    let target = build.target_directory()?;
    let mut cargo_arguments = vec![
        "build".into(),
        "--locked".into(),
        "--offline".into(),
        "--release".into(),
        "--target".into(),
        build.target_triple.clone(),
        "--bin".into(),
        "lec-distil".into(),
        "--example".into(),
        "distillation_qa".into(),
        "--example".into(),
        "study_evaluation_resume_qa".into(),
        "--example".into(),
        "quantised_prefix_qa".into(),
    ];
    if args.cuda {
        cargo_arguments.extend(["--features".into(), "cuda".into()]);
    }
    // Record only build settings: never dump inherited credentials or environment.
    let cuda_environment = if args.cuda {
        [
            "CUDA_ROOT",
            "CUDA_PATH",
            "CUDA_TOOLKIT_ROOT_DIR",
            "CUDARC_CUDA_VERSION",
            "CUDA_COMPUTE_CAP",
            "NVCC_CCBIN",
            "NVCC_PREPEND_FLAGS",
        ]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok().map(|value| (key.into(), value)))
        .collect()
    } else {
        BTreeMap::new()
    };
    let request = Request {
        schema_version: 1,
        build,
        sources: sources()?,
        cuda_environment,
        cargo_arguments,
        wall_timeout_seconds: args.wall_timeout_seconds,
    };
    let request_path = args.output.join("request.json");
    if request_path.exists() {
        let original: Request = serde_json::from_slice(&fs::read(&request_path)?)?;
        ensure!(
            original == request,
            "build inputs changed; preserve this attempt and use a new output"
        );
    } else {
        artefact::write_json(&request_path, &request)?;
    }
    let result_path = args.output.join("result.json");
    if result_path.exists() {
        let saved: BuildResult = serde_json::from_slice(&fs::read(&result_path)?)?;
        ensure!(
            saved.schema_version == 1
                && saved.request_sha256 == artefact::file_hash(&request_path)?,
            "build result belongs to another frozen request"
        );
        let engines = saved.executables;
        ensure!(
            engines.len() == PURPOSES.len(),
            "build result has an incomplete executable inventory"
        );
        for (entry, expected) in engines.iter().zip(PURPOSES) {
            ensure!(
                entry.purpose == expected,
                "build result executable roles changed"
            );
            entry.pinned.verify()?;
            ensure!(
                entry
                    .pinned
                    .cpu_build
                    .as_ref()
                    .is_some_and(|receipt| receipt.build == request.build),
                "saved executable uses another CPU/compiler configuration"
            );
        }
        println!("Verified and reused all four pinned study executables; no compilation needed.");
        return Ok(());
    }
    let state_path = args.output.join("state.json");
    artefact::write_json(
        &state_path,
        &serde_json::json!({
            "status":"building", "target_directory":target,
            "request_sha256":artefact::file_hash(&request_path)?,
            "recovery":"Restart this command with unchanged inputs to reuse Cargo's completed artefacts."
        }),
    )?;
    let mut command = Command::new("cargo");
    command
        .args(&request.cargo_arguments)
        .env("CARGO_TARGET_DIR", &target)
        .envs(request.build.build_environment()?)
        .process_group(0);
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: only async-signal-safe libc calls run between fork and exec.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                return Err(std::io::Error::other("build supervisor exited before exec"));
            }
            Ok(())
        });
    }
    eprintln!(
        "study build: {} profile, {}; cached compiler work will be reused",
        request.build.profile,
        if args.cuda { "CUDA and CPU" } else { "CPU" }
    );
    let mut child = OwnedBuild {
        child: command.spawn().context("start native study build")?,
        reaped: false,
    };
    let start = Instant::now();
    let mut heartbeat = Instant::now();
    let status = loop {
        if let Some(status) = child.child.try_wait()? {
            child.reaped = true;
            break status;
        }
        if start.elapsed() >= Duration::from_secs(request.wall_timeout_seconds) {
            artefact::write_json(
                &state_path,
                &serde_json::json!({
                    "status":"deadline_exceeded", "elapsed_seconds":start.elapsed().as_secs_f64(),
                    "request_sha256":artefact::file_hash(&request_path)?, "completed_executables_published":false
                }),
            )?;
            anyhow::bail!(
                "native study build exceeded its frozen wall deadline; owned process group will be terminated and Cargo progress retained"
            );
        }
        if heartbeat.elapsed() >= Duration::from_secs(30) {
            eprintln!(
                "study build: compiling/linking after {:.0}s; existing experiments keep their original executables",
                start.elapsed().as_secs_f64()
            );
            heartbeat = Instant::now();
        }
        thread::sleep(Duration::from_millis(250));
    };
    artefact::write_json(
        &state_path,
        &serde_json::json!({
            "status":if status.success() { "compiled" } else { "failed" },
            "exit_code":status.code(), "elapsed_seconds":start.elapsed().as_secs_f64(),
            "request_sha256":artefact::file_hash(&request_path)?
        }),
    )?;
    ensure!(
        status.success(),
        "study build failed; its request and Cargo progress are retained"
    );
    ensure!(
        sources()? == request.sources,
        "sources changed during compilation; preserve this attempt and build a new protocol"
    );
    let release = target.join(&request.build.target_triple).join("release");
    let mut engines = Vec::new();
    for (purpose, path) in PURPOSES.into_iter().zip([
        release.join("lec-distil"),
        release.join("examples/distillation_qa"),
        release.join("examples/study_evaluation_resume_qa"),
        release.join("examples/quantised_prefix_qa"),
    ]) {
        let pinned = engine::pin_executable_with_cpu(
            &path,
            Path::new("state/engines"),
            request.build.clone(),
        )?;
        engines.push(Executable {
            purpose: purpose.into(),
            built_path: path,
            pinned,
        });
    }
    let result = BuildResult {
        schema_version: 1,
        request_sha256: artefact::file_hash(&request_path)?,
        executables: engines,
    };
    artefact::write_json(&result_path, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
