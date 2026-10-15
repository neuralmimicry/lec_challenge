//! Native resource-aware build/launch boundary for Linux amd64 and arm64.
//! Model selection and evidence decisions remain in the application modules.
use anyhow::{Context, Result, ensure};
use lec_distil::{artefact, engine, platform};
use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
};

#[derive(Debug)]
struct Toolkit {
    root: PathBuf,
    major: u32,
    minor: u32,
}

fn run(mut command: Command, description: &str) -> Result<ExitStatus> {
    eprintln!("launcher: {description}");
    command
        .status()
        .with_context(|| format!("start {description}"))
}

fn version_pair(text: &str) -> Option<(u32, u32)> {
    let mut parts = text.trim().split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts
            .next()?
            .split(|c: char| !c.is_ascii_digit())
            .next()?
            .parse()
            .ok()?,
    ))
}

fn driver_ceiling() -> Option<(u32, u32)> {
    if let Ok(version) = platform::cuda_probe::driver_version() {
        return Some(version);
    }
    let output = Command::new("nvidia-smi").output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    version_pair(text.split("CUDA Version:").nth(1)?.trim_start())
}

fn toolkit_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(explicit) = std::env::var_os("LEC_CUDA_TOOLKIT") {
        roots.push(PathBuf::from(explicit));
    } else {
        for key in ["CUDA_ROOT", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR"] {
            if let Some(value) = std::env::var_os(key) {
                roots.push(PathBuf::from(value));
            }
        }
        if let Ok(entries) = fs::read_dir("/usr/local") {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("cuda") {
                    roots.push(entry.path());
                }
            }
        }
        roots.push("/opt/cuda".into());
    }
    roots
}

fn toolkit_candidates() -> Result<Vec<Toolkit>> {
    let mut toolkits = Vec::new();
    for root in toolkit_roots() {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        if toolkits
            .iter()
            .any(|toolkit: &Toolkit| toolkit.root == root)
        {
            continue;
        }
        let Ok(output) = Command::new(root.join("bin/nvcc"))
            .arg("--version")
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        if let Some((major, minor)) = text.split("release ").nth(1).and_then(version_pair) {
            toolkits.push(Toolkit { root, major, minor });
        }
    }
    toolkits.sort_by_key(|toolkit| std::cmp::Reverse((toolkit.major, toolkit.minor)));
    if let Some(ceiling) = driver_ceiling() {
        toolkits.retain(|toolkit| (toolkit.major, toolkit.minor) <= ceiling);
    }
    Ok(toolkits)
}

/// Replaying a pinned executable needs runtime libraries, never nvcc, a host
/// compiler or a newly built compatibility overlay. Preserve caller precedence
/// and also support installations containing only the CUDA runtime libraries.
fn runtime_library_paths(roots: &[PathBuf], existing: &OsStr) -> Result<OsString> {
    let mut paths = std::env::split_paths(existing)
        .filter(|path| !path.as_os_str().is_empty())
        .collect::<Vec<_>>();
    for root in roots {
        for suffix in [
            "lib64",
            "lib",
            "targets/x86_64-linux/lib",
            "targets/aarch64-linux/lib",
        ] {
            let path = root.join(suffix);
            if path.is_dir() {
                paths.push(path.canonicalize()?);
            }
        }
    }
    let mut seen = BTreeSet::new();
    paths.retain(|path| seen.insert(path.clone()));
    Ok(std::env::join_paths(paths)?)
}

fn host_compiler(toolkit: &Toolkit) -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("LEC_CUDA_HOST_COMPILER") {
        return Some(explicit.into());
    }
    let maximum = fs::read_to_string(toolkit.root.join("include/crt/host_config.h"))
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.contains("__GNUC__ > "))
                .map(str::to_owned)
        })
        .and_then(|line| {
            line.split("__GNUC__ > ")
                .nth(1)?
                .split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse::<u32>()
                .ok()
        });
    let mut names = Vec::from(["g++".to_owned()]);
    if let Some(maximum) = maximum {
        for major in (7..=maximum).rev() {
            names.push(format!("g++-{major}"));
        }
    }
    for name in names {
        let Ok(output) = Command::new(&name).args(["-dumpfullversion"]).output() else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let major = String::from_utf8_lossy(&output.stdout)
            .split('.')
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok());
        if major.is_some_and(|major| maximum.is_none_or(|maximum| major <= maximum)) {
            return Some(name.into());
        }
    }
    None
}

fn requires_local_header_overlay(toolkit: &Toolkit) -> bool {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: glibc owns this process-lifetime NUL-terminated version string.
        let version =
            unsafe { std::ffi::CStr::from_ptr(libc::gnu_get_libc_version()) }.to_string_lossy();
        toolkit.major == 12
            && toolkit.minor == 9
            && version_pair(&version).is_some_and(|version| version >= (2, 41))
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        let _ = toolkit;
        false
    }
}

fn output_directory(arguments: &[OsString]) -> Option<PathBuf> {
    let default = match arguments.first()?.to_str()? {
        "select-teachers" => "reports/comparative/search/round-001",
        "compare-models" => "reports/comparative/development",
        "study-evaluate" => "reports/local-study/evaluation",
        "explore" => "reports/local-study/explore/round-001",
        "collect-student-rollouts" => "reports/local-study/student-rollouts",
        "collect-teacher-content" => "reports/local-study/content",
        "study-plan" => "reports/local-study/study/round-001",
        "train" => "checkpoints/candidate",
        "study-run" => "reports/local-study/study/round-001/quick",
        _ => return None,
    };
    for (index, argument) in arguments.iter().enumerate() {
        if argument == "--output" {
            return arguments.get(index + 1).map(PathBuf::from);
        }
        if let Some(path) = argument.to_str().and_then(|s| s.strip_prefix("--output=")) {
            return Some(path.into());
        }
    }
    if arguments.first().is_some_and(|s| s == "study-run") {
        let value = |name: &str, fallback: &str| {
            arguments
                .iter()
                .enumerate()
                .find_map(|(index, arg)| {
                    if arg == name {
                        arguments.get(index + 1).cloned()
                    } else {
                        arg.to_str()
                            .and_then(|s| s.strip_prefix(&format!("{name}=")))
                            .map(OsString::from)
                    }
                })
                .unwrap_or_else(|| fallback.into())
        };
        return Some(
            PathBuf::from(value("--study", "reports/local-study/study/round-001"))
                .join(value("--profile", "quick")),
        );
    }
    Some(default.into())
}

/// Resolve fresh paths before recovery/building, so a new protocol receives its
/// complete engine receipt before any application work or failure is published.
fn prepare_fresh_arguments(arguments: &mut Vec<OsString>) -> Result<Option<PathBuf>> {
    let flags = arguments.iter().filter(|s| *s == "--fresh").count();
    if flags == 0 {
        return Ok(None);
    }
    ensure!(flags == 1, "--fresh was supplied more than once");
    ensure!(
        arguments
            .first()
            .and_then(|s| s.to_str())
            .is_some_and(|command| matches!(
                command,
                "explore"
                    | "select-teachers"
                    | "collect-teacher-content"
                    | "collect-student-rollouts"
                    | "study-plan"
                    | "study-run"
                    | "study-evaluate"
                    | "train"
            )),
        "this command does not support --fresh"
    );
    let mut output = None;
    for (index, argument) in arguments.iter().enumerate() {
        let value = if argument == "--output" {
            let value = arguments
                .get(index + 1)
                .context("--output requires a path")?;
            ensure!(
                !value.is_empty() && !value.to_string_lossy().starts_with("--"),
                "--output requires a path"
            );
            Some((index, true))
        } else if argument
            .to_str()
            .is_some_and(|s| s.starts_with("--output="))
        {
            ensure!(
                argument.to_str() != Some("--output="),
                "--output requires a path"
            );
            Some((index, false))
        } else {
            None
        };
        if let Some(value) = value {
            ensure!(
                output.replace(value).is_none(),
                "--output was supplied more than once"
            );
        }
    }
    let previous = output_directory(arguments).context("fresh output cannot be resolved")?;
    let resolved = engine::prepare_fresh_output(&previous)?;
    if let Some((index, separate)) = output {
        if separate {
            arguments[index + 1] = resolved.clone().into_os_string();
        } else {
            let mut replacement = OsString::from("--output=");
            replacement.push(&resolved);
            arguments[index] = replacement;
        }
    } else {
        arguments.extend(["--output".into(), resolved.clone().into_os_string()]);
    }
    arguments.retain(|s| s != "--fresh");
    eprintln!(
        "launcher: explicit fresh output {}; previous evidence retained at {}",
        resolved.display(),
        previous.display()
    );
    Ok(Some(resolved))
}

fn resume_engine(arguments: &[OsString]) -> Result<Option<engine::PinnedEngine>> {
    resume_engine_at(arguments, Path::new("state/engines"))
        .context("cannot resume the original engine; restore its pinned snapshot rather than discarding progress")
}

/// Old content policies did not bind their engine globally. Their immutable
/// per-generation plans still do. Inspect precisely the collector's three
/// directory levels; sandbox artefacts are not alternate generation plans.
fn content_plan_engine(output: &Path) -> Result<Option<String>> {
    fn directories(parent: &Path) -> Result<Vec<PathBuf>> {
        if !parent.try_exists()? {
            return Ok(Vec::new());
        }
        fs::read_dir(parent)?
            .filter_map(|entry| match entry {
                Ok(entry) => match entry.file_type() {
                    Ok(kind) if kind.is_dir() => Some(Ok(entry.path())),
                    Ok(_) => None,
                    Err(error) => Some(Err(error.into())),
                },
                Err(error) => Some(Err(error.into())),
            })
            .collect()
    }
    let mut identity: Option<String> = None;
    for teacher in directories(output)? {
        for task in directories(&teacher)? {
            for content in directories(&task)? {
                let path = content.join("plan.json");
                if !path.try_exists()? {
                    continue;
                }
                let plan: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
                let hash = plan["engine_binary_sha256"].as_str().with_context(|| {
                    format!(
                        "saved content plan has no engine identity: {}",
                        path.display()
                    )
                })?;
                if let Some(previous) = &identity {
                    ensure!(
                        previous == hash,
                        "saved content plans use mixed engine identities; preserve their evidence and recover the original collection"
                    );
                } else {
                    identity = Some(hash.to_owned());
                }
            }
        }
    }
    Ok(identity)
}

fn launch_receipt_engine(output: &Path, engines: &Path) -> Result<Option<engine::PinnedEngine>> {
    let path = output.join("launch-engine.json");
    if !path.try_exists()? {
        return Ok(None);
    }
    let receipt: engine::PinnedEngine = serde_json::from_slice(&fs::read(&path)?)?;
    // Resolve within the current content-addressed root rather than executing
    // the receipt's historical absolute path after moving an evidence bundle.
    let pinned = engine::resolve(engines, &receipt.sha256)?;
    ensure!(
        pinned.bytes == receipt.bytes && pinned.cpu_build == receipt.cpu_build,
        "saved launcher engine length or CPU build contract differs from its verified snapshot"
    );
    Ok(Some(pinned))
}

fn resume_engine_at(
    arguments: &[OsString],
    engines: &Path,
) -> Result<Option<engine::PinnedEngine>> {
    if arguments.iter().any(|argument| argument == "--fresh") {
        return Ok(None);
    }
    let Some(output) = output_directory(arguments) else {
        return Ok(None);
    };
    // The protocol is written before inference. It remains authoritative even
    // if the launcher died before recording its own launch manifest. Resolve
    // the exact executable for standalone evaluations and collectors as well
    // as the complete workflow, so a later rebuild cannot reset their work.
    let (file, pointer) = match arguments[0].to_str() {
        Some("select-teachers") => ("freeze.json", "/engine_binary_sha256"),
        Some("compare-models") => ("plan.json", "/engine_binary_sha256"),
        Some("study-evaluate") => ("plan.json", "/engine/sha256"),
        Some("study-run") => ("state.json", "/engine/sha256"),
        Some("explore") => ("protocol.json", "/stage_engine/sha256"),
        Some("collect-student-rollouts") => ("rollout-policy.json", "/engine_sha256"),
        Some("collect-teacher-content") => ("content-policy.json", "/engine_binary_sha256"),
        _ => return launch_receipt_engine(&output, engines),
    };
    let freeze = output.join(file);
    let frozen = if freeze.try_exists()? {
        Some(serde_json::from_slice::<serde_json::Value>(&fs::read(
            &freeze,
        )?)?)
    } else {
        None
    };
    let legacy_content = arguments[0] == "collect-teacher-content";
    let plan_hash = if legacy_content {
        content_plan_engine(&output)?
    } else {
        None
    };
    let protocol_hash = if let Some(frozen) = &frozen {
        match frozen.pointer(pointer) {
            Some(value) => Some(
                value
                    .as_str()
                    .context("previous protocol has an invalid engine identity")?,
            ),
            None if legacy_content => None,
            None => anyhow::bail!("previous run has no frozen engine identity"),
        }
    } else {
        None
    };
    let pinned = if let Some(hash) = protocol_hash {
        if let Some(plan_hash) = &plan_hash {
            ensure!(
                hash == plan_hash,
                "content policy and generation plans bind different engines"
            );
        }
        // The application protocol is authoritative. A stale launcher receipt
        // cannot replace the executable which produced its durable records.
        let pinned = engine::resolve(engines, hash)?;
        let launch_receipt = output.join("launch-engine.json");
        if launch_receipt.exists() {
            let receipt: engine::PinnedEngine = serde_json::from_slice(&fs::read(launch_receipt)?)?;
            if receipt.sha256 == hash {
                ensure!(
                    receipt.bytes == pinned.bytes && receipt.cpu_build == pinned.cpu_build,
                    "application and launcher CPU requirements differ"
                );
            }
        }
        if let Some(parent) = pointer.strip_suffix("/sha256")
            && let Some(value) = frozen
                .as_ref()
                .and_then(|f| f.pointer(&format!("{parent}/cpu_build")))
        {
            let expected: Option<engine::cpu::CpuBuildReceipt> =
                serde_json::from_value(value.clone())?;
            ensure!(
                pinned.cpu_build == expected,
                "frozen application CPU requirements differ from engine snapshot"
            );
        }
        Some(pinned)
    } else {
        let receipt = launch_receipt_engine(&output, engines)?;
        if let Some(hash) = plan_hash {
            if let Some(receipt) = &receipt {
                ensure!(
                    receipt.sha256 == hash,
                    "launcher receipt and historical content plans bind different engines"
                );
            }
            Some(engine::resolve(engines, &hash)?)
        } else {
            // The receipt precedes application startup, including native GGUF
            // export before the evaluation protocol exists. Without either a
            // receipt or engine-bound work it is safe to build a first engine.
            receipt
        }
    };
    let Some(pinned) = pinned else {
        return Ok(None);
    };
    eprintln!(
        "launcher: resuming validated progress in {} with engine {}",
        output.display(),
        pinned.sha256
    );
    Ok(Some(pinned))
}

fn build_engine(arguments: Vec<OsString>, environment: &[(OsString, OsString)]) -> Result<PathBuf> {
    eprintln!("launcher: building the native engine; Cargo reuses completed compilation");
    let mut child = Command::new("cargo")
        .args(arguments)
        .arg("--message-format=json-render-diagnostics")
        .envs(environment.iter().cloned())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut executable = None;
    for line in BufReader::new(child.stdout.take().context("Cargo output missing")?).lines() {
        let line = line?;
        let record: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid Cargo build record: {line}"))?;
        if record["reason"] == "compiler-artifact"
            && record["target"]["name"] == "lec-distil"
            && let Some(path) = record["executable"].as_str()
        {
            executable = Some(PathBuf::from(path));
        }
        if let Some(diagnostic) = record["message"]["rendered"].as_str() {
            eprint!("{diagnostic}");
        }
    }
    ensure!(child.wait()?.success(), "native engine build failed");
    executable.context("Cargo produced no lec-distil executable")
}

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    ensure!(
        !arguments.is_empty(),
        "usage: platform_launch COMMAND [ARGUMENTS...]"
    );
    if arguments[0] == "select-teachers"
        && !arguments.iter().any(|argument| {
            argument == "--output"
                || argument
                    .to_str()
                    .is_some_and(|s| s.starts_with("--output="))
        })
    {
        arguments.extend([
            "--output".into(),
            "reports/comparative/search/round-001".into(),
        ]);
    }
    prepare_fresh_arguments(&mut arguments)?;
    let hardware = platform::detect()?;
    eprintln!(
        "launcher: {} {} — {} available CPU threads, {:.2} GiB available RAM, {} detected CUDA GPUs",
        hardware.operating_system,
        hardware.architecture,
        hardware.available_cpu_threads,
        hardware.available_ram_bytes as f64 / (1u64 << 30) as f64,
        hardware.gpus.len()
    );
    if arguments[0] == "--probe" {
        let toolkits = toolkit_candidates()?
            .into_iter()
            .map(|toolkit| {
                serde_json::json!({
                    "root":toolkit.root,"major":toolkit.major,"minor":toolkit.minor,
                    "compatible_host_compiler":host_compiler(&toolkit)
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"hardware":hardware,
            "driver_cuda_ceiling":driver_ceiling(),"compatible_installed_toolkits":toolkits,
            "cpu_simd_features":engine::cpu::host_features()}))?
        );
        return Ok(());
    }
    let offline = arguments.iter().any(|argument| argument == "--offline");
    let cpu_only = std::env::var("LEC_DEVICE").is_ok_and(|value| value == "cpu")
        || arguments.iter().any(|argument| argument == "--device=cpu")
        || arguments
            .windows(2)
            .any(|pair| pair[0] == "--device" && pair[1] == "cpu");
    let previous_engine = resume_engine(&arguments)?;
    let mut cargo_arguments = vec![OsString::from("build"), "--locked".into()];
    if offline {
        cargo_arguments.push("--offline".into());
    }
    cargo_arguments.push("--release".into());
    let mut environment = Vec::<(OsString, OsString)>::new();
    let mut cuda_build = false;
    environment.push((
        "RAYON_NUM_THREADS".into(),
        hardware.available_cpu_threads.to_string().into(),
    ));
    environment.push((
        "TOKIO_WORKER_THREADS".into(),
        hardware.available_cpu_threads.to_string().into(),
    ));
    if previous_engine.is_some() {
        let existing_libraries = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let libraries = runtime_library_paths(&toolkit_roots(), &existing_libraries)?;
        if !libraries.is_empty() {
            environment.push(("LD_LIBRARY_PATH".into(), libraries));
        }
        eprintln!("launcher: using the original engine; CUDA build preparation is unnecessary");
    } else if !cpu_only && let Some(gpu) = hardware.gpus.first() {
        let capability = gpu
            .compute_capability
            .as_ref()
            .context("GPU compute capability unavailable")?
            .replace(' ', "");
        let capability = capability.replace('.', "");
        ensure!(
            capability.bytes().all(|byte| byte.is_ascii_digit()),
            "invalid CUDA compute capability"
        );
        let toolkit = toolkit_candidates()?.into_iter().find(|toolkit| {
            Command::new(toolkit.root.join("bin/nvcc"))
                .arg("--list-gpu-code")
                .output()
                .ok()
                .is_some_and(|output| {
                    String::from_utf8_lossy(&output.stdout)
                        .split_whitespace()
                        .any(|code| code == format!("sm_{capability}"))
                })
        });
        if let Some(toolkit) = toolkit {
            let compiler = host_compiler(&toolkit)
                .context("no compatible installed CUDA C++ host compiler")?;
            let mut root = toolkit.root.clone();
            if requires_local_header_overlay(&toolkit) {
                let overlay = std::path::absolute("state/cuda-compat")?;
                let mut preparation = Command::new("cargo");
                preparation
                    .args([
                        "run",
                        "--locked",
                        "--offline",
                        "--example",
                        "cuda_compat",
                        "--",
                    ])
                    .arg(&toolkit.root)
                    .arg(&overlay)
                    .env("CARGO_TARGET_DIR", "target")
                    .env("CUDA_COMPUTE_CAP", &capability)
                    .env("NVCC_CCBIN", &compiler);
                ensure!(
                    run(
                        preparation,
                        "prepare and compile-check local CUDA compatibility headers"
                    )?
                    .success(),
                    "CUDA compatibility preparation failed"
                );
                root = overlay;
                environment.push((
                    "NVCC_PREPEND_FLAGS".into(),
                    format!("-I{}", root.join("include").display()).into(),
                ));
            }
            eprintln!(
                "launcher: selecting CUDA {}.{} at {} for {} (SM {})",
                toolkit.major,
                toolkit.minor,
                toolkit.root.display(),
                gpu.name,
                capability
            );
            for key in ["CUDA_ROOT", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR"] {
                environment.push((key.into(), root.clone().into_os_string()));
            }
            environment.push((
                "CUDARC_CUDA_VERSION".into(),
                format!("{}{:02}0", toolkit.major, toolkit.minor).into(),
            ));
            environment.push(("CUDA_COMPUTE_CAP".into(), capability.into()));
            environment.push(("NVCC_CCBIN".into(), compiler.into_os_string()));
            let existing_path = std::env::var_os("PATH").unwrap_or_default();
            let paths = std::iter::once(toolkit.root.join("bin"))
                .chain(std::env::split_paths(&existing_path))
                .collect::<Vec<_>>();
            environment.push(("PATH".into(), std::env::join_paths(paths)?));
            let existing_libraries = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
            let libraries = std::iter::once(toolkit.root.join("lib64"))
                .chain(std::env::split_paths(&existing_libraries))
                .collect::<Vec<_>>();
            environment.push(("LD_LIBRARY_PATH".into(), std::env::join_paths(libraries)?));
            environment.push(("CARGO_TARGET_DIR".into(), "target/cuda".into()));
            cargo_arguments.extend(["--features".into(), "cuda".into()]);
            cuda_build = true;
        } else {
            eprintln!(
                "launcher: GPU detected but no driver-compatible installed CUDA toolkit supports it; building explicit automatic CPU mode"
            );
        }
    }
    let cpu_build = if previous_engine.is_none() {
        let preference = std::env::var("LEC_CPU_PROFILE").unwrap_or_else(|_| "auto".into());
        let profile = engine::cpu::select_profile(
            &preference,
            std::env::consts::ARCH,
            &engine::cpu::host_features(),
        )?;
        let mut build = engine::cpu::CpuBuildConfig::inspect(profile)?;
        if cuda_build {
            build.cargo_features = vec!["cuda".into()];
        }
        build.verify_host()?;
        eprintln!(
            "launcher: explicit CPU profile {} with frozen target {} (also used by CPU fallback)",
            build.profile, build.target_triple
        );
        environment.extend(
            build
                .build_environment()?
                .into_iter()
                .map(|(key, value)| (key.into(), value)),
        );
        environment.push((
            "CARGO_TARGET_DIR".into(),
            build.target_directory()?.into_os_string(),
        ));
        cargo_arguments.extend(["--target".into(), build.target_triple.clone().into()]);
        Some(build)
    } else {
        None
    };
    cargo_arguments.extend(["--bin".into(), "lec-distil".into()]);
    let pinned = if let Some(previous) = previous_engine {
        previous
    } else {
        let built = build_engine(cargo_arguments, &environment)?;
        let pinned = if let Some(build) = cpu_build {
            ensure!(
                artefact::file_hash(Path::new("Cargo.lock"))? == build.cargo_lock_sha256,
                "Cargo.lock changed during CPU compilation; preserve this build and start a new explicit protocol"
            );
            engine::pin_executable_with_cpu(&built, Path::new("state/engines"), build)?
        } else {
            engine::pin_executable(&built, Path::new("state/engines"))?
        };
        eprintln!(
            "launcher: pinned engine {} ({} bytes)",
            pinned.sha256, pinned.bytes
        );
        pinned
    };
    pinned.verify()?;
    if let Some(output) = output_directory(&arguments) {
        artefact::write_json(&output.join("launch-engine.json"), &pinned)?;
    }
    let mut application = Command::new(&pinned.path);
    application.args(arguments).envs(environment);
    let status = run(
        application,
        "run pinned native engine; cached progress, downloads and evaluation follow",
    )?;
    std::process::exit(status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn pinned(root: &Path, bytes: &[u8]) -> engine::PinnedEngine {
        let source = root.join("build-output");
        fs::write(&source, bytes).unwrap();
        engine::pin_executable(&source, &root.join("engines")).unwrap()
    }

    fn arguments(command: &str, output: &Path) -> Vec<OsString> {
        vec![command.into(), "--output".into(), output.into()]
    }

    fn portable_fixture() -> engine::cpu::CpuBuildConfig {
        let architecture = std::env::consts::ARCH;
        let target = format!("{architecture}-unknown-linux-gnu");
        let flags = vec![
            "-C".into(),
            "target-cpu=generic".into(),
            "-C".into(),
            "debug-assertions=no".into(),
            "-C".into(),
            "panic=unwind".into(),
        ];
        engine::cpu::CpuBuildConfig {
            schema_version: 1,
            protocol: "explicit_cpu_simd_build_v1".into(),
            architecture: architecture.into(),
            profile: "portable".into(),
            target_triple: target.clone(),
            required_features: vec![],
            encoded_rustflags: flags.join("\u{1f}"),
            rustflags: flags,
            rustc_version_verbose: format!(
                "labelled launcher fixture; no compiler run\nhost: {target}\n"
            ),
            rustc_cfg: vec![
                format!("target_arch=\"{architecture}\""),
                "target_os=\"linux\"".into(),
            ],
            cargo_lock_sha256: "a".repeat(64),
            cargo_features: Vec::new(),
            resolved_rustc: None,
        }
    }

    #[test]
    fn cpu_recovery_keeps_frozen_requirements_and_rejects_their_removal() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("build-output");
        fs::write(&source, b"labelled CPU fixture; never executed").unwrap();
        let engines = directory.path().join("engines");
        let original =
            engine::pin_executable_with_cpu(&source, &engines, portable_fixture()).unwrap();
        let output = directory.path().join("evaluation");
        artefact::write_json(&output.join("launch-engine.json"), &original).unwrap();
        let command = arguments("study-evaluate", &output);
        let recovered = resume_engine_at(&command, &engines).unwrap().unwrap();
        assert_eq!(recovered.cpu_build, original.cpu_build);
        let mut corrupt = original.clone();
        corrupt.cpu_build = None;
        artefact::write_json(&output.join("launch-engine.json"), &corrupt).unwrap();
        assert!(resume_engine_at(&command, &engines).is_err());
        fs::remove_file(output.join("launch-engine.json")).unwrap();
        artefact::write_json(
            &output.join("plan.json"),
            &serde_json::json!({"engine":original}),
        )
        .unwrap();
        artefact::write_json(
            &original.path.parent().unwrap().join("engine.json"),
            &corrupt,
        )
        .unwrap();
        assert!(resume_engine_at(&command, &engines).is_err());
    }

    fn content_plan(output: &Path, task: &str, content: &str, hash: &str) {
        artefact::write_json(
            &output
                .join("teacher")
                .join(task)
                .join(content)
                .join("plan.json"),
            &serde_json::json!({"engine_binary_sha256":hash}),
        )
        .unwrap();
    }

    #[test]
    fn fresh_is_resolved_once_and_receipt_binds_the_actual_child_directory() {
        for equals in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let original = root.path().join("original");
            fs::create_dir(&original).unwrap();
            fs::write(original.join("last-failure.json"), b"retained original").unwrap();
            let mut command = vec![OsString::from("select-teachers")];
            if equals {
                command.push(format!("--output={}", original.display()).into());
            } else {
                command.extend(["--output".into(), original.clone().into_os_string()]);
            }
            command.push("--fresh".into());
            let resolved = prepare_fresh_arguments(&mut command).unwrap().unwrap();
            assert_ne!(resolved, original);
            assert_eq!(output_directory(&command), Some(resolved.clone()));
            assert!(!command.iter().any(|s| s == "--fresh"));
            assert!(prepare_fresh_arguments(&mut command).unwrap().is_none());
            let engine = pinned(root.path(), b"labelled fresh receipt fixture");
            artefact::write_json(&resolved.join("launch-engine.json"), &engine).unwrap();
            assert_eq!(
                resume_engine_at(&command, &root.path().join("engines"))
                    .unwrap()
                    .unwrap()
                    .sha256,
                engine.sha256
            );
            assert_eq!(
                fs::read(original.join("last-failure.json")).unwrap(),
                b"retained original"
            );
            assert!(!original.join("launch-engine.json").exists());
        }
    }

    #[test]
    fn fresh_rejects_ambiguous_output_flags_before_mutating_old_output() {
        let mut command = vec![
            "select-teachers".into(),
            "--fresh".into(),
            "--output".into(),
            "--offline".into(),
        ];
        assert!(prepare_fresh_arguments(&mut command).is_err());
        let mut command = vec![
            "select-teachers".into(),
            "--fresh".into(),
            "--output=a".into(),
            "--output=b".into(),
        ];
        assert!(prepare_fresh_arguments(&mut command).is_err());
        let command = vec![
            "study-run".into(),
            "--study".into(),
            "reports/labelled-fixture-study".into(),
            "--profile=full".into(),
        ];
        assert_eq!(
            output_directory(&command),
            Some(PathBuf::from("reports/labelled-fixture-study/full"))
        );
    }

    #[test]
    fn pre_protocol_export_interruption_recovers_its_original_launch_engine() {
        let directory = tempfile::tempdir().unwrap();
        let original = pinned(directory.path(), b"original exporter fixture");
        let later = pinned(directory.path(), b"later exporter fixture");
        let output = directory.path().join("evaluation");
        artefact::write_json(&output.join("launch-engine.json"), &original).unwrap();
        // The exporter may already own durable tensor fragments even though
        // StudyEval has not yet written its application protocol.
        fs::write(
            output.join("export-progress-fixture"),
            b"saved tensor fragment",
        )
        .unwrap();
        let receipt = fs::read(output.join("launch-engine.json")).unwrap();
        let resumed = resume_engine_at(
            &arguments("study-evaluate", &output),
            &directory.path().join("engines"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resumed.sha256, original.sha256);
        assert_ne!(resumed.sha256, later.sha256);
        assert!(!output.join("plan.json").exists());
        assert_eq!(
            fs::read(output.join("launch-engine.json")).unwrap(),
            receipt
        );
        assert_eq!(
            fs::read(output.join("export-progress-fixture")).unwrap(),
            b"saved tensor fragment"
        );
    }

    #[test]
    fn application_protocol_wins_over_a_stale_launcher_receipt() {
        let directory = tempfile::tempdir().unwrap();
        let original = pinned(directory.path(), b"original engine fixture");
        let later = pinned(directory.path(), b"later engine fixture");
        let output = directory.path().join("evaluation");
        let protocol = serde_json::json!({"engine":{"sha256":original.sha256}});
        artefact::write_json(&output.join("plan.json"), &protocol).unwrap();
        artefact::write_json(&output.join("launch-engine.json"), &later).unwrap();
        let before = fs::read(output.join("plan.json")).unwrap();
        let resumed = resume_engine_at(
            &arguments("study-evaluate", &output),
            &directory.path().join("engines"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resumed.sha256, original.sha256);
        assert_eq!(fs::read(output.join("plan.json")).unwrap(), before);
    }

    #[test]
    fn launch_receipt_cannot_hide_corrupted_snapshot_bytes_or_length() {
        let directory = tempfile::tempdir().unwrap();
        let original = pinned(directory.path(), b"original engine fixture");
        let output = directory.path().join("evaluation");
        let mut bad_length = original.clone();
        bad_length.bytes += 1;
        artefact::write_json(&output.join("launch-engine.json"), &bad_length).unwrap();
        let command = arguments("study-evaluate", &output);
        assert!(resume_engine_at(&command, &directory.path().join("engines")).is_err());
        artefact::write_json(&output.join("launch-engine.json"), &original).unwrap();
        fs::set_permissions(&original.path, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(&original.path, b"changed engine fixture!").unwrap();
        fs::set_permissions(&original.path, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(resume_engine_at(&command, &directory.path().join("engines")).is_err());
    }

    #[test]
    fn legacy_content_recovers_homogeneous_plans_without_rewriting_policy() {
        let directory = tempfile::tempdir().unwrap();
        let original = pinned(directory.path(), b"original content engine fixture");
        let output = directory.path().join("content");
        artefact::write_json(
            &output.join("content-policy.json"),
            &serde_json::json!({"schema_version":1,"max_tokens":512}),
        )
        .unwrap();
        content_plan(&output, "task-a", "process", &original.sha256);
        content_plan(&output, "task-b", "context", &original.sha256);
        let policy = fs::read(output.join("content-policy.json")).unwrap();
        let resumed = resume_engine_at(
            &arguments("collect-teacher-content", &output),
            &directory.path().join("engines"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(resumed.sha256, original.sha256);
        assert_eq!(
            fs::read(output.join("content-policy.json")).unwrap(),
            policy
        );
    }

    #[test]
    fn legacy_content_rejects_mixed_plan_origins_and_conflicting_receipts() {
        let directory = tempfile::tempdir().unwrap();
        let original = pinned(directory.path(), b"original content engine fixture");
        let later = pinned(directory.path(), b"later content engine fixture");
        let output = directory.path().join("content");
        artefact::write_json(&output.join("content-policy.json"), &serde_json::json!({})).unwrap();
        content_plan(&output, "task-a", "process", &original.sha256);
        content_plan(&output, "task-b", "context", &later.sha256);
        let command = arguments("collect-teacher-content", &output);
        assert!(resume_engine_at(&command, &directory.path().join("engines")).is_err());
        content_plan(&output, "task-b", "context", &original.sha256);
        artefact::write_json(&output.join("launch-engine.json"), &later).unwrap();
        assert!(resume_engine_at(&command, &directory.path().join("engines")).is_err());
    }

    #[test]
    fn legacy_policy_with_no_engine_bound_work_can_build_its_first_engine() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("content");
        artefact::write_json(&output.join("content-policy.json"), &serde_json::json!({})).unwrap();
        assert!(
            resume_engine_at(
                &arguments("collect-teacher-content", &output),
                &directory.path().join("absent-engine-cache"),
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn explicit_fresh_does_not_consult_previous_corrupt_progress() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evaluation");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("plan.json"), b"interrupted-invalid-fixture").unwrap();
        let mut command = arguments("study-evaluate", &output);
        command.push("--fresh".into());
        assert!(
            resume_engine_at(&command, &directory.path().join("absent-engine-cache"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn runtime_library_search_does_not_require_a_compiler_or_toolkit_binary() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("runtime-only");
        let libraries = root.join("lib64");
        fs::create_dir_all(&libraries).unwrap();
        let caller = directory.path().join("caller-libraries");
        let existing = std::env::join_paths([&caller, &libraries]).unwrap();
        let resolved = runtime_library_paths(&[root], &existing).unwrap();
        assert_eq!(
            std::env::split_paths(&resolved).collect::<Vec<_>>(),
            vec![caller, libraries]
        );
    }
}
