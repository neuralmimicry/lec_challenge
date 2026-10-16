//! Trusted, persistent headless BitNet.cpp inference, supervised by Rust.
//! Construction and inference run on a blocking worker, never an async executor.
use super::*;
use crate::{
    artefact::{self, Artefact},
    model::{Generation, GenerationRequest, RuntimeOptions},
};
use futures_util::StreamExt;
use std::{
    net::TcpListener,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

const ENGINE_COMMIT: &str = "0b341e582afbf9e1011f24744b554c96a3477eb5";
const LLAMA_COMMIT: &str = "390c307752ab78fd8189f359d6954c9ba1be74af";
mod build_lease;
mod host;
mod http;
mod quantisation_v2;
mod supervisor;
pub use host::{NativeCpuRequirements, validate_execution_host};
use http::HttpDriver;
pub use quantisation_v2::bootstrap_quantisation_v2;

/// Controlled Rust process used only to test the production supervisor's
/// lifetime rules. It never runs a candidate programme or model output.
#[doc(hidden)]
pub struct SupervisionProbe {
    inner: Supervisor,
}
impl SupervisionProbe {
    pub fn pid(&self) -> u32 {
        self.inner.pid
    }
}
#[doc(hidden)]
pub fn spawn_supervision_probe(executable: &Path) -> Result<SupervisionProbe> {
    let mut command = Command::new(executable);
    command
        .arg("--leaf")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    Ok(SupervisionProbe {
        inner: Supervisor::spawn(command)?,
    })
}
use supervisor::Supervisor;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineIdentity {
    pub schema_version: u32,
    pub engine_commit: String,
    pub llama_commit: String,
    pub source_directory: PathBuf,
    pub build_directory: PathBuf,
    pub patch: Artefact,
    pub cmake_cache: Artefact,
    pub binaries: Vec<Artefact>,
    pub licences: Vec<Artefact>,
}
fn git(directory: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()?;
    ensure!(
        output.status.success(),
        "cannot inspect pinned engine source"
    );
    Ok(String::from_utf8(output.stdout)?)
}
/// Record this machine's real compiled artifacts, including every dynamic
/// library. The C++ exception is limited to the reviewed inference engine.
pub fn record_engine(source: &Path, build: &Path, output: &Path) -> Result<EngineIdentity> {
    let root = std::env::current_dir()?.canonicalize()?;
    let source = source.canonicalize()?.strip_prefix(&root)?.to_owned();
    let build = build.canonicalize()?.strip_prefix(&root)?.to_owned();
    ensure!(
        git(&source, &["rev-parse", "HEAD"])?.trim() == ENGINE_COMMIT
            && git(&source.join("3rdparty/llama.cpp"), &["rev-parse", "HEAD"])?.trim()
                == LLAMA_COMMIT,
        "engine source/submodule pin differs"
    );
    let patch = git(&source.join("3rdparty/llama.cpp"), &["diff", "HEAD", "--"])?;
    let (schema_version, patch_path) = reviewed_patch(patch.as_bytes())?;
    ensure!(
        git(&source, &["diff", "HEAD", "--ignore-submodules=all", "--"])?.is_empty(),
        "engine parent has unreviewed source changes"
    );
    let cache = fs::read_to_string(build.join("CMakeCache.txt"))?;
    ensure!(
        cache.contains("GGML_CUDA:BOOL=OFF")
            && cache.contains("LLAMA_BUILD_UI:BOOL=OFF")
            && cache.contains("LLAMA_USE_PREBUILT_UI:BOOL=OFF"),
        "engine must be CPU-only with no downloaded or built Web UI"
    );
    let mut binaries = Vec::new();
    for entry in fs::read_dir(build.join("bin"))? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("invalid engine filename")?;
        if name.starts_with("lib")
            || matches!(name, "llama-server" | "llama-cli" | "llama-quantize")
        {
            binaries.push(Artefact::capture(&root, &path)?);
        }
    }
    binaries.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(
        binaries.iter().any(|b| b.path.ends_with("llama-server")),
        "headless server executable absent"
    );
    let identity = EngineIdentity {
        schema_version,
        engine_commit: ENGINE_COMMIT.into(),
        llama_commit: LLAMA_COMMIT.into(),
        source_directory: source.clone(),
        build_directory: build.clone(),
        patch: Artefact::capture(&root, patch_path)?,
        cmake_cache: Artefact::capture(&root, build.join("CMakeCache.txt"))?,
        binaries,
        licences: [
            "BitNet-ENGINE-LICENSE",
            "LLAMA-CPP-LICENSE",
            "BitNet-MODEL-LICENSE",
            "upstream-notices/cpp-httplib-LICENSE",
            "upstream-notices/jsonhpp-LICENSE",
            "upstream-notices/gguf-py-LICENSE",
        ]
        .iter()
        .map(|name| Artefact::capture(&root, Path::new("docs/bitnet/provenance").join(name)))
        .collect::<Result<Vec<_>>>()?,
    };
    if output.exists() {
        let previous: EngineIdentity = serde_json::from_slice(&fs::read(output)?)?;
        ensure!(
            serde_json::to_value(&previous)? == serde_json::to_value(&identity)?,
            "saved engine receipt differs from the current build; use a new receipt path"
        );
    } else {
        artefact::write_json(output, &identity)?;
    }
    Ok(identity)
}

/// Explicit online bootstrap or cached-only rebuild. Normal inference never
/// calls this function and therefore cannot silently acquire network assets.
pub async fn bootstrap_engine(offline: bool, jobs: usize) -> Result<EngineIdentity> {
    host::validate_build_host()?;
    let available = std::thread::available_parallelism()?.get();
    ensure!(
        jobs > 0 && jobs <= available,
        "native build parallelism exceeds the measured CPU allocation"
    );
    let source = PathBuf::from("state/engines-src/bitnet");
    let build = PathBuf::from("state/engines/bitnet-build");
    let receipt = build.join("engine-manifest.json");
    fs::create_dir_all("state/engines")?;
    let _lease =
        build_lease::BuildLease::acquire(Path::new("state/engines/.bitnet-bootstrap.lock")).await?;
    if receipt.exists() {
        let path = receipt.clone();
        return tokio::task::spawn_blocking(move || {
            let identity = verify_engine(&path)?;
            validate_execution_host(&identity)?;
            Ok(identity)
        })
        .await?;
    }
    fs::create_dir_all("state/engines-src")?;
    fs::create_dir_all(&build)?;
    let log = build.join("bootstrap.log");
    if !source.join(".git").exists() {
        ensure!(!offline, "cached BitNet source absent in offline mode");
        run_build(
            &[
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/microsoft/BitNet.git",
                "state/engines-src/bitnet",
            ],
            &log,
        )
        .await?;
    }
    let directory = source.clone();
    let have =
        tokio::task::spawn_blocking(move || git(&directory, &["cat-file", "-t", ENGINE_COMMIT]))
            .await?;
    if have.is_err() {
        ensure!(
            !offline,
            "pinned BitNet commit absent in offline source cache"
        );
        run_build(
            &[
                "git",
                "-C",
                "state/engines-src/bitnet",
                "fetch",
                "origin",
                ENGINE_COMMIT,
            ],
            &log,
        )
        .await?;
    }
    run_build(
        &[
            "git",
            "-C",
            "state/engines-src/bitnet",
            "checkout",
            "--detach",
            ENGINE_COMMIT,
        ],
        &log,
    )
    .await?;
    let submodule = source.join("3rdparty/llama.cpp");
    let sub = submodule.clone();
    let have = tokio::task::spawn_blocking(move || git(&sub, &["rev-parse", "HEAD"])).await?;
    if !have.as_ref().is_ok_and(|s| s.trim() == LLAMA_COMMIT) {
        ensure!(
            !offline,
            "pinned BitNet submodule absent in offline source cache"
        );
        run_build(
            &[
                "git",
                "-C",
                "state/engines-src/bitnet",
                "submodule",
                "update",
                "--init",
                "--depth",
                "1",
            ],
            &log,
        )
        .await?;
    }
    let diff = git(&submodule, &["diff", "--", "src/models/bitnet.cpp"])?;
    if diff.is_empty() {
        let patch = std::env::current_dir()?.join("docs/bitnet/engine-relu2.patch");
        let patch = patch.to_str().context("non-UTF8 engine patch path")?;
        run_build(
            &[
                "git",
                "-C",
                "state/engines-src/bitnet/3rdparty/llama.cpp",
                "apply",
                patch,
            ],
            &log,
        )
        .await?;
    } else {
        ensure!(
            diff.as_bytes() == include_bytes!("../../docs/bitnet/engine-relu2.patch"),
            "cached engine has unreviewed graph changes"
        );
    }
    run_build(
        &[
            "cmake",
            "-S",
            "state/engines-src/bitnet",
            "-B",
            "state/engines/bitnet-build",
            "-DGGML_CUDA=OFF",
            "-DLLAMA_BUILD_COMMON=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
        ],
        &log,
    )
    .await?;
    let cache = build.join("CMakeCache.txt");
    tokio::task::spawn_blocking(move || host::validate_configured_build(&cache)).await??;
    run_build(
        &[
            "cmake",
            "--build",
            "state/engines/bitnet-build",
            "--target",
            "llama-cli",
            "llama-server",
            "-j",
            &jobs.to_string(),
        ],
        &log,
    )
    .await?;
    tokio::task::spawn_blocking(move || {
        let identity = record_engine(&source, &build, &receipt)?;
        validate_execution_host(&identity)?;
        Ok(identity)
    })
    .await?
}
async fn run_build(arguments: &[&str], log: &Path) -> Result<()> {
    let file = fs::OpenOptions::new().create(true).append(true).open(log)?;
    let mut command = tokio::process::Command::new(arguments[0]);
    command
        .args(&arguments[1..])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file))
        .kill_on_drop(true);
    let mut child = command.spawn()?;
    let started = Instant::now();
    let mut tick = tokio::time::interval(Duration::from_secs(15));
    loop {
        tokio::select! {status=child.wait()=>{ensure!(status?.success(),"native engine build stage {} failed; inspect {}",arguments[0],log.display());return Ok(());},_=tick.tick()=>{ensure!(started.elapsed()<Duration::from_secs(3600),"native engine build stage deadline exceeded");eprintln!("BitNet native {} progress: {:.0}s; log {}",arguments[0],started.elapsed().as_secs_f64(),log.display());}}
    }
}
pub fn verify_engine(path: &Path) -> Result<EngineIdentity> {
    let root = std::env::current_dir()?.canonicalize()?;
    let identity: EngineIdentity = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        matches!(identity.schema_version, 1 | 2)
            && identity.engine_commit == ENGINE_COMMIT
            && identity.llama_commit == LLAMA_COMMIT,
        "unsupported BitNet.cpp engine identity"
    );
    ensure!(
        identity.patch.sha256
            == artefact::hash(if identity.schema_version == 1 {
                include_bytes!("../../docs/bitnet/engine-relu2.patch").as_slice()
            } else {
                include_bytes!("../../docs/bitnet/engine-quantisation-v2.patch").as_slice()
            }),
        "engine patch identity changed"
    );
    identity.patch.verify(&root)?;
    identity.cmake_cache.verify(&root)?;
    for file in identity.binaries.iter().chain(&identity.licences) {
        file.verify(&root)?;
    }
    ensure!(
        identity
            .binaries
            .iter()
            .any(|b| b.path == identity.build_directory.join("bin/llama-server")),
        "engine receipt does not bind its executable"
    );
    Ok(identity)
}

fn reviewed_patch(bytes: &[u8]) -> Result<(u32, &'static str)> {
    if bytes == include_bytes!("../../docs/bitnet/engine-relu2.patch") {
        Ok((1, "docs/bitnet/engine-relu2.patch"))
    } else if bytes == include_bytes!("../../docs/bitnet/engine-quantisation-v2.patch") {
        Ok((2, "docs/bitnet/engine-quantisation-v2.patch"))
    } else {
        anyhow::bail!("engine differs from both reviewed, versioned numerical patches")
    }
}

#[derive(Clone, Debug)]
pub struct CppRuntimeOptions {
    pub engine_manifest: PathBuf,
    pub gguf: PathBuf,
    pub threads: usize,
    pub context_tokens: usize,
    pub cpu_mask: Option<String>,
    pub log_directory: PathBuf,
}
impl Default for CppRuntimeOptions {
    fn default() -> Self {
        Self {
            engine_manifest: "state/engines/bitnet-build/engine-manifest.json".into(),
            gguf: "offline_bundle/students/bitnet-2b-4t/model.i2_s.gguf".into(),
            threads: assigned_threads(),
            context_tokens: 4096,
            cpu_mask: None,
            log_directory: "reports/comparative/bitnet/engine-runs".into(),
        }
    }
}
fn assigned_threads() -> usize {
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(available)
        .min(available)
}
pub struct BitNetCppRuntime {
    transport: HttpDriver,
    child: Option<Supervisor>,
    endpoint: String,
    tokenizer: Tokenizer,
    assets: AssetManifest,
    checkpoint_id: String,
    options: CppRuntimeOptions,
    engine: EngineIdentity,
    native_cpu_requirements: NativeCpuRequirements,
    gguf: Artefact,
    engine_manifest: Artefact,
    log: PathBuf,
    affinity: String,
}
impl Drop for BitNetCppRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}
impl BitNetCppRuntime {
    pub fn load(base: &Path, checkpoint: &Path) -> Result<Self> {
        let options = CppRuntimeOptions {
            gguf: checkpoint.join("model.i2_s.gguf"),
            ..Default::default()
        };
        Self::load_with_options(base, Some(checkpoint), options)
    }
    pub fn load_with_options(
        base: &Path,
        checkpoint: Option<&Path>,
        options: CppRuntimeOptions,
    ) -> Result<Self> {
        ensure!(
            tokio::runtime::Handle::try_current().is_err(),
            "construct BitNet.cpp runtime on a blocking worker"
        );
        let available = std::thread::available_parallelism()?.get();
        ensure!(
            options.threads > 0
                && options.threads <= available
                && (64..=4096).contains(&options.context_tokens),
            "invalid bounded native inference resources"
        );
        if let Some(mask) = &options.cpu_mask {
            ensure!(
                !mask.is_empty()
                    && mask.len() <= 256
                    && mask.bytes().all(|b| b.is_ascii_hexdigit()),
                "CPU affinity mask must contain only bounded hexadecimal digits"
            );
        }
        let root = std::env::current_dir()?.canonicalize()?;
        let engine = verify_engine(&options.engine_manifest)?;
        let native_cpu_requirements = validate_execution_host(&engine)?;
        let exported = super::export::export(base, checkpoint, &options.gguf)?;
        let gguf: Artefact = serde_json::from_value(exported["gguf"].clone())?;
        // export() has just independently verified both original model assets
        // and the final GGUF receipt. Reuse that verified return value rather
        // than hashing the same multi-gigabyte files twice in this factory.
        let assets: AssetManifest = serde_json::from_value(exported["binding"]["source"].clone())?;
        let checkpoint_id = if let Some(path) = checkpoint {
            crate::distillation::checkpoint::validate_checkpoint(path, &assets)?.adapter_sha256
        } else {
            assets.files["model.safetensors"].clone()
        };
        let tokenizer = Tokenizer::from_file(base.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        fs::create_dir_all(&options.log_directory)?;
        let log = options
            .log_directory
            .join(format!("{}.log", uuid::Uuid::new_v4()));
        let transport = HttpDriver::new()?;
        let affinity = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines().find_map(|l| {
                    l.strip_prefix("Cpus_allowed_list:")
                        .map(str::trim)
                        .map(str::to_owned)
                })
            })
            .unwrap_or_else(|| "inherited operating-system affinity".into());
        let mut result = Self {
            transport,
            child: None,
            endpoint: String::new(),
            tokenizer,
            assets,
            checkpoint_id,
            options: options.clone(),
            engine,
            native_cpu_requirements,
            gguf,
            engine_manifest: Artefact::capture(&root, &options.engine_manifest)?,
            log,
            affinity,
        };
        result.start()?;
        result.verify_tokeniser()?;
        Ok(result)
    }
    fn stop(&mut self) {
        drop(self.child.take());
    }
    fn start(&mut self) -> Result<()> {
        let result = self.start_inner();
        if result.is_err() {
            self.stop();
        }
        result
    }
    fn start_inner(&mut self) -> Result<()> {
        ensure!(self.child.is_none(), "native server already started");
        validate_execution_host(&self.engine)?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        drop(listener);
        self.endpoint = format!("http://127.0.0.1:{port}");
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log)?;
        let mut command = Command::new(self.engine.build_directory.join("bin/llama-server"));
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C.UTF-8")
            .args(["--offline", "--host", "127.0.0.1", "--port"])
            .arg(port.to_string())
            .arg("--model")
            .arg(&self.options.gguf)
            .arg("--threads")
            .arg(self.options.threads.to_string())
            .arg("--threads-batch")
            .arg(self.options.threads.to_string())
            .arg("--ctx-size")
            .arg(self.options.context_tokens.to_string())
            .args([
                "--parallel",
                "1",
                "--n-gpu-layers",
                "0",
                "--batch-size",
                "512",
                "--ubatch-size",
                "128",
                "--flash-attn",
                "off",
                "--cache-type-k",
                "f32",
                "--cache-type-v",
                "f32",
                "--no-webui",
                "--no-jinja",
                "--log-colors",
                "off",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        if let Some(mask) = &self.options.cpu_mask {
            command.args(["--cpu-mask", mask, "--cpu-strict", "1"]);
        }
        self.child = Some(Supervisor::spawn(command)?);
        let started = Instant::now();
        let mut heartbeat = Instant::now();
        loop {
            if let Some(status) = self
                .child
                .as_ref()
                .context("native child missing")?
                .exited()
            {
                self.child = None;
                anyhow::bail!(
                    "BitNet.cpp exited before readiness ({status}); inspect {}",
                    self.log.display()
                );
            }
            if self.http("health", None, Duration::from_secs(2)).is_ok() {
                return Ok(());
            }
            ensure!(
                started.elapsed() < Duration::from_secs(120),
                "native BitNet.cpp readiness deadline exceeded; log {}",
                self.log.display()
            );
            if heartbeat.elapsed() >= Duration::from_secs(15) {
                eprintln!(
                    "BitNet.cpp waiting for pinned GGUF readiness: {:.0}s; log {}",
                    started.elapsed().as_secs_f64(),
                    self.log.display()
                );
                heartbeat = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    fn http(
        &self,
        path: &str,
        body: Option<serde_json::Value>,
        deadline: Duration,
    ) -> Result<serde_json::Value> {
        self.transport
            .call(format!("{}/{path}", self.endpoint), body, deadline)
    }
    fn verify_tokeniser(&self) -> Result<()> {
        for text in [
            "<|begin_of_text|>User: Hello!<|eot_id|>Assistant: ",
            "Rust: fn solve(x: i64) -> i64 { x + 1 }\n",
        ] {
            let expected = self
                .tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!(e.to_string()))?
                .get_ids()
                .to_vec();
            let actual = self.http(
                "tokenize",
                Some(serde_json::json!({"content":text,"add_special":false})),
                Duration::from_secs(10),
            )?;
            let ids: Vec<u32> = serde_json::from_value(actual["tokens"].clone())?;
            ensure!(
                ids == expected,
                "native GGUF token IDs differ from original HF tokeniser"
            );
        }
        Ok(())
    }
    pub fn execution_metadata(&self) -> serde_json::Value {
        let mut metadata = serde_json::json!({"engine":"pinned_bitnet_cpp_supervised_by_rust","architecture":"bitnet","model_id":self.assets.model_id,"revision":self.assets.revision,"checkpoint_id":self.checkpoint_id,"requested":RuntimeOptions::default(),"actual_device":"Cpu","actual_dtype":"F32","cuda_active":false,"cpu_fallback":false,"evaluation_only":true,"production_eligible":false,"engine_manifest":self.engine_manifest,"gguf":self.gguf,"cpu_threads":self.options.threads,"cpu_mask":self.options.cpu_mask,"inherited_affinity":self.affinity,"context_tokens":self.options.context_tokens,"kv_cache":true,"original_tokeniser_parity_verified":true,"log":self.log});
        metadata["native_server_pid"] = serde_json::to_value(self.child.as_ref().map(|s| s.pid))
            .unwrap_or(serde_json::Value::Null);
        metadata["native_cpu_requirements"] = serde_json::json!(self.native_cpu_requirements);
        metadata["lifetime_supervision"] = serde_json::json!(
            "persistent_native_owner_thread_linux_pdeathsig_sigkill_parent_pid_race_check"
        );
        metadata["key_cache_dtype"] = serde_json::json!("F32");
        metadata["value_cache_dtype"] = serde_json::json!("F32");
        metadata["inference_protocol"] = serde_json::json!(if self.engine.schema_version == 2 {
            "explicit_original_tokens_cpu_f32_kv_quant_v3"
        } else {
            "explicit_original_tokens_cpu_f32_kv_v2"
        });
        metadata
    }
    /// Small pre-sampling log-probability probe for independent Rust/native
    /// parity. It is a numerical check, not a task-quality evaluation.
    pub fn next_token_probe(&mut self, prompt: &str, top: usize) -> Result<serde_json::Value> {
        ensure!(
            (1..=64).contains(&top),
            "parity vocabulary probe outside bound"
        );
        let ids = self
            .tokenizer
            .encode(formatted_prompt(prompt), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        ensure!(
            !ids.is_empty() && ids.len() < self.options.context_tokens,
            "parity prompt exceeds native context"
        );
        self.http("completion",Some(serde_json::json!({"prompt":ids,"n_predict":1,"temperature":0.,"top_p":1.,"seed":1729,"cache_prompt":false,"stream":false,"return_tokens":true,"n_probs":top,"post_sampling_probs":false})),Duration::from_secs(120))
    }
    pub fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        ensure!(
            tokio::runtime::Handle::try_current().is_err(),
            "run native BitNet.cpp generation on a blocking worker"
        );
        ensure!(
            (1..=1024).contains(&request.max_new_tokens)
                && request.temperature.is_finite()
                && (0. ..=2.).contains(&request.temperature)
                && request.top_p.is_finite()
                && request.top_p > 0.
                && request.top_p <= 1.
                && (1..=3600).contains(&request.timeout_seconds),
            "invalid bounded native generation request"
        );
        if self.child.is_none() {
            self.start()?;
            if let Err(error) = self.verify_tokeniser() {
                self.stop();
                return Err(error);
            }
        }
        let started = Instant::now();
        let text = formatted_prompt(&request.prompt);
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        let count = ids.len();
        ensure!(
            count > 0 && count + request.max_new_tokens <= self.options.context_tokens,
            "native BitNet input plus reserved output exceeds context"
        );
        let response=self.http("completion",Some(serde_json::json!({"prompt":ids,"n_predict":request.max_new_tokens,"temperature":request.temperature,"top_p":request.top_p,"seed":request.seed,"cache_prompt":true,"stream":false,"return_tokens":true,"stop":["<|eot_id|>","<|end_of_text|>"]})),Duration::from_secs(request.timeout_seconds));
        let response = match response {
            Ok(value) => value,
            Err(error) => {
                self.stop();
                return Err(error.context(
                    "native inference stopped; completed previous observations remain reusable",
                ));
            }
        };
        let completion = response["content"]
            .as_str()
            .context("native completion missing")?
            .to_owned();
        let output_tokens = response["tokens_predicted"]
            .as_u64()
            .or_else(|| response["timings"]["predicted_n"].as_u64())
            .context("native output token usage absent")? as usize;
        ensure!(
            output_tokens <= request.max_new_tokens
                && (completion.trim().is_empty() || output_tokens > 0)
                && response["truncated"] != true,
            "native output token/context accounting invalid"
        );
        let stop = response["stop_type"].as_str().unwrap_or("");
        let complete = response["stopped_eos"] == true
            || response["stopped_word"] == true
            || matches!(stop, "eos" | "word");
        let reason = if complete { "stop" } else { "length" };
        Ok(Generation {
            completion,
            checkpoint_id: self.checkpoint_id.clone(),
            finish_reason: reason.into(),
            input_tokens: count,
            output_tokens,
            elapsed_ms: started.elapsed().as_millis(),
            complete,
        })
    }
}

#[cfg(test)]
mod patch_tests {
    use super::*;

    #[test]
    fn numerical_patch_versions_remain_distinct_and_reject_extra_edits() -> Result<()> {
        let original = include_bytes!("../../docs/bitnet/engine-relu2.patch");
        let numerical = include_bytes!("../../docs/bitnet/engine-quantisation-v2.patch");
        assert_eq!(reviewed_patch(original)?.0, 1);
        assert_eq!(reviewed_patch(numerical)?.0, 2);
        let mut changed = numerical.to_vec();
        changed.extend_from_slice(b"unreviewed change\n");
        assert!(reviewed_patch(&changed).is_err());
        assert!(reviewed_patch(b"").is_err());
        Ok(())
    }
}
