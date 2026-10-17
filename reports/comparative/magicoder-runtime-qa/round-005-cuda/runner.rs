//! Explicit CUDA diagnostic reusing a retained exact-input C++ CPU reference.
#[allow(dead_code)]
#[path = "/tmp/lec-magicoder-runtime-qa.rs"]
mod shared;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
struct Owned(Child);
impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn save(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn run(out: &Path) -> Result<()> {
    if out.exists() {
        return Err("retain existing diagnostic output".into());
    }
    fs::create_dir_all(out)?;
    fs::copy("/tmp/lec-magicoder-cuda-qa.rs", out.join("runner.rs"))?;
    fs::copy(
        "/tmp/lec-magicoder-runtime-qa.rs",
        out.join("shared-runner.rs"),
    )?;
    let reference = Path::new("reports/comparative/magicoder-runtime-qa/round-004");
    let cpp: Value = serde_json::from_slice(&fs::read(reference.join("cpp-result.json"))?)?;
    let parity: Value =
        serde_json::from_slice(&fs::read(reference.join("prompt-tokeniser-parity.json"))?)?;
    if cpp["exit_success"] != true || cpp["prompt_tokens"] != 62 || parity["success"] != true {
        return Err("retained exact-input C++ reference failed verification".into());
    }
    let request: Value = serde_json::from_slice(&fs::read(reference.join("native-request.json"))?)?;
    for name in [
        "cpp-command.json",
        "cpp-result.json",
        "cpp-stdout.txt",
        "cpp-stderr.txt",
        "cpp-completion.txt",
        "original-rendered-prompt.txt",
        "prompt-tokeniser-parity.json",
        "native-request.json",
    ] {
        fs::copy(reference.join(name), out.join(name))?;
    }
    let probe = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()?;
    fs::write(out.join("before-native-gpu.txt"), &probe.stdout)?;
    if !probe.status.success()
        || String::from_utf8(probe.stdout)?
            .lines()
            .next()
            .ok_or("GPU memory absent")?
            .trim()
            .parse::<u64>()?
            < 5500
    {
        return Err("less than 5500 MiB measured free GPU memory".into());
    }
    shared::diagnostic_memory(out, "before-native")?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);
    let config = out.join("serving.toml");
    fs::write(
        &config,
        format!(
            "address=\"{address}\"\nqueue_capacity=1\nmaximum_request_bytes=32768\nmaximum_output_tokens=16\ntimeout_seconds=120\n"
        ),
    )?;
    let executable = Path::new(
        "state/engines/949b28471a3b47255ab102ead671a6a3e719dc1303844c1c66e0d69682d1dc99/lec-distil",
    );
    let args = vec![
        "serve".to_owned(),
        "--device".into(),
        "cuda".into(),
        "--dtype".into(),
        "f32".into(),
        "--comparative".into(),
        "--base".into(),
        "offline_bundle/comparative/magicoder-s-ds-6_7b-q4_k_m".into(),
        "--config".into(),
        config.display().to_string(),
    ];
    let libraries = std::env::var_os("LD_LIBRARY_PATH")
        .unwrap_or_else(|| "/usr/local/cuda-12.9/lib64:/usr/local/cuda/lib64".into());
    save(
        &out.join("native-command.json"),
        &json!({"program":executable,"program_sha256":shared::diagnostic_hash(executable)?,"arguments":args,"environment":{"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","RAYON_NUM_THREADS":"4","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","TOKIO_WORKER_THREADS":"2","CUDA_VISIBLE_DEVICES":"0","LD_LIBRARY_PATH":libraries}}),
    )?;
    let mut server = Owned(
        Command::new(executable)
            .args(&args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C.UTF-8")
            .env("RAYON_NUM_THREADS", "4")
            .env("OMP_NUM_THREADS", "4")
            .env("MKL_NUM_THREADS", "4")
            .env("TOKIO_WORKER_THREADS", "2")
            .env("CUDA_VISIBLE_DEVICES", "0")
            .env("LD_LIBRARY_PATH", libraries)
            .stdin(Stdio::null())
            .stdout(Stdio::from(fs::File::create(
                out.join("native-stdout.txt"),
            )?))
            .stderr(Stdio::from(fs::File::create(
                out.join("native-stderr.txt"),
            )?))
            .spawn()?,
    );
    let started = Instant::now();
    eprintln!("Loading original pinned native CUDA/F32 Magicoder after CPU reference completion");
    let health = loop {
        if let Some(s) = server.0.try_wait()? {
            return Err(format!("CUDA endpoint exited during load: {s}").into());
        }
        match shared::diagnostic_http(address, "GET", "/health", &[], None) {
            Ok(v) => break v,
            Err(_) => {
                if started.elapsed() > Duration::from_secs(180) {
                    return Err("CUDA model load deadline exceeded".into());
                }
                thread::sleep(Duration::from_secs(1));
                if started.elapsed().as_secs() % 15 == 0 {
                    eprintln!("CUDA loading {}s", started.elapsed().as_secs())
                }
            }
        }
    };
    save(&out.join("native-health.json"), &health)?;
    if health["execution"]["cuda_active"] != true
        || health["execution"]["actual_dtype"] != "F32"
        || health["execution"]["candle_cpu_threads"] != 4
    {
        return Err(
            "CUDA runtime did not honour explicit device/precision/thread allocation".into(),
        );
    }
    let generation = shared::diagnostic_http(
        address,
        "POST",
        "/v1/generate",
        &serde_json::to_vec(&request)?,
        Some(&out.join("native-response.http")),
    )?;
    save(&out.join("native-generation.json"), &generation)?;
    drop(server);
    eprintln!("Native CUDA process exited; GPU allocation released");
    if generation["input_tokens"] != 62 || generation["output_tokens"] != 16 {
        return Err("native generation changed the matched input/output token denominators".into());
    }
    let cpp_completion = cpp["completion"].as_str().ok_or("C++ completion absent")?;
    let native_completion = generation["completion"]
        .as_str()
        .ok_or("native completion absent")?;
    let exact = cpp_completion == native_completion;
    let report = json!({"schema_version":1,"evidence_kind":"explicit_native_cuda_vs_retained_cpp_cpu_magicoder_synthetic_diagnostic","success":exact,"byte_exact_completion_match":exact,"original_prompt_token_ids":parity["native_ids"],"matched_input_tokens":62,"matched_new_tokens":16,"greedy_temperature":0,"repetition_penalty":1,"cpp_reference":reference,"cpp_result_sha256":shared::diagnostic_hash(&reference.join("cpp-result.json"))?,"cpp_result":cpp,"native_health":health,"native_generation":generation,"native_wall_seconds":started.elapsed().as_secs_f64(),"held_out_used":false,"frozen_trials_modified":false,"model_quality_or_learning_claim":false,"limitation":"Synthetic short-prefix diagnostic with identical original prompt/token IDs and publisher GGUF. Reference is CPU/F32; native is explicit CUDA/F32 with expanded F16 embedding. It does not establish full numerical parity, long-decoding equivalence or coding quality."});
    save(&out.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !exact {
        return Err(
            "short greedy runtime output differs; halt quality classification and diagnose".into(),
        );
    }
    Ok(())
}
fn main() {
    let out = PathBuf::from("reports/comparative/magicoder-runtime-qa/round-005-cuda");
    if let Err(error) = run(&out) {
        let _ = save(
            &out.join("failure.json"),
            &json!({"success":false,"error":error.to_string()}),
        );
        eprintln!("CUDA diagnostic failed: {error}");
        std::process::exit(1)
    }
}
