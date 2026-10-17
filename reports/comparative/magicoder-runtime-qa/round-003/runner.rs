//! Temporary, independently compiled Rust runtime diagnostic; no quality score.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn save(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn hash(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn memory(out: &Path, phase: &str) -> Result<Value> {
    let text = fs::read_to_string("/proc/meminfo")?;
    fs::write(out.join(format!("{phase}-meminfo.txt")), &text)?;
    let host = text
        .lines()
        .find_map(|l| {
            l.strip_prefix("MemAvailable:")
                .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        })
        .ok_or("MemAvailable absent")?
        * 1024;
    let groups = fs::read_to_string("/proc/self/cgroup")?;
    fs::write(out.join(format!("{phase}-cgroup.txt")), &groups)?;
    let mut available = host;
    let mut bounds = Vec::new();
    if let Some(group) = groups.lines().find_map(|l| l.strip_prefix("0::")) {
        let root = Path::new("/sys/fs/cgroup");
        let mut d = root.join(group.trim_start_matches('/'));
        loop {
            if let (Ok(max), Ok(current)) = (
                fs::read_to_string(d.join("memory.max")),
                fs::read_to_string(d.join("memory.current")),
            ) {
                if let (Ok(max), Ok(current)) =
                    (max.trim().parse::<u64>(), current.trim().parse::<u64>())
                {
                    available = available.min(max.saturating_sub(current));
                    bounds.push(json!({"path":d,"maximum":max,"current":current}));
                }
            }
            if d == root || !d.pop() || !d.starts_with(root) {
                break;
            }
        }
    }
    let usable = available
        .saturating_sub(available / 10)
        .saturating_sub(512 << 20);
    let v = json!({"host_available_bytes":host,"available_bytes":available,"usable_bytes":usable,"cgroup_bounds":bounds,"minimum_required_usable_bytes":10_u64<<30});
    save(&out.join(format!("{phase}-memory.json")), &v)?;
    eprintln!(
        "{phase}: available {:.2} GiB; guarded usable {:.2} GiB",
        available as f64 / (1_u64 << 30) as f64,
        usable as f64 / (1_u64 << 30) as f64
    );
    if usable < 10_u64 << 30 {
        return Err("less than 10 GiB guarded usable RAM; diagnostic did not load a model".into());
    }
    Ok(v)
}
fn child_command(program: &Path, args: &[String], out: &Path, name: &str) -> Result<Command> {
    save(
        &out.join(format!("{name}-command.json")),
        &json!({"program":program,"program_sha256":hash(program)?,"arguments":args,"environment":{"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","RAYON_NUM_THREADS":"4","OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","TOKIO_WORKER_THREADS":"2","CUDA_VISIBLE_DEVICES":""}}),
    )?;
    let mut c = Command::new(program);
    c.args(args)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("RAYON_NUM_THREADS", "4")
        .env("OMP_NUM_THREADS", "4")
        .env("MKL_NUM_THREADS", "4")
        .env("TOKIO_WORKER_THREADS", "2")
        .env("CUDA_VISIBLE_DEVICES", "")
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(
            out.join(format!("{name}-stdout.txt")),
        )?))
        .stderr(Stdio::from(File::create(
            out.join(format!("{name}-stderr.txt")),
        )?));
    Ok(c)
}
struct Owned(Child);
impl Drop for Owned {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(
    address: SocketAddr,
    method: &str,
    path: &str,
    body: &[u8],
    raw_path: Option<&Path>,
) -> Result<Value> {
    let mut s = TcpStream::connect_timeout(&address, Duration::from_secs(2))?;
    s.set_read_timeout(Some(Duration::from_secs(150)))?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )?;
    s.write_all(body)?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    if let Some(path) = raw_path {
        fs::write(path, &raw)?;
    }
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("HTTP header absent")?;
    let header = std::str::from_utf8(&raw[..split])?;
    if !header.starts_with("HTTP/1.1 200 ") {
        return Err(format!("HTTP failed: {header}").into());
    }
    if header
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        return Err(
            "unexpected chunked response; preserve raw diagnostic rather than infer success".into(),
        );
    }
    Ok(serde_json::from_slice(&raw[split + 4..])?)
}
fn run(out: &Path) -> Result<()> {
    if out.exists() {
        return Err("preserve existing diagnostic; select a new output".into());
    }
    fs::create_dir_all(out)?;
    fs::copy("/tmp/lec-magicoder-runtime-qa.rs", out.join("runner.rs"))?;
    let instruction =
        "Write a Rust function add(a: i64, b: i64) -> i64. Return only the complete function.";
    let rendered = format!(
        "<｜begin▁of▁sentence｜>You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n{instruction}\n\n@@ Response\n"
    );
    let prompt_path = out.join("original-rendered-prompt.txt");
    fs::write(&prompt_path, &rendered)?;
    let model = Path::new("offline_bundle/comparative/magicoder-s-ds-6_7b-q4_k_m");
    let gguf = model.join("magicoder-s-ds-6.7b.Q4_K_M.gguf");
    let original = tokenizers::Tokenizer::from_file(model.join("source/tokenizer.json"))
        .map_err(|e| e.to_string())?;
    let expected_ids = original
        .encode(rendered.as_str(), false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let parity: Value = serde_json::from_slice(&fs::read(
        "reports/comparative/magicoder-runtime-qa/tokeniser-parity/report.json",
    )?)?;
    if parity["success"] != true || parity["native_ids"] != json!(expected_ids) {
        return Err("exact prompt token-ID parity has not been established".into());
    }
    save(&out.join("prompt-tokeniser-parity.json"), &parity)?;
    let request = json!({"prompt":instruction,"max_new_tokens":16,"seed":1729,"temperature":0.0,"top_p":1.0,"timeout_seconds":120});
    save(&out.join("native-request.json"), &request)?;
    let cpp_args=vec!["--model",gguf.to_str().ok_or("model path")?,"--threads","4","--threads-batch","4","--ctx-size","1024","--predict","16","--temp","0","--top-k","0","--top-p","1","--min-p","0","--repeat-penalty","1","--repeat-last-n","0","--seed","1729","--cache-type-k","f32","--cache-type-v","f32","--n-gpu-layers","0","--device","none","--no-op-offload","--flash-attn","off","--fit","off","--offline","--no-warmup","--single-turn","--simple-io","--color","off","--no-show-timings","--verbose-prompt","--log-verbosity","3","--override-kv","tokenizer.ggml.model=str:gpt2,tokenizer.ggml.pre=str:deepseek-coder,tokenizer.ggml.add_bos_token=bool:false","--chat-template","{{ messages[0]['content'] }}","--binary-file",prompt_path.to_str().ok_or("prompt path")?,"--no-escape"].into_iter().map(str::to_owned).collect::<Vec<_>>();
    memory(out, "before-cpp")?;
    eprintln!("Starting C++ CPU oracle, four threads and at most 16 new tokens");
    let started = Instant::now();
    let mut cpp = Owned(
        child_command(
            Path::new("state/engines/bitnet-build/bin/llama-cli"),
            &cpp_args,
            out,
            "cpp",
        )?
        .spawn()?,
    );
    let cpp_exit = loop {
        if let Some(s) = cpp.0.try_wait()? {
            break s;
        }
        if started.elapsed() > Duration::from_secs(300) {
            return Err("C++ oracle deadline exceeded".into());
        }
        thread::sleep(Duration::from_secs(1));
        if started.elapsed().as_secs() % 15 == 0 {
            eprintln!("C++ oracle active: {}s", started.elapsed().as_secs())
        }
    };
    let cpp_seconds = started.elapsed().as_secs_f64();
    if !cpp_exit.success() {
        return Err(format!("C++ oracle exited {cpp_exit}").into());
    }
    drop(cpp);
    eprintln!("C++ oracle exited; all C++ model RAM released");
    let cpp_stdout = fs::read_to_string(out.join("cpp-stdout.txt"))?;
    let marker = format!("{rendered}\n\n");
    let content_start = cpp_stdout
        .rfind(&marker)
        .ok_or("exact original prompt not found in verbose C++ output")?
        + marker.len();
    let tail = "\n\nExiting...\n";
    let cpp_completion = cpp_stdout[content_start..]
        .strip_suffix(tail)
        .ok_or("C++ console suffix differs; inspect raw evidence")?
        .to_owned();
    fs::write(out.join("cpp-completion.txt"), &cpp_completion)?;
    let cpp_log = fs::read_to_string(out.join("cpp-stderr.txt"))?;
    let cpp_prompt_tokens = cpp_log
        .lines()
        .find(|line| line.contains("prompt eval time ="))
        .and_then(|line| line.split('/').nth(1))
        .and_then(|part| part.split_whitespace().next())
        .ok_or("C++ prompt token count absent")?
        .parse::<usize>()?;
    save(
        &out.join("cpp-result.json"),
        &json!({"exit_success":cpp_exit.success(),"wall_seconds":cpp_seconds,"prompt_tokens":cpp_prompt_tokens,"completion":cpp_completion,"generation_limit_tokens":16}),
    )?;
    if cpp_prompt_tokens != expected_ids.len() {
        return Err(
            "C++ frontend changed the independently verified prompt token denominator".into(),
        );
    }
    memory(out, "before-native")?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    drop(listener);
    let config = out.join("native-serving.toml");
    fs::write(
        &config,
        format!(
            "address = \"{address}\"\nqueue_capacity = 1\nmaximum_request_bytes = 32768\nmaximum_output_tokens = 16\ntimeout_seconds = 120\n"
        ),
    )?;
    let native = PathBuf::from(
        "state/engines/949b28471a3b47255ab102ead671a6a3e719dc1303844c1c66e0d69682d1dc99/lec-distil",
    );
    let native_args = vec![
        "serve".into(),
        "--device".into(),
        "cpu".into(),
        "--dtype".into(),
        "f32".into(),
        "--comparative".into(),
        "--base".into(),
        model.display().to_string(),
        "--config".into(),
        config.display().to_string(),
    ];
    eprintln!("Starting original pinned Rust CPU runtime after C++ exit");
    let native_started = Instant::now();
    let mut server = Owned(child_command(&native, &native_args, out, "native")?.spawn()?);
    let health = loop {
        if let Some(s) = server.0.try_wait()? {
            return Err(format!("native endpoint exited during load: {s}").into());
        }
        match http(address, "GET", "/health", &[], None) {
            Ok(v) => break v,
            Err(_) => {
                if native_started.elapsed() > Duration::from_secs(180) {
                    return Err("native load deadline exceeded".into());
                }
                thread::sleep(Duration::from_secs(1));
                if native_started.elapsed().as_secs() % 15 == 0 {
                    eprintln!(
                        "Rust model loading: {}s",
                        native_started.elapsed().as_secs()
                    )
                }
            }
        }
    };
    save(&out.join("native-health.json"), &health)?;
    if health["execution"]["actual_device"] != "Cpu"
        || health["execution"]["candle_cpu_threads"] != 4
        || health["execution"]["cuda_active"] != false
    {
        return Err("native runtime did not honour CPU/four-thread allocation".into());
    }
    let generated = http(
        address,
        "POST",
        "/v1/generate",
        &serde_json::to_vec(&request)?,
        Some(&out.join("native-response.http")),
    )?;
    save(&out.join("native-generation.json"), &generated)?;
    drop(server);
    eprintln!("Rust endpoint stopped; model RAM released");
    if generated["input_tokens"] != expected_ids.len() {
        return Err(
            "native frontend changed the independently verified prompt token denominator".into(),
        );
    }
    let native_completion = generated["completion"]
        .as_str()
        .ok_or("native completion absent")?;
    fs::write(out.join("native-completion.txt"), native_completion)?;
    let exact = cpp_completion == native_completion;
    let normalised = cpp_completion.trim() == native_completion.trim();
    let report = json!({"schema_version":1,"evidence_kind":"synthetic_magicoder_cross_backend_runtime_diagnostic","recorded_unix_ms":SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),"student_learning_claim":false,"held_out_used":false,"frozen_search_trials_modified":false,"synthetic_instruction":instruction,"original_source_config_sha256":hash(&model.join("source/config.json"))?,"original_source_tokeniser_sha256":hash(&model.join("source/tokenizer.json"))?,"original_source_template_sha256":hash(&model.join("source/tokenizer_config.json"))?,"original_rendered_prompt_sha256":hash(&prompt_path)?,"gguf_sha256":hash(&gguf)?,"cpp_exit_success":cpp_exit.success(),"cpp_wall_seconds":cpp_seconds,"native_total_seconds":native_started.elapsed().as_secs_f64(),"native_health":health,"native_generation":generated,"cpp_completion":cpp_completion,"byte_exact_completion_match":exact,"whitespace_trimmed_completion_match":normalised,"success":exact,"limitation":"Synthetic smoke of the same publisher GGUF and original prompt on two CPU backends; not a development reassessment, quality score, learning or promotion result. Floating-point kernels differ. C++ cache is F32; quantised matrix accumulation implementation may differ."});
    save(&out.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !exact {
        return Err("runtime mismatch; halt quality classification and diagnose against preserved raw evidence".into());
    }
    Ok(())
}
fn main() {
    let output = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .unwrap_or_else(|| "reports/comparative/magicoder-runtime-qa/round-001".into()),
    );
    if let Err(error) = run(&output) {
        let _ = save(
            &output.join("failure.json"),
            &json!({"success":false,"error":error.to_string()}),
        );
        eprintln!("diagnostic failed: {error}");
        std::process::exit(1)
    }
}
