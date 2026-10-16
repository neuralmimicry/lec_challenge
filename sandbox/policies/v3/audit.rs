//! Trusted Rust PID 1 supervisor with an independent EOF-armed deadline.
//! Candidate stdout is a separate pipe. No candidate descriptor refers to the
//! supervisor's output; candidate claims can never manufacture an audit event.
use serde_json::{Value, json};
mod ptrace_v2;
mod syscalls_v2;
use std::os::raw::{c_int, c_long, c_void};
use std::{
    fs,
    io::{Read, Write},
    os::unix::process::CommandExt,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
unsafe extern "C" {
    fn ptrace(request: c_int, pid: c_int, addr: *mut c_void, data: *mut c_void) -> c_long;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn prctl(option: c_int, ...) -> c_int;
    fn __errno_location() -> *mut c_int;
    fn getpid() -> c_int;
    fn fcntl(fd: c_int, command: c_int, ...) -> c_int;
    fn write(fd: c_int, buffer: *const c_void, count: usize) -> isize;
    fn _exit(status: c_int) -> !;
}
const DEADLINE_MS: u64 = 20_000;
const UNCLAIMED: u8 = 0;
const NORMAL: u8 = 1;
const DEADLINE: u8 = 2;

/// Exactly one trusted writer may publish the terminal audit. This state never
/// grants extra execution time: the watchdog exits PID 1 even if a normal
/// writer has claimed stdout but then blocks while flushing it.
struct Terminal {
    state: Arc<AtomicU8>,
    architecture: &'static str,
    started: Instant,
    started_unix_ms: u64,
}
fn elapsed_digits(value: u64, output: &mut [u8]) {
    output.fill(b' ');
    let mut remaining = value;
    let mut index = output.len();
    loop {
        index -= 1;
        output[index] = b'0' + (remaining % 10) as u8;
        remaining /= 10;
        if remaining == 0 {
            break;
        }
    }
}
fn failure(architecture: &str, message: &str) -> Value {
    json!({"schema_version":3,"architecture":architecture,"trace_complete":false,
        "compile_ok":false,"candidate_exit":null,"output":"","stderr":"","events":[],"failure":message})
}
impl Terminal {
    fn arm(architecture: &'static str) -> Result<Self, Box<dyn std::error::Error>> {
        // A PID 1 exit kills the remaining PID-namespace processes. Running the
        // supervisor as an ordinary child cannot provide that guarantee.
        if unsafe { getpid() } != 1 {
            return Err("trusted supervisor must be PID 1".into());
        }
        if unsafe { prctl(4, 0i32, 0i32, 0i32, 0i32) } != 0 {
            return Err("trusted supervisor dumpability protection failed".into());
        }
        let started = Instant::now();
        let started_unix_ms =
            u64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())?;
        let state = Arc::new(AtomicU8::new(UNCLAIMED));
        let watchdog_state = state.clone();
        // Allocate and serialise before candidate compilation. At expiry the
        // watchdog uses only its owned buffer, atomics, clock and raw syscalls.
        let mut envelope = failure(
            architecture,
            "trusted EOF-armed candidate deadline exceeded",
        );
        envelope["deadline_exceeded"] = json!(true);
        envelope["deadline_ms"] = json!(DEADLINE_MS);
        envelope["started_at_eof"] = json!(true);
        envelope["deadline_started_unix_ms"] = json!(started_unix_ms);
        envelope["deadline_elapsed_ms"] = json!(u64::MAX);
        let mut bytes = serde_json::to_vec(&envelope)?;
        bytes.push(b'\n');
        let placeholder = b"18446744073709551615";
        let offset = bytes
            .windows(placeholder.len())
            .position(|window| window == placeholder)
            .ok_or("watchdog elapsed placeholder absent")?;
        if bytes.len() > 4096 {
            return Err("watchdog envelope exceeds atomic pipe write bound".into());
        }
        std::thread::Builder::new()
            .name("trusted-deadline".into())
            .spawn(move || {
                std::thread::sleep(
                    Duration::from_millis(DEADLINE_MS).saturating_sub(started.elapsed()),
                );
                if watchdog_state
                    .compare_exchange(UNCLAIMED, DEADLINE, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    let elapsed = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    elapsed_digits(elapsed, &mut bytes[offset..offset + placeholder.len()]);
                    // Linux F_GETFL/F_SETFL and O_NONBLOCK share these values on
                    // both supported ABIs. A full pipe cannot delay termination.
                    // Raw write has no userspace buffer requiring a flush.
                    unsafe {
                        let flags = fcntl(1, 3);
                        if flags >= 0 && fcntl(1, 4, flags | 0x800) == 0 {
                            let _ = write(1, bytes.as_ptr().cast(), bytes.len());
                        }
                    }
                }
                // Avoid Rust/stdio exit handlers and their potentially held locks.
                // Missing output or a short write remains an explicit failed run.
                unsafe {
                    _exit(124);
                }
            })?;
        let marker = json!({"schema_version":3,"architecture":architecture,"deadline_ms":DEADLINE_MS,
            "started_at_eof":true,"deadline_started_unix_ms":started_unix_ms});
        eprintln!("LEC_TRUSTED_EOF_START {marker}");
        Ok(Self {
            state,
            architecture,
            started,
            started_unix_ms,
        })
    }
    fn publish(&self, mut value: Value) -> Result<(), Box<dyn std::error::Error>> {
        value["schema_version"] = json!(3);
        value["architecture"] = json!(self.architecture);
        value["deadline_exceeded"] = json!(false);
        value["deadline_ms"] = json!(DEADLINE_MS);
        value["started_at_eof"] = json!(true);
        value["deadline_started_unix_ms"] = json!(self.started_unix_ms);
        value["deadline_elapsed_ms"] = json!(u64::try_from(self.started.elapsed().as_millis())?);
        let mut bytes = serde_json::to_vec(&value)?;
        bytes.push(b'\n');
        if self
            .state
            .compare_exchange(UNCLAIMED, NORMAL, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(&bytes)?;
            stdout.flush()?;
        } else {
            // The watchdog owns terminal emission and PID 1 termination. Do
            // not let a simultaneous main return race its envelope with exit 0.
            loop {
                std::thread::park_timeout(Duration::from_secs(1));
            }
        }
        Ok(())
    }
}
fn trace(request: c_int, pid: i32, data: usize) -> i64 {
    unsafe { ptrace(request, pid, std::ptr::null_mut(), data as *mut c_void) as i64 }
}
fn string_at(pid: i32, address: u64) -> Result<String, Box<dyn std::error::Error>> {
    let mut bytes = vec![];
    for offset in (0..4096).step_by(8) {
        // PEEKDATA may legitimately return an all-one word. Clear errno first
        // so an inaccessible address cannot masquerade as those valid bytes.
        unsafe {
            *__errno_location() = 0;
        }
        let word = unsafe {
            ptrace(
                2,
                pid,
                (address + offset) as *mut c_void,
                std::ptr::null_mut(),
            )
        } as u64;
        if word == u64::MAX && unsafe { *__errno_location() } != 0 {
            return Err("candidate pathname cannot be read by trusted supervisor".into());
        }
        for b in word.to_ne_bytes() {
            if b == 0 {
                return Ok(String::from_utf8_lossy(&bytes).into_owned());
            }
            bytes.push(b);
        }
    }
    Err("candidate pathname exceeds trusted PATH_MAX bound".into())
}
fn bounded_read(mut r: impl Read) -> (Vec<u8>, bool) {
    let mut out = vec![];
    let mut truncated = false;
    let mut buf = [0; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if out.len() + n <= 262144 {
                    out.extend_from_slice(&buf[..n]);
                } else {
                    truncated = true;
                }
            }
        }
    }
    (out, truncated)
}
fn main() {
    if let Err(e) = run() {
        eprintln!("trusted supervisor failed: {e}");
        std::process::exit(71);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let architecture = syscalls_v2::Abi::native()
        .ok_or("unsupported native Linux ABI")?
        .name();
    let mut input = vec![];
    std::io::stdin().take(131073).read_to_end(&mut input)?;
    if input.len() > 131072 {
        return Err("input cap exceeded".into());
    }
    let terminal = Terminal::arm(architecture)?;
    match execute(architecture, input) {
        Ok(output) => terminal.publish(output),
        Err(error) => {
            terminal.publish(failure(
                architecture,
                &format!("trusted supervisor failed: {error}"),
            ))?;
            Err(error)
        }
    }
}
fn execute(architecture: &str, input: Vec<u8>) -> Result<Value, Box<dyn std::error::Error>> {
    let source = std::env::args().nth(1).ok_or("source required")?;
    let deps = fs::read_to_string("/opt/deps/serde_json.path")?;
    let compile = Command::new("rustc")
        .args([
            "--edition=2024",
            "--crate-name",
            "candidate",
            "-C",
            "opt-level=1",
            "-L",
            "dependency=/opt/deps",
            "--extern",
            &format!("serde_json={}", deps.trim()),
            &source,
            "-o",
            "/tmp/candidate",
        ])
        .env_clear()
        .env(
            "PATH",
            "/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        )
        .env("RUSTUP_HOME", "/usr/local/rustup")
        .env("CARGO_HOME", "/usr/local/cargo")
        .output()?;
    if !compile.status.success() {
        return Ok(
            json!({"schema_version":3,"architecture":architecture,"trace_complete":false,"compile_ok":false,"candidate_exit":null,"output":"","stderr":String::from_utf8_lossy(&compile.stderr).chars().take(4096).collect::<String>(),"events":[],"failure":"compilation failed"}),
        );
    }
    let mut command = Command::new("/tmp/candidate");
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();
    unsafe {
        command.pre_exec(|| {
            if ptrace(0, 0, std::ptr::null_mut(), std::ptr::null_mut()) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command.spawn()?;
    let pid = child.id() as i32;
    // Prevent /proc/PID/mem and ptrace access to the trusted supervisor.
    unsafe {
        prctl(4, 0i32, 0i32, 0i32, 0i32);
    }
    let mut stdin = child.stdin.take().ok_or("candidate stdin missing")?;
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let stdout = child.stdout.take().ok_or("candidate stdout missing")?;
    let stderr = child.stderr.take().ok_or("candidate stderr missing")?;
    let reader = std::thread::spawn(move || bounded_read(stdout));
    let errors = std::thread::spawn(move || bounded_read(stderr));
    let mut status = 0;
    unsafe {
        if waitpid(pid, &mut status, 0) != pid {
            return Err("initial wait failed".into());
        }
    }
    if status & 0xff != 0x7f {
        return Err("candidate did not enter tracing stop".into());
    }
    if trace(0x4200, pid, 1) < 0 {
        return Err("PTRACE_SETOPTIONS failed".into());
    }
    let (mut blocked, mut events, mut count) = (false, vec![], 0usize);
    let exit = loop {
        if trace(24, pid, 0) < 0 {
            return Err("PTRACE_SYSCALL failed".into());
        }
        unsafe {
            if waitpid(pid, &mut status, 0) != pid {
                return Err("syscall wait failed".into());
            }
        }
        if status & 0x7f == 0 {
            break Some((status >> 8) & 0xff);
        }
        if status & 0xff != 0x7f {
            break Some(128 + (status & 0x7f));
        }
        let signal = (status >> 8) & 0xff;
        if signal != 133 {
            if signal == 5 {
                continue;
            }
            return Err(format!("candidate stopped by signal {signal}").into());
        }
        if let ptrace_v2::Stop::Entry {
            abi,
            number,
            arguments,
        } = ptrace_v2::syscall_stop(pid)?
        {
            if blocked {
                return Err("syscall entry arrived without the required denial exit stop".into());
            }
            count += 1;
            if count > 200000 {
                return Err("syscall count cap exceeded".into());
            }
            let network = abi.network(number);
            // Descriptor theft, process signalling, symlink aliases and async
            // I/O would undermine lexical path auditing. None is needed by the
            // pure-function candidate ABI, so reject them rather than guessing.
            let process = abi.process(number);
            let mut paths = Vec::new();
            let mut forbidden_file = false;
            for index in abi.paths(number) {
                let path = string_at(pid, arguments[*index])?;
                let normal = path.split('/').all(|part| part != ".." && part != ".");
                let allowed = normal
                    && (path.starts_with("/lib/")
                        || path.starts_with("/lib64/")
                        || path.starts_with("/usr/lib/")
                        || matches!(
                            path.as_str(),
                            "/etc/ld.so.cache"
                                | "/etc/ld.so.preload"
                                | "/proc/self/maps"
                                | "/proc/self/stat"
                                | "/proc/self/exe"
                                | "/candidate/main.rs"
                                | "/dev/null"
                                | "/dev/urandom"
                        ));
                // Only stat operations with AT_EMPTY_PATH may use an empty
                // pathname; no pathname is resolved by that descriptor query.
                let descriptor_stat =
                    path.is_empty() && abi.empty_descriptor_stat(number, &arguments);
                forbidden_file |= !allowed && !descriptor_stat;
                paths.push(path);
            }
            let path = paths.join(" -> ");
            blocked = network || process || forbidden_file || number >= 0x4000_0000;
            if blocked {
                if events.len() >= 256 {
                    return Err("audit event cap exceeded".into());
                }
                events.push(json!({"syscall":number,"category":if network{"network"}else if process{"process"}else{"filesystem"},"target":path,"result":"denied:EPERM"}));
                // A nonexistent syscall cannot execute. Its return is replaced
                // at the exit stop, before any candidate instruction resumes.
                ptrace_v2::deny_entry(pid)?;
            }
        } else if blocked {
            ptrace_v2::deny_exit(pid)?;
            blocked = false;
        }
    };
    let _ = writer.join();
    let (output, output_truncated) = reader.join().map_err(|_| "output reader failed")?;
    let (err, stderr_truncated) = errors.join().map_err(|_| "stderr reader failed")?;
    Ok(
        json!({"schema_version":3,"architecture":architecture,"trace_complete":true,"compile_ok":true,"candidate_exit":exit,"output":String::from_utf8_lossy(&output),"stderr":String::from_utf8_lossy(&err),"output_truncated":output_truncated,"stderr_truncated":stderr_truncated,"events":events,"syscalls_observed":count,"failure":null}),
    )
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    #[test]
    fn elapsed_placeholder_remains_valid_json_for_all_widths() {
        for value in [0, 1, 19_999, 20_000, 20_137, u64::MAX] {
            let mut digits = [b' '; 20];
            elapsed_digits(value, &mut digits);
            let bytes = [b"{\"deadline_elapsed_ms\":".as_slice(), &digits, b"}"].concat();
            let parsed: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(parsed["deadline_elapsed_ms"], value);
        }
    }
    #[test]
    fn normal_and_watchdog_have_exactly_one_terminal_claim() {
        let state = Arc::new(AtomicU8::new(UNCLAIMED));
        let workers = (0..16)
            .map(|index| {
                let state = state.clone();
                std::thread::spawn(move || {
                    state
                        .compare_exchange(
                            UNCLAIMED,
                            if index % 2 == 0 { NORMAL } else { DEADLINE },
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .filter(|won| *won)
                .count(),
            1
        );
    }
}
