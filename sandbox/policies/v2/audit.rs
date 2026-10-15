//! Trusted Rust ptrace supervisor, compiled inside the sandbox image.
//! Candidate stdout is a separate pipe. No candidate descriptor refers to the
//! supervisor's output; candidate claims can never manufacture an audit event.
use serde_json::json;
mod ptrace_v2;
mod syscalls_v2;
use std::os::raw::{c_int, c_long, c_void};
use std::{
    fs,
    io::{Read, Write},
    os::unix::process::CommandExt,
    process::{Command, Stdio},
};
unsafe extern "C" {
    fn ptrace(request: c_int, pid: c_int, addr: *mut c_void, data: *mut c_void) -> c_long;
    fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
    fn prctl(option: c_int, ...) -> c_int;
    fn __errno_location() -> *mut c_int;
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
        println!(
            "{}",
            json!({"schema_version":2,"architecture":architecture,"trace_complete":false,"compile_ok":false,"candidate_exit":null,"output":"","stderr":String::from_utf8_lossy(&compile.stderr).chars().take(4096).collect::<String>(),"events":[],"failure":"compilation failed"})
        );
        return Ok(());
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
    println!(
        "{}",
        json!({"schema_version":2,"architecture":architecture,"trace_complete":true,"compile_ok":true,"candidate_exit":exit,"output":String::from_utf8_lossy(&output),"stderr":String::from_utf8_lossy(&err),"output_truncated":output_truncated,"stderr_truncated":stderr_truncated,"events":events,"syscalls_observed":count,"failure":null})
    );
    Ok(())
}
