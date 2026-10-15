//! Trusted Rust ptrace supervisor, compiled inside the sandbox image.
//! Candidate stdout is a separate pipe. No candidate descriptor refers to the
//! supervisor's output; candidate claims can never manufacture an audit event.
use serde_json::json;
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
}
#[repr(C)]
#[derive(Default)]
struct Registers {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbp: u64,
    rbx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    rax: u64,
    rcx: u64,
    rdx: u64,
    rsi: u64,
    rdi: u64,
    orig_rax: u64,
    rip: u64,
    cs: u64,
    eflags: u64,
    rsp: u64,
    ss: u64,
    fs_base: u64,
    gs_base: u64,
    ds: u64,
    es: u64,
    fs: u64,
    gs: u64,
}
fn trace(request: c_int, pid: i32, data: usize) -> i64 {
    unsafe { ptrace(request, pid, std::ptr::null_mut(), data as *mut c_void) as i64 }
}
fn string_at(pid: i32, address: u64) -> String {
    let mut bytes = vec![];
    for offset in (0..512).step_by(8) {
        let word = unsafe {
            ptrace(
                2,
                pid,
                (address + offset) as *mut c_void,
                std::ptr::null_mut(),
            )
        } as u64;
        for b in word.to_ne_bytes() {
            if b == 0 {
                return String::from_utf8_lossy(&bytes).into_owned();
            }
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
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
            json!({"schema_version":1,"trace_complete":false,"compile_ok":false,"candidate_exit":null,"output":"","stderr":String::from_utf8_lossy(&compile.stderr).chars().take(4096).collect::<String>(),"events":[],"failure":"compilation failed"})
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
    let (mut entering, mut blocked, mut events, mut count, mut exit) =
        (true, false, vec![], 0usize, None);
    loop {
        if trace(24, pid, 0) < 0 {
            return Err("PTRACE_SYSCALL failed".into());
        }
        unsafe {
            if waitpid(pid, &mut status, 0) != pid {
                return Err("syscall wait failed".into());
            }
        }
        if status & 0x7f == 0 {
            exit = Some((status >> 8) & 0xff);
            break;
        }
        if status & 0xff != 0x7f {
            exit = Some(128 + (status & 0x7f));
            break;
        }
        let signal = (status >> 8) & 0xff;
        if signal != 133 {
            if signal == 5 {
                continue;
            }
            return Err(format!("candidate stopped by signal {signal}").into());
        }
        let mut regs = Registers::default();
        if trace(12, pid, (&mut regs as *mut Registers) as usize) < 0 {
            return Err("PTRACE_GETREGS failed".into());
        }
        if entering {
            count += 1;
            if count > 200000 {
                return Err("syscall count cap exceeded".into());
            }
            let number = regs.orig_rax;
            let network = matches!(
                number,
                41 | 42
                    | 43
                    | 44
                    | 45
                    | 46
                    | 47
                    | 48
                    | 49
                    | 50
                    | 51
                    | 52
                    | 53
                    | 54
                    | 55
                    | 288
                    | 299
                    | 307
            );
            // Descriptor theft, process signalling, symlink aliases and async
            // I/O would undermine lexical path auditing. None is needed by the
            // pure-function candidate ABI, so reject them rather than guessing.
            let process = matches!(
                number,
                56 | 57
                    | 58
                    | 62
                    | 86
                    | 88
                    | 101
                    | 200
                    | 234
                    | 265
                    | 266
                    | 270
                    | 298
                    | 303
                    | 310
                    | 311
                    | 424
                    | 425
                    | 426
                    | 427
                    | 434
                    | 435
                    | 438
            );
            let path_pointer = match number {
                2 | 4 | 6 | 21 | 59 | 76 | 80 | 82 | 83 | 84 | 85 | 87 | 89 | 90 | 92 => {
                    Some(regs.rdi)
                }
                257 | 258 | 259 | 260 | 261 | 262 | 263 | 264 | 265 | 266 | 267 | 268 | 269
                | 316 | 322 | 332 | 437 => Some(regs.rsi),
                _ => None,
            };
            let path = path_pointer.map(|p| string_at(pid, p)).unwrap_or_default();
            let normal = path.split('/').all(|p| p != ".." && p != ".");
            let allowed = normal
                && (path.starts_with("/lib/")
                    || path.starts_with("/lib64/")
                    || path.starts_with("/usr/lib/")
                    || path == "/etc/ld.so.cache"
                    || path == "/etc/ld.so.preload"
                    || path == "/proc/self/maps"
                    || path == "/proc/self/stat"
                    || path == "/proc/self/exe"
                    || path == "/candidate/main.rs"
                    || path == "/dev/null"
                    || path == "/dev/urandom");
            // The dynamic loader stats an already-open descriptor using
            // AT_EMPTY_PATH. No new path is resolved by that operation.
            let empty_descriptor_stat = path.is_empty()
                && ((number == 262 && regs.r10 & 0x1000 != 0)
                    || (number == 332 && regs.rdx & 0x1000 != 0));
            let forbidden_file = path_pointer.is_some() && !allowed && !empty_descriptor_stat;
            // The decoder is deliberately native x86_64 only. Reject x32 syscall
            // numbers as well as relying on the native-only seccomp profile.
            blocked = network || process || forbidden_file || number >= 0x4000_0000;
            if blocked {
                if events.len() >= 256 {
                    return Err("audit event cap exceeded".into());
                }
                events.push(json!({"syscall":number,"category":if network{"network"}else if process{"process"}else{"filesystem"},"target":path,"result":"denied:EPERM"}));
                // A nonexistent syscall cannot execute. Its return is replaced
                // at the exit stop, before any candidate instruction resumes.
                regs.orig_rax = u64::MAX;
                if trace(13, pid, (&mut regs as *mut Registers) as usize) < 0 {
                    return Err("blocking syscall failed".into());
                }
            }
        } else if blocked {
            regs.rax = (-1i64) as u64;
            if trace(13, pid, (&mut regs as *mut Registers) as usize) < 0 {
                return Err("return rewrite failed".into());
            }
            blocked = false;
        }
        entering = !entering;
    }
    let _ = writer.join();
    let (output, output_truncated) = reader.join().map_err(|_| "output reader failed")?;
    let (err, stderr_truncated) = errors.join().map_err(|_| "stderr reader failed")?;
    println!(
        "{}",
        json!({"schema_version":1,"trace_complete":true,"compile_ok":true,"candidate_exit":exit,"output":String::from_utf8_lossy(&output),"stderr":String::from_utf8_lossy(&err),"output_truncated":output_truncated,"stderr_truncated":stderr_truncated,"events":events,"syscalls_observed":count,"failure":null})
    );
    Ok(())
}
