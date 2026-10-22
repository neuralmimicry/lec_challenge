//! Portable resource discovery and explicit, memory-aware execution decisions.
//!
//! Automatic selection may choose CPU for a stated reason. An explicit GPU
//! request remains an error when its backend or memory budget is unavailable.
use crate::model::{ExecutionDevice, InferenceDtype, RuntimeOptions};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, process::Command};

const MIB: u64 = 1024 * 1024;

pub mod cuda_build;
pub mod cuda_probe;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuProfile {
    pub backend: String,
    pub ordinal: usize,
    pub name: String,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    pub compute_capability: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub architecture: String,
    pub operating_system: String,
    pub available_cpu_threads: usize,
    pub host_available_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub usable_ram_bytes: u64,
    pub memory_constraint: String,
    pub gpus: Vec<GpuProfile>,
    pub cuda_backend_compiled: bool,
    pub gpu_probe_note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeSelection {
    pub options: RuntimeOptions,
    pub reason: String,
}

fn meminfo_bytes(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key)
            .then(|| {
                value
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
            .flatten()
    })
}

fn guarded_capacity(bytes: u64, fixed_reserve: u64) -> u64 {
    // Retain ten per cent plus a small allocator/driver reserve, using checked
    // integer arithmetic rather than rounding an oversized allocation upward.
    bytes
        .saturating_sub(bytes / 10)
        .saturating_sub(fixed_reserve)
}

#[cfg(target_os = "linux")]
fn ram_available() -> Result<(u64, u64, String)> {
    let text = fs::read_to_string("/proc/meminfo").context("read available Linux RAM")?;
    let host = meminfo_bytes(&text, "MemAvailable").context("Linux MemAvailable missing")?;
    let mut available = host;
    let mut constraint = String::from("Linux MemAvailable");
    let groups = fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    if let Some(group) = groups.lines().find_map(|line| line.strip_prefix("0::")) {
        let relative = group.trim_start_matches('/');
        ensure!(
            Path::new(relative)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unexpected cgroup path"
        );
        let root = Path::new("/sys/fs/cgroup");
        let mut directory = root.join(relative);
        loop {
            if let (Ok(limit), Ok(current)) = (
                fs::read_to_string(directory.join("memory.max")),
                fs::read_to_string(directory.join("memory.current")),
            ) && let (Ok(limit), Ok(current)) =
                (limit.trim().parse::<u64>(), current.trim().parse::<u64>())
            {
                let remaining = limit.saturating_sub(current);
                if remaining < available {
                    available = remaining;
                    constraint = format!("remaining cgroup v2 memory at {}", directory.display());
                }
            }
            if directory == root || !directory.pop() || !directory.starts_with(root) {
                break;
            }
        }
    }
    Ok((host, available, constraint))
}

#[cfg(target_os = "macos")]
fn ram_available() -> Result<(u64, u64, String)> {
    let total = Command::new("sysctl").args(["-n", "hw.memsize"]).output()?;
    ensure!(total.status.success(), "cannot query macOS physical RAM");
    let total = String::from_utf8(total.stdout)?.trim().parse::<u64>()?;
    let pages = Command::new("vm_stat").output()?;
    ensure!(
        pages.status.success(),
        "cannot query macOS available memory"
    );
    let pages = String::from_utf8(pages.stdout)?;
    let page_size = pages
        .lines()
        .next()
        .and_then(|s| s.split("page size of ").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u64>().ok())
        .context("macOS memory page size missing")?;
    let mut reclaimable = 0_u64;
    for key in ["Pages free", "Pages inactive", "Pages speculative"] {
        if let Some(value) = pages.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            (name == key)
                .then(|| value.trim().trim_end_matches('.').parse::<u64>().ok())
                .flatten()
        }) {
            reclaimable = reclaimable.saturating_add(value.saturating_mul(page_size));
        }
    }
    Ok((
        total,
        reclaimable.min(total),
        "macOS free/inactive/speculative pages; unified GPU memory is shared".into(),
    ))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ram_available() -> Result<(u64, u64, String)> {
    bail!("automatic available-memory discovery is currently supported on Linux and macOS")
}

fn parse_nvidia(text: &str) -> Result<Vec<GpuProfile>> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            ensure!(fields.len() == 4, "unexpected NVIDIA memory query shape");
            let total = fields[2]
                .parse::<u64>()?
                .checked_mul(MIB)
                .context("VRAM size overflow")?;
            let free = fields[3]
                .parse::<u64>()?
                .checked_mul(MIB)
                .context("free VRAM overflow")?;
            ensure!(free <= total, "reported free VRAM exceeds total");
            Ok(GpuProfile {
                backend: "cuda".into(),
                ordinal: fields[0].parse()?,
                name: fields[1].into(),
                total_memory_bytes: total,
                free_memory_bytes: free,
                compute_capability: None,
            })
        })
        .collect()
}

pub fn detect() -> Result<HardwareProfile> {
    let cpu = std::thread::available_parallelism()?.get();
    let (host, ram, memory_constraint) = ram_available()?;
    let (mut gpus, mut gpu_probe_note) = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) if output.status.success() => (
            parse_nvidia(&String::from_utf8(output.stdout)?)?,
            "NVIDIA driver reported memory availability".into(),
        ),
        Ok(_) => (
            Vec::new(),
            "NVIDIA query unavailable; CPU remains eligible".into(),
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (
            Vec::new(),
            "No NVIDIA management utility; CPU remains eligible".into(),
        ),
        Err(error) => (Vec::new(), format!("NVIDIA probe failed: {error}")),
    };
    if gpus.is_empty() {
        match cuda_probe::detect() {
            Ok(devices) if !devices.is_empty() => {
                gpus = devices;
                gpu_probe_note =
                    "Native CUDA driver API discovered GPU memory; no nvidia-smi dependency".into();
            }
            Ok(_) => {}
            Err(error) => gpu_probe_note.push_str(&format!("; native CUDA probe: {error:#}")),
        }
    }
    if !gpus.is_empty()
        && let Ok(output) = Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,compute_cap",
                "--format=csv,noheader,nounits",
            ])
            .output()
        && output.status.success()
    {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some((index, capability)) = line.split_once(',')
                && let Ok(index) = index.trim().parse::<usize>()
                && let Some(gpu) = gpus.iter_mut().find(|gpu| gpu.ordinal == index)
            {
                gpu.compute_capability = Some(capability.trim().into());
            }
        }
    }
    Ok(HardwareProfile {
        architecture: std::env::consts::ARCH.into(),
        operating_system: std::env::consts::OS.into(),
        available_cpu_threads: cpu,
        host_available_ram_bytes: host,
        available_ram_bytes: ram,
        usable_ram_bytes: guarded_capacity(ram, 256 * MIB),
        memory_constraint,
        gpus,
        cuda_backend_compiled: cfg!(feature = "cuda"),
        gpu_probe_note,
    })
}

pub fn select_runtime(
    profile: &HardwareProfile,
    requested_device: &str,
    gpu_ordinal: usize,
    quantised: bool,
    estimated_gpu_bytes: u64,
    estimated_cpu_bytes: u64,
) -> Result<RuntimeSelection> {
    ensure!(
        matches!(requested_device, "auto" | "cpu" | "cuda"),
        "unknown execution device"
    );
    if requested_device != "cpu" {
        let gpu = profile
            .gpus
            .iter()
            .find(|gpu| gpu.backend == "cuda" && gpu.ordinal == gpu_ordinal);
        if profile.cuda_backend_compiled
            && let Some(gpu) = gpu
            && estimated_gpu_bytes <= guarded_capacity(gpu.free_memory_bytes, 128 * MIB)
        {
            return Ok(RuntimeSelection {
                options: RuntimeOptions {
                    device: ExecutionDevice::Cuda {
                        ordinal: gpu.ordinal,
                    },
                    dtype: if quantised {
                        InferenceDtype::F32
                    } else {
                        InferenceDtype::F16
                    },
                },
                reason: format!(
                    "CUDA GPU {} selected: estimated resident {} bytes fits guarded free VRAM {} bytes",
                    gpu.name,
                    estimated_gpu_bytes,
                    guarded_capacity(gpu.free_memory_bytes, 128 * MIB)
                ),
            });
        }
        if requested_device == "cuda" {
            bail!(
                "explicit CUDA request lacks a compiled backend, available GPU, or sufficient free VRAM; CPU fallback is disabled"
            );
        }
    }
    ensure!(
        estimated_cpu_bytes <= profile.usable_ram_bytes,
        "estimated CPU residency {} bytes exceeds guarded available RAM {} bytes",
        estimated_cpu_bytes,
        profile.usable_ram_bytes
    );
    Ok(RuntimeSelection {
        options: RuntimeOptions::default(),
        reason: if requested_device == "cpu" {
            format!(
                "Explicit CPU selected with {} available threads and {} guarded RAM bytes",
                profile.available_cpu_threads, profile.usable_ram_bytes
            )
        } else {
            format!(
                "Automatic CPU selected: no compiled, available CUDA device has sufficient free VRAM; {} threads and {} guarded RAM bytes available",
                profile.available_cpu_threads, profile.usable_ram_bytes
            )
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> HardwareProfile {
        HardwareProfile {
            architecture: "aarch64".into(),
            operating_system: "linux".into(),
            available_cpu_threads: 24,
            host_available_ram_bytes: 32 << 30,
            available_ram_bytes: 8 << 30,
            usable_ram_bytes: 7 << 30,
            memory_constraint: "labelled constrained fixture".into(),
            gpus: vec![GpuProfile {
                backend: "cuda".into(),
                ordinal: 0,
                name: "labelled fixture".into(),
                total_memory_bytes: 8 << 30,
                free_memory_bytes: 6 << 30,
                compute_capability: Some("8.7".into()),
            }],
            cuda_backend_compiled: true,
            gpu_probe_note: "fixture".into(),
        }
    }
    #[test]
    fn constrained_memory_and_vram_block_oversized_models() {
        let host = profile();
        assert!(matches!(
            select_runtime(&host, "auto", 0, true, 7 << 30, 5 << 30)
                .unwrap()
                .options
                .device,
            ExecutionDevice::Cpu
        ));
        assert!(select_runtime(&host, "cuda", 0, true, 7 << 30, 5 << 30).is_err());
        assert!(select_runtime(&host, "auto", 0, true, 7 << 30, 9 << 30).is_err());
        let selection = select_runtime(&host, "auto", 0, true, 4 << 30, 5 << 30).unwrap();
        assert!(matches!(
            selection.options.device,
            ExecutionDevice::Cuda { ordinal: 0 }
        ));
        assert_eq!(selection.options.dtype, InferenceDtype::F32);
    }
    #[test]
    fn missing_compiled_backend_is_an_explicit_automatic_cpu_decision() {
        let mut host = profile();
        host.cuda_backend_compiled = false;
        assert!(matches!(
            select_runtime(&host, "auto", 0, false, 1 << 30, 2 << 30)
                .unwrap()
                .options
                .device,
            ExecutionDevice::Cpu
        ));
        assert!(select_runtime(&host, "cuda", 0, false, 1 << 30, 2 << 30).is_err());
    }
    #[test]
    fn memory_units_and_multiple_gpu_ordinals_are_checked() {
        assert_eq!(
            meminfo_bytes("MemTotal: 999 kB\nMemAvailable: 123 kB\n", "MemAvailable"),
            Some(123 * 1024)
        );
        let gpus = parse_nvidia("0, First, 8192, 4096\n1, Second, 16384, 12000\n").unwrap();
        assert_eq!(gpus[1].ordinal, 1);
        assert_eq!(gpus[0].free_memory_bytes, 4096 * MIB);
        assert!(parse_nvidia("0, Broken, 1, 2").is_err());
        assert_eq!(guarded_capacity(32, 256), 0);
    }
}
