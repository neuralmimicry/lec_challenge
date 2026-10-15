//! CUDA Driver API discovery works without nvcc, nvidia-smi or a CUDA build.
//! Linux amd64/aarch64 share this ABI; no inference context survives discovery.
use super::GpuProfile;
use anyhow::Result;

#[cfg(not(target_os = "linux"))]
pub fn detect() -> Result<Vec<GpuProfile>> {
    anyhow::bail!("native CUDA driver discovery is currently supported on Linux")
}

#[cfg(target_os = "linux")]
pub fn detect() -> Result<Vec<GpuProfile>> {
    linux::detect()
}

/// Highest CUDA toolkit version supported by the installed driver. Querying
/// the driver directly also works on ARM64 systems without `nvidia-smi`.
pub fn driver_version() -> Result<(u32, u32)> {
    #[cfg(target_os = "linux")]
    {
        linux::driver_version()
    }
    #[cfg(not(target_os = "linux"))]
    {
        anyhow::bail!("native CUDA driver version discovery is currently supported on Linux")
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use anyhow::{Context, ensure};
    use std::{
        ffi::{CStr, c_char, c_int, c_uint, c_void},
        ptr,
    };

    type ContextHandle = *mut c_void;
    type Init = unsafe extern "C" fn(c_uint) -> c_int;
    type Version = unsafe extern "C" fn(*mut c_int) -> c_int;
    type Count = unsafe extern "C" fn(*mut c_int) -> c_int;
    type GetDevice = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type Name = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
    type TotalMemory = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
    type Attribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    type Retain = unsafe extern "C" fn(*mut ContextHandle, c_int) -> c_int;
    type Release = unsafe extern "C" fn(c_int) -> c_int;
    type GetCurrent = unsafe extern "C" fn(*mut ContextHandle) -> c_int;
    type SetCurrent = unsafe extern "C" fn(ContextHandle) -> c_int;
    type MemoryInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;

    struct Driver {
        handle: *mut c_void,
        init: Init,
        count: Count,
        get_device: GetDevice,
        name: Name,
        total: TotalMemory,
        attribute: Attribute,
        retain: Retain,
        release: Release,
        get_current: GetCurrent,
        set_current: SetCurrent,
        memory: MemoryInfo,
    }
    impl Drop for Driver {
        fn drop(&mut self) {
            // SAFETY: this handle came from dlopen and outlives every symbol.
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }

    fn status(code: c_int, operation: &str) -> Result<()> {
        ensure!(
            code == 0,
            "CUDA driver {operation} failed with status {code}"
        );
        Ok(())
    }

    fn version_pair(encoded: c_int) -> Result<(u32, u32)> {
        ensure!(
            encoded >= 1000 && encoded % 10 == 0,
            "CUDA driver returned invalid encoded version {encoded}"
        );
        Ok(((encoded / 1000) as u32, ((encoded % 1000) / 10) as u32))
    }

    pub(super) fn driver_version() -> Result<(u32, u32)> {
        let driver = Driver::open()?;
        let mut encoded = 0;
        // SAFETY: the installed symbol has CUDA's documented C signature. The
        // library and writable output remain live throughout the call. This
        // query creates no device context and does not require cuInit.
        unsafe {
            let version: Version = symbol(driver.handle, &[c"cuDriverGetVersion"])?;
            status(version(&mut encoded), "cuDriverGetVersion")?;
        }
        version_pair(encoded)
    }

    unsafe fn symbol<T: Copy>(handle: *mut c_void, names: &[&CStr]) -> Result<T> {
        ensure!(
            std::mem::size_of::<T>() == std::mem::size_of::<*mut c_void>(),
            "CUDA symbol pointer has incompatible size"
        );
        for name in names {
            // SAFETY: handle is a live dlopen result and names are NUL terminated.
            let pointer = unsafe { libc::dlsym(handle, name.as_ptr()) };
            if !pointer.is_null() {
                // SAFETY: callers specify the documented CUDA Driver C signature;
                // dlsym returns that symbol and its pointer size was checked.
                return Ok(unsafe { std::mem::transmute_copy(&pointer) });
            }
        }
        anyhow::bail!(
            "CUDA driver symbol is absent: {}",
            names[0].to_string_lossy()
        )
    }

    impl Driver {
        fn open() -> Result<Self> {
            // SAFETY: fixed installed library name and standard loader flags.
            let handle = unsafe {
                libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL)
            };
            ensure!(
                !handle.is_null(),
                "installed CUDA driver library libcuda.so.1 is unavailable"
            );
            let load = (|| -> Result<Self> {
                // SAFETY: each fixed CUDA Driver symbol is paired with its
                // published C ABI; the handle remains live in the result.
                unsafe {
                    Ok(Self {
                        handle,
                        init: symbol(handle, &[c"cuInit"])?,
                        count: symbol(handle, &[c"cuDeviceGetCount"])?,
                        get_device: symbol(handle, &[c"cuDeviceGet"])?,
                        name: symbol(handle, &[c"cuDeviceGetName"])?,
                        total: symbol(handle, &[c"cuDeviceTotalMem_v2"])?,
                        attribute: symbol(handle, &[c"cuDeviceGetAttribute"])?,
                        retain: symbol(handle, &[c"cuDevicePrimaryCtxRetain"])?,
                        release: symbol(
                            handle,
                            &[
                                c"cuDevicePrimaryCtxRelease_v2",
                                c"cuDevicePrimaryCtxRelease",
                            ],
                        )?,
                        get_current: symbol(handle, &[c"cuCtxGetCurrent"])?,
                        set_current: symbol(handle, &[c"cuCtxSetCurrent"])?,
                        memory: symbol(handle, &[c"cuMemGetInfo_v2"])?,
                    })
                }
            })();
            if load.is_err() {
                // SAFETY: failed construction transferred ownership to no Driver.
                unsafe {
                    libc::dlclose(handle);
                }
            }
            load
        }
    }

    struct PrimaryContext<'a> {
        driver: &'a Driver,
        device: c_int,
        previous: ContextHandle,
        active: bool,
    }
    impl<'a> PrimaryContext<'a> {
        fn acquire(driver: &'a Driver, device: c_int) -> Result<Self> {
            let mut previous = ptr::null_mut();
            let mut context = ptr::null_mut();
            // SAFETY: local output pointers remain valid for these synchronous
            // calls; device was returned by cuDeviceGet.
            unsafe {
                status((driver.get_current)(&mut previous), "cuCtxGetCurrent")?;
                status(
                    (driver.retain)(&mut context, device),
                    "cuDevicePrimaryCtxRetain",
                )?;
                if let Err(error) = status((driver.set_current)(context), "cuCtxSetCurrent") {
                    let _ = (driver.set_current)(previous);
                    let _ = (driver.release)(device);
                    return Err(error);
                }
            }
            Ok(Self {
                driver,
                device,
                previous,
                active: true,
            })
        }

        fn close(&mut self) -> Result<()> {
            // SAFETY: these context/device handles belong to this active retain.
            let (restored, released) = unsafe {
                (
                    (self.driver.set_current)(self.previous),
                    (self.driver.release)(self.device),
                )
            };
            if released == 0 {
                self.active = false;
            }
            status(restored, "restore calling thread context")?;
            status(released, "release primary discovery context")
        }
    }
    impl Drop for PrimaryContext<'_> {
        fn drop(&mut self) {
            if !self.active {
                return;
            }
            // SAFETY: restore the calling thread's original context, then release
            // precisely this probe's retain, including every error return path.
            unsafe {
                let _ = (self.driver.set_current)(self.previous);
                let _ = (self.driver.release)(self.device);
            }
        }
    }

    fn probe(driver: &Driver, ordinal: usize) -> Result<GpuProfile> {
        let ordinal_c = c_int::try_from(ordinal)?;
        let mut device = 0;
        let mut name = [0 as c_char; 256];
        let mut total = 0_usize;
        let mut major = 0;
        let mut minor = 0;
        // SAFETY: every output buffer has the declared size and stays live;
        // attribute IDs 75/76 are CUDA's compute capability major/minor.
        unsafe {
            status((driver.get_device)(&mut device, ordinal_c), "cuDeviceGet")?;
            status(
                (driver.name)(name.as_mut_ptr(), name.len() as c_int, device),
                "cuDeviceGetName",
            )?;
            status((driver.total)(&mut total, device), "cuDeviceTotalMem_v2")?;
            status(
                (driver.attribute)(&mut major, 75, device),
                "compute capability major",
            )?;
            status(
                (driver.attribute)(&mut minor, 76, device),
                "compute capability minor",
            )?;
        }
        name[255] = 0;
        // SAFETY: the final element was NUL terminated and the buffer is live.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut context = PrimaryContext::acquire(driver, device)?;
        let mut free = 0_usize;
        let mut contextual_total = 0_usize;
        // SAFETY: output pointers remain valid and the retained context is current.
        unsafe {
            status(
                (driver.memory)(&mut free, &mut contextual_total),
                "cuMemGetInfo_v2",
            )?;
        }
        ensure!(
            total > 0 && free <= total && free <= contextual_total,
            "CUDA driver returned inconsistent memory capacities"
        );
        ensure!(
            major > 0 && minor >= 0,
            "CUDA driver returned invalid compute capability"
        );
        context.close()?;
        Ok(GpuProfile {
            backend: "cuda".into(),
            ordinal,
            name,
            total_memory_bytes: u64::try_from(total)?,
            free_memory_bytes: u64::try_from(free)?,
            compute_capability: Some(format!("{major}.{minor}")),
        })
    }

    pub(super) fn detect() -> Result<Vec<GpuProfile>> {
        let driver = Driver::open()?;
        // SAFETY: cuInit takes a reserved zero flags argument, no pointer access.
        let initialized = unsafe { (driver.init)(0) };
        if initialized == 100 {
            return Ok(Vec::new());
        } // CUDA_ERROR_NO_DEVICE.
        status(initialized, "cuInit")?;
        let mut count = 0;
        // SAFETY: the local count pointer is writable for the synchronous call.
        unsafe {
            status((driver.count)(&mut count), "cuDeviceGetCount")?;
        }
        ensure!(
            (0..=64).contains(&count),
            "CUDA driver returned an invalid device count"
        );
        (0..count as usize)
            .map(|ordinal| {
                probe(&driver, ordinal).with_context(|| format!("inspect CUDA device {ordinal}"))
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn driver_status_errors_are_explicit() {
            status(0, "fixture").unwrap();
            let error = status(201, "cuMemGetInfo_v2").unwrap_err().to_string();
            assert!(error.contains("201") && error.contains("cuMemGetInfo_v2"));
        }

        #[test]
        fn driver_version_preserves_toolkit_minor_and_rejects_invalid_results() {
            assert_eq!(version_pair(12090).unwrap(), (12, 9));
            assert_eq!(version_pair(13030).unwrap(), (13, 3));
            assert_eq!(version_pair(11000).unwrap(), (11, 0));
            for invalid in [0, -1, 999, 12091] {
                assert!(version_pair(invalid).is_err());
            }
        }

        #[test]
        #[ignore = "actual installed CUDA driver QA; runs without the Cargo CUDA feature"]
        fn actual_driver_probe_restores_thread_and_primary_context_state() {
            type PrimaryState = unsafe extern "C" fn(c_int, *mut c_uint, *mut c_int) -> c_int;
            let driver = Driver::open().unwrap();
            let mut before = ptr::null_mut();
            let mut after = ptr::null_mut();
            let mut device = 0;
            let mut flags = 0;
            let mut active_before = 0;
            let mut active_after = 0;
            // SAFETY: the documented fixed symbol signature and local output
            // pointers remain valid for every synchronous Driver API call.
            unsafe {
                status((driver.init)(0), "cuInit").unwrap();
                status((driver.get_current)(&mut before), "cuCtxGetCurrent before").unwrap();
                status(
                    (driver.get_device)(&mut device, 0),
                    "cuDeviceGet fixture device",
                )
                .unwrap();
                let state: PrimaryState =
                    symbol(driver.handle, &[c"cuDevicePrimaryCtxGetState"]).unwrap();
                status(
                    state(device, &mut flags, &mut active_before),
                    "primary state before",
                )
                .unwrap();
                let profiles = detect().unwrap();
                assert!(!profiles.is_empty());
                assert!(
                    profiles
                        .iter()
                        .all(|gpu| gpu.free_memory_bytes <= gpu.total_memory_bytes
                            && gpu.total_memory_bytes > 0
                            && gpu.compute_capability.is_some())
                );
                status((driver.get_current)(&mut after), "cuCtxGetCurrent after").unwrap();
                status(
                    state(device, &mut flags, &mut active_after),
                    "primary state after",
                )
                .unwrap();
                assert_eq!(
                    before, after,
                    "resource discovery changed the caller's CUDA context"
                );
                assert_eq!(
                    active_before, active_after,
                    "resource discovery retained a primary CUDA context"
                );
                let report = serde_json::json!({"evidence_kind":"actual_CUDA_driver_probe_without_management_utility","cuda_feature_compiled":cfg!(feature="cuda"),"architecture":std::env::consts::ARCH,"uses_nvidia_smi":false,"thread_context_restored":true,"primary_context_active_before":active_before,"primary_context_active_after":active_after,"gpus":profiles});
                let output = std::path::PathBuf::from(format!(
                    "reports/comparative/search/cuda-driver-probe/{}.json",
                    uuid::Uuid::new_v4()
                ));
                crate::artefact::write_json(&output, &report).unwrap();
                println!(
                    "{}\nreport={}",
                    serde_json::to_string_pretty(&report).unwrap(),
                    output.display()
                );
            }
        }
    }
}
