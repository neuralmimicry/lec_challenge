//! Reviewed Linux ptrace layouts; candidates can never select another ABI.

use super::syscalls_v2::Abi;
use std::{ffi::c_void, io, mem::size_of};

unsafe extern "C" {
    fn ptrace(request: i32, pid: i32, address: *mut c_void, data: *mut c_void) -> i64;
}

#[repr(C)]
#[derive(Default)]
struct SyscallInfo {
    operation: u8,
    padding: [u8; 3],
    architecture: u32,
    instruction_pointer: u64,
    stack_pointer: u64,
    payload: [u64; 8],
}

pub enum Stop {
    Entry {
        abi: Abi,
        number: u64,
        arguments: [u64; 6],
    },
    Exit,
}

pub fn syscall_stop(pid: i32) -> io::Result<Stop> {
    let mut info = SyscallInfo::default();
    // SAFETY: GET_SYSCALL_INFO writes at most the supplied buffer length. The
    // all-integer repr(C) object matches Linux uapi linux/ptrace.h, including
    // the largest union member. This request requires Linux 5.3 or newer.
    let read = unsafe {
        ptrace(
            0x420e,
            pid,
            size_of::<SyscallInfo>() as *mut c_void,
            (&mut info as *mut SyscallInfo).cast(),
        )
    };
    if read < 0 {
        return Err(io::Error::last_os_error());
    }
    let abi = Abi::from_audit_arch(info.architecture)
        .filter(|abi| Some(*abi) == Abi::native())
        .ok_or_else(|| io::Error::other("non-native candidate syscall ABI rejected"))?;
    match info.operation {
        1 if read >= 80 => {
            let mut arguments = [0; 6];
            arguments.copy_from_slice(&info.payload[1..7]);
            Ok(Stop::Entry {
                abi,
                number: info.payload[0],
                arguments,
            })
        }
        2 if read >= 33 => Ok(Stop::Exit),
        _ => Err(io::Error::other(
            "missing or truncated kernel syscall-stop information",
        )),
    }
}

#[cfg(target_arch = "x86_64")]
#[repr(C)]
#[derive(Default)]
struct Registers {
    words: [u64; 27],
}

#[cfg(target_arch = "aarch64")]
#[repr(C)]
#[derive(Default)]
struct Registers {
    x: [u64; 31],
    sp: u64,
    pc: u64,
    pstate: u64,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn registers(pid: i32, write: bool, regs: &mut Registers) -> io::Result<()> {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: GETREGS/SETREGS accesses a complete native user_regs_struct.
    let result = unsafe {
        ptrace(
            if write { 13 } else { 12 },
            pid,
            std::ptr::null_mut(),
            (regs as *mut Registers).cast(),
        )
    };
    #[cfg(target_arch = "aarch64")]
    let result = {
        #[repr(C)]
        struct IoVec {
            base: *mut c_void,
            length: usize,
        }
        let mut vector = IoVec {
            base: (regs as *mut Registers).cast(),
            length: size_of::<Registers>(),
        };
        // SAFETY: NT_PRSTATUS uses the complete native user_pt_regs layout;
        // iovec bounds the kernel access and remains alive throughout ptrace.
        let result = unsafe {
            ptrace(
                if write { 0x4205 } else { 0x4204 },
                pid,
                std::ptr::without_provenance_mut::<c_void>(1),
                (&mut vector as *mut IoVec).cast(),
            )
        };
        if result >= 0 && vector.length != size_of::<Registers>() {
            return Err(io::Error::other(
                "kernel returned a truncated native register set",
            ));
        }
        result
    };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

pub fn deny_entry(pid: i32) -> io::Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut regs = Registers::default();
        registers(pid, false, &mut regs)?;
        regs.words[15] = u64::MAX; // orig_rax: skip the system call entirely.
        registers(pid, true, &mut regs)
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[repr(C)]
        struct IoVec {
            base: *mut c_void,
            length: usize,
        }
        let mut number = -1i32;
        let mut vector = IoVec {
            base: (&mut number as *mut i32).cast(),
            length: size_of::<i32>(),
        };
        // SAFETY: NT_ARM_SYSTEM_CALL is a single signed native integer. Setting
        // -1 suppresses dispatch; changing x8 alone would not change syscallno.
        let result = unsafe {
            ptrace(
                0x4205,
                pid,
                std::ptr::without_provenance_mut::<c_void>(0x404),
                (&mut vector as *mut IoVec).cast(),
            )
        };
        if result < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = pid;
        Err(io::Error::other("unsupported native register ABI"))
    }
}

pub fn deny_exit(pid: i32) -> io::Result<()> {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let mut regs = Registers::default();
        registers(pid, false, &mut regs)?;
        #[cfg(target_arch = "x86_64")]
        {
            regs.words[10] = (-1i64) as u64;
        } // rax = -EPERM.
        #[cfg(target_arch = "aarch64")]
        {
            regs.x[0] = (-1i64) as u64;
        } // x0 = -EPERM.
        registers(pid, true, &mut regs)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = pid;
        Err(io::Error::other("unsupported native register ABI"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn syscall_information_and_native_register_layouts_match_linux_uapi() {
        assert_eq!(size_of::<SyscallInfo>(), 88);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(size_of::<Registers>(), 216);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(size_of::<Registers>(), 272);
    }
}
