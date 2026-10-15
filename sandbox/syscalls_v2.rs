//! Explicit native Linux syscall ABIs. No host-number guessing or x32 fallback.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Abi {
    X86_64,
    Aarch64,
}
impl Abi {
    pub fn from_audit_arch(value: u32) -> Option<Self> {
        match value {
            0xc000_003e => Some(Self::X86_64),
            0xc000_00b7 => Some(Self::Aarch64),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
    pub fn network(self, number: u64) -> bool {
        match self {
            Self::X86_64 => matches!(number, 41..=55 | 288 | 299 | 307),
            Self::Aarch64 => matches!(number, 198..=212 | 242 | 243 | 269),
        }
    }
    pub fn process(self, number: u64) -> bool {
        let modern = matches!(number, 424..=427 | 434 | 435 | 438);
        modern
            || match self {
                Self::X86_64 => matches!(
                    number,
                    56..=58
                        | 62
                        | 86
                        | 88
                        | 101
                        | 165
                        | 166
                        | 200
                        | 234
                        | 265
                        | 266
                        | 270
                        | 272
                        | 298
                        | 303
                        | 308
                        | 310
                        | 311
                        | 321
                ),
                Self::Aarch64 => matches!(
                    number,
                    27 | 36 | 37 | 39..=41 | 97 | 117 | 129..=131 | 220
                        | 241 | 262..=265 | 268 | 270 | 271 | 280
                ),
            }
    }
    /// Every argument resolving a pathname, including both rename operands.
    pub fn paths(self, number: u64) -> &'static [usize] {
        match self {
            Self::X86_64 => match number {
                2 | 4 | 6 | 21 | 59 | 76 | 80 | 83..=85 | 87 | 89 | 90 | 92 | 161 | 166 => &[0],
                82 | 86 | 88 | 165 => &[0, 1],
                264..=266 | 316 => &[1, 3],
                257..=263 | 267..=269 | 322 | 332 | 437 | 439 | 452 => &[1],
                _ => &[],
            },
            Self::Aarch64 => match number {
                45 | 49 | 51 | 221 => &[0],
                36 => &[0, 2],
                37 | 38 | 276 => &[1, 3],
                34 | 35 | 48 | 53 | 54 | 56 | 78 | 79 | 88 | 281 | 291 | 437 | 439 | 452 => &[1],
                263 => &[4],
                _ => &[],
            },
        }
    }
    pub fn empty_descriptor_stat(self, number: u64, args: &[u64; 6]) -> bool {
        let flags = match (self, number) {
            (Self::X86_64, 262) | (Self::Aarch64, 79) => args[3],
            (Self::X86_64, 332) | (Self::Aarch64, 291) => args[2],
            _ => return false,
        };
        flags & 0x1000 != 0
    }
    pub fn native() -> Option<Self> {
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            Some(Self::X86_64)
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            Some(Self::Aarch64)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_architecture_and_syscall_maps_do_not_mix_abis() {
        assert_eq!(Abi::from_audit_arch(0xc000_00b7), Some(Abi::Aarch64));
        assert_eq!(Abi::from_audit_arch(0x4000_0003), None);
        assert!(Abi::Aarch64.network(198));
        assert!(!Abi::X86_64.network(198));
        assert!(Abi::X86_64.network(41));
        assert!(!Abi::Aarch64.network(41));
        assert!(Abi::Aarch64.process(220));
        assert!(Abi::Aarch64.process(117));
        assert_eq!(Abi::Aarch64.paths(276), &[1, 3]);
        assert_eq!(Abi::X86_64.paths(82), &[0, 1]);
        assert_eq!(Abi::Aarch64.paths(56), &[1]);
        assert_eq!(Abi::X86_64.paths(257), &[1]);
    }
    #[test]
    fn empty_path_exemption_requires_correct_stat_flags_argument() {
        let mut args = [0; 6];
        args[3] = 0x1000;
        assert!(Abi::Aarch64.empty_descriptor_stat(79, &args));
        assert!(!Abi::Aarch64.empty_descriptor_stat(291, &args));
        args[2] = 0x1000;
        assert!(Abi::Aarch64.empty_descriptor_stat(291, &args));
        assert!(!Abi::Aarch64.empty_descriptor_stat(56, &args));
    }
}
