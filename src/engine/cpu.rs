//! Explicit CPU build profiles; replay needs host features, never a compiler.
use super::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuProfile {
    Portable,
    Avx2FmaF16c,
    Neon,
}
impl CpuProfile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Avx2FmaF16c => "avx2-fma-f16c",
            Self::Neon => "neon",
        }
    }
    fn flags(self) -> Vec<String> {
        let mut flags = vec![
            "-C".into(),
            "target-cpu=generic".into(),
            "-C".into(),
            "debug-assertions=no".into(),
            "-C".into(),
            "panic=unwind".into(),
        ];
        match self {
            Self::Portable => {}
            Self::Avx2FmaF16c => {
                flags.extend(["-C".into(), "target-feature=+avx2,+fma,+f16c".into()])
            }
            Self::Neon => flags.extend(["-C".into(), "target-feature=+neon".into()]),
        }
        flags
    }
    fn required(self) -> Vec<String> {
        match self {
            Self::Portable => vec![],
            Self::Avx2FmaF16c => vec!["avx", "avx2", "f16c", "fma"],
            Self::Neon => vec!["neon"],
        }
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}

/// Runtime detection includes the OS state needed to execute AVX safely.
pub fn host_features() -> BTreeSet<String> {
    let mut features = BTreeSet::new();
    #[cfg(target_arch = "x86_64")]
    for (name, available) in [
        ("avx", std::is_x86_feature_detected!("avx")),
        ("avx2", std::is_x86_feature_detected!("avx2")),
        ("f16c", std::is_x86_feature_detected!("f16c")),
        ("fma", std::is_x86_feature_detected!("fma")),
    ] {
        if available {
            features.insert(name.into());
        }
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        features.insert("neon".into());
    }
    features
}

pub fn select_profile(
    preference: &str,
    architecture: &str,
    features: &BTreeSet<String>,
) -> Result<CpuProfile> {
    ensure!(
        matches!(preference, "auto" | "portable"),
        "LEC_CPU_PROFILE must be auto or portable"
    );
    ensure!(
        matches!(architecture, "x86_64" | "aarch64"),
        "CPU launcher supports Linux amd64 and arm64"
    );
    Ok(if preference == "portable" {
        CpuProfile::Portable
    } else if architecture == "x86_64"
        && CpuProfile::Avx2FmaF16c
            .required()
            .iter()
            .all(|f| features.contains(f))
    {
        CpuProfile::Avx2FmaF16c
    } else if architecture == "aarch64" && features.contains("neon") {
        CpuProfile::Neon
    } else {
        CpuProfile::Portable
    })
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuBuildConfig {
    pub schema_version: u32,
    pub protocol: String,
    pub architecture: String,
    pub profile: String,
    pub target_triple: String,
    pub required_features: Vec<String>,
    pub rustflags: Vec<String>,
    pub encoded_rustflags: String,
    pub rustc_version_verbose: String,
    pub rustc_cfg: Vec<String>,
    pub cargo_lock_sha256: String,
    /// Absent in the original CPU-only receipt, preserving its JSON identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cargo_features: Vec<String>,
    /// New builds bind the real compiler rather than a mutable rustup proxy.
    /// Historical executable replay remains independent of compiler availability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_rustc: Option<ResolvedRustCompiler>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResolvedRustCompiler {
    pub executable: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuBuildReceipt {
    pub build: CpuBuildConfig,
    pub executable_sha256: String,
    pub executable_bytes: u64,
}

fn output(command: &mut std::process::Command) -> Result<String> {
    let result = command.output().context("inspect CPU build compiler")?;
    ensure!(
        result.status.success(),
        "CPU compiler inspection failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(String::from_utf8(result.stdout)?)
}

fn unambiguous_environment(values: impl Iterator<Item = (String, String)>) -> Result<()> {
    for (key, value) in values {
        if value.trim().is_empty() {
            continue;
        }
        ensure!(
            !matches!(
                key.as_str(),
                "RUSTFLAGS"
                    | "CARGO_ENCODED_RUSTFLAGS"
                    | "CARGO_BUILD_TARGET"
                    | "RUSTC"
                    | "RUSTC_WRAPPER"
                    | "RUSTC_WORKSPACE_WRAPPER"
                    | "CARGO_BUILD_RUSTC"
                    | "CARGO_BUILD_RUSTC_WRAPPER"
                    | "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER"
                    | "CARGO_BUILD_RUSTFLAGS"
            ) && !(key.starts_with("CARGO_TARGET_") && key.ends_with("_RUSTFLAGS")),
            "ambiguous inherited CPU build setting {key}; clear it before starting a new launcher protocol"
        );
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl ResolvedRustCompiler {
    fn validate_identity(&self) -> Result<()> {
        ensure!(
            self.executable.is_absolute()
                && !self
                    .executable
                    .components()
                    .any(|part| matches!(part, std::path::Component::ParentDir))
                && valid_sha256(&self.sha256)
                && self.bytes > 0,
            "frozen real Rust compiler identity is incomplete"
        );
        Ok(())
    }

    fn inspect() -> Result<Self> {
        // rustup's PATH entry is a dispatcher. Resolve its chosen sysroot once,
        // then inspect and build with that actual toolchain executable.
        let sysroot = output(std::process::Command::new("rustc").args(["--print", "sysroot"]))?;
        let sysroot = Path::new(sysroot.trim());
        ensure!(sysroot.is_absolute(), "rustc returned a relative sysroot");
        let executable = sysroot.join("bin/rustc").canonicalize()?;
        let metadata = fs::metadata(&executable)?;
        ensure!(metadata.is_file(), "resolved rustc is not a regular file");
        let resolved = Self {
            sha256: artefact::file_hash(&executable)?,
            bytes: metadata.len(),
            executable,
        };
        resolved.validate_identity()?;
        Ok(resolved)
    }

    fn verify_for_build(&self, expected_version: &str) -> Result<()> {
        self.validate_identity()?;
        ensure!(
            self.executable.canonicalize()? == self.executable
                && fs::metadata(&self.executable)?.len() == self.bytes
                && artefact::file_hash(&self.executable)? == self.sha256,
            "resolved Rust compiler changed after inspection; start a new build protocol"
        );
        // The driver may be a separate shared library in the selected sysroot;
        // recheck its reported identity as well as the executable bytes.
        ensure!(
            output(std::process::Command::new(&self.executable).arg("-vV"))? == expected_version,
            "resolved Rust compiler version changed after inspection"
        );
        Ok(())
    }
}

fn compiler_environment(
    compiler: &ResolvedRustCompiler,
    encoded_rustflags: &str,
    target: &str,
) -> BTreeMap<String, OsString> {
    [
        ("RUSTC", compiler.executable.as_os_str().to_owned()),
        (
            "CARGO_BUILD_RUSTC",
            compiler.executable.as_os_str().to_owned(),
        ),
        ("RUSTC_WRAPPER", OsString::new()),
        ("RUSTC_WORKSPACE_WRAPPER", OsString::new()),
        ("CARGO_BUILD_RUSTC_WRAPPER", OsString::new()),
        ("CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", OsString::new()),
        ("RUSTFLAGS", OsString::new()),
        ("CARGO_BUILD_RUSTFLAGS", OsString::new()),
        ("CARGO_ENCODED_RUSTFLAGS", encoded_rustflags.into()),
        ("CARGO_BUILD_TARGET", target.into()),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value))
    .collect()
}

impl CpuBuildConfig {
    pub fn inspect(profile: CpuProfile) -> Result<Self> {
        unambiguous_environment(std::env::vars())?;
        let compiler = ResolvedRustCompiler::inspect()?;
        let version = output(std::process::Command::new(&compiler.executable).arg("-vV"))?;
        let target = version
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .context("rustc host triple absent")?
            .to_owned();
        let flags = profile.flags();
        let cfg = output(
            std::process::Command::new(&compiler.executable)
                .args(["--print", "cfg", "--target", &target])
                .args(&flags),
        )?;
        let mut rustc_cfg = cfg.lines().map(str::to_owned).collect::<Vec<_>>();
        rustc_cfg.sort();
        let config = Self {
            schema_version: 1,
            protocol: "explicit_cpu_simd_build_v1".into(),
            architecture: std::env::consts::ARCH.into(),
            profile: profile.name().into(),
            target_triple: target,
            required_features: profile.required(),
            encoded_rustflags: flags.join("\u{1f}"),
            rustflags: flags,
            rustc_version_verbose: version,
            rustc_cfg,
            cargo_lock_sha256: artefact::file_hash(Path::new("Cargo.lock"))?,
            cargo_features: Vec::new(),
            resolved_rustc: Some(compiler),
        };
        config.verify_host()?;
        Ok(config)
    }

    /// Bind Cargo to the exact compiler inspected for this new build. Empty
    /// wrapper variables override both Cargo configuration files and aliases.
    /// Verifying/replaying an already pinned executable never calls this method.
    pub fn build_environment(&self) -> Result<BTreeMap<String, OsString>> {
        self.verify_host()?;
        let compiler = self.resolved_rustc.as_ref().context(
            "historical CPU receipt has no resolved compiler; replay its pinned executable or start a new build protocol",
        )?;
        compiler.verify_for_build(&self.rustc_version_verbose)?;
        Ok(compiler_environment(
            compiler,
            &self.encoded_rustflags,
            &self.target_triple,
        ))
    }

    pub fn target_directory(&self) -> Result<PathBuf> {
        let digest = artefact::hash(&serde_json::to_vec(self)?);
        Ok(Path::new(if self.cargo_features.is_empty() {
            "target/cpu"
        } else {
            "target/cuda-simd"
        })
        .join(&self.architecture)
        .join(&self.profile)
        .join(&digest[..16]))
    }

    pub fn verify_frozen(&self) -> Result<()> {
        if let Some(compiler) = &self.resolved_rustc {
            compiler.validate_identity()?;
        }
        let profile = match self.profile.as_str() {
            "portable" => CpuProfile::Portable,
            "avx2-fma-f16c" => CpuProfile::Avx2FmaF16c,
            "neon" => CpuProfile::Neon,
            _ => anyhow::bail!("unknown frozen CPU profile"),
        };
        ensure!(
            self.schema_version == 1
                && self.protocol == "explicit_cpu_simd_build_v1"
                && self.rustflags == profile.flags()
                && self.encoded_rustflags == self.rustflags.join("\u{1f}")
                && self.required_features == profile.required()
                && (self.cargo_features.is_empty() || self.cargo_features == ["cuda"])
                && valid_sha256(&self.cargo_lock_sha256)
                && self
                    .rustc_version_verbose
                    .lines()
                    .any(|line| line == format!("host: {}", self.target_triple))
                && self
                    .target_triple
                    .starts_with(&format!("{}-", self.architecture))
                && self
                    .rustc_cfg
                    .contains(&format!("target_arch=\"{}\"", self.architecture))
                && self.rustc_cfg.contains(&"target_os=\"linux\"".into())
                && self.rustc_cfg.windows(2).all(|pair| pair[0] < pair[1]),
            "frozen CPU compiler, target, features or exact flags are inconsistent"
        );
        let enabled = self
            .rustc_cfg
            .iter()
            .filter_map(|line| {
                line.strip_prefix("target_feature=\"")
                    .and_then(|s| s.strip_suffix('"'))
            })
            .collect::<BTreeSet<_>>();
        let allowed = match (self.architecture.as_str(), profile) {
            ("x86_64", CpuProfile::Portable) => vec!["fxsr", "sse", "sse2"],
            ("x86_64", CpuProfile::Avx2FmaF16c) => vec![
                "avx", "avx2", "f16c", "fma", "fxsr", "sse", "sse2", "sse3", "sse4.1", "sse4.2",
                "ssse3",
            ],
            ("aarch64", CpuProfile::Portable | CpuProfile::Neon) => vec!["neon"],
            _ => anyhow::bail!("CPU profile does not belong to frozen architecture"),
        }
        .into_iter()
        .collect::<BTreeSet<_>>();
        ensure!(
            enabled.is_subset(&allowed)
                && self
                    .required_features
                    .iter()
                    .all(|f| enabled.contains(f.as_str())),
            "rustc cfg enabled undeclared or omitted required CPU features"
        );
        Ok(())
    }

    pub fn verify_host_requirements(
        &self,
        architecture: &str,
        features: &BTreeSet<String>,
    ) -> Result<()> {
        self.verify_frozen()?;
        ensure!(
            self.architecture == architecture
                && self.required_features.iter().all(|f| features.contains(f)),
            "host cannot safely execute frozen CPU engine {}; restore it on a compatible host or start an explicit new portable protocol",
            self.profile
        );
        Ok(())
    }

    pub fn verify_host(&self) -> Result<()> {
        self.verify_host_requirements(std::env::consts::ARCH, &host_features())
    }
}

impl CpuBuildReceipt {
    pub(super) fn verify(&self, engine: &PinnedEngine) -> Result<()> {
        ensure!(
            self.executable_sha256 == engine.sha256 && self.executable_bytes == engine.bytes,
            "CPU build receipt belongs to another executable"
        );
        self.build.verify_host()
    }
}

#[cfg(test)]
pub(super) fn fixture_config(profile: CpuProfile) -> CpuBuildConfig {
    let architecture = if profile == CpuProfile::Neon {
        "aarch64"
    } else {
        "x86_64"
    };
    let target = format!("{architecture}-unknown-linux-gnu");
    let mut cfg = vec![
        format!("target_arch=\"{architecture}\""),
        "target_os=\"linux\"".into(),
    ];
    cfg.extend(
        profile
            .required()
            .iter()
            .map(|feature| format!("target_feature=\"{feature}\"")),
    );
    cfg.sort();
    let rustflags = profile.flags();
    CpuBuildConfig {
        schema_version: 1,
        protocol: "explicit_cpu_simd_build_v1".into(),
        architecture: architecture.into(),
        profile: profile.name().into(),
        target_triple: target.clone(),
        required_features: profile.required(),
        encoded_rustflags: rustflags.join("\u{1f}"),
        rustflags,
        rustc_version_verbose: format!(
            "labelled compiler fixture; no build performed\nhost: {target}\n"
        ),
        rustc_cfg: cfg,
        cargo_lock_sha256: "a".repeat(64),
        cargo_features: Vec::new(),
        resolved_rustc: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn features(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|s| (*s).into()).collect()
    }

    #[test]
    fn auto_requires_complete_safe_feature_set_and_honours_portable_override() {
        let complete = features(&["avx", "avx2", "f16c", "fma"]);
        assert_eq!(
            select_profile("auto", "x86_64", &complete).unwrap(),
            CpuProfile::Avx2FmaF16c
        );
        for required in &complete {
            let mut missing = complete.clone();
            missing.remove(required);
            assert_eq!(
                select_profile("auto", "x86_64", &missing).unwrap(),
                CpuProfile::Portable
            );
        }
        assert_eq!(
            select_profile("portable", "x86_64", &complete).unwrap(),
            CpuProfile::Portable
        );
        assert_eq!(
            select_profile("auto", "aarch64", &features(&["neon"])).unwrap(),
            CpuProfile::Neon
        );
        assert!(select_profile("native", "x86_64", &complete).is_err());
    }

    #[test]
    fn frozen_requirements_are_checked_without_compiler_and_cannot_be_reduced() {
        let frozen = fixture_config(CpuProfile::Avx2FmaF16c);
        frozen
            .verify_host_requirements("x86_64", &features(&["avx", "avx2", "f16c", "fma"]))
            .unwrap();
        assert!(
            frozen
                .verify_host_requirements("x86_64", &features(&["avx", "avx2", "f16c"]))
                .is_err()
        );
        assert!(
            frozen
                .verify_host_requirements("aarch64", &features(&["neon"]))
                .is_err()
        );
        let mut changed = frozen.clone();
        changed.required_features.clear();
        assert!(changed.verify_frozen().is_err());
        changed = frozen.clone();
        changed.rustflags[1] = "target-cpu=native".into();
        assert!(changed.verify_frozen().is_err());
        changed = frozen;
        changed.rustc_cfg.push("target_feature=\"avx512f\"".into());
        changed.rustc_cfg.sort();
        assert!(changed.verify_frozen().is_err());
    }

    #[test]
    fn inherited_flags_and_wrappers_cannot_hide_a_different_build_target() {
        for key in [
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "CARGO_BUILD_TARGET",
            "RUSTC",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTFLAGS",
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
        ] {
            assert!(
                unambiguous_environment(
                    [(key.into(), "unexpected fixture setting".into())].into_iter()
                )
                .is_err()
            );
        }
        unambiguous_environment([("RUSTFLAGS".into(), String::new())].into_iter()).unwrap();
    }

    #[test]
    fn actual_host_selection_requires_only_detected_features() {
        let available = host_features();
        let profile = select_profile("auto", std::env::consts::ARCH, &available).unwrap();
        assert!(
            profile
                .required()
                .iter()
                .all(|feature| available.contains(feature))
        );
        let portable = fixture_config(CpuProfile::Portable);
        let optimised = fixture_config(CpuProfile::Avx2FmaF16c);
        assert_ne!(
            portable.target_directory().unwrap(),
            optimised.target_directory().unwrap()
        );
    }

    #[test]
    fn cuda_and_cpu_share_isa_contract_but_never_the_build_cache() {
        let cpu = fixture_config(CpuProfile::Portable);
        let old = serde_json::to_value(&cpu).unwrap();
        assert!(old.get("cargo_features").is_none());
        assert!(old.get("resolved_rustc").is_none());
        let reloaded: CpuBuildConfig = serde_json::from_value(old.clone()).unwrap();
        assert_eq!(serde_json::to_value(reloaded).unwrap(), old);
        let mut cuda = cpu.clone();
        cuda.cargo_features = vec!["cuda".into()];
        cuda.verify_frozen().unwrap();
        assert_eq!(cuda.required_features, cpu.required_features);
        assert_eq!(cuda.rustflags, cpu.rustflags);
        assert_ne!(
            cuda.target_directory().unwrap(),
            cpu.target_directory().unwrap()
        );
        assert!(
            cuda.target_directory()
                .unwrap()
                .starts_with("target/cuda-simd")
        );
        cuda.cargo_features.push("unreviewed".into());
        assert!(cuda.verify_frozen().is_err());
    }

    #[test]
    fn compiler_availability_is_only_required_for_new_builds() {
        let mut frozen = fixture_config(if std::env::consts::ARCH == "aarch64" {
            CpuProfile::Neon
        } else {
            CpuProfile::Portable
        });
        frozen.verify_frozen().unwrap();
        assert!(
            frozen
                .build_environment()
                .unwrap_err()
                .to_string()
                .contains("historical CPU receipt")
        );
        frozen.resolved_rustc = Some(ResolvedRustCompiler {
            executable: "/deliberately-absent-labelled-test-toolchain/bin/rustc".into(),
            sha256: "1".repeat(64),
            bytes: 1,
        });
        // Replay checks the stored contract without reopening a toolchain.
        frozen.verify_frozen().unwrap();
        frozen
            .verify_host_requirements(std::env::consts::ARCH, &host_features())
            .unwrap();
        assert!(frozen.build_environment().is_err());
        let rebound: CpuBuildConfig =
            serde_json::from_value(serde_json::to_value(&frozen).unwrap()).unwrap();
        assert_eq!(rebound, frozen);
        frozen.resolved_rustc.as_mut().unwrap().executable = "relative-rustc".into();
        assert!(frozen.verify_frozen().is_err());
    }

    #[test]
    fn compiler_changes_fail_before_the_replacement_can_be_executed() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("rustc");
        fs::write(&executable, b"labelled non-executable compiler fixture A").unwrap();
        let compiler = ResolvedRustCompiler {
            executable: executable.canonicalize().unwrap(),
            sha256: artefact::file_hash(&executable).unwrap(),
            bytes: fs::metadata(&executable).unwrap().len(),
        };
        // Preserve length to require an actual content check, not stat trust.
        fs::write(&executable, b"labelled non-executable compiler fixture B").unwrap();
        assert_eq!(fs::metadata(&executable).unwrap().len(), compiler.bytes);
        assert!(
            compiler
                .verify_for_build("labelled fixture; never executed")
                .unwrap_err()
                .to_string()
                .contains("changed after inspection")
        );
    }

    #[test]
    fn cargo_build_environment_binds_compiler_and_clears_every_wrapper_alias() {
        let compiler = ResolvedRustCompiler {
            executable: "/labelled test path with spaces/bin/rustc".into(),
            sha256: "1".repeat(64),
            bytes: 1,
        };
        let values =
            compiler_environment(&compiler, "-C\u{1f}target-cpu=generic", "fixture-target");
        for key in ["RUSTC", "CARGO_BUILD_RUSTC"] {
            assert_eq!(values[key].as_os_str(), compiler.executable.as_os_str());
        }
        for key in [
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "CARGO_BUILD_RUSTC_WRAPPER",
            "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
            "RUSTFLAGS",
            "CARGO_BUILD_RUSTFLAGS",
        ] {
            assert!(values[key].is_empty());
        }
        assert_eq!(
            values["CARGO_ENCODED_RUSTFLAGS"],
            "-C\u{1f}target-cpu=generic"
        );
        assert_eq!(values["CARGO_BUILD_TARGET"], "fixture-target");
    }
}
