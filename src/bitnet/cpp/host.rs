//! Execution guards for the reviewed x86 I2_S codec and native build cache.
//! Historical receipt verification remains independent of the current host.
use super::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
};

const REQUIRED: [(&str, &str); 6] = [
    ("GGML_AVX", "avx"),
    ("GGML_AVX2", "avx2"),
    ("GGML_BMI2", "bmi2"),
    ("GGML_F16C", "f16c"),
    ("GGML_FMA", "fma"),
    ("GGML_SSE42", "sse4.2"),
];
const DISABLED: [&str; 8] = [
    "GGML_AVX512",
    "GGML_AVX512_BF16",
    "GGML_AVX512_VBMI",
    "GGML_AVX512_VNNI",
    "GGML_AVX_VNNI",
    "GGML_AMX_BF16",
    "GGML_AMX_INT8",
    "GGML_AMX_TILE",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeCpuRequirements {
    pub operating_system: String,
    pub architecture: String,
    pub i2_s_block_values: usize,
    pub required_cpu_features: Vec<String>,
    pub cmake_cache: Artefact,
}

fn features() -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    #[cfg(target_arch = "x86_64")]
    for (name, available) in [
        ("avx", std::is_x86_feature_detected!("avx")),
        ("avx2", std::is_x86_feature_detected!("avx2")),
        ("bmi2", std::is_x86_feature_detected!("bmi2")),
        ("f16c", std::is_x86_feature_detected!("f16c")),
        ("fma", std::is_x86_feature_detected!("fma")),
        ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
    ] {
        if available {
            result.insert(name.into());
        }
    }
    result
}

fn compatible(
    os: &str,
    architecture: &str,
    available: &BTreeSet<String>,
    required: &[String],
) -> Result<()> {
    ensure!(
        os == "linux" && architecture == "x86_64",
        "reviewed native BitNet I2_S128 execution requires Linux x86_64; this host is {os}/{architecture}"
    );
    let missing = required
        .iter()
        .filter(|name| !available.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        missing.is_empty(),
        "native BitNet engine requires unavailable CPU/OS features: {}",
        missing.join(", ")
    );
    Ok(())
}

pub(super) fn validate_build_host() -> Result<()> {
    compatible(
        std::env::consts::OS,
        std::env::consts::ARCH,
        &features(),
        &REQUIRED
            .iter()
            .map(|(_, name)| (*name).into())
            .collect::<Vec<_>>(),
    )
}

fn entries(text: &str) -> Result<BTreeMap<String, (String, String)>> {
    let mut result = BTreeMap::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#') && !s.starts_with("//"))
    {
        let (head, value) = line
            .split_once('=')
            .context("native CMake cache entry lacks value")?;
        let (key, kind) = head
            .rsplit_once(':')
            .context("native CMake cache entry lacks type")?;
        ensure!(
            !key.is_empty()
                && !kind.is_empty()
                && result
                    .insert(key.into(), (kind.into(), value.into()))
                    .is_none(),
            "native CMake cache has an empty or duplicated key: {key}"
        );
    }
    Ok(result)
}

fn configured(text: &str, receipt: Artefact) -> Result<NativeCpuRequirements> {
    let cache = entries(text)?;
    let exact = |name: &str, kind: &str, value: &str| -> Result<()> {
        ensure!(
            cache
                .get(name)
                .is_some_and(|actual| actual.0 == kind && actual.1 == value),
            "native CMake cache requires {name}:{kind}={value}"
        );
        Ok(())
    };
    exact("CMAKE_BUILD_TYPE", "STRING", "Release")?;
    for name in [
        "GGML_NATIVE",
        "GGML_CUDA",
        "GGML_CPU_ALL_VARIANTS",
        "GGML_BACKEND_DL",
        "GGML_BLAS",
        "GGML_LTO",
        "BITNET_ARM_TL1",
        "BITNET_X86_TL2",
    ] {
        exact(name, "BOOL", "OFF")?;
    }
    exact("GGML_CPU", "BOOL", "ON")?;
    for (name, _) in REQUIRED {
        exact(name, "BOOL", "ON")?;
    }
    for name in DISABLED {
        exact(name, "BOOL", "OFF")?;
    }
    for (key, (kind, value)) in &cache {
        if kind == "BOOL"
            && [
                "GGML_AVX", "GGML_SSE", "GGML_BMI", "GGML_FMA", "GGML_F16", "GGML_AMX",
            ]
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            ensure!(
                REQUIRED.iter().any(|(known, _)| key == known) || DISABLED.contains(&key.as_str()),
                "native CMake cache contains an unreviewed ISA switch: {key}"
            );
        }
        // General flags must be empty below; reject hidden ISA/math additions
        // elsewhere too, including inactive CMake configurations and link flags.
        ensure!(
            ![
                "-march",
                "-mcpu",
                "-mavx",
                "-msse",
                "-mbmi",
                "-mfma",
                "-mf16",
                "-mamx",
                "-ffast-math",
                "-funsafe-math",
                "-Ofast",
                "/arch:"
            ]
            .iter()
            .any(|flag| value.contains(flag)),
            "native cache contains custom ISA/unsafe maths flags in {key}"
        );
        if key == "CMAKE_TOOLCHAIN_FILE" {
            ensure!(
                value.is_empty(),
                "native cross-toolchain cache is unreviewed"
            );
        }
        if key.starts_with("CMAKE_") && key.ends_with("_COMPILER_TARGET") {
            ensure!(
                value.is_empty(),
                "native compiler target override is unreviewed"
            );
        }
    }
    for language in ["C", "CXX", "ASM"] {
        for (suffix, expected) in [
            ("", ""),
            ("_DEBUG", "-g"),
            ("_RELEASE", "-O3 -DNDEBUG"),
            ("_MINSIZEREL", "-Os -DNDEBUG"),
            ("_RELWITHDEBINFO", "-O2 -g -DNDEBUG"),
        ] {
            exact(
                &format!("CMAKE_{language}_FLAGS{suffix}"),
                "STRING",
                expected,
            )?;
        }
    }
    for name in ["GGML_CPU_ARM_ARCH", "GGML_CPU_POWERPC_CPUTYPE"] {
        exact(name, "STRING", "")?;
    }
    Ok(NativeCpuRequirements {
        operating_system: "linux".into(),
        architecture: "x86_64".into(),
        i2_s_block_values: 128,
        required_cpu_features: REQUIRED.iter().map(|(_, name)| (*name).into()).collect(),
        cmake_cache: receipt,
    })
}

pub(super) fn validate_configured_build(path: &Path) -> Result<NativeCpuRequirements> {
    validate_build_host()?;
    let root = std::env::current_dir()?.canonicalize()?;
    let receipt = Artefact::capture(&root, path.canonicalize()?.strip_prefix(&root)?)?;
    ensure!(receipt.bytes <= 4 << 20, "native CMake cache exceeds bound");
    configured(&fs::read_to_string(receipt.verify(&root)?)?, receipt)
}

/// Verify actual cached settings and binary architecture before a native
/// executable starts. This does not change or reinterpret its saved receipt.
pub fn validate_execution_host(identity: &EngineIdentity) -> Result<NativeCpuRequirements> {
    validate_build_host()?;
    let root = std::env::current_dir()?.canonicalize()?;
    ensure!(
        identity.cmake_cache.bytes <= 4 << 20,
        "native CMake cache exceeds bound"
    );
    let requirement = configured(
        &fs::read_to_string(identity.cmake_cache.verify(&root)?)?,
        identity.cmake_cache.clone(),
    )?;
    compatible(
        std::env::consts::OS,
        std::env::consts::ARCH,
        &features(),
        &requirement.required_cpu_features,
    )?;
    let binary = identity
        .binaries
        .iter()
        .find(|file| file.path == identity.build_directory.join("bin/llama-server"))
        .context("native server is absent from engine receipt")?;
    let mut file = fs::File::open(binary.verify(&root)?)?;
    let mut header = [0_u8; 20];
    file.read_exact(&mut header)?;
    ensure!(
        &header[..4] == b"\x7fELF"
            && header[4] == 2
            && header[5] == 1
            && u16::from_le_bytes([header[18], header[19]]) == 62,
        "native BitNet server must be the reviewed Linux x86_64 ELF executable"
    );
    Ok(requirement)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> String {
        let mut lines = vec![
            "CMAKE_BUILD_TYPE:STRING=Release".into(),
            "GGML_CPU:BOOL=ON".into(),
            "GGML_CPU_ARM_ARCH:STRING=".into(),
            "GGML_CPU_POWERPC_CPUTYPE:STRING=".into(),
        ];
        for key in [
            "GGML_NATIVE",
            "GGML_CUDA",
            "GGML_CPU_ALL_VARIANTS",
            "GGML_BACKEND_DL",
            "GGML_BLAS",
            "GGML_LTO",
            "BITNET_ARM_TL1",
            "BITNET_X86_TL2",
        ] {
            lines.push(format!("{key}:BOOL=OFF"));
        }
        for (key, _) in REQUIRED {
            lines.push(format!("{key}:BOOL=ON"));
        }
        for key in DISABLED {
            lines.push(format!("{key}:BOOL=OFF"));
        }
        for language in ["C", "CXX", "ASM"] {
            for (suffix, value) in [
                ("", ""),
                ("_DEBUG", "-g"),
                ("_RELEASE", "-O3 -DNDEBUG"),
                ("_MINSIZEREL", "-Os -DNDEBUG"),
                ("_RELWITHDEBINFO", "-O2 -g -DNDEBUG"),
            ] {
                lines.push(format!("CMAKE_{language}_FLAGS{suffix}:STRING={value}"));
            }
        }
        lines.join("\n")
    }
    fn receipt() -> Artefact {
        Artefact {
            path: "labelled-cache-fixture-only".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        }
    }
    #[test]
    fn reviewed_cache_freezes_every_required_feature_without_mutating_receipt() {
        let parsed = configured(&fixture(), receipt()).unwrap();
        assert_eq!(parsed.i2_s_block_values, 128);
        let all = parsed
            .required_cpu_features
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        compatible("linux", "x86_64", &all, &parsed.required_cpu_features).unwrap();
        for missing in ["bmi2", "avx2"] {
            let mut reduced = all.clone();
            reduced.remove(missing);
            assert!(
                compatible("linux", "x86_64", &reduced, &parsed.required_cpu_features).is_err()
            );
        }
        assert!(compatible("linux", "aarch64", &all, &parsed.required_cpu_features).is_err());
        assert!(compatible("macos", "x86_64", &all, &parsed.required_cpu_features).is_err());
    }
    #[test]
    fn unknown_duplicate_native_and_custom_compiler_flags_fail_closed() {
        for changed in [
            fixture().replace("GGML_NATIVE:BOOL=OFF", "GGML_NATIVE:BOOL=ON"),
            fixture().replace("GGML_BMI2:BOOL=ON", "GGML_BMI2:BOOL=OFF"),
            fixture() + "\nGGML_AVX512_NEW:BOOL=ON",
            fixture() + "\nGGML_AVX2:BOOL=OFF",
            fixture().replace(
                "CMAKE_C_FLAGS:STRING=",
                "CMAKE_C_FLAGS:STRING=-march=native",
            ),
            fixture().replace("-O3 -DNDEBUG", "-Ofast -DNDEBUG"),
        ] {
            assert!(configured(&changed, receipt()).is_err());
        }
    }
}
