//! Freeze numerical launches before model allocation, without rewriting history.
use anyhow::{Context, Result, ensure};
use lec_distil::{
    artefact::{self, Artefact},
    distillation::DistillationWorkerSpec,
    engine,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path},
};

pub(super) enum Phase {
    Worker,
    Prewarm,
    Verify,
}
impl Phase {
    fn name(&self) -> &'static str {
        match self {
            Self::Worker => "worker",
            Self::Prewarm => "prewarm",
            Self::Verify => "verify",
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
struct NumericalThreads {
    environment: BTreeMap<String, Option<String>>,
    effective_candle_threads: usize,
    effective_rayon_threads: usize,
    linux_cpu_affinity: Option<String>,
}

fn thread_contract() -> Result<NumericalThreads> {
    let mut environment = BTreeMap::new();
    for key in [
        "RAYON_NUM_THREADS",
        "LEC_STUDY_CPU_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ] {
        let value = std::env::var(key);
        let value = match value {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("non-UTF-8 numerical thread setting {key}"));
            }
        };
        environment.insert(key.to_owned(), value);
    }
    let positive = |key: &str| -> Result<usize> {
        let value = environment[key]
            .as_deref()
            .with_context(|| format!("new numerical helper launches require explicit {key}"))?;
        let value = value
            .parse::<usize>()
            .with_context(|| format!("invalid positive thread count in {key}"))?;
        ensure!(value > 0, "numerical thread setting {key} must be positive");
        Ok(value)
    };
    let requested = positive("RAYON_NUM_THREADS")?;
    let assigned = positive("LEC_STUDY_CPU_THREADS")?;
    let available = std::thread::available_parallelism()?.get();
    ensure!(
        requested <= assigned && assigned <= available,
        "numerical thread allocation exceeds available affinity/cgroup capacity"
    );
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"] {
        if environment[key].is_some() {
            ensure!(
                positive(key)? <= assigned,
                "numerical thread setting {key} exceeds its declared CPU allocation"
            );
        }
    }
    let effective_candle_threads = candle_core::utils::get_num_threads();
    // This public re-export queries the actual shared Rayon pool. No new
    // dependency or approximation from available logical cores is required.
    let effective_rayon_threads = tokenizers::utils::parallelism::current_num_threads();
    ensure!(
        effective_candle_threads == requested && effective_rayon_threads == requested,
        "effective Candle/Rayon thread counts differ from the declared launch"
    );
    let linux_cpu_affinity = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .map(|value| value.trim().to_owned())
            })
        });
    Ok(NumericalThreads {
        environment,
        effective_candle_threads,
        effective_rayon_threads,
        linux_cpu_affinity,
    })
}

fn check_history(output: &Path) -> Result<()> {
    ensure!(
        !output.join("freeze.json").exists() || output.join("helper-launch.json").exists(),
        "historical worker freeze has no common helper launch receipt; use its original pinned executable or an explicit fresh protocol"
    );
    Ok(())
}

fn check_frozen(path: &Path, expected: &Value) -> Result<()> {
    if path.exists() {
        ensure!(
            serde_json::from_slice::<Value>(&fs::read(path)?)? == *expected,
            "numerical helper launch contract or phase request changed; preserve the original receipt and resume its exact executable/thread environment"
        );
    }
    Ok(())
}

fn freeze(
    root: &Path,
    output: &Path,
    common: &Value,
    phase: &str,
    request: &Artefact,
    canonical_request_sha256: &str,
) -> Result<()> {
    // This guard precedes directory creation, lock files and every write.
    check_history(output)?;
    let common_path = output.join("helper-launch.json");
    let phase_path = output.join(format!("helper-launch-{phase}.json"));
    check_frozen(&common_path, common)?;
    ensure!(
        !phase_path.exists() || common_path.exists(),
        "phase launch receipt has no original common launch contract"
    );
    fs::create_dir_all(output)?;
    let _lease = engine::RunLease::acquire(&output.join(".helper-launch.lock"))?;
    check_history(output)?;
    check_frozen(&common_path, common)?;
    if !common_path.exists() {
        artefact::write_json(&common_path, common)?;
    }
    let common = Artefact::capture(root, common_path.strip_prefix(root)?)?;
    let binding = json!({"schema_version":1,"evidence_kind":"immutable_native_distillation_helper_phase_launch","phase":phase,"common_launch":common,"request":request,"canonical_request_sha256":canonical_request_sha256,"production_eligible":false});
    check_frozen(&phase_path, &binding)?;
    if !phase_path.exists() {
        artefact::write_json(&phase_path, &binding)?;
    }
    Ok(())
}

pub(super) fn bind(path: &Path, spec: &DistillationWorkerSpec, phase: Phase) -> Result<()> {
    ensure!(
        !spec.output.as_os_str().is_empty()
            && spec
                .output
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "numerical helper output must remain relative without traversal"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    let output = root.join(&spec.output);
    check_history(&output)?;
    let request = Artefact::capture(&root, path.to_owned())?;
    let bytes = fs::read(request.verify(&root)?)?;
    let current: DistillationWorkerSpec = serde_json::from_slice(&bytes)?;
    ensure!(
        artefact::hash(&bytes) == request.sha256
            && serde_json::to_value(&current)? == serde_json::to_value(spec)?,
        "helper request changed between parsing and launch binding"
    );
    let executable = std::env::current_exe()?.canonicalize()?;
    let common = json!({"schema_version":1,"evidence_kind":"immutable_native_distillation_helper_launch_contract","executable":executable,"engine_sha256":artefact::file_hash(&executable)?,"engine_bytes":fs::metadata(&executable)?.len(),"numerical_threads":thread_contract()?,"production_eligible":false});
    freeze(
        &root,
        &output,
        &common,
        phase.name(),
        &request,
        &artefact::hash(&serde_json::to_vec(spec)?),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn historical_freeze_is_rejected_without_creating_a_retroactive_receipt() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("round");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("freeze.json"), b"preserved historical fixture").unwrap();
        let placeholder = Artefact {
            path: "labelled-unused".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        assert!(
            freeze(
                root.path(),
                &output,
                &json!({"fixture":true}),
                "worker",
                &placeholder,
                "labelled"
            )
            .is_err()
        );
        assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
        assert_eq!(
            fs::read(output.join("freeze.json")).unwrap(),
            b"preserved historical fixture"
        );
    }
    #[test]
    fn phases_bind_distinct_requests_to_one_immutable_engine_and_thread_contract() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("round");
        let request = Artefact {
            path: "labelled-acquire-request".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        };
        let common = json!({"evidence_kind":"labelled-unit-fixture","engine_sha256":"b".repeat(64),"rayon_threads":4});
        freeze(
            root.path(),
            &output,
            &common,
            "prewarm",
            &request,
            "acquire",
        )
        .unwrap();
        let original = fs::read(output.join("helper-launch.json")).unwrap();
        let mut cached = request.clone();
        cached.path = "labelled-cache-only-request".into();
        cached.sha256 = "c".repeat(64);
        freeze(
            root.path(),
            &output,
            &common,
            "worker",
            &cached,
            "cache_only",
        )
        .unwrap();
        freeze(
            root.path(),
            &output,
            &common,
            "worker",
            &cached,
            "cache_only",
        )
        .unwrap();
        let before = fs::read(output.join("helper-launch-worker.json")).unwrap();
        assert!(freeze(root.path(), &output, &common, "worker", &request, "acquire").is_err());
        let mut changed = common.clone();
        changed["rayon_threads"] = json!(3);
        assert!(
            freeze(
                root.path(),
                &output,
                &changed,
                "worker",
                &cached,
                "cache_only"
            )
            .is_err()
        );
        assert_eq!(
            fs::read(output.join("helper-launch.json")).unwrap(),
            original
        );
        assert_eq!(
            fs::read(output.join("helper-launch-worker.json")).unwrap(),
            before
        );
        let prewarm: Value =
            serde_json::from_slice(&fs::read(output.join("helper-launch-prewarm.json")).unwrap())
                .unwrap();
        let worker: Value = serde_json::from_slice(&before).unwrap();
        assert_eq!(prewarm["common_launch"], worker["common_launch"]);
        assert_ne!(prewarm["request"], worker["request"]);
    }
}
