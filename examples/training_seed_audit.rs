//! Independently audit whole-training seed variants without constructing a model.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lec_distil::{
    artefact::{self, Artefact},
    distillation::{DistillationWorkerSpec, checkpoint},
    experiments::StudyConfig,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    left: PathBuf,
    #[arg(long)]
    right: PathBuf,
    #[arg(long)]
    left_reload: PathBuf,
    #[arg(long)]
    right_reload: PathBuf,
    #[arg(long, default_value = "config/distillation-study.toml")]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn read(path: impl AsRef<Path>) -> Result<Value> {
    let path = path.as_ref();
    ensure!(
        fs::metadata(path)?.len() <= 32 << 20,
        "audit JSON exceeds its bound"
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(root, path.canonicalize()?.strip_prefix(root)?.to_owned())
}

fn common_spec(spec: &Value) -> Result<Value> {
    let mut value = spec.clone();
    let object = value
        .as_object_mut()
        .context("worker spec is not an object")?;
    for field in ["arm_id", "seed", "output"] {
        ensure!(
            object.remove(field).is_some(),
            "worker spec omitted {field}"
        );
    }
    Ok(value)
}

fn matched(left: &Value, right: &Value) -> Result<()> {
    ensure!(
        common_spec(left)? == common_spec(right)?,
        "seed variants changed another training method, source, runtime or optimiser setting"
    );
    ensure!(
        left["seed"].as_u64().is_some()
            && right["seed"].as_u64().is_some()
            && left["seed"] != right["seed"],
        "seed variants require different explicit seeds"
    );
    Ok(())
}

fn exposures(order: &Value) -> Result<BTreeMap<String, usize>> {
    let mut counts = BTreeMap::new();
    for row in order.as_array().context("frozen update order absent")? {
        *counts
            .entry(
                row["sample_id"]
                    .as_str()
                    .context("update sample absent")?
                    .to_owned(),
            )
            .or_default() += 1;
    }
    Ok(counts)
}

fn phase(
    root: &Path,
    directory: &Path,
    name: &str,
    proof: &checkpoint::ExperimentalCheckpointManifest,
) -> Result<Artefact> {
    let value = read(directory.join(format!("helper-launch-{name}.json")))?;
    let request: Artefact = serde_json::from_value(value["request"].clone())?;
    let actual: DistillationWorkerSpec = serde_json::from_slice(&fs::read(request.verify(root)?)?)?;
    let expected = serde_json::to_value(&proof.spec)?;
    ensure!(
        value["schema_version"] == 1
            && value["evidence_kind"] == "immutable_native_distillation_helper_phase_launch"
            && value["phase"] == name
            && value["production_eligible"] == false
            && value["common_launch"]
                == serde_json::to_value(capture(root, &directory.join("helper-launch.json"))?)?
            && value["canonical_request_sha256"]
                == artefact::hash(&serde_json::to_vec(&proof.spec)?)
            && serde_json::to_value(actual)? == expected,
        "helper {name} phase changed its schema, request or common launch"
    );
    Ok(request)
}

fn common_launch(root: &Path, launch: &Value, freeze: &Value) -> Result<()> {
    ensure!(
        launch["schema_version"] == 1
            && launch["evidence_kind"] == "immutable_native_distillation_helper_launch_contract"
            && launch["production_eligible"] == false
            && launch["engine_sha256"] == freeze["engine_binary_sha256"],
        "common training helper identity differs"
    );
    // Resolve the exact retained bytes in this installation. The original
    // absolute launch path stays untouched as historical evidence.
    let pinned = lec_distil::engine::resolve(
        &root.join("state/engines"),
        launch["engine_sha256"]
            .as_str()
            .context("helper SHA absent")?,
    )?;
    ensure!(
        launch["engine_bytes"] == pinned.bytes,
        "helper executable length changed"
    );
    let threads = &launch["numerical_threads"];
    let positive = |key: &str| -> Result<usize> {
        let count = threads["environment"][key]
            .as_str()
            .context("numerical setting absent")?
            .parse::<usize>()?;
        ensure!(count > 0, "numerical setting {key} is zero");
        Ok(count)
    };
    let requested = positive("RAYON_NUM_THREADS")?;
    let allocated = positive("LEC_STUDY_CPU_THREADS")?;
    ensure!(
        requested <= allocated
            && positive("OMP_NUM_THREADS")? <= allocated
            && positive("MKL_NUM_THREADS")? <= allocated
            && threads["effective_candle_threads"] == requested
            && threads["effective_rayon_threads"] == requested,
        "effective numerical thread pools differ from their allocation"
    );
    Ok(())
}

fn reload(
    root: &Path,
    path: &Path,
    proof: &checkpoint::ExperimentalCheckpointManifest,
    launch: &Value,
    directory: &Path,
) -> Result<Value> {
    let value = read(path)?;
    let config = read(proof.spec.student_base.join("config.json"))?;
    let logits = config["vocab_size"]
        .as_u64()
        .context("base vocabulary absent")?
        .checked_mul(2)
        .context("reload vocabulary count overflow")?;
    ensure!(
        value["schema_version"] == 1
            && value["evidence_kind"] == "actual_fresh_process_checkpoint_reload"
            && value["checkpoint_sha256"] == proof.adapter_sha256
            && value["engine_sha256"] == launch["engine_sha256"]
            && value["finite_logits"] == logits
            && value["execution"]["engine"] == "native_rust_candle"
            && value["execution"]["checkpoint_id"] == proof.adapter_sha256
            && value["execution"]["actual_device"] == "Cpu"
            && value["execution"]["actual_dtype"] == "F32"
            && value["execution"]["cpu_fallback"] == false
            && value["execution"]["cuda_active"] == false
            && value["execution"]["requested"]
                == serde_json::to_value(&proof.spec.student_runtime)?
            && value["production_eligible"] == false,
        "separate-process reload is missing or belongs to another runtime/checkpoint"
    );
    let request: Artefact = serde_json::from_value(value["request"].clone())?;
    let actual: DistillationWorkerSpec = serde_json::from_slice(&fs::read(request.verify(root)?)?)?;
    ensure!(
        serde_json::to_value(actual)? == serde_json::to_value(&proof.spec)?,
        "reload request differs"
    );
    let verify_request = phase(root, directory, "verify", proof)?;
    let worker_request = phase(root, directory, "worker", proof)?;
    ensure!(
        serde_json::to_value(&verify_request)? == serde_json::to_value(&request)?
            && serde_json::to_value(&worker_request)? == serde_json::to_value(&request)?,
        "reload did not use the frozen common helper launch"
    );
    Ok(
        json!({"receipt":capture(root,path)?,"finite_logits":value["finite_logits"],"request":request}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "preserve the existing audit; choose a new output"
    );
    let root = std::env::current_dir()?.canonicalize()?;
    let config = StudyConfig::load(&args.config)?;
    let mut proofs = Vec::new();
    let mut freezes = Vec::new();
    let mut launches = Vec::new();
    let mut rows = Vec::new();
    for (directory, reload_path) in [
        (&args.left, &args.left_reload),
        (&args.right, &args.right_reload),
    ] {
        let raw = read(directory.join("manifest.json"))?;
        let spec: DistillationWorkerSpec = serde_json::from_value(raw["spec"].clone())?;
        ensure!(
            raw["spec"]["student_kind"] == "conventional_qwen_lora"
                && spec.output.canonicalize()? == directory.canonicalize()?
                && config.seeds.contains(&spec.seed),
            "audit requires conventional checkpoints and declared training seeds"
        );
        let assets = lec_distil::model::validate_assets(&spec.student_base)?;
        // This public verifier reads adapter tensors and full run provenance;
        // it does not instantiate a transformer or perform inference.
        let proof = checkpoint::validate_checkpoint(directory, &assets)?;
        let freeze = read(directory.join("freeze.json"))?;
        let launch = read(directory.join("helper-launch.json"))?;
        common_launch(&root, &launch, &freeze)?;
        let loaded = reload(&root, reload_path, &proof, &launch, directory)?;
        rows.push(json!({"manifest":capture(&root,&directory.join("manifest.json"))?,
            "freeze":capture(&root,&directory.join("freeze.json"))?,"launch":capture(&root,&directory.join("helper-launch.json"))?,
            "seed":proof.spec.seed,"steps":proof.steps,"adapter_sha256":proof.adapter_sha256,
            "initial_adapter_sha256":proof.initial_adapter_sha256,"sample_update_counts":exposures(&freeze["order"])?,
            "reload":loaded}));
        proofs.push(proof);
        freezes.push(freeze);
        launches.push(launch);
    }
    matched(
        &serde_json::to_value(&proofs[0].spec)?,
        &serde_json::to_value(&proofs[1].spec)?,
    )?;
    for field in [
        "encoded_sequences",
        "excluded_overlength_samples",
        "base",
        "evidence",
        "engine_binary_sha256",
        "initial_checkpoint",
    ] {
        ensure!(
            freezes[0][field] == freezes[1][field],
            "seed variants changed frozen {field}"
        );
    }
    let mut thread_contracts = Vec::new();
    for launch in &launches {
        let mut threads = launch["numerical_threads"].clone();
        let affinity = threads
            .as_object_mut()
            .context("numerical launch absent")?
            .remove("linux_cpu_affinity")
            .context("launch affinity absent")?;
        thread_contracts.push((threads, affinity));
    }
    ensure!(
        thread_contracts[0].0 == thread_contracts[1].0,
        "seed variants changed effective numerical pools or thread settings"
    );
    let report = json!({"schema_version":1,"evidence_kind":"independent_verified_whole_training_seed_audit",
        "passed":true,"model_constructed":false,"inference_performed":false,
        "auditor":capture(&root,&std::env::current_exe()?)?,"source_sha256":artefact::hash(include_bytes!("training_seed_audit.rs")),
        "declared_config":capture(&root,&args.config)?,"variants":rows,
        "eligible_complete_samples":freezes[0]["encoded_sequences"].as_object().context("encoded samples absent")?.len(),
        "same_realised_order":freezes[0]["order"] == freezes[1]["order"],
        "same_initial_adapter":proofs[0].initial_adapter_sha256 == proofs[1].initial_adapter_sha256,
        "same_final_adapter":proofs[0].adapter_sha256 == proofs[1].adapter_sha256,
        "left_affinity":thread_contracts[0].1,"right_affinity":thread_contracts[1].1,
        "quality_evaluation":false,"held_out_used":false,"production_eligible":false,
        "limitations":["The captured configuration proves declared seed membership; this audit alone does not establish preregistration chronology.",
            "Whole-training seed changes initialisation and sample shuffling, including any partial final cycle.",
            "Affinity placement is disclosed separately; effective numerical pools and all remaining training settings must match.",
            "Changed or reloaded weights do not establish coding improvement, statistical seed equivalence or an assessment gate result."]});
    artefact::write_json(&args.output, &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"passed":true,"report":args.output,
        "eligible_complete_samples":report["eligible_complete_samples"],"quality_evaluation":false}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_labels_cannot_hide_an_optimiser_or_source_change() {
        let left = json!({"arm_id":"left","output":"left","seed":1729,"learning_rate":0.0002,"evidence_bundle":"original"});
        let mut right = json!({"arm_id":"right","output":"right","seed":2718,"learning_rate":0.0002,"evidence_bundle":"original"});
        matched(&left, &right).unwrap();
        for (field, value) in [
            ("learning_rate", json!(0.001)),
            ("evidence_bundle", json!("replacement")),
        ] {
            let previous = right[field].clone();
            right[field] = value;
            assert!(matched(&left, &right).is_err());
            right[field] = previous;
        }
        right["seed"] = left["seed"].clone();
        assert!(matched(&left, &right).is_err());
    }

    #[test]
    fn partial_cycles_retain_their_actual_per_sample_exposure() {
        let left = json!([{"sample_id":"a"},{"sample_id":"b"},{"sample_id":"a"}]);
        let right = json!([{"sample_id":"b"},{"sample_id":"a"},{"sample_id":"b"}]);
        assert_eq!(
            exposures(&left).unwrap(),
            BTreeMap::from([("a".into(), 2), ("b".into(), 1)])
        );
        assert_ne!(exposures(&left).unwrap(), exposures(&right).unwrap());
        assert!(exposures(&json!([{}])).is_err());
    }
}
