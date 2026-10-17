//! Model-free reconstruction of the completed teacher-search evidence.
//! This diagnostic reads original artefacts and writes only its own report.
use std::{fs, path::Path};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Path::new("reports/comparative/search/round-002-review-001");
    let paths = [
        "reports/comparative/search/round-001/trials/qwen-coder-7b-q4_k_m/comparison",
        "reports/comparative/search/round-002/trials/deepseek-coder-6_7b-q4_k_m/comparison",
        "reports/comparative/search/round-002/trials/qwen3-4b-instruct-2507-q4_k_m/comparison",
    ];
    let mut verified = Vec::new();
    let mut all_valid = true;
    for path in paths {
        let result = lec_distil::comparative::verify_completed_comparison(Path::new(path));
        let mut observations = Vec::new();
        let plan: Value = serde_json::from_slice(&fs::read(Path::new(path).join("plan.json"))?)?;
        for model in plan["models"].as_array().ok_or("models missing")? {
            let id = model["model"]["id"].as_str().ok_or("model ID missing")?;
            for task in plan["tasks"].as_array().ok_or("tasks missing")? {
                let task_id = task["id"].as_str().ok_or("task ID missing")?;
                let observation_path = Path::new(path).join(id).join(format!("{task_id}.json"));
                let o: Value = serde_json::from_slice(&fs::read(&observation_path)?)?;
                observations.push(json!({"path":observation_path,"task_id":task_id,
                    "success":o["success"],"generation_error":o["generation_error"],
                    "grading_error":o["grading_error"],"finish_reason":o["generation"]["finish_reason"],
                    "complete":o["generation"]["complete"]}));
            }
        }
        match result {
            Ok(report) => verified.push(json!({"path":path,"verified":true,"report":report,"observations":observations})),
            Err(error) => {
                all_valid = false;
                verified.push(json!({"path":path,"verified":false,"error":format!("{error:#}"),"observations":observations}));
            }
        }
    }
    let registry = lec_distil::comparative::Registry::load(Path::new("reports/comparative/search/round-002/trials/qwen3-4b-instruct-2507-q4_k_m/registry.toml"))?;
    let pruning = match lec_distil::teacher_search::verify_owned_pruned_model(&registry.models[0]) {
        Ok(value) => json!({"verified":value}),
        Err(error) => { all_valid = false; json!({"verified":false,"error":format!("{error:#}")}) }
    };
    if pruning["verified"] != true { all_valid = false; }
    let evidence_paths = [
        "reports/comparative/search/round-002/report.json",
        "reports/comparative/search/round-002/state.json",
        "reports/comparative/search/round-002/freeze.json",
        "reports/comparative/search/round-002/qualification-pack.json",
        "config/teacher-search-qwen3.toml",
        "offline_bundle/comparative/qwen3-4b-instruct-2507-q4_k_m/.pruned-weights.json",
        "offline_bundle/comparative/qwen3-4b-instruct-2507-q4_k_m/.teacher-search-owned.json",
        "reports/comparative/search/round-002/attempts/qwen3-4b-instruct-2507-q4_k_m/1fd2c1b1-3728-4d0f-b831-e0137880fa07.json",
        "state/engines/b45d32ca11ccc84633ab2de0bb9d195dc38a61c78e2ef1a87c03ea90091aa301/lec-distil",
        "reports/comparative/search/round-002-review-001/review.rs",
        "reports/comparative/search/round-002-review-001/review",
    ];
    let mut evidence = Vec::new();
    for path in evidence_paths {
        evidence.push(json!({"path":path,"bytes":fs::metadata(path)?.len(),
            "sha256":lec_distil::artefact::file_hash(Path::new(path))?}));
    }
    let result = json!({"schema_version":1,"review":"read_only_model_free_teacher_search_round_002",
        "all_source_verifications_passed":all_valid,"comparisons":verified,"qwen3_owned_pruning":pruning,
        "evidence":evidence,"inference_executed":false,"containers_started":false,"held_out_access":false});
    fs::write(base.join("verification.json"), serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !all_valid { std::process::exit(1); }
    Ok(())
}
