//! Read retained failures only; diagnostic causes never replace grading verdicts.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
fn read(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn json_files(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut files = fs::read_dir(path)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.extension().is_some_and(|s| s == "json"));
    files.sort();
    Ok(files)
}
fn matching_journals(run_id: &str) -> Result<Vec<Value>> {
    let root = Path::new("state/sandbox-owned-runs");
    let mut results = Vec::new();
    if root.exists() {
        for scope in fs::read_dir(root)? {
            let scope = scope?;
            if !scope.file_type()?.is_dir() {
                continue;
            }
            let path = scope.path().join(format!("lec-{run_id}.json"));
            if path.exists() {
                results.push(json!({"artefact":capture(&path)?,"record":read(&path)?}));
            }
        }
    }
    Ok(results)
}

pub(super) fn collect(candidates: &[PathBuf], event_paths: &[PathBuf]) -> Result<Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    let mut reports = Vec::new();
    let mut all_names = BTreeSet::new();
    for directory in candidates {
        let plan = read(&directory.join("plan.json"))?;
        let aggregate = read(&directory.join("report.json"))?;
        let mut counts = BTreeMap::<String, usize>::new();
        let mut rows = Vec::new();
        let mut complete = 0;
        let mut original = BTreeMap::<PathBuf, Artefact>::new();
        for path in json_files(&directory.join("generations"))? {
            let raw = read(&path)?;
            complete += usize::from(
                raw["generation"]["complete"] == true
                    && raw["generation"]["completion"]
                        .as_str()
                        .is_some_and(|s| !s.trim().is_empty())
                    && raw["model_error"].is_null(),
            );
            let file = capture(&path)?;
            original.insert(file.path.clone(), file);
        }
        for observation in json_files(&directory.join("observations"))? {
            let row = read(&observation)?;
            let file = capture(&observation)?;
            original.insert(file.path.clone(), file.clone());
            let Some(error) = row["grading_error"].as_str() else {
                continue;
            };
            let category = if error.contains("trusted audit artifact missing") {
                "missing_trusted_audit"
            } else if error.contains("owned sandbox inspection deadline") {
                "owned_cleanup_inspection_timeout"
            } else {
                "other_grading_error"
            };
            *counts.entry(category.into()).or_default() += 1;
            let raw_file: Artefact = serde_json::from_value(row["raw_generation"].clone())?;
            let raw = read(&raw_file.verify(&root)?)?;
            let mut stored = Vec::new();
            for path in json_files(&directory.join("sandbox"))? {
                if path
                    .file_name()
                    .is_some_and(|s| s.to_string_lossy().ends_with(".audit.json"))
                {
                    continue;
                }
                let grade = read(&path)?;
                if grade["task_id"] != row["task_id"] {
                    continue;
                }
                let run_id = grade["run_id"]
                    .as_str()
                    .context("stored sandbox report lacks run ID")?;
                all_names.insert(format!("lec-{run_id}"));
                let audit =
                    PathBuf::from(grade["audit_path"].as_str().context("audit path absent")?);
                let audit = if audit.is_absolute() {
                    audit
                } else {
                    root.join(audit)
                };
                let audit_file = if audit.try_exists()? {
                    Some(capture(&audit)?)
                } else {
                    None
                };
                let file = capture(&path)?;
                original.insert(file.path.clone(), file.clone());
                stored.push(json!({"artefact":file,"failure":grade["failure"],"quarantined":grade["quarantined"],
                    "trace_complete":grade["trace_complete"],"duration_ms":grade["duration_ms"],
                    "timeout_evidence":grade["timeout_evidence"],"declared_audit_path":grade["audit_path"],
                    "declared_audit_sha256":grade["audit_sha256"],"retained_audit":audit_file,
                    "ownership_journals":matching_journals(run_id)?}));
            }
            rows.push(json!({"task_id":row["task_id"],"source_id":row["source_id"],"category":category,
                "observation":file,"grading_error":error,"raw_generation":raw_file,
                "generation_complete":raw["generation"]["complete"],"model_error":raw["model_error"],
                "stored_sandbox_reports":stored,"success_retained":row["success"],
                "candidate_quality_inferred":false}));
        }
        for file in original.values() {
            file.verify(&root)?;
        }
        reports.push(json!({"evaluation":directory,"plan":capture(&directory.join("plan.json"))?,
            "report":capture(&directory.join("report.json"))?,"complete_generations":complete,
            "planned":aggregate["planned"],"retained_error_count":rows.len(),"category_counts":counts,
            "frozen_sandbox":plan["sandbox"],"rows":rows,"input_artefacts_unchanged":true,
            "original_artefacts":original}));
    }
    let mut events = Vec::new();
    for path in event_paths {
        let contents = fs::read_to_string(path)?;
        let mut matched = Vec::new();
        let mut first = None::<i64>;
        let mut last = None::<i64>;
        let mut count = 0;
        for line in contents.lines().filter(|s| !s.trim().is_empty()) {
            let event: Value = serde_json::from_str(line)?;
            count += 1;
            if let Some(time) = event["time"].as_i64() {
                first = Some(first.map_or(time, |v| v.min(time)));
                last = Some(last.map_or(time, |v| v.max(time)));
            }
            if event["Name"]
                .as_str()
                .is_some_and(|name| all_names.contains(name))
            {
                matched.push(event);
            }
        }
        events.push(
            json!({"artefact":capture(path)?,"event_count":count,"first_unix_seconds":first,
            "last_unix_seconds":last,"matching_error_container_events":matched}),
        );
    }
    Ok(
        json!({"schema_version":1,"evidence_kind":"read_only_retained_study_grading_failure_diagnosis",
        "evaluations":reports,"retained_runtime_event_files":events,"inference_performed":false,
        "generations_replayed_or_replaced":false,"containers_started":false,"held_out_used":false,"production_eligible":false,
        "conclusions":["Complete model generations and grading infrastructure failures are separate observations; the original zero scores and error denominator remain unchanged.",
            "Stored timeout reports and ownership journals are diagnostic evidence, not missing trusted execution audits or successful grades.",
            "A later cleanup acknowledgement does not establish the candidate's missing execution result.",
            "Without matching retained runtime events/audits, the reports cannot distinguish runtime startup/cleanup delay, compiler delay, host contention or candidate runtime as the underlying cause."]}),
    )
}
