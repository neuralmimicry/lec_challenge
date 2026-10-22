//! Rejection diagnostics supplement, and never replace, independent verification.
use crate::{artefact, study_evaluate};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    SplitMismatch,
    CheckpointMismatch,
    EvidenceInvalid,
    GenerationIncomplete,
    InferenceTimeout,
    GradingInfrastructureFailure,
    UnhealthyCompletion,
}

#[derive(Clone, Debug, Serialize, Deserialize, thiserror::Error)]
#[error("evaluation rejected ({reason:?}) at {evidence_path}: {context}")]
pub struct EvaluationFailure {
    pub schema_version: u32,
    pub reason: Reason,
    pub evidence_path: PathBuf,
    pub expected: Value,
    pub actual: Value,
    pub context: String,
}

fn failure(
    path: &Path,
    reason: Reason,
    expected: Value,
    actual: Value,
    context: impl Into<String>,
) -> EvaluationFailure {
    EvaluationFailure {
        schema_version: 1,
        reason,
        evidence_path: path.to_owned(),
        expected,
        actual,
        context: context.into(),
    }
}

fn identities(
    report: &Value,
    path: &Path,
    split: &str,
    checkpoint: &Value,
) -> Result<(), EvaluationFailure> {
    if report["split"] != split || report["held_out_used"] == true {
        return Err(failure(
            path,
            Reason::SplitMismatch,
            split.into(),
            report["split"].clone(),
            "undeclared split or held-out access",
        ));
    }
    if report["checkpoint_id"] != *checkpoint {
        return Err(failure(
            path,
            Reason::CheckpointMismatch,
            checkpoint.clone(),
            report["checkpoint_id"].clone(),
            "evaluation belongs to another checkpoint",
        ));
    }
    Ok(())
}

/// Only the independently reconstructed report and its verified raw observations
/// can establish health. A worker exit code or status string cannot override it.
pub fn verify(path: &Path, split: &str, checkpoint: &Value) -> Result<Value, EvaluationFailure> {
    let invalid = |error: anyhow::Error| {
        failure(
            path,
            Reason::EvidenceInvalid,
            Value::Null,
            Value::Null,
            format!("{error:#}"),
        )
    };
    let directory = path
        .parent()
        .ok_or_else(|| invalid(anyhow::anyhow!("evaluation directory missing")))?;
    // Preliminary denial is useful even when a held-out plan cannot deserialize.
    // It grants no acceptance: every accepted identity is reconstructed below.
    let header: Value = serde_json::from_slice(&fs::read(path).map_err(|e| invalid(e.into()))?)
        .map_err(|e| invalid(e.into()))?;
    identities(&header, path, split, checkpoint)?;
    let report = study_evaluate::verify_completed(directory).map_err(invalid)?;
    identities(&report, path, split, checkpoint)?;
    if report["healthy_complete"] == true {
        return Ok(report);
    }
    let plan: Value = serde_json::from_slice(
        &fs::read(directory.join("plan.json")).map_err(|e| invalid(e.into()))?,
    )
    .map_err(|e| invalid(e.into()))?;
    if plan["evidence_kind"] == "frozen_local_study_checkpoint_evaluation" {
        let root = std::env::current_dir().map_err(|e| invalid(e.into()))?;
        for task in plan["tasks"]
            .as_array()
            .ok_or_else(|| invalid(anyhow::anyhow!("verified task list absent")))?
        {
            let row_path = directory
                .join("observations")
                .join(format!("{}.json", task["id"].as_str().unwrap_or_default()));
            let row: Value =
                serde_json::from_slice(&fs::read(&row_path).map_err(|e| invalid(e.into()))?)
                    .map_err(|e| invalid(e.into()))?;
            let raw: artefact::Artefact = serde_json::from_value(row["raw_generation"].clone())
                .map_err(|e| invalid(e.into()))?;
            let raw_path = raw.verify(&root).map_err(invalid)?;
            let generated: Value =
                serde_json::from_slice(&fs::read(&raw_path).map_err(|e| invalid(e.into()))?)
                    .map_err(|e| invalid(e.into()))?;
            if let Some(reason) = raw_failure(&row, &generated) {
                return Err(failure(
                    &raw_path,
                    reason,
                    "complete generation and grade".into(),
                    generated["generation"]["finish_reason"].clone(),
                    format!(
                        "task {}; observation retained at {}",
                        task["id"],
                        row_path.display()
                    ),
                ));
            }
        }
    }
    Err(failure(
        path,
        Reason::UnhealthyCompletion,
        true.into(),
        report["healthy_complete"].clone(),
        "verified evaluation has unhealthy completion; complete checkpoint remains available",
    ))
}

fn raw_failure(row: &Value, generated: &Value) -> Option<Reason> {
    if generated["generation"]["finish_reason"] == "timeout" {
        Some(Reason::InferenceTimeout)
    } else if !generated["model_error"].is_null()
        || generated["generation"]["complete"] != true
        || generated["generation"]["completion"]
            .as_str()
            .is_none_or(|s| s.trim().is_empty())
    {
        Some(Reason::GenerationIncomplete)
    } else if !row["grading_error"].is_null() {
        Some(Reason::GradingInfrastructureFailure)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn generation_health_never_masquerades_as_split_or_checkpoint_failure() {
        let mut report = json!({"split":"development","checkpoint_id":"actual-adapter","healthy_complete":false});
        let path = Path::new("labelled-evidence-fixture/report.json");
        identities(&report, path, "development", &json!("actual-adapter")).unwrap();
        let raw = json!({"generation":{"finish_reason":"length","complete":false,"completion":"nonsense"},"model_error":null});
        assert_eq!(
            raw_failure(&json!({}), &raw),
            Some(Reason::GenerationIncomplete)
        );
        report["split"] = json!("held_out");
        assert_eq!(
            identities(&report, path, "development", &json!("actual-adapter"))
                .unwrap_err()
                .reason,
            Reason::SplitMismatch
        );
        report["split"] = json!("development");
        assert_eq!(
            identities(&report, path, "development", &json!("wrong-adapter"))
                .unwrap_err()
                .reason,
            Reason::CheckpointMismatch
        );
    }
    #[test]
    fn forged_health_and_absent_bound_evidence_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("report.json");
        fs::write(
            &path,
            br#"{"split":"development","checkpoint_id":"a","healthy_complete":true}"#,
        )
        .unwrap();
        assert_eq!(
            verify(&path, "development", &json!("a"))
                .unwrap_err()
                .reason,
            Reason::EvidenceInvalid
        );
        assert_eq!(
            raw_failure(
                &json!({"grading_error":"injected"}),
                &json!({"generation":{"complete":true,"finish_reason":"stop","completion":"code"}})
            ),
            Some(Reason::GradingInfrastructureFailure)
        );
        assert_eq!(
            raw_failure(
                &json!({}),
                &json!({"generation":{"finish_reason":"timeout"}})
            ),
            Some(Reason::InferenceTimeout)
        );
    }
}
