//! Read-only audit of the original assessment, separate from exploratory success.
use crate::{
    artefact::{self, Artefact},
    tasks,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{fs, path::Path};

mod checks;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EvidenceStatus {
    Verified,
    Observed,
    Incomplete,
    Invalid,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub criterion: String,
    pub status: EvidenceStatus,
    pub finding: String,
    pub next_action: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AssessmentAudit {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub observed_unix_ms: i64,
    pub original_request_satisfied: bool,
    pub requirements: Vec<Requirement>,
    pub inputs: Vec<Artefact>,
    pub exploration: Option<Value>,
    pub limitations: Vec<String>,
}

impl AssessmentAudit {
    pub fn markdown(&self) -> String {
        let mut text = format!(
            "# Original assessment evidence audit\n\nOriginal request satisfied: **{}**. Read-only inspection; no provider calls, model execution, training or sealed evaluation.\n\n| Requirement | Status | Finding | Next action |\n| --- | --- | --- | --- |\n",
            self.original_request_satisfied
        );
        let cell = |s: &str| s.replace('|', "\\|").replace('\n', " ");
        for row in &self.requirements {
            text.push_str(&format!(
                "| {} | {:?} | {} | {} |\n",
                cell(&row.criterion),
                row.status,
                cell(&row.finding),
                cell(&row.next_action)
            ));
        }
        if let Some(exploration) = &self.exploration {
            text.push_str(&format!(
                "\nLatest local exploration (reported observations):\n\n```json\n{}\n```\n",
                serde_json::to_string_pretty(exploration).unwrap_or_default()
            ));
        }
        text.push_str("\nObservation is distinct from independent verification. Missing, malformed, fixture-only and failed observations cannot establish completion. This report assigns no speculative assessment score.\n");
        text
    }
}

struct Inspector<'a> {
    root: &'a Path,
    inputs: Vec<Artefact>,
    rows: Vec<Requirement>,
}
impl Inspector<'_> {
    fn read(&mut self, path: &Path) -> Result<Value> {
        let full = self.root.join(path);
        let metadata =
            fs::symlink_metadata(&full).with_context(|| format!("missing {}", path.display()))?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink() && metadata.len() <= 16 << 20,
            "unsafe or oversized evidence {}",
            path.display()
        );
        let artifact = Artefact::capture(self.root, path)?;
        // Hash before and after parsing so a concurrent writer cannot produce an
        // audit containing one version's values with another version's identity.
        let bytes = fs::read(&full)?;
        ensure!(
            artefact::hash(&bytes) == artifact.sha256,
            "evidence changed during inspection"
        );
        self.inputs.push(artifact);
        serde_json::from_slice(&bytes).with_context(|| format!("malformed {}", path.display()))
    }
    fn record(
        &mut self,
        id: &str,
        criterion: &str,
        status: EvidenceStatus,
        finding: impl Into<String>,
        next: &str,
    ) {
        self.rows.push(Requirement {
            id: id.into(),
            criterion: criterion.into(),
            status,
            finding: finding.into(),
            next_action: next.into(),
        });
    }
    fn checked(&mut self, id: &str, criterion: &str, result: Result<String>, next: &str) {
        let (status, finding) = match result {
            Ok(finding) => (EvidenceStatus::Verified, finding),
            Err(error) => {
                let missing = error.chain().any(|cause| {
                    cause
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound)
                });
                (
                    if missing {
                        EvidenceStatus::Incomplete
                    } else {
                        EvidenceStatus::Invalid
                    },
                    format!("{error:#}"),
                )
            }
        };
        self.record(id, criterion, status, finding, next);
    }
}

/// Runs on a blocking worker. It never creates evidence or changes run state.
pub fn inspect(root: &Path, exploration: &Path) -> Result<AssessmentAudit> {
    let mut inspector = Inspector {
        root,
        inputs: vec![],
        rows: vec![],
    };
    let corpus = tasks::validate_corpus().and_then(|summary| {
        ensure!(summary.tasks >= 40, "fewer than forty tasks");
        Ok(format!(
            "{} executable Rust contracts; train/development/held-out source separation verified",
            summary.tasks
        ))
    });
    inspector.checked(
        "tasks",
        "At least 40 executable source-separated tasks",
        corpus,
        "Repair corpus/source contracts before collecting outputs.",
    );
    checks::original_evidence(&mut inspector);
    let exploration = if root.join(exploration).exists() {
        match inspector.read(exploration) {
            Ok(value) => {
                let mut summary = checks::exploration_summary(&value);
                summary["verified_failed_child_attempts"] =
                    json!(checks::failed_children(&mut inspector, &value));
                Some(summary)
            }
            Err(error) => {
                inspector.record(
                    "latest_exploration",
                    "Latest exploration report",
                    EvidenceStatus::Invalid,
                    format!("{error:#}"),
                    "Retain the failed report and investigate its original receipts.",
                );
                None
            }
        }
    } else {
        None
    };
    inspector.inputs.sort_by(|a, b| a.path.cmp(&b.path));
    inspector
        .inputs
        .dedup_by(|a, b| a.path == b.path && a.sha256 == b.sha256);
    Ok(AssessmentAudit{schema_version:1,evidence_kind:"read_only_original_assessment_evidence_audit".into(),
        observed_unix_ms:crate::ledger::unix_millis(),
        original_request_satisfied:inspector.rows.iter().all(|row|row.status==EvidenceStatus::Verified),
        requirements:inspector.rows,inputs:inspector.inputs,exploration,
        limitations:vec!["Observed historical reports are not a fresh execution certificate; unverified observations remain distinct from reconstructed proof.".into(),
            "Local teacher checkpoints, development gains and API protocol fixtures cannot replace three independent real provider businesses or sealed assessment evidence.".into(),
            "No paid calls, model/cache pruning, candidate execution, training, held-out generation or publication occurs in this audit.".into()]})
}

pub fn write(report: &AssessmentAudit, output: &Path) -> Result<()> {
    ensure!(
        !output.join("audit.json").exists() && !output.join("audit.md").exists(),
        "audit output already exists; use a new output directory to retain observations"
    );
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    // Atomic directory creation also prevents two audits from racing to publish
    // different observations under one explicit output name.
    fs::create_dir(output).context("audit output must be a new directory")?;
    artefact::write_json(&output.join("audit.json"), report)?;
    artefact::atomic_write(&output.join("audit.md"), report.markdown().as_bytes())
}
