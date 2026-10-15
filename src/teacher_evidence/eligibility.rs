//! Training roles are distinct from an unchanged quarantined grading verdict.
use super::*;
use crate::sandbox::{self, SandboxConfig};

pub const CONTENT_ROLE_POLICY: &str = "verified_content_roles_v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContentRole {
    PassingSolution,
    OracleNegative,
    CompilationFeedbackOnly,
    Excluded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContentEligibility {
    pub policy: String,
    pub role: ContentRole,
    pub report: Artefact,
    pub audit: Artefact,
    pub completion_sha256: String,
    pub generation_complete: bool,
    pub original_quarantined: bool,
    pub reason: String,
}
impl ContentEligibility {
    pub fn negative_allowed(&self) -> bool {
        self.policy == CONTENT_ROLE_POLICY && self.role == ContentRole::OracleNegative
    }
    pub fn feedback_allowed(&self) -> bool {
        self.policy == CONTENT_ROLE_POLICY
            && matches!(
                self.role,
                ContentRole::PassingSolution
                    | ContentRole::OracleNegative
                    | ContentRole::CompilationFeedbackOnly
            )
    }
}

#[derive(Deserialize)]
struct AuditFacts {
    trace_complete: bool,
    compile_ok: bool,
    candidate_exit: Option<i32>,
    output: String,
    stderr: String,
    events: Vec<sandbox::AuditEvent>,
    #[serde(default)]
    syscalls_observed: usize,
    failure: Option<String>,
    #[serde(default)]
    output_truncated: bool,
    #[serde(default)]
    stderr_truncated: bool,
}

pub fn classify_content(
    root: &Path,
    artifact: &Artefact,
    config: &SandboxConfig,
    task: &Task,
    completion: &str,
    complete: bool,
) -> Result<ContentEligibility> {
    ensure!(
        artifact.bytes <= 4 << 20,
        "grading report exceeds content-classification bound"
    );
    let report: SandboxReport = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
    sandbox::verify_report(
        &report,
        config,
        task,
        &crate::evaluate::extract_code(completion),
    )?;
    let audit = capture(root, &report.audit_path)?;
    ensure!(
        audit.sha256 == report.audit_sha256 && audit.bytes <= 8 << 20,
        "content-role audit identity or size differs"
    );
    let facts: AuditFacts = serde_json::from_slice(&fs::read(audit.verify(root)?)?)?;
    let role = role(&report, &facts, completion, complete);
    let reason = match role {
        ContentRole::PassingSolution => {
            "Complete solution passes its independently reconstructed controller verdict."
        }
        ContentRole::OracleNegative => {
            "Complete executed answer has an oracle-only mismatch, valid complete JSON responses, exit zero and no audit events, runtime errors or truncation. Its original failure/quarantine remains unchanged; it is a negative only."
        }
        ContentRole::CompilationFeedbackOnly => {
            "Trusted compiler failure with no candidate execution or audit event. Bounded original text may be shown as untrusted correction context; it is never a positive or DPO negative."
        }
        ContentRole::Excluded => {
            "No independently verified permitted content role; retain the original failed/quarantined observation without feedback or training eligibility."
        }
    };
    Ok(ContentEligibility {
        policy: CONTENT_ROLE_POLICY.into(),
        role,
        report: artifact.clone(),
        audit,
        completion_sha256: artefact::hash(completion.as_bytes()),
        generation_complete: complete,
        original_quarantined: report.quarantined,
        reason: reason.into(),
    })
}

fn role(report: &SandboxReport, audit: &AuditFacts, text: &str, complete: bool) -> ContentRole {
    if !complete
        || text.trim().is_empty()
        || text.len() > 32 << 10
        || !audit.events.is_empty()
        || !report.audit_events.is_empty()
        || audit.output_truncated
        || audit.stderr_truncated
        || report.timeout_evidence.is_some()
    {
        return ContentRole::Excluded;
    }
    if report.passed && !report.quarantined && report.trace_complete {
        return ContentRole::PassingSolution;
    }
    let oracle = report.failure.as_ref().filter(|f| {
        f.strip_prefix("host oracle mismatch at case ")
            .and_then(|i| i.parse::<usize>().ok())
            .is_some_and(|i| i < report.cases_total)
    });
    if !report.passed
        && report.quarantined
        && report.trace_complete
        && audit.trace_complete
        && audit.compile_ok
        && audit.syscalls_observed > 0
        && audit.candidate_exit == Some(0)
        && audit.failure.is_none()
        && audit.stderr.trim().is_empty()
        && report.cases_passed < report.cases_total
        && oracle.is_some_and(|f| {
            report.quarantine_reasons.len() == 1 && report.quarantine_reasons[0] == *f
        })
        && audit.output.lines().count() == report.cases_total
        && audit
            .output
            .lines()
            .all(|line| serde_json::from_str::<serde_json::Value>(line).is_ok())
    {
        return ContentRole::OracleNegative;
    }
    if !report.passed
        && report.quarantined
        && !report.trace_complete
        && !audit.trace_complete
        && !audit.compile_ok
        && audit.candidate_exit.is_none()
        && audit.syscalls_observed == 0
        && audit.output.is_empty()
        && audit.failure.as_deref() == Some("compilation failed")
        && report.failure.as_ref().is_some_and(|f| {
            f.starts_with("compilation failed: ")
                && report.quarantine_reasons.len() == 1
                && report.quarantine_reasons[0] == *f
        })
    {
        return ContentRole::CompilationFeedbackOnly;
    }
    ContentRole::Excluded
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixtures() -> (SandboxReport, AuditFacts) {
        let config = SandboxConfig::default();
        let report = SandboxReport {
            run_id: "unit-role-fixture".into(),
            task_id: "unit-only".into(),
            passed: false,
            quarantined: true,
            quarantine_reasons: vec!["host oracle mismatch at case 0".into()],
            cases_total: 2,
            cases_passed: 1,
            trace_complete: true,
            policy_sha256: String::new(),
            image_id: String::new(),
            oracle_sha256: String::new(),
            code_sha256: String::new(),
            stdout_sha256: String::new(),
            sandbox_config: config,
            audit_events: vec![],
            audit_path: "unit-only".into(),
            audit_sha256: String::new(),
            failure: Some("host oracle mismatch at case 0".into()),
            duration_ms: 1,
            timeout_evidence: None,
        };
        let audit = AuditFacts {
            trace_complete: true,
            compile_ok: true,
            candidate_exit: Some(0),
            output: "0\n1\n".into(),
            stderr: String::new(),
            events: vec![],
            syscalls_observed: 32,
            failure: None,
            output_truncated: false,
            stderr_truncated: false,
        };
        (report, audit)
    }
    #[test]
    fn an_oracle_failure_can_only_be_negative_while_security_or_schema_failures_are_excluded() {
        let (r, mut a) = fixtures();
        assert_eq!(
            role(&r, &a, "unit-only code", true),
            ContentRole::OracleNegative
        );
        assert!(r.quarantined);
        assert!(!r.passed);
        a.events.push(sandbox::AuditEvent {
            syscall: 41,
            category: "network".into(),
            target: "unit".into(),
            result: "denied".into(),
        });
        assert_eq!(role(&r, &a, "unit-only code", true), ContentRole::Excluded);
        a.events.clear();
        a.output = "0\nnot JSON\n".into();
        assert_eq!(role(&r, &a, "unit-only code", true), ContentRole::Excluded);
        a.output = "0\n1\n".into();
        a.stderr = "panic: index out of bounds".into();
        assert_eq!(role(&r, &a, "unit-only code", true), ContentRole::Excluded);
        a.stderr.clear();
        assert_eq!(role(&r, &a, "unit-only code", false), ContentRole::Excluded);
    }
    #[test]
    fn a_compiler_failure_is_feedback_only_and_never_a_negative_target() {
        let (mut r, mut a) = fixtures();
        r.trace_complete = false;
        r.cases_passed = 0;
        r.failure = Some("compilation failed: unit type error".into());
        r.quarantine_reasons = vec![r.failure.clone().unwrap()];
        a.trace_complete = false;
        a.compile_ok = false;
        a.candidate_exit = None;
        a.syscalls_observed = 0;
        a.output.clear();
        a.failure = Some("compilation failed".into());
        assert_eq!(
            role(&r, &a, "unit-only code", true),
            ContentRole::CompilationFeedbackOnly
        );
        r.quarantine_reasons
            .push("prohibited filesystem attempt".into());
        assert_eq!(role(&r, &a, "unit-only code", true), ContentRole::Excluded);
    }
}
