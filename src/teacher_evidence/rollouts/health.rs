//! Current-batch health never inherits success from an earlier bundle.
use super::*;

fn generation_failed(raw: &Raw) -> bool {
    raw.generation_error.is_some()
        || !raw.generation.as_ref().is_some_and(|generation| {
            generation.complete && !generation.completion.trim().is_empty()
        })
}

pub(super) fn batch_report(
    root: &Path,
    records: &[Artefact],
    expected: usize,
    new_samples: usize,
) -> Result<serde_json::Value> {
    let mut students = BTreeSet::new();
    let mut student_generation_failures = 0;
    let mut student_grading_failures = 0;
    let mut feedback_generation_failures = 0;
    let mut feedback_grading_failures = 0;
    let mut feedback_not_attempted = 0;
    let mut successful_records = 0;
    let mut error_records = 0;
    for artifact in records {
        let record: Record = serde_json::from_slice(&fs::read(artifact.verify(root)?)?)?;
        if students.insert(record.student_raw.sha256.clone()) {
            let raw: Raw = serde_json::from_slice(&fs::read(record.student_raw.verify(root)?)?)?;
            student_generation_failures += usize::from(generation_failed(&raw));
            student_grading_failures += usize::from(
                record.student_grade.is_none() || record.unverified_student_grade.is_some(),
            );
        }
        if let Some(raw) = &record.feedback_raw {
            let raw: Raw = serde_json::from_slice(&fs::read(raw.verify(root)?)?)?;
            feedback_generation_failures += usize::from(generation_failed(&raw));
            feedback_grading_failures += usize::from(
                record.feedback_grade.is_none() || record.unverified_feedback_grade.is_some(),
            );
        } else {
            feedback_not_attempted += 1;
        }
        successful_records += usize::from(record.success);
        error_records += usize::from(!record.errors.is_empty());
    }
    let errors = student_generation_failures
        + student_grading_failures
        + feedback_generation_failures
        + feedback_grading_failures;
    let healthy_complete = expected > 0
        && records.len() == expected
        && new_samples > 0
        && successful_records == expected
        && errors == 0
        && error_records == 0;
    Ok(
        serde_json::json!({"schema_version":2,"evidence_kind":"actual_current_batch_checkpoint_bound_rollout_report",
        "healthy_complete":healthy_complete,"coverage_complete":expected > 0 && records.len()==expected,
        "expected_new_records":expected,"new_rollout_records":records.len(),"new_verified_rollout_samples":new_samples,
        "successful_new_records":successful_records,"failed_new_records":records.len()-successful_records,
        "student_generation_failures":student_generation_failures,"student_grading_failures":student_grading_failures,
        "feedback_generation_failures":feedback_generation_failures,"feedback_grading_failures":feedback_grading_failures,
        "feedback_not_attempted":feedback_not_attempted,"error_records":error_records,
        "generation_or_infrastructure_errors":errors,"production_eligible":false,
        "semantics":"Only this frozen batch counts. Inherited passing samples cannot turn absent/error/truncated/unsafe corrective observations into success. Failed rows remain in all record denominators; trusted compiler feedback is not a passing solution."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_empty_new_batch_never_inherits_a_positive_bundle_success() {
        let report = batch_report(Path::new("."), &[], 4, 0).unwrap();
        assert_eq!(report["healthy_complete"], false);
        assert_eq!(report["new_verified_rollout_samples"], 0);
        assert_eq!(report["coverage_complete"], false);
    }
    #[test]
    fn no_output_and_token_limit_termination_are_generation_failures() {
        let mut raw = Raw {
            schema_version: 1,
            evidence_kind: "labelled-test-only".into(),
            plan: Artefact {
                path: "unused".into(),
                sha256: String::new(),
                bytes: 0,
            },
            generation: None,
            generation_error: Some("labelled error".into()),
            execution: None,
        };
        assert!(generation_failed(&raw));
        raw.generation_error = None;
        raw.generation = Some(Generation {
            completion: "partial code".into(),
            checkpoint_id: "labelled".into(),
            finish_reason: "length".into(),
            input_tokens: 1,
            output_tokens: 1,
            elapsed_ms: 1,
            complete: false,
        });
        assert!(generation_failed(&raw));
        raw.generation.as_mut().unwrap().complete = true;
        assert!(!generation_failed(&raw));
        raw.generation.as_mut().unwrap().completion.clear();
        assert!(generation_failed(&raw));
    }
}
