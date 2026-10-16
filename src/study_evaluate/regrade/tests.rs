//! Explicitly synthetic failure/receipt fixtures; no model or container is run.
use super::*;

fn placeholder(label: &str) -> Artefact {
    Artefact {
        path: format!("labelled-unit-fixture/{label}").into(),
        sha256: artefact::hash(label.as_bytes()),
        bytes: 0,
    }
}

fn original(label: &str, split: StudySplit, checkpoint: &str) -> OriginalEvaluation {
    let mut plan = super::super::tests::plan();
    plan.split = split;
    plan.tasks = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == split.corpus_split())
        .collect();
    plan.checkpoint.checkpoint_id = artefact::hash(checkpoint.as_bytes());
    let directory = PathBuf::from(format!("labelled-unit-fixture/{label}"));
    let mut artifact = placeholder(label);
    artifact.path = directory.join("plan.json");
    let rows = plan
        .tasks
        .iter()
        .map(|task| ImportedGeneration {
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            original_raw: placeholder(&task.id),
            original_plan_sha256: artifact.sha256.clone(),
            request_sha256: "unused cohort-only fixture".into(),
            extracted_code_sha256: None,
        })
        .collect();
    OriginalEvaluation {
        directory,
        plan: artifact,
        generation_plan: plan,
        execution_receipts: Vec::new(),
        rows,
    }
}

fn six_originals() -> Vec<OriginalEvaluation> {
    [StudySplit::Development, StudySplit::Train]
        .into_iter()
        .flat_map(|split| {
            ["base", "sft", "composed"]
                .map(|checkpoint| original(&format!("{split:?}-{checkpoint}"), split, checkpoint))
        })
        .collect()
}

#[test]
fn full_declared_cohort_is_required_before_any_grading() {
    let originals = six_originals();
    validate_cohort(&originals, 192).unwrap();
    assert!(validate_cohort(&originals, 191).is_err());
    assert!(validate_cohort(&originals[..5], 192).is_err());
    assert!(validate_cohort(&[], 192).is_err());
}

#[test]
fn cohort_rejects_duplicate_directory_plan_and_checkpoint_split() {
    let originals = six_originals();
    for field in 0..3 {
        let mut changed = originals.clone();
        match field {
            0 => changed[1].directory = changed[0].directory.clone(),
            1 => changed[1].plan.sha256 = changed[0].plan.sha256.clone(),
            _ => {
                changed[1].generation_plan.checkpoint =
                    changed[0].generation_plan.checkpoint.clone()
            }
        }
        assert!(validate_cohort(&changed, 192).is_err());
    }
}

#[test]
fn cohort_rejects_reordered_missing_and_duplicate_rows() {
    let originals = six_originals();
    for field in 0..3 {
        let mut changed = originals.clone();
        match field {
            0 => changed[0].rows.swap(0, 1),
            1 => {
                changed[0].rows.pop();
            }
            _ => changed[0].rows[1] = changed[0].rows[0].clone(),
        }
        assert!(validate_cohort(&changed, 192).is_err());
    }
}

#[test]
fn cohort_freezes_original_generation_engine_runtime_and_base() {
    let originals = six_originals();
    for field in 0..7 {
        let mut changed = originals.clone();
        let plan = &mut changed[1].generation_plan;
        match field {
            0 => plan.engine.sha256 = artefact::hash(b"different engine"),
            1 => plan.seed += 1,
            2 => plan.runtime = serde_json::json!({"changed":true}),
            3 => plan.max_new_tokens += 1,
            4 => plan.checkpoint.base_assets.sha256 = artefact::hash(b"different base"),
            5 => plan.checkpoint.tokenizer.sha256 = artefact::hash(b"different tokeniser"),
            _ => plan.checkpoint.checkpoint_id = "invalid".into(),
        }
        assert!(validate_cohort(&changed, 192).is_err());
    }
}

/// The fixture records only a synthetic error or synthetic raw text. It cannot
/// pass checkpoint validation or become training/promotion evidence.
struct RawFixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
    plan: RegradePlan,
    artifact: Artefact,
    output: PathBuf,
}
impl RawFixture {
    fn new(kind: &str, in_current_root: bool) -> Self {
        let cwd = std::env::current_dir().unwrap().canonicalize().unwrap();
        let temporary = if in_current_root {
            tempfile::tempdir_in(&cwd).unwrap()
        } else {
            tempfile::tempdir().unwrap()
        };
        let root = if in_current_root {
            cwd
        } else {
            temporary.path().canonicalize().unwrap()
        };
        let directory = temporary
            .path()
            .strip_prefix(&root)
            .unwrap()
            .join("original");
        let output = temporary
            .path()
            .strip_prefix(&root)
            .unwrap()
            .join("derived");
        let mut original = original("raw", StudySplit::Train, "synthetic-checkpoint");
        original.directory = directory.clone();
        original.generation_plan.tasks.truncate(1);
        artefact::write_json(
            &root.join(directory.join("plan.json")),
            &original.generation_plan,
        )
        .unwrap();
        original.plan = capture(&root, &directory.join("plan.json")).unwrap();
        let task = &original.generation_plan.tasks[0];
        let generation = match kind {
            "error" => None,
            "empty" => Some((true, "   ", "stop")),
            "length" => Some((false, "synthetic truncated text", "length")),
            "timeout" => Some((false, "synthetic timed-out text", "timeout")),
            _ => Some((true, "synthetic complete text; no model was run", "stop")),
        }
        .map(|(complete, text, reason)| Generation {
            completion: text.into(),
            checkpoint_id: original.generation_plan.checkpoint.checkpoint_id.clone(),
            finish_reason: reason.into(),
            input_tokens: 1,
            output_tokens: 1,
            elapsed_ms: 1,
            complete,
        });
        let raw = RawGeneration {
            schema_version: 1,
            plan_sha256: original.plan.sha256.clone(),
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            request: original.generation_plan.request(task),
            model_error: generation
                .is_none()
                .then(|| "Injected unit error; no model was run".into()),
            generation,
        };
        let path = directory
            .join("generations")
            .join(format!("{}.json", task.id));
        artefact::write_json(&root.join(&path), &raw).unwrap();
        original.rows = vec![ImportedGeneration {
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            original_raw: capture(&root, &path).unwrap(),
            original_plan_sha256: original.plan.sha256.clone(),
            request_sha256: artefact::hash(&serde_json::to_vec(&raw.request).unwrap()),
            extracted_code_sha256: raw
                .generation
                .as_ref()
                .filter(|_| usable(&raw))
                .map(|g| tasks::hash(crate::evaluate::extract_code(&g.completion))),
        }];
        let grading = GradingProtocol {
            engine: placeholder("grading-engine"),
            sandbox: SandboxConfig {
                runtime: "unreachable-unit-test-runtime".into(),
                expected_image_id: "labelled-unit-fixture-image; never launched".into(),
                ..Default::default()
            },
            policy_sha256: "unused observation-only fixture".into(),
            image_id: "labelled-unit-fixture-image; never launched".into(),
            preflight: placeholder("preflight"),
            extractor_sha256: extractor_hash(),
        };
        let plan = RegradePlan {
            schema_version: 1,
            evidence_kind: PLAN_KIND.into(),
            batch_plan: placeholder("batch-plan"),
            original,
            grading,
            maximum_concurrent_graders: 1,
            production_eligible: false,
        };
        artefact::write_json(&root.join(output.join("plan.json")), &plan).unwrap();
        let artifact = capture(&root, &output.join("plan.json")).unwrap();
        Self {
            _temporary: temporary,
            root,
            plan,
            artifact,
            output,
        }
    }
    fn row(&self) -> RegradedObservation {
        let imported = &self.plan.original.rows[0];
        RegradedObservation {
            schema_version: 1,
            regrade_plan_sha256: self.artifact.sha256.clone(),
            task_id: imported.task_id.clone(),
            source_id: imported.source_id.clone(),
            original_plan: self.plan.original.plan.clone(),
            original_raw: imported.original_raw.clone(),
            outcome: Outcome::OriginalGenerationFailure,
            success: false,
        }
    }
    fn verify(&self, row: &RegradedObservation) -> Result<RawGeneration> {
        verify::observation(
            &self.root,
            &self.plan,
            &self.artifact,
            &self.plan.original.generation_plan.tasks[0],
            &self.plan.original.rows[0],
            row,
        )
    }
}

#[test]
fn original_failures_empty_and_partial_outputs_cannot_become_passes_or_grades() {
    for kind in ["error", "empty", "length", "timeout"] {
        let fixture = RawFixture::new(kind, false);
        let mut row = fixture.row();
        fixture.verify(&row).unwrap();
        row.success = true;
        assert!(fixture.verify(&row).is_err());
        row.success = false;
        row.outcome = Outcome::InfrastructureFailure {
            prepared: None,
            error: "invented grading of original failure".into(),
        };
        assert!(fixture.verify(&row).is_err());
        row.outcome = Outcome::Graded {
            prepared: placeholder("prepared"),
            report: placeholder("report"),
            audit: placeholder("audit"),
        };
        assert!(fixture.verify(&row).is_err());
    }
}

#[test]
fn complete_output_needs_a_real_grade_or_an_explicit_infrastructure_failure() {
    let fixture = RawFixture::new("complete", false);
    let mut row = fixture.row();
    assert!(fixture.verify(&row).is_err());
    row.outcome = Outcome::InfrastructureFailure {
        prepared: None,
        error: "Injected pre-launch infrastructure failure; no runtime was called".into(),
    };
    fixture.verify(&row).unwrap();
    row.success = true;
    assert!(fixture.verify(&row).is_err());
    row.success = false;
    row.outcome = Outcome::InfrastructureFailure {
        prepared: None,
        error: String::new(),
    };
    assert!(fixture.verify(&row).is_err());
}

#[test]
fn immutable_raw_request_checkpoint_and_plan_pointers_are_rechecked() {
    for field in 0..5 {
        let mut fixture = RawFixture::new("complete", false);
        let task = &fixture.plan.original.generation_plan.tasks[0];
        let imported = fixture.plan.original.rows[0].clone();
        let path = fixture.root.join(&imported.original_raw.path);
        let mut raw: RawGeneration = read(&path).unwrap();
        match field {
            0 => raw.request.seed += 1,
            1 => raw.plan_sha256 = artefact::hash(b"another generation plan"),
            2 => {
                raw.generation.as_mut().unwrap().checkpoint_id =
                    artefact::hash(b"another checkpoint")
            }
            3 => raw.generation.as_mut().unwrap().completion.push('!'),
            _ => raw.request.max_new_tokens += 1,
        }
        artefact::write_json(&path, &raw).unwrap();
        assert!(original_raw(&fixture.root, &fixture.plan.original, &imported, task).is_err());
        // Even recapturing altered bytes cannot repair the original request,
        // checkpoint or extraction identity saved in the imported wrapper.
        fixture.plan.original.rows[0].original_raw =
            capture(&fixture.root, &imported.original_raw.path).unwrap();
        assert!(
            original_raw(
                &fixture.root,
                &fixture.plan.original,
                &fixture.plan.original.rows[0],
                &fixture.plan.original.generation_plan.tasks[0]
            )
            .is_err()
        );
    }
}

#[test]
fn original_plan_bytes_cannot_be_changed_after_import() {
    let fixture = RawFixture::new("error", false);
    let path = fixture.root.join(&fixture.plan.original.plan.path);
    artefact::atomic_write(&path, b"changed original plan fixture").unwrap();
    assert!(import::verify(&fixture.root, &fixture.plan.original).is_err());
}

#[test]
fn output_overlap_and_symlink_ancestors_are_rejected_without_writes() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let original = root.join("original");
    fs::create_dir(&original).unwrap();
    fs::write(original.join("retained"), b"original fixture bytes").unwrap();
    let originals = vec![original.clone()];
    for path in ["original", "original/derived", "../escape", ""] {
        assert!(run::output_path(&root, Path::new(path), &originals).is_err());
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&original, root.join("alias")).unwrap();
        assert!(run::output_path(&root, Path::new("alias/new/derived"), &originals).is_err());
        assert!(!original.join("new").exists());
    }
    assert_eq!(
        run::output_path(&root, Path::new("fresh/derived"), &originals).unwrap(),
        PathBuf::from("fresh/derived")
    );
    assert_eq!(
        fs::read(original.join("retained")).unwrap(),
        b"original fixture bytes"
    );
    assert_eq!(fs::read_dir(&original).unwrap().count(), 1);
}

#[tokio::test]
async fn original_error_rows_finish_without_a_model_factory_or_sandbox_launch() {
    for kind in ["error", "empty", "length", "timeout"] {
        let fixture = RawFixture::new(kind, true);
        run::grade(
            fixture.plan.clone(),
            fixture.artifact.clone(),
            0,
            fixture.output.clone(),
        )
        .await
        .unwrap();
        let path = fixture.root.join(
            fixture
                .output
                .join("observations")
                .join(format!("{}.json", fixture.plan.original.rows[0].task_id)),
        );
        let before = fs::read(&path).unwrap();
        let row: RegradedObservation = read(&path).unwrap();
        fixture.verify(&row).unwrap();
        assert!(!row.success);
        assert!(!fixture.root.join(fixture.output.join("prepared")).exists());
        run::grade(
            fixture.plan.clone(),
            fixture.artifact.clone(),
            0,
            fixture.output.clone(),
        )
        .await
        .unwrap();
        assert_eq!(fs::read(path).unwrap(), before);
    }
}

#[test]
fn existing_prepared_receipt_is_adopted_exactly_and_wrong_identity_is_rejected() {
    let mut fixture = RawFixture::new("complete", true);
    let task = fixture.plan.original.generation_plan.tasks[0].clone();
    let imported = fixture.plan.original.rows[0].clone();
    let code = "synthetic candidate receipt only";
    let (first, receipt) = run::prepare_or_adopt(
        &fixture.root,
        &fixture.plan,
        &fixture.artifact,
        &task,
        &imported,
        &fixture.output,
        code,
    )
    .unwrap();
    let bytes = fs::read(fixture.root.join(&receipt.path)).unwrap();
    let (second, again) = run::prepare_or_adopt(
        &fixture.root,
        &fixture.plan,
        &fixture.artifact,
        &task,
        &imported,
        &fixture.output,
        code,
    )
    .unwrap();
    assert_eq!(first.run_id, second.run_id);
    assert_eq!(first.ownership_nonce, second.ownership_nonce);
    assert_eq!(receipt.sha256, again.sha256);
    assert!(
        run::prepare_or_adopt(
            &fixture.root,
            &fixture.plan,
            &fixture.artifact,
            &task,
            &imported,
            &fixture.output,
            "different code"
        )
        .is_err()
    );
    let mut changed_task = task.clone();
    changed_task.public_examples[0].input = serde_json::json!({"changed_public_input":true});
    assert!(
        run::prepare_or_adopt(
            &fixture.root,
            &fixture.plan,
            &fixture.artifact,
            &changed_task,
            &imported,
            &fixture.output,
            code
        )
        .is_err()
    );
    let mut changed_plan = fixture.artifact.clone();
    changed_plan.sha256 = artefact::hash(b"another derived protocol");
    assert!(
        run::prepare_or_adopt(
            &fixture.root,
            &fixture.plan,
            &changed_plan,
            &task,
            &imported,
            &fixture.output,
            code
        )
        .is_err()
    );
    fixture.plan.grading.sandbox.memory_mb += 1;
    assert!(
        run::prepare_or_adopt(
            &fixture.root,
            &fixture.plan,
            &fixture.artifact,
            &task,
            &imported,
            &fixture.output,
            code
        )
        .is_err()
    );
    assert_eq!(fs::read(fixture.root.join(&receipt.path)).unwrap(), bytes);
    assert!(!first.report_path.exists());
}

#[tokio::test]
async fn interrupted_prepared_launch_never_relaunches_or_invents_a_report() {
    let fixture = RawFixture::new("complete", true);
    let task = &fixture.plan.original.generation_plan.tasks[0];
    let imported = &fixture.plan.original.rows[0];
    let (prepared, _) = run::prepare_or_adopt(
        &fixture.root,
        &fixture.plan,
        &fixture.artifact,
        task,
        imported,
        &fixture.output,
        "synthetic receipt-only candidate",
    )
    .unwrap();
    let launch = prepared
        .config
        .report_dir
        .join(format!("{}.launch.json", prepared.run_id));
    fs::write(
        &launch,
        b"labelled interrupted launch fixture; no runtime was called",
    )
    .unwrap();
    let error = sandbox::execute_prepared(&prepared).await.unwrap_err();
    assert!(format!("{error:#}").contains("already launched without a complete verified report"));
    assert!(!prepared.report_path.exists());
    assert_eq!(
        fs::read(launch).unwrap(),
        b"labelled interrupted launch fixture; no runtime was called"
    );
}

#[test]
fn comparisons_require_identical_effective_grading_identity() {
    let fixture = RawFixture::new("error", false);
    let baseline = verify::identity(&fixture.plan.grading);
    compare::matched_grading(&baseline, &baseline).unwrap();
    for field in 0..6 {
        let mut changed = baseline.clone();
        match field {
            0 => changed.kind = "original_local_study_grading".into(),
            1 => changed.engine_sha256 = artefact::hash(b"other grading engine"),
            2 => changed.controller_version += 1,
            3 => changed.policy_sha256 = artefact::hash(b"other policy"),
            4 => changed.image_id = "other image".into(),
            _ => changed.extractor_sha256 = Some(artefact::hash(b"other extractor")),
        };
        assert!(compare::matched_grading(&baseline, &changed).is_err());
    }
}
