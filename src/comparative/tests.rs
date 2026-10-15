//! Labelled fixture evidence verifies control flow without model or container execution.
use super::*;

struct Fixture {
    directory: tempfile::TempDir,
    frozen: FrozenModel,
    plan: Plan,
    task: Task,
    raw: GenerationRecord,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let model = RegistryModel {
            id: "labelled-fixture".into(),
            model_id: "LABELLED_SYNTHETIC_MODEL".into(),
            revision: "1".repeat(40),
            directory: "unavailable-fixture-weights".into(),
            architecture: "fixture".into(),
            role: "fixture".into(),
            licence: "fixture-only".into(),
            developer_id: "fixture-only".into(),
        };
        let assets = model::AssetManifest {
            model_id: model.model_id.clone(),
            revision: model.revision.clone(),
            licence: "fixture-only".into(),
            files: BTreeMap::from([("model.safetensors".into(), "a".repeat(64))]),
        };
        artefact::write_json(&root.join("assets.json"), &assets).unwrap();
        fs::write(root.join("registry.toml"), "# labelled fixture only").unwrap();
        fs::write(
            root.join("preflight.json"),
            "labelled fixture; no actual controls claimed",
        )
        .unwrap();
        let frozen = FrozenModel {
            model,
            assets: Artefact::capture(root, "assets.json").unwrap(),
        };
        let tasks = tasks::corpus()
            .into_iter()
            .filter(|t| t.split == Split::Development)
            .collect::<Vec<_>>();
        let task = tasks[0].clone();
        let plan = Plan {
            schema_version: 1,
            evidence_kind: "labelled_synthetic_plan_fixture".into(),
            frozen_unix_ms: 1,
            registry: Artefact::capture(root, "registry.toml").unwrap(),
            models: vec![frozen.clone()],
            tasks,
            split: ComparisonSplit::Development,
            runtime: serde_json::to_value(RuntimeOptions::default()).unwrap(),
            max_tokens: 64,
            seed: 1729,
            temperature: 0.2,
            top_p: 0.95,
            sandbox: SandboxConfig {
                runtime: "/nonexistent-labelled-fixture-runtime".into(),
                report_dir: "sandbox".into(),
                ..Default::default()
            },
            sandbox_preflight: Artefact::capture(root, "preflight.json").unwrap(),
            sandbox_policy_sha256: "fixture".into(),
            engine_binary_sha256: "fixture".into(),
            promotable: false,
        };
        let execution = serde_json::json!({"engine":"native_rust_candle","requested":plan.runtime,"actual_device":"Cpu","actual_dtype":"F32","cuda_active":false,"cpu_fallback":false,"cuda_feature_compiled":false,"model_id":frozen.model.model_id,"revision":frozen.model.revision,"checkpoint_id":"a".repeat(64)});
        let mut raw = GenerationRecord::new(
            &frozen,
            &task,
            &plan,
            "labelled-plan-hash",
            Ok(Generation {
                completion: "pub fn solve(x:serde_json::Value)->serde_json::Value{x}".into(),
                checkpoint_id: "a".repeat(64),
                finish_reason: "stop".into(),
                input_tokens: 10,
                output_tokens: 20,
                elapsed_ms: 1,
                complete: true,
            }),
            Some(execution),
        );
        raw.evidence_kind = "labelled_synthetic_generation_fixture".into();
        Self {
            directory,
            frozen,
            plan,
            task,
            raw,
        }
    }

    fn save(&self) -> Artefact {
        publish_generation(
            Path::new("raw.generation.json"),
            &self.raw,
            &self.frozen,
            &self.task,
            &self.plan,
            "labelled-plan-hash",
            self.directory.path(),
        )
        .unwrap()
    }

    fn observation(&self, artifact: Artefact) -> Observation {
        Observation {
            model_id: self.raw.model_id.clone(),
            task_id: self.raw.task_id.clone(),
            source_id: self.raw.source_id.clone(),
            specification_sha256: self.raw.specification_sha256.clone(),
            seed: self.raw.seed,
            plan_sha256: self.raw.plan_sha256.clone(),
            request: self.raw.request.clone(),
            generation: self.raw.generation.clone(),
            generation_error: self.raw.generation_error.clone(),
            raw_generation: artifact,
            sandbox_report: None,
            grading_error: Some("labelled grading failure".into()),
            success: false,
        }
    }
}

#[test]
fn exploratory_split_and_plan_cannot_open_sealed_sources() {
    for forbidden in ["held_out", "held-out", "test", "unknown"] {
        assert!(serde_json::from_value::<ComparisonSplit>(serde_json::json!(forbidden)).is_err());
    }
    let mut fixture = Fixture::new();
    validate_plan(&fixture.plan).unwrap();
    fixture.plan.tasks.push(
        tasks::corpus()
            .into_iter()
            .find(|t| t.split == Split::HeldOut)
            .unwrap(),
    );
    assert!(validate_plan(&fixture.plan).is_err());
    fixture.plan.tasks.pop();
    fixture.plan.tasks.push(fixture.task.clone());
    assert!(validate_plan(&fixture.plan).is_err());
}

#[test]
fn every_tensor_file_contributes_to_checkpoint_identity() {
    let mut assets = model::AssetManifest {
        model_id: "fixture".into(),
        revision: "0".repeat(40),
        licence: "fixture".into(),
        files: BTreeMap::from([
            ("model.safetensors".into(), "a".repeat(64)),
            ("model-2.safetensors".into(), "b".repeat(64)),
        ]),
    };
    let original = comparative_checkpoint_id(&assets).unwrap();
    assert_ne!(original, "a".repeat(64));
    assets
        .files
        .insert("model-2.safetensors".into(), "c".repeat(64));
    assert_ne!(original, comparative_checkpoint_id(&assets).unwrap());
    assets.files.remove("model.safetensors");
    assert_eq!(comparative_checkpoint_id(&assets).unwrap(), "c".repeat(64));
    assets.files.clear();
    assert!(comparative_checkpoint_id(&assets).is_err());
}

#[test]
fn cached_output_cannot_change_raw_inference_or_be_marked_success() {
    let fixture = Fixture::new();
    let artifact = fixture.save();
    let mut observation = fixture.observation(artifact);
    verify_observation(
        &observation,
        &fixture.frozen,
        &fixture.task,
        &fixture.plan,
        "labelled-plan-hash",
        fixture.directory.path(),
    )
    .unwrap();
    observation
        .generation
        .as_mut()
        .unwrap()
        .completion
        .push_str(" // altered");
    assert!(
        verify_observation(
            &observation,
            &fixture.frozen,
            &fixture.task,
            &fixture.plan,
            "labelled-plan-hash",
            fixture.directory.path()
        )
        .is_err()
    );
    observation.generation = fixture.raw.generation.clone();
    observation.success = true;
    assert!(
        verify_observation(
            &observation,
            &fixture.frozen,
            &fixture.task,
            &fixture.plan,
            "labelled-plan-hash",
            fixture.directory.path()
        )
        .is_err()
    );
    observation.success = false;
    fs::write(
        fixture.directory.path().join("raw.generation.json"),
        b"modified after capture",
    )
    .unwrap();
    assert!(
        verify_observation(
            &observation,
            &fixture.frozen,
            &fixture.task,
            &fixture.plan,
            "labelled-plan-hash",
            fixture.directory.path()
        )
        .is_err()
    );
}

#[test]
fn raw_cache_rejects_prompt_identity_device_termination_and_token_tampering() {
    let fixture = Fixture::new();
    for case in [
        "seed",
        "prompt",
        "model",
        "source",
        "tensor",
        "dtype",
        "fallback",
        "device",
        "finish",
        "cap",
        "missing_execution",
    ] {
        let mut raw = fixture.raw.clone();
        match case {
            "seed" => raw.seed += 1,
            "prompt" => raw.request.prompt.push('x'),
            "model" => raw.model_id.push('x'),
            "source" => raw.source_id.push('x'),
            "tensor" => raw.generation.as_mut().unwrap().checkpoint_id = "b".repeat(64),
            "dtype" => raw.execution.as_mut().unwrap()["actual_dtype"] = serde_json::json!("F16"),
            "fallback" => raw.execution.as_mut().unwrap()["cpu_fallback"] = serde_json::json!(true),
            "device" => raw.execution.as_mut().unwrap()["cuda_active"] = serde_json::json!(true),
            "finish" => raw.generation.as_mut().unwrap().finish_reason = "length".into(),
            "cap" => raw.generation.as_mut().unwrap().output_tokens = fixture.plan.max_tokens + 1,
            "missing_execution" => raw.execution = None,
            _ => unreachable!(),
        }
        assert!(
            verify_generation(
                &raw,
                &fixture.frozen,
                &fixture.task,
                &fixture.plan,
                "labelled-plan-hash",
                fixture.directory.path()
            )
            .is_err(),
            "case={case}"
        );
    }
}

#[test]
fn missing_truncated_empty_and_error_outcomes_remain_in_the_frozen_denominator() {
    let fixture = Fixture::new();
    let mut observation = fixture.observation(fixture.save());
    observation.grading_error = None;
    observation.generation.as_mut().unwrap().complete = false;
    observation.generation.as_mut().unwrap().finish_reason = "length".into();
    let report = summary(&fixture.plan, &[observation.clone()]).unwrap();
    assert_eq!(report["models"][0]["planned"], 16);
    assert_eq!(report["models"][0]["observed"], 1);
    assert_eq!(report["models"][0]["passed"], 0);
    assert_eq!(report["models"][0]["pass_rate"], 0.0);
    assert_eq!(
        report["models"][0]["infrastructure_or_generation_errors"],
        1
    );
    assert_eq!(report["coverage_complete"], false);
    observation.generation.as_mut().unwrap().complete = true;
    observation.generation.as_mut().unwrap().finish_reason = "stop".into();
    observation.generation.as_mut().unwrap().completion.clear();
    assert_eq!(
        summary(&fixture.plan, &[observation.clone()]).unwrap()["models"][0]["infrastructure_or_generation_errors"],
        1
    );
    observation.generation = None;
    observation.generation_error = Some("labelled load failure".into());
    assert_eq!(
        summary(&fixture.plan, &[observation.clone()]).unwrap()["models"][0]["infrastructure_or_generation_errors"],
        1
    );
    assert!(summary(&fixture.plan, &[observation.clone(), observation]).is_err());
}

#[test]
fn raw_evidence_paths_reject_traversal_and_symlinks_before_publication() {
    let fixture = Fixture::new();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), fixture.directory.path().join("redirected"))
        .unwrap();
    for path in [
        "../outside.generation.json",
        "redirected/raw.generation.json",
    ] {
        assert!(
            publish_generation(
                Path::new(path),
                &fixture.raw,
                &fixture.frozen,
                &fixture.task,
                &fixture.plan,
                "labelled-plan-hash",
                fixture.directory.path()
            )
            .is_err()
        );
        assert!(
            load_generation(
                Path::new(path),
                &fixture.frozen,
                &fixture.task,
                &fixture.plan,
                "labelled-plan-hash",
                fixture.directory.path()
            )
            .is_err()
        );
    }
    assert!(!outside.path().join("raw.generation.json").exists());
}

#[tokio::test]
async fn model_loading_failure_is_durable_and_never_turns_into_success_on_resume() {
    let mut fixture = Fixture::new();
    fixture.raw.generation = None;
    fixture.raw.generation_error = Some("labelled model-loading failure".into());
    fixture.raw.execution = None;
    let artifact = fixture.save();
    let (raw, loaded) = load_generation(
        Path::new("raw.generation.json"),
        &fixture.frozen,
        &fixture.task,
        &fixture.plan,
        "labelled-plan-hash",
        fixture.directory.path(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(artifact.sha256, loaded.sha256);
    let observation = grade_generation(
        raw,
        loaded,
        fixture.task.clone(),
        fixture.plan.sandbox.clone(),
        fixture.directory.path().join("failure.json"),
        fixture.directory.path().to_owned(),
    )
    .await
    .unwrap();
    verify_observation(
        &observation,
        &fixture.frozen,
        &fixture.task,
        &fixture.plan,
        "labelled-plan-hash",
        fixture.directory.path(),
    )
    .unwrap();
    assert!(!observation.success && observation.sandbox_report.is_none());
    assert_eq!(
        summary(&fixture.plan, &[observation]).unwrap()["models"][0]["infrastructure_or_generation_errors"],
        1
    );
}

#[tokio::test]
async fn durable_generation_resumes_grading_without_model_weights_or_a_second_inference() {
    let fixture = Fixture::new();
    let artifact = fixture.save();
    assert!(
        !fixture
            .directory
            .path()
            .join(&fixture.frozen.model.directory)
            .exists()
    );
    let (raw, reloaded) = load_generation(
        Path::new("raw.generation.json"),
        &fixture.frozen,
        &fixture.task,
        &fixture.plan,
        "labelled-plan-hash",
        fixture.directory.path(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(artifact.sha256, reloaded.sha256);
    let path = fixture.directory.path().join("observation.json");
    // The genuine sandbox entry point refuses the absent runtime; candidate code
    // is never executed on the host. The original generation still survives.
    let observation = grade_generation(
        raw,
        reloaded,
        fixture.task.clone(),
        fixture.plan.sandbox.clone(),
        path.clone(),
        fixture.directory.path().to_owned(),
    )
    .await
    .unwrap();
    assert!(observation.grading_error.is_some());
    assert!(!observation.success);
    assert!(path.exists());
    verify_observation(
        &observation,
        &fixture.frozen,
        &fixture.task,
        &fixture.plan,
        "labelled-plan-hash",
        fixture.directory.path(),
    )
    .unwrap();
    assert_eq!(observation.raw_generation.sha256, artifact.sha256);
    assert!(
        publish_generation(
            Path::new("raw.generation.json"),
            &fixture.raw,
            &fixture.frozen,
            &fixture.task,
            &fixture.plan,
            "labelled-plan-hash",
            fixture.directory.path()
        )
        .is_err()
    );
}
