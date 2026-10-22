//! Labelled policy fixtures; these tests never manufacture training checkpoints.
use super::*;

fn artifact() -> Artefact {
    Artefact {
        path: "labelled-fixture-only.json".into(),
        sha256: "a".repeat(64),
        bytes: 0,
    }
}
fn capability(available: bool) -> Capability {
    Capability {
        supported: available,
        available,
        acquirable_offline: false,
        reason: if available {
            "labelled policy fixture"
        } else {
            "not implemented in this labelled fixture"
        }
        .into(),
        artefacts: if available { vec![artifact()] } else { vec![] },
    }
}
fn inputs() -> StudyInputs {
    let teachers = ["alpha", "beta"]
        .into_iter()
        .map(|id| TeacherProfile {
            id: id.into(),
            source_model_id: format!("fixture/{id}"),
            revision: "1".repeat(40),
            tokenizer_sha256: "2".repeat(64),
            token_alignment_verified: true,
            rights_approved: true,
            signals: [SignalMethod::SequenceSft, SignalMethod::SoftTokenKl]
                .into_iter()
                .map(|s| (s, capability(true)))
                .collect(),
            contents: [(ContentMethod::Preference, capability(true))]
                .into_iter()
                .collect(),
            on_policy: capability(false),
            self_distillation: capability(false),
            rollout_data: None,
        })
        .collect();
    StudyInputs {
        schema_version: 1,
        evidence_bundle: artifact(),
        student_base: artifact(),
        student_tokenizer_sha256: "2".repeat(64),
        teachers,
        students: vec![
            StudentProfile {
                architecture: StudentArchitecture::ConventionalQwenLora,
                base_assets: Some(artifact()),
                tokenizer_sha256: Some("2".repeat(64)),
                teacher_token_alignment: BTreeMap::new(),
                training: capability(true),
                exported_checkpoint_reload: capability(true),
            },
            StudentProfile {
                architecture: StudentArchitecture::TernaryBitnet,
                base_assets: None,
                tokenizer_sha256: None,
                teacher_token_alignment: BTreeMap::new(),
                training: capability(false),
                exported_checkpoint_reload: capability(false),
            },
        ],
        resources: ResourceCapacity {
            cpu_threads: 4,
            ram_bytes: 64 << 30,
            gpu_ordinal: Some(0),
            vram_bytes: 8 << 30,
            maximum_parallel_jobs: 4,
        },
        worker_resources: ResourceRequest {
            cpu_threads: 2,
            ram_bytes: 8 << 30,
            vram_bytes: 4 << 30,
        },
        student_resources: vec![],
        teacher_runtime: RuntimeOptions::default(),
        student_runtime: RuntimeOptions::default(),
        evaluation: None,
    }
}
fn config() -> StudyConfig {
    StudyConfig {
        schema_version: 1,
        student_base_model: crate::model::MODEL_ID.into(),
        signals: vec![SignalMethod::SequenceSft, SignalMethod::SoftTokenKl],
        contents: vec![ContentMethod::Preference],
        policies: vec![DataPolicy::Offline, DataPolicy::OnPolicy],
        students: vec![
            StudentArchitecture::ConventionalQwenLora,
            StudentArchitecture::TernaryBitnet,
        ],
        progressive: vec![false, true],
        self_distillation: vec![false, true],
        seeds: vec![1729],
        minimum_teachers: 1,
        maximum_teachers: 2,
        maximum_catalogue_arms: 1000,
        student_training: vec![],
        bitnet_backend: None,
        quick: FidelityConfig {
            steps: 3,
            max_sequence_tokens: 64,
            sequence_policy: SequencePolicy::AssistantWindow,
            rank: 2,
            alpha: 4.,
            learning_rate: 0.0002,
            temperature: 2.,
            dpo_beta: 0.1,
            maximum_trials: 24,
            maximum_total_steps: 72,
            timeout_seconds: 60,
            objective_weights: BTreeMap::new(),
            health_check: None,
        },
        full: FidelityConfig {
            steps: 48,
            max_sequence_tokens: 512,
            sequence_policy: SequencePolicy::RejectOverlength,
            rank: 8,
            alpha: 16.,
            learning_rate: 0.0002,
            temperature: 2.,
            dpo_beta: 0.1,
            maximum_trials: 8,
            maximum_total_steps: 384,
            timeout_seconds: 600,
            objective_weights: BTreeMap::new(),
            health_check: None,
        },
    }
}

#[test]
fn product_keeps_every_configured_arm_and_ordered_assistant_permutation() {
    let config = config();
    config.validate().unwrap();
    let inputs = inputs();
    config::validate_inputs(&inputs).unwrap();
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    assert_eq!(arms.len(), 336);
    assert_eq!(arms.iter().filter(|a| a.applicable).count(), 30);
    assert!(
        arms.iter()
            .any(|a| a.progressive && a.teacher_ids == ["alpha", "beta"])
    );
    assert!(
        arms.iter()
            .any(|a| a.progressive && a.teacher_ids == ["beta", "alpha"])
    );
    assert!(
        !arms
            .iter()
            .any(|a| !a.progressive && a.teacher_ids == ["beta", "alpha"])
    );
    assert!(
        arms.iter()
            .filter(|a| a.student == StudentArchitecture::TernaryBitnet)
            .all(|a| !a.applicable && !a.inapplicable_reasons.is_empty())
    );
    let again = catalogue::enumerate(&config, &inputs).unwrap();
    assert_eq!(
        arms.iter().map(|a| &a.id).collect::<Vec<_>>(),
        again.iter().map(|a| &a.id).collect::<Vec<_>>()
    );
}

#[test]
fn mismatched_tokenisers_block_tensor_methods_without_blocking_verified_sequences() {
    let config = config();
    let mut inputs = inputs();
    inputs.teachers[0].tokenizer_sha256 = "9".repeat(64);
    inputs.teachers[0].token_alignment_verified = false;
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    let relevant = arms
        .iter()
        .filter(|a| {
            a.teacher_ids == ["alpha"]
                && a.data_policy == DataPolicy::Offline
                && !a.progressive
                && !a.self_distillation
                && a.contents.is_empty()
                && a.student == StudentArchitecture::ConventionalQwenLora
        })
        .collect::<Vec<_>>();
    assert!(
        relevant
            .iter()
            .any(|a| a.signals == [SignalMethod::SequenceSft] && a.applicable)
    );
    assert!(
        relevant
            .iter()
            .filter(|a| a.signals.contains(&SignalMethod::SoftTokenKl))
            .all(|a| !a.applicable)
    );
}

#[test]
fn bounded_quick_cover_names_unavailable_families_and_never_selects_them() {
    let config = config();
    let inputs = inputs();
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    let (selected, gaps) =
        catalogue::quick_cover(&arms, config.quick.maximum_trials, &config.quick);
    assert!(!selected.is_empty() && selected.len() <= config.quick.maximum_trials);
    assert!(
        selected
            .iter()
            .all(|id| arms.iter().any(|a| &a.id == id && a.applicable))
    );
    assert!(gaps.iter().any(|s| s.contains("TernaryBitnet")));
    assert!(gaps.iter().any(|s| s == "organiser:self_distillation"));
    assert!(
        !gaps
            .iter()
            .any(|s| s == "signal:SoftTokenKl" || s == "content:Preference")
    );
}

#[test]
fn default_quick_budget_selects_real_mixtures_with_preregistered_seed_pairs() {
    let config = StudyConfig::load(Path::new("config/distillation-study.toml")).unwrap();
    let mut evidence = inputs();
    evidence.teachers.truncate(1);
    evidence.students.truncate(1);
    evidence.teachers[0].signals = config
        .signals
        .iter()
        .map(|signal| (*signal, capability(true)))
        .collect();
    evidence.teachers[0].contents = config
        .contents
        .iter()
        .map(|content| (*content, capability(true)))
        .collect();
    let arms = catalogue::enumerate(&config, &evidence).unwrap();
    let (selected, gaps) = catalogue::quick_cover(&arms, 24, &config.quick);
    assert_eq!(selected.len(), 24);
    assert_eq!(
        selected.iter().collect::<BTreeSet<_>>().len(),
        selected.len()
    );
    let chosen = arms
        .iter()
        .filter(|arm| selected.contains(&arm.id))
        .collect::<Vec<_>>();
    assert!(
        chosen
            .iter()
            .all(|arm| arm.applicable
                && fidelity_inapplicable_reasons(arm, &config.quick).is_empty())
    );
    assert!(chosen.iter().any(|arm| arm.signals.len() == 4));
    let mixture = chosen.iter().find(|arm| arm.signals.len() > 1).unwrap();
    for seed in &config.seeds {
        assert!(chosen.iter().any(|arm| arm.student == mixture.student
            && arm.teacher_ids == mixture.teacher_ids
            && arm.signals == mixture.signals
            && arm.contents == mixture.contents
            && arm.data_policy == mixture.data_policy
            && arm.progressive == mixture.progressive
            && arm.self_distillation == mixture.self_distillation
            && arm.seed == *seed));
    }
    assert_eq!(
        catalogue::quick_cover(&arms, 24, &config.quick),
        (selected.clone(), gaps)
    );
    let mut bounded = config.quick.clone();
    bounded.maximum_total_steps = 9;
    let (limited, _) = catalogue::quick_cover(&arms, 24, &bounded);
    assert!(limited.len() * bounded.steps <= bounded.maximum_total_steps as usize);
    assert!(limited.len() <= bounded.maximum_trials);
}

#[test]
fn atomic_multi_resource_reservations_prevent_aggregate_overcommit() {
    let capacity = inputs().resources;
    let mut resources = scheduler::Resources::new(capacity);
    let request = ResourceRequest {
        cpu_threads: 2,
        ram_bytes: 20 << 30,
        vram_bytes: 4 << 30,
    };
    assert!(resources.reserve(&request));
    assert!(resources.reserve(&request));
    assert!(!resources.reserve(&request));
    resources.release(&request).unwrap();
    assert!(resources.reserve(&request));
    let oversized = ResourceRequest {
        cpu_threads: 1,
        ram_bytes: 100 << 30,
        vram_bytes: 0,
    };
    assert!(!resources.reserve(&oversized));
    resources.release(&request).unwrap();
    resources.release(&request).unwrap();
    assert!(resources.release(&request).is_err());
}

#[test]
fn per_arm_resources_keep_text_cpu_jobs_parallel_to_tensor_acquisition() {
    let mut inputs = inputs();
    let config = config();
    inputs.student_resources = vec![StudentResources {
        student: StudentArchitecture::ConventionalQwenLora,
        quick: ResourceRequest {
            cpu_threads: 2,
            ram_bytes: 8 << 30,
            vram_bytes: 0,
        },
        full: ResourceRequest {
            cpu_threads: 2,
            ram_bytes: 16 << 30,
            vram_bytes: 0,
        },
        tensor_capture_vram_bytes: 6 << 30,
        teacher_targets_cache_only: false,
    }];
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    let text = arms
        .iter()
        .find(|a| a.applicable && a.signals == [SignalMethod::SequenceSft])
        .unwrap();
    let tensor = arms
        .iter()
        .find(|a| a.applicable && a.signals == [SignalMethod::SoftTokenKl])
        .unwrap();
    assert_eq!(resources_for(&inputs, text, Fidelity::Quick).vram_bytes, 0);
    assert_eq!(
        resources_for(&inputs, tensor, Fidelity::Quick).vram_bytes,
        6 << 30
    );
    assert_eq!(
        resources_for(&inputs, text, Fidelity::Full).ram_bytes,
        16 << 30
    );
    inputs.student_resources[0].teacher_targets_cache_only = true;
    assert_eq!(resources_for(&inputs, tensor, Fidelity::Full).vram_bytes, 0);
}

#[test]
fn cap_overflow_and_unproven_invented_content_fail_closed() {
    let mut config = config();
    config.maximum_catalogue_arms = 10;
    assert!(catalogue::enumerate(&config, &inputs()).is_err());
    let mut input = inputs();
    input.teachers[0]
        .contents
        .get_mut(&ContentMethod::Preference)
        .unwrap()
        .acquirable_offline = true;
    assert!(config::validate_inputs(&input).is_err());
}

#[test]
fn committed_default_is_a_finite_bounded_screening_protocol() {
    let config = StudyConfig::load(Path::new("config/distillation-study.toml")).unwrap();
    assert_eq!(config.signals.len(), 4);
    assert_eq!(config.contents.len(), 4);
    assert!(
        config
            .students
            .contains(&StudentArchitecture::TernaryBitnet)
    );
    assert_eq!(config.quick.maximum_total_steps, 72);
    assert_eq!(
        config.quick.sequence_policy,
        SequencePolicy::AssistantWindow
    );
    assert_eq!(
        config.full.sequence_policy,
        SequencePolicy::RejectOverlength
    );
    assert_eq!(
        config.student_training[0].student,
        StudentArchitecture::TernaryBitnet
    );
    assert_eq!(config.student_training[0].quick_learning_rate, 0.005);
}

#[test]
fn every_teacher_content_stratum_requires_a_real_update_within_fidelity() {
    let config = config();
    let inputs = inputs();
    let mut arm = catalogue::enumerate(&config, &inputs).unwrap().remove(0);
    arm.teacher_ids = vec!["alpha".into(), "beta".into()];
    arm.contents = vec![
        ContentMethod::Process,
        ContentMethod::ToolTrajectory,
        ContentMethod::Preference,
    ];
    assert_eq!(minimum_updates_for(&arm), 4);
    assert!(!fidelity_inapplicable_reasons(&arm, &config.quick).is_empty());
    assert!(fidelity_inapplicable_reasons(&arm, &config.full).is_empty());
}

#[test]
fn success_flags_without_real_training_tensors_cannot_complete_an_arm() {
    let input = inputs();
    let config = config();
    let arm = catalogue::enumerate(&config, &input)
        .unwrap()
        .into_iter()
        .find(|a| a.applicable)
        .unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let request = WorkerRequest {
        schema_version: 1,
        study_plan: artifact(),
        arm: arm.clone(),
        fidelity: Fidelity::Quick,
        training: config.quick,
        inputs: input.clone(),
        resources: input.worker_resources,
        teacher_targets_cache_only: false,
        output: tmp.path().to_owned(),
        offline: true,
    };
    let outcome = WorkerOutcome {
        schema_version: 1,
        arm_id: arm.id,
        request_sha256: "fixture-request".into(),
        training_artifact: artifact(),
        checkpoint_artifacts: vec![],
        checkpoint_changed: true,
        checkpoint_reloaded: true,
        completed_steps: 3,
        finite_losses: vec![1., 0.9, 0.8],
        finite_gradient_norms: vec![1., 1., 1.],
        train_evaluation: None,
        development_evaluation: None,
        measured_costs: serde_json::Value::Null,
        limitations: vec!["Labelled invalid policy fixture; no training occurred".into()],
    };
    let path = tmp.path().join("outcome.json");
    artefact::write_json(&path, &outcome).unwrap();
    assert!(scheduler::validate_outcome(&path, &request, "fixture-request").is_err());
    assert!(
        scheduler::validate_outcome(
            &tmp.path().join("missing.json"),
            &request,
            "fixture-request"
        )
        .is_err()
    );
}

fn rollout_inputs(adapted: bool) -> StudyInputs {
    let mut inputs = inputs();
    inputs.students[0].base_assets.as_mut().unwrap().path =
        "labelled-fixture/student/assets.json".into();
    let initial = crate::teacher_evidence::RolloutInitialMetadata {
        student_base: "labelled-fixture/student".into(),
        student_kind: crate::distillation::DistillationStudent::ConventionalQwenLora,
        base_sha256: "a".repeat(64),
        checkpoint_sha256: if adapted {
            "b".repeat(64)
        } else {
            "a".repeat(64)
        },
        checkpoint: adapted.then(artifact),
        rank: adapted.then_some(2),
        alpha: adapted.then_some(4.),
    };
    for teacher in &mut inputs.teachers {
        teacher.on_policy = capability(true);
        teacher.self_distillation = capability(true);
        teacher.rollout_data = Some(RolloutProfile {
            initial: Some(initial.clone()),
            reason: "Labelled compatibility fixture; no rollout was executed".into(),
            complete_sequences: true,
            contents: vec![],
            complete_sequence_preferences: true,
            preference_contents: vec![],
        });
    }
    inputs
}

#[test]
fn adapted_rollouts_require_exact_fidelity_rank_and_alpha_but_base_rollouts_do_not() {
    let config = config();
    let inputs = rollout_inputs(true);
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    let arm = arms
        .iter()
        .find(|arm| arm.applicable && arm.data_policy == DataPolicy::OnPolicy)
        .unwrap();
    assert!(fidelity_inapplicable_reasons(arm, &config.quick).is_empty());
    let reasons = fidelity_inapplicable_reasons(arm, &config.full);
    assert!(reasons.iter().any(|reason| reason.contains("rank 2")));
    assert!(reasons.iter().any(|reason| reason.contains("alpha 4")));
    let (selected, _) = catalogue::quick_cover(&arms, config.quick.maximum_trials, &config.full);
    assert!(selected.iter().all(
        |id| arms.iter().find(|arm| &arm.id == id).unwrap().data_policy == DataPolicy::Offline
    ));
    let base = rollout_inputs(false);
    let arms = catalogue::enumerate(&config, &base).unwrap();
    let arm = arms
        .iter()
        .find(|arm| arm.applicable && arm.data_policy == DataPolicy::OnPolicy)
        .unwrap();
    assert!(fidelity_inapplicable_reasons(arm, &config.full).is_empty());
}

#[test]
fn rollout_architecture_and_initial_identity_are_specific_to_each_student_and_teacher() {
    let config = config();
    let mut inputs = rollout_inputs(true);
    inputs.teachers[0]
        .rollout_data
        .as_mut()
        .unwrap()
        .initial
        .as_mut()
        .unwrap()
        .student_kind = crate::distillation::DistillationStudent::TernaryBitnet;
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    assert!(
        arms.iter()
            .filter(
                |arm| arm.student == StudentArchitecture::ConventionalQwenLora
                    && arm.data_policy == DataPolicy::OnPolicy
                    && arm.teacher_ids == ["alpha"]
            )
            .all(|arm| !arm.applicable
                && arm
                    .inapplicable_reasons
                    .iter()
                    .any(|reason| reason.contains("architecture")))
    );
    let mut inputs = rollout_inputs(true);
    inputs.teachers[1]
        .rollout_data
        .as_mut()
        .unwrap()
        .initial
        .as_mut()
        .unwrap()
        .checkpoint_sha256 = "c".repeat(64);
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    assert!(
        arms.iter()
            .filter(|arm| arm.data_policy == DataPolicy::OnPolicy && arm.teacher_ids.len() == 2)
            .all(|arm| !arm.applicable
                && arm
                    .inapplicable_reasons
                    .iter()
                    .any(|reason| reason.contains("one verified initial student")))
    );
}

#[test]
fn offline_content_and_global_preferences_cannot_manufacture_rollout_intersections() {
    let mut config = config();
    config.contents.push(ContentMethod::Process);
    let mut inputs = rollout_inputs(false);
    for teacher in &mut inputs.teachers {
        teacher
            .contents
            .insert(ContentMethod::Process, capability(true));
    }
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    assert!(
        arms.iter()
            .filter(|arm| arm.data_policy == DataPolicy::OnPolicy
                && arm.contents.contains(&ContentMethod::Process))
            .all(|arm| !arm.applicable)
    );
    assert!(arms.iter().any(|arm| arm.applicable
        && arm.data_policy == DataPolicy::Offline
        && arm.contents == [ContentMethod::Process]));
    for teacher in &mut inputs.teachers {
        teacher.rollout_data.as_mut().unwrap().contents = vec![ContentMethod::Process];
    }
    let arms = catalogue::enumerate(&config, &inputs).unwrap();
    assert!(arms.iter().any(|arm| arm.applicable
        && arm.data_policy == DataPolicy::OnPolicy
        && arm.contents == [ContentMethod::Process]));
    assert!(
        arms.iter()
            .filter(|arm| arm.data_policy == DataPolicy::OnPolicy
                && arm.contents.contains(&ContentMethod::Process)
                && arm.contents.contains(&ContentMethod::Preference))
            .all(|arm| !arm.applicable
                && arm
                    .inapplicable_reasons
                    .iter()
                    .any(|reason| reason.contains("genuine preference pair")))
    );
}

#[test]
fn optional_bitnet_backend_preserves_historical_and_explicit_config_identities() {
    let historical = serde_json::to_value(config()).unwrap();
    assert!(historical.get("bitnet_backend").is_none());
    let restored: StudyConfig = serde_json::from_value(historical.clone()).unwrap();
    assert!(restored.bitnet_backend.is_none());
    assert_eq!(serde_json::to_value(&restored).unwrap(), historical);
    let old_toml = toml::to_string(&restored).unwrap();
    assert!(!old_toml.contains("bitnet_backend"));
    let old_restored: StudyConfig = toml::from_str(&old_toml).unwrap();
    assert_eq!(serde_json::to_value(old_restored).unwrap(), historical);

    let mut explicit = historical;
    explicit["bitnet_backend"] = serde_json::json!({
        "engine_manifest":"reports/labelled-backend-fixture/engine.json",
        "gguf":"reports/labelled-backend-fixture/model.gguf",
        "parity_report":"reports/labelled-backend-fixture/parity.json",
    });
    let restored: StudyConfig = serde_json::from_value(explicit.clone()).unwrap();
    assert_eq!(serde_json::to_value(&restored).unwrap(), explicit);
    let encoded = toml::to_string(&restored).unwrap();
    let decoded: StudyConfig = toml::from_str(&encoded).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), explicit);
}
