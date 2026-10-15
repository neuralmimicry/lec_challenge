//! Reuse only task/seed/checkpoint-bound raw results and trusted graded outcomes.
use super::*;

pub(super) struct Intent<'a> {
    pub checkpoint: &'a CheckpointEvidence,
    pub config_sha256: &'a str,
    pub corpus_sha256: &'a str,
    pub split: Split,
    pub seeds: &'a [u64],
    pub inference: &'a InferencePlan,
    pub sandbox: &'a SandboxConfig,
}

pub(super) fn validate(prereg: &Preregistration, intent: &Intent<'_>) -> Result<()> {
    ensure!(
        prereg.schema_version == 1
            && prereg.frozen_unix_ms > 0
            && same_serialised(&prereg.checkpoint, intent.checkpoint)?
            && prereg.gate_config_sha256 == intent.config_sha256
            && prereg.corpus_sha256 == intent.corpus_sha256
            && prereg.split == intent.split
            && prereg.evaluation_seeds == intent.seeds
            && prereg.inference == *intent.inference
            && same_serialised(&prereg.sandbox_config, intent.sandbox)?
            && prereg.sandbox_policy_sha256 == sandbox::policy_sha256(intent.sandbox)?
            && !prereg.sandbox_image_id.is_empty()
            && prereg.sandbox_image_id == intent.sandbox.expected_image_id,
        "evaluation resume changed checkpoint, tasks, seeds, decoding, gate or sandbox; use an explicit fresh output"
    );
    prereg.inference.validate()
}

pub(super) fn generation(
    path: &Path,
    prereg: &Preregistration,
    task: &tasks::Task,
    seed: u64,
) -> Result<Option<GenerationEvidence>> {
    if !path.exists() {
        return Ok(None);
    }
    let record: GenerationEvidence = serde_json::from_slice(&std::fs::read(path)?)?;
    verify_generation(
        &record,
        &prereg.inference,
        task.prompt(),
        seed,
        &prereg.checkpoint.checkpoint_id,
    )?;
    Ok(Some(record))
}

fn insert(
    rows: &mut BTreeMap<(String, u64), Outcome>,
    row: Outcome,
    prereg: &Preregistration,
    corpus: &[tasks::Task],
    root: &Path,
) -> Result<()> {
    ensure!(
        prereg.evaluation_seeds.contains(&row.seed),
        "cached evaluation contains an unplanned decoding seed"
    );
    verify_cached_outcome(&row, &prereg.checkpoint, prereg, corpus, root)?;
    let key = (row.task_id.clone(), row.seed);
    if let Some(existing) = rows.get(&key) {
        ensure!(
            same_serialised(existing, &row)?,
            "conflicting partial and per-task cached outcomes"
        );
    } else {
        rows.insert(key, row);
    }
    Ok(())
}

pub(super) fn outcomes(
    output: &Path,
    prereg: &Preregistration,
    corpus: &[tasks::Task],
    root: &Path,
) -> Result<BTreeMap<(String, u64), Outcome>> {
    let mut rows = BTreeMap::new();
    let partial = output.join("partial.json");
    if partial.exists() {
        let cached: Vec<Outcome> = serde_json::from_slice(&std::fs::read(partial)?)?;
        let mut duplicate = BTreeSet::new();
        for row in cached {
            ensure!(
                duplicate.insert((row.task_id.clone(), row.seed)),
                "duplicate cached partial task/seed"
            );
            insert(&mut rows, row, prereg, corpus, root)?;
        }
    }
    for entry in std::fs::read_dir(output)? {
        let entry = entry?;
        if entry
            .file_name()
            .to_string_lossy()
            .ends_with("-outcome.json")
        {
            ensure!(
                entry.file_type()?.is_file(),
                "cached outcome must be a regular file"
            );
            let row: Outcome = serde_json::from_slice(&std::fs::read(entry.path())?)?;
            ensure!(
                entry.file_name().to_string_lossy()
                    == format!("{}-{}-outcome.json", row.task_id, row.seed),
                "cached outcome filename disagrees with task/seed"
            );
            insert(&mut rows, row, prereg, corpus, root)?;
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn placeholder() -> Artefact {
        Artefact {
            path: "labelled-unit-fixture-not-checkpoint.json".into(),
            sha256: "a".repeat(64),
            bytes: 0,
        }
    }
    fn fixture() -> (Preregistration, tasks::Task) {
        let task = tasks::corpus()
            .into_iter()
            .find(|t| t.split == Split::Train)
            .unwrap();
        let sandbox = SandboxConfig {
            expected_image_id: "labelled-fixture-image".into(),
            ..Default::default()
        };
        let prereg = Preregistration {
            schema_version: 1,
            frozen_unix_ms: 1,
            checkpoint: CheckpointEvidence {
                checkpoint_id: "labelled-fixture-checkpoint".into(),
                tensor: placeholder(),
                metadata: placeholder(),
                base_assets: placeholder(),
                base_tensor_sha256: "b".repeat(64),
                tokenizer_sha256: "c".repeat(64),
            },
            gate_config_sha256: "d".repeat(64),
            corpus_sha256: "e".repeat(64),
            split: Split::Train,
            evaluation_seeds: vec![17],
            inference: InferencePlan {
                max_new_tokens: 32,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: 120,
            },
            sandbox_policy_sha256: sandbox::policy_sha256(&sandbox).unwrap(),
            sandbox_image_id: sandbox.expected_image_id.clone(),
            sandbox_config: sandbox,
            sandbox_preflight: placeholder(),
        };
        (prereg, task)
    }
    fn intent(prereg: &Preregistration) -> Intent<'_> {
        Intent {
            checkpoint: &prereg.checkpoint,
            config_sha256: &prereg.gate_config_sha256,
            corpus_sha256: &prereg.corpus_sha256,
            split: prereg.split,
            seeds: &prereg.evaluation_seeds,
            inference: &prereg.inference,
            sandbox: &prereg.sandbox_config,
        }
    }
    fn error_record(prereg: &Preregistration, task: &tasks::Task) -> GenerationEvidence {
        let request = prereg.inference.request(task.prompt(), 17);
        GenerationEvidence {
            schema_version: 1,
            checkpoint_id: prereg.checkpoint.checkpoint_id.clone(),
            request_sha256: artefact::hash(&serde_json::to_vec(&request).unwrap()),
            request,
            generation: None,
            model_error: Some("Injected unit-test failure; no model was invoked".into()),
        }
    }
    #[test]
    fn resumption_rejects_changed_checkpoint_seed_split_caps_and_sandbox() {
        let (original, _) = fixture();
        validate(&original, &intent(&original)).unwrap();
        for field in ["checkpoint", "seed", "split", "cap", "sandbox", "corpus"] {
            let mut changed = original.clone();
            match field {
                "checkpoint" => changed.checkpoint.checkpoint_id = "different".into(),
                "seed" => changed.evaluation_seeds.push(18),
                "split" => changed.split = Split::Development,
                "cap" => changed.inference.max_new_tokens += 1,
                "sandbox" => changed.sandbox_config.memory_mb += 64,
                "corpus" => changed.corpus_sha256 = "different".into(),
                _ => unreachable!(),
            }
            assert!(
                validate(&changed, &intent(&original)).is_err(),
                "accepted changed {field}"
            );
        }
    }
    #[test]
    fn raw_error_survives_restart_and_cannot_be_reused_for_another_seed() {
        let (prereg, task) = fixture();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("raw.json");
        artefact::write_json(&path, &error_record(&prereg, &task)).unwrap();
        assert!(
            generation(&path, &prereg, &task, 17)
                .unwrap()
                .unwrap()
                .model_error
                .is_some()
        );
        assert!(generation(&path, &prereg, &task, 18).is_err());
        assert!(
            generation(&tmp.path().join("missing.json"), &prereg, &task, 17)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn atomic_row_recovers_without_partial_and_forged_success_is_rejected() {
        let (prereg, task) = fixture();
        let tmp = tempfile::tempdir().unwrap();
        let raw_name = format!("{}-17-generation.json", task.id);
        let raw_path = tmp.path().join(&raw_name);
        artefact::write_json(&raw_path, &error_record(&prereg, &task)).unwrap();
        let mut row = Outcome {
            task_id: task.id.clone(),
            source_id: task.source_id.clone(),
            split: Split::Train,
            seed: 17,
            specification_sha256: task.specification_sha256.clone(),
            oracle_sha256: task.oracle_sha256.clone(),
            checkpoint_id: prereg.checkpoint.checkpoint_id.clone(),
            success: false,
            generation_complete: false,
            generation: Artefact::capture(tmp.path(), raw_name).unwrap(),
            sandbox_report: None,
        };
        let row_path = tmp.path().join(format!("{}-17-outcome.json", task.id));
        artefact::write_json(&row_path, &row).unwrap();
        assert_eq!(
            outcomes(tmp.path(), &prereg, std::slice::from_ref(&task), tmp.path())
                .unwrap()
                .len(),
            1
        );
        artefact::write_json(&tmp.path().join("partial.json"), &vec![row.clone()]).unwrap();
        assert_eq!(
            outcomes(tmp.path(), &prereg, std::slice::from_ref(&task), tmp.path())
                .unwrap()
                .len(),
            1
        );
        row.success = true;
        artefact::write_json(&row_path, &row).unwrap();
        assert!(outcomes(tmp.path(), &prereg, std::slice::from_ref(&task), tmp.path()).is_err());
    }
}
