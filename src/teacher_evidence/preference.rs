//! Preference conditioning is explicit and shared by both responses/reference.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreferencePromptTransform {
    pub policy: String,
    pub task_id: String,
    pub task_specification_sha256: String,
    pub common_prompt_sha256: String,
    pub chosen_generation_prompt_sha256: String,
    pub rejected_generation_prompt_sha256: String,
    pub semantics: String,
}

pub fn preference_conditioning(
    task: &Task,
    chosen_actual_prompt: &str,
    rejected_actual_prompt: &str,
) -> (String, PreferencePromptTransform) {
    let prompt = task.prompt();
    let transform=PreferencePromptTransform{policy:"same_original_task_conditioning_v1".into(),task_id:task.id.clone(),task_specification_sha256:task.specification_sha256.clone(),common_prompt_sha256:artefact::hash(prompt.as_bytes()),chosen_generation_prompt_sha256:artefact::hash(chosen_actual_prompt.as_bytes()),rejected_generation_prompt_sha256:artefact::hash(rejected_actual_prompt.as_bytes()),semantics:"Both chosen/rejected responses and frozen reference are evaluated under this same original task prompt. Actual possibly richer generation prompts remain separately bound; SFT and teacher tensor contexts are not overwritten.".into()};
    (prompt, transform)
}

pub fn negative_pair_is_eligible(pair: &PreferencePair) -> bool {
    pair.rejected_eligibility.as_ref().is_some_and(|e| {
        e.negative_allowed()
            && !pair.rejected.passed
            && e.original_quarantined == pair.rejected.quarantined
            && e.report.sha256 == pair.rejected.sandbox_report.sha256
            && e.audit.sha256 == pair.rejected.audit.sha256
            && e.completion_sha256 == artefact::hash(pair.rejected.completion.as_bytes())
            && e.generation_complete
    })
}

fn rejected_prompt(pair: &PreferencePair) -> Result<String> {
    let root = std::env::current_dir()?;
    let raw: serde_json::Value =
        serde_json::from_slice(&fs::read(pair.rejected.raw_generation.verify(&root)?)?)?;
    if raw["evidence_kind"] == "actual_local_model_generation" {
        ensure!(
            raw["task_id"] == pair.task_id,
            "preference rejected generation belongs to another task"
        );
        return raw["request"]["prompt"]
            .as_str()
            .map(str::to_owned)
            .context("original rejected generation prompt absent");
    }
    ensure!(
        matches!(
            raw["evidence_kind"].as_str(),
            Some(
                "actual_checkpoint_bound_rollout_generation"
                    | "actual_local_teacher_content_generation"
            )
        ),
        "unsupported rejected generation prompt provenance"
    );
    let plan: Artefact = serde_json::from_value(raw["plan"].clone())?;
    let plan: serde_json::Value = serde_json::from_slice(&fs::read(plan.verify(&root)?)?)?;
    if let Some(prompt) = plan["student_request"]["prompt"].as_str() {
        ensure!(
            plan["task"]["id"] == pair.task_id,
            "rejected student plan task changed"
        );
        return Ok(prompt.into());
    }
    ensure!(
        plan["task_id"] == pair.task_id,
        "rejected content plan task changed"
    );
    plan["request"]["prompt"]
        .as_str()
        .map(str::to_owned)
        .context("rejected content request prompt absent")
}

pub fn common_preference_prompt(bundle: &EvidenceBundle, pair: &PreferencePair) -> Result<String> {
    ensure!(
        negative_pair_is_eligible(pair),
        "DPO requires independently verified oracle-only negative eligibility; quarantine is never cleared"
    );
    let chosen = bundle
        .samples
        .iter()
        .find(|s| s.id == pair.chosen_sample_id)
        .context("preference chosen sample absent")?;
    ensure!(
        chosen.task.id == pair.task_id,
        "preference task/choice mismatch"
    );
    let rejected_actual = rejected_prompt(pair)?;
    let (expected, transform) =
        preference_conditioning(&chosen.task, &chosen.prompt, &rejected_actual);
    match (&pair.preference_prompt, &pair.prompt_transform) {
        (Some(prompt), Some(recorded)) => {
            ensure!(
                prompt == &expected && json_equal(recorded, &transform)?,
                "preference conditioning/transform differs from original task or actual generation contexts"
            );
            Ok(prompt.clone())
        }
        (None, None) => {
            ensure!(
                chosen.prompt == expected && rejected_actual == expected,
                "legacy preference generation contexts differ; explicit common conditioning transform required"
            );
            Ok(expected)
        }
        _ => anyhow::bail!("preference prompt and transform must be recorded together"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rich_generation_contexts_do_not_become_different_preference_conditioning() {
        let task = train_tasks().remove(0);
        let rich = format!(
            "Privileged verified teacher example\n{}\nPrevious student candidate",
            task.prompt()
        );
        let (prompt, proof) = preference_conditioning(&task, &rich, &task.prompt());
        assert_eq!(prompt, task.prompt());
        assert!(!prompt.contains("Privileged verified teacher example"));
        assert_ne!(
            proof.chosen_generation_prompt_sha256,
            proof.rejected_generation_prompt_sha256
        );
        assert_eq!(
            proof.common_prompt_sha256,
            artefact::hash(prompt.as_bytes())
        );
    }
}
