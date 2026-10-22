use super::*;
use crate::{
    artefact,
    model::{AdapterConfig, AssetManifest},
};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentalCheckpointManifest {
    pub schema_version: u32,
    pub evidence_kind: String,
    pub spec: DistillationWorkerSpec,
    pub freeze_sha256: String,
    pub base_sha256: String,
    pub base_sha256_after: String,
    pub tokenizer_sha256: String,
    pub evidence_bundle: Artefact,
    pub adapter: AdapterConfig,
    pub adapter_sha256: String,
    pub optimizer_sha256: String,
    pub initial_adapter_sha256: String,
    pub initial_adapter: Artefact,
    pub initial_checkpoint: Option<Artefact>,
    pub dpo_reference: Option<Artefact>,
    pub steps: usize,
    pub losses: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub component_losses: Vec<BTreeMap<LossKind, f64>>,
    pub alignment_diagnostics: Vec<serde_json::Value>,
    pub qat_effective_trits: Option<serde_json::Value>,
    pub sample_ids: Vec<String>,
    pub realised_update_counts: BTreeMap<String, usize>,
    pub trace_records: Vec<Artefact>,
    pub teacher_rights: serde_json::Value,
    pub student_execution: serde_json::Value,
    pub reloaded: bool,
    pub production_eligible: bool,
    pub elapsed_seconds: f64,
    pub limitations: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub training_health: Vec<Artefact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Progress {
    pub freeze_sha256: String,
    pub steps: usize,
    pub adapter: Artefact,
    pub optimizer: Artefact,
    pub initial_adapter_sha256: String,
    pub losses: Vec<f64>,
    pub gradient_norms: Vec<f64>,
    pub component_losses: Vec<BTreeMap<LossKind, f64>>,
    pub alignment_diagnostics: Vec<serde_json::Value>,
    pub sample_ids: Vec<String>,
}

pub(super) fn save_tensors(path: &Path, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    let parent = path.parent().context("checkpoint tensor parent absent")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(".{}.safetensors", uuid::Uuid::new_v4()));
    let named = tensors
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<HashMap<_, _>>();
    candle_core::safetensors::save(&named, &tmp)?;
    fs::File::open(&tmp)?.sync_all()?;
    fs::rename(tmp, path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn validate_components(
    objectives: &[WeightedLoss],
    losses: &[f64],
    components: &[BTreeMap<LossKind, f64>],
) -> Result<()> {
    let expected = objectives.iter().map(|l| l.kind).collect::<BTreeSet<_>>();
    let total_weight = objectives.iter().map(|l| l.weight).sum::<f64>();
    ensure!(
        !expected.is_empty()
            && expected.len() == objectives.len()
            && total_weight.is_finite()
            && total_weight > 0.
            && objectives
                .iter()
                .all(|l| l.weight.is_finite() && l.weight > 0.)
            && losses.len() == components.len(),
        "checkpoint objective identities or weights are invalid"
    );
    for (loss, parts) in losses.iter().zip(components) {
        ensure!(
            parts.keys().copied().collect::<BTreeSet<_>>() == expected
                && parts.values().all(|x| x.is_finite()),
            "checkpoint omitted, added or invalidated an objective component"
        );
        let expected_loss = objectives
            .iter()
            .map(|l| l.weight / total_weight * parts[&l.kind])
            .sum::<f64>();
        // Components were evaluated and accumulated in f32 on the declared
        // device. Permit rounding, never an independently invented total.
        ensure!(
            loss.is_finite() && (loss - expected_loss).abs() <= 1e-5 * expected_loss.abs().max(1.),
            "checkpoint total loss differs from its declared weighted objectives"
        );
    }
    Ok(())
}

pub fn validate_checkpoint(
    path: &Path,
    assets: &AssetManifest,
) -> Result<ExperimentalCheckpointManifest> {
    validate_checkpoint_inner(path, assets, true)
}

/// Metadata imports retain the complete provenance/optimiser checks without
/// allocating tensor storage. The caller independently validates tensor headers.
pub(crate) fn validate_checkpoint_metadata(
    path: &Path,
    assets: &AssetManifest,
) -> Result<ExperimentalCheckpointManifest> {
    validate_checkpoint_inner(path, assets, false)
}

fn validate_checkpoint_inner(
    path: &Path,
    assets: &AssetManifest,
    materialise_tensor: bool,
) -> Result<ExperimentalCheckpointManifest> {
    let m: ExperimentalCheckpointManifest =
        serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
    ensure!(
        matches!(m.schema_version, 1 | 2)
            && m.schema_version == m.spec.schema_version
            && m.evidence_kind == "verified_local_teacher_experimental_distillation"
            && !m.production_eligible
            && m.reloaded,
        "invalid experimental checkpoint purpose or reload evidence"
    );
    ensure!(
        assets.files.get("model.safetensors") == Some(&m.base_sha256)
            && m.base_sha256_after == m.base_sha256
            && m.tokenizer_sha256 == assets.files["tokenizer.json"],
        "experimental checkpoint base identity changed"
    );
    ensure!(
        m.steps == m.spec.steps
            && m.steps > 0
            && m.losses.len() == m.steps
            && m.gradient_norms.len() == m.steps
            && m.component_losses.len() == m.steps
            && m.sample_ids.len() == m.steps,
        "incomplete experimental optimiser evidence"
    );
    ensure!(
        m.losses.iter().all(|x| x.is_finite())
            && m.gradient_norms.iter().all(|x| x.is_finite() && *x > 0.),
        "invalid experimental numerical evidence"
    );
    validate_components(&m.spec.losses, &m.losses, &m.component_losses)?;
    ensure!(
        m.adapter.rank == m.spec.rank && m.adapter.alpha == m.spec.alpha,
        "checkpoint adapter configuration differs from its frozen request"
    );
    ensure!(
        m.adapter_sha256 != m.initial_adapter_sha256
            && artefact::file_hash(&path.join("adapter.safetensors"))? == m.adapter_sha256
            && artefact::file_hash(&path.join("optimizer.safetensors"))? == m.optimizer_sha256,
        "experimental adapter or optimiser identity invalid"
    );
    if m.spec.student_kind == DistillationStudent::TernaryBitnet {
        let trits = m
            .qat_effective_trits
            .as_ref()
            .context("ternary checkpoint lacks effective-lattice evidence")?;
        ensure!(
            trits["initial"]["sha256"] != trits["final"]["sha256"]
                && trits["final"]["changed_from_frozen_base"]
                    .as_u64()
                    .is_some_and(|n| n > 0),
            "latent-only changes are not a genuinely adapted ternary checkpoint"
        );
    }
    let root = std::env::current_dir()?;
    ensure!(
        artefact::file_hash(&path.join("freeze.json"))? == m.freeze_sha256,
        "experimental frozen order/configuration checksum changed"
    );
    let frozen: serde_json::Value = serde_json::from_slice(&fs::read(path.join("freeze.json"))?)?;
    ensure!(
        frozen["spec"] == serde_json::to_value(&m.spec)?
            && frozen["evidence"] == serde_json::to_value(&m.evidence_bundle)?,
        "experimental manifest differs from its frozen inputs"
    );
    m.evidence_bundle.verify(&root)?;
    let bundle = crate::teacher_evidence::load_verified(&m.evidence_bundle.path)?;
    let order = frozen["order"]
        .as_array()
        .context("checkpoint frozen order absent")?;
    ensure!(
        order.len() == m.steps,
        "checkpoint order length differs from realised updates"
    );
    let mut counts = BTreeMap::<String, usize>::new();
    let preference = m
        .spec
        .losses
        .iter()
        .any(|l| l.kind == LossKind::PreferenceDpo);
    let tokenizer = tokenizers::Tokenizer::from_file(m.spec.student_base.join("tokenizer.json"))
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if m.schema_version >= 2 {
        let selection = super::worker::choose(&m.spec, &bundle, &tokenizer)?;
        ensure!(
            frozen["schema_version"] == 2
                && frozen["order"] == serde_json::to_value(&selection.order)?
                && frozen["encoded_sequences"] == serde_json::to_value(&selection.encoded)?
                && frozen["prepared_selection"]["canonical_request_sha256"]
                    == artefact::hash(&serde_json::to_vec(&m.spec)?)
                && frozen["prepared_selection"]["engine_sha256"] == frozen["engine_binary_sha256"]
                && frozen["prepared_selection"]["selection"]["order"] == frozen["order"],
            "prepared selection, optimiser order, student encoding or original engine differs"
        );
    }
    let mut expected_contexts = BTreeMap::new();
    let mut expected_preferences = BTreeMap::new();
    for (index, step) in order.iter().enumerate() {
        let id = step["sample_id"]
            .as_str()
            .context("checkpoint order sample absent")?;
        let teacher = step["teacher_id"]
            .as_str()
            .context("checkpoint order teacher absent")?;
        let sample = bundle
            .samples
            .iter()
            .find(|s| s.id == id)
            .context("checkpoint sample absent from verified evidence")?;
        ensure!(
            m.sample_ids[index] == id
                && sample.teacher_id == teacher
                && m.spec.teacher_ids.iter().any(|t| t == teacher),
            "checkpoint realised sample/teacher differs from its frozen order"
        );
        if let Some(transform) = conditioning::transform(sample)? {
            conditioning::verify_transform(sample, &transform)?;
            expected_contexts.insert(id.to_owned(), transform);
        }
        if preference && !expected_preferences.contains_key(id) {
            expected_preferences.insert(
                id.to_owned(),
                super::preference::prepare(&m.spec, &bundle, sample, &tokenizer)?,
            );
        }
        // Independently reconstruct the exact assistant masks/window and
        // conditioning, rather than trusting a plausible frozen JSON record.
        let sequence = if m
            .spec
            .losses
            .iter()
            .all(|loss| loss.kind == LossKind::PreferenceDpo)
        {
            expected_preferences[id].chosen.clone()
        } else {
            alignment::encode_with_policy(
                &tokenizer,
                &student::prefix(&m.spec, &conditioning::student_prompt(sample)),
                &sample.completion,
                student::eos(&m.spec),
                m.spec.max_sequence_tokens,
                m.spec.sequence_policy,
            )?
        };
        ensure!(
            frozen["encoded_sequences"][id] == serde_json::to_value(sequence)?,
            "checkpoint student conditioning, assistant masks or sequence window changed"
        );
        *counts.entry(format!("teacher/{teacher}")).or_default() += 1;
        let content = serde_json::to_value(sample.content)?;
        *counts
            .entry(format!(
                "content/{}",
                content
                    .as_str()
                    .context("checkpoint content identity absent")?
            ))
            .or_default() += 1;
        if preference {
            ensure!(
                bundle.preferences.iter().any(|p| p.chosen_sample_id == id
                    && crate::teacher_evidence::negative_pair_is_eligible(p)),
                "checkpoint preference update lacks a real permitted pair"
            );
            *counts.entry("content/preference".into()).or_default() += 1;
        }
    }
    let optional_map_matches = |key: &str, expected: serde_json::Value| {
        frozen[key] == expected
            || (expected.as_object().is_some_and(|map| map.is_empty()) && frozen[key].is_null())
    };
    ensure!(
        optional_map_matches(
            "context_prompt_transforms",
            serde_json::to_value(&expected_contexts)?
        ),
        "checkpoint context transforms do not match its exact privileged teacher/shorter student prompts"
    );
    ensure!(
        optional_map_matches(
            "preference_sequences",
            serde_json::to_value(&expected_preferences)?
        ),
        "checkpoint DPO pair identity/common conditioning/sequences changed"
    );
    ensure!(
        optional_map_matches(
            "excluded_overlength_samples",
            serde_json::to_value(super::worker::excluded_overlength(
                &m.spec, &bundle, &tokenizer
            )?)?
        ),
        "checkpoint overlength exclusions differ from its frozen sequence policy and verified evidence"
    );
    ensure!(
        counts == m.realised_update_counts,
        "checkpoint realised-axis counts differ from its exact recorded updates"
    );
    for teacher in &m.spec.teacher_ids {
        ensure!(
            counts
                .get(&format!("teacher/{teacher}"))
                .is_some_and(|n| *n > 0),
            "checkpoint never updated a named teacher"
        );
    }
    if let Some(policy) = &m.spec.health_check {
        ensure!(
            m.training_health.len() == policy.checkpoints.len(),
            "checkpoint lacks predeclared train generation health observations"
        );
        for (artifact, step) in m.training_health.iter().zip(&policy.checkpoints) {
            let observation: serde_json::Value =
                serde_json::from_slice(&fs::read(artifact.verify(&root)?)?)?;
            ensure!(
                observation["step"] == *step
                    && observation["policy"] == serde_json::to_value(policy)?
                    && observation["sample_id"] == m.sample_ids[0]
                    && observation["held_out_used"] == false,
                "train generation health binding differs"
            );
            let adapter: Artefact = serde_json::from_value(observation["adapter"].clone())?;
            adapter.verify(&root)?;
            ensure!(
                adapter.path
                    == m.spec
                        .output
                        .join(format!("step-{step:06}/adapter.safetensors")),
                "health probe used another checkpoint"
            );
            let healthy = observation["generation"]["Ok"]["complete"] == true
                && observation["generation"]["Ok"]["completion"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty());
            let sample = bundle
                .samples
                .iter()
                .find(|s| s.id == m.sample_ids[0])
                .context("health sample absent")?;
            let prompt = if m
                .spec
                .losses
                .iter()
                .all(|l| l.kind == LossKind::PreferenceDpo)
            {
                expected_preferences[&sample.id].common_prompt.clone()
            } else {
                conditioning::student_prompt(sample)
            };
            let expected_request = crate::model::GenerationRequest {
                prompt,
                max_new_tokens: policy.max_new_tokens,
                seed: m.spec.seed,
                temperature: 0.2,
                top_p: 0.95,
                timeout_seconds: policy.timeout_seconds,
            };
            ensure!(
                observation["generation_healthy"] == healthy
                    && (healthy || !policy.stop_on_unhealthy)
                    && observation["request"] == serde_json::to_value(expected_request)?
                    && observation["execution"]["checkpoint_id"] == adapter.sha256
                    && observation["execution"]["requested"]
                        == serde_json::to_value(&m.spec.student_runtime)?,
                "health status differs from original generation or failed frozen stop policy"
            );
        }
    } else {
        ensure!(
            m.training_health.is_empty(),
            "undeclared health observations in checkpoint"
        );
    }
    for content in &m.spec.contents {
        let label = serde_json::to_value(content)?;
        ensure!(
            counts
                .get(&format!(
                    "content/{}",
                    label.as_str().context("content identity absent")?
                ))
                .is_some_and(|n| *n > 0),
            "checkpoint never updated a named content kind"
        );
    }
    m.initial_adapter.verify(&root)?;
    ensure!(
        m.initial_adapter.sha256 == m.initial_adapter_sha256,
        "initial experimental adapter artefact differs from its declared identity"
    );
    let tensor_objective = m.spec.losses.iter().any(|l| {
        matches!(
            l.kind,
            LossKind::SoftTokenKl | LossKind::HiddenFeature | LossKind::AttentionRelation
        )
    });
    let expected_traces = if tensor_objective {
        m.sample_ids.iter().cloned().collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let mut actual_traces = BTreeSet::new();
    for trace in &m.trace_records {
        let record = super::targets::validate_record(&trace.verify(&root)?)?;
        ensure!(
            actual_traces.insert(record.sample_id),
            "duplicate checkpoint target trace"
        );
    }
    ensure!(
        actual_traces == expected_traces,
        "checkpoint target traces do not cover its actual tensor objectives"
    );
    let kl = m
        .spec
        .losses
        .iter()
        .any(|l| l.kind == LossKind::SoftTokenKl);
    ensure!(
        m.alignment_diagnostics.len() == if kl { m.steps } else { 0 },
        "checkpoint KL alignment diagnostics are incomplete"
    );
    for (index, diagnostic) in m.alignment_diagnostics.iter().enumerate() {
        ensure!(
            diagnostic["step"] == index + 1
                && diagnostic["sample_id"] == m.sample_ids[index]
                && diagnostic["teacher_id"] == order[index]["teacher_id"]
                && diagnostic["padding_mass"]["discarded_probability_mass"] == 0.
                && diagnostic["padding_mass"]["temperature"] == m.spec.temperature,
            "checkpoint KL alignment diagnostics differ from actual updates"
        );
    }
    if let Some(initial) = &m.initial_checkpoint {
        initial.verify(&root)?;
    }
    ensure!(
        m.dpo_reference.is_some() == preference,
        "checkpoint DPO reference presence differs from its objective"
    );
    if let Some(reference) = &m.dpo_reference {
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(reference.verify(&root)?)?)?;
        ensure!(
            value["initial_adapter_sha256"] == m.initial_adapter_sha256
                && value["freeze_sha256"] == m.freeze_sha256
                && value["preference_sequences_sha256"]
                    == artefact::hash(&serde_json::to_vec(&expected_preferences)?),
            "frozen DPO reference identity changed"
        );
        let logp: BTreeMap<String, (f64, f64)> =
            serde_json::from_value(value["log_probabilities"].clone())?;
        ensure!(
            logp.keys().collect::<BTreeSet<_>>()
                == expected_preferences.keys().collect::<BTreeSet<_>>()
                && logp
                    .values()
                    .all(|(chosen, rejected)| chosen.is_finite() && rejected.is_finite()),
            "frozen DPO references lack complete finite common-prompt coverage"
        );
    }
    if materialise_tensor {
        let _ = candle_core::safetensors::load(path.join("adapter.safetensors"), &Device::Cpu)?;
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn objective_evidence_cannot_omit_a_component_or_change_the_total() {
        let objectives = vec![
            WeightedLoss {
                kind: LossKind::SequenceSft,
                weight: 3.,
            },
            WeightedLoss {
                kind: LossKind::HiddenFeature,
                weight: 1.,
            },
        ];
        let complete = BTreeMap::from([(LossKind::SequenceSft, 2.), (LossKind::HiddenFeature, 6.)]);
        assert!(validate_components(&objectives, &[3.], std::slice::from_ref(&complete)).is_ok());
        assert!(validate_components(&objectives, &[2.], std::slice::from_ref(&complete)).is_err());
        let missing = BTreeMap::from([(LossKind::SequenceSft, 4.)]);
        assert!(validate_components(&objectives, &[3.], &[missing]).is_err());
    }
}
