use super::*;

pub(super) fn teacher(root: &Path, frozen: &FrozenTeacher) -> Result<TeacherEvidence> {
    let manifest_path = frozen.assets.verify(root)?;
    let assets: model::AssetManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    ensure!(
        assets.model_id == frozen.model.model_id
            && assets.revision == frozen.model.revision
            && equivalent_licence(&assets.licence, &frozen.model.licence),
        "teacher model/asset/licence pin mismatch"
    );
    let directory = root.join(&frozen.model.directory);
    let (original, source) = match frozen.model.weight_format {
        WeightFormat::Gguf => {
            let proof = hub::validate_gguf_asset_metadata(&directory)?;
            ensure!(
                json_equal(&proof.assets, &assets)?,
                "teacher manifest differs from original pinned GGUF identity"
            );
            (proof.source_assets.manifest, directory.join("source"))
        }
        WeightFormat::Safetensors => {
            let verified = if frozen.model.directory == Path::new("offline_bundle/base") {
                model::validate_assets(&directory)?
            } else {
                hub::validate_downloaded_asset_metadata(&directory)?
            };
            ensure!(
                json_equal(&verified, &assets)?,
                "teacher manifest differs from verified weights"
            );
            (verified, directory.clone())
        }
    };
    let mut licences = Vec::new();
    for name in original.files.keys().filter(|n| {
        let n = n.to_ascii_lowercase();
        n.contains("licen") || n.contains("copying") || n == "readme.md"
    }) {
        licences.push(capture(root, source.join(name).strip_prefix(root)?)?);
    }
    ensure!(
        !licences.is_empty(),
        "teacher has no preserved original licence evidence"
    );
    let recognised = matches!(
        original.licence.as_str(),
        "Apache-2.0" | "apache-2.0" | "MIT" | "mit" | "DeepSeek-Model-Licence-1.0" | "deepseek"
    );
    ensure!(
        recognised,
        "unreviewed local model licence; teacher disabled for training"
    );
    let restricted = original.model_id.starts_with("deepseek-ai/");
    let rights=LocalRightsDecision {eligible:true,scope:"local open-weight output adaptation only; no API terms inferred".into(),licence:original.licence.clone(),licence_artefacts:licences,inherited_restrictions:if restricted{vec!["DeepSeek model output distillation is a derivative: retain model licence, attribution, modification notices and inherited use restrictions.".into()]}else{vec!["Retain original model licence, copyright notices and attribution; no ownership guarantee for independently reproduced code.".into()]},reasoning:"Explicit recognised original model licence and publisher-verified local assets, subject to preserved licence obligations".into()};
    let tokeniser_sha256 = original
        .files
        .get("tokenizer.json")
        .context("original tokeniser absent")?
        .clone();
    let student_tokeniser_sha256 = model::TOKENIZER_SHA256.to_owned();
    ensure!(
        crate::artefact::file_hash(&root.join("offline_bundle/base/tokenizer.json"))?
            == student_tokeniser_sha256,
        "original student tokeniser checksum mismatch"
    );
    let aligned = tokeniser_sha256 == student_tokeniser_sha256;
    let native_trace = matches!(frozen.model.architecture.as_str(), "qwen2" | "llama");
    let capabilities = EvidenceCapabilities {
        complete_sequences: SignalAvailability::absent(
            true,
            "No passing sequence from this teacher in this bundle",
        ),
        logits: SignalAvailability::absent(
            native_trace,
            "Not recorded: a native teacher-forced worker must produce and bind actual token logits; KL also requires tokeniser alignment",
        ),
        hidden_states: SignalAvailability::absent(
            native_trace,
            "Not recorded: require actual layer states and declared dimensional projection",
        ),
        attention: SignalAvailability::absent(
            native_trace,
            "Not recorded: require actual causal attention matrices and declared head mapping",
        ),
        process: SignalAvailability::absent(
            true,
            "No independently generated and checked planning/process trace",
        ),
        tool_trajectories: SignalAvailability::absent(
            true,
            "A sandbox grade alone is not a teacher feedback/revision trajectory",
        ),
        context: SignalAvailability::absent(
            true,
            "No generation with pinned additional train-only context",
        ),
        preferences: SignalAvailability::absent(
            true,
            "No safe same-task verified positive/negative pair",
        ),
        on_policy: SignalAvailability::absent(
            true,
            "No checkpoint-bound student rollout plus teacher feedback and regrading",
        ),
        self_distillation: SignalAvailability::absent(
            true,
            "No original student-checkpoint teacher outputs; supervised teacher rows cannot be relabelled",
        ),
        tokeniser_sha256,
        student_tokeniser_sha256,
        tokeniser_aligned: aligned,
    };
    Ok(TeacherEvidence {
        id: frozen.model.id.clone(),
        model: frozen.model.clone(),
        assets: frozen.assets.clone(),
        adapter_checkpoint: None,
        capabilities,
        rights,
    })
}

fn equivalent_licence(a: &str, b: &str) -> bool {
    if ["apache-2.0", "mit"]
        .iter()
        .any(|known| a.eq_ignore_ascii_case(known) && b.eq_ignore_ascii_case(known))
    {
        true
    } else {
        a == b
    }
}

/// Reconstruct the complete accepted bundle from its original observations.
/// Editing a convenience completion, rights decision or advertised capability
/// cannot bypass the original generation, model, grader or audit bindings.
pub fn load_verified(path: &Path) -> Result<EvidenceBundle> {
    let path = if path.is_dir() {
        path.join("bundle.json")
    } else {
        path.to_owned()
    };
    let root = std::env::current_dir()?.canonicalize()?;
    let path = path.canonicalize()?;
    ensure!(path.starts_with(&root), "bundle outside workspace");
    ensure!(
        fs::metadata(&path)?.len() <= 128 << 20,
        "teacher evidence bundle oversized"
    );
    let recorded: EvidenceBundle = serde_json::from_slice(&fs::read(&path)?)?;
    ensure!(
        recorded.schema_version == 1
            && recorded.evidence_kind
                == "actual_local_teacher_training_evidence_not_provider_assessment"
            && recorded.created_unix_ms > 0,
        "unsupported teacher evidence kind"
    );
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(
        path.parent()
            .context("bundle parent absent")?
            .join("manifest.json"),
    )?)?;
    let bound: Artefact = serde_json::from_value(manifest["bundle"].clone())?;
    ensure!(
        bound.verify(&root)? == path,
        "bundle manifest names another file"
    );
    let mut comparisons = Vec::new();
    for input in &recorded.input_comparisons {
        input.verify(&root)?;
        if input.path.file_name().is_some_and(|n| n == "report.json") {
            comparisons.push(
                input
                    .path
                    .parent()
                    .context("comparison parent absent")?
                    .to_owned(),
            );
        }
    }
    let mut rebuilt = collect::build_existing(&root, &comparisons)?;
    for record in &recorded.content_records {
        variants::append_verified(&root, &mut rebuilt, record)?;
    }
    for record in &recorded.rollout_records {
        rollouts::append_verified(&root, &mut rebuilt, record)?;
    }
    rebuilt.created_unix_ms = recorded.created_unix_ms;
    ensure!(
        json_equal(&recorded, &rebuilt)?,
        "teacher bundle differs from independently revalidated original evidence"
    );
    ensure!(
        !recorded.samples.is_empty()
            && manifest["samples"] == recorded.samples.len()
            && manifest["preferences"] == recorded.preferences.len()
            && manifest["train_corpus_sha256"] == recorded.train_corpus_sha256
            && manifest["promotable"] == false,
        "teacher bundle coverage/manifest mismatch"
    );
    Ok(recorded)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recognised_spdx_case_preserves_terms_without_conflating_licences() {
        assert!(equivalent_licence("Apache-2.0", "apache-2.0"));
        assert!(equivalent_licence("MIT", "mit"));
        assert!(!equivalent_licence("MIT", "Apache-2.0"));
        assert!(!equivalent_licence(
            "UnreviewedLicence",
            "unreviewedlicence"
        ));
    }
}
