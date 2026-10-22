//! Teacher-forced targets are acquired from verified native models, published
//! before optimisation, and reused only after their complete binding verifies.
use super::{
    DistillationWorkerSpec, LossKind,
    alignment::{self, AnchorAlignment, EncodedSequence},
    checkpoint,
};
use crate::{
    artefact::{self, Artefact},
    teacher_evidence::{EvidenceBundle, EvidenceSample, RecordedModel},
};
use anyhow::{Context, Result, ensure};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    os::fd::AsRawFd,
    path::{Path, PathBuf},
};
use tokenizers::Tokenizer;

pub(super) fn required(spec: &DistillationWorkerSpec) -> bool {
    spec.losses.iter().any(|l| {
        matches!(
            l.kind,
            LossKind::SoftTokenKl | LossKind::HiddenFeature | LossKind::AttentionRelation
        )
    })
}

/// Render the original teacher template and tokenise without loading weights.
/// The teacher keeps privileged context; the student prompt is never substituted.
pub(super) fn encode_teacher(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    sample: &EvidenceSample,
) -> Result<EncodedSequence> {
    let teacher = bundle
        .teachers
        .iter()
        .find(|t| t.id == sample.teacher_id)
        .context("teacher absent")?;
    let directory = &teacher.model.directory;
    let gguf = directory.join("gguf-assets.json").is_file();
    let source = if gguf {
        directory.join("source")
    } else {
        directory.clone()
    };
    let config: serde_json::Value = serde_json::from_slice(&fs::read(source.join("config.json"))?)?;
    let (tokenizer, prefix, eos) = if teacher.model.architecture == "bitnet" {
        (
            Tokenizer::from_file(source.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!(e.to_string()))?,
            crate::bitnet::formatted_prompt(&sample.prompt),
            crate::bitnet::END_OF_TURN,
        )
    } else {
        let encoder = crate::model::ComparativePromptEncoder::load(directory, gguf)?;
        let generation = source.join("generation_config.json");
        let generation: serde_json::Value = if generation.is_file() {
            serde_json::from_slice(&fs::read(generation)?)?
        } else {
            serde_json::Value::Null
        };
        let eos = if directory == Path::new("offline_bundle/base") {
            151645
        } else {
            crate::model::parse_eos_ids(
                generation
                    .get("eos_token_id")
                    .filter(|v| !v.is_null())
                    .unwrap_or(&config["eos_token_id"]),
            )?[0]
        };
        (
            encoder.tokenizer().clone(),
            encoder.formatted_prompt(&sample.prompt)?,
            eos,
        )
    };
    let capacity = config["max_position_embeddings"]
        .as_u64()
        .context("teacher context capacity absent")? as usize;
    let cap = spec.max_sequence_tokens.min(capacity);
    alignment::encode_with_policy(&tokenizer, &prefix, &sample.completion, eos, cap, spec.sequence_policy)
        .with_context(|| format!("teacher sequence sample={} content={:?}, teacher={}, effective cap={cap}, frozen training cap={} (collection/evaluation limits are independent)", sample.id, sample.content, sample.teacher_id, spec.max_sequence_tokens))
}

pub(super) fn validate_capacity(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    sample: &EvidenceSample,
    student: &EncodedSequence,
    teacher: &EncodedSequence,
) -> Result<()> {
    let selected = bundle
        .teachers
        .iter()
        .find(|t| t.id == sample.teacher_id)
        .context("teacher absent")?;
    let source = if selected.model.directory.join("gguf-assets.json").is_file() {
        selected.model.directory.join("source")
    } else {
        selected.model.directory.clone()
    };
    let cfg: serde_json::Value = serde_json::from_slice(&fs::read(source.join("config.json"))?)?;
    let width = cfg["vocab_size"]
        .as_u64()
        .context("teacher vocabulary capacity absent")?;
    let hidden = cfg["hidden_size"]
        .as_u64()
        .context("teacher feature width absent")?;
    let features = spec.losses.iter().any(|l| {
        matches!(
            l.kind,
            LossKind::HiddenFeature | LossKind::AttentionRelation
        )
    });
    let logits = spec.losses.iter().any(|l| l.kind == LossKind::SoftTokenKl);
    let anchors = if features {
        Some(alignment::feature_anchors(student, teacher)?)
    } else {
        None
    };
    if logits {
        let teacher_tokens = Tokenizer::from_file(source.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let student_tokens = Tokenizer::from_file(spec.student_base.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        ensure!(
            teacher_tokens.get_vocab(true) == student_tokens.get_vocab(true),
            "full-vocabulary KL unavailable before capture: teacher/student semantic token-ID mappings differ for sample {}",
            sample.id
        );
        alignment::prediction_rows(student, teacher)?;
    }
    let n = anchors.as_ref().map_or(0, |a| a.teacher_rows.len() as u64);
    let payload = 4
        * (if logits {
            student.targets.len() as u64 * width
        } else {
            0
        } + n * hidden
            + n * n);
    ensure!(
        payload <= 512_u64 << 20,
        "teacher trace payload {payload} bytes exceeds 512 MiB for sample {}",
        sample.id
    );
    // forward_trace materialises head attention before reducing to anchors.
    let heads = cfg["num_attention_heads"]
        .as_u64()
        .context("teacher attention capacity absent")?;
    let t = teacher.input_ids.len() as u64;
    ensure!(
        !features || 4 * heads * t * t <= 512_u64 << 20,
        "teacher attention capture exceeds 512 MiB for sample {}: heads={heads}, sequence={t}",
        sample.id
    );
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceRecord {
    pub schema_version: u32,
    pub sample_id: String,
    pub teacher_id: String,
    pub fingerprint: String,
    pub tensors: Artefact,
    pub anchors: Option<AnchorAlignment>,
    pub execution: serde_json::Value,
    pub source_evidence: Artefact,
    pub assistant_tokens: usize,
    pub full_vocabulary: bool,
    pub tail_mass_discarded: f64,
    pub vocabulary_alignment: Option<serde_json::Value>,
}

#[derive(Clone)]
pub(super) struct PreparedTrace {
    pub record: TraceRecord,
    pub artefact: Artefact,
}

fn ensure_disk_capacity(directory: &Path, planned: u64) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(directory.as_os_str().as_bytes())?;
    let mut info = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path is NUL terminated and info points to valid writable storage.
    ensure!(
        unsafe { libc::statvfs(path.as_ptr(), info.as_mut_ptr()) } == 0,
        "native teacher-cache free space could not be measured"
    );
    // SAFETY: statvfs succeeded and initialised the complete structure.
    let info = unsafe { info.assume_init() };
    let free = (info.f_bavail as u128) * (info.f_frsize as u128);
    ensure!(
        free >= (planned as u128) + (512_u128 << 20),
        "native teacher target capture would breach its 512 MiB measured disk floor"
    );
    let mut used = 0_u64;
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "safetensors") {
            used = used
                .checked_add(entry.metadata()?.len())
                .context("teacher-cache inventory overflow")?;
        }
    }
    ensure!(
        used.saturating_add(planned) <= 8_u64 << 30,
        "native shared teacher targets exceed the 8 GiB cache budget; preserve evidence and explicitly expand the study budget"
    );
    Ok(())
}

pub fn validate_record(path: &Path) -> Result<TraceRecord> {
    let record: TraceRecord = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        record.schema_version == 1 && record.tail_mass_discarded == 0.,
        "invalid native trace contract"
    );
    let root = std::env::current_dir()?;
    record.tensors.verify(&root)?;
    record.source_evidence.verify(&root)?;
    Ok(record)
}

pub(super) fn prepare(
    spec: &DistillationWorkerSpec,
    bundle: &EvidenceBundle,
    samples: &BTreeMap<String, EvidenceSample>,
    encoded: &BTreeMap<String, EncodedSequence>,
    tokenizer: &Tokenizer,
) -> Result<BTreeMap<String, PreparedTrace>> {
    let logits = spec.losses.iter().any(|x| x.kind == LossKind::SoftTokenKl);
    let features = spec.losses.iter().any(|x| {
        matches!(
            x.kind,
            LossKind::HiddenFeature | LossKind::AttentionRelation
        )
    });
    let attention = spec
        .losses
        .iter()
        .any(|x| x.kind == LossKind::AttentionRelation);
    if !logits && !features {
        return Ok(BTreeMap::new());
    }
    let cache_directory = Path::new("offline_bundle/distillation-traces");
    if spec.teacher_target_policy == super::TeacherTargetPolicy::Acquire {
        fs::create_dir_all(cache_directory)?;
    }
    // Only capture mutates the shared cache. Cache-only CPU workers remain
    // parallel readers; acquisition is a separately reserved prewarm phase.
    let _cache_lease = if spec.teacher_target_policy == super::TeacherTargetPolicy::Acquire {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(cache_directory.join(".capture.lock"))?;
        // SAFETY: the File owns this descriptor until the phase returns.
        ensure!(
            unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another prewarm owns the teacher cache; validated progress can resume after it finishes"
        );
        Some(file)
    } else {
        None
    };
    let root = std::env::current_dir()?;
    let engine = artefact::file_hash(&std::env::current_exe()?)?;
    let tokeniser_hash = artefact::file_hash(&spec.student_base.join("tokenizer.json"))?;
    let student_config: serde_json::Value =
        serde_json::from_slice(&fs::read(spec.student_base.join("config.json"))?)?;
    let student_width = student_config["vocab_size"]
        .as_u64()
        .context("student vocabulary width absent")? as usize;
    let mut semantic_ids = tokenizer.get_vocab(true).into_values().collect::<Vec<_>>();
    semantic_ids.sort_unstable();
    semantic_ids.dedup();
    let semantic_hash = artefact::hash(&serde_json::to_vec(&semantic_ids)?);
    let mut prepared = BTreeMap::new();
    let mut budget = 0_u64;
    for id in &spec.teacher_ids {
        let teacher = bundle
            .teachers
            .iter()
            .find(|t| &t.id == id)
            .context("selected teacher missing from verified bundle")?;
        ensure!(
            teacher.rights.eligible,
            "teacher output training rights are unresolved"
        );
        let selected = samples
            .values()
            .filter(|s| &s.teacher_id == id)
            .collect::<Vec<_>>();
        let mut pending = Vec::<(&EvidenceSample, String, PathBuf)>::new();
        for sample in selected {
            let mut binding = serde_json::json!({"sample":sample.sequence_sha256,"sample_id":sample.id,"teacher_assets":teacher.assets,"teacher_checkpoint":teacher.adapter_checkpoint,"student_tokeniser":tokeniser_hash,"student_sequence":encoded[&sample.id],"sequence_policy":spec.sequence_policy,"runtime":spec.teacher_runtime,"logits":logits,"features":features,"attention":attention,"max_tokens":spec.max_sequence_tokens,"engine":engine});
            if spec.schema_version >= 2 {
                binding["teacher_sequence"] =
                    serde_json::to_value(encode_teacher(spec, bundle, sample)?)?;
                binding["selection_contract_version"] = 2.into();
            }
            let fingerprint = artefact::hash(&serde_json::to_vec(&binding)?);
            let mut selected_fingerprint = fingerprint.clone();
            let mut metadata = cache_directory.join(format!("{fingerprint}.json"));
            // A previously recorded superset can satisfy a smaller requested
            // signal set without recapturing or duplicating full-vocabulary data.
            if !metadata.exists() {
                for (l, f, a) in [(true, true, true), (true, true, false), (false, true, true)] {
                    if (!logits || l) && (!features || f) && (!attention || a) {
                        let mut candidate = binding.clone();
                        candidate["logits"] = serde_json::json!(l);
                        candidate["features"] = serde_json::json!(f);
                        candidate["attention"] = serde_json::json!(a);
                        let hash = artefact::hash(&serde_json::to_vec(&candidate)?);
                        let path = cache_directory.join(format!("{hash}.json"));
                        if path.exists() {
                            selected_fingerprint = hash;
                            metadata = path;
                            break;
                        }
                    }
                }
            }
            if metadata.exists() {
                let record = validate_record(&metadata)?;
                budget = budget
                    .checked_add(record.tensors.bytes)
                    .context("teacher target budget overflow")?;
                ensure!(
                    budget <= 8_u64 << 30,
                    "cached selected teacher targets exceed the 8 GiB worker disk budget"
                );
                ensure!(
                    record.fingerprint == selected_fingerprint
                        && record.sample_id == sample.id
                        && record.teacher_id == *id,
                    "cached teacher trace binding differs"
                );
                prepared.insert(
                    sample.id.clone(),
                    PreparedTrace {
                        record,
                        artefact: Artefact::capture(&root, metadata)?,
                    },
                );
            } else {
                pending.push((sample, fingerprint, metadata));
            }
        }
        if pending.is_empty() {
            continue;
        }
        ensure!(
            spec.teacher_target_policy == super::TeacherTargetPolicy::Acquire,
            "required immutable native teacher targets are missing; run the separately reserved prewarm phase"
        );
        let mut runtime = RecordedModel::teacher(teacher, spec.teacher_runtime.clone())?;
        let (teacher_vocabulary, teacher_hidden, teacher_layers) = runtime.dimensions();
        if logits {
            ensure!(
                runtime.tokenizer().get_vocab(true) == tokenizer.get_vocab(true),
                "full-vocabulary KL unavailable: original teacher/student token-ID mappings differ"
            );
        }
        for (sample, fingerprint, metadata) in pending {
            let student = &encoded[&sample.id];
            let teacher_sequence = encode_teacher(spec, bundle, sample)?;
            let actual_sequence = alignment::encode_with_policy(
                runtime.tokenizer(),
                &runtime.formatted_prompt(&sample.prompt)?,
                &sample.completion,
                runtime.primary_eos_token_id(),
                spec.max_sequence_tokens,
                spec.sequence_policy,
            )?;
            ensure!(
                serde_json::to_value(&teacher_sequence)? == serde_json::to_value(actual_sequence)?,
                "weight-free teacher encoding differs from loaded capture runtime"
            );
            let anchors = if features {
                Some(alignment::feature_anchors(student, &teacher_sequence)?)
            } else {
                None
            };
            let logit_bytes = if logits {
                (student.targets.len() as u64)
                    .checked_mul(teacher_vocabulary as u64)
                    .and_then(|n| n.checked_mul(4))
                    .context("teacher trace size overflow")?
            } else {
                0
            };
            let feature_bytes = if let Some(a) = &anchors {
                let n = a.teacher_rows.len() as u64;
                n.checked_mul(teacher_hidden as u64)
                    .and_then(|h| {
                        if attention {
                            h.checked_add(n.checked_mul(n)?)
                        } else {
                            Some(h)
                        }
                    })
                    .and_then(|n| n.checked_mul(4))
                    .context("teacher feature trace size overflow")?
            } else {
                0
            };
            let estimated = logit_bytes
                .checked_add(feature_bytes)
                .context("teacher trace size overflow")?;
            budget = budget
                .checked_add(estimated)
                .context("teacher target budget overflow")?;
            ensure!(
                estimated <= 512_u64 << 20,
                "one native teacher trace exceeds the 512 MiB resident-payload bound"
            );
            ensure!(
                budget <= 8_u64 << 30,
                "selected teacher targets exceed the 8 GiB worker disk budget; use an explicit larger study budget"
            );
            ensure_disk_capacity(cache_directory, estimated.saturating_add(16 << 20))?;
            runtime.clear_cache();
            let input = Tensor::from_slice(
                &teacher_sequence.input_ids,
                (1, teacher_sequence.input_ids.len()),
                runtime.device(),
            )?;
            let layer = (teacher_layers - 1) / 2;
            let trace = if features {
                Some(runtime.forward_trace(&input, layer)?)
            } else {
                None
            };
            let final_hidden = match &trace {
                Some(t) => t.final_hidden.clone(),
                None => runtime.hidden(&input)?,
            };
            let mut tensors = BTreeMap::new();
            if logits {
                let teacher_rows = alignment::prediction_rows(student, &teacher_sequence)?;
                let positions = teacher_rows
                    .iter()
                    .map(|&r| teacher_sequence.prediction_positions[r as usize])
                    .collect::<Vec<_>>();
                let hidden = final_hidden.squeeze(0)?.index_select(
                    &Tensor::from_slice(&positions, positions.len(), runtime.device())?,
                    0,
                )?;
                tensors.insert(
                    "logits".into(),
                    runtime
                        .logits(&hidden)?
                        .detach()
                        .to_dtype(candle_core::DType::F32)?
                        .to_device(&Device::Cpu)?,
                );
            }
            if let Some(a) = &anchors {
                let trace = trace
                    .as_ref()
                    .context("selected teacher features were not captured")?;
                let index =
                    Tensor::from_slice(&a.teacher_rows, a.teacher_rows.len(), runtime.device())?;
                tensors.insert(
                    "hidden".into(),
                    trace
                        .layer_hidden
                        .index_select(&index, 0)?
                        .detach()
                        .to_dtype(candle_core::DType::F32)?
                        .to_device(&Device::Cpu)?,
                );
                if attention {
                    tensors.insert(
                        "attention".into(),
                        trace
                            .mean_attention
                            .index_select(&index, 0)?
                            .index_select(&index, 1)?
                            .detach()
                            .to_dtype(candle_core::DType::F32)?
                            .to_device(&Device::Cpu)?,
                    );
                }
            }
            for tensor in tensors.values() {
                ensure!(
                    tensor
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|x| x.is_finite()),
                    "native teacher trace contains non-finite values"
                );
            }
            let path = metadata.with_extension("safetensors");
            checkpoint::save_tensors(&path, &tensors)?;
            let record = TraceRecord {
                schema_version: 1,
                sample_id: sample.id.clone(),
                teacher_id: id.clone(),
                fingerprint,
                tensors: Artefact::capture(&root, path)?,
                anchors,
                execution: runtime.execution_metadata(),
                source_evidence: sample.observation.clone(),
                assistant_tokens: student.targets.len(),
                full_vocabulary: logits,
                tail_mass_discarded: 0.,
                vocabulary_alignment: if logits {
                    Some(serde_json::json!({
                        "semantic_token_ids_sha256":semantic_hash,
                        "semantic_token_count":semantic_ids.len(),
                        "student_logits_width":student_width,
                        "teacher_logits_width":teacher_vocabulary,
                        "student_residual_token_count":student_width-semantic_ids.len(),
                        "teacher_residual_token_count":teacher_vocabulary-semantic_ids.len(),
                        "semantics":"Identical defined semantic token IDs plus one residual category containing every padding output. Full original-head normalisation before coarsening; zero discarded probability mass. Per-temperature residual masses are recorded with each optimisation update."
                    }))
                } else {
                    None
                },
            };
            artefact::write_json(&metadata, &record)?;
            prepared.insert(
                sample.id.clone(),
                PreparedTrace {
                    record,
                    artefact: Artefact::capture(&root, metadata)?,
                },
            );
            eprintln!(
                "distillation: cached native teacher trace {} / {}",
                id, sample.id
            );
        }
        runtime.device().synchronize()?;
    }
    ensure!(
        prepared.len() == samples.len(),
        "teacher target coverage is incomplete"
    );
    Ok(prepared)
}
