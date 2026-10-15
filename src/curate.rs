//! Fail-closed curation. Metadata is a claim; durable provider and controller
//! artifacts are checked again immediately before training opens an optimiser.
use crate::{
    fleet::{Recording, parse_response},
    sandbox::SandboxReport,
    tasks::{self, HeldoutEntry, Split},
};
use anyhow::{Context, Result, bail, ensure};
use quote::ToTokens;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRow {
    pub task_id: String,
    pub source_id: String,
    pub lineage_id: String,
    pub specification: String,
    pub code: String,
    pub run_id: String,
    pub attempt_id: String,
    pub request_key: String,
    pub provider_id: String,
    pub training_permission_decision_id: String,
    pub training_permission_approved: bool,
    pub real_provider_recording: bool,
    pub passed: bool,
    pub trace_complete: bool,
    pub quarantine_reasons: Vec<String>,
    pub sandbox_report_path: PathBuf,
    pub provider_recording_path: PathBuf,
    pub response_artifact_sha256: String,
    pub attempt_proof_path: PathBuf,
    pub attempt_proof_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub id: String,
    pub task_id: String,
    pub source_id: String,
    pub lineage_id: String,
    pub prompt: String,
    pub completion: String,
    pub specification: String,
    pub run_id: String,
    pub attempt_id: String,
    pub request_key: String,
    pub provider_id: String,
    pub permission_decision_id: String,
    pub specification_sha256: String,
    pub code_sha256: String,
    pub sandbox_evidence: String,
    pub sandbox_evidence_sha256: String,
    pub provider_evidence: String,
    pub provider_evidence_sha256: String,
    pub attempt_evidence: String,
    pub attempt_evidence_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub data_sha256: String,
    pub heldout_index_sha256: String,
    pub sample_count: usize,
    pub source_ids: Vec<String>,
    pub manifest_sha256: String,
}
/// The private field prevents accidental construction from unvalidated rows.
#[allow(
    clippy::manual_non_exhaustive,
    reason = "The private seal prevents bypassing validation inside this crate too; #[non_exhaustive] only restricts external crates"
)]
pub struct ValidatedTrainingSet {
    pub rows: Vec<TrainingSample>,
    pub manifest: DatasetManifest,
    _validated: (),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rejection {
    pub task_id: String,
    pub run_id: String,
    pub reason: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurateReport {
    pub accepted: usize,
    pub rejected: Vec<Rejection>,
    pub dataset_directory: PathBuf,
    pub manifest: DatasetManifest,
}

fn index(path: &Path) -> Result<(Vec<HeldoutEntry>, String)> {
    let bytes = fs::read(path).context("immutable held-out index is required")?;
    let entries: Vec<HeldoutEntry> = serde_json::from_slice(&bytes)?;
    // Comparing with the compiled catalogue prevents swapping in an empty or
    // shortened index just before the optimiser starts.
    let expected: Vec<_> = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::HeldOut)
        .collect();
    ensure!(
        entries.len() == expected.len(),
        "held-out index is incomplete"
    );
    for t in expected {
        let e = entries
            .iter()
            .find(|e| e.task_id == t.id)
            .context("held-out task absent")?;
        ensure!(
            e.source_id == t.source_id
                && e.lineage_id == t.lineage_id
                && e.specification_sha256 == t.specification_sha256
                && e.specification == t.specification,
            "held-out index changed"
        );
    }
    Ok((entries, tasks::hash(bytes)))
}
fn words(text: &str) -> BTreeSet<String> {
    tasks::substantive_specification(text)
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| s.len() > 2)
        .map(str::to_ascii_lowercase)
        .collect()
}
fn similarity(a: &str, b: &str) -> f64 {
    let a = words(a);
    let b = words(b);
    let u = a.union(&b).count();
    let overlap = a.intersection(&b).count();
    if u == 0 {
        0.0
    } else {
        let j = overlap as f64 / u as f64;
        let shorter = a.len().min(b.len());
        if shorter >= 8 && overlap as f64 / shorter as f64 >= 0.95 {
            1.0
        } else {
            j
        }
    }
}
/// Conservative lexical screening supplements exact hashes and lineage. This
/// is not a proof of semantic non-equivalence or a full Rust AST analyser.
fn check_origin(
    task_id: &str,
    source: &str,
    lineage: &str,
    specification: &str,
    heldout: &[HeldoutEntry],
) -> Result<()> {
    for e in heldout {
        if task_id == e.task_id || source == e.source_id || lineage == e.lineage_id {
            bail!("held-out lineage contamination: {}", e.task_id);
        }
        if tasks::hash(specification) == e.specification_sha256
            || similarity(specification, &e.specification) >= 0.80
        {
            bail!(
                "held-out exact/near specification match: {} (review required)",
                e.task_id
            );
        }
    }
    Ok(())
}
struct VerifiedEvidence {
    sandbox: Vec<u8>,
    provider: Vec<u8>,
    audit: Vec<u8>,
    attempt: Vec<u8>,
}

fn validate_evidence(row: &TrainingRow, audit_root: Option<&Path>) -> Result<VerifiedEvidence> {
    ensure!(
        row.passed && row.trace_complete && row.quarantine_reasons.is_empty(),
        "untrusted or quarantined controller verdict"
    );
    ensure!(
        row.training_permission_approved && row.real_provider_recording,
        "fixtures or uncleared provider outputs cannot train"
    );
    let report_bytes = fs::read(&row.sandbox_report_path).context("sandbox evidence missing")?;
    let mut report: SandboxReport = serde_json::from_slice(&report_bytes)?;
    ensure!(
        report.passed
            && !report.quarantined
            && report.trace_complete
            && report.quarantine_reasons.is_empty(),
        "sandbox evidence is not a complete pass"
    );
    ensure!(
        report.task_id == row.task_id
            && report.run_id == row.run_id
            && report.code_sha256 == tasks::hash(&row.code),
        "sandbox evidence does not bind this candidate"
    );
    let task = tasks::corpus()
        .into_iter()
        .find(|t| t.id == row.task_id)
        .context("unknown graded task")?;
    if let Some(root) = audit_root {
        report.audit_path = confined(
            root,
            &format!("evidence/{}.audit.json", report.audit_sha256),
        )?;
    }
    crate::sandbox::verify_report(&report, &report.sandbox_config, &task, &row.code)?;
    let audit_bytes = fs::read(&report.audit_path)?;
    let provider_bytes =
        fs::read(&row.provider_recording_path).context("provider evidence missing")?;
    ensure!(
        tasks::hash(&provider_bytes) == row.response_artifact_sha256,
        "provider artifact checksum mismatch"
    );
    let record: Recording = serde_json::from_slice(&provider_bytes)?;
    record.validate_identity()?;
    ensure!(
        record.origin == "real_provider"
            && record.request_key == row.request_key
            && record.provider_id == row.provider_id,
        "provider evidence origin/key mismatch"
    );
    ensure!(
        record.permission.approved_for_output_training
            && record.permission.decision_id == row.training_permission_decision_id
            && !record.permission.evidence_sha256.is_empty(),
        "provider rights evidence is not approved"
    );
    ensure!(
        !record.business_id.is_empty()
            && !record.model.is_empty()
            && !record.model_revision.is_empty(),
        "provider/model identity missing"
    );
    ensure!(
        record.request.task_id == task.id
            && record.request.source_id == task.source_id
            && record.request.specification_sha256 == task.specification_sha256
            && record.request.prompt == task.prompt(),
        "provider request metadata does not bind canonical task"
    );
    ensure!(
        record.request_body["messages"]
            .as_array()
            .map(|messages| messages
                .iter()
                .any(|m| m["role"] == "user" && m["content"] == task.prompt()))
            .unwrap_or(false),
        "recorded request does not contain the canonical task prompt"
    );
    let (code, _usage) = parse_response(&record.response)?;
    let code = crate::evaluate::extract_code(&code);
    ensure!(
        code == row.code,
        "candidate differs from recorded completion"
    );
    let attempt_bytes =
        fs::read(&row.attempt_proof_path).context("completed attempt proof missing")?;
    ensure!(
        tasks::hash(&attempt_bytes) == row.attempt_proof_sha256,
        "attempt proof checksum mismatch"
    );
    validate_attempt_proof(&attempt_bytes, row, &record)?;
    Ok(VerifiedEvidence {
        sandbox: report_bytes,
        provider: provider_bytes,
        audit: audit_bytes,
        attempt: attempt_bytes,
    })
}
fn canonical_code(code: &str) -> Result<String> {
    let ast = syn::parse_file(code).context("candidate is not valid Rust source")?;
    Ok(ast.to_token_stream().to_string())
}
fn code_similarity(a: &str, b: &str) -> f64 {
    let a: BTreeSet<_> = a.split_whitespace().collect();
    let b: BTreeSet<_> = b.split_whitespace().collect();
    let u = a.union(&b).count();
    if u == 0 {
        0.0
    } else {
        a.intersection(&b).count() as f64 / u as f64
    }
}
fn validate_attempt_proof(bytes: &[u8], row: &TrainingRow, record: &Recording) -> Result<()> {
    let proof: crate::ledger::AttemptProof = serde_json::from_slice(bytes)?;
    ensure!(
        proof.schema_version == 1 && proof.state == "completed" && !proof.round_id.is_empty(),
        "attempt proof is not a completed acquisition"
    );
    ensure!(
        proof.attempt_id == row.attempt_id
            && proof.request_key == row.request_key
            && proof.task_id == row.task_id
            && proof.source_id == row.source_id,
        "attempt proof does not bind the sample/request/source"
    );
    ensure!(
        proof.provider_id == record.provider_id
            && proof.business_id == record.business_id
            && proof.model == record.model
            && proof.model_revision == record.model_revision
            && proof.ordinal == record.attempt_ordinal,
        "attempt proof provider/model/ordinal mismatch"
    );
    ensure!(
        proof.response_artifact_sha256.as_deref() == Some(row.response_artifact_sha256.as_str()),
        "attempt proof response artifact mismatch"
    );
    ensure!(
        serde_json::from_str::<serde_json::Value>(&proof.request_json)? == record.request_body,
        "attempt proof full request body mismatch"
    );
    ensure!(
        serde_json::from_str::<serde_json::Value>(&proof.price_json)?
            == serde_json::to_value(&record.price)?,
        "attempt proof price snapshot mismatch"
    );
    let usage = crate::fleet::extract_usage(&record.response)?;
    ensure!(
        proof.input_tokens == Some(usage.input_tokens)
            && proof.cached_input_tokens == Some(usage.cached_input_tokens)
            && proof.output_tokens == Some(usage.output_tokens)
            && proof.reasoning_tokens == Some(usage.reasoning_tokens),
        "attempt proof usage mismatch"
    );
    let cost = record.price.cost(&usage)?;
    ensure!(
        proof.historical_lower_nanos == cost
            && proof.historical_upper_nanos == cost
            && cost <= proof.reserved_nanos,
        "attempt proof cost/reservation mismatch"
    );
    ensure!(
        proof.started_unix_ms > 0
            && proof
                .finished_unix_ms
                .is_some_and(|end| end >= proof.started_unix_ms),
        "attempt proof completion timestamps missing/invalid"
    );
    Ok(())
}
fn manifest_digest(manifest: &DatasetManifest) -> Result<String> {
    let mut m = manifest.clone();
    m.manifest_sha256.clear();
    Ok(tasks::hash(serde_json::to_vec(&m)?))
}

pub fn curate(rows: &[TrainingRow], output: &Path, heldout_index: &Path) -> Result<CurateReport> {
    fs::create_dir_all(output)?;
    let lease = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(output.join(".curation.lock"))?;
    use std::os::fd::AsRawFd;
    // SAFETY: flock uses a live descriptor and accesses no Rust memory. The
    // kernel releases this exclusive publication lease after interruption.
    ensure!(
        unsafe { libc::flock(lease.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
        "another curator owns this dataset directory"
    );
    ensure!(
        !output.join("manifest.json").exists(),
        "completed curated datasets are immutable; select a fresh output directory"
    );
    let (heldout, index_sha) = index(heldout_index)?;
    fs::create_dir_all(output.join("evidence"))?;
    let catalogue: BTreeMap<_, _> = tasks::corpus()
        .into_iter()
        .map(|t| (t.id.clone(), t))
        .collect();
    let (mut accepted, mut rejected, mut used_tasks, mut used_code, mut selected_ast) = (
        vec![],
        vec![],
        BTreeSet::new(),
        BTreeSet::new(),
        Vec::<String>::new(),
    );
    // Asynchronous acquisition order must not decide the winner or data hash.
    // Only measured ledger-compatible usage/pricing determines the primary
    // tie-break; invalid/missing billing is sorted last and rejected below.
    let mut ranked: Vec<_> = rows
        .iter()
        .map(|row| {
            let measured_cost = (|| -> Result<i64> {
                let record: Recording =
                    serde_json::from_slice(&fs::read(&row.provider_recording_path)?)?;
                record
                    .price
                    .cost(&crate::fleet::extract_usage(&record.response)?)
            })()
            .unwrap_or(i64::MAX);
            (measured_cost, row)
        })
        .collect();
    ranked.sort_by(|(ac, a), (bc, b)| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.task_id.cmp(&b.task_id))
            .then(ac.cmp(bc))
            .then(a.code.len().cmp(&b.code.len()))
            .then(a.request_key.cmp(&b.request_key))
            .then(a.provider_id.cmp(&b.provider_id))
            .then(a.run_id.cmp(&b.run_id))
    });
    for (_, row) in ranked {
        let result = (|| -> Result<TrainingSample> {
            check_origin(
                &row.task_id,
                &row.source_id,
                &row.lineage_id,
                &row.specification,
                &heldout,
            )?;
            let task = catalogue.get(&row.task_id).context("unknown task ID")?;
            ensure!(
                task.split == Split::Train,
                "only training sources are eligible"
            );
            ensure!(
                row.source_id == task.source_id
                    && row.lineage_id == task.lineage_id
                    && row.specification == task.specification,
                "task metadata differs from immutable catalogue"
            );
            ensure!(
                !used_tasks.contains(&row.task_id),
                "a winner was already selected for this task"
            );
            let code_sha = tasks::hash(&row.code);
            ensure!(!used_code.contains(&code_sha), "exact duplicate code");
            let canonical = canonical_code(&row.code)?;
            ensure!(
                !selected_ast
                    .iter()
                    .any(|s| s == &canonical || code_similarity(s, &canonical) >= 0.97),
                "canonical AST-token duplicate/near duplicate (review required)"
            );
            let VerifiedEvidence {
                sandbox,
                provider,
                audit,
                attempt,
            } = validate_evidence(row, None)?;
            let sh = tasks::hash(&sandbox);
            let ph = tasks::hash(&provider);
            let sp = format!("evidence/{sh}.sandbox.json");
            let pp = format!("evidence/{ph}.provider.json");
            let ah = tasks::hash(&attempt);
            let ap = format!("evidence/{ah}.attempt.json");
            fs::write(output.join(&sp), sandbox)?;
            fs::write(output.join(&pp), provider)?;
            fs::write(output.join(&ap), attempt)?;
            fs::write(
                output.join(format!("evidence/{}.audit.json", tasks::hash(&audit))),
                audit,
            )?;
            Ok(TrainingSample {
                id: tasks::hash(format!("{}:{}", row.run_id, code_sha)),
                task_id: row.task_id.clone(),
                source_id: row.source_id.clone(),
                lineage_id: row.lineage_id.clone(),
                prompt: task.prompt(),
                completion: row.code.clone(),
                specification: row.specification.clone(),
                run_id: row.run_id.clone(),
                attempt_id: row.attempt_id.clone(),
                request_key: row.request_key.clone(),
                provider_id: row.provider_id.clone(),
                permission_decision_id: row.training_permission_decision_id.clone(),
                specification_sha256: task.specification_sha256.clone(),
                code_sha256: code_sha,
                sandbox_evidence: sp,
                sandbox_evidence_sha256: sh,
                provider_evidence: pp,
                provider_evidence_sha256: ph,
                attempt_evidence: ap,
                attempt_evidence_sha256: ah,
            })
        })();
        match result {
            Ok(sample) => {
                used_tasks.insert(sample.task_id.clone());
                used_code.insert(sample.code_sha256.clone());
                selected_ast.push(canonical_code(&sample.completion)?);
                accepted.push(sample);
            }
            Err(e) => rejected.push(Rejection {
                task_id: row.task_id.clone(),
                run_id: row.run_id.clone(),
                reason: format!("{e:#}"),
            }),
        }
    }
    ensure!(
        !accepted.is_empty(),
        "no eligible real-provider training samples; {} rows rejected",
        rejected.len()
    );
    accepted.sort_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then(a.task_id.cmp(&b.task_id))
            .then(a.id.cmp(&b.id))
    });
    let mut data = Vec::new();
    for row in &accepted {
        serde_json::to_writer(&mut data, row)?;
        data.push(b'\n');
    }
    let mut manifest = DatasetManifest {
        schema_version: 1,
        data_sha256: tasks::hash(&data),
        heldout_index_sha256: index_sha,
        sample_count: accepted.len(),
        source_ids: accepted
            .iter()
            .map(|r| r.source_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = manifest_digest(&manifest)?;
    fs::write(output.join("dataset.jsonl"), data)?;
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let report = CurateReport {
        accepted: accepted.len(),
        rejected,
        dataset_directory: output.to_path_buf(),
        manifest,
    };
    fs::write(
        output.join("curation-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    validate_dataset(output, heldout_index)?;
    Ok(report)
}

fn confined(root: &Path, relative: &str) -> Result<PathBuf> {
    let p = Path::new(relative);
    ensure!(
        !p.is_absolute()
            && p.components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
        "evidence path escapes dataset"
    );
    let root = root.canonicalize()?;
    let out = root.join(p).canonicalize()?;
    ensure!(out.starts_with(root), "evidence symlink escapes dataset");
    Ok(out)
}

pub fn validate_dataset(path: &Path, heldout_index: &Path) -> Result<ValidatedTrainingSet> {
    let (heldout, index_sha) = index(heldout_index)?;
    let manifest: DatasetManifest = serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
    ensure!(
        manifest.schema_version == 1 && manifest.heldout_index_sha256 == index_sha,
        "dataset version/index hash mismatch"
    );
    ensure!(
        manifest.manifest_sha256 == manifest_digest(&manifest)?,
        "dataset manifest checksum mismatch"
    );
    let bytes = fs::read(path.join("dataset.jsonl"))?;
    ensure!(
        tasks::hash(&bytes) == manifest.data_sha256,
        "training data checksum mismatch"
    );
    let rows: Vec<TrainingSample> = std::str::from_utf8(&bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    ensure!(
        !rows.is_empty() && rows.len() == manifest.sample_count,
        "empty/incomplete training dataset"
    );
    let catalogue: BTreeMap<_, _> = tasks::corpus()
        .into_iter()
        .map(|t| (t.id.clone(), t))
        .collect();
    let (mut ids, mut code_hashes, mut asts) =
        (BTreeSet::new(), BTreeSet::new(), Vec::<String>::new());
    for sample in &rows {
        check_origin(
            &sample.task_id,
            &sample.source_id,
            &sample.lineage_id,
            &sample.specification,
            &heldout,
        )?;
        let task = catalogue
            .get(&sample.task_id)
            .context("unknown sample task")?;
        ensure!(
            task.split == Split::Train
                && task.source_id == sample.source_id
                && task.lineage_id == sample.lineage_id
                && task.specification == sample.specification,
            "non-training or altered source"
        );
        ensure!(
            sample.prompt == task.prompt()
                && sample.specification_sha256 == task.specification_sha256
                && tasks::hash(&sample.completion) == sample.code_sha256,
            "training text/hash mismatch"
        );
        ensure!(
            ids.insert(&sample.id) && code_hashes.insert(&sample.code_sha256),
            "duplicate sample/code"
        );
        let ast = canonical_code(&sample.completion)?;
        ensure!(
            !asts
                .iter()
                .any(|a| a == &ast || code_similarity(a, &ast) >= 0.97),
            "canonical AST-token duplicate/near duplicate"
        );
        asts.push(ast);
        let sp = confined(path, &sample.sandbox_evidence)?;
        let pp = confined(path, &sample.provider_evidence)?;
        let ap = confined(path, &sample.attempt_evidence)?;
        ensure!(
            tasks::hash(fs::read(&sp)?) == sample.sandbox_evidence_sha256
                && tasks::hash(fs::read(&pp)?) == sample.provider_evidence_sha256,
            "training provenance checksum mismatch"
        );
        validate_evidence(
            &TrainingRow {
                task_id: sample.task_id.clone(),
                source_id: sample.source_id.clone(),
                lineage_id: sample.lineage_id.clone(),
                specification: sample.specification.clone(),
                code: sample.completion.clone(),
                run_id: sample.run_id.clone(),
                attempt_id: sample.attempt_id.clone(),
                request_key: sample.request_key.clone(),
                provider_id: sample.provider_id.clone(),
                training_permission_decision_id: sample.permission_decision_id.clone(),
                training_permission_approved: true,
                real_provider_recording: true,
                passed: true,
                trace_complete: true,
                quarantine_reasons: vec![],
                sandbox_report_path: sp,
                provider_recording_path: pp,
                response_artifact_sha256: sample.provider_evidence_sha256.clone(),
                attempt_proof_path: ap,
                attempt_proof_sha256: sample.attempt_evidence_sha256.clone(),
            },
            Some(path),
        )?;
    }
    ensure!(
        rows.iter()
            .map(|r| r.source_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            == manifest.source_ids,
        "dataset source manifest mismatch"
    );
    Ok(ValidatedTrainingSet {
        rows,
        manifest,
        _validated: (),
    })
}

/// This fixture tests contamination handling only; it never constructs a real
/// provider sample or opens an optimiser.
pub fn contamination_drill() -> Result<serde_json::Value> {
    let heldout: Vec<_> = tasks::corpus()
        .into_iter()
        .filter(|t| t.split == Split::HeldOut)
        .map(|t| HeldoutEntry {
            task_id: t.id,
            source_id: t.source_id,
            lineage_id: t.lineage_id,
            specification: t.specification,
            specification_sha256: t.specification_sha256,
        })
        .collect();
    let e = &heldout[0];
    let temporary = tempfile::tempdir()?;
    let index_path = temporary.path().join("heldout-index.json");
    fs::write(&index_path, serde_json::to_vec_pretty(&heldout)?)?;
    let (_, index_sha) = index(&index_path)?;
    let rejection = |name: &str,
                     task_id: &str,
                     source: &str,
                     lineage: &str,
                     specification: &str|
     -> Result<bool> {
        let directory = temporary.path().join(name);
        fs::create_dir_all(&directory)?;
        let sample = TrainingSample {
            id: format!("labelled-contamination-fixture-{name}"),
            task_id: task_id.into(),
            source_id: source.into(),
            lineage_id: lineage.into(),
            prompt: "deliberately invalid fixture: no provider claim".into(),
            completion:
                "pub fn solve(_:serde_json::Value)->serde_json::Value{serde_json::Value::Null}"
                    .into(),
            specification: specification.into(),
            run_id: "fixture-no-run".into(),
            attempt_id: "fixture-no-call".into(),
            request_key: "fixture-no-key".into(),
            provider_id: "fixture-no-provider".into(),
            permission_decision_id: "fixture-no-rights".into(),
            specification_sha256: tasks::hash(specification),
            code_sha256: String::new(),
            sandbox_evidence: String::new(),
            sandbox_evidence_sha256: String::new(),
            provider_evidence: String::new(),
            provider_evidence_sha256: String::new(),
            attempt_evidence: String::new(),
            attempt_evidence_sha256: String::new(),
        };
        let mut data = serde_json::to_vec(&sample)?;
        data.push(b'\n');
        let mut manifest = DatasetManifest {
            schema_version: 1,
            data_sha256: tasks::hash(&data),
            heldout_index_sha256: index_sha.clone(),
            sample_count: 1,
            source_ids: vec![source.into()],
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = manifest_digest(&manifest)?;
        fs::write(directory.join("dataset.jsonl"), data)?;
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        // Exercise the exact entry point called immediately before optimisation.
        // Check its reason, so an unrelated absent-evidence error cannot pass.
        match validate_dataset(&directory, &index_path) {
            Ok(_) => Ok(false),
            Err(error) => Ok(format!("{error:#}").contains("held-out")),
        }
    };
    let exact = rejection(
        "exact",
        &e.task_id,
        &e.source_id,
        &e.lineage_id,
        &e.specification,
    )?;
    let reworded = rejection(
        "reworded",
        "planted-renamed",
        "planted-source",
        &e.lineage_id,
        "A bounding rectangle around a collection of planar locations.",
    )?;
    let near = rejection(
        "near",
        "planted-near",
        "planted-source",
        "planted-lineage",
        &format!("{} Please implement carefully.", e.specification),
    )?;
    ensure!(exact && reworded && near, "contamination drill failed");
    Ok(
        serde_json::json!({"evidence_kind":"labelled_contamination_fixture","passed":true,"exact_rejected":exact,"reworded_lineage_rejected":reworded,"near_specification_rejected":near,"optimiser_steps":0}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_dataset_cannot_be_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("curated");
        fs::create_dir_all(&output).unwrap();
        fs::write(
            output.join("manifest.json"),
            b"preserved immutable manifest",
        )
        .unwrap();
        assert!(curate(&[], &output, &directory.path().join("unused-index.json")).is_err());
        assert_eq!(
            fs::read(output.join("manifest.json")).unwrap(),
            b"preserved immutable manifest"
        );
        assert!(!output.join("dataset.jsonl").exists());
    }
    #[test]
    fn lineage_survives_renaming() {
        let r = contamination_drill().unwrap();
        assert_eq!(r["optimiser_steps"], 0);
    }
    #[test]
    fn empty_and_modified_index_fail_closed() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("index.json"), b"[]").unwrap();
        assert!(validate_dataset(d.path(), &d.path().join("index.json")).is_err());
    }
    #[test]
    fn evidence_paths_cannot_escape() {
        let d = tempfile::tempdir().unwrap();
        assert!(confined(d.path(), "../provider.json").is_err());
        assert!(confined(d.path(), "/tmp/provider.json").is_err());
    }
    #[test]
    fn formatting_and_comments_do_not_evade_ast_deduplication() {
        let a = "pub fn solve(input:serde_json::Value)->serde_json::Value{input}";
        let b = "// deliberately reformatted\npub fn solve ( input : serde_json::Value ) -> serde_json::Value { input }";
        assert_eq!(canonical_code(a).unwrap(), canonical_code(b).unwrap());
    }
}
