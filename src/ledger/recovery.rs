//! Conservative reconciliation of the publish-before-database-completion window.
//!
//! Only trusted, immutable local recordings can settle an interrupted attempt.
//! This module has no transport or provider configuration: recovery cannot purchase
//! another response or replace the original price/permission evidence.

use super::*;
use crate::fleet::{Recording, TokenUsage, canonical_json, extract_usage, parse_response};
use std::collections::HashMap;

struct Pending {
    attempt_id: String,
    ordinal: u32,
    round_id: String,
    request_key: String,
    task_id: String,
    source_id: String,
    provider_id: String,
    business_id: String,
    model: String,
    model_revision: String,
    request_json: String,
    price_json: String,
    reserved_nanos: i64,
}

struct Candidate {
    hash: String,
    record: Result<Recording>,
}

struct Settlement {
    success: bool,
    lower: i64,
    upper: i64,
    usage: Option<TokenUsage>,
    reason: Option<String>,
}

pub(super) fn recover(ledger: &Ledger) -> Result<usize> {
    let mut conn = ledger.connection()?;
    let pending = {
        let mut statement = conn.prepare("SELECT attempt_id,ordinal,round_id,request_key,task_id,source_id,provider_id,business_id,model,model_revision,request_json,price_json,reserved_nanos FROM attempts WHERE state='in_flight' ORDER BY attempt_id")?;
        statement
            .query_map([], |row| {
                Ok(Pending {
                    attempt_id: row.get(0)?,
                    ordinal: row.get(1)?,
                    round_id: row.get(2)?,
                    request_key: row.get(3)?,
                    task_id: row.get(4)?,
                    source_id: row.get(5)?,
                    provider_id: row.get(6)?,
                    business_id: row.get(7)?,
                    model: row.get(8)?,
                    model_revision: row.get(9)?,
                    request_json: row.get(10)?,
                    price_json: row.get(11)?,
                    reserved_nanos: row.get(12)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
    };
    if pending.is_empty() {
        return Ok(0);
    }
    // Scan once, before taking the database write lock. The caller must already
    // own orchestration exclusively; no worker may still be publishing responses.
    let (inventory, uninspectable) = scan(ledger, &pending)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut unknown = 0;
    for attempt in &pending {
        let state: String = tx.query_row(
            "SELECT state FROM attempts WHERE attempt_id=?1",
            [&attempt.attempt_id],
            |row| row.get(0),
        )?;
        ensure!(
            state == "in_flight",
            "exclusive recovery ownership was lost"
        );
        let request_state: String = tx.query_row(
            "SELECT state FROM requests WHERE request_key=?1",
            [&attempt.request_key],
            |row| row.get(0),
        )?;
        ensure!(
            request_state == "in_flight",
            "attempt/request recovery states disagree"
        );
        let candidates = inventory.get(&attempt.request_key);
        let recovered = if uninspectable {
            Err(anyhow::anyhow!(
                "recording inventory contains uninspectable evidence"
            ))
        } else if let Some(candidates) = candidates {
            if candidates.len() == 1 {
                let candidate = &candidates[0];
                candidate
                    .record
                    .as_ref()
                    .map_err(|_| anyhow::anyhow!("matching recording is corrupt or invalid"))
                    .and_then(|record| {
                        // Recheck the immutable content immediately before settlement;
                        // a changed or missing blob follows the UNKNOWN path too.
                        ledger.read_artifact(&candidate.hash)?;
                        validate(record, attempt)
                    })
                    .map(|settlement| (candidate, settlement))
            } else {
                Err(anyhow::anyhow!(
                    "multiple recordings match the interrupted key/ordinal"
                ))
            }
        } else {
            Err(anyhow::anyhow!(
                "no durable recording matches the interrupted key/ordinal"
            ))
        };
        match recovered {
            Ok((candidate, settlement)) => {
                // Updates and the recovery event share one FULL-synchronous
                // transaction, so another interruption cannot leave partial work.
                let status = if settlement.success {
                    "completed"
                } else {
                    "failed"
                };
                let (input, cached, output, reasoning) = settlement
                    .usage
                    .as_ref()
                    .map(|u| {
                        (
                            Some(u.input_tokens),
                            Some(u.cached_input_tokens),
                            Some(u.output_tokens),
                            Some(u.reasoning_tokens),
                        )
                    })
                    .unwrap_or((None, None, None, None));
                tx.execute("UPDATE attempts SET state=?2,finished_ms=?3,artifact_sha256=?4,cost_lower_nanos=?5,cost_upper_nanos=?6,reason=?7,input_tokens=?8,cached_input_tokens=?9,output_tokens=?10,reasoning_tokens=?11 WHERE attempt_id=?1 AND state='in_flight'", params![attempt.attempt_id,status,unix_millis(),candidate.hash,settlement.lower,settlement.upper,settlement.reason,input,cached,output,reasoning])?;
                tx.execute("UPDATE requests SET state=?2,completed_attempt_id=?3,artifact_sha256=?4 WHERE request_key=?1 AND state='in_flight'", params![attempt.request_key,status,settlement.success.then_some(&attempt.attempt_id),candidate.hash])?;
                let record = candidate
                    .record
                    .as_ref()
                    .map_err(|_| anyhow::anyhow!("validated recording disappeared"))?;
                let detail = serde_json::json!({"attempt_id":attempt.attempt_id,"artifact_sha256":candidate.hash,"state":status,"cost_lower_nanos":settlement.lower,"cost_upper_nanos":settlement.upper,"origin":record.origin});
                tx.execute("INSERT INTO events(timestamp_ms,round_id,request_key,kind,detail) VALUES (?1,?2,?3,'recovery_reconciled',?4)", params![unix_millis(),attempt.round_id,attempt.request_key,detail.to_string()])?;
            }
            Err(error) => {
                unknown += 1;
                let reason =
                    format!("orchestrator interrupted; remote acceptance UNKNOWN: {error}");
                // Do not change the cost interval or reservation without evidence.
                tx.execute("UPDATE attempts SET state='unknown',finished_ms=?2,reason=?3 WHERE attempt_id=?1 AND state='in_flight'", params![attempt.attempt_id,unix_millis(),reason])?;
                tx.execute("UPDATE requests SET state='unknown' WHERE request_key=?1 AND state='in_flight'", [&attempt.request_key])?;
                tx.execute("INSERT OR IGNORE INTO blocked_rounds(round_id,reason) VALUES (?1,'orchestrator interrupted; remote acceptance UNKNOWN')", [&attempt.round_id])?;
                tx.execute("INSERT INTO events(timestamp_ms,round_id,request_key,kind,detail) VALUES (?1,?2,?3,'recovery_unknown',?4)", params![unix_millis(),attempt.round_id,attempt.request_key,reason])?;
            }
        }
    }
    tx.commit()?;
    Ok(unknown)
}

fn scan(ledger: &Ledger, pending: &[Pending]) -> Result<(HashMap<String, Vec<Candidate>>, bool)> {
    let ordinals: HashMap<_, _> = pending
        .iter()
        .map(|p| (p.request_key.as_str(), p.ordinal))
        .collect();
    ensure!(
        ordinals.len() == pending.len(),
        "multiple interrupted attempts share one request key"
    );
    let mut inventory: HashMap<String, Vec<Candidate>> = HashMap::new();
    let mut uninspectable = false;
    for entry in fs::read_dir(&ledger.artifacts)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue; // A .tmp file was never durably published.
        }
        let inspected = (|| -> Result<(Vec<u8>, serde_json::Value)> {
            ensure!(
                entry.file_type()?.is_file(),
                "recording is not a regular file"
            );
            let bytes = fs::read(&path)?;
            let value = serde_json::from_slice(&bytes)?;
            Ok((bytes, value))
        })();
        let Ok((bytes, value)) = inspected else {
            // Without a readable identity we cannot rule out competing evidence.
            uninspectable = true;
            continue;
        };
        let Some(key) = value.get("request_key").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Some(ordinal) = ordinals.get(key) else {
            continue;
        };
        if let Some(recorded) = value
            .get("attempt_ordinal")
            .and_then(serde_json::Value::as_u64)
            && recorded != u64::from(*ordinal)
        {
            continue; // Earlier retry evidence cannot settle the current attempt.
        }
        let hash = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_owned();
        let record = (|| -> Result<Recording> {
            ensure!(
                hash.len() == 64 && sha256(&bytes) == hash,
                "recording filename/checksum mismatch"
            );
            Ok(serde_json::from_slice(&bytes)?)
        })();
        inventory
            .entry(key.to_owned())
            .or_default()
            .push(Candidate { hash, record });
    }
    Ok((inventory, uninspectable))
}

fn validate(record: &Recording, attempt: &Pending) -> Result<Settlement> {
    record.validate_identity()?;
    ensure!(
        record.request_key == attempt.request_key && record.attempt_ordinal == attempt.ordinal,
        "recording key/ordinal mismatch"
    );
    ensure!(
        record.provider_id == attempt.provider_id
            && record.business_id == attempt.business_id
            && record.model == attempt.model
            && record.model_revision == attempt.model_revision,
        "recording provider identity mismatch"
    );
    ensure!(
        record.request.task_id == attempt.task_id && record.request.source_id == attempt.source_id,
        "recording task/source mismatch"
    );
    let body: serde_json::Value = serde_json::from_str(&attempt.request_json)?;
    let price: serde_json::Value = serde_json::from_str(&attempt.price_json)?;
    ensure!(
        canonical_json(&record.request_body)? == canonical_json(&body)?,
        "recording request body mismatch"
    );
    ensure!(
        canonical_json(&serde_json::to_value(&record.price)?)? == canonical_json(&price)?,
        "recording original price mismatch"
    );
    ensure!(
        (100..=599).contains(&record.response.status),
        "invalid recorded HTTP status"
    );
    let input_bound = u64::try_from(serde_json::to_vec(&body)?.len())?
        .checked_add(1024)
        .context("input bound overflow")?;
    let output_bound = record.request.generation.max_output_tokens;
    ensure!(
        output_bound > 0
            && record.price.reserve(input_bound, output_bound)? == attempt.reserved_nanos,
        "recording maximum reservation mismatch"
    );
    let usage = match extract_usage(&record.response) {
        Ok(usage) => {
            let response: serde_json::Value = serde_json::from_str(&record.response.raw_body)?;
            for pointer in [
                "/usage/prompt_tokens_details/cached_tokens",
                "/usage/prompt_cache_hit_tokens",
                "/usage/completion_tokens_details/reasoning_tokens",
            ] {
                ensure!(
                    response
                        .pointer(pointer)
                        .is_none_or(|v| v.as_u64().is_some()),
                    "recorded optional usage field is malformed"
                );
            }
            let nested_cached = response
                .pointer("/usage/prompt_tokens_details/cached_tokens")
                .and_then(serde_json::Value::as_u64);
            let legacy_cached = response
                .pointer("/usage/prompt_cache_hit_tokens")
                .and_then(serde_json::Value::as_u64);
            ensure!(
                nested_cached.zip(legacy_cached).is_none_or(|(a, b)| a == b),
                "recorded cached usage fields disagree"
            );
            ensure!(
                usage.input_tokens <= input_bound && usage.output_tokens <= output_bound,
                "recorded usage exceeds reserved token bounds"
            );
            Some(usage)
        }
        Err(_) => {
            let body: Option<serde_json::Value> =
                serde_json::from_str(&record.response.raw_body).ok();
            ensure!(
                !body.as_ref().is_some_and(|b| b.get("usage").is_some()),
                "recorded usage is malformed or inconsistent"
            );
            None
        }
    };
    let cost = usage.as_ref().map(|u| record.price.cost(u)).transpose()?;
    let lower = cost.unwrap_or(0);
    let upper = cost.unwrap_or(attempt.reserved_nanos);
    ensure!(
        upper <= attempt.reserved_nanos,
        "recorded cost exceeds reservation"
    );
    let parsed = parse_response(&record.response);
    Ok(Settlement {
        success: parsed.is_ok(),
        lower,
        upper,
        usage,
        reason: parsed
            .err()
            .map(|e| format!("recovered durable failure: {e}")),
    })
}

#[cfg(test)]
mod tests;
