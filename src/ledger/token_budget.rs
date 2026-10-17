//! Cumulative token liabilities checked inside the same transaction as money.
use super::*;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenBudgetLimits {
    pub round: Option<u64>,
    pub task: Option<u64>,
    pub provider: Option<u64>,
    pub model: Option<u64>,
}

pub(super) fn check(
    tx: &rusqlite::Transaction<'_>,
    acquisition: &Acquisition,
    limits: TokenBudgetLimits,
) -> Result<Option<Claim>> {
    for (scope, cap, field, id) in [
        (
            "round",
            limits.round,
            "round_id",
            acquisition.round_id.as_str(),
        ),
        ("task", limits.task, "task_id", acquisition.task_id.as_str()),
        (
            "provider",
            limits.provider,
            "provider_id",
            acquisition.provider_id.as_str(),
        ),
        ("model", limits.model, "model", acquisition.model.as_str()),
    ] {
        let Some(cap) = cap else {
            continue;
        };
        ensure!(
            cap <= i64::MAX as u64,
            "token budget exceeds database integer range"
        );
        let requested = acquisition
            .reserved_tokens
            .context("token budget requires a bounded token reservation")?;
        // The only interpolated identifier is from the fixed list above. Model
        // scope is the provider/model pair, matching the monetary budget.
        let sql = format!(
            "SELECT CASE WHEN a.input_tokens IS NOT NULL AND a.output_tokens IS NOT NULL
                THEN a.input_tokens + a.output_tokens ELSE t.reserved_tokens END
            FROM attempts a LEFT JOIN token_reservations t ON t.attempt_id=a.attempt_id
            WHERE a.round_id=?1 AND a.{field}=?2 {}",
            if scope == "model" {
                "AND a.provider_id=?3"
            } else {
                ""
            }
        );
        let mut statement = tx.prepare(&sql)?;
        let mut rows = if scope == "model" {
            statement.query(params![acquisition.round_id, id, acquisition.provider_id])?
        } else {
            statement.query(params![acquisition.round_id, id])?
        };
        let mut used = 0u64;
        while let Some(row) = rows.next()? {
            let liability: Option<u64> = row.get(0)?;
            used = used
                .checked_add(liability.context(
                    "historical token liability is unknown; cannot authorise this token budget",
                )?)
                .context("token budget overflow")?;
        }
        if used.checked_add(requested).is_none_or(|total| total > cap) {
            tx.execute("INSERT INTO events(timestamp_ms,round_id,request_key,kind,detail) VALUES (?1,?2,?3,'token_budget_stop',?4)",
                params![unix_millis(),acquisition.round_id,acquisition.request_key,
                    format!("{scope}: used={used}; requested={requested}; cap={cap}")])?;
            return Ok(Some(Claim::TokenBudgetStopped {
                scope: scope.into(),
                used,
                requested,
                limit: cap,
            }));
        }
    }
    Ok(None)
}
