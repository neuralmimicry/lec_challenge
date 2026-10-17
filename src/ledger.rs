//! Durable acquisition ledger. Reservations are liabilities until evidence settles them.
//!
//! Monetary amounts use integer USD nanodollars. A transport timeout is not evidence
//! of a free call, and recovery therefore never automatically releases its reservation.

use anyhow::{Context, Result, bail, ensure};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

mod recovery;
mod token_budget;
pub use token_budget::TokenBudgetLimits;

pub fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
pub fn unix_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetLimits {
    pub round_nanos: i64,
    pub task_nanos: i64,
    pub provider_nanos: i64,
    pub model_nanos: i64,
    #[serde(default)]
    pub tokens: TokenBudgetLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Acquisition {
    pub round_id: String,
    pub task_id: String,
    pub source_id: String,
    pub request_key: String,
    pub provider_id: String,
    pub business_id: String,
    pub model: String,
    pub model_revision: String,
    pub request_json: String,
    pub price_json: String,
    pub reserved_nanos: i64,
    #[serde(default)]
    pub reserved_tokens: Option<u64>,
    pub replay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptClaim {
    pub attempt_id: String,
    pub ordinal: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Claim {
    Reserved(AttemptClaim),
    Cached {
        attempt_id: String,
        artifact_sha256: String,
    },
    InFlight,
    Unknown,
    BudgetStopped {
        scope: String,
        used_nanos: i64,
        requested_nanos: i64,
        limit_nanos: i64,
    },
    TokenBudgetStopped {
        scope: String,
        used: u64,
        requested: u64,
        limit: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostRow {
    #[serde(default)]
    pub reserved_tokens: Option<u64>,
    #[serde(default)]
    pub token_upper_bound: Option<u64>,
    pub attempt_id: String,
    pub round_id: String,
    pub task_id: String,
    pub provider_id: String,
    pub business_id: String,
    pub model: String,
    pub state: String,
    pub ordinal: u32,
    pub historical_lower_nanos: i64,
    pub historical_upper_nanos: i64,
    pub new_api_lower_nanos: i64,
    pub new_api_upper_nanos: i64,
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub artifact_sha256: Option<String>,
    pub reason: Option<String>,
    #[serde(default)]
    pub request_key: String,
    #[serde(default)]
    pub replayed: bool,
}

/// Immutable export of facts read from the trusted acquisition transaction.
/// This is copied into curated evidence so a sample's UUID is a checked link,
/// rather than an arbitrary nonempty provenance string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptProof {
    pub schema_version: u32,
    pub attempt_id: String,
    pub ordinal: u32,
    pub round_id: String,
    pub task_id: String,
    pub source_id: String,
    pub request_key: String,
    pub provider_id: String,
    pub business_id: String,
    pub model: String,
    pub model_revision: String,
    pub state: String,
    pub request_json: String,
    pub price_json: String,
    pub response_artifact_sha256: Option<String>,
    pub replayed: bool,
    pub reserved_nanos: i64,
    pub historical_lower_nanos: i64,
    pub historical_upper_nanos: i64,
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
    pub started_unix_ms: i64,
    pub finished_unix_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedCompletion {
    pub task_id: String,
    pub source_id: String,
    pub specification_sha256: String,
    pub sample_slot: u32,
    pub preferred_provider_id: String,
    pub allowed_request_keys: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundPlan {
    pub schema_version: u32,
    pub round_id: String,
    pub configuration_sha256: String,
    pub jobs: Vec<PlannedCompletion>,
}
impl RoundPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && !self.round_id.is_empty()
                && self.configuration_sha256.len() == 64
                && !self.jobs.is_empty(),
            "invalid acquisition round plan"
        );
        let mut samples = std::collections::HashSet::new();
        let mut keys = std::collections::HashSet::new();
        for job in &self.jobs {
            ensure!(
                !job.task_id.is_empty()
                    && !job.source_id.is_empty()
                    && job.specification_sha256.len() == 64
                    && !job.preferred_provider_id.is_empty()
                    && !job.allowed_request_keys.is_empty(),
                "incomplete acquisition job"
            );
            ensure!(
                samples.insert((&job.task_id, job.sample_slot)),
                "duplicate logical task/sample slot in acquisition plan"
            );
            for key in &job.allowed_request_keys {
                ensure!(
                    key.len() == 64
                        && key.bytes().all(|c| c.is_ascii_hexdigit())
                        && keys.insert(key),
                    "invalid or overlapping request keys in acquisition plan"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEvent {
    pub timestamp_ms: i64,
    pub round_id: String,
    pub request_key: Option<String>,
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Ledger {
    database: PathBuf,
    artifacts: PathBuf,
}

impl Ledger {
    pub fn open(database: impl AsRef<Path>, artifacts: impl AsRef<Path>) -> Result<Self> {
        let ledger = Self {
            database: database.as_ref().to_owned(),
            artifacts: artifacts.as_ref().to_owned(),
        };
        if let Some(parent) = ledger.database.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(&ledger.artifacts)?;
        let conn = ledger.connection()?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS requests (
              request_key TEXT PRIMARY KEY, state TEXT NOT NULL,
              completed_attempt_id TEXT, artifact_sha256 TEXT);
            CREATE TABLE IF NOT EXISTS attempts (
              attempt_id TEXT PRIMARY KEY, request_key TEXT NOT NULL REFERENCES requests(request_key),
              ordinal INTEGER NOT NULL, round_id TEXT NOT NULL, task_id TEXT NOT NULL,
              source_id TEXT NOT NULL, provider_id TEXT NOT NULL, business_id TEXT NOT NULL,
              model TEXT NOT NULL, model_revision TEXT NOT NULL,
              request_json TEXT NOT NULL, price_json TEXT NOT NULL,
              reserved_nanos INTEGER NOT NULL CHECK(reserved_nanos >= 0),
              cost_lower_nanos INTEGER NOT NULL, cost_upper_nanos INTEGER NOT NULL,
              replay INTEGER NOT NULL, state TEXT NOT NULL, started_ms INTEGER NOT NULL,
              finished_ms INTEGER, artifact_sha256 TEXT, reason TEXT,
              input_tokens INTEGER, cached_input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER,
              UNIQUE(request_key, ordinal));
            CREATE INDEX IF NOT EXISTS attempts_round ON attempts(round_id);
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY, timestamp_ms INTEGER NOT NULL, round_id TEXT NOT NULL,
              request_key TEXT, kind TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS blocked_rounds (round_id TEXT PRIMARY KEY, reason TEXT NOT NULL);")?;
        conn.execute_batch("CREATE TABLE IF NOT EXISTS round_plans (round_id TEXT PRIMARY KEY,plan_json TEXT NOT NULL,plan_sha256 TEXT NOT NULL,declared_ms INTEGER NOT NULL);")?;
        // A separate additive table preserves historical databases unchanged.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS token_reservations (
            attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
            reserved_tokens INTEGER NOT NULL CHECK(reserved_tokens >= 0));",
        )?;
        Ok(ledger)
    }

    fn connection(&self) -> Result<Connection> {
        let conn = Connection::open(&self.database).context("open acquisition ledger")?;
        conn.busy_timeout(std::time::Duration::from_secs(15))?;
        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")?;
        Ok(conn)
    }

    /// Check completed/ambiguous work before credentials, throttling or failover.
    /// `claim_and_reserve` repeats this under its write transaction to close races.
    pub fn lookup_request(&self, request_key: &str) -> Result<Option<Claim>> {
        let conn = self.connection()?;
        let found:Option<(String,Option<String>,Option<String>)>=conn.query_row("SELECT state,completed_attempt_id,artifact_sha256 FROM requests WHERE request_key=?1",[request_key],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        match found {
            Some((state, attempt, artifact)) if state == "completed" => Ok(Some(Claim::Cached {
                attempt_id: attempt.context("completed request lacks attempt")?,
                artifact_sha256: artifact.context("completed request lacks artifact")?,
            })),
            Some((state, _, _)) if state == "in_flight" => Ok(Some(Claim::InFlight)),
            Some((state, _, _)) if state == "unknown" => Ok(Some(Claim::Unknown)),
            Some((state, _, _)) if state == "invalid" => {
                bail!("request is invalid and requires reconciliation")
            }
            _ => Ok(None),
        }
    }

    pub fn record_round_plan(&self, plan: &RoundPlan) -> Result<()> {
        plan.validate()?;
        let payload = serde_json::to_string(plan)?;
        let digest = sha256(payload.as_bytes());
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT plan_sha256 FROM round_plans WHERE round_id=?1",
                [&plan.round_id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(existing) = existing {
            ensure!(
                existing == digest,
                "acquisition round plan changed; use a new explicitly declared round"
            );
        } else {
            let attempts: i64 = tx.query_row(
                "SELECT COUNT(*) FROM attempts WHERE round_id=?1",
                [&plan.round_id],
                |r| r.get(0),
            )?;
            ensure!(
                attempts == 0,
                "round plan must be declared before the first acquisition attempt"
            );
            tx.execute("INSERT INTO round_plans(round_id,plan_json,plan_sha256,declared_ms) VALUES (?1,?2,?3,?4)",params![plan.round_id,payload,digest,unix_millis()])?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn round_plan(&self, round: &str) -> Result<Option<RoundPlan>> {
        let value: Option<(String, String)> = self
            .connection()?
            .query_row(
                "SELECT plan_json,plan_sha256 FROM round_plans WHERE round_id=?1",
                [round],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        value
            .map(|(payload, digest)| {
                ensure!(
                    sha256(payload.as_bytes()) == digest,
                    "acquisition plan checksum mismatch"
                );
                let plan: RoundPlan = serde_json::from_str(&payload)?;
                plan.validate()?;
                Ok(plan)
            })
            .transpose()
    }

    pub fn attempt_proof(&self, attempt_id: &str) -> Result<AttemptProof> {
        let proof=self.connection()?.query_row("SELECT attempt_id,ordinal,round_id,task_id,source_id,request_key,provider_id,business_id,model,model_revision,state,request_json,price_json,artifact_sha256,replay,reserved_nanos,cost_lower_nanos,cost_upper_nanos,input_tokens,cached_input_tokens,output_tokens,reasoning_tokens,started_ms,finished_ms FROM attempts WHERE attempt_id=?1",[attempt_id],|row|Ok(AttemptProof{
            schema_version:1,attempt_id:row.get(0)?,ordinal:row.get(1)?,round_id:row.get(2)?,task_id:row.get(3)?,source_id:row.get(4)?,request_key:row.get(5)?,provider_id:row.get(6)?,business_id:row.get(7)?,model:row.get(8)?,model_revision:row.get(9)?,state:row.get(10)?,request_json:row.get(11)?,price_json:row.get(12)?,response_artifact_sha256:row.get(13)?,replayed:row.get(14)?,reserved_nanos:row.get(15)?,historical_lower_nanos:row.get(16)?,historical_upper_nanos:row.get(17)?,input_tokens:row.get(18)?,cached_input_tokens:row.get(19)?,output_tokens:row.get(20)?,reasoning_tokens:row.get(21)?,started_unix_ms:row.get(22)?,finished_unix_ms:row.get(23)?,
        })).context("acquisition attempt proof is missing from trusted ledger")?;
        if let Some(hash) = &proof.response_artifact_sha256 {
            self.read_artifact(hash)?;
        }
        Ok(proof)
    }

    /// Atomically claims a semantic request and reserves every applicable scope.
    pub fn claim_and_reserve(
        &self,
        acquisition: &Acquisition,
        limits: BudgetLimits,
    ) -> Result<Claim> {
        ensure!(acquisition.reserved_nanos >= 0, "negative reservation");
        ensure!(
            [
                limits.round_nanos,
                limits.task_nanos,
                limits.provider_nanos,
                limits.model_nanos
            ]
            .iter()
            .all(|n| *n >= 0),
            "negative budget"
        );
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<(String, Option<String>, Option<String>)> = tx.query_row(
            "SELECT state, completed_attempt_id, artifact_sha256 FROM requests WHERE request_key=?1",
            [&acquisition.request_key], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
        match existing {
            Some((state, attempt, artifact)) if state == "completed" => {
                return Ok(Claim::Cached {
                    attempt_id: attempt.context("completed request lacks attempt")?,
                    artifact_sha256: artifact
                        .context("completed request lacks response artifact")?,
                });
            }
            Some((state, _, _)) if state == "in_flight" => return Ok(Claim::InFlight),
            Some((state, _, _)) if state == "unknown" => return Ok(Claim::Unknown),
            Some((state, _, _)) if state == "invalid" => {
                bail!("request is invalid and requires reconciliation")
            }
            _ => {}
        }
        let blocked: Option<String> = tx
            .query_row(
                "SELECT reason FROM blocked_rounds WHERE round_id=?1",
                [&acquisition.round_id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(reason) = blocked {
            bail!("round blocked: {reason}");
        }
        let declared: Option<(String, String)> = tx
            .query_row(
                "SELECT plan_json,plan_sha256 FROM round_plans WHERE round_id=?1",
                [&acquisition.round_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((payload, digest)) = declared {
            ensure!(
                sha256(payload.as_bytes()) == digest,
                "round plan checksum mismatch before reservation"
            );
            let plan: RoundPlan = serde_json::from_str(&payload)?;
            plan.validate()?;
            ensure!(
                plan.jobs
                    .iter()
                    .any(|job| job.task_id == acquisition.task_id
                        && job.source_id == acquisition.source_id
                        && job.allowed_request_keys.contains(&acquisition.request_key)),
                "unplanned request refused before acquisition reservation"
            );
        }
        let scopes = [
            (
                "round",
                "SELECT COALESCE(SUM(cost_upper_nanos),0) FROM attempts WHERE round_id=?1",
                limits.round_nanos,
                None,
            ),
            (
                "task",
                "SELECT COALESCE(SUM(cost_upper_nanos),0) FROM attempts WHERE round_id=?1 AND task_id=?2",
                limits.task_nanos,
                Some(acquisition.task_id.as_str()),
            ),
            (
                "provider",
                "SELECT COALESCE(SUM(cost_upper_nanos),0) FROM attempts WHERE round_id=?1 AND provider_id=?2",
                limits.provider_nanos,
                Some(acquisition.provider_id.as_str()),
            ),
            (
                "model",
                "SELECT COALESCE(SUM(cost_upper_nanos),0) FROM attempts WHERE round_id=?1 AND provider_id=?2 AND model=?3",
                limits.model_nanos,
                Some(""),
            ),
        ];
        for (scope, sql, cap, identifier) in scopes {
            let used: i64 = match identifier {
                None => tx.query_row(sql, [&acquisition.round_id], |r| r.get(0))?,
                Some(_) if scope == "model" => tx.query_row(
                    sql,
                    params![
                        acquisition.round_id,
                        acquisition.provider_id,
                        acquisition.model
                    ],
                    |r| r.get(0),
                )?,
                Some(id) => tx.query_row(sql, params![acquisition.round_id, id], |r| r.get(0))?,
            };
            if used
                .checked_add(acquisition.reserved_nanos)
                .is_none_or(|n| n > cap)
            {
                tx.execute("INSERT INTO events(timestamp_ms,round_id,request_key,kind,detail) VALUES (?1,?2,?3,'budget_stop',?4)", params![unix_millis(),acquisition.round_id,acquisition.request_key,format!("{scope}: used={used}; requested={}; cap={cap}",acquisition.reserved_nanos)])?;
                tx.commit()?;
                return Ok(Claim::BudgetStopped {
                    scope: scope.to_owned(),
                    used_nanos: used,
                    requested_nanos: acquisition.reserved_nanos,
                    limit_nanos: cap,
                });
            }
        }
        if let Some(stop) = token_budget::check(&tx, acquisition, limits.tokens)? {
            tx.commit()?;
            return Ok(stop);
        }
        let ordinal: u32 = tx.query_row(
            "SELECT COALESCE(MAX(ordinal),0)+1 FROM attempts WHERE request_key=?1",
            [&acquisition.request_key],
            |r| r.get(0),
        )?;
        let attempt_id = uuid::Uuid::new_v4().to_string();
        tx.execute("INSERT INTO requests(request_key,state) VALUES (?1,'in_flight') ON CONFLICT(request_key) DO UPDATE SET state='in_flight'", [&acquisition.request_key])?;
        tx.execute("INSERT INTO attempts(attempt_id,request_key,ordinal,round_id,task_id,source_id,provider_id,business_id,model,model_revision,request_json,price_json,reserved_nanos,cost_lower_nanos,cost_upper_nanos,replay,state,started_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,0,?13,?14,'in_flight',?15)",
          params![attempt_id, acquisition.request_key, ordinal, acquisition.round_id, acquisition.task_id, acquisition.source_id, acquisition.provider_id, acquisition.business_id, acquisition.model, acquisition.model_revision, acquisition.request_json, acquisition.price_json, acquisition.reserved_nanos, acquisition.replay, unix_millis()])?;
        if let Some(tokens) = acquisition.reserved_tokens {
            let tokens = i64::try_from(tokens)
                .context("token reservation exceeds database integer range")?;
            tx.execute(
                "INSERT INTO token_reservations(attempt_id,reserved_tokens) VALUES (?1,?2)",
                params![attempt_id, tokens],
            )?;
        }
        tx.commit()?;
        Ok(Claim::Reserved(AttemptClaim {
            attempt_id,
            ordinal,
        }))
    }

    /// Store and fsync the immutable response before a completion can reference it.
    pub fn publish_artifact(&self, bytes: &[u8]) -> Result<String> {
        let hash = sha256(bytes);
        let target = self.artifacts.join(format!("{hash}.json"));
        if target.exists() {
            ensure!(
                sha256(&fs::read(&target)?) == hash,
                "corrupt existing artifact"
            );
            return Ok(hash);
        }
        let temporary = self
            .artifacts
            .join(format!(".{}.tmp", uuid::Uuid::new_v4()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, &target)?;
        File::open(&self.artifacts)?.sync_all()?;
        Ok(hash)
    }

    pub fn read_artifact(&self, hash: &str) -> Result<Vec<u8>> {
        ensure!(
            hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid artifact hash"
        );
        let bytes = fs::read(self.artifacts.join(format!("{hash}.json")))
            .context("missing durable response artifact")?;
        ensure!(
            sha256(&bytes) == hash,
            "response artifact checksum mismatch"
        );
        Ok(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &self,
        attempt_id: &str,
        artifact: Option<&str>,
        success: bool,
        lower: i64,
        upper: i64,
        usage: Option<(u64, u64, u64, u64)>,
        reason: Option<&str>,
    ) -> Result<()> {
        ensure!(lower >= 0 && upper >= lower, "invalid cost interval");
        if let Some(hash) = artifact {
            self.read_artifact(hash)?;
        }
        ensure!(
            !success || artifact.is_some(),
            "completion requires durable response"
        );
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (key, round, reserved, state): (String, String, i64, String) = tx.query_row(
            "SELECT request_key,round_id,reserved_nanos,state FROM attempts WHERE attempt_id=?1",
            [attempt_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        ensure!(state == "in_flight", "attempt already settled: {state}");
        let discrepancy = upper > reserved;
        let request_state = if discrepancy {
            "invalid"
        } else if success {
            "completed"
        } else {
            "failed"
        };
        let attempt_state = if discrepancy {
            "billing_discrepancy"
        } else if success {
            "completed"
        } else {
            "failed"
        };
        let (input, cached, output, reasoning) = usage
            .map(|(a, b, c, d)| (Some(a), Some(b), Some(c), Some(d)))
            .unwrap_or((None, None, None, None));
        tx.execute("UPDATE attempts SET state=?2,finished_ms=?3,artifact_sha256=?4,cost_lower_nanos=?5,cost_upper_nanos=?6,reason=?7,input_tokens=?8,cached_input_tokens=?9,output_tokens=?10,reasoning_tokens=?11 WHERE attempt_id=?1", params![attempt_id,attempt_state,unix_millis(),artifact,lower,upper,reason,input,cached,output,reasoning])?;
        tx.execute("UPDATE requests SET state=?2,completed_attempt_id=?3,artifact_sha256=?4 WHERE request_key=?1", params![key,request_state,if success && !discrepancy {Some(attempt_id)} else {None},artifact])?;
        if discrepancy {
            tx.execute("INSERT OR REPLACE INTO blocked_rounds(round_id,reason) VALUES (?1,'reported billing exceeded maximum reservation')", [&round])?;
        }
        tx.commit()?;
        ensure!(
            !discrepancy,
            "reported billing exceeds reservation; round stopped for reconciliation"
        );
        Ok(())
    }

    pub fn mark_unknown(&self, attempt_id: &str, reason: &str) -> Result<()> {
        let mut conn = self.connection()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (key, round): (String, String) = tx.query_row(
            "SELECT request_key,round_id FROM attempts WHERE attempt_id=?1 AND state='in_flight'",
            [attempt_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        tx.execute(
            "UPDATE attempts SET state='unknown',finished_ms=?2,reason=?3 WHERE attempt_id=?1",
            params![attempt_id, unix_millis(), reason],
        )?;
        tx.execute(
            "UPDATE requests SET state='unknown' WHERE request_key=?1",
            [key],
        )?;
        tx.execute("INSERT OR IGNORE INTO blocked_rounds(round_id,reason) VALUES (?1,'remote acceptance UNKNOWN; reconciliation required')",[round])?;
        tx.commit()?;
        Ok(())
    }

    /// Run only after obtaining exclusive orchestration ownership, never per worker.
    /// Reconcile durable orphan recordings locally and return the number left UNKNOWN.
    pub fn recover_interrupted(&self) -> Result<usize> {
        recovery::recover(self)
    }

    pub fn event(&self, round: &str, key: Option<&str>, kind: &str, detail: &str) -> Result<()> {
        self.connection()?.execute("INSERT INTO events(timestamp_ms,round_id,request_key,kind,detail) VALUES (?1,?2,?3,?4,?5)", params![unix_millis(),round,key,kind,detail])?;
        Ok(())
    }

    pub fn events(&self, round: Option<&str>) -> Result<Vec<LedgerEvent>> {
        let conn = self.connection()?;
        let mut statement=conn.prepare("SELECT timestamp_ms,round_id,request_key,kind,detail FROM events WHERE (?1 IS NULL OR round_id=?1) ORDER BY id")?;
        let rows = statement.query_map([round], |row| {
            Ok(LedgerEvent {
                timestamp_ms: row.get(0)?,
                round_id: row.get(1)?,
                request_key: row.get(2)?,
                kind: row.get(3)?,
                detail: row.get(4)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn block_round(&self, round: &str, reason: &str) -> Result<()> {
        self.connection()?.execute(
            "INSERT OR REPLACE INTO blocked_rounds(round_id,reason) VALUES (?1,?2)",
            params![round, reason],
        )?;
        Ok(())
    }

    pub fn costs(&self, round: Option<&str>) -> Result<Vec<CostRow>> {
        let conn = self.connection()?;
        let mut statement=conn.prepare("SELECT attempt_id,round_id,task_id,provider_id,business_id,model,state,ordinal,cost_lower_nanos,cost_upper_nanos,replay,input_tokens,cached_input_tokens,output_tokens,reasoning_tokens,artifact_sha256,reason,request_key,(SELECT reserved_tokens FROM token_reservations t WHERE t.attempt_id=attempts.attempt_id) FROM attempts WHERE (?1 IS NULL OR round_id=?1) ORDER BY started_ms,attempt_id")?;
        let rows = statement.query_map([round], |r| {
            let low: i64 = r.get(8)?;
            let high: i64 = r.get(9)?;
            let replay: bool = r.get(10)?;
            let reserved_tokens: Option<u64> = r.get(18)?;
            let input: Option<u64> = r.get(11)?;
            let output: Option<u64> = r.get(13)?;
            Ok(CostRow {
                reserved_tokens,
                token_upper_bound: input
                    .zip(output)
                    .and_then(|(a, b)| a.checked_add(b))
                    .or(reserved_tokens),
                attempt_id: r.get(0)?,
                round_id: r.get(1)?,
                task_id: r.get(2)?,
                provider_id: r.get(3)?,
                business_id: r.get(4)?,
                model: r.get(5)?,
                state: r.get(6)?,
                ordinal: r.get(7)?,
                historical_lower_nanos: low,
                historical_upper_nanos: high,
                new_api_lower_nanos: if replay { 0 } else { low },
                new_api_upper_nanos: if replay { 0 } else { high },
                input_tokens: r.get(11)?,
                cached_input_tokens: r.get(12)?,
                output_tokens: r.get(13)?,
                reasoning_tokens: r.get(14)?,
                artifact_sha256: r.get(15)?,
                reason: r.get(16)?,
                request_key: r.get(17)?,
                replayed: replay,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(Into::into)
    }

    pub fn export_costs(&self, round: Option<&str>, directory: impl AsRef<Path>) -> Result<()> {
        let rows = self.costs(round)?;
        fs::create_dir_all(directory.as_ref())?;
        let mut csv = String::from(
            "attempt_id,round_id,task_id,provider_id,business_id,model,state,ordinal,historical_lower_usd,historical_upper_usd,new_api_lower_usd,new_api_upper_usd,input_tokens,cached_input_tokens,output_tokens,reasoning_tokens,reserved_tokens,token_upper_bound\n",
        );
        let mut markdown = String::from(
            "# Acquisition costs\n\nGenerated from the durable attempt ledger. USD intervals retain unresolved billing; replay creates no new provider charge. CPU, training, storage and serving costs require separate measured resource records.\n\n| Task | Provider/model | State | Historical USD interval | New API USD interval |\n| --- | --- | --- | ---: | ---: |\n",
        );
        for row in &rows {
            let quote = |v: &str| format!("\"{}\"", v.replace('"', "\"\""));
            let fields = [
                quote(&row.attempt_id),
                quote(&row.round_id),
                quote(&row.task_id),
                quote(&row.provider_id),
                quote(&row.business_id),
                quote(&row.model),
                quote(&row.state),
                row.ordinal.to_string(),
                usd(row.historical_lower_nanos),
                usd(row.historical_upper_nanos),
                usd(row.new_api_lower_nanos),
                usd(row.new_api_upper_nanos),
                row.input_tokens.map(|x| x.to_string()).unwrap_or_default(),
                row.cached_input_tokens
                    .map(|x| x.to_string())
                    .unwrap_or_default(),
                row.output_tokens.map(|x| x.to_string()).unwrap_or_default(),
                row.reasoning_tokens
                    .map(|x| x.to_string())
                    .unwrap_or_default(),
                row.reserved_tokens
                    .map(|x| x.to_string())
                    .unwrap_or_default(),
                row.token_upper_bound
                    .map(|x| x.to_string())
                    .unwrap_or_default(),
            ];
            csv.push_str(&fields.join(","));
            csv.push('\n');
            markdown.push_str(&format!(
                "| {} | {}/{} | {} | {}–{} | {}–{} |\n",
                row.task_id.replace('|', "\\|"),
                row.provider_id.replace('|', "\\|"),
                row.model.replace('|', "\\|"),
                row.state,
                usd(row.historical_lower_nanos),
                usd(row.historical_upper_nanos),
                usd(row.new_api_lower_nanos),
                usd(row.new_api_upper_nanos)
            ));
        }
        if rows.is_empty() {
            markdown.push_str("\nNo acquisition attempts have been recorded. This is absence of evidence, not a measured zero-cost round.\n");
        }
        fs::write(directory.as_ref().join("costs.csv"), csv)?;
        fs::write(directory.as_ref().join("costs.md"), markdown)?;
        fs::write(
            directory.as_ref().join("costs.json"),
            serde_json::to_vec_pretty(&rows)?,
        )?;
        Ok(())
    }
}

pub fn usd(nanos: i64) -> String {
    let magnitude = nanos.unsigned_abs();
    format!(
        "{}{}.{:09}",
        if nanos < 0 { "-" } else { "" },
        magnitude / 1_000_000_000,
        magnitude % 1_000_000_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (tempfile::TempDir, Ledger, Acquisition, BudgetLimits) {
        let dir = tempfile::tempdir().unwrap();
        let ledger =
            Ledger::open(dir.path().join("ledger.db"), dir.path().join("artifacts")).unwrap();
        let request = Acquisition {
            round_id: "round".into(),
            task_id: "task".into(),
            source_id: "source".into(),
            request_key: "key".into(),
            provider_id: "provider".into(),
            business_id: "business".into(),
            model: "model".into(),
            model_revision: "revision".into(),
            request_json: "{}".into(),
            price_json: "{}".into(),
            reserved_nanos: 60,
            reserved_tokens: Some(60),
            replay: false,
        };
        (
            dir,
            ledger,
            request,
            BudgetLimits {
                round_nanos: 100,
                task_nanos: 100,
                provider_nanos: 100,
                model_nanos: 100,
                tokens: Default::default(),
            },
        )
    }
    #[test]
    fn concurrent_claim_is_unique_and_reservations_count() {
        let (_dir, ledger, request, limits) = setup();
        let copies: Vec<_> = (0..8)
            .map(|_| {
                let ledger = ledger.clone();
                let request = request.clone();
                std::thread::spawn(move || ledger.claim_and_reserve(&request, limits).unwrap())
            })
            .collect();
        let claims: Vec<_> = copies.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(
            claims
                .iter()
                .filter(|c| matches!(c, Claim::Reserved(_)))
                .count(),
            1
        );
        let mut next = request.clone();
        next.request_key = "other".into();
        assert!(matches!(
            ledger.claim_and_reserve(&next, limits).unwrap(),
            Claim::BudgetStopped { .. }
        ));
        assert_eq!(ledger.costs(None).unwrap().len(), 1);
    }
    #[test]
    fn unknown_window_retains_reservation_across_restart() {
        let (dir, ledger, request, limits) = setup();
        assert!(matches!(
            ledger.claim_and_reserve(&request, limits).unwrap(),
            Claim::Reserved(_)
        ));
        assert_eq!(ledger.recover_interrupted().unwrap(), 1);
        let reloaded =
            Ledger::open(dir.path().join("ledger.db"), dir.path().join("artifacts")).unwrap();
        assert!(matches!(
            reloaded.claim_and_reserve(&request, limits).unwrap(),
            Claim::Unknown
        ));
        assert_eq!(reloaded.costs(None).unwrap()[0].historical_upper_nanos, 60);
    }
    #[test]
    fn finished_work_reuses_verified_durable_blob() {
        let (_dir, ledger, request, limits) = setup();
        let Claim::Reserved(claim) = ledger.claim_and_reserve(&request, limits).unwrap() else {
            panic!()
        };
        let hash = ledger.publish_artifact(b"{\"real\":false}").unwrap();
        ledger
            .finish(
                &claim.attempt_id,
                Some(&hash),
                true,
                20,
                20,
                Some((2, 0, 3, 0)),
                None,
            )
            .unwrap();
        assert!(matches!(
            ledger.claim_and_reserve(&request, limits).unwrap(),
            Claim::Cached { .. }
        ));
        assert_eq!(ledger.costs(None).unwrap().len(), 1);
        let proof = ledger.attempt_proof(&claim.attempt_id).unwrap();
        assert_eq!(proof.state, "completed");
        assert_eq!(proof.request_key, request.request_key);
        assert_eq!(proof.round_id, request.round_id);
        assert_eq!(
            proof.response_artifact_sha256.as_deref(),
            Some(hash.as_str())
        );
        assert_eq!(proof.output_tokens, Some(3));
        assert_eq!(proof.historical_upper_nanos, 20);
        assert!(ledger.attempt_proof("nonexistent-attempt").is_err());
    }
    #[test]
    fn billing_above_reservation_stops_round() {
        let (_dir, ledger, request, limits) = setup();
        let Claim::Reserved(claim) = ledger.claim_and_reserve(&request, limits).unwrap() else {
            panic!()
        };
        assert!(
            ledger
                .finish(
                    &claim.attempt_id,
                    None,
                    false,
                    70,
                    70,
                    None,
                    Some("usage discrepancy")
                )
                .is_err()
        );
        let mut next = request;
        next.request_key = "next".into();
        assert!(ledger.claim_and_reserve(&next, limits).is_err());
    }

    #[test]
    fn token_reservations_are_atomic_and_survive_failure_restart_and_unknown_usage() {
        let (dir, ledger, request, mut limits) = setup();
        limits.round_nanos = 10_000;
        limits.task_nanos = 10_000;
        limits.provider_nanos = 10_000;
        limits.model_nanos = 10_000;
        limits.tokens.round = Some(100);
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let ledger = ledger.clone();
                let mut request = request.clone();
                request.request_key = format!("token-fixture-{i}");
                std::thread::spawn(move || ledger.claim_and_reserve(&request, limits).unwrap())
            })
            .collect();
        let claims: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        assert_eq!(
            claims
                .iter()
                .filter(|c| matches!(c, Claim::Reserved(_)))
                .count(),
            1
        );
        assert_eq!(
            claims
                .iter()
                .filter(|c| matches!(c, Claim::TokenBudgetStopped { .. }))
                .count(),
            7
        );
        let row = ledger.costs(None).unwrap().remove(0);
        // An explicit provider failure without usage still owes its full token cap.
        ledger
            .finish(
                &row.attempt_id,
                None,
                false,
                0,
                60,
                None,
                Some("LABELLED failure without usage"),
            )
            .unwrap();
        let restarted =
            Ledger::open(dir.path().join("ledger.db"), dir.path().join("artifacts")).unwrap();
        let mut next = request;
        next.request_key = "next-token-fixture".into();
        assert!(matches!(
            restarted.claim_and_reserve(&next, limits).unwrap(),
            Claim::TokenBudgetStopped { used: 60, .. }
        ));
        assert_eq!(
            restarted.costs(None).unwrap()[0].token_upper_bound,
            Some(60)
        );
    }

    #[test]
    fn known_usage_releases_token_headroom_but_legacy_unknown_liability_fails_closed() {
        let (_dir, ledger, mut request, mut limits) = setup();
        limits.round_nanos = 10_000;
        limits.tokens.round = Some(100);
        let Claim::Reserved(first) = ledger.claim_and_reserve(&request, limits).unwrap() else {
            panic!()
        };
        ledger
            .finish(
                &first.attempt_id,
                None,
                false,
                1,
                1,
                Some((10, 0, 20, 0)),
                Some("LABELLED failed generation"),
            )
            .unwrap();
        request.request_key = "second-token-fixture".into();
        let Claim::Reserved(second) = ledger.claim_and_reserve(&request, limits).unwrap() else {
            panic!()
        };
        ledger
            .mark_unknown(&second.attempt_id, "LABELLED unknown remote acceptance")
            .unwrap();
        assert_eq!(ledger.costs(None).unwrap()[1].token_upper_bound, Some(60));

        let (_dir, legacy, mut request, mut old_limits) = setup();
        old_limits.round_nanos = 10_000;
        old_limits.task_nanos = 10_000;
        old_limits.provider_nanos = 10_000;
        old_limits.model_nanos = 10_000;
        request.reserved_tokens = None;
        legacy.claim_and_reserve(&request, old_limits).unwrap();
        old_limits.tokens.round = Some(100);
        request.request_key = "bounded-follow-up".into();
        request.reserved_tokens = Some(10);
        assert!(
            legacy
                .claim_and_reserve(&request, old_limits)
                .unwrap_err()
                .to_string()
                .contains("historical token liability is unknown")
        );
    }

    #[test]
    fn every_token_scope_includes_failed_and_in_flight_liabilities() {
        for scope in ["round", "task", "provider", "model"] {
            let (_dir, ledger, mut request, mut limits) = setup();
            request.reserved_nanos = 0;
            match scope {
                "round" => limits.tokens.round = Some(100),
                "task" => limits.tokens.task = Some(100),
                "provider" => limits.tokens.provider = Some(100),
                _ => limits.tokens.model = Some(100),
            }
            let Claim::Reserved(first) = ledger.claim_and_reserve(&request, limits).unwrap() else {
                panic!()
            };
            ledger
                .finish(
                    &first.attempt_id,
                    None,
                    false,
                    0,
                    0,
                    None,
                    Some("LABELLED failed call with absent usage"),
                )
                .unwrap();
            request.request_key = "next-scoped-fixture".into();
            let Claim::TokenBudgetStopped { scope: stopped, .. } =
                ledger.claim_and_reserve(&request, limits).unwrap()
            else {
                panic!("scope {scope} did not stop")
            };
            assert_eq!(stopped, scope);
            if scope == "model" {
                request.provider_id = "another-provider".into();
                assert!(matches!(
                    ledger.claim_and_reserve(&request, limits).unwrap(),
                    Claim::Reserved(_)
                ));
            }
        }
    }

    #[test]
    fn immutable_plan_refuses_unplanned_work_before_reserving() {
        let (_dir, ledger, mut request, limits) = setup();
        request.request_key = "a".repeat(64);
        let plan = RoundPlan {
            schema_version: 1,
            round_id: request.round_id.clone(),
            configuration_sha256: "b".repeat(64),
            jobs: vec![PlannedCompletion {
                task_id: request.task_id.clone(),
                source_id: request.source_id.clone(),
                specification_sha256: "c".repeat(64),
                sample_slot: 0,
                preferred_provider_id: request.provider_id.clone(),
                allowed_request_keys: vec![request.request_key.clone()],
            }],
        };
        ledger.record_round_plan(&plan).unwrap();
        ledger.record_round_plan(&plan).unwrap();
        let mut other = request.clone();
        other.request_key = "d".repeat(64);
        assert!(ledger.claim_and_reserve(&other, limits).is_err());
        assert!(ledger.costs(None).unwrap().is_empty());
        assert!(matches!(
            ledger.claim_and_reserve(&request, limits).unwrap(),
            Claim::Reserved(_)
        ));
        let mut changed = plan;
        changed.configuration_sha256 = "e".repeat(64);
        assert!(ledger.record_round_plan(&changed).is_err());
        assert_eq!(ledger.round_plan("round").unwrap().unwrap().jobs.len(), 1);
    }
}
