//! Checked ledger accounting, complete acquisition plans and explicit cash evidence.
use crate::{
    artefact, evaluate,
    fleet::Recording,
    gate::Decision,
    ledger::{Claim, CostRow, Ledger, RoundPlan},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoneyInterval {
    pub lower_nanos: i64,
    pub upper_nanos: i64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostGroup {
    pub attempts: usize,
    pub historical: MoneyInterval,
    pub new_api: MoneyInterval,
    /// Known usage only. Missing usage is never silently asserted to be zero.
    pub input_tokens: u64,
    pub cached_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub uncertain_attempts: usize,
    pub missing_usage_attempts: usize,
}
impl CostGroup {
    fn add(&mut self, row: &CostRow) -> Result<()> {
        ensure!(
            row.historical_lower_nanos >= 0
                && row.historical_upper_nanos >= row.historical_lower_nanos
                && row.new_api_lower_nanos >= 0
                && row.new_api_upper_nanos >= row.new_api_lower_nanos,
            "invalid ledger cost interval"
        );
        let money = |a: i64, b: i64| a.checked_add(b).context("aggregate money overflow");
        let tokens = |a: u64, b: Option<u64>| {
            a.checked_add(b.unwrap_or(0))
                .context("aggregate token count overflow")
        };
        let missing = row.input_tokens.is_none()
            || row.cached_input_tokens.is_none()
            || row.output_tokens.is_none()
            || row.reasoning_tokens.is_none();
        // Build the complete replacement first: an error cannot leave half a total.
        let next = Self {
            attempts: self
                .attempts
                .checked_add(1)
                .context("attempt count overflow")?,
            historical: MoneyInterval {
                lower_nanos: money(self.historical.lower_nanos, row.historical_lower_nanos)?,
                upper_nanos: money(self.historical.upper_nanos, row.historical_upper_nanos)?,
            },
            new_api: MoneyInterval {
                lower_nanos: money(self.new_api.lower_nanos, row.new_api_lower_nanos)?,
                upper_nanos: money(self.new_api.upper_nanos, row.new_api_upper_nanos)?,
            },
            input_tokens: tokens(self.input_tokens, row.input_tokens)?,
            cached_input_tokens: tokens(self.cached_input_tokens, row.cached_input_tokens)?,
            output_tokens: tokens(self.output_tokens, row.output_tokens)?,
            reasoning_tokens: tokens(self.reasoning_tokens, row.reasoning_tokens)?,
            uncertain_attempts: self
                .uncertain_attempts
                .checked_add(usize::from(
                    row.historical_lower_nanos != row.historical_upper_nanos,
                ))
                .context("uncertain count overflow")?,
            missing_usage_attempts: self
                .missing_usage_attempts
                .checked_add(usize::from(missing))
                .context("missing usage count overflow")?,
        };
        *self = next;
        Ok(())
    }
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AcquisitionCompletion {
    pub complete: bool,
    pub plan_sha256: Option<String>,
    pub planned_samples: usize,
    pub verified_samples: usize,
    pub real_businesses: usize,
    pub all_successes_acquired_in_selected_round: bool,
    pub reasons: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct CostSummary {
    pub evidence_kind: String,
    pub acquisition: CostGroup,
    pub per_task: BTreeMap<String, CostGroup>,
    pub per_model: BTreeMap<String, CostGroup>,
    pub per_round: BTreeMap<String, CostGroup>,
    pub real_businesses: usize,
    pub non_real_or_unverified_response_attempts: usize,
    pub acquisition_round_complete: bool,
    pub acquisition_completion: AcquisitionCompletion,
    pub compute_observations: BTreeMap<String, serde_json::Value>,
    pub api_cash_reconciled: bool,
    pub attributed_api_cash_nanos: Option<i64>,
    pub attributed_api_credit_nanos: Option<i64>,
    pub attributed_api_discount_nanos: Option<i64>,
    pub round_total_usd: Option<f64>,
    pub descriptive_cost_per_percentage_point: Option<f64>,
    pub proven_cost_per_point: Option<f64>,
    pub break_even_verified_tasks: Option<f64>,
    pub explanation: Vec<String>,
}
/// Operator metering/invoice allocation. Free compute retains measured device time.
#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeCharge {
    pub round_id: String,
    pub phase: String,
    pub device: String,
    pub elapsed_seconds: f64,
    pub cash_nanos: i64,
    pub basis: String,
    pub evidence: artefact::Artefact,
}
/// Price schedules are accrual estimates, not evidence of invoice cash after credit.
/// Each selected attempt needs its own checksum-linked settlement before a complete
/// cash cost is reported. A settlement does not silently release ledger liabilities.
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiSettlement {
    pub round_id: String,
    pub attempt_id: String,
    pub attributed_historical_nanos: i64,
    pub cash_nanos: i64,
    pub credit_nanos: i64,
    pub discount_nanos: i64,
    pub basis: String,
    pub evidence: artefact::Artefact,
}

fn real_recording(ledger: &Ledger, row: &CostRow) -> Result<Option<Recording>> {
    let Some(hash) = &row.artifact_sha256 else {
        return Ok(None);
    };
    let record: Recording = serde_json::from_slice(&ledger.read_artifact(hash)?)?;
    record.validate_identity()?;
    ensure!(
        record.request_key == row.request_key
            && record.provider_id == row.provider_id
            && record.business_id == row.business_id
            && record.model == row.model,
        "attempt and recording provenance differ"
    );
    Ok((record.origin == "real_provider").then_some(record))
}
fn completed_plan(
    ledger: &Ledger,
    plan: Option<RoundPlan>,
    rows: &[CostRow],
) -> Result<AcquisitionCompletion> {
    let Some(plan) = plan else {
        return Ok(AcquisitionCompletion{reasons:vec!["No immutable complete plan was declared before acquisition; observed attempts cannot establish the round's intended denominator.".into()],..Default::default()});
    };
    plan.validate()?;
    let mut result = AcquisitionCompletion {
        plan_sha256: Some(artefact::hash(&serde_json::to_vec(&plan)?)),
        planned_samples: plan.jobs.len(),
        all_successes_acquired_in_selected_round: true,
        ..Default::default()
    };
    let allowed: BTreeSet<_> = plan
        .jobs
        .iter()
        .flat_map(|job| job.allowed_request_keys.iter())
        .collect();
    for row in rows {
        if !allowed.contains(&row.request_key) {
            result
                .reasons
                .push(format!("unplanned attempt {}", row.attempt_id));
        }
        if !matches!(row.state.as_str(), "completed" | "failed") {
            result.reasons.push(format!(
                "unresolved attempt {}: {}",
                row.attempt_id, row.state
            ));
        }
    }
    let acquired: BTreeSet<_> = rows.iter().map(|r| r.attempt_id.as_str()).collect();
    let mut businesses = BTreeSet::new();
    for job in &plan.jobs {
        let mut successes = Vec::new();
        for key in &job.allowed_request_keys {
            if let Some(Claim::Cached {
                attempt_id,
                artifact_sha256,
            }) = ledger.lookup_request(key)?
            {
                let record: Recording =
                    serde_json::from_slice(&ledger.read_artifact(&artifact_sha256)?)?;
                record.validate_identity()?;
                ensure!(
                    record.request_key == *key
                        && record.request.task_id == job.task_id
                        && record.request.source_id == job.source_id
                        && record.request.specification_sha256 == job.specification_sha256
                        && record.request.sample_slot == job.sample_slot,
                    "completed acquisition differs from immutable plan"
                );
                if record.origin == "real_provider" {
                    crate::fleet::parse_response(&record.response)?;
                    businesses.insert(record.business_id);
                    successes.push(attempt_id);
                }
            }
        }
        if successes.len() == 1 {
            result.verified_samples += 1;
            if !acquired.contains(successes[0].as_str()) {
                result.all_successes_acquired_in_selected_round = false;
            }
        } else {
            result.reasons.push(format!(
                "task {} sample {} has {} verified real completions; expected one",
                job.task_id,
                job.sample_slot,
                successes.len()
            ));
        }
    }
    result.real_businesses = businesses.len();
    if businesses.len() < 3 {
        result
            .reasons
            .push("fewer than three independently owned real response businesses".into());
    }
    result.complete =
        result.reasons.is_empty() && result.verified_samples == result.planned_samples;
    Ok(result)
}
fn settlement_totals(
    rows: &[CostRow],
    round: &str,
    path: &Path,
    root: &Path,
) -> Result<Option<(i64, i64, i64)>> {
    if !path.exists() {
        return Ok(None);
    }
    let settlements: Vec<ApiSettlement> = serde_json::from_slice(&std::fs::read(path)?)?;
    let mut by_attempt = BTreeMap::new();
    for settlement in settlements.iter().filter(|s| s.round_id == round) {
        ensure!(
            by_attempt
                .insert(settlement.attempt_id.as_str(), settlement)
                .is_none(),
            "duplicate API cash settlement"
        );
    }
    let mut cash = 0i64;
    let mut credit = 0i64;
    let mut discount = 0i64;
    for row in rows {
        let Some(settlement) = by_attempt.remove(row.attempt_id.as_str()) else {
            return Ok(None);
        };
        ensure!(
            settlement.cash_nanos >= 0
                && settlement.credit_nanos >= 0
                && settlement.discount_nanos >= 0
                && !settlement.basis.trim().is_empty(),
            "invalid API cash/credit settlement"
        );
        let attributed = settlement
            .cash_nanos
            .checked_add(settlement.credit_nanos)
            .and_then(|n| n.checked_add(settlement.discount_nanos))
            .context("API settlement overflow")?;
        ensure!(
            attributed == settlement.attributed_historical_nanos
                && attributed >= row.historical_lower_nanos
                && attributed <= row.historical_upper_nanos,
            "API cash/credit evidence does not reconcile the historical interval"
        );
        settlement.evidence.verify(root)?;
        cash = cash
            .checked_add(settlement.cash_nanos)
            .context("API cash overflow")?;
        credit = credit
            .checked_add(settlement.credit_nanos)
            .context("API credit overflow")?;
        discount = discount
            .checked_add(settlement.discount_nanos)
            .context("API discount overflow")?;
    }
    ensure!(
        by_attempt.is_empty(),
        "API settlements contain attempts outside selected round"
    );
    Ok(Some((cash, credit, discount)))
}

fn bind_training_to_round(candidate_path: &Path, rows: &[CostRow], root: &Path) -> Result<()> {
    let evaluation: evaluate::Evaluation = serde_json::from_slice(&std::fs::read(candidate_path)?)?;
    let checkpoint: crate::train::CheckpointManifest = serde_json::from_slice(&std::fs::read(
        evaluation.checkpoint.metadata.verify(root)?,
    )?)?;
    let dataset = crate::curate::validate_dataset(
        &root.join("data/curated"),
        &root.join("grader_private/heldout-index.json"),
    )?;
    ensure!(
        dataset.manifest.data_sha256 == checkpoint.dataset_sha256
            && dataset.manifest.manifest_sha256 == checkpoint.dataset_manifest_sha256,
        "cost round uses a different dataset from the evaluated checkpoint"
    );
    let attempts: BTreeMap<_, _> = rows.iter().map(|r| (r.attempt_id.as_str(), r)).collect();
    for sample in &dataset.rows {
        let attempt = attempts
            .get(sample.attempt_id.as_str())
            .context("training data acquisition is not attributed to selected cost round")?;
        ensure!(
            attempt.request_key == sample.request_key
                && attempt.task_id == sample.task_id
                && attempt.state == "completed",
            "training sample and selected round provenance differ"
        );
    }
    Ok(())
}

pub fn export(ledger: &Ledger, round: Option<&str>, directory: &Path) -> Result<CostSummary> {
    ledger.export_costs(round, directory)?;
    let rows = ledger.costs(round)?;
    let mut total = CostGroup::default();
    let (mut task_groups, mut models, mut rounds) = (
        BTreeMap::<String, CostGroup>::new(),
        BTreeMap::<String, CostGroup>::new(),
        BTreeMap::<String, CostGroup>::new(),
    );
    let mut businesses = BTreeSet::new();
    let mut unverified = 0usize;
    for row in &rows {
        total.add(row)?;
        task_groups
            .entry(row.task_id.clone())
            .or_default()
            .add(row)?;
        // A JSON tuple is unambiguous even when a provider/model contains '/'.
        models
            .entry(serde_json::to_string(&(&row.provider_id, &row.model))?)
            .or_default()
            .add(row)?;
        rounds.entry(row.round_id.clone()).or_default().add(row)?;
        if let Some(record) = real_recording(ledger, row)? {
            businesses.insert(record.business_id);
        } else {
            unverified = unverified.checked_add(1).context("origin count overflow")?;
        }
    }
    let plan = round.map(|id| ledger.round_plan(id)).transpose()?.flatten();
    let mut completion = completed_plan(ledger, plan, &rows)?;
    if unverified > 0 {
        completion.complete = false;
        completion.reasons.push(format!("{unverified} acquisition attempts lack a verified real-provider response; local refusals/fixtures are not real calls"));
    }
    let root = std::env::current_dir()?;
    let mut observations = BTreeMap::new();
    for (name, path) in [
        ("base_inference_qa", "reports/base-inference-smoke.json"),
        ("native_gradient_qa", "reports/native-gradient-smoke.json"),
        (
            "native_gradient_512_token_qa",
            "reports/native-gradient-512.json",
        ),
        ("local_serving_qa", "reports/serving-smoke.json"),
        ("candidate_training", "checkpoints/candidate/manifest.json"),
        (
            "seed_control_training",
            "checkpoints/seed-control/manifest.json",
        ),
        (
            "degraded_control_training",
            "checkpoints/degraded/manifest.json",
        ),
        (
            "overfit_control_training",
            "checkpoints/overfit/manifest.json",
        ),
    ] {
        if Path::new(path).exists() {
            observations.insert(name.into(),serde_json::json!({"path":path,"sha256":artefact::file_hash(Path::new(path))?,"measurement":serde_json::from_slice::<serde_json::Value>(&std::fs::read(path)?)?}));
        }
    }
    let gradient_time = Path::new("reports/native-gradient-512-time.txt");
    if gradient_time.exists() {
        observations.insert("native_gradient_512_process_resource_qa".into(),serde_json::json!({"path":gradient_time,"sha256":artefact::file_hash(gradient_time)?,"evidence_kind":"measured_process_resources_for_synthetic_tensor_qa_not_a_learning_round","format":"usr_bin_time_text","measurement_text":std::fs::read_to_string(gradient_time)?,"cash_price":"unmeasured_owned_workstation"}));
    }
    let settlement = round
        .map(|id| settlement_totals(&rows, id, &directory.join("api-settlements.json"), &root))
        .transpose()?
        .flatten();
    let mut round_total_usd = None;
    let mut descriptive = None;
    let mut proven = None;
    let metering = directory.join("compute-charges.json");
    if let Some(round_id) = round.filter(|_| {
        completion.complete
            && completion.all_successes_acquired_in_selected_round
            && settlement.is_some()
            && metering.exists()
    }) {
        let charges: Vec<ComputeCharge> = serde_json::from_slice(&std::fs::read(metering)?)?;
        let mut phases = BTreeSet::new();
        let mut unique_charges = BTreeSet::new();
        let mut cash = 0i64;
        for charge in charges.iter().filter(|c| c.round_id == round_id) {
            ensure!(
                unique_charges.insert(artefact::hash(&serde_json::to_vec(charge)?)),
                "duplicate measured compute charge"
            );
            ensure!(
                charge.elapsed_seconds.is_finite()
                    && charge.elapsed_seconds >= 0.0
                    && charge.cash_nanos >= 0
                    && !charge.basis.trim().is_empty()
                    && !charge.device.trim().is_empty(),
                "invalid measured compute charge"
            );
            charge.evidence.verify(&root)?;
            phases.insert(charge.phase.as_str());
            cash = cash
                .checked_add(charge.cash_nanos)
                .context("compute cost overflow")?;
        }
        if ["training", "evaluation", "grading", "storage", "serving"]
            .iter()
            .all(|phase| phases.contains(phase))
        {
            let api_cash = settlement.context("cash settlement disappeared")?.0;
            let attributable = api_cash
                .checked_add(cash)
                .context("round cash total overflow")?;
            round_total_usd = Some(attributable as f64 / 1e9);
            let a = Path::new("reports/incumbent-evaluation/evaluation.json");
            let b = Path::new("reports/candidate-evaluation/evaluation.json");
            if a.exists() && b.exists() {
                bind_training_to_round(b, &rows, &root)?;
                let verified = evaluate::gate_files(a, b, &root, Path::new("config/gate.toml"))?;
                if let Some((d, p)) = cost_per_point(attributable as f64 / 1e9, &verified) {
                    descriptive = Some(d);
                    proven = p;
                }
            }
        }
    }
    let summary=CostSummary{evidence_kind:"verified_attempt_accruals_with_explicit_cash_and_compute_evidence".into(),acquisition:total,per_task:task_groups,per_model:models,per_round:rounds,real_businesses:businesses.len(),non_real_or_unverified_response_attempts:unverified,acquisition_round_complete:completion.complete,acquisition_completion:completion,compute_observations:observations,api_cash_reconciled:settlement.is_some(),attributed_api_cash_nanos:settlement.map(|s|s.0),attributed_api_credit_nanos:settlement.map(|s|s.1),attributed_api_discount_nanos:settlement.map(|s|s.2),round_total_usd,descriptive_cost_per_percentage_point:descriptive,proven_cost_per_point:proven,break_even_verified_tasks:None,explanation:vec![
        "Historical token-price accruals and replay's zero new API accruals are separate integer-nanodollar intervals. A price schedule multiplied by real usage is not an invoice cash payment after credits/discounts.".into(),
        "Real business counts require checksum-verified real-provider response envelopes and recomputed canonical request identities; fixture/local-refusal rows do not establish provider acquisition.".into(),
        "Round completeness requires the immutable full plan declared before the first attempt and exactly one real durable completion for every planned task/sample. Failed HTTP retries remain in costs without invalidating an otherwise completed plan; UNKNOWN, unplanned work and missing completions block it.".into(),
        "Token totals contain known usage only. Missing-usage attempts are counted explicitly; zero known tokens is not a claim that unresolved calls used no tokens.".into(),
        "A cash total additionally requires per-attempt api-settlements.json invoice/credit/discount evidence and checksum-linked compute-charges.json metering for training/evaluation/grading/storage/serving. No tariff, free credit or electricity allocation is invented.".into(),
        "Round total attributes the selected round's historical acquisition cash plus its declared measured compute cash. Replay's new provider spend remains zero; historical acquisition is not charged again. Cached acquisitions from another round need an explicit allocation before a new complete cash round is claimed.".into(),
        "Compute smoke wall times are labelled observations, not summed CPU core-hours or machine rental duration. Some workloads ran concurrently.".into(),
        "Cost per point divides a complete attributable round cash total by 100 times verified rate gain. Nonpositive gain is N/A; inconclusive positive gain is descriptive only. Break-even/amortised success costs need a SHIP quality comparison and measured comparable marginal costs.".into(),
    ]};
    artefact::write_json(&directory.join("cost-summary.json"), &summary)?;
    let usd = |n: i64| format!("${}", crate::ledger::usd(n));
    let optional = |v: Option<f64>| {
        v.map(|v| format!("${v:.9}"))
            .unwrap_or_else(|| "unavailable".into())
    };
    let mut markdown = format!(
        "# Measured costs and limits\n\n{} attempts; {} verified real response businesses. Full acquisition plan complete: {}.\n\n| Quantity | Observed amount |\n| --- | ---: |\n| Historical API accrual lower | {} |\n| Historical API accrual upper | {} |\n| New API accrual lower | {} |\n| New API accrual upper | {} |\n| Reconciled acquisition cash | {} |\n| Credits / discounts | {} / {} |\n| Complete attributable cash round | {} |\n| Descriptive cost per point | {} |\n| Proven cost per point | {} |\n| Break-even volume | unavailable |\n| Missing-usage attempts | {} |\n\n",
        summary.acquisition.attempts,
        summary.real_businesses,
        summary.acquisition_round_complete,
        usd(summary.acquisition.historical.lower_nanos),
        usd(summary.acquisition.historical.upper_nanos),
        usd(summary.acquisition.new_api.lower_nanos),
        usd(summary.acquisition.new_api.upper_nanos),
        summary
            .attributed_api_cash_nanos
            .map(usd)
            .unwrap_or_else(|| "unreconciled".into()),
        summary
            .attributed_api_credit_nanos
            .map(usd)
            .unwrap_or_else(|| "unreconciled".into()),
        summary
            .attributed_api_discount_nanos
            .map(usd)
            .unwrap_or_else(|| "unreconciled".into()),
        optional(summary.round_total_usd),
        optional(summary.descriptive_cost_per_percentage_point),
        optional(summary.proven_cost_per_point),
        summary.acquisition.missing_usage_attempts
    );
    for explanation in &summary.explanation {
        markdown.push_str(&format!("- {explanation}\n"));
    }
    for reason in &summary.acquisition_completion.reasons {
        markdown.push_str(&format!("- Incomplete: {reason}\n"));
    }
    markdown.push_str("\nPer-attempt usage and historical/new uncertainty are in costs.csv. JSON includes task, unambiguous provider/model, round and checksum-linked compute observations. The dominating actual line remains unestablished until a complete measured round exists.\n");
    artefact::atomic_write(&directory.join("costs.md"), markdown.as_bytes())?;
    Ok(summary)
}

pub fn cost_per_point(
    round_usd: f64,
    verified: &evaluate::VerifiedGate,
) -> Option<(f64, Option<f64>)> {
    if !round_usd.is_finite()
        || round_usd < 0.0
        || verified.report.evidence_kind != "verified_checkpoint_evaluation"
        || verified.report.decision == Decision::InvalidEvaluation
    {
        return None;
    }
    let gain = verified.report.effect?;
    if !gain.is_finite() || gain <= 0.0 {
        return None;
    }
    let ratio = round_usd / (100.0 * gain);
    if !ratio.is_finite() {
        return None;
    }
    Some((
        ratio,
        (verified.report.decision == Decision::Ship).then_some(ratio),
    ))
}
pub fn break_even_volume(
    round_usd: f64,
    teacher_usd_per_success: f64,
    student_usd_per_success: f64,
    decision: &Decision,
) -> Option<f64> {
    if *decision != Decision::Ship
        || ![round_usd, teacher_usd_per_success, student_usd_per_success]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    {
        return None;
    }
    let saving = teacher_usd_per_success - student_usd_per_success;
    if saving <= 0.0 {
        return None;
    }
    let volume = (round_usd / saving).ceil();
    volume.is_finite().then_some(volume)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row() -> CostRow {
        CostRow {
            attempt_id: "attempt".into(),
            round_id: "round".into(),
            task_id: "task".into(),
            provider_id: "provider".into(),
            business_id: "business".into(),
            model: "model".into(),
            state: "failed".into(),
            ordinal: 1,
            historical_lower_nanos: 0,
            historical_upper_nanos: 10,
            new_api_lower_nanos: 0,
            new_api_upper_nanos: 10,
            input_tokens: None,
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
            artifact_sha256: None,
            reason: None,
            request_key: "a".repeat(64),
            replayed: false,
        }
    }
    #[test]
    fn aggregate_overflow_fails_without_partial_mutation() {
        let mut group = CostGroup::default();
        let mut first = row();
        first.historical_upper_nanos = i64::MAX;
        group.add(&first).unwrap();
        let before = serde_json::to_vec(&group).unwrap();
        assert!(group.add(&row()).is_err());
        assert_eq!(serde_json::to_vec(&group).unwrap(), before);
        assert_eq!(group.missing_usage_attempts, 1);
    }
    #[test]
    fn no_declared_plan_cannot_be_inferred_from_attempt_counts() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::open(dir.path().join("db"), dir.path().join("blobs")).unwrap();
        let report = completed_plan(&ledger, None, &[row()]).unwrap();
        assert!(!report.complete);
        assert_eq!(report.planned_samples, 0);
    }
    #[test]
    fn percentage_point_denominator_and_inconclusive_limit_are_exact() {
        let cfg = crate::gate::GateConfig::load(Path::new("config/gate.toml")).unwrap();
        let mut report = crate::gate::GateReport::invalid("test fixture", &cfg);
        report.decision = Decision::NoShipInconclusive;
        report.effect = Some(0.05);
        report.evidence_kind = "verified_checkpoint_evaluation".into();
        let mut verified = evaluate::VerifiedGate {
            report,
            incumbent_checkpoint_id: "TEST FIXTURE".into(),
            candidate_checkpoint_id: "TEST FIXTURE".into(),
            incumbent_evaluation_sha256: "TEST FIXTURE".into(),
            candidate_evaluation_sha256: "TEST FIXTURE".into(),
            gate_config_sha256: "TEST FIXTURE".into(),
        };
        assert_eq!(cost_per_point(10.0, &verified), Some((2.0, None)));
        verified.report.decision = Decision::Ship;
        assert_eq!(cost_per_point(10.0, &verified), Some((2.0, Some(2.0))));
        verified.report.effect = Some(-0.01);
        assert_eq!(cost_per_point(10.0, &verified), None);
        verified.report.effect = Some(0.0);
        assert_eq!(cost_per_point(10.0, &verified), None);
    }
    #[test]
    fn break_even_requires_quality_and_positive_measured_savings() {
        assert_eq!(
            break_even_volume(5.0, 0.01, 0.005, &Decision::Ship),
            Some(1000.0)
        );
        assert_eq!(break_even_volume(5.0, 0.01, 0.01, &Decision::Ship), None);
        assert_eq!(
            break_even_volume(5.0, 0.01, 0.005, &Decision::NoShipInconclusive),
            None
        );
    }
}
