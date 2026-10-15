//! Freeze development-only rankings without choosing favourable seeds.
use super::*;
fn method_group(arm: &experiments::ArmSpec) -> Result<String> {
    Ok(artefact::hash(&serde_json::to_vec(&serde_json::json!({
        "student":arm.student,"teachers":arm.teacher_ids,"signals":arm.signals,"contents":arm.contents,
        "data_policy":arm.data_policy,"progressive":arm.progressive,"self_distillation":arm.self_distillation,
    }))?))
}

fn selection_value(
    plan_path: &Path,
    quick_output: &Path,
    top_groups: usize,
    snapshot: Option<&BTreeSet<String>>,
    schema_version: u32,
) -> Result<serde_json::Value> {
    ensure!(
        (1..=8).contains(&top_groups),
        "selection group count must be 1–8"
    );
    let plan = experiments::load_plan(plan_path)?;
    let completed = experiments::completed_results(quick_output, plan_path)?
        .into_iter()
        .filter(|r| snapshot.is_none_or(|ids| ids.contains(&r.arm.id)))
        .collect::<Vec<_>>();
    if let Some(ids) = snapshot {
        ensure!(
            completed
                .iter()
                .map(|r| r.arm.id.clone())
                .collect::<BTreeSet<_>>()
                == *ids,
            "frozen selection snapshot has lost a completed verified arm"
        );
    }
    ensure!(
        !completed.is_empty(),
        "no complete, verified development comparisons to select"
    );
    let mut ranking_inputs = Vec::new();
    let mut groups = BTreeMap::<String, (f64, f64, Vec<String>)>::new();
    let mut exclusions = Vec::new();
    for result in &completed {
        let evidence = result
            .outcome
            .development_evaluation
            .as_ref()
            .context("completed arm lacks development evidence")?;
        let directory = evidence
            .path
            .parent()
            .context("evaluation directory missing")?;
        let report = crate::study_evaluate::verify_completed(directory)?;
        ensure!(
            report["split"] == "development"
                && report["healthy_complete"] == true
                && report["checkpoint_id"]
                    == result
                        .outcome
                        .checkpoint_artifacts
                        .iter()
                        .find(|a| a
                            .path
                            .file_name()
                            .is_some_and(|n| n == "adapter.safetensors"))
                        .context("selected arm has no adapter tensor")?
                        .sha256,
            "selection requires healthy, checkpoint-bound development observations"
        );
        let sources = report["per_source"]
            .as_array()
            .context("per-source development scores absent")?;
        ensure!(
            sources.len() == 4,
            "selection requires every original development source"
        );
        let score = sources
            .iter()
            .map(|s| s["pass_rate"].as_f64().context("source score absent"))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>()
            / sources.len() as f64;
        let milliseconds = report["summed_generation_ms"]
            .as_f64()
            .context("actual generation timing missing")?;
        ensure!(
            schema_version >= 2 || result.arm.student != StudentArchitecture::TernaryBitnet,
            "legacy BitNet selection lacks frozen numerical qualification; create a new selection output"
        );
        let numerical = crate::study_evaluate::verify_numerical_equivalence(directory)?;
        let key = method_group(&result.arm)?;
        if numerical.qualified {
            let group = groups.entry(key).or_default();
            group.0 += score;
            group.1 += milliseconds;
            group.2.push(result.arm.id.clone());
        } else {
            exclusions
                .push(serde_json::json!({"arm_id":result.arm.id,"reasons":numerical.reasons}));
        }
        let mut row = serde_json::json!({"arm_id":result.arm.id,"outcome":result.outcome_artefact,
            "development_evaluation":evidence,"source_macro_pass_rate":score,"summed_generation_ms":milliseconds,
            "worker_wall_seconds":result.worker_wall_seconds});
        if schema_version >= 2 {
            row["ranking_eligible"] = numerical.qualified.into();
            row["numerical_equivalence"] = serde_json::to_value(numerical)?;
        }
        ranking_inputs.push(row);
    }
    ranking_inputs.sort_by(|a, b| a["arm_id"].as_str().cmp(&b["arm_id"].as_str()));
    let mut ranking = groups
        .into_iter()
        .map(|(key, (score, milliseconds, mut arms))| {
            arms.sort();
            let count = arms.len() as f64;
            (key, score / count, milliseconds / count, arms)
        })
        .collect::<Vec<_>>();
    ranking.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut selected = Vec::new();
    let mut selected_groups = 0;
    for (key, _, _, _) in &ranking {
        if selected_groups == top_groups {
            break;
        }
        let candidates = plan
            .arms
            .iter()
            .filter(|a| {
                a.applicable
                    && experiments::fidelity_inapplicable_reasons(
                        a,
                        &experiments::training_for(&plan, a, experiments::Fidelity::Full),
                    )
                    .is_empty()
            })
            .map(|a| Ok((a, method_group(a)?)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(_, group)| group == key)
            .map(|(a, _)| a.id.clone())
            .collect::<Vec<_>>();
        if candidates.is_empty() || selected.len() + candidates.len() > plan.full.maximum_trials {
            continue;
        }
        selected.extend(candidates);
        selected_groups += 1;
    }
    ensure!(
        schema_version >= 2 || !selected.is_empty(),
        "full-fidelity budget cannot accommodate a complete selected seed group"
    );
    selected.sort();
    let ranking = ranking.into_iter().map(|(key,score,milliseconds,arms)| serde_json::json!({
        "method_group":key,"source_macro_pass_rate":score,"mean_summed_generation_ms":milliseconds,
        "observed_arm_ids":arms,"observed_training_seeds":arms.len(),
    })).collect::<Vec<_>>();
    let mut value = serde_json::json!({"schema_version":schema_version,"evidence_kind":"verified_development_only_study_selection",
        "study_plan":capture(plan_path)?,"quick_output":quick_output,"top_groups":top_groups,
        "ranking_inputs":ranking_inputs,"ranking":ranking,"selected_arms":selected,
        "selection_rule":"Average source-macro development success within method groups across every observed training seed; break ties by lower measured generation time, then group hash. Freeze every applicable preregistered seed of selected groups within the full trial cap.",
        "held_out_used":false,"production_eligible":false,
        "limitations":["Best tested in the executed local development catalogue; queued and unavailable arms have not been compared.",
            "Quick assistant-window screening is an operator experiment, not evidence of general learning.",
            "Four jointly authored development sources, repeated selection and one decoding seed cannot establish a significant improvement or global optimum.",
            "Generation timings exclude model loading and grading; local energy and amortised cash costs are unmeasured.",
            "Full-fidelity training and the independent sealed promotion gate remain separate work."]});
    if schema_version >= 2 {
        let unavailable = selected.is_empty();
        value["status"] = if unavailable {
            "unavailable"
        } else {
            "selected"
        }
        .into();
        value["selection_available"] = (!unavailable).into();
        value["excluded_from_ranking"] = serde_json::to_value(exclusions)?;
        value["unavailable_reasons"] = if unavailable {
            serde_json::json!([if ranking.is_empty() {
                "No observed method group has the required independently reconstructed numerical equivalence."
            } else {
                "The full-fidelity budget cannot accommodate a complete qualified seed group."
            }])
        } else {
            serde_json::json!([])
        };
        value["selection_rule"] = "Exclude BitNet without independently reconstructed checkpoint-bound schema-2 numerical parity; retain its descriptive score. Average source-macro development success within qualified method groups across every observed training seed; break ties by lower measured generation time, then group hash. Freeze every applicable preregistered seed of selected groups within the full trial cap.".into();
    }
    Ok(value)
}

pub fn select(
    plan_path: &Path,
    quick_output: &Path,
    output: &Path,
    top_groups: usize,
) -> Result<serde_json::Value> {
    if output.exists() {
        let recorded: serde_json::Value = serde_json::from_slice(&fs::read(output)?)?;
        let selected: Vec<String> = serde_json::from_value(recorded["selected_arms"].clone())?;
        verify_record(output, plan_path, &selected, true)?;
        ensure!(
            recorded["top_groups"] == top_groups,
            "selection group budget is already frozen; use a new output for a changed decision"
        );
        return Ok(recorded);
    }
    let selection = selection_value(plan_path, quick_output, top_groups, None, 2)?;
    artefact::write_json(output, &selection)?;
    Ok(selection)
}

/// A full-fidelity caller cannot promote arbitrary JSON into a selection record.
pub fn verify_selection(path: &Path, plan_path: &Path, selected: &[String]) -> Result<()> {
    verify_record(path, plan_path, selected, false)
}

fn verify_record(
    path: &Path,
    plan_path: &Path,
    selected: &[String],
    allow_unavailable: bool,
) -> Result<()> {
    let recorded: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        matches!(recorded["schema_version"].as_u64(), Some(1 | 2))
            && recorded["evidence_kind"] == "verified_development_only_study_selection"
            && recorded["held_out_used"] == false
            && recorded["production_eligible"] == false,
        "unsupported or non-development selection record"
    );
    let bound: Artefact = serde_json::from_value(recorded["study_plan"].clone())?;
    ensure!(
        bound.verify(&std::env::current_dir()?)? == plan_path.canonicalize()?,
        "selection uses another study plan"
    );
    let quick = PathBuf::from(
        recorded["quick_output"]
            .as_str()
            .context("selection quick output missing")?,
    );
    let top = usize::try_from(
        recorded["top_groups"]
            .as_u64()
            .context("selection group count absent")?,
    )?;
    let snapshot = recorded["ranking_inputs"]
        .as_array()
        .context("selection snapshot missing")?
        .iter()
        .map(|row| {
            row["arm_id"]
                .as_str()
                .map(str::to_owned)
                .context("selection snapshot arm ID missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let unique = snapshot.iter().cloned().collect::<BTreeSet<_>>();
    ensure!(
        unique.len() == snapshot.len(),
        "duplicate arm in selection snapshot"
    );
    ensure!(
        recorded
            == selection_value(
                plan_path,
                &quick,
                top,
                Some(&unique),
                u32::try_from(
                    recorded["schema_version"]
                        .as_u64()
                        .context("selection schema missing")?
                )?
            )?,
        "selection differs from independently reconstructed development ranking"
    );
    let expected: Vec<String> = serde_json::from_value(recorded["selected_arms"].clone())?;
    ensure!(
        allow_unavailable || !expected.is_empty(),
        "selection is unavailable and cannot start a full-fidelity study"
    );
    let mut actual = selected.to_vec();
    actual.sort();
    ensure!(
        actual == expected,
        "requested full arms differ from frozen complete seed groups"
    );
    Ok(())
}
