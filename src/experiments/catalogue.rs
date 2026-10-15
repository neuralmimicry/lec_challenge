//! Enumerate a finite operator product; aliases do not become extra methods.
use super::*;

fn subsets<T: Clone>(items: &[T], empty: bool) -> Vec<Vec<T>> {
    ((if empty { 0 } else { 1 })..(1usize << items.len()))
        .map(|mask| {
            items
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, v)| v.clone())
                .collect()
        })
        .collect()
}
fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut result = Vec::new();
    for i in 0..items.len() {
        let mut remaining = items.to_vec();
        let first = remaining.remove(i);
        for tail in permutations(&remaining) {
            let mut row = vec![first.clone()];
            row.extend(tail);
            result.push(row);
        }
    }
    result
}

fn applicability(arm: &ArmSpec, inputs: &StudyInputs) -> Vec<String> {
    let mut reasons = BTreeSet::new();
    match inputs
        .students
        .iter()
        .find(|s| s.architecture == arm.student)
    {
        Some(s) => {
            if !s.training.usable() {
                reasons.insert(format!(
                    "student training unavailable: {}",
                    s.training.reason
                ));
            }
            if !s.exported_checkpoint_reload.usable() {
                reasons.insert(format!(
                    "student export/reload unavailable: {}",
                    s.exported_checkpoint_reload.reason
                ));
            }
        }
        None => {
            reasons.insert("student has no verified training/export/reload capability".into());
        }
    }
    if arm.progressive && arm.teacher_ids.len() < 2 {
        reasons.insert("progressive assistants require at least two ordered real teachers".into());
    }
    let mut sources = BTreeSet::new();
    for id in &arm.teacher_ids {
        let teacher = inputs
            .teachers
            .iter()
            .find(|t| &t.id == id)
            .expect("enumerated declared teacher");
        if !sources.insert(&teacher.source_model_id) {
            reasons.insert("an ensemble repeats the same original source model".into());
        }
        if !teacher.rights_approved {
            reasons.insert(format!(
                "teacher {id} lacks approved output-training rights"
            ));
        }
        for signal in &arm.signals {
            match teacher.signals.get(signal) {
                Some(cap) if cap.usable() => {}
                Some(cap) => {
                    reasons.insert(format!(
                        "teacher {id} signal {signal:?} unavailable: {}",
                        cap.reason
                    ));
                }
                None => {
                    reasons.insert(format!(
                        "teacher {id} signal {signal:?} has no verified capability"
                    ));
                }
            }
            let student_alignment = inputs
                .students
                .iter()
                .find(|s| s.architecture == arm.student)
                .and_then(|s| s.teacher_token_alignment.get(id))
                .copied()
                .unwrap_or(
                    arm.student == StudentArchitecture::ConventionalQwenLora
                        && teacher.token_alignment_verified,
                );
            if *signal == SignalMethod::SoftTokenKl && !student_alignment {
                reasons.insert(format!("teacher {id} lacks verified complete vocabulary/token-ID equivalence for soft-token KL"));
            }
        }
        for content in &arm.contents {
            match teacher.contents.get(content) {
                Some(cap) if cap.usable() => {}
                Some(cap) => {
                    reasons.insert(format!(
                        "teacher {id} content {content:?} unavailable: {}",
                        cap.reason
                    ));
                }
                None => {
                    reasons.insert(format!(
                        "teacher {id} content {content:?} has no verified evidence"
                    ));
                }
            }
        }
        if arm.data_policy == DataPolicy::OnPolicy && !teacher.on_policy.usable() {
            reasons.insert(format!(
                "teacher {id} lacks checkpoint-linked on-policy rollouts: {}",
                teacher.on_policy.reason
            ));
        }
        if arm.self_distillation && !teacher.self_distillation.usable() {
            reasons.insert(format!(
                "teacher {id} lacks frozen student-generated self-distillation evidence: {}",
                teacher.self_distillation.reason
            ));
        }
        reasons.extend(rollout::reasons(arm, teacher, inputs));
    }
    reasons.into_iter().collect()
}

pub(super) fn enumerate(config: &StudyConfig, inputs: &StudyInputs) -> Result<Vec<ArmSpec>> {
    let mut teacher_ids = inputs
        .teachers
        .iter()
        .map(|t| t.id.clone())
        .collect::<Vec<_>>();
    teacher_ids.sort();
    let teacher_subsets = subsets(&teacher_ids, false)
        .into_iter()
        .filter(|s| (config.minimum_teachers..=config.maximum_teachers).contains(&s.len()))
        .collect::<Vec<_>>();
    ensure!(
        !teacher_subsets.is_empty(),
        "no configured teacher subset is available"
    );
    let signals = subsets(&config.signals, false);
    let contents = subsets(&config.contents, true);
    let mut arms = Vec::new();
    let mut ids = BTreeSet::new();
    for student in &config.students {
        for policy in &config.policies {
            for progressive in &config.progressive {
                for self_distillation in &config.self_distillation {
                    for teachers in &teacher_subsets {
                        let orders = if *progressive {
                            permutations(teachers)
                        } else {
                            vec![teachers.clone()]
                        };
                        for order in orders {
                            for signal_set in &signals {
                                for content_set in &contents {
                                    for seed in &config.seeds {
                                        ensure!(
                                            arms.len() < config.maximum_catalogue_arms,
                                            "finite study product exceeds maximum_catalogue_arms; increase the explicit cap or reduce configured axes"
                                        );
                                        let identity = serde_json::json!({"student":student,"teacher_ids":order,"signals":signal_set,"contents":content_set,
                    "data_policy":policy,"progressive":progressive,"self_distillation":self_distillation,"seed":seed});
                                        let id = format!(
                                            "arm-{}",
                                            &artefact::hash(&serde_json::to_vec(&identity)?)[..24]
                                        );
                                        ensure!(
                                            ids.insert(id.clone()),
                                            "duplicate arm identity in configured study"
                                        );
                                        let mut arm = ArmSpec {
                                            id,
                                            student: *student,
                                            teacher_ids: order.clone(),
                                            signals: signal_set.clone(),
                                            contents: content_set.clone(),
                                            data_policy: *policy,
                                            progressive: *progressive,
                                            self_distillation: *self_distillation,
                                            seed: *seed,
                                            applicable: false,
                                            inapplicable_reasons: vec![],
                                            rollout_initial: None,
                                        };
                                        arm.rollout_initial = rollout::initial_for(&arm, inputs);
                                        arm.inapplicable_reasons = applicability(&arm, inputs);
                                        arm.applicable = arm.inapplicable_reasons.is_empty();
                                        arms.push(arm);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(arms)
}

fn coverage(arm: &ArmSpec) -> BTreeSet<String> {
    let mut covered = BTreeSet::new();
    covered.insert(format!("student:{:?}", arm.student));
    covered.insert(format!("policy:{:?}", arm.data_policy));
    for id in &arm.teacher_ids {
        covered.insert(format!("teacher:{id}"));
    }
    for signal in &arm.signals {
        covered.insert(format!("signal:{signal:?}"));
    }
    for content in &arm.contents {
        covered.insert(format!("content:{content:?}"));
    }
    if arm.teacher_ids.len() > 1 {
        covered.insert("organiser:multi_teacher".into());
    }
    if arm.progressive {
        covered.insert("organiser:progressive".into());
    }
    if arm.self_distillation {
        covered.insert("organiser:self_distillation".into());
    }
    covered
}

fn method_group(arm: &ArmSpec) -> String {
    serde_json::json!([
        arm.student,
        arm.teacher_ids,
        arm.signals,
        arm.contents,
        arm.data_policy,
        arm.progressive,
        arm.self_distillation
    ])
    .to_string()
}

fn mixture_coverage(arm: &ArmSpec) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    if arm.signals.len() > 1 {
        result.insert(format!(
            "mixture:{:?}:signals:{:?}",
            arm.student, arm.signals
        ));
        for first in 0..arm.signals.len() {
            for second in first + 1..arm.signals.len() {
                result.insert(format!(
                    "mixture:{:?}:pair:{:?}+{:?}",
                    arm.student, arm.signals[first], arm.signals[second]
                ));
            }
        }
    }
    if arm.contents.len() > 1 {
        result.insert(format!(
            "mixture:{:?}:contents:{:?}",
            arm.student, arm.contents
        ));
    }
    if arm.contents.contains(&ContentMethod::Preference) {
        for signal in &arm.signals {
            result.insert(format!("mixture:{:?}:preference+{signal:?}", arm.student));
        }
    }
    result
}

/// First cover isolated operators and organiser axes. Then replicate selected
/// groups across their preregistered seeds and spend the remaining bounded
/// budget on signal/content mixtures, with each new group retaining its seeds.
/// Unavailable or unaffordable combinations remain explicit catalogue gaps.
pub(super) fn quick_cover(
    arms: &[ArmSpec],
    maximum: usize,
    training: &FidelityConfig,
) -> (Vec<String>, Vec<String>) {
    let maximum = maximum.min(training.maximum_trials).min(
        training
            .maximum_total_steps
            .checked_div(training.steps as u64)
            .unwrap_or(0)
            .min(usize::MAX as u64) as usize,
    );
    let mut wanted = arms.iter().flat_map(coverage).collect::<BTreeSet<_>>();
    wanted.extend(arms.iter().flat_map(mixture_coverage));
    let mut covered = BTreeSet::new();
    let mut selected = Vec::new();
    let mut available = arms
        .iter()
        .filter(|a| a.applicable && fidelity_inapplicable_reasons(a, training).is_empty())
        .collect::<Vec<_>>();
    available.sort_by_key(|a| {
        (
            a.signals.len() + a.contents.len(),
            a.teacher_ids.len(),
            a.progressive,
            a.self_distillation,
            a.id.clone(),
        )
    });
    for arm in &available {
        if selected.len() >= maximum {
            break;
        }
        let family_isolation = arm.signals.len() == 1
            && arm.contents.len() <= 1
            && !arm.progressive
            && !arm.self_distillation;
        let features = coverage(arm);
        if family_isolation && features.iter().any(|f| !covered.contains(f)) {
            selected.push(arm.id.clone());
            covered.extend(features);
        }
    }
    while selected.len() < maximum {
        let best = available
            .iter()
            .filter(|a| !selected.contains(&a.id))
            .map(|a| (a, coverage(a)))
            .max_by(|(a, af), (b, bf)| {
                af.difference(&covered)
                    .count()
                    .cmp(&bf.difference(&covered).count())
                    .then_with(|| {
                        (b.signals.len() + b.contents.len() + b.teacher_ids.len())
                            .cmp(&(a.signals.len() + a.contents.len() + a.teacher_ids.len()))
                    })
                    .then_with(|| b.id.cmp(&a.id))
            });
        let Some((arm, features)) = best else {
            break;
        };
        if features.is_subset(&covered) {
            break;
        }
        selected.push(arm.id.clone());
        covered.extend(features);
    }
    // Family coverage alone is insufficient: otherwise a four-signal bundle
    // can stop after a handful of isolated arms despite a 24-trial budget.
    let mut groups = BTreeMap::<String, Vec<&ArmSpec>>::new();
    for arm in &available {
        groups.entry(method_group(arm)).or_default().push(arm);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|arm| (arm.seed, &arm.id));
    }
    let mut chosen = available
        .iter()
        .filter(|arm| selected.contains(&arm.id))
        .map(|arm| method_group(arm))
        .collect::<BTreeSet<_>>();
    for key in &chosen {
        for arm in &groups[key] {
            if selected.len() >= maximum {
                break;
            }
            if !selected.contains(&arm.id) {
                selected.push(arm.id.clone());
            }
        }
    }
    covered.extend(
        available
            .iter()
            .filter(|arm| selected.contains(&arm.id))
            .flat_map(|arm| mixture_coverage(arm)),
    );
    while selected.len() < maximum {
        let remaining = maximum - selected.len();
        let best = groups
            .iter()
            .filter(|(key, group)| !chosen.contains(*key) && group.len() <= remaining)
            .filter(|(_, group)| !mixture_coverage(group[0]).is_empty())
            .map(|(key, group)| (key, group, mixture_coverage(group[0])))
            .max_by(|(ak, ag, af), (bk, bg, bf)| {
                af.difference(&covered)
                    .count()
                    .cmp(&bf.difference(&covered).count())
                    .then_with(|| {
                        (bg[0].signals.len() + bg[0].contents.len() + bg[0].teacher_ids.len()).cmp(
                            &(ag[0].signals.len() + ag[0].contents.len() + ag[0].teacher_ids.len()),
                        )
                    })
                    .then_with(|| bk.cmp(ak))
            });
        let Some((key, group, features)) = best else {
            break;
        };
        chosen.insert(key.clone());
        for arm in group {
            selected.push(arm.id.clone());
            covered.extend(coverage(arm));
        }
        covered.extend(features);
    }
    (selected, wanted.difference(&covered).cloned().collect())
}
