//! Predeclared source-cluster promotion decision; fixtures are never model evidence.
use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ContinuousCDF, StudentsT};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    pub alpha: f64,
    pub practical_margin: f64,
    pub bootstrap_trials: usize,
    pub bootstrap_seed: u64,
    pub evaluation_seeds: Vec<u64>,
    pub minimum_sources: usize,
    pub maximum_source_regression: f64,
    pub power_trials: usize,
    pub power_seed: u64,
    pub power_target: f64,
}
impl GateConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let value: Self = toml::from_str(&std::fs::read_to_string(path)?)?;
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.alpha > 0.0 && self.alpha < 0.5, "invalid alpha");
        ensure!(
            (0.0..1.0).contains(&self.practical_margin),
            "invalid practical margin"
        );
        ensure!(
            self.bootstrap_trials >= 100 && self.minimum_sources >= 2,
            "insufficient interval settings"
        );
        ensure!(
            self.evaluation_seeds.len() >= 2
                && self.evaluation_seeds.iter().collect::<BTreeSet<_>>().len()
                    == self.evaluation_seeds.len(),
            "seeds must be distinct"
        );
        ensure!(
            (0.0..=1.0).contains(&self.maximum_source_regression),
            "invalid slice ceiling"
        );
        ensure!(
            self.power_trials >= 10 && self.power_target > 0.5 && self.power_target < 1.0,
            "invalid power settings"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pair {
    pub source_id: String,
    pub task_id: String,
    pub seed: u64,
    pub incumbent_pass: bool,
    pub candidate_pass: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Decision {
    Ship,
    NoShipRegression,
    NoShipInconclusive,
    InvalidEvaluation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateReport {
    pub decision: Decision,
    pub reason: String,
    pub effect: Option<f64>,
    pub incumbent_rate: Option<f64>,
    pub candidate_rate: Option<f64>,
    /// Older reports did not retain alpha. Do not infer their confidence from
    /// today's configuration when rendering historical evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_level: Option<f64>,
    pub t_interval: Option<Interval>,
    pub source_bootstrap_interval: Option<Interval>,
    pub source_effects: BTreeMap<String, f64>,
    pub sources: usize,
    pub tasks: usize,
    pub observations: usize,
    pub practical_margin: f64,
    pub evidence_kind: String,
    pub limitations: Vec<String>,
}
impl GateReport {
    pub fn invalid(reason: impl Into<String>, cfg: &GateConfig) -> Self {
        Self {
            decision: Decision::InvalidEvaluation,
            reason: reason.into(),
            effect: None,
            incumbent_rate: None,
            candidate_rate: None,
            confidence_level: (cfg.alpha > 0.0 && cfg.alpha < 0.5).then_some(1.0 - cfg.alpha),
            t_interval: None,
            source_bootstrap_interval: None,
            source_effects: BTreeMap::new(),
            sources: 0,
            tasks: 0,
            observations: 0,
            practical_margin: cfg.practical_margin,
            evidence_kind: "invalid_or_missing_evidence".into(),
            limitations: limitations(),
        }
    }
    pub fn markdown(&self) -> String {
        let pct = |x: Option<f64>| {
            x.map(|v| format!("{:.2}%", 100.0 * v))
                .unwrap_or_else(|| "unavailable".into())
        };
        let interval = |x: Option<Interval>| {
            x.map(|v| {
                format!(
                    "[{:.2}, {:.2}] percentage points",
                    100.0 * v.lower,
                    100.0 * v.upper
                )
            })
            .unwrap_or_else(|| "unavailable".into())
        };
        let confidence = self
            .confidence_level
            .map(|v| format!("{:.2}%", 100.0 * v))
            .unwrap_or_else(|| "confidence not retained in historical report".into());
        format!(
            "# Promotion evaluation\n\nDecision: **{:?}**\n\n{}\n\nEvidence: {}. Incumbent: {}; candidate: {}; difference: {}.\n\nSource t interval ({confidence}): {}. Source bootstrap ({confidence}): {}. Ship margin: {:.2} points.\n\n{} sources; {} tasks; {} paired observations. Seeds do not increase the independent source count.\n\n{}\n",
            self.decision,
            self.reason,
            self.evidence_kind,
            pct(self.incumbent_rate),
            pct(self.candidate_rate),
            pct(self.effect),
            interval(self.t_interval),
            interval(self.source_bootstrap_interval),
            100.0 * self.practical_margin,
            self.sources,
            self.tasks,
            self.observations,
            self.limitations
                .iter()
                .map(|s| format!("- {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
fn limitations() -> Vec<String> {
    vec!["Eight locally authored source groups give fragile interval coverage; shared authorship may create dependence between groups.".into(),"A confidence interval containing zero is insufficient evidence of improvement, and does not prove equality.".into(),"Public-algorithm familiarity and pretraining contamination cannot be ruled out by novel local wording.".into(),"A NO SHIP result retains the incumbent; train gains cannot rescue a held-out regression.".into()]
}

/// This computes statistics only. The public command first verifies every artifact.
pub fn decide(
    pairs: &[Pair],
    cfg: &GateConfig,
    operational_ok: bool,
    evidence_kind: &str,
) -> GateReport {
    let compute = || -> Result<GateReport> {
        cfg.validate()?;
        ensure!(operational_ok, "security or operational checks failed");
        ensure!(!pairs.is_empty(), "no paired outcomes");
        let mut grouped: BTreeMap<&str, BTreeMap<&str, BTreeMap<u64, &Pair>>> = BTreeMap::new();
        let mut task_sources = BTreeMap::new();
        for p in pairs {
            ensure!(
                !p.source_id.is_empty() && !p.task_id.is_empty(),
                "missing source/task identity"
            );
            if let Some(previous) = task_sources.insert(&p.task_id, &p.source_id) {
                ensure!(
                    previous == &p.source_id,
                    "task appears under multiple sources"
                );
            }
            ensure!(
                cfg.evaluation_seeds.contains(&p.seed),
                "unregistered decoding seed"
            );
            ensure!(
                grouped
                    .entry(&p.source_id)
                    .or_default()
                    .entry(&p.task_id)
                    .or_default()
                    .insert(p.seed, p)
                    .is_none(),
                "duplicate task/seed"
            );
        }
        ensure!(
            grouped.len() >= cfg.minimum_sources,
            "too few independent source groups"
        );
        let mut effects = BTreeMap::new();
        let mut incumbent = 0.0;
        let mut candidate = 0.0;
        let mut tasks = 0;
        for (source, source_tasks) in &grouped {
            let mut effect = 0.0;
            let mut base = 0.0;
            let mut next = 0.0;
            for seeds in source_tasks.values() {
                ensure!(
                    seeds.len() == cfg.evaluation_seeds.len()
                        && cfg.evaluation_seeds.iter().all(|s| seeds.contains_key(s)),
                    "missing task/seed outcome"
                );
                let b =
                    seeds.values().filter(|p| p.incumbent_pass).count() as f64 / seeds.len() as f64;
                let n =
                    seeds.values().filter(|p| p.candidate_pass).count() as f64 / seeds.len() as f64;
                base += b;
                next += n;
                effect += n - b;
                tasks += 1;
            }
            let n = source_tasks.len() as f64;
            effects.insert((*source).to_owned(), effect / n);
            incumbent += base / n;
            candidate += next / n;
        }
        let n = effects.len() as f64;
        incumbent /= n;
        candidate /= n;
        let values = effects.values().copied().collect::<Vec<_>>();
        let (effect, t_interval, boot) = intervals(&values, cfg)?;
        let slice_bad = values.iter().any(|&x| x < -cfg.maximum_source_regression);
        let decision = if effect < 0.0 || slice_bad {
            Decision::NoShipRegression
        } else if t_interval.lower > cfg.practical_margin && boot.lower > cfg.practical_margin {
            Decision::Ship
        } else {
            Decision::NoShipInconclusive
        };
        let reason=match decision {Decision::Ship=>"Both predeclared interval lower bounds exceed the practical margin and all checks pass.",Decision::NoShipRegression=>"Measured held-out difference is negative or a predeclared source slice regresses beyond its ceiling.",_=>"The complete evaluation does not establish an improvement beyond the practical margin."}.into();
        Ok(GateReport {
            decision,
            reason,
            effect: Some(effect),
            incumbent_rate: Some(incumbent),
            candidate_rate: Some(candidate),
            confidence_level: Some(1.0 - cfg.alpha),
            t_interval: Some(t_interval),
            source_bootstrap_interval: Some(boot),
            source_effects: effects,
            sources: n as usize,
            tasks,
            observations: pairs.len(),
            practical_margin: cfg.practical_margin,
            evidence_kind: evidence_kind.into(),
            limitations: limitations(),
        })
    };
    match compute() {
        Ok(r) => r,
        Err(e) => GateReport::invalid(format!("{e:#}"), cfg),
    }
}

fn intervals(values: &[f64], cfg: &GateConfig) -> Result<(f64, Interval, Interval)> {
    ensure!(
        values.len() >= 2 && values.iter().all(|x| x.is_finite()),
        "invalid source effects"
    );
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let quantile = StudentsT::new(0.0, 1.0, n - 1.0)?.inverse_cdf(1.0 - cfg.alpha / 2.0);
    let half = quantile * (variance / n).sqrt();
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.bootstrap_seed);
    let mut boot = Vec::with_capacity(cfg.bootstrap_trials);
    for _ in 0..cfg.bootstrap_trials {
        boot.push(
            (0..values.len())
                .map(|_| values[rng.random_range(0..values.len())])
                .sum::<f64>()
                / n,
        );
    }
    boot.sort_by(f64::total_cmp);
    let lo = ((cfg.alpha / 2.0) * (boot.len() - 1) as f64).floor() as usize;
    let hi = ((1.0 - cfg.alpha / 2.0) * (boot.len() - 1) as f64).ceil() as usize;
    Ok((
        mean,
        Interval {
            lower: mean - half,
            upper: mean + half,
        },
        Interval {
            lower: boot[lo],
            upper: boot[hi],
        },
    ))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PowerPoint {
    pub effect: f64,
    pub discordance: f64,
    pub source_correlation: f64,
    pub estimated_power: f64,
    pub monte_carlo_standard_error: f64,
    pub monte_carlo_interval: Interval,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PowerReport {
    pub evidence_kind: String,
    pub gate_configuration: GateConfig,
    pub tested_effects: Vec<f64>,
    pub seed: u64,
    pub trials: usize,
    pub sources: usize,
    pub tasks_per_source: usize,
    pub decoding_repeats: usize,
    pub points: Vec<PowerPoint>,
    pub mde_by_assumption: BTreeMap<String, Option<f64>>,
    pub interpretation: String,
}
impl PowerReport {
    pub fn markdown(&self) -> String {
        let mut md = format!(
            "# Gate detectability sensitivity\n\nSimulation only; no learning result. {} source clusters, {} tasks per source and {} shared decoding repeats. {} trials per point; {} bootstrap resamples; simulation seed {}. The source sharing construction has the stated pairwise intraclass correlation.\n\n| Assumed discordance / source correlation | Smallest tested effect at estimated {:.2}% power |\n| --- | ---: |\n",
            self.sources,
            self.tasks_per_source,
            self.decoding_repeats,
            self.trials,
            self.gate_configuration.bootstrap_trials,
            self.seed,
            100.0 * self.gate_configuration.power_target
        );
        for (assumption, mde) in &self.mde_by_assumption {
            md.push_str(&format!(
                "| {assumption} | {} |\n",
                mde.map(|v| format!("{:.1} percentage points", 100.0 * v))
                    .unwrap_or_else(|| "MDE above tested range".into())
            ));
        }
        md.push_str(&format!("\n{}\n\nEach point in power.json also reports Monte Carlo standard error and an exact binomial 95% interval, including zero/one estimated power. A reduced diagnostic run is identified by its complete embedded configuration; only the frozen full configuration is the assessment gate's power calculation. Development calibration and shared-authorship limitations still apply.\n",self.interpretation));
        md
    }
}

fn sample_source_effect(rng: &mut ChaCha8Rng, tasks: usize, q: f64, effect: f64, rho: f64) -> f64 {
    let common_discord = rng.random::<f64>() < q;
    let common_win = rng.random::<f64>() < (1.0 + effect / q) / 2.0;
    let mut sum = 0.0;
    for _ in 0..tasks {
        // Two tasks share their source draw with probability p². Choosing p as
        // sqrt(rho) makes rho the resulting intraclass correlation, rather than
        // accidentally reporting a mixing probability as the correlation.
        let shared = rng.random::<f64>() < rho.sqrt();
        let discord = if shared {
            common_discord
        } else {
            rng.random::<f64>() < q
        };
        let win = if shared {
            common_win
        } else {
            rng.random::<f64>() < (1.0 + effect / q) / 2.0
        };
        sum += if discord {
            if win { 1.0 } else { -1.0 }
        } else {
            0.0
        };
    }
    sum / tasks as f64
}

fn bootstrap_draws(sources: usize, cfg: &GateConfig) -> Vec<Vec<usize>> {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.bootstrap_seed);
    (0..cfg.bootstrap_trials)
        .map(|_| (0..sources).map(|_| rng.random_range(0..sources)).collect())
        .collect()
}
/// Reuse exactly the public gate's deterministic resamples across simulations.
/// A lower percentile exceeds the margin iff at most its zero-based index of
/// resamples are <= the margin. This avoids repeated RNG, sorting and allocation.
fn simulated_ship(values: &[f64], cfg: &GateConfig, draws: &[Vec<usize>], t_critical: f64) -> bool {
    if values.iter().any(|x| *x < -cfg.maximum_source_regression) {
        return false;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    if mean - t_critical * (variance / n).sqrt() <= cfg.practical_margin {
        return false;
    }
    let allowed = ((cfg.alpha / 2.0) * (draws.len() - 1) as f64).floor() as usize;
    let mut below = 0;
    for draw in draws {
        let boot_mean = draw.iter().map(|&i| values[i]).sum::<f64>() / n;
        if boot_mean <= cfg.practical_margin {
            below += 1;
            if below > allowed {
                return false;
            }
        }
    }
    true
}

/// Development-calibrated sensitivity simulation, never fitted to sealed outcomes.
/// A source shock changes direction probabilities; shared task outcomes are reused
/// across decoding repeats to avoid pretending that seeds are independent tasks.
pub fn simulate_power(
    cfg: &GateConfig,
    sources: usize,
    tasks_per_source: usize,
    effects: &[f64],
    discordances: &[f64],
    correlations: &[f64],
) -> Result<PowerReport> {
    cfg.validate()?;
    ensure!(
        sources >= cfg.minimum_sources && tasks_per_source > 0,
        "invalid source structure"
    );
    ensure!(
        !effects.is_empty() && effects.windows(2).all(|w| w[0] < w[1]),
        "tested effects must be strictly increasing to identify the smallest tested detectable effect"
    );
    ensure!(
        !discordances.is_empty() && !correlations.is_empty(),
        "detectability requires discordance and source-correlation assumptions"
    );
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.power_seed);
    let mut points = Vec::new();
    let mut mdes = BTreeMap::new();
    let draws = bootstrap_draws(sources, cfg);
    let t_critical =
        StudentsT::new(0.0, 1.0, (sources - 1) as f64)?.inverse_cdf(1.0 - cfg.alpha / 2.0);
    for &q in discordances {
        ensure!(q > 0.0 && q <= 1.0, "invalid discordance");
        for &rho in correlations {
            ensure!((0.0..=1.0).contains(&rho), "invalid source correlation");
            let mut mde = None;
            for &effect in effects {
                ensure!((0.0..=1.0).contains(&effect), "invalid tested effect");
                if effect > q {
                    continue;
                }
                let mut ships = 0usize;
                for _ in 0..cfg.power_trials {
                    let mut values = Vec::with_capacity(sources);
                    for _ in 0..sources {
                        // Task outcome is shared across the prescribed seeds.
                        values.push(sample_source_effect(
                            &mut rng,
                            tasks_per_source,
                            q,
                            effect,
                            rho,
                        ));
                    }
                    if simulated_ship(&values, cfg, &draws, t_critical) {
                        ships += 1;
                    }
                }
                let p = ships as f64 / cfg.power_trials as f64;
                if p >= cfg.power_target && mde.is_none() {
                    mde = Some(effect);
                }
                let lo = if ships == 0 {
                    0.0
                } else {
                    Beta::new(ships as f64, (cfg.power_trials - ships + 1) as f64)?
                        .inverse_cdf(0.025)
                };
                let hi = if ships == cfg.power_trials {
                    1.0
                } else {
                    Beta::new((ships + 1) as f64, (cfg.power_trials - ships) as f64)?
                        .inverse_cdf(0.975)
                };
                points.push(PowerPoint {
                    effect,
                    discordance: q,
                    source_correlation: rho,
                    estimated_power: p,
                    monte_carlo_standard_error: (p * (1.0 - p) / cfg.power_trials as f64).sqrt(),
                    monte_carlo_interval: Interval {
                        lower: lo,
                        upper: hi,
                    },
                });
            }
            mdes.insert(format!("discordance={q:.2}, correlation={rho:.2}"), mde);
        }
    }
    Ok(PowerReport {
        evidence_kind: "simulation_assumptions_not_measured_learning".into(),
        gate_configuration: cfg.clone(),
        tested_effects: effects.to_vec(),
        seed: cfg.power_seed,
        trials: cfg.power_trials,
        sources,
        tasks_per_source,
        decoding_repeats: cfg.evaluation_seeds.len(),
        points,
        mde_by_assumption: mdes,
        interpretation: format!(
            "MDE is the first tested effect reaching {:.2}% power, not an interpolated precision claim; null means MDE above tested range or unachievable at the specified discordance. Includes the exact reported t/bootstrap/slice gate configuration, with maximally shared seed outcomes; reduced settings are diagnostic, not the final predeclared MDE. Task/source dependence assumptions require development calibration and sensitivity review.",
            100.0 * cfg.power_target
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> GateConfig {
        GateConfig::load(Path::new("config/gate.toml")).unwrap()
    }
    fn pairs(base: bool, next: bool) -> Vec<Pair> {
        (0..8)
            .flat_map(|s| {
                (0..4).flat_map(move |t| {
                    [1729, 2718, 3141].map(move |seed| Pair {
                        source_id: format!("s{s}"),
                        task_id: format!("s{s}t{t}"),
                        seed,
                        incumbent_pass: base,
                        candidate_pass: next,
                    })
                })
            })
            .collect()
    }
    #[test]
    fn identical_and_degraded_are_never_wins() {
        let c = cfg();
        assert_eq!(
            decide(&pairs(true, true), &c, true, "synthetic_fixture").decision,
            Decision::NoShipInconclusive
        );
        assert_eq!(
            decide(&pairs(true, false), &c, true, "synthetic_fixture").decision,
            Decision::NoShipRegression
        );
    }
    #[test]
    fn strong_consistent_gain_and_margin() {
        let c = cfg();
        assert_eq!(
            decide(&pairs(false, true), &c, true, "synthetic_fixture").decision,
            Decision::Ship
        );
        let mut p = pairs(true, true);
        p[0].candidate_pass = false;
        assert_ne!(
            decide(&p, &c, true, "synthetic_fixture").decision,
            Decision::Ship
        );
    }
    #[test]
    fn missing_duplicate_and_security_are_invalid() {
        let c = cfg();
        let mut p = pairs(false, true);
        p.pop();
        assert_eq!(
            decide(&p, &c, true, "synthetic_fixture").decision,
            Decision::InvalidEvaluation
        );
        let mut p = pairs(false, true);
        p.push(p[0].clone());
        assert_eq!(
            decide(&p, &c, true, "synthetic_fixture").decision,
            Decision::InvalidEvaluation
        );
        assert_eq!(
            decide(&pairs(false, true), &c, false, "synthetic_fixture").decision,
            Decision::InvalidEvaluation
        );
    }
    #[test]
    fn bootstrap_reproduces() {
        let c = cfg();
        let mut p = pairs(false, true);
        p[0].candidate_pass = false;
        let a = decide(&p, &c, true, "synthetic_fixture");
        let b = decide(&p, &c, true, "synthetic_fixture");
        assert_eq!(
            a.source_bootstrap_interval.unwrap().lower,
            b.source_bootstrap_interval.unwrap().lower
        );
    }
    #[test]
    fn simulated_source_moments_match_declared_effect_and_correlation() {
        let mut rng = ChaCha8Rng::seed_from_u64(92261);
        let n = 30000;
        let q = 0.5;
        let effect = 0.1;
        let rho = 0.25;
        let tasks = 4;
        let values = (0..n)
            .map(|_| sample_source_effect(&mut rng, tasks, q, effect, rho))
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let expected = (q - effect * effect) * (1.0 + (tasks - 1) as f64 * rho) / tasks as f64;
        assert!((mean - effect).abs() < 0.015);
        assert!(
            (variance - expected).abs() < 0.015,
            "{variance} versus {expected}"
        );
    }
    #[test]
    fn optimised_power_rule_matches_full_public_intervals() {
        let mut c = cfg();
        c.bootstrap_trials = 1000;
        let draws = bootstrap_draws(8, &c);
        let t = StudentsT::new(0.0, 1.0, 7.0).unwrap().inverse_cdf(0.975);
        let mut rng = ChaCha8Rng::seed_from_u64(73261);
        for _ in 0..100 {
            let values = (0..8)
                .map(|_| rng.random_range(-4..=12) as f64 / 12.0)
                .collect::<Vec<_>>();
            let (_, a, b) = intervals(&values, &c).unwrap();
            let expected = a.lower > c.practical_margin
                && b.lower > c.practical_margin
                && values.iter().all(|x| *x >= -c.maximum_source_regression);
            assert_eq!(simulated_ship(&values, &c, &draws, t), expected);
        }
    }
}
