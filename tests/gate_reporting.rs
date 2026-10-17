//! Non-default gate settings must survive reporting and historical readback.
//! These are statistical fixtures, never measured checkpoint evaluations.
use anyhow::Result;
use lec_distil::gate::{self, GateConfig, GateReport, Pair};
use std::path::Path;

fn diagnostic_config() -> Result<GateConfig> {
    let mut cfg = GateConfig::load(Path::new("config/gate.toml"))?;
    cfg.alpha = 0.10;
    cfg.power_target = 0.90;
    cfg.bootstrap_trials = 100;
    cfg.power_trials = 10;
    Ok(cfg)
}

#[test]
fn confidence_labels_follow_frozen_alpha_and_survive_roundtrip() -> Result<()> {
    let cfg = diagnostic_config()?;
    let pairs = (0..8)
        .flat_map(|source| {
            cfg.evaluation_seeds.iter().map(move |&seed| Pair {
                source_id: format!("fixture-source-{source}"),
                task_id: format!("fixture-task-{source}"),
                seed,
                incumbent_pass: true,
                candidate_pass: true,
            })
        })
        .collect::<Vec<_>>();
    let report = gate::decide(&pairs, &cfg, true, "synthetic_reporting_fixture");
    let restored: GateReport = serde_json::from_slice(&serde_json::to_vec(&report)?)?;
    assert_eq!(restored.confidence_level, Some(0.90));
    assert!(restored.markdown().contains("Source t interval (90.00%)"));
    assert!(restored.markdown().contains("Source bootstrap (90.00%)"));
    assert!(!restored.markdown().contains("95%"));

    // An earlier report has no alpha identity. Its reader must not retrofit
    // today's settings or claim the historical interval had 95% confidence.
    let mut historical = serde_json::to_value(&report)?;
    historical
        .as_object_mut()
        .unwrap()
        .remove("confidence_level");
    let historical: GateReport = serde_json::from_value(historical)?;
    assert_eq!(historical.confidence_level, None);
    assert!(historical.markdown().contains("confidence not retained"));
    assert!(!historical.markdown().contains("95%"));
    Ok(())
}

#[test]
fn detectability_labels_use_the_actual_power_target() -> Result<()> {
    let cfg = diagnostic_config()?;
    let report = gate::simulate_power(&cfg, 8, 4, &[0.0, 0.5], &[0.5], &[0.0])?;
    assert!(report.markdown().contains("estimated 90.00% power"));
    assert!(report.interpretation.contains("reaching 90.00% power"));
    assert!(!report.interpretation.contains("80%"));
    Ok(())
}

#[test]
fn empty_sensitivity_assumptions_cannot_report_detectability() -> Result<()> {
    let cfg = diagnostic_config()?;
    assert!(gate::simulate_power(&cfg, 8, 4, &[0.0], &[], &[0.0]).is_err());
    assert!(gate::simulate_power(&cfg, 8, 4, &[0.0], &[0.5], &[]).is_err());
    Ok(())
}
