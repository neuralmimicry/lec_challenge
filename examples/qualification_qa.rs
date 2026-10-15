//! Trusted qualification references run in fresh isolated containers only.
use anyhow::{Result, ensure};
use lec_distil::{
    artefact,
    sandbox::{self, SandboxConfig},
    teacher_search::qualification,
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() -> Result<()> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "reports/comparative/search/qualification-qa/{}",
                uuid::Uuid::new_v4()
            ))
        });
    ensure!(
        !output.join("report.json").exists(),
        "retain previous qualification QA evidence"
    );
    std::fs::create_dir_all(&output)?;
    let frozen = qualification::freeze(&output.join("qualification-pack.json"))?;
    let mut cfg = SandboxConfig::default();
    cfg.expected_image_id = sandbox::policy_evidence(&cfg).await?.1;
    cfg.report_dir = output.join("runs");
    let preflight = sandbox::preflight(&cfg).await?;
    artefact::write_json(&output.join("preflight.json"), &preflight)?;
    let started = Instant::now();
    let mut observations = Vec::new();
    let mut complete = true;
    let tasks = qualification::corpus();
    for group in tasks.chunks(2) {
        let mut work = tokio::task::JoinSet::new();
        for task in group {
            let task = task.clone();
            let cfg = cfg.clone();
            work.spawn(async move {
                let source = qualification::reference_source(&task.id)?;
                let report = sandbox::grade(&task, &source, &cfg).await?;
                sandbox::verify_report(&report, &cfg, &task, &source)?;
                Ok::<_, anyhow::Error>((task, report))
            });
        }
        while let Some(result) = work.join_next().await {
            match result {
                Ok(Ok((task, report))) => {
                    let passed = report.passed && !report.quarantined && report.trace_complete;
                    complete &= passed;
                    eprintln!(
                        "qualification-reference: {} {} ({}/{})",
                        if passed { "PASS" } else { "FAIL" },
                        task.id,
                        report.cases_passed,
                        report.cases_total
                    );
                    observations
                        .push(json!({"task_id":task.id,"passed":passed,"sandbox_report":report}));
                }
                Ok(Err(error)) => {
                    complete = false;
                    observations.push(json!({"passed":false,"error":format!("{error:#}")}));
                }
                Err(error) => {
                    complete = false;
                    observations.push(json!({"passed":false,"worker_error":error.to_string()}));
                }
            }
            artefact::write_json(&output.join("progress.json"), &observations)?;
        }
    }
    complete &= observations.len() == 16;
    let report = json!({"schema_version":1,"evidence_kind":"actual_trusted_reference_programmes_in_isolated_containers","pack_id":qualification::PACK_ID,"pack_sha256":frozen,"planned":16,"observations":observations,"complete":complete,"elapsed_seconds":started.elapsed().as_secs_f64(),"maximum_concurrent_graders":2,"learning_or_promotion_evidence":false});
    artefact::write_json(&output.join("report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"report":output.join("report.json"),"complete":complete,"planned":16,"elapsed_seconds":started.elapsed().as_secs_f64()})
        )?
    );
    ensure!(
        complete,
        "qualification reference QA failed; failed observations retained"
    );
    Ok(())
}
