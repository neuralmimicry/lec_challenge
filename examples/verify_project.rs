//! Explicit local QA, with captured native-tool evidence and separate assessment status.
use anyhow::{Context, Result, ensure};
use lec_distil::artefact;
use serde_json::json;
use std::{fs, path::Path, process::Command, time::Instant};

fn main() -> Result<()> {
    fs::create_dir_all("reports")?;
    let checks: [(&str, &[&str]); 5] = [
        ("format", &["fmt", "--all", "--", "--check"]),
        (
            "clippy",
            &[
                "clippy",
                "--locked",
                "--offline",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        ("unit", &["test", "--locked", "--offline"]),
        (
            "original_tokeniser",
            &[
                "test",
                "--locked",
                "--offline",
                "original_tokenizer_masks_prompt_and_preserves_assistant_eos",
                "--",
                "--ignored",
            ],
        ),
        (
            "real_container_qa",
            &[
                "test",
                "--locked",
                "--offline",
                "--test",
                "sandbox_contract",
                "--",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ],
        ),
    ];
    let root = std::env::current_dir()?;
    let mut steps = Vec::new();
    let mut passed = true;
    for (name, args) in checks {
        eprintln!("verification: START {name}");
        let start = Instant::now();
        let output = Command::new("cargo")
            .args(args)
            .output()
            .with_context(|| format!("start {name}"))?;
        let mut log = output.stdout;
        log.extend_from_slice(&output.stderr);
        let path = format!("reports/verification-{name}.log");
        artefact::atomic_write(Path::new(&path), &log)?;
        let success = output.status.success();
        passed &= success;
        eprintln!(
            "verification: {} {name} {:.2}s",
            if success { "PASSED" } else { "FAILED" },
            start.elapsed().as_secs_f64()
        );
        steps.push(json!({"name":name,"command":args,"success":success,"exit_code":output.status.code(),"elapsed_seconds":start.elapsed().as_secs_f64(),"evidence":artefact::Artefact::capture(&root,path)?}));
    }
    let skill = fs::read_to_string(".agents/skills/rust-distillation-evidence/SKILL.md")?;
    let frontmatter = skill
        .strip_prefix("---\n")
        .and_then(|s| s.split_once("\n---\n"))
        .context("project skill frontmatter missing")?
        .0;
    ensure!(
        frontmatter
            .lines()
            .any(|l| l == "name: rust-distillation-evidence")
            && frontmatter.lines().any(|l| l.starts_with("description: ")),
        "project skill metadata invalid"
    );
    let notices: serde_json::Value =
        serde_json::from_slice(&fs::read("offline_bundle/notices/manifest.json")?)?;
    ensure!(
        notices["remaining_notice_gaps"]
            .as_array()
            .is_some_and(|a| a.is_empty()),
        "original dependency notice gaps remain"
    );
    ensure!(
        notices["cargo_lock_sha256"] == artefact::file_hash(Path::new("Cargo.lock"))?,
        "dependency notice lock is stale"
    );
    let demo = fs::read("reports/demo.json")
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .unwrap_or_default();
    let report = json!({"schema_version":1,"observed_unix_ms":lec_distil::ledger::unix_millis(),"local_quality_checks_passed":passed,"assessment_complete":demo["status"]=="COMPLETE","steps":steps,"project_skill_validated":true,"resolved_dependency_packages":notices["packages"].as_array().map(Vec::len),"original_notice_gaps":0,"evidence_rule":"Passing local QA does not replace absent real provider recordings, approved data, assessment checkpoints or measured held-out learning."});
    artefact::write_json(Path::new("reports/verification.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(passed, "one or more local QA steps failed");
    Ok(())
}
