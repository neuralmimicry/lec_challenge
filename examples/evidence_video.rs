//! Render an honest, captioned evidence walkthrough from actual repository reports.
//! Rust owns the evidence and captions; ffmpeg is used only as a media encoder.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Serialize)]
struct VideoManifest {
    evidence_kind: String,
    duration_seconds: u64,
    width: u32,
    height: u32,
    input_sha256: BTreeMap<String, String>,
    video_sha256: String,
    slides: Vec<Slide>,
    limitations: Vec<String>,
}
#[derive(Serialize)]
struct Slide {
    title: String,
    caption: String,
}

fn read(path: &str, hashes: &mut BTreeMap<String, String>) -> Result<Option<Value>> {
    let p = Path::new(path);
    if !p.exists() {
        return Ok(None);
    }
    hashes.insert(path.into(), lec_distil::model::sha256_file(p)?);
    Ok(Some(serde_json::from_slice(&std::fs::read(p)?)?))
}
fn field(value: &Value, key: &str) -> String {
    match &value[key] {
        Value::String(s) => s.clone(),
        Value::Number(n) if n.is_f64() => format!("{:.6}", n.as_f64().unwrap_or_default()),
        Value::Null => "unavailable".into(),
        other => other.to_string(),
    }
}
fn case_status(drill: &Option<Value>, name: &str) -> String {
    drill
        .as_ref()
        .and_then(|d| d["cases"].as_array())
        .and_then(|cases| cases.iter().find(|c| c["name"] == name))
        .map(|c| field(c, "status"))
        .unwrap_or_else(|| "REPORT UNAVAILABLE".into())
}
fn wrap(text: &str, width: usize) -> String {
    text.lines()
        .map(|line| {
            let mut result = String::new();
            let mut count = 0;
            for word in line.split_whitespace() {
                let len = word.chars().count();
                if count > 0 && count + 1 + len > width {
                    result.push('\n');
                    count = 0;
                }
                if count > 0 {
                    result.push(' ');
                    count += 1;
                }
                result.push_str(word);
                count += len;
            }
            result
        })
        .collect::<Vec<_>>()
        .join("\n")
}
fn slide(title: &str, caption: impl Into<String>) -> Slide {
    Slide {
        title: title.into(),
        caption: wrap(&caption.into(), 64),
    }
}

fn main() -> Result<()> {
    let mut hashes = BTreeMap::new();
    let base = read("reports/base-inference-smoke.json", &mut hashes)?
        .context("actual base inference report required")?;
    let gradient = read("reports/native-gradient-512.json", &mut hashes)?
        .context("actual native gradient QA report required")?;
    let serving = read("reports/serving-smoke.json", &mut hashes)?
        .context("actual serving report required")?;
    let drill = read("reports/drill.json", &mut hashes)?;
    let eval = read("reports/eval.json", &mut hashes)?;
    let _costs = read("reports/cost-summary.json", &mut hashes)?;
    let design = Path::new("docs/design.md");
    hashes.insert(
        "docs/design.md".into(),
        lec_distil::model::sha256_file(design)?,
    );
    let fleet = drill.as_ref().and_then(|d| d["invariant_qa"].get("fleet"));
    let fleet_summary = fleet
        .map(|f| {
            format!(
                "Recorded local fault observations: {}",
                f.get("evidence_kind")
                    .and_then(Value::as_str)
                    .unwrap_or("labelled fixture transport invariants")
            )
        })
        .unwrap_or_else(|| "Local runner report not yet present.".into());
    let slides = vec![
        slide(
            "Principal AI Engineer assessment",
            "Native Rust implementation and measured local evidence.\nThis is a captioned walkthrough, not a screen recording.\nThe assessment remains INCOMPLETE.\nNo provider-trained student or quality improvement is claimed.\nReports and code are the reviewable evidence.",
        ),
        slide(
            "Design: the ceiling drives the loop",
            "Tasks -> configurable fleet -> isolated grading -> curation\n-> genuine LoRA -> held-out gate -> local serving.\nA specialised student may beat an individual teacher on this narrow workload by learning verified complementary outputs.\nThis does not establish general superiority over frontier models.\nAn inconclusive gate means NO SHIP.",
        ),
        slide(
            "Task supply and source separation",
            "The authored catalogue has 96 Rust coding contracts:\n48 train, 16 development, 32 held out; 24 source groups.\nComplete sources are assigned before model collection.\nLineage, executable oracles and provenance accompany tasks.\nOne author may induce dependence between source groups.\nThe sealed set must never become a tuning dashboard.",
        ),
        slide(
            "Fleet and durable budget invariants",
            format!(
                "{fleet_summary}\nTransactional reservations precede every attempted call.\nFinished request keys are cached; unknown remote acceptance retains its reservation.\nSynthetic faults are labelled as fixtures.\nThree independent real provider recordings are still required.\nNo missing completion is fabricated."
            ),
        ),
        slide(
            "Sandbox and contamination evidence",
            format!(
                "Grader-gaming drill: {}\nEscape-attempt drill: {}\nPlanted held-out task: {}\nA complete assessment requires observed container enforcement and trusted syscall evidence.\nCandidate self-reports and score fixtures cannot complete these drills.",
                case_status(&drill, "grader_gaming"),
                case_status(&drill, "escape_attempts"),
                case_status(&drill, "planted_heldout_task")
            ),
        ),
        slide(
            "Actual Qwen base and gradient QA",
            format!(
                "Pinned real Qwen2.5-Coder-0.5B-Instruct on CPU.\nBase generated {} output tokens; complete={}; {} ms.\nFull-base synthetic tensor QA: {} input tokens.\nLoss {}; gradient norm {}.\nChanged adapter L2 {}; safetensors round trip={}.\nThis is technical QA, not provider-output training.",
                field(&base, "output_tokens"),
                field(&base, "complete"),
                field(&base, "elapsed_ms"),
                field(&gradient, "sequence_tokens"),
                field(&gradient, "loss"),
                field(&gradient, "gradient_norm"),
                field(&gradient, "changed_parameter_l2"),
                field(&gradient, "adapter_safetensors_roundtrip")
            ),
        ),
        slide(
            "Actual bounded local endpoint",
            format!(
                "Real model health includes checksum and inference checks.\nComplete generated output: {} tokens.\nOversized body HTTP {}; excessive tokens HTTP {}.\nConcurrent pressure statuses: {}.\nInitialise -> explicit evaluation route -> atomic rollback.\nThe evaluation alias serves the same base weights.\nNo trained candidate canary is claimed.",
                field(&serving["generation"], "output_tokens"),
                field(&serving, "oversized_body_status"),
                field(&serving, "excessive_output_status"),
                field(&serving, "pressure_status_counts")
            ),
        ),
        slide(
            "The gate: quality must be measured",
            format!(
                "Actual evaluation status: {}.\nShip requires both source-level interval lower bounds above the predeclared +2 percentage-point margin.\nAll task and decoding-seed outcomes remain in the denominator.\nEight held-out sources give weak statistical power.\nNegative/null score fixtures test the decision logic only.\nActual checkpoint controls remain a separate requirement.",
                eval.as_ref()
                    .map(|e| field(e, "decision"))
                    .unwrap_or_else(|| "no real paired evaluation report".into())
            ),
        ),
        slide(
            "Honest remaining evidence",
            "Rights/model/account clearance and spending authority.\nReal calls to three independent provider businesses.\nPassing isolated grades and a validated provider dataset.\nActual candidate, second-seed and negative/overfit adapters.\nFrozen held-out evaluation, measured costs and a real gate.\nFull provider-trained disconnected rehearsal.\nLocal security passes do not replace these missing artifacts.",
        ),
        slide(
            "Reproduce and review",
            "./run.sh demo --offline\nThe full command fails if required evidence is absent.\nInspect reports/drill.json and reports/eval.json.\nNative QA examples: model_smoke, cpu_gradient_smoke, serving_smoke.\nRead docs/training.md and docs/reproduction.md.\nThe repository includes source, checksums, tests and explicit limitations.\nNO SHIP keeps the incumbent.",
        ),
    ];
    let duration = 120u64;
    let seconds = duration / slides.len() as u64;
    let temp = tempfile::tempdir()?;
    let font = PathBuf::from("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    ensure!(
        font.is_file(),
        "DejaVu Sans font is required for caption rendering"
    );
    let mut filters = Vec::new();
    for (i, s) in slides.iter().enumerate() {
        let title = temp.path().join(format!("title-{i}.txt"));
        std::fs::write(&title, &s.title)?;
        let start = i as u64 * seconds;
        let end = start + seconds;
        filters.push(format!("drawtext=fontfile={}:textfile={}:fontcolor=0x67e8f9:fontsize=42:x=70:y=55:enable='gte(t,{start})*lt(t,{end})'",font.display(),title.display()));
        // Render one caption line per filter. Some packaged drawtext builds also
        // render a missing-glyph box for LF; separate lines avoid that artefact.
        ensure!(
            s.caption.lines().count() <= 11,
            "caption exceeds the visible slide area"
        );
        for (line_index, line) in s.caption.lines().enumerate() {
            let caption = temp.path().join(format!("caption-{i}-{line_index}.txt"));
            std::fs::write(&caption, line)?;
            filters.push(format!("drawtext=fontfile={}:textfile={}:fontcolor=white:fontsize=29:x=70:y={}:enable='gte(t,{start})*lt(t,{end})'",font.display(),caption.display(),155+line_index*45));
        }
    }
    let output = Path::new("reports/evidence-walkthrough.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("color=c=0x101827:s=1280x720:r=10:d={duration}"),
            "-vf",
            &filters.join(","),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "25",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
        ])
        .arg(output)
        .status()
        .context("ffmpeg media encoder unavailable")?;
    ensure!(
        status.success() && std::fs::metadata(output)?.len() > 0,
        "video encoding failed"
    );
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(output)
        .output()?;
    ensure!(probe.status.success(), "video verification failed");
    let measured = std::str::from_utf8(&probe.stdout)?.trim().parse::<f64>()?;
    ensure!(
        measured > 0. && measured <= 300.,
        "video duration exceeds the assessment limit"
    );
    let manifest=VideoManifest{evidence_kind:"captioned_walkthrough_of_actual_reports_with_explicit_gaps".into(),duration_seconds:duration,width:1280,height:720,input_sha256:hashes,video_sha256:lec_distil::model::sha256_file(output)?,slides,limitations:vec!["Captioned slides display actual report facts; this is not a claimed screen capture or narrated live demo.".into(),"Fixture QA, base inference and synthetic tensor updates are explicitly distinguished from actual provider-trained assessment evidence.".into(),"ffmpeg/ffprobe provide media encoding and verification; all evidence generation logic is Rust.".into()]};
    std::fs::write(
        "reports/evidence-walkthrough-manifest.json",
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!(
        "Created {} ({measured:.1} seconds, {} bytes)",
        output.display(),
        std::fs::metadata(output)?.len()
    );
    Ok(())
}
