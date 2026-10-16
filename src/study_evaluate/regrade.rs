//! Derived grading of immutable train/development generations, without inference.
//! Original provenance and every denominator entry remain independently bound.
use super::*;

mod import;
mod run;
#[cfg(test)]
mod tests;
mod verify;
pub use run::run_batch;
pub use verify::verify_batch;
pub(super) use verify::{verified_batch, verify_view};

pub const PLAN_KIND: &str = "frozen_local_study_generation_regrade";
const BATCH_KIND: &str = "frozen_local_study_regrade_batch";

pub struct RegradeBatchOptions {
    pub originals: Vec<PathBuf>,
    pub output: PathBuf,
    /// Explicit denominator declared before importing or grading any task.
    pub expected_generations: usize,
    pub maximum_concurrent_graders: usize,
    pub sandbox: SandboxConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OriginalEvaluation {
    directory: PathBuf,
    plan: Artefact,
    generation_plan: Plan,
    execution_receipts: Vec<Artefact>,
    rows: Vec<ImportedGeneration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ImportedGeneration {
    task_id: String,
    source_id: String,
    original_raw: Artefact,
    /// The raw record keeps this original pointer, never the regrade plan SHA.
    original_plan_sha256: String,
    request_sha256: String,
    extracted_code_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GradingProtocol {
    engine: Artefact,
    sandbox: SandboxConfig,
    policy_sha256: String,
    image_id: String,
    preflight: Artefact,
    extractor_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BatchPlan {
    schema_version: u32,
    evidence_kind: String,
    originals: Vec<OriginalEvaluation>,
    grading: GradingProtocol,
    expected_generations: usize,
    maximum_concurrent_graders: usize,
    production_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RegradePlan {
    schema_version: u32,
    evidence_kind: String,
    batch_plan: Artefact,
    original: OriginalEvaluation,
    grading: GradingProtocol,
    maximum_concurrent_graders: usize,
    production_eligible: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Outcome {
    OriginalGenerationFailure,
    Graded {
        prepared: Artefact,
        report: Artefact,
        audit: Artefact,
    },
    InfrastructureFailure {
        prepared: Option<Artefact>,
        error: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RegradedObservation {
    schema_version: u32,
    regrade_plan_sha256: String,
    task_id: String,
    source_id: String,
    original_plan: Artefact,
    original_raw: Artefact,
    outcome: Outcome,
    success: bool,
}

fn root_relative(root: &Path, path: &Path) -> Result<PathBuf> {
    let canonical = path.canonicalize()?;
    Ok(canonical.strip_prefix(root)?.to_owned())
}

fn capture(root: &Path, path: &Path) -> Result<Artefact> {
    Artefact::capture(
        root,
        if path.is_absolute() {
            path.strip_prefix(root)?.to_owned()
        } else {
            path.to_owned()
        },
    )
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    ensure!(
        fs::metadata(path)?.len() <= 32 << 20,
        "regrade JSON exceeds 32 MiB bound"
    );
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn extractor_hash() -> String {
    // Freeze the actual implementation as well as the grading executable.
    artefact::hash(include_bytes!("../evaluate.rs"))
}

fn usable(raw: &RawGeneration) -> bool {
    raw.generation
        .as_ref()
        .is_some_and(|g| g.complete && !g.completion.trim().is_empty())
}

fn original_raw(
    root: &Path,
    original: &OriginalEvaluation,
    imported: &ImportedGeneration,
    task: &Task,
) -> Result<RawGeneration> {
    ensure!(
        imported.task_id == task.id
            && imported.source_id == task.source_id
            && imported.original_plan_sha256 == original.plan.sha256
            && imported.original_raw.path
                == original
                    .directory
                    .join("generations")
                    .join(format!("{}.json", task.id)),
        "imported generation task/plan/path differs"
    );
    let raw: RawGeneration = read(&imported.original_raw.verify(root)?)?;
    evidence::raw(&raw, task, &original.generation_plan, &original.plan.sha256)?;
    ensure!(
        artefact::hash(&serde_json::to_vec(&raw.request)?) == imported.request_sha256,
        "imported generation request changed"
    );
    let code_hash = raw
        .generation
        .as_ref()
        .filter(|_| usable(&raw))
        .map(|g| tasks::hash(crate::evaluate::extract_code(&g.completion)));
    ensure!(
        code_hash == imported.extracted_code_sha256,
        "imported extracted code changed"
    );
    Ok(raw)
}

fn validate_cohort(originals: &[OriginalEvaluation], expected: usize) -> Result<()> {
    ensure!(
        !originals.is_empty() && expected > 0,
        "empty original regrade cohort"
    );
    let mut directories = BTreeSet::new();
    let mut plans = BTreeSet::new();
    let mut identities = BTreeSet::new();
    let mut total = 0usize;
    let reference = &originals[0].generation_plan;
    for original in originals {
        let plan = &original.generation_plan;
        ensure!(
            plan.checkpoint.checkpoint_id.len() == 64
                && plan
                    .checkpoint
                    .checkpoint_id
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "invalid original checkpoint identity"
        );
        ensure!(
            directories.insert(&original.directory)
                && plans.insert(&original.plan.sha256)
                && identities.insert((
                    plan.checkpoint.checkpoint_id.clone(),
                    serde_json::to_string(&plan.split)?
                )),
            "duplicate original directory, plan or checkpoint/split identity"
        );
        ensure!(
            original.plan.path == original.directory.join("plan.json")
                && original.rows.len() == plan.tasks.len(),
            "original plan path or denominator differs"
        );
        let mut tasks = BTreeSet::new();
        for (task, row) in plan.tasks.iter().zip(&original.rows) {
            ensure!(
                tasks.insert(&task.id) && row.task_id == task.id && row.source_id == task.source_id,
                "original cohort task order/uniqueness differs"
            );
        }
        total = total
            .checked_add(original.rows.len())
            .context("cohort denominator overflow")?;
        ensure!(
            same(&plan.engine, &reference.engine)?
                && plan.runtime == reference.runtime
                && plan.seed == reference.seed
                && plan.max_new_tokens == reference.max_new_tokens
                && plan.temperature == reference.temperature
                && plan.top_p == reference.top_p
                && plan.timeout_seconds == reference.timeout_seconds
                && plan.checkpoint.architecture == reference.checkpoint.architecture
                && same(
                    &plan.checkpoint.base_assets,
                    &reference.checkpoint.base_assets
                )?
                && same(&plan.checkpoint.tokenizer, &reference.checkpoint.tokenizer)?,
            "original cohort changes generation engine/runtime/decoding/base/tokeniser"
        );
    }
    ensure!(
        total == expected,
        "original cohort contains {total} generations; expected {expected}"
    );
    Ok(())
}

fn child_name(index: usize, original: &OriginalEvaluation) -> String {
    let split = match original.generation_plan.split {
        StudySplit::Train => "train",
        StudySplit::Development => "development",
    };
    format!(
        "evaluation-{index:03}-{split}-{}",
        &original.generation_plan.checkpoint.checkpoint_id[..12]
    )
}
