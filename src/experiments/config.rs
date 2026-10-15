//! Validate the finite operator grid and measured resource envelope before work.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FidelityConfig {
    pub steps: usize,
    pub max_sequence_tokens: usize,
    pub sequence_policy: SequencePolicy,
    pub rank: usize,
    pub alpha: f64,
    pub learning_rate: f64,
    pub temperature: f64,
    pub dpo_beta: f64,
    pub maximum_trials: usize,
    pub maximum_total_steps: u64,
    pub timeout_seconds: u64,
}
impl FidelityConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=100_000).contains(&self.steps)
                && (16..=4096).contains(&self.max_sequence_tokens)
                && (1..=128).contains(&self.rank)
                && self.alpha.is_finite()
                && self.alpha > 0.
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.
                && self.learning_rate <= 0.1
                && self.temperature.is_finite()
                && (0.1..=10.).contains(&self.temperature)
                && self.dpo_beta.is_finite()
                && (0.001..=10.).contains(&self.dpo_beta)
                && (1..=4096).contains(&self.maximum_trials)
                && self.maximum_total_steps >= self.steps as u64
                && self.maximum_total_steps <= 10_000_000
                && (1..=86_400).contains(&self.timeout_seconds),
            "invalid fidelity/training budget"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudyConfig {
    pub schema_version: u32,
    pub student_base_model: String,
    pub signals: Vec<SignalMethod>,
    pub contents: Vec<ContentMethod>,
    pub policies: Vec<DataPolicy>,
    pub students: Vec<StudentArchitecture>,
    pub progressive: Vec<bool>,
    pub self_distillation: Vec<bool>,
    pub seeds: Vec<u64>,
    pub minimum_teachers: usize,
    pub maximum_teachers: usize,
    pub maximum_catalogue_arms: usize,
    #[serde(default)]
    pub student_training: Vec<StudentTraining>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitnet_backend: Option<crate::bitnet::BitNetBackendOptions>,
    pub quick: FidelityConfig,
    pub full: FidelityConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StudentTraining {
    pub student: StudentArchitecture,
    pub quick_learning_rate: f64,
    pub full_learning_rate: f64,
}
fn unique<T: Ord>(items: &[T]) -> bool {
    items.iter().collect::<BTreeSet<_>>().len() == items.len()
}
impl StudyConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1
                && self.student_base_model == crate::model::MODEL_ID
                && !self.signals.is_empty()
                && self.signals.len() <= 4
                && unique(&self.signals)
                && self.contents.len() <= 4
                && unique(&self.contents)
                && !self.policies.is_empty()
                && self.policies.len() <= 2
                && unique(&self.policies)
                && !self.students.is_empty()
                && self.students.len() <= 2
                && unique(&self.students)
                && !self.progressive.is_empty()
                && self.progressive.len() <= 2
                && unique(&self.progressive)
                && !self.self_distillation.is_empty()
                && self.self_distillation.len() <= 2
                && unique(&self.self_distillation)
                && !self.seeds.is_empty()
                && self.seeds.len() <= 8
                && unique(&self.seeds)
                && self.minimum_teachers >= 1
                && self.maximum_teachers >= self.minimum_teachers
                && self.maximum_teachers <= 6
                && (1..=262_144).contains(&self.maximum_catalogue_arms),
            "invalid or unbounded study operator grid"
        );
        self.quick.validate()?;
        self.full.validate()?;
        let mut students = BTreeSet::new();
        for override_ in &self.student_training {
            ensure!(
                students.insert(override_.student)
                    && self.students.contains(&override_.student)
                    && [override_.quick_learning_rate, override_.full_learning_rate]
                        .iter()
                        .all(|rate| rate.is_finite() && *rate > 0. && *rate <= 0.1),
                "invalid or duplicate preregistered student training override"
            );
        }
        ensure!(
            self.quick.steps <= self.full.steps
                && self.quick.max_sequence_tokens <= self.full.max_sequence_tokens,
            "quick fidelity must not exceed full fidelity"
        );
        Ok(())
    }
}

pub(super) fn validate_inputs(inputs: &StudyInputs) -> Result<()> {
    ensure!(
        inputs.schema_version == 1
            && !inputs.teachers.is_empty()
            && inputs.teachers.len() <= 12
            && inputs.student_tokenizer_sha256.len() == 64
            && inputs.resources.cpu_threads > 0
            && inputs.resources.ram_bytes > 0
            && inputs.resources.maximum_parallel_jobs > 0
            && inputs.resources.maximum_parallel_jobs <= inputs.resources.cpu_threads as usize
            && (inputs.resources.gpu_ordinal.is_some() || inputs.resources.vram_bytes == 0),
        "invalid study inputs/resource capacity"
    );
    let request = &inputs.worker_resources;
    ensure!(
        request.cpu_threads > 0
            && request.cpu_threads <= inputs.resources.cpu_threads
            && request.ram_bytes > 0
            && request.ram_bytes <= inputs.resources.ram_bytes
            && request.vram_bytes <= inputs.resources.vram_bytes,
        "one worker cannot fit the measured aggregate resource envelope"
    );
    let mut resource_students = BTreeSet::new();
    for profile in &inputs.student_resources {
        ensure!(
            resource_students.insert(profile.student),
            "duplicate student resource profile"
        );
        for request in [&profile.quick, &profile.full] {
            ensure!(
                request.cpu_threads > 0
                    && request.cpu_threads <= inputs.resources.cpu_threads
                    && request.ram_bytes > 0
                    && request.ram_bytes <= inputs.resources.ram_bytes
                    && request.vram_bytes <= inputs.resources.vram_bytes,
                "student/fidelity resource request cannot fit the declared capacity"
            );
        }
        ensure!(
            profile.tensor_capture_vram_bytes <= inputs.resources.vram_bytes,
            "teacher tensor capture cannot fit the declared GPU capacity"
        );
    }
    let mut ids = BTreeSet::new();
    for teacher in &inputs.teachers {
        ensure!(
            !teacher.id.is_empty()
                && teacher
                    .id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
                && ids.insert(&teacher.id)
                && teacher.revision.len() == 40
                && !teacher.source_model_id.is_empty(),
            "unsafe or duplicate teacher identity"
        );
        for cap in teacher
            .signals
            .values()
            .chain(teacher.contents.values())
            .chain([&teacher.on_policy, &teacher.self_distillation])
        {
            ensure!(
                !cap.reason.trim().is_empty()
                    && (!(cap.available || cap.acquirable_offline) || cap.supported),
                "capability claims need an explicit reason and cannot be available while unsupported"
            );
            ensure!(
                !cap.usable() || !cap.artefacts.is_empty(),
                "usable teacher capability lacks linked evidence"
            );
        }
        for cap in teacher
            .contents
            .values()
            .chain([&teacher.on_policy, &teacher.self_distillation])
        {
            ensure!(
                !cap.acquirable_offline,
                "content/rollouts cannot be invented through the native tensor-acquisition exception"
            );
        }
        if let Some(cap) = teacher.signals.get(&SignalMethod::SequenceSft) {
            ensure!(
                !cap.acquirable_offline,
                "verified sequence winners must exist before planning"
            );
        }
    }
    let mut architectures = BTreeSet::new();
    for student in &inputs.students {
        ensure!(
            architectures.insert(student.architecture),
            "duplicate student capability identity"
        );
        ensure!(
            !student.training.usable()
                || (student.base_assets.is_some()
                    && student
                        .tokenizer_sha256
                        .as_ref()
                        .is_some_and(|h| h.len() == 64)),
            "trainable student requires its independently frozen base and tokeniser identity"
        );
        for cap in [&student.training, &student.exported_checkpoint_reload] {
            ensure!(
                !cap.reason.is_empty()
                    && !cap.acquirable_offline
                    && (!cap.available || cap.supported)
                    && (!cap.usable() || !cap.artefacts.is_empty()),
                "unproven student training/export capability"
            );
        }
    }
    Ok(())
}
