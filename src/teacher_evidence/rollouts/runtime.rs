//! Actual base-or-adapter runtimes; a self teacher can never silently use its base.
use super::*;
use crate::{bitnet::BitNetRuntime, distillation::DistillationStudent, model::ModelTrace};
use candle_core::{Device, Tensor};

mod context;
pub(crate) use context::{validate_context_observation, validate_teacher_request};

pub(crate) enum RecordedModel {
    Qwen(ModelRuntime),
    BitNet(BitNetRuntime),
}
impl RecordedModel {
    pub fn student_optional(
        base: &Path,
        checkpoint: Option<&Path>,
        kind: DistillationStudent,
        options: RuntimeOptions,
    ) -> Result<Self> {
        if let Some(path) = checkpoint {
            return Self::student(base, path, options);
        }
        match kind {
            DistillationStudent::ConventionalQwenLora => Ok(Self::Qwen(
                ModelRuntime::load_with_options(base, None, options)?,
            )),
            DistillationStudent::TernaryBitnet => {
                ensure!(
                    options.device == model::ExecutionDevice::Cpu
                        && options.dtype == model::InferenceDtype::F32,
                    "base BitNet requires explicit CPU/F32"
                );
                Ok(Self::BitNet(BitNetRuntime::load_base(base)?))
            }
        }
    }
    pub fn student(base: &Path, checkpoint: &Path, options: RuntimeOptions) -> Result<Self> {
        let m: crate::distillation::checkpoint::ExperimentalCheckpointManifest =
            serde_json::from_slice(&fs::read(checkpoint.join("manifest.json"))?)?;
        match m.spec.student_kind {
            DistillationStudent::ConventionalQwenLora => Ok(Self::Qwen(
                crate::distillation::load_runtime(base, checkpoint, options)?,
            )),
            DistillationStudent::TernaryBitnet => {
                ensure!(
                    options.device == model::ExecutionDevice::Cpu
                        && options.dtype == model::InferenceDtype::F32,
                    "adapted BitNet runtime requires explicit CPU/F32"
                );
                Ok(Self::BitNet(BitNetRuntime::load(base, checkpoint)?))
            }
        }
    }
    pub fn teacher(teacher: &TeacherEvidence, options: RuntimeOptions) -> Result<Self> {
        if let Some(checkpoint) = &teacher.adapter_checkpoint {
            checkpoint.verify(&std::env::current_dir()?)?;
            return Self::student(
                &teacher.model.directory,
                checkpoint
                    .path
                    .parent()
                    .context("self-teacher checkpoint parent absent")?,
                options,
            );
        }
        if teacher.model.architecture == "bitnet" {
            return Self::student_optional(
                &teacher.model.directory,
                None,
                DistillationStudent::TernaryBitnet,
                options,
            );
        }
        if teacher.model.directory == Path::new("offline_bundle/base") {
            return Ok(Self::Qwen(ModelRuntime::load_with_options(
                &teacher.model.directory,
                None,
                options,
            )?));
        }
        Ok(Self::Qwen(ModelRuntime::load_comparative_with_options(
            &teacher.model.directory,
            options,
        )?))
    }
    /// Explicit context is a new protocol for an original GGUF teacher only.
    /// Omission retains the exact historical loader, including its 1,024 cap.
    pub fn teacher_with_context(
        teacher: &TeacherEvidence,
        options: RuntimeOptions,
        context_tokens: Option<usize>,
    ) -> Result<Self> {
        let Some(context_tokens) = context_tokens else {
            return Self::teacher(teacher, options);
        };
        context::preflight(teacher, &options, context_tokens)?;
        Ok(Self::Qwen(ModelRuntime::load_quantised_with_context(
            &teacher.model.directory,
            options,
            context_tokens,
        )?))
    }
    pub fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        match self {
            Self::Qwen(m) => m.generate(request),
            Self::BitNet(m) => m.generate(request),
        }
    }
    pub fn execution_metadata(&self) -> serde_json::Value {
        match self {
            Self::Qwen(m) => m.execution_metadata(),
            Self::BitNet(m) => m.execution_metadata(),
        }
    }
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        match self {
            Self::Qwen(m) => &m.tokenizer,
            Self::BitNet(m) => &m.tokenizer,
        }
    }
    pub fn formatted_prompt(&self, prompt: &str) -> Result<String> {
        match self {
            Self::Qwen(m) => m.formatted_prompt(prompt),
            Self::BitNet(_) => Ok(crate::bitnet::formatted_prompt(prompt)),
        }
    }
    pub fn primary_eos_token_id(&self) -> u32 {
        match self {
            Self::Qwen(m) => m.primary_eos_token_id(),
            Self::BitNet(_) => crate::bitnet::END_OF_TURN,
        }
    }
    pub fn device(&self) -> &Device {
        match self {
            Self::Qwen(m) => m.model.device(),
            Self::BitNet(m) => m.model.device(),
        }
    }
    pub fn dimensions(&self) -> (usize, usize, usize) {
        match self {
            Self::Qwen(m) => {
                let c = m.model.configuration();
                (c.vocab_size, c.hidden_size, c.num_hidden_layers)
            }
            Self::BitNet(m) => {
                let c = m.model.configuration();
                (c.vocab_size, c.hidden_size, c.num_hidden_layers)
            }
        }
    }
    pub fn clear_cache(&mut self) {
        if let Self::Qwen(m) = self {
            m.model.clear_cache()
        }
    }
    pub fn forward_trace(&mut self, input: &Tensor, layer: usize) -> Result<ModelTrace> {
        match self {
            Self::Qwen(m) => m.model.forward_trace(input, layer),
            Self::BitNet(m) => m.model.forward_trace(input, layer),
        }
    }
    pub fn hidden(&mut self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Qwen(m) => m.model.hidden(input, 0, false),
            Self::BitNet(m) => m.model.hidden(input, 0, false),
        }
    }
    pub fn logits(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Qwen(m) => m.model.logits(input),
            Self::BitNet(m) => m.model.logits(input),
        }
    }
}
