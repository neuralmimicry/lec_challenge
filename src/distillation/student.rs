//! Shared optimiser surface for conventional LoRA and genuine BitNet QAT.
use super::DistillationWorkerSpec;
use crate::{
    bitnet::{self, BitNetStudent},
    model::{
        self, AdapterConfig, AdapterVariables, AssetManifest, ModelTrace, QwenConfig, QwenModel,
    },
};
use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DistillationStudent {
    #[default]
    ConventionalQwenLora,
    TernaryBitnet,
}

pub(super) enum Student {
    Qwen(QwenModel),
    BitNet(BitNetStudent),
}

impl Student {
    pub fn execution_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "engine":"native_rust_candle",
            "student_backend":match self {Self::Qwen(_)=>"conventional_qwen_lora",Self::BitNet(_)=>"bitnet_ternary_qat"},
            "actual_device":format!("{:?}",self.device().location()),
            "actual_dtype":"F32",
            "cuda_active":self.device().is_cuda(),
            "cpu_fallback":false,
            "ternary_quantisation_during_every_forward":matches!(self,Self::BitNet(_)),
            "available_cpu_threads":std::thread::available_parallelism().ok().map(usize::from),
            "candle_cpu_threads":candle_core::utils::get_num_threads(),
            "trainable_adapter_tensors":self.variables().len(),
        })
    }
    pub fn effective_trits(&self) -> Result<Option<crate::bitnet::EffectiveTritIdentity>> {
        match self {
            Self::Qwen(_) => Ok(None),
            Self::BitNet(x) => Ok(Some(x.effective_trit_identity()?)),
        }
    }
    pub fn device(&self) -> &Device {
        match self {
            Self::Qwen(x) => x.device(),
            Self::BitNet(x) => x.device(),
        }
    }
    pub fn variables(&self) -> &AdapterVariables {
        match self {
            Self::Qwen(x) => &x.adapter_vars,
            Self::BitNet(x) => &x.adapter_vars,
        }
    }
    pub fn layer_count(&self) -> usize {
        match self {
            Self::Qwen(x) => x.configuration().num_hidden_layers,
            Self::BitNet(x) => x.configuration().num_hidden_layers,
        }
    }
    pub fn clear_cache(&mut self) {
        if let Self::Qwen(x) = self {
            x.clear_cache();
        }
    }
    pub fn hidden(&mut self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Qwen(x) => x.hidden(input, 0, false),
            Self::BitNet(x) => x.hidden(input, 0, false),
        }
    }
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Qwen(x) => x.logits(hidden),
            Self::BitNet(x) => x.logits(hidden),
        }
    }
    pub fn forward_trace(&mut self, input: &Tensor, layer: usize) -> Result<ModelTrace> {
        match self {
            Self::Qwen(x) => x.forward_trace(input, layer),
            Self::BitNet(x) => x.forward_trace(input, layer),
        }
    }
}

pub(super) fn assets(spec: &DistillationWorkerSpec) -> Result<AssetManifest> {
    match spec.student_kind {
        DistillationStudent::ConventionalQwenLora => model::validate_assets(&spec.student_base),
        DistillationStudent::TernaryBitnet => {
            crate::hub::validate_bitnet_student_assets(&spec.student_base)
        }
    }
}

pub(super) fn prefix(spec: &DistillationWorkerSpec, prompt: &str) -> String {
    match spec.student_kind {
        DistillationStudent::ConventionalQwenLora => model::chat_prefix(prompt),
        DistillationStudent::TernaryBitnet => bitnet::formatted_prompt(prompt),
    }
}
pub(super) fn eos(spec: &DistillationWorkerSpec) -> u32 {
    match spec.student_kind {
        DistillationStudent::ConventionalQwenLora => 151645,
        DistillationStudent::TernaryBitnet => bitnet::END_OF_TURN,
    }
}

pub(super) fn load(spec: &DistillationWorkerSpec, adapter: Option<&Path>) -> Result<Student> {
    load_with_training(spec, adapter, true)
}

pub(super) fn load_for_inference(spec: &DistillationWorkerSpec, adapter: &Path) -> Result<Student> {
    load_with_training(spec, Some(adapter), false)
}

fn load_with_training(
    spec: &DistillationWorkerSpec,
    adapter: Option<&Path>,
    trainable: bool,
) -> Result<Student> {
    let cfg = AdapterConfig {
        rank: spec.rank,
        alpha: spec.alpha,
    };
    match spec.student_kind {
        DistillationStudent::ConventionalQwenLora => {
            let (device, dtype) = spec.student_runtime.resolve()?;
            let weights = candle_core::safetensors::load(
                spec.student_base.join("model.safetensors"),
                &device,
            )?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_dtype(dtype)?)))
            .collect::<Result<HashMap<_, _>>>()?;
            let config: QwenConfig =
                serde_json::from_slice(&fs::read(spec.student_base.join("config.json"))?)?;
            let loaded = adapter
                .map(|p| -> Result<HashMap<String, Tensor>> {
                    candle_core::safetensors::load(p, &device)?
                        .into_iter()
                        .map(|(name, t)| Ok((name, t.to_dtype(dtype)?)))
                        .collect()
                })
                .transpose()?;
            Ok(Student::Qwen(QwenModel::from_weights(
                config,
                weights,
                Some(cfg),
                spec.seed,
                loaded,
                trainable,
            )?))
        }
        DistillationStudent::TernaryBitnet => {
            ensure!(
                spec.student_runtime.device == model::ExecutionDevice::Cpu
                    && spec.student_runtime.dtype == model::InferenceDtype::F32,
                "native BitNet QAT supports explicit CPU F32 only; no device fallback"
            );
            let mut model = BitNetStudent::load_with_adapters(
                &spec.student_base,
                Some(cfg),
                spec.seed,
                adapter,
            )?;
            if !trainable {
                model.freeze_for_inference()?;
            }
            Ok(Student::BitNet(model))
        }
    }
}
