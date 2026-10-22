//! Native Qwen2/Qwen3 and Llama inference with differentiable adaptation.
//!
//! The architecture follows Qwen2 (Apache-2.0). Unlike the optimised inference
//! kernels, the tensor RoPE, RMS normalisation and softmax paths retain gradients.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, IndexOp, Module, Tensor, Var};
use candle_nn::{Embedding, LayerNorm, Linear};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

mod asset_lease;
mod quantised;
#[cfg(test)]
mod qwen3_tests;
use asset_lease::AssetLease;
pub use quantised::{QuantisedMemoryEstimate, estimate_quantised_memory};

pub const MODEL_ID: &str = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
pub const MODEL_REVISION: &str = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25";
pub const MODEL_SHA256: &str = "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44";
pub const TOKENIZER_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
pub const SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
pub(crate) const PINNED_FILES: &[(&str, &str)] = &[
    (
        "config.json",
        "b1e58593cd31852f7da5c2fc31ddf6135b9c066c0fd9177a4bbe95717083adff",
    ),
    ("tokenizer.json", TOKENIZER_SHA256),
    (
        "tokenizer_config.json",
        "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    ),
    (
        "generation_config.json",
        "fdaccbcb02f3e1e7914ccb0f69ebe899071ffd27cf825166d823163e156870f2",
    ),
    (
        "LICENSE",
        "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    ),
    (
        "README.md",
        "50cd33dcab1f7b9d206d4691df527a419cfddadd0c4648b8d9bffd5a1b7a032d",
    ),
    ("model.safetensors", MODEL_SHA256),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetManifest {
    pub model_id: String,
    pub revision: String,
    pub licence: String,
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDevice {
    #[default]
    Cpu,
    Cuda {
        ordinal: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InferenceDtype {
    #[default]
    F32,
    F16,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeOptions {
    pub device: ExecutionDevice,
    pub dtype: InferenceDtype,
}

impl RuntimeOptions {
    /// Check executable support without loading weights or creating a GPU context.
    pub fn validate_build_support(&self) -> Result<()> {
        match self.device {
            ExecutionDevice::Cpu => ensure!(
                self.dtype == InferenceDtype::F32,
                "CPU inference requires explicit float32 precision"
            ),
            ExecutionDevice::Cuda { .. } => ensure!(
                cfg!(feature = "cuda"),
                "CUDA was requested but this executable has no cuda feature; no CPU fallback is allowed"
            ),
        }
        Ok(())
    }
    pub fn cuda(ordinal: usize) -> Self {
        Self {
            device: ExecutionDevice::Cuda { ordinal },
            dtype: InferenceDtype::F16,
        }
    }

    pub(crate) fn resolve(&self) -> Result<(Device, DType)> {
        self.validate_build_support()?;
        let dtype = match self.dtype {
            InferenceDtype::F32 => DType::F32,
            InferenceDtype::F16 => DType::F16,
        };
        let device = match self.device {
            ExecutionDevice::Cpu => {
                ensure!(
                    dtype == DType::F32,
                    "CPU inference requires explicit float32 precision"
                );
                Device::Cpu
            }
            ExecutionDevice::Cuda { ordinal } => {
                ensure!(
                    cfg!(feature = "cuda"),
                    "CUDA was requested but this executable has no cuda feature; no CPU fallback is allowed"
                );
                let device = Device::new_cuda(ordinal)
                    .context("create explicitly requested CUDA device; no CPU fallback")?;
                ensure!(
                    device.is_cuda(),
                    "CUDA request did not create a CUDA device"
                );
                device
            }
        };
        Ok((device, dtype))
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Publish only complete files; fsync both file and directory before returning.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact has no parent")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .context("artifact filename")?
            .to_string_lossy(),
        std::process::id()
    ));
    {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

/// Explicit online bootstrap. Normal loading never consults the Hugging Face Hub.
pub async fn bootstrap(base_dir: &Path) -> Result<AssetManifest> {
    fs::create_dir_all(base_dir)?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(1800))
        .build()?;
    let mut files = BTreeMap::new();
    for &(name, expected) in PINNED_FILES {
        let path = base_dir.join(name);
        let checked = path.clone();
        let state =
            tokio::task::spawn_blocking(move || crate::assets::inspect_file(&checked, expected))
                .await??;
        if state["state"] != "missing" {
            ensure!(
                state["state"] == "valid_regular_file",
                "pinned asset conflict at {}: {}; inspect with assets-doctor and explicitly quarantine before restoring; no existing payload was removed",
                path.display(),
                state["state"]
            );
        }
        if state["state"] == "missing" {
            eprintln!("bootstrap: downloading pinned {name}");
            let url = format!("https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}");
            let mut response = client.get(url).send().await?.error_for_status()?;
            let partial = base_dir.join(format!(".{name}.download"));
            let mut file = tokio::fs::File::create(&partial).await?;
            use tokio::io::AsyncWriteExt;
            while let Some(chunk) = response.chunk().await? {
                file.write_all(&chunk).await?;
            }
            file.sync_all().await?;
            drop(file);
            let hash_path = partial.clone();
            let downloaded = tokio::task::spawn_blocking(move || sha256_file(&hash_path)).await??;
            ensure!(
                downloaded == expected,
                "downloaded asset checksum mismatch before publication: {name}; partial payload retained"
            );
            tokio::fs::rename(&partial, &path).await?;
        }
        let hash_path = path.clone();
        let hash = tokio::task::spawn_blocking(move || sha256_file(&hash_path)).await??;
        ensure!(
            hash == expected,
            "pinned model asset SHA-256 mismatch: {name}"
        );
        files.insert(name.to_owned(), hash);
    }
    let manifest = AssetManifest {
        model_id: MODEL_ID.into(),
        revision: MODEL_REVISION.into(),
        licence: "Apache-2.0".into(),
        files,
    };
    let manifest_path = base_dir.join("assets.json");
    if manifest_path.exists() {
        let saved: AssetManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        ensure!(
            serde_json::to_value(saved)? == serde_json::to_value(&manifest)?,
            "existing asset provenance differs; preserve and explicitly resolve the manifest conflict"
        );
    } else {
        atomic_write(&manifest_path, &serde_json::to_vec_pretty(&manifest)?)?;
    }
    let validate_dir = base_dir.to_owned();
    tokio::task::spawn_blocking(move || validate_assets(&validate_dir)).await??;
    Ok(manifest)
}

pub fn validate_assets(base_dir: &Path) -> Result<AssetManifest> {
    let manifest: AssetManifest = serde_json::from_slice(
        &fs::read(base_dir.join("assets.json"))
            .context("model assets missing; run bootstrap-online explicitly")?,
    )?;
    ensure!(
        manifest.model_id == MODEL_ID
            && manifest.revision == MODEL_REVISION
            && manifest.licence == "Apache-2.0",
        "unrecognised model asset provenance"
    );
    ensure!(
        manifest.files.len() == PINNED_FILES.len(),
        "unexpected model assets in manifest"
    );
    for &(required, hash) in PINNED_FILES {
        ensure!(
            manifest.files.get(required).map(String::as_str) == Some(hash),
            "asset manifest missing or changed: {required}"
        );
    }
    ensure!(
        manifest.files["model.safetensors"] == MODEL_SHA256,
        "unpinned base tensor hash"
    );
    for (name, expected) in &manifest.files {
        ensure!(
            !name.contains('/') && !name.contains('\\') && name != "..",
            "unsafe asset filename"
        );
        ensure!(
            &sha256_file(&base_dir.join(name))? == expected,
            "asset checksum mismatch: {name}"
        );
    }
    Ok(manifest)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenConfig {
    #[serde(default = "qwen_model_type")]
    pub model_type: String,
    #[serde(default)]
    pub attention_bias: Option<bool>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default = "silu_activation")]
    pub hidden_act: String,
    #[serde(default)]
    pub mlp_bias: bool,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Qwen3 may project to more attention channels than its residual width.
    /// Omission preserves historical Qwen2/Llama configuration fingerprints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
}

fn qwen_model_type() -> String {
    "qwen2".into()
}

fn silu_activation() -> String {
    "silu".into()
}

impl QwenConfig {
    pub fn head_dimension(&self) -> usize {
        self.head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads.max(1))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.model_type.as_str(), "qwen2" | "qwen3" | "llama"),
            "unsupported model architecture"
        );
        ensure!(
            self.hidden_act == "silu" && !self.mlp_bias && !self.use_sliding_window,
            "unsupported activation, feed-forward bias or sliding-window semantics"
        );
        ensure!(
            (1..=1_000_000).contains(&self.vocab_size)
                && (2..=16384).contains(&self.hidden_size)
                && (1..=65536).contains(&self.intermediate_size)
                && (1..=128).contains(&self.num_hidden_layers)
                && (1..=1_000_000).contains(&self.max_position_embeddings),
            "model dimensions are empty or outside native runtime bounds"
        );
        ensure!(
            (1..=256).contains(&self.num_attention_heads)
                && self.num_key_value_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads)
                && (2..=512).contains(&self.head_dimension())
                && self.head_dimension().is_multiple_of(2),
            "invalid attention or rotary head dimensions"
        );
        if self.model_type == "qwen3" {
            ensure!(
                self.head_dim.is_some()
                    && self.attention_bias == Some(false)
                    && self.num_attention_heads * self.head_dimension() <= 16384,
                "Qwen3 requires explicit bounded head_dim and bias-free projections"
            );
        } else {
            ensure!(
                self.hidden_size.is_multiple_of(self.num_attention_heads)
                    && self.head_dimension() == self.hidden_size / self.num_attention_heads,
                "legacy attention head width differs from residual dimensions"
            );
        }
        ensure!(
            self.rope_theta.is_finite()
                && self.rope_theta > 0.
                && self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.
                && self.rms_norm_eps <= 1.,
            "invalid rotary base or RMS normalisation epsilon"
        );
        if let Some(scaling) = &self.rope_scaling {
            ensure!(
                scaling.scaling_type == "linear"
                    && scaling.factor.is_finite()
                    && (1. ..=64.).contains(&scaling.factor),
                "unsupported or invalid rotary scaling"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(rename = "type", alias = "rope_type")]
    pub scaling_type: String,
    pub factor: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterConfig {
    pub rank: usize,
    pub alpha: f64,
}
impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.,
        }
    }
}

/// A small reproducible generator avoids dependence on platform RNG behaviour.
#[derive(Clone)]
pub(crate) struct StableRng(u64);
impl StableRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(crate) fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)
    }
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

pub type AdapterVariables = BTreeMap<String, Var>;

/// Candle's generic matmul backward also allocates gradients for constant weights.
/// This frozen projection retains the exact input gradient and intentionally omits
/// a base-weight gradient. The CPU GEMM is Candle's existing parallel backend.
#[derive(Clone, Debug)]
struct FrozenMatmul;
impl candle_core::CustomOp2 for FrozenMatmul {
    fn name(&self) -> &'static str {
        "frozen-base-matmul"
    }
    fn cpu_fwd(
        &self,
        lhs: &candle_core::CpuStorage,
        ll: &candle_core::Layout,
        rhs: &candle_core::CpuStorage,
        rl: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        let (m, k) = ll.shape().dims2()?;
        let (rk, n) = rl.shape().dims2()?;
        if k != rk {
            candle_core::bail!("frozen matmul dimensions disagree");
        }
        Ok((lhs.matmul(rhs, (1, m, n, k), ll, rl)?, (m, n).into()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        lhs: &candle_core::CudaStorage,
        ll: &candle_core::Layout,
        rhs: &candle_core::CudaStorage,
        rl: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        let (m, k) = ll.shape().dims2()?;
        let (rk, n) = rl.shape().dims2()?;
        if k != rk {
            candle_core::bail!("frozen CUDA matmul dimensions disagree");
        }
        Ok((lhs.matmul(rhs, (1, m, n, k), ll, rl)?, (m, n).into()))
    }
    fn bwd(
        &self,
        _lhs: &Tensor,
        rhs: &Tensor,
        _result: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some(grad.matmul(&rhs.t()?)?), None))
    }
}

fn frozen_linear(base: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    let mut output_shape = x.dims().to_vec();
    let input = *output_shape
        .last()
        .ok_or_else(|| candle_core::Error::Msg("linear input must have dimensions".into()))?;
    let rows = x.elem_count() / input;
    let out = base.weight().dim(0)?;
    *output_shape.last_mut().expect("checked shape") = out;
    let y = x
        .contiguous()?
        .reshape((rows, input))?
        .apply_op2(&base.weight().t()?, FrozenMatmul)?
        .reshape(output_shape)?;
    match base.bias() {
        None => Ok(y),
        Some(bias) => y.broadcast_add(bias),
    }
}

/// Float16 Qwen queries can have a finite dot product above 65,504. Casting
/// an already-overflowed GEMM result cannot recover it. Promote activations
/// before the score GEMM, while retaining the requested weight/KV precision.
fn stable_attention_scores(
    q: &Tensor,
    k: &Tensor,
    head_dim: usize,
    mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let k = k.to_dtype(DType::F32)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
    match mask {
        Some(mask) => scores.broadcast_add(&mask.to_dtype(DType::F32)?),
        None => Ok(scores),
    }
}

#[derive(Clone)]
enum LinearBackend {
    Dense(Linear),
    Quantised {
        weight: candle_core::quantized::QMatMul,
        bias: Option<Tensor>,
    },
}

impl LinearBackend {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(base) => frozen_linear(base, input),
            Self::Quantised { weight, bias } => {
                let output = weight.forward(&input.contiguous()?)?;
                match bias {
                    None => Ok(output),
                    Some(bias) => output.broadcast_add(bias),
                }
            }
        }
    }
}

impl From<Linear> for LinearBackend {
    fn from(linear: Linear) -> Self {
        Self::Dense(linear)
    }
}

#[derive(Clone)]
struct Projection {
    base: LinearBackend,
    adapter: Option<(Tensor, Tensor, f64)>,
}
impl Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let base = self.base.forward(x)?;
        match &self.adapter {
            None => Ok(base),
            Some((a, b, scale)) => {
                let (batch, seq, hidden) = x.dims3()?;
                let delta = x
                    .reshape((batch * seq, hidden))?
                    .matmul(&a.t()?)?
                    .matmul(&b.t()?)?
                    .reshape(base.shape())?;
                base + (delta * *scale)?
            }
        }
    }
}

fn apply_head_norm(input: &Tensor, norm: &Option<LayerNorm>) -> candle_core::Result<Tensor> {
    match norm {
        Some(norm) => norm.forward(&input.contiguous()?),
        None => Ok(input.clone()),
    }
}

struct Layer {
    q: Projection,
    k: LinearBackend,
    v: Projection,
    o: LinearBackend,
    gate: LinearBackend,
    up: LinearBackend,
    down: LinearBackend,
    norm1: LayerNorm,
    norm2: LayerNorm,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    kv: Option<(Tensor, Tensor)>,
}

/// Base tensors are constants. Only q/v adapter tensors can enter this variable map.
pub struct QwenModel {
    cfg: QwenConfig,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: LayerNorm,
    lm_head: LinearBackend,
    cos: Tensor,
    sin: Tensor,
    device: Device,
    dtype: DType,
    quantisation: Option<serde_json::Value>,
    interleaved_rotary: bool,
    pub adapter_vars: AdapterVariables,
    // Fields drop in declaration order. Keep the shared asset descriptor until
    // all model tensors have dropped, including when moved out of a runtime.
    _asset_lease: Option<AssetLease>,
}

/// Selected-layer traces retain the student's backward graph. Teacher callers
/// detach and transfer only their selected rows before releasing a model.
pub struct ModelTrace {
    pub final_hidden: Tensor,
    pub layer_hidden: Tensor,
    pub mean_attention: Tensor,
}

fn constant(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    Ok(weights
        .get(name)
        .with_context(|| format!("missing base tensor {name}"))?
        .clone())
}
fn base_linear(weights: &HashMap<String, Tensor>, name: &str, bias: bool) -> Result<Linear> {
    Ok(Linear::new(
        constant(weights, &format!("{name}.weight"))?,
        if bias {
            Some(constant(weights, &format!("{name}.bias"))?)
        } else {
            None
        },
    ))
}

impl QwenModel {
    pub fn from_weights(
        cfg: QwenConfig,
        weights: HashMap<String, Tensor>,
        adapter: Option<AdapterConfig>,
        seed: u64,
        loaded: Option<HashMap<String, Tensor>>,
        trainable: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        ensure!(
            !cfg.use_sliding_window,
            "sliding-window attention is not supported by this pinned architecture"
        );
        let bias = cfg.attention_bias.unwrap_or(cfg.model_type == "qwen2");
        let embed = constant(&weights, "model.embed_tokens.weight")?;
        let dev = embed.device().clone();
        let dtype = embed.dtype();
        ensure!(
            matches!(dtype, DType::F32 | DType::F16),
            "model precision must be float32 or float16"
        );
        ensure!(
            !trainable || dtype == DType::F32,
            "trainable adapters require float32; mixed-precision optimisation is not implemented"
        );
        ensure!(
            weights
                .values()
                .all(|t| t.device().same_device(&dev) && t.dtype() == dtype),
            "base weights have inconsistent devices or precision"
        );
        let expect = |name: &str, dimensions: &[usize]| -> Result<()> {
            ensure!(
                constant(&weights, name)?.dims() == dimensions,
                "base tensor has incorrect architecture dimensions: {name}"
            );
            Ok(())
        };
        expect(
            "model.embed_tokens.weight",
            &[cfg.vocab_size, cfg.hidden_size],
        )?;
        expect("model.norm.weight", &[cfg.hidden_size])?;
        if !cfg.tie_word_embeddings {
            expect("lm_head.weight", &[cfg.vocab_size, cfg.hidden_size])?;
        }
        let head_dim = cfg.head_dimension();
        let q_dimension = head_dim * cfg.num_attention_heads;
        let kv_dimension = head_dim * cfg.num_key_value_heads;
        for index in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            for (name, out, input) in [
                ("self_attn.q_proj", q_dimension, cfg.hidden_size),
                ("self_attn.k_proj", kv_dimension, cfg.hidden_size),
                ("self_attn.v_proj", kv_dimension, cfg.hidden_size),
                ("self_attn.o_proj", cfg.hidden_size, q_dimension),
                ("mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size),
                ("mlp.up_proj", cfg.intermediate_size, cfg.hidden_size),
                ("mlp.down_proj", cfg.hidden_size, cfg.intermediate_size),
            ] {
                expect(&format!("{prefix}.{name}.weight"), &[out, input])?;
            }
            for name in ["input_layernorm", "post_attention_layernorm"] {
                expect(&format!("{prefix}.{name}.weight"), &[cfg.hidden_size])?;
            }
            if cfg.model_type == "qwen3" {
                for name in ["q_norm", "k_norm"] {
                    expect(&format!("{prefix}.self_attn.{name}.weight"), &[head_dim])?;
                }
                ensure!(
                    ["q_proj", "k_proj", "v_proj", "o_proj"].iter().all(|name| {
                        !weights.contains_key(&format!("{prefix}.self_attn.{name}.bias"))
                    }),
                    "Qwen3 projection bias is inconsistent with the verified architecture"
                );
            }
            if bias {
                for (name, size) in [
                    ("q_proj", q_dimension),
                    ("k_proj", kv_dimension),
                    ("v_proj", kv_dimension),
                ] {
                    expect(&format!("{prefix}.self_attn.{name}.bias"), &[size])?;
                }
                if cfg.model_type == "llama" {
                    expect(
                        &format!("{prefix}.self_attn.o_proj.bias"),
                        &[cfg.hidden_size],
                    )?;
                }
            }
        }
        let embedding = Embedding::new(embed.clone(), cfg.hidden_size);
        let mut adapter_vars = BTreeMap::new();
        let mut expected_adapter_names = std::collections::BTreeSet::new();
        let mut rng = StableRng::new(seed);
        let mut projection = |name: &str| -> Result<Projection> {
            let base = base_linear(&weights, name, bias)?;
            let Some(ac) = &adapter else {
                return Ok(Projection {
                    base: base.into(),
                    adapter: None,
                });
            };
            ensure!(
                ac.rank > 0 && ac.rank <= cfg.hidden_size && ac.alpha.is_finite() && ac.alpha > 0.,
                "invalid LoRA rank/alpha"
            );
            let (out, input) = base.weight().dims2()?;
            let mut params = Vec::new();
            for (suffix, shape, zero) in [
                ("lora_A.weight", (ac.rank, input), false),
                ("lora_B.weight", (out, ac.rank), true),
            ] {
                let key = format!("{name}.{suffix}");
                expected_adapter_names.insert(key.clone());
                let t = if let Some(loaded) = &loaded {
                    loaded
                        .get(&key)
                        .with_context(|| format!("adapter missing {key}"))?
                        .clone()
                } else if zero {
                    Tensor::zeros(shape, dtype, &dev)?
                } else {
                    let bound = (6. / input as f64).sqrt();
                    Tensor::from_vec(
                        (0..shape.0 * shape.1)
                            .map(|_| ((rng.uniform() * 2. - 1.) * bound) as f32)
                            .collect::<Vec<_>>(),
                        shape,
                        &dev,
                    )?
                    .to_dtype(dtype)?
                };
                ensure!(
                    t.dims() == [shape.0, shape.1]
                        && t.dtype() == dtype
                        && t.device().same_device(&dev),
                    "invalid adapter tensor {key}"
                );
                if trainable {
                    let var = Var::from_tensor(&t)?;
                    params.push(var.as_tensor().clone());
                    adapter_vars.insert(key, var);
                } else {
                    params.push(t.detach());
                }
            }
            Ok(Projection {
                base: base.into(),
                adapter: Some((
                    params.remove(0),
                    params.remove(0),
                    ac.alpha / ac.rank as f64,
                )),
            })
        };
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                q: projection(&format!("{p}.self_attn.q_proj"))?,
                k: base_linear(&weights, &format!("{p}.self_attn.k_proj"), bias)?.into(),
                v: projection(&format!("{p}.self_attn.v_proj"))?,
                o: base_linear(
                    &weights,
                    &format!("{p}.self_attn.o_proj"),
                    cfg.model_type == "llama" && bias,
                )?
                .into(),
                gate: base_linear(&weights, &format!("{p}.mlp.gate_proj"), false)?.into(),
                up: base_linear(&weights, &format!("{p}.mlp.up_proj"), false)?.into(),
                down: base_linear(&weights, &format!("{p}.mlp.down_proj"), false)?.into(),
                norm1: LayerNorm::rms_norm(
                    constant(&weights, &format!("{p}.input_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
                norm2: LayerNorm::rms_norm(
                    constant(&weights, &format!("{p}.post_attention_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
                q_norm: if cfg.model_type == "qwen3" {
                    Some(LayerNorm::rms_norm(
                        constant(&weights, &format!("{p}.self_attn.q_norm.weight"))?,
                        cfg.rms_norm_eps,
                    ))
                } else {
                    None
                },
                k_norm: if cfg.model_type == "qwen3" {
                    Some(LayerNorm::rms_norm(
                        constant(&weights, &format!("{p}.self_attn.k_norm.weight"))?,
                        cfg.rms_norm_eps,
                    ))
                } else {
                    None
                },
                kv: None,
            });
        }
        if let Some(loaded) = &loaded {
            ensure!(
                loaded
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>()
                    == expected_adapter_names,
                "unexpected adapter tensor names"
            );
        }
        let head = if cfg.tie_word_embeddings {
            embed
        } else {
            constant(&weights, "lm_head.weight")?
        };
        let norm = LayerNorm::rms_norm(constant(&weights, "model.norm.weight")?, cfg.rms_norm_eps);
        let dim = head_dim;
        ensure!(dim.is_multiple_of(2), "rotary head dimension must be even");
        let rope_scale = match &cfg.rope_scaling {
            None => 1.,
            Some(scaling) => {
                ensure!(
                    scaling.scaling_type == "linear"
                        && scaling.factor.is_finite()
                        && scaling.factor >= 1.,
                    "unsupported or invalid rotary scaling"
                );
                scaling.factor
            }
        };
        let inv = Tensor::from_vec(
            (0..dim)
                .step_by(2)
                .map(|i| (1. / (rope_scale * cfg.rope_theta.powf(i as f64 / dim as f64))) as f32)
                .collect::<Vec<_>>(),
            (1, dim / 2),
            &dev,
        )?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let phase = t.matmul(&inv)?;
        Ok(Self {
            cfg,
            embedding,
            layers,
            norm,
            lm_head: Linear::new(head, None).into(),
            cos: phase.cos()?.to_dtype(dtype)?,
            sin: phase.sin()?.to_dtype(dtype)?,
            device: dev,
            dtype,
            quantisation: None,
            interleaved_rotary: false,
            adapter_vars,
            _asset_lease: None,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn configuration(&self) -> &QwenConfig {
        &self.cfg
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv = None;
        }
    }

    pub fn hidden(&mut self, input: &Tensor, offset: usize, cache: bool) -> Result<Tensor> {
        Ok(self.hidden_inner(input, offset, cache, None)?.0)
    }

    pub fn forward_trace(&mut self, input: &Tensor, layer: usize) -> Result<ModelTrace> {
        ensure!(layer < self.layers.len(), "trace layer outside model");
        ensure!(
            input.dim(0)? == 1,
            "trace capture requires one bounded sequence"
        );
        let (final_hidden, hidden, attention) = self.hidden_inner(input, 0, false, Some(layer))?;
        Ok(ModelTrace {
            final_hidden,
            layer_hidden: hidden.context("selected hidden trace was not captured")?,
            mean_attention: attention.context("selected attention trace was not captured")?,
        })
    }

    fn hidden_inner(
        &mut self,
        input: &Tensor,
        offset: usize,
        cache: bool,
        capture: Option<usize>,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let (batch, seq) = input.dims2()?;
        ensure!(
            input.device().same_device(&self.device),
            "model input is on a different execution device"
        );
        ensure!(
            seq > 0 && offset + seq <= self.cfg.max_position_embeddings,
            "empty input or context limit exceeded"
        );
        ensure!(
            cache || offset == 0,
            "training cannot reuse a decoding cache"
        );
        let heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let head_dim = self.cfg.head_dimension();
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let mask = if seq > 1 {
            Some(
                Tensor::from_vec(
                    (0..seq)
                        .flat_map(|i| {
                            (0..offset + seq).map(move |j| {
                                if j > offset + i {
                                    f32::NEG_INFINITY
                                } else {
                                    0.
                                }
                            })
                        })
                        .collect::<Vec<_>>(),
                    (1, 1, seq, offset + seq),
                    &self.device,
                )?
                .to_dtype(self.dtype)?,
            )
        } else {
            None
        };
        let mut x = self.embedding.forward(input)?.to_dtype(self.dtype)?;
        let mut captured_hidden = None;
        let mut captured_attention = None;
        for (layer_index, layer) in self.layers.iter_mut().enumerate() {
            let norm = layer.norm1.forward(&x)?;
            let q = layer
                .q
                .forward(&norm)?
                .reshape((batch, seq, heads, head_dim))?
                .transpose(1, 2)?;
            let k = layer
                .k
                .forward(&norm)?
                .reshape((batch, seq, kv_heads, head_dim))?
                .transpose(1, 2)?;
            // Qwen3 normalises each projected head before the rotary transform;
            // the per-channel learned scale is shared across heads.
            let q = apply_head_norm(&q, &layer.q_norm)?;
            let k = apply_head_norm(&k, &layer.k_norm)?;
            let v = layer
                .v
                .forward(&norm)?
                .reshape((batch, seq, kv_heads, head_dim))?
                .transpose(1, 2)?;
            // Slow here means ordinary tensor operations with a correct backward graph.
            let rotate = |input: &Tensor| {
                if self.interleaved_rotary {
                    // GGUF Llama conversion interleaves q/k rows. Dense HF and
                    // GGUF Qwen2 use the original split-half rotary convention.
                    candle_nn::rotary_emb::rope_i(&input.contiguous()?, &cos, &sin)
                } else {
                    candle_nn::rotary_emb::rope_slow(input, &cos, &sin)
                }
            };
            let q = rotate(&q)?;
            let k = rotate(&k)?;
            let (k, v) = if cache && offset > 0 {
                let (pk, pv) = layer.kv.as_ref().context("missing decoding cache")?;
                ensure!(pk.dim(2)? == offset, "decoding cache offset mismatch");
                (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?)
            } else {
                (k, v)
            };
            if cache {
                layer.kv = Some((k.clone(), v.clone()));
            }
            let repeat = |x: Tensor| -> candle_core::Result<Tensor> {
                let len = x.dim(2)?;
                x.unsqueeze(2)?
                    .broadcast_as((batch, kv_heads, heads / kv_heads, len, head_dim))?
                    .reshape((batch, heads, len, head_dim))?
                    .contiguous()
            };
            let k = repeat(k)?;
            let v = repeat(v)?;
            let scores = stable_attention_scores(&q, &k, head_dim, mask.as_ref())?;
            let probabilities = candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?;
            if capture == Some(layer_index) {
                captured_attention = Some(probabilities.mean(1)?.squeeze(0)?);
            }
            let attention = probabilities
                .to_dtype(self.dtype)?
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, seq, heads * head_dim))?;
            x = (&x + layer.o.forward(&attention)?)?;
            let norm = layer.norm2.forward(&x)?;
            let ff =
                (candle_nn::ops::silu(&layer.gate.forward(&norm)?)? * layer.up.forward(&norm)?)?;
            x = (&x + layer.down.forward(&ff)?)?;
            if capture == Some(layer_index) {
                captured_hidden = Some(x.squeeze(0)?);
            }
        }
        Ok((self.norm.forward(&x)?, captured_hidden, captured_attention))
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward(hidden)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    #[serde(default = "default_tokens")]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub seed: u64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,
}
fn default_tokens() -> usize {
    256
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.8
}
fn default_timeout() -> u64 {
    120
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Generation {
    pub completion: String,
    pub checkpoint_id: String,
    pub finish_reason: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub elapsed_ms: u128,
    pub complete: bool,
}

pub fn chat_prefix(prompt: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}

pub struct ModelRuntime {
    pub model: QwenModel,
    pub tokenizer: Tokenizer,
    pub assets: AssetManifest,
    pub checkpoint_id: String,
    pub options: RuntimeOptions,
    chat: ChatFormat,
    eos_token_ids: Vec<u32>,
    context_cap: usize,
}

enum ChatFormat {
    PinnedQwen,
    Original {
        template: String,
        configuration: serde_json::Value,
    },
}

/// Reuse one original tokeniser/template for memory planning across task prompts.
/// Small auxiliary hashes are checked here; the full trial/runtime validators
/// separately verify publisher provenance and every weight before inference.
pub struct ComparativePromptEncoder {
    tokenizer: Tokenizer,
    chat: ChatFormat,
}

impl ComparativePromptEncoder {
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn formatted_prompt(&self, prompt: &str) -> Result<String> {
        match &self.chat {
            ChatFormat::PinnedQwen => Ok(chat_prefix(prompt)),
            ChatFormat::Original {
                template,
                configuration,
            } => render_original_chat(template, configuration, prompt),
        }
    }

    pub fn load(base_dir: &Path, gguf: bool) -> Result<Self> {
        let manifest: AssetManifest = serde_json::from_slice(&fs::read(base_dir.join(if gguf {
            "gguf-assets.json"
        } else {
            "assets.json"
        }))?)?;
        let source = if gguf {
            base_dir.join("source")
        } else {
            base_dir.to_owned()
        };
        let root = base_dir.canonicalize()?;
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            let relative = if gguf {
                format!("source/{name}")
            } else {
                name.to_owned()
            };
            let expected = manifest
                .files
                .get(&relative)
                .context("tokeniser/template checksum is absent from the model manifest")?;
            let path = source.join(name).canonicalize()?;
            ensure!(
                path.starts_with(&root),
                "tokeniser/template escapes the model directory"
            );
            ensure!(
                &sha256_file(&path)? == expected,
                "original tokeniser/template bytes changed"
            );
        }
        let tokenizer = Tokenizer::from_file(source.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let configuration =
            serde_json::from_slice(&fs::read(source.join("tokenizer_config.json"))?)?;
        let chat = if !gguf && manifest.model_id == MODEL_ID && manifest.revision == MODEL_REVISION
        {
            ChatFormat::PinnedQwen
        } else {
            original_chat_format(configuration)?
        };
        Ok(Self { tokenizer, chat })
    }

    pub fn input_tokens(&self, prompt: &str) -> Result<usize> {
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= 65536,
            "prompt must contain 1–65536 bytes"
        );
        let rendered = match &self.chat {
            ChatFormat::PinnedQwen => chat_prefix(prompt),
            ChatFormat::Original {
                template,
                configuration,
            } => render_original_chat(template, configuration, prompt)?,
        };
        Ok(self
            .tokenizer
            .encode(rendered, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?
            .get_ids()
            .len())
    }
}

pub fn comparative_input_tokens(base_dir: &Path, gguf: bool, prompt: &str) -> Result<usize> {
    ComparativePromptEncoder::load(base_dir, gguf)?.input_tokens(prompt)
}

fn render_original_chat(
    template: &str,
    configuration: &serde_json::Value,
    prompt: &str,
) -> Result<String> {
    let mut environment = minijinja::Environment::new();
    // Hugging Face's original apply_chat_template environment strips block
    // whitespace. Matching those settings avoids adding untrained prompt tokens.
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_fuel(Some(1_000_000));
    environment.set_recursion_limit(64);
    environment.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    environment.add_function(
        "raise_exception",
        |message: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    environment.add_template("original-chat", template)?;
    let mut context = configuration
        .as_object()
        .context("tokeniser configuration must be an object")?
        .clone();
    // Original tokeniser metadata can encode special tokens as AddedToken objects.
    // Jinja receives their original content, exactly as apply_chat_template does.
    for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
        if let Some(value) = context.get_mut(key)
            && let Some(content) = value.get("content").and_then(serde_json::Value::as_str)
        {
            *value = serde_json::Value::String(content.to_owned());
        }
    }
    context.insert(
        "messages".into(),
        serde_json::json!([{"role":"user","content":prompt}]),
    );
    context.insert("add_generation_prompt".into(), serde_json::json!(true));
    context.insert("tools".into(), serde_json::Value::Null);
    context.insert("documents".into(), serde_json::Value::Null);
    struct BoundedPrompt(Vec<u8>);
    impl Write for BoundedPrompt {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0.len().saturating_add(bytes.len()) > 262144 {
                return Err(std::io::Error::other(
                    "original chat template exceeds the prompt byte limit",
                ));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut output = BoundedPrompt(Vec::new());
    environment
        .get_template("original-chat")?
        .render_captured_to(context, &mut output)?;
    let rendered = String::from_utf8(output.0)?;
    ensure!(
        !rendered.trim().is_empty(),
        "original chat template produced an empty prompt"
    );
    Ok(rendered)
}

fn original_chat_format(configuration: serde_json::Value) -> Result<ChatFormat> {
    let value = &configuration["chat_template"];
    let template = value
        .as_str()
        .or_else(|| {
            value
                .as_array()?
                .iter()
                .find(|item| item["name"] == "default")?["template"]
                .as_str()
        })
        .context("an original default chat template is required; template guessing is forbidden")?
        .to_owned();
    Ok(ChatFormat::Original {
        template,
        configuration,
    })
}

pub(crate) fn parse_eos_ids(value: &serde_json::Value) -> Result<Vec<u32>> {
    let values = match value {
        serde_json::Value::Array(values) => values.clone(),
        _ => vec![value.clone()],
    };
    let ids = values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .context("original EOS token ID must be a nonnegative u32")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!ids.is_empty(), "original model has no EOS token IDs");
    Ok(ids)
}

impl ModelRuntime {
    /// Trusted train-only health probe reuses the resident model. No second base
    /// is allocated and no optimiser state is altered by this constructor.
    pub(crate) fn from_pinned_qwen_model(
        model: QwenModel,
        tokenizer: Tokenizer,
        assets: AssetManifest,
        checkpoint_id: String,
        options: RuntimeOptions,
    ) -> Self {
        let context_cap = 4096.min(model.configuration().max_position_embeddings);
        Self {
            model,
            tokenizer,
            assets,
            checkpoint_id,
            options,
            chat: ChatFormat::PinnedQwen,
            eos_token_ids: vec![151645, 151643],
            context_cap,
        }
    }
    pub fn formatted_prompt(&self, prompt: &str) -> Result<String> {
        match &self.chat {
            ChatFormat::PinnedQwen => Ok(chat_prefix(prompt)),
            ChatFormat::Original {
                template,
                configuration,
            } => render_original_chat(template, configuration, prompt),
        }
    }

    pub fn primary_eos_token_id(&self) -> u32 {
        self.eos_token_ids[0]
    }
    pub fn load(base_dir: &Path, checkpoint: Option<&Path>) -> Result<Self> {
        Self::load_with_options(base_dir, checkpoint, RuntimeOptions::default())
    }

    /// The assessment base retains its exact revision/hash/template contract.
    pub fn load_with_options(
        base_dir: &Path,
        checkpoint: Option<&Path>,
        options: RuntimeOptions,
    ) -> Result<Self> {
        let assets = validate_assets(base_dir)?;
        Self::load_validated(
            base_dir,
            checkpoint,
            assets,
            options,
            ChatFormat::PinnedQwen,
            vec![151645, 151643],
            4096,
        )
    }

    /// Comparator assets have independent publisher/revision/hash provenance.
    /// They cannot weaken the pinned assessment loader or create an adapter.
    pub fn load_comparative_with_options(base_dir: &Path, options: RuntimeOptions) -> Result<Self> {
        if base_dir.join("gguf-assets.json").exists() {
            return Self::load_quantised_with_options(base_dir, options);
        }
        let lease = AssetLease::acquire(base_dir)?;
        let assets = crate::hub::validate_downloaded_assets(base_dir)?;
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
        let tokeniser_config =
            serde_json::from_slice(&fs::read(base_dir.join("tokenizer_config.json"))?)?;
        let generation = if assets.files.contains_key("generation_config.json") {
            serde_json::from_slice::<serde_json::Value>(&fs::read(
                base_dir.join("generation_config.json"),
            )?)?
        } else {
            serde_json::Value::Null
        };
        let eos = generation
            .get("eos_token_id")
            .filter(|value| !value.is_null())
            .unwrap_or(&config["eos_token_id"]);
        let eos = parse_eos_ids(eos)?;
        let mut runtime = Self::load_validated(
            base_dir,
            None,
            assets,
            options,
            original_chat_format(tokeniser_config)?,
            eos,
            2048,
        )?;
        runtime.model._asset_lease = Some(lease.with_device(runtime.model.device()));
        Ok(runtime)
    }

    /// GGUF weights and original source tokeniser/config have independent,
    /// immutable publisher provenance. Quantised matmul uses F32 activations;
    /// this never expands the complete matrix set into floating-point weights.
    pub fn load_quantised_with_options(base_dir: &Path, options: RuntimeOptions) -> Result<Self> {
        Self::load_quantised_with_context(base_dir, options, 1024)
    }

    pub fn load_quantised_with_context(
        base_dir: &Path,
        options: RuntimeOptions,
        context_tokens: usize,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&context_tokens),
            "quantised runtime context must be 1–4096 tokens"
        );
        ensure!(
            options.dtype == InferenceDtype::F32,
            "native GGUF matmul requires explicit F32 activations; choose dtype f32, while weights remain quantised"
        );
        let lease = AssetLease::acquire(base_dir)?;
        let proof = crate::hub::validate_gguf_assets(base_dir)?;
        let source = base_dir.join(&proof.source_assets.directory);
        let configuration: serde_json::Value =
            serde_json::from_slice(&fs::read(source.join("config.json"))?)?;
        let cfg: QwenConfig = serde_json::from_value(configuration.clone())?;
        cfg.validate()?;
        let context_cap = context_tokens.min(cfg.max_position_embeddings);
        let tokeniser_configuration =
            serde_json::from_slice(&fs::read(source.join("tokenizer_config.json"))?)?;
        let generation = if proof
            .source_assets
            .manifest
            .files
            .contains_key("generation_config.json")
        {
            serde_json::from_slice::<serde_json::Value>(&fs::read(
                source.join("generation_config.json"),
            )?)?
        } else {
            serde_json::Value::Null
        };
        let eos_token_ids = parse_eos_ids(
            generation
                .get("eos_token_id")
                .filter(|value| !value.is_null())
                .unwrap_or(&configuration["eos_token_id"]),
        )?;
        ensure!(
            eos_token_ids
                .iter()
                .all(|id| (*id as usize) < cfg.vocab_size),
            "GGUF original EOS is outside its vocabulary"
        );
        let tokenizer = Tokenizer::from_file(source.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let chat = original_chat_format(tokeniser_configuration)?;
        let (device, _) = options.resolve()?;
        let mut file = fs::File::open(base_dir.join(&proof.filename))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        quantised::validate_tokeniser_metadata(
            &content,
            &tokenizer,
            &configuration,
            &eos_token_ids,
        )?;
        let mut model = QwenModel::from_gguf(
            cfg,
            &content,
            &mut file,
            &device,
            proof.file_bytes,
            context_cap,
        )?;
        if let Some(metadata) = model.quantisation.as_mut() {
            metadata["original_source_model_id"] =
                serde_json::json!(proof.source_assets.manifest.model_id);
            metadata["original_source_revision"] =
                serde_json::json!(proof.source_assets.manifest.revision);
            metadata["quantised_publisher_model_id"] = serde_json::json!(proof.model_id);
            metadata["quantised_publisher_revision"] = serde_json::json!(proof.revision);
            metadata["weight_sha256"] = serde_json::json!(proof.file_sha256);
        }
        model._asset_lease = Some(lease.with_device(model.device()));
        Ok(Self {
            model,
            tokenizer,
            assets: proof.assets,
            checkpoint_id: proof.file_sha256,
            options,
            chat,
            eos_token_ids,
            context_cap,
        })
    }

    fn load_validated(
        base_dir: &Path,
        checkpoint: Option<&Path>,
        assets: AssetManifest,
        options: RuntimeOptions,
        chat: ChatFormat,
        eos_token_ids: Vec<u32>,
        context_cap: usize,
    ) -> Result<Self> {
        let (device, dtype) = options.resolve()?;
        let cfg: QwenConfig = serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
        cfg.validate()?;
        ensure!(
            eos_token_ids
                .iter()
                .all(|id| (*id as usize) < cfg.vocab_size),
            "original EOS token is outside the model vocabulary"
        );
        let context_cap = context_cap.min(cfg.max_position_embeddings);
        let weight_files = assets
            .files
            .iter()
            .filter(|(name, _)| name.ends_with(".safetensors"))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            !weight_files.is_empty(),
            "validated model assets contain no safetensors weights"
        );
        let base_checkpoint_id = if weight_files.len() == 1 {
            (*weight_files.values().next().expect("nonempty weights")).clone()
        } else {
            crate::artefact::hash(&serde_json::to_vec(&weight_files)?)
        };
        let mut weights = HashMap::new();
        for name in weight_files.keys() {
            for (name, value) in candle_core::safetensors::load(base_dir.join(name), &device)? {
                ensure!(
                    weights
                        .insert(name.clone(), value.to_dtype(dtype)?)
                        .is_none(),
                    "duplicate model tensor across shards: {name}"
                );
            }
        }
        let tokenizer = Tokenizer::from_file(base_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let (adapter, loaded, checkpoint_id) = match checkpoint {
            None => (None, None, base_checkpoint_id),
            Some(path) => {
                let manifest: crate::train::CheckpointManifest =
                    serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
                crate::train::validate_checkpoint(path, &manifest, &assets)?;
                let loaded =
                    candle_core::safetensors::load(path.join("adapter.safetensors"), &device)?
                        .into_iter()
                        .map(|(name, tensor)| Ok((name, tensor.to_dtype(dtype)?)))
                        .collect::<Result<HashMap<_, _>>>()?;
                (
                    Some(manifest.adapter),
                    Some(loaded),
                    manifest.adapter_sha256,
                )
            }
        };
        let model = QwenModel::from_weights(cfg, weights, adapter, 0, loaded, false)?;
        Ok(Self {
            model,
            tokenizer,
            assets,
            checkpoint_id,
            options,
            chat,
            eos_token_ids,
            context_cap,
        })
    }

    pub fn execution_metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "engine": "native_rust_candle",
            "requested": self.options,
            "actual_device": format!("{:?}", self.model.device().location()),
            "actual_dtype": format!("{:?}", self.model.dtype()),
            "weight_format": if self.model.quantisation.is_some() {"gguf"} else {"safetensors"},
            "quantisation": self.model.quantisation,
            "cuda_feature_compiled": cfg!(feature = "cuda"),
            "cuda_active": self.model.device().is_cuda(),
            "cpu_fallback": false,
            "asset_lifetime_shared_lease": self.model._asset_lease.is_some(),
            "available_cpu_threads": std::thread::available_parallelism().ok().map(usize::from),
            "candle_cpu_threads": candle_core::utils::get_num_threads(),
            "model_id": self.assets.model_id,
            "revision": self.assets.revision,
            "checkpoint_id": self.checkpoint_id,
            "context_cap": self.context_cap,
        })
    }

    pub fn generate(&mut self, request: GenerationRequest) -> Result<Generation> {
        self.generate_cancellable(request, Arc::new(AtomicBool::new(false)))
    }
    pub fn generate_cancellable(
        &mut self,
        request: GenerationRequest,
        cancel: Arc<AtomicBool>,
    ) -> Result<Generation> {
        ensure!(
            !request.prompt.trim().is_empty() && request.prompt.len() <= 65536,
            "prompt must contain 1–65536 bytes"
        );
        ensure!(
            (1..=1024).contains(&request.max_new_tokens),
            "output token limit must be 1–1024"
        );
        ensure!(
            request.temperature.is_finite()
                && (0.0..=2.0).contains(&request.temperature)
                && request.top_p.is_finite()
                && (0.0..=1.0).contains(&request.top_p)
                && request.top_p > 0.,
            "invalid sampling parameters"
        );
        ensure!(
            (1..=1800).contains(&request.timeout_seconds),
            "invalid inference deadline"
        );
        let started = Instant::now();
        let deadline = started + Duration::from_secs(request.timeout_seconds);
        let enc = self
            .tokenizer
            .encode(
                match &self.chat {
                    ChatFormat::PinnedQwen => chat_prefix(&request.prompt),
                    ChatFormat::Original {
                        template,
                        configuration,
                    } => render_original_chat(template, configuration, &request.prompt)?,
                },
                false,
            )
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut tokens = enc.get_ids().to_vec();
        let input_tokens = tokens.len();
        ensure!(
            input_tokens + request.max_new_tokens <= self.context_cap,
            "local inference context cap is {} tokens",
            self.context_cap
        );
        let mut rng = StableRng::new(request.seed);
        let mut generated = Vec::new();
        let mut reason = "length";
        self.model.clear_cache();
        let result = (|| -> Result<()> {
            for i in 0..request.max_new_tokens {
                if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    reason = "timeout";
                    break;
                }
                let (ids, offset) = if i == 0 {
                    (&tokens[..], 0)
                } else {
                    (&tokens[tokens.len() - 1..], tokens.len() - 1)
                };
                let input = Tensor::from_slice(ids, (1, ids.len()), self.model.device())?;
                let hidden = self.model.hidden(&input, offset, true)?;
                let logits = self
                    .model
                    .logits(&hidden.i((.., ids.len() - 1, ..))?)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                // Host transfer synchronises CUDA. Expiry during a forward pass
                // must not turn a late EOS into a successful completion.
                if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    reason = "timeout";
                    break;
                }
                let next = sample(&logits, request.temperature, request.top_p, &mut rng)?;
                if self.eos_token_ids.contains(&next) {
                    reason = "stop";
                    break;
                }
                tokens.push(next);
                generated.push(next);
            }
            Ok(())
        })();
        self.model.clear_cache();
        result?;
        let completion = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
            reason = "timeout";
        }
        ensure!(
            !completion.trim().is_empty(),
            "model returned no output ({reason})"
        );
        Ok(Generation {
            completion,
            checkpoint_id: self.checkpoint_id.clone(),
            finish_reason: reason.into(),
            input_tokens,
            output_tokens: generated.len(),
            elapsed_ms: started.elapsed().as_millis(),
            complete: reason == "stop",
        })
    }
}

pub(crate) fn sample(
    logits: &[f32],
    temperature: f64,
    top_p: f64,
    rng: &mut StableRng,
) -> Result<u32> {
    ensure!(
        !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
        "non-finite or empty generation logits"
    );
    let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if temperature == 0. {
        return Ok(ranked[0].0 as u32);
    }
    let max = ranked[0].1 as f64;
    let mut probs = ranked
        .into_iter()
        .map(|(i, x)| (i, ((x as f64 - max) / temperature).exp()))
        .collect::<Vec<_>>();
    let sum: f64 = probs.iter().map(|x| x.1).sum();
    let mut cumulative = 0.;
    let mut keep = 0;
    for (_, p) in &probs {
        cumulative += p / sum;
        keep += 1;
        if cumulative >= top_p {
            break;
        }
    }
    probs.truncate(keep);
    let sum: f64 = probs.iter().map(|x| x.1).sum();
    let draw = rng.uniform() * sum;
    let mut acc = 0.;
    for (i, p) in &probs {
        acc += p;
        if draw <= acc {
            return Ok(*i as u32);
        }
    }
    probs.last().map(|x| x.0 as u32).context("empty nucleus")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsupported_phase_runtimes_fail_without_allocating_a_model() {
        RuntimeOptions::default().validate_build_support().unwrap();
        assert!(
            RuntimeOptions {
                device: ExecutionDevice::Cpu,
                dtype: InferenceDtype::F16
            }
            .validate_build_support()
            .is_err()
        );
        if !cfg!(feature = "cuda") {
            assert!(RuntimeOptions::cuda(0).validate_build_support().is_err());
        }
    }
    #[test]
    fn deterministic_sampling_and_rejection() {
        let logits = [0., 1., 2.];
        let mut a = StableRng::new(17);
        let mut b = StableRng::new(17);
        for _ in 0..20 {
            assert_eq!(
                sample(&logits, 0.7, 0.9, &mut a).unwrap(),
                sample(&logits, 0.7, 0.9, &mut b).unwrap()
            );
        }
        assert!(sample(&[f32::NAN], 1., 1., &mut a).is_err());
        assert_eq!(sample(&logits, 0., 1., &mut a).unwrap(), 2);
    }
    #[test]
    fn chat_template_plain_message_branch() {
        assert_eq!(
            chat_prefix("Hello"),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn original_deepseek_template_and_eos_preserve_family_contract() {
        let configuration: serde_json::Value = serde_json::from_str(include_str!(
            "../docs/comparative-models/provenance/deepseek-coder-1.3b/tokenizer_config.json"
        ))
        .unwrap();
        let template = configuration["chat_template"].as_str().unwrap();
        let actual = render_original_chat(template, &configuration, "fn add").unwrap();
        assert_eq!(
            actual,
            "<｜begin▁of▁sentence｜>You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\nfn add\n### Response:\n"
        );
        assert_eq!(
            parse_eos_ids(&serde_json::json!(32021)).unwrap(),
            vec![32021]
        );
        assert_eq!(
            parse_eos_ids(&serde_json::json!([151645, 151643])).unwrap(),
            vec![151645, 151643]
        );
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!([]),
            serde_json::json!("32021"),
        ] {
            assert!(parse_eos_ids(&invalid).is_err());
        }
    }

    #[test]
    fn explicit_device_selection_never_silently_falls_back() {
        let (device, dtype) = RuntimeOptions::default().resolve().unwrap();
        assert!(!device.is_cuda());
        assert_eq!(dtype, DType::F32);
        assert!(
            RuntimeOptions {
                device: ExecutionDevice::Cpu,
                dtype: InferenceDtype::F16
            }
            .resolve()
            .is_err()
        );
        #[cfg(not(feature = "cuda"))]
        assert!(RuntimeOptions::cuda(0).resolve().is_err());
    }

    #[test]
    fn original_llama_configuration_rejects_invalid_dimensions_and_semantics() {
        let original: QwenConfig = serde_json::from_str(include_str!(
            "../docs/comparative-models/provenance/deepseek-coder-1.3b/config.json"
        ))
        .unwrap();
        original.validate().unwrap();
        assert_eq!(original.model_type, "llama");
        assert!(
            !original
                .attention_bias
                .unwrap_or(original.model_type == "qwen2")
        );
        assert_eq!(original.rope_scaling.as_ref().unwrap().factor, 4.);
        for change in 0..8 {
            let mut cfg = original.clone();
            match change {
                0 => cfg.hidden_size = 0,
                1 => cfg.num_attention_heads = 0,
                2 => cfg.max_position_embeddings = usize::MAX,
                3 => cfg.rope_theta = f64::NAN,
                4 => cfg.rms_norm_eps = 0.,
                5 => cfg.hidden_act = "gelu".into(),
                6 => cfg.mlp_bias = true,
                _ => cfg.rope_scaling.as_mut().unwrap().scaling_type = "unverified_dynamic".into(),
            }
            assert!(
                cfg.validate().is_err(),
                "invalid configuration variant {change}"
            );
        }
    }

    #[test]
    fn original_chat_rendering_bounds_output_and_computation() {
        let cfg = serde_json::json!({"bos_token":"", "eos_token":""});
        assert!(
            render_original_chat("{% for i in range(100000000) %}x{% endfor %}", &cfg, "test")
                .is_err()
        );
        assert!(
            render_original_chat(
                "{% for i in range(100) %}{{ messages[0].content }}{% endfor %}",
                &cfg,
                &"x".repeat(4096)
            )
            .is_err()
        );
        assert!(render_original_chat("{{ unknown_mandatory_variable }}", &cfg, "test").is_err());
    }

    #[test]
    fn frozen_projection_matches_generic_forward_and_input_gradient() {
        let input = Var::new(&[[1_f32, 2., 3.], [4., 5., 6.]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[[0.1_f32, 0.2, 0.3], [0.4, 0.5, 0.6]], &Device::Cpu).unwrap();
        let base = Linear::new(weight, None);
        let frozen = frozen_linear(&base, input.as_tensor()).unwrap();
        let generic = base.forward(input.as_tensor()).unwrap();
        assert_eq!(
            frozen.to_vec2::<f32>().unwrap(),
            generic.to_vec2::<f32>().unwrap()
        );
        let fg = frozen.sqr().unwrap().sum_all().unwrap().backward().unwrap();
        let gg = generic
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(
            fg.get(&input).unwrap().to_vec2::<f32>().unwrap(),
            gg.get(&input).unwrap().to_vec2::<f32>().unwrap()
        );
        assert!(fg.get(base.weight()).is_none());
    }

    #[test]
    fn finite_float16_queries_cannot_overflow_attention_scores_before_scaling() {
        let q = Tensor::from_slice(&[300_f32, 300.], (1, 1, 1, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let k = Tensor::from_slice(&[300_f32, 300., 0., 0.], (1, 1, 2, 2), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let naive = q
            .matmul(&k.transpose(2, 3).unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            naive[0].is_infinite(),
            "the fixture must exercise a dot product outside float16's finite range"
        );
        let stable = stable_attention_scores(&q, &k, 2, None).unwrap();
        assert_eq!(stable.dtype(), DType::F32);
        let scores = stable.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(scores.iter().all(|value| value.is_finite()));
        assert!((scores[0] - 180000_f32 / 2_f32.sqrt()).abs() < 0.02);
        let probabilities = candle_nn::ops::softmax(&stable, D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(probabilities, vec![1., 0.]);
    }

    /// A random unit model verifies the complete backward architecture. It is not
    /// a provider-trained checkpoint or evidence of assessment improvement.
    #[test]
    fn full_qwen_architecture_backpropagates_and_preserves_frozen_base() {
        let cfg = QwenConfig {
            model_type: "qwen2".into(),
            attention_bias: None,
            rope_scaling: None,
            hidden_act: "silu".into(),
            mlp_bias: false,
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: None,
            max_position_embeddings: 64,
            rope_theta: 10000.,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            use_sliding_window: false,
        };
        let mut weights = HashMap::new();
        let mut rng = StableRng::new(4);
        let mut weight = |name: String, shape: (usize, usize)| {
            weights.insert(
                name,
                Tensor::from_vec(
                    (0..shape.0 * shape.1)
                        .map(|_| ((rng.uniform() * 2. - 1.) * 0.15) as f32)
                        .collect::<Vec<_>>(),
                    shape,
                    &Device::Cpu,
                )
                .unwrap(),
            );
        };
        weight("model.embed_tokens.weight".into(), (32, 16));
        for i in 0..2 {
            for (p, shape) in [
                ("self_attn.q_proj", (16, 16)),
                ("self_attn.k_proj", (8, 16)),
                ("self_attn.v_proj", (8, 16)),
                ("self_attn.o_proj", (16, 16)),
                ("mlp.gate_proj", (32, 16)),
                ("mlp.up_proj", (32, 16)),
                ("mlp.down_proj", (16, 32)),
            ] {
                weight(format!("model.layers.{i}.{p}.weight"), shape);
            }
        }
        for i in 0..2 {
            for (p, n) in [("q_proj", 16), ("k_proj", 8), ("v_proj", 8)] {
                weights.insert(
                    format!("model.layers.{i}.self_attn.{p}.bias"),
                    Tensor::zeros(n, DType::F32, &Device::Cpu).unwrap(),
                );
            }
            for p in ["input_layernorm", "post_attention_layernorm"] {
                weights.insert(
                    format!("model.layers.{i}.{p}.weight"),
                    Tensor::ones(16, DType::F32, &Device::Cpu).unwrap(),
                );
            }
        }
        weights.insert(
            "model.norm.weight".into(),
            Tensor::ones(16, DType::F32, &Device::Cpu).unwrap(),
        );
        let frozen = weights["model.embed_tokens.weight"].clone();
        let original = frozen.to_vec2::<f32>().unwrap();
        let reload_weights = weights.clone();
        let mut model = QwenModel::from_weights(
            cfg.clone(),
            weights,
            Some(AdapterConfig { rank: 2, alpha: 4. }),
            1701,
            None,
            true,
        )
        .unwrap();
        let sample = crate::train::EncodedSample {
            id: "unit tensor only".into(),
            input_ids: vec![1, 2, 3, 4, 5],
            positions: vec![2, 3, 4],
            targets: vec![4, 5, 6],
        };
        let loss = crate::train::assistant_loss(&mut model, &sample).unwrap();
        let gradients = loss.backward().unwrap();
        assert_eq!(model.adapter_vars.len(), 8);
        for (name, var) in &model.adapter_vars {
            let g = gradients
                .get(var)
                .unwrap_or_else(|| panic!("missing gradient {name}"));
            let norm = g
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(norm.is_finite());
            if name.contains("lora_B") {
                assert!(norm > 0., "no attention gradient {name}");
            }
        }
        let before = model
            .adapter_vars
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut optimiser = crate::train::ResumableAdam::new(
            &model.adapter_vars,
            &crate::train::TrainConfig::default(),
        )
        .unwrap();
        let norm = optimiser.update(&model.adapter_vars, &loss).unwrap();
        assert!(norm > 0.);
        assert!(
            model
                .adapter_vars
                .iter()
                .any(|(k, v)| before[k] != v.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        );
        assert_eq!(frozen.to_vec2::<f32>().unwrap(), original);
        let loss = crate::train::assistant_loss(&mut model, &sample).unwrap();
        let gradients = loss.backward().unwrap();
        for (name, var) in &model.adapter_vars {
            let norm = gradients
                .get(var)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                norm.is_finite() && norm > 0.,
                "zero/non-finite gradient after update: {name}"
            );
        }
        let input =
            Tensor::from_slice(&sample.input_ids, (1, sample.input_ids.len()), &Device::Cpu)
                .unwrap();
        let full_hidden = model
            .hidden(&input, 0, false)
            .unwrap()
            .i((.., sample.input_ids.len() - 1, ..))
            .unwrap();
        let expected = model
            .logits(&full_hidden)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        model.clear_cache();
        let prefix = Tensor::from_slice(&sample.input_ids[..3], (1, 3), &Device::Cpu).unwrap();
        model.hidden(&prefix, 0, true).unwrap();
        let mut last = None;
        for offset in 3..sample.input_ids.len() {
            let next =
                Tensor::from_slice(&sample.input_ids[offset..offset + 1], (1, 1), &Device::Cpu)
                    .unwrap();
            last = Some(model.hidden(&next, offset, true).unwrap());
        }
        let cached = model
            .logits(&last.unwrap().squeeze(1).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(
            expected
                .iter()
                .zip(&cached)
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "KV decoding disagrees with full causal attention"
        );
        model.clear_cache();
        let tmp = tempfile::tempdir().unwrap();
        let adapter_path = tmp.path().join("unit-only.safetensors");
        let saved = model
            .adapter_vars
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
            .collect::<HashMap<_, _>>();
        candle_core::safetensors::save(&saved, &adapter_path).unwrap();
        let loaded = candle_core::safetensors::load(&adapter_path, &Device::Cpu).unwrap();
        let mut reloaded = QwenModel::from_weights(
            cfg,
            reload_weights,
            Some(AdapterConfig { rank: 2, alpha: 4. }),
            0,
            Some(loaded),
            false,
        )
        .unwrap();
        assert!(reloaded.adapter_vars.is_empty());
        let hidden = reloaded
            .hidden(&input, 0, false)
            .unwrap()
            .i((.., sample.input_ids.len() - 1, ..))
            .unwrap();
        assert_eq!(
            expected,
            reloaded
                .logits(&hidden)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let trace = model.forward_trace(&input, 0).unwrap();
        assert_eq!(trace.layer_hidden.dims(), &[5, 16]);
        assert_eq!(trace.mean_attention.dims(), &[5, 5]);
        assert_eq!(trace.final_hidden.dims(), &[1, 5, 16]);
        assert!(
            trace
                .mean_attention
                .sum(1)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|x| (x - 1.).abs() < 1e-6)
        );
        let trace_loss = (trace.layer_hidden.sqr().unwrap().mean_all().unwrap()
            + trace.mean_attention.sqr().unwrap().mean_all().unwrap())
        .unwrap();
        let trace_gradients = trace_loss.backward().unwrap();
        for (name, var) in &model.adapter_vars {
            if name.contains("layers.0") {
                let gradient = trace_gradients
                    .get(var)
                    .unwrap_or_else(|| panic!("captured layer lost gradient {name}"));
                assert!(
                    gradient
                        .sqr()
                        .unwrap()
                        .sum_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                        .is_finite()
                );
            } else {
                assert!(
                    trace_gradients.get(var).is_none(),
                    "later layer unexpectedly enters a selected-layer objective"
                );
            }
        }
    }
}
