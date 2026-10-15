//! Native, CPU-capable Qwen2 inference and differentiable rank-limited adaptation.
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

pub const MODEL_ID: &str = "Qwen/Qwen2.5-Coder-0.5B-Instruct";
pub const MODEL_REVISION: &str = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25";
pub const MODEL_SHA256: &str = "f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44";
pub const TOKENIZER_SHA256: &str =
    "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539";
pub const SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const PINNED_FILES: &[(&str, &str)] = &[
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
        if !path.exists() {
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
    atomic_write(
        &base_dir.join("assets.json"),
        &serde_json::to_vec_pretty(&manifest)?,
    )?;
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
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
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

#[derive(Clone)]
struct Projection {
    base: Linear,
    adapter: Option<(Tensor, Tensor, f64)>,
}
impl Projection {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let base = frozen_linear(&self.base, x)?;
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

struct Layer {
    q: Projection,
    k: Linear,
    v: Projection,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    kv: Option<(Tensor, Tensor)>,
}

/// Base tensors are constants. Only q/v adapter tensors can enter this variable map.
pub struct QwenModel {
    cfg: QwenConfig,
    embedding: Embedding,
    layers: Vec<Layer>,
    norm: LayerNorm,
    lm_head: Linear,
    cos: Tensor,
    sin: Tensor,
    pub adapter_vars: AdapterVariables,
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
        ensure!(
            !cfg.use_sliding_window,
            "sliding-window attention is not supported by this pinned architecture"
        );
        ensure!(
            cfg.num_attention_heads > 0
                && cfg.num_key_value_heads > 0
                && cfg.hidden_size.is_multiple_of(cfg.num_attention_heads)
                && cfg
                    .num_attention_heads
                    .is_multiple_of(cfg.num_key_value_heads),
            "invalid Qwen head dimensions"
        );
        let dev = Device::Cpu;
        let embed = constant(&weights, "model.embed_tokens.weight")?;
        let embedding = Embedding::new(embed.clone(), cfg.hidden_size);
        let mut adapter_vars = BTreeMap::new();
        let mut expected_adapter_names = std::collections::BTreeSet::new();
        let mut rng = StableRng::new(seed);
        let mut projection = |name: &str| -> Result<Projection> {
            let base = base_linear(&weights, name, true)?;
            let Some(ac) = &adapter else {
                return Ok(Projection {
                    base,
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
                    Tensor::zeros(shape, DType::F32, &dev)?
                } else {
                    let bound = (6. / input as f64).sqrt();
                    Tensor::from_vec(
                        (0..shape.0 * shape.1)
                            .map(|_| ((rng.uniform() * 2. - 1.) * bound) as f32)
                            .collect::<Vec<_>>(),
                        shape,
                        &dev,
                    )?
                };
                ensure!(
                    t.dims() == [shape.0, shape.1] && t.dtype() == DType::F32,
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
                base,
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
                k: base_linear(&weights, &format!("{p}.self_attn.k_proj"), true)?,
                v: projection(&format!("{p}.self_attn.v_proj"))?,
                o: base_linear(&weights, &format!("{p}.self_attn.o_proj"), false)?,
                gate: base_linear(&weights, &format!("{p}.mlp.gate_proj"), false)?,
                up: base_linear(&weights, &format!("{p}.mlp.up_proj"), false)?,
                down: base_linear(&weights, &format!("{p}.mlp.down_proj"), false)?,
                norm1: LayerNorm::rms_norm(
                    constant(&weights, &format!("{p}.input_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
                norm2: LayerNorm::rms_norm(
                    constant(&weights, &format!("{p}.post_attention_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
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
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        ensure!(dim.is_multiple_of(2), "rotary head dimension must be even");
        let inv = Tensor::from_vec(
            (0..dim)
                .step_by(2)
                .map(|i| (1. / cfg.rope_theta.powf(i as f64 / dim as f64)) as f32)
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
            lm_head: Linear::new(head, None),
            cos: phase.cos()?,
            sin: phase.sin()?,
            adapter_vars,
        })
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.kv = None;
        }
    }

    pub fn hidden(&mut self, input: &Tensor, offset: usize, cache: bool) -> Result<Tensor> {
        let (batch, seq) = input.dims2()?;
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
        let head_dim = self.cfg.hidden_size / heads;
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let mask = if seq > 1 {
            Some(Tensor::from_vec(
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
                &Device::Cpu,
            )?)
        } else {
            None
        };
        let mut x = self.embedding.forward(input)?;
        for layer in &mut self.layers {
            let norm = layer.norm1.forward(&x)?;
            let q = layer
                .q
                .forward(&norm)?
                .reshape((batch, seq, heads, head_dim))?
                .transpose(1, 2)?;
            let k = frozen_linear(&layer.k, &norm)?
                .reshape((batch, seq, kv_heads, head_dim))?
                .transpose(1, 2)?;
            let v = layer
                .v
                .forward(&norm)?
                .reshape((batch, seq, kv_heads, head_dim))?
                .transpose(1, 2)?;
            // Slow here means ordinary tensor operations with a correct backward graph.
            let q = candle_nn::rotary_emb::rope_slow(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_slow(&k, &cos, &sin)?;
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
            let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt())?;
            let scores = match &mask {
                Some(mask) => scores.broadcast_add(mask)?,
                None => scores,
            };
            let attention = candle_nn::ops::softmax(&scores, D::Minus1)?
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape((batch, seq, self.cfg.hidden_size))?;
            x = (&x + frozen_linear(&layer.o, &attention)?)?;
            let norm = layer.norm2.forward(&x)?;
            let ff = (candle_nn::ops::silu(&frozen_linear(&layer.gate, &norm)?)?
                * frozen_linear(&layer.up, &norm)?)?;
            x = (&x + frozen_linear(&layer.down, &ff)?)?;
        }
        Ok(self.norm.forward(&x)?)
    }

    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(frozen_linear(&self.lm_head, hidden)?)
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
}
impl ModelRuntime {
    pub fn load(base_dir: &Path, checkpoint: Option<&Path>) -> Result<Self> {
        let assets = validate_assets(base_dir)?;
        let cfg: QwenConfig = serde_json::from_slice(&fs::read(base_dir.join("config.json"))?)?;
        let weights =
            candle_core::safetensors::load(base_dir.join("model.safetensors"), &Device::Cpu)?
                .into_iter()
                .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
                .collect::<Result<HashMap<_, _>>>()?;
        let tokenizer = Tokenizer::from_file(base_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let (adapter, loaded, checkpoint_id) = match checkpoint {
            None => (None, None, MODEL_SHA256.to_owned()),
            Some(path) => {
                let manifest: crate::train::CheckpointManifest =
                    serde_json::from_slice(&fs::read(path.join("manifest.json"))?)?;
                crate::train::validate_checkpoint(path, &manifest, &assets)?;
                let loaded =
                    candle_core::safetensors::load(path.join("adapter.safetensors"), &Device::Cpu)?;
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
            .encode(chat_prefix(&request.prompt), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let mut tokens = enc.get_ids().to_vec();
        let input_tokens = tokens.len();
        ensure!(
            input_tokens + request.max_new_tokens <= 4096,
            "local inference context cap is 4096 tokens"
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
                let input = Tensor::from_slice(ids, (1, ids.len()), &Device::Cpu)?;
                let hidden = self.model.hidden(&input, offset, true)?;
                let logits = self
                    .model
                    .logits(&hidden.i((.., ids.len() - 1, ..))?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let next = sample(&logits, request.temperature, request.top_p, &mut rng)?;
                if next == 151645 || next == 151643 {
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

fn sample(logits: &[f32], temperature: f64, top_p: f64, rng: &mut StableRng) -> Result<u32> {
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

    /// A random unit model verifies the complete backward architecture. It is not
    /// a provider-trained checkpoint or evidence of assessment improvement.
    #[test]
    fn full_qwen_architecture_backpropagates_and_preserves_frozen_base() {
        let cfg = QwenConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
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
    }
}
