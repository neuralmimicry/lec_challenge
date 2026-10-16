//! Streaming native GGUF/I2_S export from the official student and real QAT
//! adapters. Completed tensor fragments and assembly progress survive restart.
use super::*;
use crate::artefact::{self, Artefact};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

fn safe_output(root: &Path, path: &Path) -> Result<()> {
    use std::path::Component;
    ensure!(
        !path.as_os_str().is_empty()
            && path.components().all(|c| matches!(c, Component::Normal(_))),
        "export paths must remain relative to the workspace"
    );
    let mut current = root.to_owned();
    for component in path.components() {
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(m) => {
                ensure!(!m.file_type().is_symlink(), "symlink in GGUF export path");
                ensure!(
                    m.is_file() || m.is_dir(),
                    "special file in GGUF export path"
                );
                #[cfg(unix)]
                if m.is_file() {
                    use std::os::unix::fs::MetadataExt;
                    ensure!(m.nlink() == 1, "hard-linked GGUF export file");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
struct ExportLease(fs::File);
impl ExportLease {
    fn acquire(path: &Path) -> Result<Self> {
        let f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        // SAFETY: descriptor remains live and flock does not access Rust memory.
        ensure!(
            unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another exporter owns this GGUF destination"
        );
        Ok(Self(f))
    }
}
impl Drop for ExportLease {
    fn drop(&mut self) {
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Part {
    name: String,
    shape: Vec<usize>,
    ggml_type: u32,
    file: Artefact,
    changed_trits: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Assembly {
    header_sha256: String,
    completed_parts: usize,
    bytes: u64,
}
#[derive(Clone)]
enum Meta {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
}
fn string(w: &mut impl Write, s: &str) -> Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
fn meta(w: &mut impl Write, m: &Meta) -> Result<()> {
    let kind = match m {
        Meta::U32(_) => 4,
        Meta::F32(_) => 6,
        Meta::Bool(_) => 7,
        Meta::String(_) => 8,
        Meta::Strings(_) | Meta::I32s(_) => 9,
    };
    w.write_all(&(kind as u32).to_le_bytes())?;
    match m {
        Meta::U32(v) => w.write_all(&v.to_le_bytes())?,
        Meta::F32(v) => w.write_all(&v.to_le_bytes())?,
        Meta::Bool(v) => w.write_all(&[u8::from(*v)])?,
        Meta::String(v) => string(w, v)?,
        Meta::Strings(v) => {
            w.write_all(&8u32.to_le_bytes())?;
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for s in v {
                string(w, s)?;
            }
        }
        Meta::I32s(v) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for n in v {
                w.write_all(&n.to_le_bytes())?;
            }
        }
    }
    Ok(())
}
fn header(base: &Path, cfg: &BitNetConfig, parts: &[Part]) -> Result<Vec<u8>> {
    let tokeniser = tokenizers::Tokenizer::from_file(base.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let vocab = tokeniser.get_vocab(true);
    ensure!(
        vocab.len() == cfg.vocab_size,
        "GGUF tokeniser vocabulary count differs"
    );
    let mut tokens = vec![String::new(); cfg.vocab_size];
    for (token, id) in vocab {
        ensure!(
            (id as usize) < tokens.len() && tokens[id as usize].is_empty(),
            "duplicate/out-of-range original token ID"
        );
        tokens[id as usize] = token;
    }
    ensure!(
        tokens.iter().all(|t| !t.is_empty()),
        "original token ID mapping has gaps"
    );
    let raw: serde_json::Value = serde_json::from_slice(&fs::read(base.join("tokenizer.json"))?)?;
    let added = raw["added_tokens"]
        .as_array()
        .context("original added tokens missing")?;
    let mut types = vec![1_i32; cfg.vocab_size];
    for token in added {
        let id = token["id"].as_u64().context("added token ID missing")? as usize;
        ensure!(id < types.len(), "added token ID outside vocabulary");
        types[id] = if token["special"] == true { 3 } else { 4 };
    }
    let merges = raw["model"]["merges"]
        .as_array()
        .context("BPE merges absent")?
        .iter()
        .map(|m| {
            if let Some(s) = m.as_str() {
                Ok(s.to_owned())
            } else {
                let pair = m.as_array().context("malformed BPE merge")?;
                ensure!(pair.len() == 2, "malformed BPE merge pair");
                Ok(format!(
                    "{} {}",
                    pair[0].as_str().context("BPE merge token missing")?,
                    pair[1].as_str().context("BPE merge token missing")?
                ))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(base.join("tokenizer_config.json"))?)?;
    let arch = "bitnet-b1.58";
    let mut metadata = BTreeMap::from([
        ("general.architecture".into(), Meta::String(arch.into())),
        (
            "general.name".into(),
            Meta::String("Official BitNet 2B 4T native Rust QAT export".into()),
        ),
        ("general.file_type".into(), Meta::U32(40)),
        ("general.quantization_version".into(), Meta::U32(2)),
        ("general.alignment".into(), Meta::U32(32)),
        ("tokenizer.ggml.model".into(), Meta::String("gpt2".into())),
        (
            "tokenizer.ggml.pre".into(),
            Meta::String("llama-bpe".into()),
        ),
        ("tokenizer.ggml.tokens".into(), Meta::Strings(tokens)),
        ("tokenizer.ggml.token_type".into(), Meta::I32s(types)),
        ("tokenizer.ggml.merges".into(), Meta::Strings(merges)),
        ("tokenizer.ggml.bos_token_id".into(), Meta::U32(128000)),
        ("tokenizer.ggml.eos_token_id".into(), Meta::U32(128009)),
        ("tokenizer.ggml.add_bos_token".into(), Meta::Bool(true)),
        ("tokenizer.ggml.add_eos_token".into(), Meta::Bool(false)),
        (
            "tokenizer.chat_template".into(),
            Meta::String(
                config["chat_template"]
                    .as_str()
                    .context("original chat template absent")?
                    .into(),
            ),
        ),
    ]);
    for (name, value) in [
        ("context_length", cfg.max_position_embeddings),
        ("embedding_length", cfg.hidden_size),
        ("feed_forward_length", cfg.intermediate_size),
        ("block_count", cfg.num_hidden_layers),
        ("attention.head_count", cfg.num_attention_heads),
        ("attention.head_count_kv", cfg.num_key_value_heads),
        (
            "rope.dimension_count",
            cfg.hidden_size / cfg.num_attention_heads,
        ),
        ("vocab_size", cfg.vocab_size),
    ] {
        metadata.insert(format!("{arch}.{name}"), Meta::U32(u32::try_from(value)?));
    }
    metadata.insert(
        format!("{arch}.rope.freq_base"),
        Meta::F32(cfg.rope_theta as f32),
    );
    metadata.insert(
        format!("{arch}.attention.layer_norm_rms_epsilon"),
        Meta::F32(cfg.rms_norm_eps as f32),
    );
    let mut bytes = Vec::new();
    bytes.write_all(b"GGUF")?;
    bytes.write_all(&3u32.to_le_bytes())?;
    bytes.write_all(&(parts.len() as u64).to_le_bytes())?;
    bytes.write_all(&(metadata.len() as u64).to_le_bytes())?;
    for (name, value) in metadata {
        string(&mut bytes, &name)?;
        meta(&mut bytes, &value)?;
    }
    let mut offset = 0_u64;
    for part in parts {
        string(&mut bytes, &part.name)?;
        bytes.write_all(&(part.shape.len() as u32).to_le_bytes())?;
        for &d in part.shape.iter().rev() {
            bytes.write_all(&(d as u64).to_le_bytes())?;
        }
        bytes.write_all(&part.ggml_type.to_le_bytes())?;
        bytes.write_all(&offset.to_le_bytes())?;
        offset = aligned(
            offset
                .checked_add(part.file.bytes)
                .context("GGUF offset overflow")?,
        );
    }
    bytes.resize(aligned(bytes.len() as u64) as usize, 0);
    Ok(bytes)
}
fn aligned(n: u64) -> u64 {
    (n + 31) & !31
}
fn same_binding(a: &serde_json::Value, b: &serde_json::Value) -> Result<bool> {
    // The versioned codec is the semantic contract. Preserve the original
    // implementation hash as provenance, while allowing verified fragments to
    // survive comment/refactoring/test-only changes in the surrounding module.
    for value in [a, b] {
        let hash = value["implementation_sha256"]
            .as_str()
            .context("export implementation identity absent")?;
        ensure!(
            hash.len() == 64 && hash.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid export implementation digest"
        );
    }
    let mut left = a.clone();
    let mut right = b.clone();
    left.as_object_mut()
        .context("export binding is not an object")?
        .remove("implementation_sha256");
    right
        .as_object_mut()
        .context("export binding is not an object")?
        .remove("implementation_sha256");
    Ok(left == right)
}
fn mapped(name: &str) -> Result<String> {
    if name == "model.embed_tokens.weight" {
        return Ok("token_embd.weight".into());
    }
    if name == "model.norm.weight" {
        return Ok("output_norm.weight".into());
    }
    let rest = name
        .strip_prefix("model.layers.")
        .context("unrecognised official BitNet tensor name")?;
    let (layer, name) = rest
        .split_once('.')
        .context("invalid BitNet layer tensor")?;
    let _: usize = layer.parse()?;
    let suffix = match name {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.attn_sub_norm.weight" => "attn_sub_norm.weight",
        "mlp.ffn_sub_norm.weight" => "ffn_sub_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => anyhow::bail!("unrecognised official BitNet tensor {name}"),
    };
    Ok(format!("blk.{layer}.{suffix}"))
}
fn read_at(file: &mut fs::File, data: u64, entry: &TensorHeader) -> Result<Vec<u8>> {
    let size = entry.data_offsets[1]
        .checked_sub(entry.data_offsets[0])
        .context("invalid safetensors range")?;
    ensure!(
        size <= 32 << 20,
        "packed/scalar tensor exceeds bounded conversion buffer"
    );
    file.seek(SeekFrom::Start(data + entry.data_offsets[0]))?;
    let mut bytes = vec![0; size as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}
fn hf_i2(packed: &[u8], rows: usize, columns: usize, scale: f32) -> Result<Vec<u8>> {
    ensure!(
        rows.is_multiple_of(4) && columns.is_multiple_of(128) && packed.len() == rows / 4 * columns,
        "invalid official matrix shape for I2_S"
    );
    let mut output = vec![0; rows * columns / 4 + 32];
    let packed_rows = rows / 4;
    for row in 0..rows {
        let shift = (row / packed_rows) * 2;
        let r = row % packed_rows;
        for block in 0..columns / 128 {
            for column in 0..32 {
                let mut byte = 0;
                for group in 0..4 {
                    let code =
                        (packed[r * columns + block * 128 + group * 32 + column] >> shift) & 3;
                    ensure!(code < 3, "reserved code3 cannot enter native I2_S export");
                    byte |= code << (6 - 2 * group);
                }
                output[row * columns / 4 + block * 32 + column] = byte;
            }
        }
    }
    output[rows * columns / 4..rows * columns / 4 + 4].copy_from_slice(&scale.to_le_bytes());
    Ok(output)
}

/// Exporting alone does not prove training. A supplied adapter must pass the
/// separate experimental checkpoint verifier and have changed effective trits.
pub fn export(base: &Path, checkpoint: Option<&Path>, output: &Path) -> Result<serde_json::Value> {
    let root = std::env::current_dir()?.canonicalize()?;
    safe_output(&root, output)?;
    let _source_lease = SharedAssetLease::acquire(base)?;
    let assets = crate::hub::validate_bitnet_student_assets(base)?;
    ensure!(
        output.extension().is_some_and(|e| e == "gguf"),
        "BitNet export requires a GGUF destination"
    );
    let parent = output.parent().context("GGUF output parent absent")?;
    fs::create_dir_all(parent)?;
    let lock = output.with_extension("gguf.export.lock");
    safe_output(&root, &lock)?;
    let _lease = ExportLease::acquire(&lock)?;
    let adapter = if let Some(checkpoint) = checkpoint {
        let m = crate::distillation::checkpoint::validate_checkpoint(checkpoint, &assets)?;
        let _ = crate::teacher_evidence::load_verified(&m.evidence_bundle.path)?;
        ensure!(
            m.spec.student_kind == crate::distillation::DistillationStudent::TernaryBitnet,
            "export adapter is not BitNet"
        );
        Some((
            Artefact::capture(&root, checkpoint.join("adapter.safetensors"))?,
            m,
        ))
    } else {
        None
    };
    let mut binding = serde_json::json!({"source":assets,"adapter":adapter.as_ref().map(|(a,_)|a),"codec":"native_hf_axis_to_i2s_v1","implementation_sha256":artefact::hash(include_bytes!("export.rs"))});
    let evidence_path = output.with_extension("export.json");
    safe_output(&root, &evidence_path)?;
    if evidence_path.exists() {
        let old: serde_json::Value = serde_json::from_slice(&fs::read(&evidence_path)?)?;
        ensure!(
            same_binding(&old["binding"], &binding)?,
            "existing GGUF export has different immutable inputs"
        );
        let file: Artefact = serde_json::from_value(old["gguf"].clone())?;
        ensure!(
            file.verify(&root)? == output.canonicalize()?,
            "GGUF receipt names another destination"
        );
        return Ok(old);
    }
    let parts_directory = output.with_extension("gguf.parts");
    safe_output(&root, &parts_directory)?;
    fs::create_dir_all(&parts_directory)?;
    let binding_path = parts_directory.join("binding.json");
    safe_output(&root, &binding_path)?;
    ensure!(
        !output.exists() || binding_path.exists(),
        "unclaimed GGUF output exists without saved export authority"
    );
    if binding_path.exists() {
        let saved = serde_json::from_slice::<serde_json::Value>(&fs::read(&binding_path)?)?;
        ensure!(
            same_binding(&saved, &binding)?,
            "saved tensor fragments belong to different exporter inputs"
        );
        binding = saved;
    } else {
        artefact::write_json(&binding_path, &binding)?;
    }
    let cfg: BitNetConfig = serde_json::from_slice(&fs::read(base.join("config.json"))?)?;
    cfg.validate()?;
    let mut source = fs::File::open(base.join("model.safetensors"))?;
    let mut length = [0; 8];
    source.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    ensure!(length <= 8 << 20, "safetensors header oversized");
    let mut bytes = vec![0; length as usize];
    source.read_exact(&mut bytes)?;
    let entries: serde_json::Value = serde_json::from_slice(&bytes)?;
    let data = length + 8;
    let mut tensors = BTreeMap::new();
    for (name, value) in entries
        .as_object()
        .context("safetensors header is not an object")?
    {
        if name != "__metadata__" {
            tensors.insert(
                name.clone(),
                serde_json::from_value::<TensorHeader>(value.clone())?,
            );
        }
    }
    let adapter_weights = adapter
        .as_ref()
        .map(|(a, _)| candle_core::safetensors::load(&a.path, &Device::Cpu))
        .transpose()?;
    let mut parts = Vec::new();
    let mut changed = 0_u64;
    let retained_tensors = tensors
        .keys()
        .filter(|n| !n.ends_with("weight_scale"))
        .count();
    for (name, entry) in &tensors {
        if name.ends_with("weight_scale") {
            continue;
        }
        let mapped = mapped(name)?;
        let receipt = parts_directory.join(format!("{mapped}.json"));
        safe_output(&root, &receipt)?;
        if receipt.exists() {
            let part: Part = serde_json::from_slice(&fs::read(&receipt)?)?;
            ensure!(part.name == mapped, "fragment mapping changed");
            part.file.verify(&root)?;
            changed = changed
                .checked_add(part.changed_trits)
                .context("changed trit count overflow")?;
            parts.push(part);
            continue;
        }
        let changed_before = changed;
        let path = parts_directory.join(format!("{mapped}.bin"));
        let temporary = path.with_extension("partial");
        safe_output(&root, &path)?;
        safe_output(&root, &temporary)?;
        let mut writer = BufWriter::new(fs::File::create(&temporary)?);
        let (shape, ggml_type) = if entry.dtype == "U8" {
            ensure!(entry.shape.len() == 2, "packed BitLinear rank differs");
            let rows = entry.shape[0] * 4;
            let columns = entry.shape[1];
            let prefix = name
                .strip_suffix(".weight")
                .context("packed weight suffix absent")?;
            let scale_entry = tensors
                .get(&format!("{prefix}.weight_scale"))
                .context("original weight scale absent")?;
            let scale = read_at(&mut source, data, scale_entry)?;
            ensure!(
                scale_entry.dtype == "BF16" && scale.len() == 2,
                "original scale type/shape differs"
            );
            let scale = f32::from_bits(u32::from(u16::from_le_bytes([scale[0], scale[1]])) << 16);
            ensure!(
                scale.is_finite() && scale > 0.,
                "non-finite original export scale"
            );
            let packed = read_at(&mut source, data, entry)?;
            let a = adapter_weights
                .as_ref()
                .and_then(|w| w.get(&format!("{prefix}.qat_a")));
            let b = adapter_weights
                .as_ref()
                .and_then(|w| w.get(&format!("{prefix}.qat_b")));
            let encoded = match (a, b) {
                (Some(a), Some(b)) => {
                    let original = codec::unpack_rows(&packed, rows, columns)?
                        .into_iter()
                        .map(|x| x * scale)
                        .collect::<Vec<_>>();
                    let base_tensor =
                        Tensor::from_vec(original.clone(), (rows, columns), &Device::Cpu)?;
                    let manifest = &adapter.as_ref().context("adapter receipt absent")?.1;
                    let latent = (&base_tensor
                        + (b.matmul(a)?
                            * (manifest.adapter.alpha / manifest.adapter.rank as f64))?)?;
                    let effective = ternary_quantise_ste(&latent, f64::from(scale))?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    changed += original
                        .iter()
                        .zip(&effective)
                        .filter(|(a, b)| a != b)
                        .count() as u64;
                    codec::pack_i2_s(&effective, scale)?
                }
                (None, None) => hf_i2(&packed, rows, columns, scale)?,
                _ => anyhow::bail!("incomplete latent QAT adapter projection"),
            };
            writer.write_all(&encoded)?;
            (vec![rows, columns], 36)
        } else {
            ensure!(
                entry.dtype == "BF16",
                "unsupported original unquantised tensor type"
            );
            source.seek(SeekFrom::Start(data + entry.data_offsets[0]))?;
            let mut remaining = entry.data_offsets[1] - entry.data_offsets[0];
            ensure!(
                remaining.is_multiple_of(2),
                "invalid BF16 tensor byte count"
            );
            let mut buffer = vec![0; 65536];
            while remaining > 0 {
                let n = usize::try_from(remaining.min(buffer.len() as u64))?;
                source.read_exact(&mut buffer[..n])?;
                for pair in buffer[..n].as_chunks::<2>().0 {
                    writer.write_all(
                        &(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16).to_le_bytes(),
                    )?;
                }
                remaining -= n as u64;
            }
            (entry.shape.clone(), 0)
        };
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, &path)?;
        let part = Part {
            name: mapped,
            shape,
            ggml_type,
            file: Artefact::capture(&root, &path)?,
            changed_trits: changed - changed_before,
        };
        artefact::write_json(&receipt, &part)?;
        parts.push(part);
        eprintln!(
            "BitNet GGUF export verified tensor fragment {}/{}: {}",
            parts.len(),
            retained_tensors,
            name
        );
    }
    // An adapted export must demonstrate a discrete model change. Count is
    // recomputed on a fresh adapter export; resumed parts carry this in a receipt.
    let change_path = parts_directory.join("effective-change.json");
    safe_output(&root, &change_path)?;
    if adapter.is_some() {
        if change_path.exists() {
            let saved=serde_json::from_slice::<serde_json::Value>(&fs::read(&change_path)?)?["changed_trits"].as_u64().context("changed trit count absent")?;
            ensure!(
                saved == changed && changed > 0,
                "saved effective change count differs from retained tensor fragments"
            );
        } else {
            ensure!(
                changed > 0,
                "latent adapter changed but no effective ternary weights changed"
            );
            artefact::write_json(&change_path, &serde_json::json!({"changed_trits":changed}))?;
        }
    }
    let header = header(base, &cfg, &parts)?;
    let temporary = output.with_extension("gguf.partial");
    let progress_path = parts_directory.join("assembly.json");
    safe_output(&root, &temporary)?;
    safe_output(&root, &progress_path)?;
    let hash = artefact::hash(&header);
    let mut progress = if progress_path.exists() {
        serde_json::from_slice::<Assembly>(&fs::read(&progress_path)?)?
    } else {
        Assembly {
            header_sha256: hash.clone(),
            completed_parts: 0,
            bytes: header.len() as u64,
        }
    };
    ensure!(
        progress.header_sha256 == hash && progress.completed_parts <= parts.len(),
        "saved GGUF assembly header/progress changed"
    );
    if output.exists() {
        ensure!(
            !temporary.exists(),
            "both unpublished final and partial GGUF files exist; refusing ambiguous recovery"
        );
        fs::rename(output, &temporary)?;
    }
    let mut assembled = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&temporary)?;
    if assembled.metadata()?.len() == 0 {
        assembled.write_all(&header)?;
        assembled.sync_all()?;
        artefact::write_json(&progress_path, &progress)?;
    } else {
        let mut saved = vec![0; header.len()];
        assembled.read_exact(&mut saved)?;
        ensure!(saved == header, "GGUF assembly prefix header changed");
        for part in parts.iter().take(progress.completed_parts) {
            let start = assembled.stream_position()?;
            let mut bytes = Read::by_ref(&mut assembled).take(part.file.bytes);
            use sha2::{Digest, Sha256};
            let mut digest = Sha256::new();
            let mut buffer = vec![0; 65536];
            loop {
                let n = bytes.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                digest.update(&buffer[..n]);
            }
            ensure!(
                hex::encode(digest.finalize()) == part.file.sha256,
                "saved GGUF tensor prefix changed"
            );
            assembled.seek(SeekFrom::Start(aligned(start + part.file.bytes)))?;
        }
        ensure!(
            assembled.stream_position()? == progress.bytes,
            "GGUF assembly progress boundary differs"
        );
        assembled.set_len(progress.bytes)?;
        assembled.seek(SeekFrom::Start(progress.bytes))?;
    }
    for part in parts.iter().skip(progress.completed_parts) {
        let start = assembled.stream_position()?;
        let mut source = fs::File::open(part.file.verify(&root)?)?;
        std::io::copy(&mut source, &mut assembled)?;
        let padding = aligned(part.file.bytes) - part.file.bytes;
        assembled.write_all(&[0; 32][..padding as usize])?;
        assembled.sync_all()?;
        progress.completed_parts += 1;
        progress.bytes = start + aligned(part.file.bytes);
        artefact::write_json(&progress_path, &progress)?;
    }
    assembled.sync_all()?;
    drop(assembled);
    fs::rename(&temporary, output)?;
    fs::File::open(parent)?.sync_all()?;
    let receipt = serde_json::json!({"schema_version":1,"evidence_kind":"native_official_bitnet_qat_gguf_export","binding":binding,"gguf":Artefact::capture(&root,output)?,"tensors":parts.len(),"changed_effective_trits":changed,"requires_engine_patch":"official ReLU² BitNet graph; preserve local patch provenance","production_eligible":false,"limitations":["Export is separate from training and quality evidence; baseline export has no adaptation claim.","Requires compatible pinned BitNet.cpp I2_S engine and an actual generation/parity smoke test."]});
    artefact::write_json(&evidence_path, &receipt)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_hf_to_i2s_matches_independent_ternary_codec() -> Result<()> {
        let values = (0..4 * 256)
            .map(|i| (((i * 7 + i / 31) % 3) as f32 - 1.) * 0.125)
            .collect::<Vec<_>>();
        let hf = codec::pack_rows(&values, 4, 256, 0.125)?;
        assert_eq!(
            hf_i2(&hf, 4, 256, 0.125)?,
            codec::pack_i2_s(&values, 0.125)?
        );
        let mut corrupt = hf;
        corrupt[0] = 255;
        assert!(hf_i2(&corrupt, 4, 256, 0.125).is_err());
        Ok(())
    }
    #[test]
    fn source_mapping_and_export_path_guards_fail_closed() -> Result<()> {
        assert_eq!(
            mapped("model.layers.29.self_attn.q_proj.weight")?,
            "blk.29.attn_q.weight"
        );
        assert!(mapped("unreviewed.code").is_err());
        let root = tempfile::tempdir()?;
        assert!(safe_output(root.path(), Path::new("../escape.gguf")).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.path(), root.path().join("link"))?;
            assert!(safe_output(root.path(), Path::new("link/model.gguf")).is_err());
        }
        Ok(())
    }
}
