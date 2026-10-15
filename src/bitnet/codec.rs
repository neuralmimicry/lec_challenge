//! Official Hugging Face first-axis two-bit packing, not GGUF I2_S layout.
use super::*;

pub fn pack_rows(values: &[f32], rows: usize, columns: usize, scale: f32) -> Result<Vec<u8>> {
    ensure!(
        rows > 0
            && columns > 0
            && rows.is_multiple_of(4)
            && values.len() == rows * columns
            && scale.is_finite()
            && scale > 0.,
        "invalid ternary packing dimensions or scale"
    );
    let packed_rows = rows / 4;
    let mut packed = vec![0; packed_rows * columns];
    for group in 0..4 {
        for row in 0..packed_rows {
            for column in 0..columns {
                let value = values[(group * packed_rows + row) * columns + column] / scale;
                let trit = value.round_ties_even();
                ensure!(
                    value.is_finite() && (-1. ..=1.).contains(&trit) && (value - trit).abs() < 1e-5,
                    "effective weight is not on the declared ternary lattice"
                );
                packed[row * columns + column] |= ((trit as i8 + 1) as u8) << (group * 2);
            }
        }
    }
    Ok(packed)
}

/// Native I2_S is a different codec: each 128-value block is interleaved into
/// four 32-value groups, followed by a scalar scale and zero padding.
pub fn pack_i2_s(values: &[f32], scale: f32) -> Result<Vec<u8>> {
    ensure!(
        !values.is_empty() && values.len().is_multiple_of(128) && scale.is_finite() && scale > 0.,
        "I2_S requires complete 128-value blocks and a finite positive scale"
    );
    let mut output = vec![0; values.len() / 4 + 32];
    for (block, chunk) in values.as_chunks::<128>().0.iter().enumerate() {
        for column in 0..32 {
            let mut byte = 0;
            for group in 0..4 {
                let value = chunk[group * 32 + column] / scale;
                let trit = value.round_ties_even();
                ensure!(
                    value.is_finite() && (-1. ..=1.).contains(&trit) && (value - trit).abs() < 1e-5,
                    "I2_S export encountered a non-ternary effective weight"
                );
                byte |= ((trit as i8 + 1) as u8) << (6 - 2 * group);
            }
            output[block * 32 + column] = byte;
        }
    }
    output[values.len() / 4..values.len() / 4 + 4].copy_from_slice(&scale.to_le_bytes());
    Ok(output)
}

pub fn unpack_rows(packed: &[u8], rows: usize, columns: usize) -> Result<Vec<f32>> {
    ensure!(
        rows > 0 && columns > 0 && rows.is_multiple_of(4) && packed.len() == rows / 4 * columns,
        "invalid official packed BitLinear shape"
    );
    let mut output = vec![0.; rows * columns];
    let packed_rows = rows / 4;
    for group in 0..4 {
        for row in 0..packed_rows {
            for col in 0..columns {
                let value = (packed[row * columns + col] >> (group * 2)) & 3;
                ensure!(
                    value <= 2,
                    "invalid reserved ternary code3 in official BitNet weights"
                );
                output[(group * packed_rows + row) * columns + col] = value as f32 - 1.;
            }
        }
    }
    Ok(output)
}

pub(super) fn decode_official(
    mut weights: HashMap<String, Tensor>,
    cfg: &BitNetConfig,
) -> Result<(HashMap<String, Tensor>, BTreeMap<String, f64>)> {
    let mut decoded = HashMap::new();
    let mut scales = BTreeMap::new();
    for (name, tensor) in &weights {
        if !name.ends_with("weight_scale") && tensor.dtype() != DType::U8 {
            decoded.insert(name.clone(), tensor.to_dtype(DType::F32)?);
        }
    }
    let kv = cfg.hidden_size / cfg.num_attention_heads * cfg.num_key_value_heads;
    for layer in 0..cfg.num_hidden_layers {
        for (name, out, input) in [
            ("self_attn.q_proj", cfg.hidden_size, cfg.hidden_size),
            ("self_attn.k_proj", kv, cfg.hidden_size),
            ("self_attn.v_proj", kv, cfg.hidden_size),
            ("self_attn.o_proj", cfg.hidden_size, cfg.hidden_size),
            ("mlp.gate_proj", cfg.intermediate_size, cfg.hidden_size),
            ("mlp.up_proj", cfg.intermediate_size, cfg.hidden_size),
            ("mlp.down_proj", cfg.hidden_size, cfg.intermediate_size),
        ] {
            let prefix = format!("model.layers.{layer}.{name}");
            let key = format!("{prefix}.weight");
            let packed = weights
                .remove(&key)
                .with_context(|| format!("missing packed BitNet tensor {key}"))?;
            ensure!(
                packed.dtype() == DType::U8 && packed.dims() == [out / 4, input],
                "official BitNet packed dimensions differ"
            );
            let scale = weights
                .remove(&format!("{prefix}.weight_scale"))
                .context("official BitNet scale absent")?;
            ensure!(scale.elem_count() == 1, "BitNet scale must be scalar");
            let scale = f64::from(
                scale
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?[0],
            );
            ensure!(
                scale.is_finite() && scale > 0.,
                "invalid original BitNet scale"
            );
            let values = unpack_rows(&packed.flatten_all()?.to_vec1::<u8>()?, out, input)?;
            let matrix = Tensor::from_vec(values, (out, input), &Device::Cpu)?.affine(scale, 0.)?;
            decoded.insert(key, matrix);
            scales.insert(prefix, scale);
        }
        eprintln!(
            "BitNet decoded official ternary layer {}/{} on CPU",
            layer + 1,
            cfg.num_hidden_layers
        );
    }
    ensure!(
        weights.values().all(|t| t.dtype() != DType::U8),
        "unrecognised packed BitNet tensors"
    );
    Ok((decoded, scales))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_axis_packing_and_reserved_codes_are_checked() -> Result<()> {
        let input = [0b10_01_00_10, 0b00_10_01_00];
        assert_eq!(
            unpack_rows(&input, 4, 2)?,
            vec![1., -1., -1., 0., 0., 1., 1., -1.]
        );
        assert!(unpack_rows(&[255], 4, 1).is_err());
        assert!(unpack_rows(&input, 3, 2).is_err());
        Ok(())
    }
}
