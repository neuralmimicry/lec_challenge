//! CPU/F32 BitNet RMS arithmetic matching the reviewed native inference path.
//!
//! Forward squares in F32, accumulates the rounded products sequentially in
//! F64, then uses the native F32 mean/epsilon/square-root/reciprocal/product.
//! Backward implements the first-order smooth RMS vector-Jacobian product at
//! that rounded reciprocal; floating-point rounding itself is not differentiated.
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

#[derive(Clone, Copy, Debug)]
struct NativeRms {
    epsilon: f32,
}

fn reciprocal(row: &[f32], epsilon: f32) -> Result<f32> {
    let mut sum = 0_f64;
    for &value in row {
        let square = value * value;
        if !value.is_finite() || !square.is_finite() {
            candle_core::bail!("BitNet RMS received non-finite input or squared input");
        }
        // Retain the F32 multiplication before promotion and the sequential
        // F64 reduction. An F64 square or reassociated sum is another protocol.
        sum += square as f64;
    }
    let mean = (sum / row.len() as f64) as f32;
    let scale = 1_f32 / (mean + epsilon).sqrt();
    if !scale.is_finite() || scale <= 0. {
        candle_core::bail!("BitNet RMS produced an invalid reciprocal");
    }
    Ok(scale)
}

impl CustomOp1 for NativeRms {
    fn name(&self) -> &'static str {
        "bitnet-native-f32-products-f64-sum-rms"
    }
    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("BitNet RMS requires explicit CPU/F32 storage");
        };
        let width = *layout
            .shape()
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("BitNet RMS has no feature axis".into()))?;
        if width == 0 {
            candle_core::bail!("BitNet RMS cannot normalise an empty feature row");
        }
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("BitNet RMS requires contiguous internal storage".into())
        })?;
        let values = &values[start..end];
        if !values.len().is_multiple_of(width) {
            candle_core::bail!("BitNet RMS row shape is inconsistent");
        }
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks_exact(width) {
            let scale = reciprocal(row, self.epsilon)?;
            output.extend(row.iter().map(|&value| value * scale));
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
    fn bwd(&self, input: &Tensor, _result: &Tensor, grad: &Tensor) -> Result<Option<Tensor>> {
        if input.dims() != grad.dims()
            || input.dtype() != DType::F32
            || grad.dtype() != DType::F32
            || !input.device().is_cpu()
            || !grad.device().is_cpu()
        {
            candle_core::bail!("BitNet RMS backward requires matching CPU/F32 tensors");
        }
        let width = *input.dims().last().ok_or_else(|| {
            candle_core::Error::Msg("BitNet RMS backward has no feature axis".into())
        })?;
        if width == 0 {
            candle_core::bail!("BitNet RMS backward has an empty feature axis");
        }
        let values = input.flatten_all()?.to_vec1::<f32>()?;
        let upstream = grad.flatten_all()?.to_vec1::<f32>()?;
        let mut derivative = Vec::with_capacity(values.len());
        for (row, upstream) in values.chunks_exact(width).zip(upstream.chunks_exact(width)) {
            if upstream.iter().any(|value| !value.is_finite()) {
                candle_core::bail!("non-finite BitNet RMS upstream gradient");
            }
            let scale = reciprocal(row, self.epsilon)? as f64;
            let dot = row
                .iter()
                .zip(upstream)
                .map(|(&x, &g)| x as f64 * g as f64)
                .sum::<f64>();
            let correction = scale * scale * scale * dot / width as f64;
            for (&value, &upstream) in row.iter().zip(upstream) {
                let gradient = (scale * upstream as f64 - correction * value as f64) as f32;
                if !gradient.is_finite() {
                    candle_core::bail!("non-finite BitNet RMS derivative");
                }
                derivative.push(gradient);
            }
        }
        // This explicit tensor supports first-order adaptation. Higher-order
        // differentiation through this backward implementation is unsupported.
        Ok(Some(Tensor::from_vec(
            derivative,
            input.shape().clone(),
            input.device(),
        )?))
    }
}

pub(super) fn forward(input: &Tensor, epsilon: f64) -> Result<Tensor> {
    let epsilon_f32 = epsilon as f32;
    if !epsilon.is_finite() || epsilon <= 0. || !epsilon_f32.is_finite() || epsilon_f32 <= 0. {
        candle_core::bail!("BitNet RMS requires a positive finite F32 epsilon");
    }
    if input.dtype() != DType::F32 || !input.device().is_cpu() {
        candle_core::bail!("BitNet RMS requires explicit CPU/F32 input");
    }
    // Candle retains the contiguous-copy operation in the graph, including a
    // non-contiguous original view, so upstream gradients preserve their axes.
    input.contiguous()?.apply_op1(NativeRms {
        epsilon: epsilon_f32,
    })
}

pub(super) fn weighted(input: &Tensor, weight: &Tensor, epsilon: f64) -> Result<Tensor> {
    let width = *input
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("BitNet weighted RMS has no feature axis".into()))?;
    if weight.dtype() != DType::F32 || !weight.device().is_cpu() || weight.dims() != [width] {
        candle_core::bail!("BitNet RMS weight must be one CPU/F32 feature vector");
    }
    forward(input, epsilon)?.broadcast_mul(weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};
    fn objective(input: &[f32], upstream: &[f32], rows: usize, width: usize) -> Result<f64> {
        Ok(forward(
            &Tensor::from_vec(input.to_vec(), (rows, width), &Device::Cpu)?,
            1e-5,
        )?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .zip(upstream)
        .map(|(&y, &g)| y as f64 * g as f64)
        .sum())
    }
    #[test]
    fn full_vjp_matches_central_differences_for_asymmetric_rows() -> Result<()> {
        let input = vec![-1.3, 0.2, 0.7, 2.1, -0.4, 0.8, -1.7, 0.3, -0.6, 1.1];
        let upstream = vec![0.7, -0.2, 1.3, 0.4, -0.9, -0.5, 0.8, 0.2, -1.2, 0.6];
        let variable = Var::from_vec(input.clone(), (2, 5), &Device::Cpu)?;
        let loss = forward(variable.as_tensor(), 1e-5)?
            .mul(&Tensor::from_vec(upstream.clone(), (2, 5), &Device::Cpu)?)?
            .sum_all()?;
        let gradient = loss
            .backward()?
            .get(variable.as_tensor())
            .ok_or_else(|| candle_core::Error::Msg("RMS gradient absent".into()))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for step in [2e-3_f32, 1e-3, 5e-4] {
            for index in 0..input.len() {
                let mut plus = input.clone();
                let mut minus = input.clone();
                plus[index] += step;
                minus[index] -= step;
                let estimate = (objective(&plus, &upstream, 2, 5)?
                    - objective(&minus, &upstream, 2, 5)?)
                    / (plus[index] as f64 - minus[index] as f64);
                assert!(
                    (estimate - gradient[index] as f64).abs()
                        <= 1e-3 + 1e-3 * (gradient[index] as f64).abs()
                );
            }
        }
        Ok(())
    }
    #[test]
    fn non_contiguous_view_and_weight_keep_their_gradients() -> Result<()> {
        let input = Var::from_vec(
            vec![0.2_f32, -0.7, 1.1, 0.4, -1.3, 0.8],
            (2, 3),
            &Device::Cpu,
        )?;
        let weight = Var::from_vec(vec![0.8_f32, -1.1], 2, &Device::Cpu)?;
        let upstream = Tensor::from_vec(
            vec![0.3_f32, -0.8, 0.4, 0.9, -0.5, 1.2],
            (3, 2),
            &Device::Cpu,
        )?;
        let loss = weighted(&input.t()?, weight.as_tensor(), 1e-5)?
            .mul(&upstream)?
            .sum_all()?;
        let gradient = loss.backward()?;
        for variable in [&input, &weight] {
            let values = gradient
                .get(variable.as_tensor())
                .ok_or_else(|| candle_core::Error::Msg("weighted RMS gradient absent".into()))?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(values.iter().all(|x| x.is_finite()) && values.iter().any(|x| *x != 0.));
        }
        Ok(())
    }
    #[test]
    fn invalid_dtype_epsilon_and_overflow_fail_closed() -> Result<()> {
        let input = Tensor::from_vec(vec![0.2_f32, 1.], (1, 2), &Device::Cpu)?;
        for epsilon in [0., -1., f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
            assert!(forward(&input, epsilon).is_err());
        }
        let wrong = Tensor::from_vec(vec![0.2_f64, 1.], (1, 2), &Device::Cpu)?;
        assert!(forward(&wrong, 1e-5).is_err());
        let overflow = Tensor::from_vec(vec![1e30_f32, 1.], (1, 2), &Device::Cpu)?;
        assert!(forward(&overflow, 1e-5).is_err());
        let wrong_weight = Tensor::ones(3, DType::F32, &Device::Cpu)?;
        assert!(weighted(&input, &wrong_weight, 1e-5).is_err());
        Ok(())
    }
}
