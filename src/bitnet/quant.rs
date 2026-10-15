//! Quantised forward values with identity straight-through derivatives.
use super::*;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Shape};

#[derive(Clone, Debug)]
enum Quantiser {
    Ternary(f32),
    Activation,
}
impl CustomOp1 for Quantiser {
    fn name(&self) -> &'static str {
        "bitnet-quantisation-aware-ste"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let CpuStorage::F32(values) = storage else {
            candle_core::bail!("BitNet STE requires F32")
        };
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg("BitNet quantiser input must be contiguous".into())
        })?;
        let values = &values[start..end];
        let mut output = Vec::with_capacity(values.len());
        match self {
            Self::Ternary(scale) => {
                for &value in values {
                    if !value.is_finite() {
                        candle_core::bail!("non-finite BitNet latent weight")
                    };
                    output.push((value / scale).round_ties_even().clamp(-1., 1.) * scale);
                }
            }
            Self::Activation => {
                let width = *layout.shape().dims().last().ok_or_else(|| {
                    candle_core::Error::Msg("BitNet activations have no feature dimension".into())
                })?;
                if width == 0 {
                    candle_core::bail!("empty BitNet activation width")
                };
                for row in values.chunks_exact(width) {
                    if row.iter().any(|x| !x.is_finite()) {
                        candle_core::bail!("non-finite BitNet activation")
                    };
                    let max = row.iter().map(|x| x.abs()).fold(1e-5_f32, f32::max);
                    let scale = 127. / max;
                    output.extend(
                        row.iter()
                            .map(|x| (x * scale).round_ties_even().clamp(-128., 127.) / scale),
                    );
                }
            }
        }
        Ok((CpuStorage::F32(output), layout.shape().clone()))
    }
    fn bwd(
        &self,
        _input: &Tensor,
        _result: &Tensor,
        grad: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        Ok(Some(grad.clone()))
    }
}
pub fn ternary_quantise_ste(weight: &Tensor, scale: f64) -> Result<Tensor> {
    ensure!(
        scale.is_finite() && scale > 0. && scale <= f32::MAX as f64,
        "invalid BitNet quantisation scale"
    );
    Ok(weight
        .contiguous()?
        .apply_op1(Quantiser::Ternary(scale as f32))?)
}
pub fn activation_quantise_ste(input: &Tensor) -> Result<Tensor> {
    Ok(input.contiguous()?.apply_op1(Quantiser::Activation)?)
}

#[derive(Clone, Debug)]
pub(super) struct FrozenMatmul;
impl CustomOp2 for FrozenMatmul {
    fn name(&self) -> &'static str {
        "bitnet-frozen-base-matmul"
    }
    fn cpu_fwd(
        &self,
        lhs: &CpuStorage,
        ll: &Layout,
        rhs: &CpuStorage,
        rl: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let (m, k) = ll.shape().dims2()?;
        let (rk, n) = rl.shape().dims2()?;
        if k != rk {
            candle_core::bail!("BitNet frozen matmul dimensions disagree")
        };
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ternary_and_activation_forwards_keep_nonzero_ste_gradients() -> Result<()> {
        let v = Var::from_vec(vec![-2_f32, -0.5, 0.2, 0.5, 2.], 5, &Device::Cpu)?;
        let q = ternary_quantise_ste(v.as_tensor(), 1.)?;
        assert_eq!(q.to_vec1::<f32>()?, vec![-1., 0., 0., 0., 1.]);
        let gradients = q.sum_all()?.backward()?;
        assert_eq!(
            gradients
                .get(v.as_tensor())
                .context("STE gradient absent")?
                .to_vec1::<f32>()?,
            vec![1.; 5]
        );
        let a = Var::from_vec(vec![0_f32, 0.25, 1., -1.], (1, 4), &Device::Cpu)?;
        let quant = activation_quantise_ste(a.as_tensor())?;
        assert_eq!(quant.to_vec2::<f32>()?[0][2], 1.);
        let gradients = quant.sum_all()?.backward()?;
        assert_eq!(
            gradients
                .get(a.as_tensor())
                .context("activation STE gradient absent")?
                .to_vec2::<f32>()?,
            vec![vec![1.; 4]]
        );
        assert!(ternary_quantise_ste(v.as_tensor(), 0.).is_err());
        Ok(())
    }
}
