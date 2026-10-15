//! Differentiable distillation objectives. Teacher and reference tensors are
//! detached explicitly; only the student receives optimisation gradients.
use anyhow::{Result, ensure};
use candle_core::{D, IndexOp, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaddingMassDiagnostics {
    pub semantic_tokens: usize,
    pub student_padding_tokens: usize,
    pub teacher_padding_tokens: usize,
    pub student_residual_probability_mean: f64,
    pub teacher_residual_probability_mean: f64,
    pub temperature: f64,
    pub discarded_probability_mass: f64,
    pub semantics: String,
}

/// Padded output heads can differ despite identical original token-ID maps.
/// Retain each defined semantic ID and coarsen every remaining output into one
/// residual category. Both distributions first normalise over their FULL heads.
pub fn aligned_softened_kl(
    student: &Tensor,
    teacher: &Tensor,
    semantic_ids: &[u32],
    temperature: f64,
) -> Result<(Tensor, PaddingMassDiagnostics)> {
    let (rows, sw) = student.dims2()?;
    let (tr, tw) = teacher.dims2()?;
    ensure!(
        rows > 0 && rows == tr && !semantic_ids.is_empty(),
        "aligned KL requires nonempty matched rows and semantic IDs"
    );
    ensure!(
        temperature.is_finite() && (0.01..=100.).contains(&temperature),
        "invalid distillation temperature"
    );
    ensure!(
        semantic_ids.windows(2).all(|w| w[0] < w[1])
            && semantic_ids
                .iter()
                .all(|&i| (i as usize) < sw && (i as usize) < tw),
        "semantic token IDs must be unique, sorted and present in both heads"
    );
    let student_logp = candle_nn::ops::log_softmax(&(student / temperature)?, 1)?;
    let teacher_logp = candle_nn::ops::log_softmax(&(teacher.detach() / temperature)?, 1)?;
    let project = |logp: &Tensor, width: usize| -> Result<(Tensor, Option<Tensor>, usize)> {
        let selected = logp.index_select(
            &Tensor::from_slice(semantic_ids, semantic_ids.len(), logp.device())?,
            1,
        )?;
        let residual = (0..width as u32)
            .filter(|i| semantic_ids.binary_search(i).is_err())
            .collect::<Vec<_>>();
        if residual.is_empty() {
            Ok((selected, None, 0))
        } else {
            let tail = logp
                .index_select(
                    &Tensor::from_slice(&residual, residual.len(), logp.device())?,
                    1,
                )?
                .log_sum_exp(1)?
                .unsqueeze(1)?;
            Ok((
                Tensor::cat(&[&selected, &tail], 1)?,
                Some(tail),
                residual.len(),
            ))
        }
    };
    let (student, student_tail, sp) = project(&student_logp, sw)?;
    let (teacher, teacher_tail, tp) = project(&teacher_logp, tw)?;
    ensure!(
        student_tail.is_some() == teacher_tail.is_some(),
        "one model has no residual category for the other's padding mass"
    );
    let mass = |tail: &Option<Tensor>| -> Result<f64> {
        Ok(match tail {
            None => 0.,
            Some(t) => t.detach().exp()?.mean_all()?.to_scalar::<f32>()? as f64,
        })
    };
    let diagnostics=PaddingMassDiagnostics{semantic_tokens:semantic_ids.len(),student_padding_tokens:sp,teacher_padding_tokens:tp,student_residual_probability_mean:mass(&student_tail)?,teacher_residual_probability_mean:mass(&teacher_tail)?,temperature,discarded_probability_mass:0.,semantics:"Full-head temperature normalisation; exact shared semantic IDs plus one mass-preserving residual bucket for every padding/unmapped ID. Physical padding identities are deliberately coarsened; no probability tail is discarded.".into()};
    let loss = ((teacher.exp()? * (&teacher - student)?)?
        .sum(1)?
        .mean_all()?
        * (temperature * temperature))?;
    Ok((loss, diagnostics))
}

pub fn sequence_sft(student_logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let (rows, vocabulary) = student_logits.dims2()?;
    ensure!(
        rows > 0 && vocabulary > 1 && targets.dims() == [rows],
        "SFT targets/logits have incompatible shapes"
    );
    let logp = candle_nn::ops::log_softmax(student_logits, D::Minus1)?;
    Ok(logp.gather(&targets.unsqueeze(1)?, 1)?.mean_all()?.neg()?)
}

/// Full-vocabulary KL requires an independently checked identical token-ID
/// mapping and aligned assistant spans. No unreported top-k tail is discarded.
pub fn softened_kl(student: &Tensor, teacher: &Tensor, temperature: f64) -> Result<Tensor> {
    ensure!(
        student.dims() == teacher.dims() && student.rank() == 2,
        "KL requires aligned complete vocabulary logits"
    );
    ensure!(
        temperature.is_finite() && (0.01..=100.).contains(&temperature),
        "invalid distillation temperature"
    );
    let student = candle_nn::ops::log_softmax(&(student / temperature)?, D::Minus1)?;
    let teacher = candle_nn::ops::log_softmax(&(teacher.detach() / temperature)?, D::Minus1)?;
    Ok(((teacher.exp()? * (&teacher - student)?)?
        .sum(D::Minus1)?
        .mean_all()?
        * (temperature * temperature))?)
}

fn unit_rows(features: &Tensor) -> Result<Tensor> {
    ensure!(
        features.rank() == 2 && features.dim(0)? > 0 && features.dim(1)? > 0,
        "nonempty feature matrix required"
    );
    let norm = (features.sqr()?.sum_keepdim(D::Minus1)? + 1e-8)?.sqrt()?;
    Ok(features.broadcast_div(&norm)?)
}

/// Fixed contiguous channel pooling avoids learning an unconstrained projection
/// that could minimise feature error without improving the student itself.
pub fn pool_channels(features: &Tensor, width: usize) -> Result<Tensor> {
    let (_, original) = features.dims2()?;
    ensure!(
        width > 0 && width <= original,
        "invalid feature-pooling width"
    );
    let pooled = (0..width)
        .map(|i| {
            let first = i * original / width;
            let end = (i + 1) * original / width;
            features.narrow(1, first, end - first)?.mean_keepdim(1)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::cat(&pooled, 1)?)
}

pub fn hidden_feature(student: &Tensor, teacher: &Tensor) -> Result<Tensor> {
    ensure!(
        student.dim(0)? == teacher.dim(0)?,
        "feature rows need byte-aligned assistant anchors"
    );
    let width = 64.min(student.dim(1)?).min(teacher.dim(1)?);
    let student = unit_rows(&pool_channels(student, width)?)?;
    let teacher = unit_rows(&pool_channels(&teacher.detach(), width)?)?;
    Ok((student - teacher)?.sqr()?.mean_all()?)
}

pub fn token_relations(features: &Tensor) -> Result<Tensor> {
    let features = unit_rows(features)?;
    Ok(features.matmul(&features.t()?)?)
}

/// Compare actual head-averaged attention on common byte-boundary anchors and
/// pairwise hidden-state cosine relations. Hidden width and head count may differ.
pub fn attention_relation(
    student_attention: &Tensor,
    teacher_attention: &Tensor,
    student_hidden: &Tensor,
    teacher_hidden: &Tensor,
) -> Result<Tensor> {
    ensure!(
        student_attention.dims() == teacher_attention.dims() && student_attention.rank() == 2,
        "attention anchors are not aligned"
    );
    let (n, m) = student_attention.dims2()?;
    ensure!(
        n > 0 && n == m && student_hidden.dim(0)? == n && teacher_hidden.dim(0)? == n,
        "attention/feature anchors disagree"
    );
    let normalise = |a: &Tensor| -> candle_core::Result<Tensor> {
        a.broadcast_div(&(a.sum_keepdim(1)? + 1e-8)?)
    };
    let attention_error = (normalise(student_attention)?
        - normalise(&teacher_attention.detach())?)?
    .sqr()?
    .mean_all()?;
    let relation_error = (token_relations(student_hidden)?
        - token_relations(&teacher_hidden.detach())?)?
    .sqr()?
    .mean_all()?;
    Ok(((attention_error + relation_error)? * 0.5)?)
}

/// Reference-corrected DPO on summed assistant log probabilities. Sequence
/// lengths remain part of that defined objective; it is not mean-token DPO.
pub fn preference_dpo(
    chosen: &Tensor,
    rejected: &Tensor,
    reference_chosen: &Tensor,
    reference_rejected: &Tensor,
    beta: f64,
) -> Result<Tensor> {
    ensure!(
        beta.is_finite() && (0.0001..=10.).contains(&beta),
        "invalid DPO beta"
    );
    ensure!(
        [chosen, rejected, reference_chosen, reference_rejected]
            .iter()
            .all(|x| x.rank() == 0),
        "DPO requires scalar summed log probabilities"
    );
    let margin = (((chosen - rejected)?
        - (reference_chosen.detach() - reference_rejected.detach())?)?
        * beta)?;
    // log-softmax([0, margin])[1] is a stable log-sigmoid, including large
    // margins, without an overflowing explicit exp(-margin).
    let pair = Tensor::stack(&[&margin.zeros_like()?, &margin], 0)?;
    Ok(candle_nn::ops::log_softmax(&pair, 0)?.i(1)?.neg()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn full_distribution_kl_matches_known_value_and_never_updates_teacher() {
        let device = Device::Cpu;
        let student = Var::new(&[[0_f32, 0.]], &device).unwrap();
        let teacher = Var::new(&[[3_f32.ln(), 1_f32.ln()]], &device).unwrap();
        let loss = softened_kl(&student, &teacher, 1.).unwrap();
        let expected = 0.75_f32 * (1.5_f32).ln() + 0.25_f32 * (0.5_f32).ln();
        assert!((loss.to_scalar::<f32>().unwrap() - expected).abs() < 1e-6);
        let grads = loss.backward().unwrap();
        assert!(grads.get(&teacher).is_none());
        let gradient = grads.get(&student).unwrap().to_vec2::<f32>().unwrap();
        assert!(gradient[0][0] < 0. && gradient[0][1] > 0.);
        let same = softened_kl(&teacher, &teacher, 2.)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(same.abs() < 1e-6);
    }

    #[test]
    fn different_padding_widths_preserve_nonzero_teacher_tail_and_student_gradients() {
        let student = Var::new(&[[0_f32, 0., 0.]], &Device::Cpu).unwrap();
        let teacher = Var::new(&[[0_f32, 0., 0., 0.]], &Device::Cpu).unwrap();
        let (loss, stats) = aligned_softened_kl(&student, &teacher, &[0, 1], 1.).unwrap();
        // Teacher categories are [1/4,1/4,1/2], student [1/3,1/3,1/3].
        let expected = 0.5_f32 * (0.75_f32).ln() + 0.5_f32 * (1.5_f32).ln();
        assert!((loss.to_scalar::<f32>().unwrap() - expected).abs() < 1e-6);
        assert!((stats.teacher_residual_probability_mean - 0.5).abs() < 1e-6);
        assert!((stats.student_residual_probability_mean - 1. / 3.).abs() < 1e-6);
        assert_eq!(stats.discarded_probability_mass, 0.);
        assert_eq!(
            (stats.student_padding_tokens, stats.teacher_padding_tokens),
            (1, 2)
        );
        let grad = loss.backward().unwrap();
        assert!(grad.get(&teacher).is_none());
        let values = grad.get(&student).unwrap().to_vec2::<f32>().unwrap();
        assert!(values[0][2] < 0. && values[0][0] > 0. && values[0].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn feature_pooling_retains_student_gradients_across_different_widths() {
        let student = Var::new(&[[1_f32, 2., 4., 8.], [2., 1., 3., 7.]], &Device::Cpu).unwrap();
        let teacher = Var::new(
            &[[4_f32, 3., 2., 1., 0., -1.], [1., 3., 5., 7., 9., 11.]],
            &Device::Cpu,
        )
        .unwrap();
        let loss = hidden_feature(&student, &teacher).unwrap();
        assert!(loss.to_scalar::<f32>().unwrap() > 0.);
        let grads = loss.backward().unwrap();
        assert!(grads.get(&teacher).is_none());
        assert!(
            grads
                .get(&student)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 0.
        );
    }

    #[test]
    fn attention_and_relations_are_invariant_to_feature_width_when_geometry_agrees() {
        let attention = Tensor::new(&[[1_f32, 0.], [0.25, 0.75]], &Device::Cpu).unwrap();
        let narrow = Tensor::new(&[[1_f32, 0.], [0., 1.]], &Device::Cpu).unwrap();
        let wide = Tensor::new(&[[1_f32, 0., 0., 0.], [0., 1., 0., 0.]], &Device::Cpu).unwrap();
        assert!(
            attention_relation(&attention, &attention, &narrow, &wide)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .abs()
                < 1e-7
        );
    }

    #[test]
    fn dpo_reference_correction_and_gradient_direction_are_real() {
        let chosen = Var::new(-2_f32, &Device::Cpu).unwrap();
        let rejected = Var::new(-3_f32, &Device::Cpu).unwrap();
        let ref_chosen = Var::new(-2_f32, &Device::Cpu).unwrap();
        let ref_rejected = Var::new(-3_f32, &Device::Cpu).unwrap();
        let loss = preference_dpo(&chosen, &rejected, &ref_chosen, &ref_rejected, 0.5).unwrap();
        assert!((loss.to_scalar::<f32>().unwrap() - 2_f32.ln()).abs() < 1e-6);
        let grads = loss.backward().unwrap();
        assert!(grads.get(&chosen).unwrap().to_scalar::<f32>().unwrap() < 0.);
        assert!(grads.get(&rejected).unwrap().to_scalar::<f32>().unwrap() > 0.);
        assert!(grads.get(&ref_chosen).is_none() && grads.get(&ref_rejected).is_none());
        let large = Tensor::new(10000_f32, &Device::Cpu).unwrap();
        assert!(
            preference_dpo(&large, &rejected, &ref_chosen, &ref_rejected, 0.5)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite()
        );
    }
}
