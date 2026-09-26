//! Root-mean-square layer normalisation.

/// `out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]`
///
/// RMSNorm is LayerNorm without the mean subtraction or bias; every model we
/// support uses it before attention, before the MLP, and on the final output.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(x.len(), out.len());
    let scale = inverse_rms(x, eps);
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

/// In-place variant of [`rms_norm`] (used for per-head QK-norm).
pub fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let scale = inverse_rms(x, eps);
    for (v, w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
}

fn inverse_rms(x: &[f32], eps: f32) -> f32 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (mean_sq + eps).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_to_unit_rms() {
        let x = [3.0, -4.0, 0.0, 5.0];
        let mut out = [0.0; 4];
        rms_norm(&x, &[1.0; 4], 0.0, &mut out);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);

        let mut y = x;
        rms_norm_in_place(&mut y, &[2.0; 4], 0.0);
        for (a, b) in y.iter().zip(&out) {
            assert!((a - 2.0 * b).abs() < 1e-6);
        }
    }
}
