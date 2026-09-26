//! Element-wise activations and reductions.

/// SiLU (a.k.a. swish): `x * sigmoid(x)`. Used inside SwiGLU experts.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Numerically stable softmax: subtracts the max before exponentiating so
/// large logits cannot overflow.
pub fn softmax_in_place(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Index of the largest value (first one wins on ties). Returns 0 for an empty slice.
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_survives_huge_logits() {
        let mut x = [1000.0, 1001.0, 1002.0];
        softmax_in_place(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn silu_and_argmax_basics() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(10.0) - 10.0).abs() < 1e-3);
        assert_eq!(argmax(&[1.0, 3.0, 3.0, -1.0]), 1);
    }
}
