//! Rotary position embeddings (RoPE).
//!
//! RoPE encodes a token's position by rotating pairs of query/key features
//! by an angle proportional to the position. We follow the Hugging Face
//! "rotate-half" convention: feature `i` is paired with feature `i + d/2`.

/// Pre-computed inverse frequencies for one head size.
#[derive(Debug, Clone)]
pub struct Rope {
    inv_freq: Vec<f32>,
}

impl Rope {
    /// `theta` is the model's `rope_theta` (e.g. 1e6 for Qwen3).
    pub fn new(head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();
        Rope { inv_freq }
    }

    /// Fills `cos`/`sin` (length `head_dim / 2`) for `position`.
    ///
    /// Computed once per token and shared by every head in every layer.
    pub fn angles(&self, position: usize, cos: &mut [f32], sin: &mut [f32]) {
        for ((c, s), f) in cos.iter_mut().zip(sin.iter_mut()).zip(&self.inv_freq) {
            let angle = position as f32 * f;
            *c = angle.cos();
            *s = angle.sin();
        }
    }

    /// Rotates one head (length `head_dim`) in place.
    pub fn apply(head: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = head.len() / 2;
        let (first, second) = head.split_at_mut(half);
        for i in 0..half {
            let (a, b) = (first[i], second[i]);
            first[i] = a * cos[i] - b * sin[i];
            second[i] = b * cos[i] + a * sin[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_is_identity_and_rotation_preserves_norm() {
        let rope = Rope::new(8, 10_000.0);
        let (mut cos, mut sin) = ([0.0; 4], [0.0; 4]);
        let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        rope.angles(0, &mut cos, &mut sin);
        let mut h = original;
        Rope::apply(&mut h, &cos, &sin);
        assert_eq!(h, original);

        rope.angles(17, &mut cos, &mut sin);
        Rope::apply(&mut h, &cos, &sin);
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((norm(&h) - norm(&original)).abs() < 1e-3);
        assert_ne!(h, original);
    }
}
