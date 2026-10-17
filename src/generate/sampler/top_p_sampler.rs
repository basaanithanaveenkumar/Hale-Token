//! Temperature + top-k + nucleus sampling with a seedable PRNG.

use super::{Sampler, SamplingParams};

/// Temperature + top-k + top-p (nucleus) sampling.
pub struct TopPSampler {
    params: SamplingParams,
    rng: XorShift64,
}

impl TopPSampler {
    pub fn new(params: SamplingParams) -> Self {
        TopPSampler {
            params,
            rng: XorShift64::new(params.seed),
        }
    }
}

impl Sampler for TopPSampler {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let p = self.params;
        // Candidates sorted by logit, best first; optionally truncated to top-k.
        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        let k = if p.top_k == 0 {
            candidates.len()
        } else {
            p.top_k.min(candidates.len())
        };
        if k < candidates.len() {
            candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // Softmax with temperature over the survivors.
        let max = candidates[0].1;
        let mut total = 0.0;
        for c in candidates.iter_mut() {
            c.1 = ((c.1 - max) / p.temperature).exp();
            total += c.1;
        }
        // Nucleus cut: smallest prefix whose mass reaches top_p.
        let mut mass = 0.0;
        let mut keep = candidates.len();
        for (i, c) in candidates.iter().enumerate() {
            mass += c.1 / total;
            if mass >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        candidates.truncate(keep);

        let kept_total: f32 = candidates.iter().map(|c| c.1).sum();
        let mut r = self.rng.next_f32() * kept_total;
        for (index, weight) in &candidates {
            r -= weight;
            if r <= 0.0 {
                return *index as u32;
            }
        }
        // Rounding can leave `r` a hair above zero after the last subtraction; the last kept
        // candidate is then the right pick. `candidates` is never empty for a non-empty vocab.
        candidates.last().map_or(0, |c| c.0 as u32)
    }
}

/// Tiny, fast, seedable PRNG (Marsaglia xorshift64*). Good enough for
/// sampling, and avoids an extra dependency.
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}
