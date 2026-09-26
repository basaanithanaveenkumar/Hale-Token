//! Strategies for picking the next token from logits.

/// Picks the next token id from a vector of logits.
///
/// A trait so new strategies (min-p, mirostat, ...) can be added without
/// touching the generation loop (Open/Closed principle).
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}

/// Always takes the most likely token. Deterministic; ideal for tests.
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        crate::ops::argmax(logits) as u32
    }
}

/// User-facing sampling settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// 0 means greedy; higher values flatten the distribution.
    pub temperature: f32,
    /// Nucleus sampling: keep the smallest set of tokens whose probability
    /// mass reaches `top_p` (1.0 disables it).
    pub top_p: f32,
    /// Keep only the `top_k` most likely tokens (0 disables it).
    pub top_k: usize,
    /// Random seed, so runs are reproducible.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: 42,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding.
    pub fn greedy() -> Self {
        SamplingParams {
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Builds the matching sampler (Factory).
    pub fn build(&self) -> Box<dyn Sampler> {
        if self.temperature <= 0.0 {
            Box::new(Greedy)
        } else {
            Box::new(TopPSampler::new(*self))
        }
    }
}

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
        candidates.last().expect("at least one candidate").0 as u32
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_argmax() {
        assert_eq!(Greedy.sample(&[0.1, 3.0, -1.0]), 1);
        assert_eq!(SamplingParams::greedy().build().sample(&[5.0, 1.0]), 0);
    }

    #[test]
    fn top_p_only_samples_from_the_nucleus_and_is_reproducible() {
        let logits = [10.0, 9.5, 0.0, -5.0, -5.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            seed: 7,
        };
        let mut a = TopPSampler::new(params);
        let mut b = TopPSampler::new(params);
        let mut seen = [0usize; 5];
        for _ in 0..500 {
            let t = a.sample(&logits);
            assert_eq!(t, b.sample(&logits), "same seed, same stream");
            seen[t as usize] += 1;
        }
        assert!(
            seen[0] > 0 && seen[1] > 0,
            "both nucleus tokens appear: {seen:?}"
        );
        assert_eq!(
            seen[2] + seen[3] + seen[4],
            0,
            "tail is never sampled: {seen:?}"
        );
    }

    #[test]
    fn top_k_one_is_greedy() {
        let params = SamplingParams {
            temperature: 2.0,
            top_p: 1.0,
            top_k: 1,
            seed: 3,
        };
        let mut s = TopPSampler::new(params);
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.0, 0.1, 0.05]), 1);
        }
    }
}
