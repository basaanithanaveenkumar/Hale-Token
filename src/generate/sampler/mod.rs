//! Strategies for picking the next token from logits.

mod greedy;
mod sampling_params;
mod top_p_sampler;

pub use self::greedy::Greedy;
pub use self::sampling_params::SamplingParams;
pub use self::top_p_sampler::TopPSampler;

/// Picks the next token id from a vector of logits.
///
/// A trait so new strategies (min-p, mirostat, ...) can be added without
/// touching the generation loop (Open/Closed principle).
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
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
