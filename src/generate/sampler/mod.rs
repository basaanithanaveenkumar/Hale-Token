//! Strategies for picking the next token from logits.

mod greedy;
mod sampling_params;
mod top_p_sampler;

pub use self::greedy::Greedy;
pub use self::sampling_params::SamplingParams;
pub use self::top_p_sampler::TopPSampler;

use std::sync::OnceLock;

use crate::registry::Registry;

/// Picks the next token id from a vector of logits.
///
/// A trait so new strategies (min-p, mirostat, ...) can be added without
/// touching the generation loop (Open/Closed principle).
pub trait Sampler {
    /// Returns the chosen token id for `logits` (one entry per vocabulary token).
    fn sample(&mut self, logits: &[f32]) -> u32;
}

/// Registry of samplers by name (`"greedy"`, `"top_p"`), for selecting a
/// strategy from configuration or a command-line flag.
///
/// # Examples
///
/// ```
/// use hale::generate::{sampler_registry, SamplingParams};
///
/// let mut sampler = sampler_registry()
///     .build("greedy", &SamplingParams::default())
///     .unwrap();
/// assert_eq!(sampler.sample(&[0.0, 2.0, 1.0]), 1);
/// ```
pub fn sampler_registry() -> &'static Registry<dyn Sampler, SamplingParams> {
    static REGISTRY: OnceLock<Registry<dyn Sampler, SamplingParams>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut registry: Registry<dyn Sampler, SamplingParams> = Registry::new("sampler");
        registry.register_force("greedy", |_| Box::new(Greedy));
        registry.register_force("top_p", |params| Box::new(TopPSampler::new(*params)));
        registry
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_builds_samplers_by_name() {
        let params = SamplingParams::default();
        let mut greedy = sampler_registry().build("greedy", &params).unwrap();
        assert_eq!(greedy.sample(&[0.0, 1.0]), 1);
        assert!(sampler_registry().build("top_p", &params).is_ok());
        assert!(sampler_registry().build("nucleus", &params).is_err());
    }

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
