//! The autoregressive decode loop.

use std::time::Instant;

use serde::Serialize;

use super::SamplingParams;
use crate::error::{HaleError, Result};
use crate::expert::CacheStats;
use crate::model::{KvCache, Transformer};

/// Settings for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    /// Generation stops after emitting any of these ids.
    pub stop_tokens: Vec<u32>,
}

/// Timing and cache figures for one generation call.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub cache: CacheStats,
}

impl GenerationStats {
    /// Prompt processing speed.
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens, self.prefill_seconds)
    }

    /// Generation speed (excluding the first token, which comes from prefill).
    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.generated_tokens.saturating_sub(1), self.decode_seconds)
    }
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

/// Generates up to `options.max_new_tokens` tokens after `prompt`.
///
/// `on_token` is called with every new token id as soon as it is sampled;
/// return `false` from it to stop early. The KV cache is created fresh.
pub fn generate(
    model: &Transformer,
    prompt: &[u32],
    options: &GenerationOptions,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<(Vec<u32>, GenerationStats)> {
    let max = model.config.max_position_embeddings;
    let needed = prompt.len().saturating_add(options.max_new_tokens);
    if needed > max {
        return Err(HaleError::ContextOverflow { needed, max });
    }
    let mut cache: KvCache = model.new_cache(needed);
    let mut sampler = options.sampling.build();

    let started = Instant::now();
    let mut logits = model.prefill(prompt, &mut cache)?;
    let prefill_seconds = started.elapsed().as_secs_f64();

    let decode_started = Instant::now();
    let mut output = Vec::with_capacity(options.max_new_tokens);
    for step in 0..options.max_new_tokens {
        let token = sampler.sample(&logits);
        output.push(token);
        let keep_going = on_token(token);
        let is_last = step + 1 == options.max_new_tokens;
        if !keep_going || options.stop_tokens.contains(&token) || is_last {
            break;
        }
        logits = model.forward(token, &mut cache)?;
    }

    let stats = GenerationStats {
        prompt_tokens: prompt.len(),
        generated_tokens: output.len(),
        prefill_seconds,
        decode_seconds: decode_started.elapsed().as_secs_f64(),
        cache: model.experts().stats(),
    };
    Ok((output, stats))
}
