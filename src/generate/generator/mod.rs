//! The autoregressive decode loop.

mod generation_options;
mod generation_stats;

pub use self::generation_options::GenerationOptions;
pub use self::generation_stats::GenerationStats;

use crate::error::{HaleTokenError, Result};
use crate::model::{KvCache, Transformer};
use std::time::Instant;

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
        return Err(HaleTokenError::ContextOverflow { needed, max });
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
