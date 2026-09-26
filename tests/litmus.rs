//! Litmus tests: the engine must reproduce an independent reference.
//!
//! The reference (`scripts/reference/make_fixtures.py`) is a NumPy float64
//! implementation using full-sequence attention. The engine uses an f32
//! incremental KV cache and a tiered expert store. If both agree on every
//! logit of every position, the attention, RoPE, QK-norm, routing, expert
//! and dense-MLP maths are all right.

mod common;

use common::*;
use hale::generate::{generate, GenerationOptions, SamplingParams};

/// f32 accumulation vs float64 reference; logits are O(10).
const LOGIT_TOLERANCE: f32 = 2e-3;

#[test]
fn logits_match_the_reference_for_every_position() {
    for name in FIXTURES {
        let stats = tempfile::tempdir().unwrap();
        let engine = load(&fixture(name), 1 << 30, stats.path());
        let exp = expected(name);
        let got = teacher_forced_logits(&engine, &exp.tokens);
        let diff = max_abs_diff(&got, &exp.logits);
        assert!(diff < LOGIT_TOLERANCE, "{name}: max |logit diff| = {diff}");
    }
}

#[test]
fn greedy_generation_matches_the_reference() {
    for name in FIXTURES {
        let stats = tempfile::tempdir().unwrap();
        let engine = load(&fixture(name), 1 << 30, stats.path());
        let exp = expected(name);
        let options = GenerationOptions {
            max_new_tokens: exp.greedy_continuation.len(),
            sampling: SamplingParams::greedy(),
            stop_tokens: vec![],
        };
        let (tokens, stats) = generate(engine.model(), &exp.prompt, &options, |_| true).unwrap();
        assert_eq!(tokens, exp.greedy_continuation, "{name}");
        assert_eq!(stats.prompt_tokens, exp.prompt.len());
        assert_eq!(stats.generated_tokens, exp.greedy_continuation.len());
    }
}

#[test]
fn prefill_equals_token_by_token_forward() {
    let name = FIXTURES[0];
    let stats = tempfile::tempdir().unwrap();
    let engine = load(&fixture(name), 1 << 30, stats.path());
    let exp = expected(name);
    let model = engine.model();
    let mut cache = model.new_cache(64);
    let prefill = model.prefill(&exp.prompt, &mut cache).unwrap();
    let stepwise = teacher_forced_logits(&engine, &exp.prompt);
    assert_eq!(&prefill, stepwise.last().unwrap(), "bit-identical");
    assert_eq!(cache.len(), exp.prompt.len());
}
