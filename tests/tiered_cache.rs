//! The central promise of Hale-Token: where an expert lives (pinned RAM,
//! LRU RAM, or SSD) must never change the answer.

mod common;

use common::*;
use hale::convert::convert_checkpoint;
use hale::tensor::DType;

/// Converts a fixture into a pack with experts stored as `dtype`.
fn convert(name: &str, dtype: DType) -> tempfile::TempDir {
    let out = tempfile::tempdir().unwrap();
    convert_checkpoint(&fixture(name), out.path(), dtype, |_, _| {}).unwrap();
    out
}

/// The fixture's native dtype, so conversion is lossless.
fn native_dtype(name: &str) -> DType {
    if name.contains("mixtral") {
        DType::BF16
    } else {
        DType::F32
    }
}

#[test]
fn ssd_streaming_is_bit_identical_to_resident() {
    for name in FIXTURES {
        let packed = convert(name, native_dtype(name));
        let exp = expected(name);
        let stats = tempfile::tempdir().unwrap();

        let resident = load(packed.path(), 1 << 30, stats.path());
        let resident_logits = teacher_forced_logits(&resident, &exp.tokens);
        let r = resident.model().experts().stats();
        assert!(
            r.misses == 0 && r.pinned_experts > 0,
            "{name}: everything pinned: {r:?}"
        );

        // Room for a single expert: nearly every request goes to the SSD.
        let one_expert = resident.model().experts().stats().pinned_bytes / r.pinned_experts;
        let streaming = load(packed.path(), one_expert, stats.path());
        let streaming_logits = teacher_forced_logits(&streaming, &exp.tokens);
        let s = streaming.model().experts().stats();
        assert!(
            s.misses > s.lru_hits,
            "{name}: expected mostly misses, got {s:?}"
        );
        assert!(s.bytes_loaded > 0);

        assert_eq!(
            resident_logits, streaming_logits,
            "{name}: tiers changed the result"
        );
        assert!(
            max_abs_diff(&resident_logits, &exp.logits) < 2e-3,
            "{name}: pack diverged from reference"
        );
    }
}

/// Relative RMS error `||got - want|| / ||want||` over all logits.
fn relative_rms_error(got: &[Vec<f32>], want: &[Vec<f32>]) -> f32 {
    let (mut err, mut norm) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
        err += f64::from(g - w).powi(2);
        norm += f64::from(*w).powi(2);
    }
    (err / norm).sqrt() as f32
}

/// Fraction of positions whose most likely next token is unchanged.
fn top1_agreement(got: &[Vec<f32>], want: &[Vec<f32>]) -> f32 {
    let same = got
        .iter()
        .zip(want)
        .filter(|(g, w)| hale::ops::argmax(g) == hale::ops::argmax(w))
        .count();
    same as f32 / got.len() as f32
}

#[test]
fn quantized_packs_stay_close_to_the_reference() {
    // Random Gaussian weights are the worst case for block quantization (no
    // structure to exploit) and errors compound over layers, so these bounds
    // are loose; trained models fare better.
    for name in FIXTURES {
        let exp = expected(name);
        for (dtype, max_rms, min_top1) in [(DType::Q8_0, 0.01, 0.95), (DType::Q4_0, 0.25, 0.6)] {
            let packed = convert(name, dtype);
            let stats = tempfile::tempdir().unwrap();
            let engine = load(packed.path(), 1 << 30, stats.path());
            let logits = teacher_forced_logits(&engine, &exp.tokens);
            let rms = relative_rms_error(&logits, &exp.logits);
            let top1 = top1_agreement(&logits, &exp.logits);
            println!("{name} {dtype}: relative RMS error {rms:.4}, top-1 agreement {top1:.2}");
            assert!(rms < max_rms, "{name} {dtype}: relative RMS error {rms}");
            assert!(top1 >= min_top1, "{name} {dtype}: top-1 agreement {top1}");
        }
    }
}

#[test]
fn routing_stats_warm_start_pins_the_hottest_experts() {
    let name = FIXTURES[0];
    let packed = convert(name, DType::F32);
    let exp = expected(name);
    let stats_dir = tempfile::tempdir().unwrap();

    // Run once with a tiny cache and save what the router chose.
    let first = load(packed.path(), 1, stats_dir.path());
    teacher_forced_logits(&first, &exp.tokens);
    first.save_routing_stats().unwrap();
    let expert_bytes = first.model().experts().stats().bytes_loaded as usize
        / first.model().experts().stats().misses as usize;

    // Second start with room for 4 experts: 2 pinned (hottest) + 2 LRU.
    let second = load(packed.path(), 4 * expert_bytes, stats_dir.path());
    let s = second.model().experts().stats();
    assert_eq!(s.pinned_experts, 2, "{s:?}");
    teacher_forced_logits(&second, &exp.tokens);
    let s = second.model().experts().stats();
    assert!(s.pinned_hits > 0, "hottest experts should be hit: {s:?}");
}
