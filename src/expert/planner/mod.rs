//! Deciding what lives in RAM and predicting tokens per second.
//!
//! # Cost model
//!
//! Decoding one token reads every *dense* weight once and, in each MoE
//! layer, `k` experts. Weights already in unified memory stream at memory
//! bandwidth; experts that miss the cache stream from the SSD. The CPU also
//! has to do `2 x active_params` floating-point operations. So:
//!
//! ```text
//! bytes_ram  = dense_bytes + hit_rate       * active_expert_bytes
//! bytes_ssd  =               (1 - hit_rate) * active_expert_bytes
//! t_memory   = bytes_ram / (memory_bw * MEMORY_EFFICIENCY)
//! t_compute  = 2 * active_params / cpu_flops
//! t_token    = max(t_memory, t_compute) + bytes_ssd / ssd_bw
//! ```
//!
//! SSD time is *added* rather than overlapped because a layer cannot run
//! until its experts have arrived.
//!
//! The hit rate assumes uniform routing (hit rate = fraction of experts in
//! RAM). Real routers are skewed, so real hit rates - reported after every
//! `hale run` - are higher: treat the estimate as a floor.
//!
//! # Placement policy
//!
//! 1. If every expert fits in the RAM budget, pin them all ("resident").
//! 2. Otherwise split the budget: half pinned to the historically hottest
//!    experts (if routing stats exist), the rest a recency (LRU) cache.

mod hardware_profile;
mod model_shape;
mod plan;
mod plan_request;
mod tier_split;

pub use self::hardware_profile::HardwareProfile;
pub use self::model_shape::ModelShape;
pub use self::plan::Plan;
pub use self::plan_request::PlanRequest;
pub use self::tier_split::TierSplit;

use crate::tensor::DType;

/// Fraction of advertised memory bandwidth the CPU cores actually achieve.
pub const MEMORY_EFFICIENCY: f64 = 0.6;

/// Sustained GFLOP/s per performance core for our quantized mat-vec kernels.
pub const GFLOPS_PER_PERFORMANCE_CORE: f64 = 24.0;

/// Sustained GFLOP/s per efficiency core.
pub const GFLOPS_PER_EFFICIENCY_CORE: f64 = 6.0;

/// Share of installed RAM Hale-Token may use; macOS and apps need the rest.
pub const DEFAULT_RAM_FRACTION: f64 = 0.75;

/// Conservative sustained random-read speed of an Apple internal SSD.
pub const DEFAULT_SSD_GBPS: f64 = 5.0;

/// Bytes kept free for activations, KV cache and the runtime.
pub const RUNTIME_OVERHEAD_BYTES: f64 = 2.0e9;

/// Divides `experts_in_ram` slots between the pinned and LRU tiers
/// (the placement policy from the module docs).
///
/// `per_token_experts` is how many experts one token touches
/// (`moe_layers x k`). Every token walks the layers in the same order, so
/// the access pattern is cyclic; an LRU smaller than one token's working
/// set evicts each expert just before it is needed again and hits almost
/// never (measured: 0% on TinyMixtral with a 25% cache). Below twice the
/// working set every slot is therefore pinned instead.
pub fn split_budget(
    experts_in_ram: usize,
    total_experts: usize,
    per_token_experts: usize,
) -> TierSplit {
    let n = experts_in_ram.min(total_experts);
    if n == total_experts || n < 2 * per_token_experts {
        TierSplit { pinned: n, lru: 0 }
    } else {
        TierSplit {
            pinned: n / 2,
            lru: n - n / 2,
        }
    }
}

/// Computes a placement plan and speed estimate. Pure function: easy to test.
pub fn plan(req: &PlanRequest) -> Plan {
    let m = &req.model;
    let hw = &req.hardware;
    let bytes = |params: f64, d: DType| params * d.bits_per_weight() / 8.0;

    let dense_bytes = bytes(m.dense_params, req.dense_dtype);
    let expert_bytes = bytes(m.params_per_expert, req.expert_dtype);
    let total_experts = m.total_experts();
    let total_expert_bytes = expert_bytes * total_experts as f64;

    let ram_budget_bytes = hw.ram_bytes * req.ram_fraction;
    let expert_ram_bytes = (ram_budget_bytes - dense_bytes - RUNTIME_OVERHEAD_BYTES).max(0.0);
    let feasible = ram_budget_bytes > dense_bytes + RUNTIME_OVERHEAD_BYTES;

    let experts_in_ram = ((expert_ram_bytes / expert_bytes).floor() as usize).min(total_experts);
    let fully_resident = experts_in_ram == total_experts;
    let split = split_budget(
        experts_in_ram,
        total_experts,
        m.moe_layers * m.experts_per_token,
    );
    let (pinned_experts, lru_bytes) = (split.pinned, split.lru as f64 * expert_bytes);
    let expected_hit_rate = if total_experts == 0 {
        1.0
    } else {
        experts_in_ram as f64 / total_experts as f64
    };

    let active_expert_bytes = (m.moe_layers * m.experts_per_token) as f64 * expert_bytes;
    let t_memory = (dense_bytes + expected_hit_rate * active_expert_bytes)
        / (hw.memory_bandwidth_gbps * 1e9 * MEMORY_EFFICIENCY);
    let t_compute = 2.0 * m.active_params() / (hw.cpu_gflops * 1e9);
    let t_ssd = (1.0 - expected_hit_rate) * active_expert_bytes / (hw.ssd_bandwidth_gbps * 1e9);
    let t_token = t_memory.max(t_compute) + t_ssd;

    let bottleneck = if t_ssd > t_memory.max(t_compute) {
        "ssd"
    } else if t_compute > t_memory {
        "compute"
    } else {
        "memory"
    };

    Plan {
        dense_bytes,
        expert_bytes,
        total_expert_bytes,
        ram_budget_bytes,
        expert_ram_bytes,
        feasible,
        fully_resident,
        pinned_experts,
        lru_bytes,
        expected_hit_rate,
        est_tokens_per_second: if feasible && t_token > 0.0 {
            1.0 / t_token
        } else {
            0.0
        },
        bottleneck,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::lookup_chip;

    fn request(model: &str, chip: &str, ram_gb: f64) -> PlanRequest {
        PlanRequest {
            model: ModelShape::preset(model).unwrap(),
            hardware: HardwareProfile::from_chip(lookup_chip(chip).unwrap(), ram_gb),
            dense_dtype: DType::BF16,
            expert_dtype: DType::Q4_0,
            ram_fraction: DEFAULT_RAM_FRACTION,
            have_routing_stats: true,
        }
    }

    #[test]
    fn preset_sizes_match_published_totals() {
        let expect = [
            ("Qwen3-30B", 30.5, 3.3),
            ("Qwen3-235B", 235.0, 22.0),
            ("Mixtral-8x7B", 46.7, 12.9),
            ("Mixtral-8x22B", 141.0, 39.0),
            ("DeepSeek-V3", 671.0, 37.0),
        ];
        for (name, total, active) in expect {
            let s = ModelShape::preset(name).unwrap();
            let (t, a) = (s.total_params() / 1e9, s.active_params() / 1e9);
            assert!((t - total).abs() / total < 0.03, "{name}: total {t}");
            assert!((a - active).abs() / active < 0.06, "{name}: active {a}");
        }
    }

    #[test]
    fn small_model_on_big_mac_is_fully_resident() {
        let p = plan(&request("Qwen3-30B", "M3 Max", 128.0));
        assert!(p.feasible && p.fully_resident);
        assert_eq!(p.lru_bytes, 0.0);
        assert!((p.expected_hit_rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn datacenter_model_on_laptop_streams_from_ssd() {
        let p = plan(&request("Qwen3-235B", "M4 Max", 64.0));
        assert!(p.feasible);
        assert!(!p.fully_resident);
        assert!(p.pinned_experts > 0 && p.lru_bytes > 0.0);
        assert!(p.expected_hit_rate > 0.1 && p.expected_hit_rate < 1.0);
        assert!(p.est_tokens_per_second > 0.0);
    }

    #[test]
    fn more_ram_never_makes_it_slower() {
        let mut last = 0.0;
        for ram in [32.0, 64.0, 128.0, 192.0] {
            let tps = plan(&request("Qwen3-235B", "M2 Ultra", ram)).est_tokens_per_second;
            assert!(tps >= last, "{ram} GB: {tps} < {last}");
            last = tps;
        }
    }

    #[test]
    fn small_caches_pin_instead_of_thrashing_an_lru() {
        // 12 layers x top-2 = 24 experts per token.
        assert_eq!(split_budget(12, 48, 24), TierSplit { pinned: 12, lru: 0 });
        assert_eq!(split_budget(47, 48, 24), TierSplit { pinned: 47, lru: 0 });
        assert_eq!(
            split_budget(100, 1000, 24),
            TierSplit {
                pinned: 50,
                lru: 50
            }
        );
        assert_eq!(
            split_budget(2000, 1000, 24),
            TierSplit {
                pinned: 1000,
                lru: 0
            }
        );
    }

    #[test]
    fn too_little_ram_is_infeasible() {
        let p = plan(&request("DeepSeek-V3", "M1", 8.0));
        assert!(!p.feasible);
        assert_eq!(p.est_tokens_per_second, 0.0);
    }
}
