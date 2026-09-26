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

use serde::Serialize;

use crate::config::ModelConfig;
use crate::hardware::ChipSpec;
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

/// The sizes that matter for planning, independent of the architecture.
#[derive(Debug, Clone, Serialize)]
pub struct ModelShape {
    pub name: String,
    /// Parameters that are not routed experts.
    pub dense_params: f64,
    /// Parameters in one routed expert.
    pub params_per_expert: f64,
    pub moe_layers: usize,
    pub experts_per_layer: usize,
    pub experts_per_token: usize,
    /// Whether Hale-Token can execute this architecture today.
    pub runnable: bool,
}

impl ModelShape {
    /// Shape of a model we can load.
    pub fn from_config(name: &str, c: &ModelConfig) -> Self {
        ModelShape {
            name: name.to_string(),
            dense_params: c.dense_params() as f64,
            params_per_expert: c.params_per_expert() as f64,
            moe_layers: c.num_moe_layers(),
            experts_per_layer: c.num_experts,
            experts_per_token: c.experts_per_token,
            runnable: true,
        }
    }

    pub fn total_experts(&self) -> usize {
        self.moe_layers * self.experts_per_layer
    }
    pub fn total_params(&self) -> f64 {
        self.dense_params + self.total_experts() as f64 * self.params_per_expert
    }
    pub fn active_params(&self) -> f64 {
        self.dense_params
            + (self.moe_layers * self.experts_per_token) as f64 * self.params_per_expert
    }

    /// Well-known MoE models for comparison. Figures derive from each
    /// model's published `config.json`.
    pub fn presets() -> Vec<ModelShape> {
        let shape =
            |name: &str, dense: f64, per_expert: f64, layers, experts, k, runnable| ModelShape {
                name: name.to_string(),
                dense_params: dense,
                params_per_expert: per_expert,
                moe_layers: layers,
                experts_per_layer: experts,
                experts_per_token: k,
                runnable,
            };
        vec![
            // hidden 2048, expert inter 768, 48 layers x 128 experts, top-8
            shape(
                "Qwen3-30B-A3B",
                1.53e9,
                3.0 * 2048.0 * 768.0,
                48,
                128,
                8,
                true,
            ),
            // hidden 4096, expert inter 1536, 94 layers x 128 experts, top-8
            shape(
                "Qwen3-235B-A22B",
                7.98e9,
                3.0 * 4096.0 * 1536.0,
                94,
                128,
                8,
                true,
            ),
            // hidden 4096, inter 14336, 32 layers x 8 experts, top-2
            shape(
                "Mixtral-8x7B",
                1.60e9,
                3.0 * 4096.0 * 14336.0,
                32,
                8,
                2,
                true,
            ),
            // hidden 6144, inter 16384, 56 layers x 8 experts, top-2
            shape(
                "Mixtral-8x22B",
                5.34e9,
                3.0 * 6144.0 * 16384.0,
                56,
                8,
                2,
                true,
            ),
            // hidden 7168, expert inter 2048, 58 MoE layers x 256 experts, top-8
            shape(
                "DeepSeek-V3 (671B)",
                17.1e9,
                3.0 * 7168.0 * 2048.0,
                58,
                256,
                8,
                false,
            ),
            // hidden 7168, expert inter 2048, 60 MoE layers x 384 experts, top-8
            shape(
                "Kimi-K2 (1T)",
                11.5e9,
                3.0 * 7168.0 * 2048.0,
                60,
                384,
                8,
                false,
            ),
        ]
    }

    /// Finds a preset by case-insensitive prefix, e.g. `"qwen3-235b"`.
    pub fn preset(name: &str) -> Option<ModelShape> {
        let wanted = name.to_ascii_lowercase();
        Self::presets()
            .into_iter()
            .find(|p| p.name.to_ascii_lowercase().starts_with(&wanted))
    }
}

/// The machine being planned for.
#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub name: String,
    pub memory_bandwidth_gbps: f64,
    pub ssd_bandwidth_gbps: f64,
    pub ram_bytes: f64,
    pub cpu_gflops: f64,
}

impl HardwareProfile {
    /// Profile for a chip with `ram_gb` of unified memory.
    pub fn from_chip(chip: &ChipSpec, ram_gb: f64) -> Self {
        HardwareProfile {
            name: format!("{} {:.0}GB", chip.name, ram_gb),
            memory_bandwidth_gbps: chip.memory_bandwidth_gbps,
            ssd_bandwidth_gbps: DEFAULT_SSD_GBPS,
            ram_bytes: ram_gb * 1e9,
            cpu_gflops: chip.performance_cores as f64 * GFLOPS_PER_PERFORMANCE_CORE
                + chip.efficiency_cores as f64 * GFLOPS_PER_EFFICIENCY_CORE,
        }
    }
}

/// Planner inputs.
#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub model: ModelShape,
    pub hardware: HardwareProfile,
    /// Storage format of the dense weights (usually the checkpoint's bf16).
    pub dense_dtype: DType,
    /// Storage format of the experts (e.g. q4_0 after `hale convert`).
    pub expert_dtype: DType,
    /// Share of RAM we may use (see [`DEFAULT_RAM_FRACTION`]).
    pub ram_fraction: f64,
    /// Whether routing statistics exist to choose a pinned set.
    pub have_routing_stats: bool,
}

/// How the tiers should be sized, and what speed to expect.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub dense_bytes: f64,
    pub expert_bytes: f64,
    pub total_expert_bytes: f64,
    pub ram_budget_bytes: f64,
    /// RAM available to experts after dense weights and overhead.
    pub expert_ram_bytes: f64,
    /// Dense weights plus overhead fit in RAM.
    pub feasible: bool,
    /// Every expert fits in RAM - no SSD traffic at all.
    pub fully_resident: bool,
    pub pinned_experts: usize,
    pub lru_bytes: f64,
    /// Expected fraction of expert requests served from RAM (floor).
    pub expected_hit_rate: f64,
    pub est_tokens_per_second: f64,
    /// Which resource bounds the speed: "memory", "compute" or "ssd".
    pub bottleneck: &'static str,
}

/// How many RAM-resident experts go to each tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSplit {
    pub pinned: usize,
    pub lru: usize,
}

/// Divides `experts_in_ram` slots between the pinned and LRU tiers
/// (the placement policy from the module docs).
pub fn split_budget(
    experts_in_ram: usize,
    total_experts: usize,
    have_routing_stats: bool,
) -> TierSplit {
    let n = experts_in_ram.min(total_experts);
    if n == total_experts {
        TierSplit { pinned: n, lru: 0 }
    } else if have_routing_stats {
        TierSplit {
            pinned: n / 2,
            lru: n - n / 2,
        }
    } else {
        TierSplit { pinned: 0, lru: n }
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
    let split = split_budget(experts_in_ram, total_experts, req.have_routing_stats);
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
    fn too_little_ram_is_infeasible() {
        let p = plan(&request("DeepSeek-V3", "M1", 8.0));
        assert!(!p.feasible);
        assert_eq!(p.est_tokens_per_second, 0.0);
    }
}
