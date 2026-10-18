//! The planner output: tier split and speed estimate.

use serde::Serialize;

/// How the tiers should be sized, and what speed to expect.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    /// Bytes of dense (non-expert) weights.
    pub dense_bytes: f64,
    /// Bytes of one expert.
    pub expert_bytes: f64,
    /// Bytes of all experts.
    pub total_expert_bytes: f64,
    /// RAM available to the whole model.
    pub ram_budget_bytes: f64,
    /// RAM available to experts after dense weights and overhead.
    pub expert_ram_bytes: f64,
    /// Dense weights plus overhead fit in RAM.
    pub feasible: bool,
    /// Every expert fits in RAM - no SSD traffic at all.
    pub fully_resident: bool,
    /// Experts to pin in RAM.
    pub pinned_experts: usize,
    /// Bytes to give the LRU tier.
    pub lru_bytes: f64,
    /// Expected fraction of expert requests served from RAM (floor).
    pub expected_hit_rate: f64,
    /// Estimated decode speed in tokens per second.
    pub est_tokens_per_second: f64,
    /// Which resource bounds the speed: "memory", "compute" or "ssd".
    pub bottleneck: &'static str,
}
