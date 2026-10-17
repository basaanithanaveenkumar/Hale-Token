//! The planner output: tier split and speed estimate.

use serde::Serialize;

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
