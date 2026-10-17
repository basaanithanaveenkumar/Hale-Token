//! The machine quantities the planner needs.

use super::{DEFAULT_SSD_GBPS, GFLOPS_PER_EFFICIENCY_CORE, GFLOPS_PER_PERFORMANCE_CORE};
use crate::hardware::ChipSpec;
use serde::Serialize;

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
