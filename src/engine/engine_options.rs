//! User-tunable engine settings.

use crate::expert::planner::{self};
use std::path::PathBuf;

/// Knobs for loading a model.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// RAM for experts in bytes. `None` = let the planner decide from the
    /// machine's memory.
    pub expert_ram_bytes: Option<usize>,
    /// Share of total RAM the planner may use.
    pub ram_fraction: f64,
    /// Where routing counters are loaded from / saved to.
    /// `None` = `<model>/hale-routing.json`.
    pub routing_stats_path: Option<PathBuf>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            expert_ram_bytes: None,
            ram_fraction: planner::DEFAULT_RAM_FRACTION,
            routing_stats_path: None,
        }
    }
}
