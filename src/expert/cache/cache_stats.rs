//! Hit/miss and byte counters of the [`ExpertCache`](super::ExpertCache).

use serde::Serialize;

/// Counters describing how well the cache is doing.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct CacheStats {
    /// Requests served from the pinned tier.
    pub pinned_hits: u64,
    /// Requests served from the LRU tier.
    pub lru_hits: u64,
    /// Requests that had to be loaded from the source.
    pub misses: u64,
    /// Bytes loaded from the source.
    pub bytes_loaded: u64,
    /// Wall-clock seconds spent loading from the source.
    pub load_seconds: f64,
    /// Number of experts held in the pinned tier.
    pub pinned_experts: usize,
    /// Bytes held in the pinned tier.
    pub pinned_bytes: usize,
    /// Number of experts currently in the LRU tier.
    pub lru_experts: usize,
    /// Bytes currently in the LRU tier.
    pub lru_bytes: usize,
    /// Byte budget of the LRU tier.
    pub lru_capacity_bytes: usize,
    /// True when the source is memory-mapped and caching is left to the OS.
    pub passthrough: bool,
}

impl CacheStats {
    /// Fraction of requests served from RAM tiers (0 when nothing was requested).
    pub fn hit_rate(&self) -> f64 {
        let hits = self.pinned_hits + self.lru_hits;
        let total = hits + self.misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Observed source read throughput in GB/s.
    pub fn load_gbps(&self) -> f64 {
        if self.load_seconds > 0.0 {
            self.bytes_loaded as f64 / self.load_seconds / 1e9
        } else {
            0.0
        }
    }
}
