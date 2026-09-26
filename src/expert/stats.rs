//! Per-expert routing counters.
//!
//! Routing in trained MoE models is skewed: some experts are chosen far more
//! often than others. Counting selections lets the planner pin the hottest
//! experts in RAM. Counters are saved to JSON after a run and reloaded next
//! time, so the cache starts "warm".

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ExpertKey;
use crate::error::{HaleError, Result};

/// Selection counts for every `(layer, expert)` pair, stored as one flat
/// vector indexed by `layer * num_experts + expert` (cache friendly and
/// trivially serialisable).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingStats {
    num_layers: usize,
    num_experts: usize,
    counts: Vec<u64>,
}

impl RoutingStats {
    /// All-zero counters for a model with the given shape.
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        RoutingStats {
            num_layers,
            num_experts,
            counts: vec![0; num_layers * num_experts],
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn slot(&self, key: ExpertKey) -> usize {
        key.layer as usize * self.num_experts + key.expert as usize
    }

    /// Records one selection of `key`.
    pub fn record(&mut self, key: ExpertKey) {
        let slot = self.slot(key);
        self.counts[slot] += 1;
    }

    /// Number of times `key` was selected.
    pub fn count(&self, key: ExpertKey) -> u64 {
        self.counts[self.slot(key)]
    }

    /// Total selections recorded.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Adds another set of counters of the same shape into this one.
    pub fn merge(&mut self, other: &RoutingStats) {
        if other.num_layers == self.num_layers && other.num_experts == self.num_experts {
            for (a, b) in self.counts.iter_mut().zip(&other.counts) {
                *a += b;
            }
        }
    }

    /// The `n` most frequently selected experts that were used at least once,
    /// hottest first.
    ///
    /// Uses quickselect (`select_nth_unstable_by`, average `O(len)`) to find
    /// the top `n` and then sorts only those `n`: `O(len + n log n)` instead
    /// of sorting everything.
    pub fn hottest(&self, n: usize) -> Vec<ExpertKey> {
        let mut used: Vec<usize> = (0..self.counts.len())
            .filter(|&i| self.counts[i] > 0)
            .collect();
        let n = n.min(used.len());
        if n == 0 {
            return Vec::new();
        }
        // Hotter first; ties broken by slot so the result is deterministic.
        let hotter = |a: &usize, b: &usize| self.counts[*b].cmp(&self.counts[*a]).then(a.cmp(b));
        if n < used.len() {
            used.select_nth_unstable_by(n - 1, hotter);
            used.truncate(n);
        }
        used.sort_unstable_by(hotter);
        used.into_iter()
            .map(|i| ExpertKey::new(i / self.num_experts, i % self.num_experts))
            .collect()
    }

    /// Fraction of all selections that went to the `n` hottest experts.
    /// This is the hit rate a pinned tier of `n` experts would have had.
    pub fn coverage(&self, n: usize) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let covered: u64 = self.hottest(n).iter().map(|k| self.count(*k)).sum();
        covered as f64 / total as f64
    }

    /// Loads counters from `path`.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| HaleError::io(path, e))?;
        let stats: RoutingStats = serde_json::from_str(&text).map_err(|e| HaleError::Json {
            path: path.to_path_buf(),
            source: e,
        })?;
        if stats.counts.len() != stats.num_layers * stats.num_experts {
            return Err(HaleError::InvalidArgument(format!(
                "{}: counter length does not match its shape",
                path.display()
            )));
        }
        Ok(stats)
    }

    /// Saves counters to `path` as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(self).expect("stats serialise");
        std::fs::write(path, text).map_err(|e| HaleError::io(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hottest_orders_by_count_and_skips_unused() {
        let mut s = RoutingStats::new(2, 4);
        for _ in 0..5 {
            s.record(ExpertKey::new(1, 2));
        }
        for _ in 0..3 {
            s.record(ExpertKey::new(0, 1));
        }
        s.record(ExpertKey::new(0, 3));
        assert_eq!(
            s.hottest(10),
            vec![
                ExpertKey::new(1, 2),
                ExpertKey::new(0, 1),
                ExpertKey::new(0, 3)
            ]
        );
        assert_eq!(s.hottest(1), vec![ExpertKey::new(1, 2)]);
        assert!((s.coverage(2) - 8.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn save_load_and_merge() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stats.json");
        let mut s = RoutingStats::new(1, 3);
        s.record(ExpertKey::new(0, 2));
        s.save(&path).unwrap();
        let mut loaded = RoutingStats::load(&path).unwrap();
        assert_eq!(loaded, s);
        loaded.merge(&s);
        assert_eq!(loaded.count(ExpertKey::new(0, 2)), 2);
    }
}
