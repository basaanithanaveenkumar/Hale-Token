//! Mixture-of-Experts weight management - the core of Hale-Token.
//!
//! A frontier MoE model is mostly experts: in Qwen3-235B-A22B, 97% of the
//! weights are routed experts, yet each token only uses 8 of 128 per layer.
//! Keeping *all* experts in RAM is therefore wasteful; keeping *none* makes
//! every token wait on the SSD. Hale-Token places them in tiers:
//!
//! ```text
//!   tier 0  pinned   hottest experts, loaded at start-up, never evicted
//!   tier 1  LRU      recently used experts, evicted least-recently-used first
//!   tier 2  SSD      everything, in a page-aligned pack file read with pread
//! ```
//!
//! On Apple Silicon tiers 0-1 are *unified memory*: the same bytes are
//! visible to CPU and GPU without copies, so "in RAM" is also "on the GPU".
//!
//! The pieces, each with one job (Single Responsibility):
//!
//! * [`Expert`]        - one SwiGLU expert's three matrices and its forward pass.
//! * [`ExpertSource`]  - trait: *cold* storage that can load any expert.
//!   * [`CheckpointExpertSource`] - reads a Hugging Face checkpoint (mmap).
//!   * [`PackExpertSource`]       - reads Hale's SSD-optimised pack file.
//! * [`ExpertProvider`] - trait: what the MoE layer asks for experts.
//!   * [`ExpertCache`]  - the tiered implementation described above.
//! * [`LruCache`]      - O(1) byte-budgeted LRU used by tier 1.
//! * [`RoutingStats`]  - per-expert usage counters, persisted to pick tier 0.
//! * [`planner`]       - decides the tier sizes and predicts tokens/second.

mod cache;
mod checkpoint_source;
mod lru;
mod pack;
pub mod planner;
mod stats;

pub use cache::{CachePolicy, CacheStats, ExpertCache};
pub use checkpoint_source::CheckpointExpertSource;
pub use lru::LruCache;
pub use pack::{PackExpertSource, PackHeader, PackWriter, PACK_FILE_NAME};
pub use stats::RoutingStats;

use std::sync::Arc;

use crate::error::Result;
use crate::ops::silu;
use crate::tensor::WeightMatrix;

/// Identifies one routed expert: `(layer, expert index within the layer)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: usize, expert: usize) -> Self {
        ExpertKey {
            layer: layer as u32,
            expert: expert as u32,
        }
    }
}

impl std::fmt::Display for ExpertKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}E{}", self.layer, self.expert)
    }
}

/// A SwiGLU feed-forward block: `down( silu(gate x) * up x )`.
///
/// Routed experts and the dense MLP layers of some models share this shape,
/// so both are represented by this one type.
#[derive(Debug, Clone)]
pub struct Expert {
    /// `[intermediate, hidden]`
    pub gate: WeightMatrix,
    /// `[intermediate, hidden]`
    pub up: WeightMatrix,
    /// `[hidden, intermediate]`
    pub down: WeightMatrix,
}

impl Expert {
    /// Bytes of weight data held by this expert.
    pub fn size_bytes(&self) -> usize {
        self.gate.size_bytes() + self.up.size_bytes() + self.down.size_bytes()
    }

    /// Computes the expert's output for one token into `out` (`hidden` long).
    pub fn forward(&self, x: &[f32], out: &mut [f32]) {
        let inter = self.gate.rows();
        let mut gate = vec![0.0; inter];
        let mut up = vec![0.0; inter];
        rayon::join(
            || self.gate.matvec(x, &mut gate),
            || self.up.matvec(x, &mut up),
        );
        for (g, u) in gate.iter_mut().zip(&up) {
            *g = silu(*g) * u;
        }
        self.down.matvec(&gate, out);
    }
}

/// Cold storage able to load any expert on demand (tier 2).
///
/// Implementations must be thread-safe: the cache loads missing experts in
/// parallel to keep the SSD's command queue full.
pub trait ExpertSource: Send + Sync {
    /// Loads expert `key`.
    fn load(&self, key: ExpertKey) -> Result<Expert>;

    /// Size in bytes of one loaded expert (all experts are the same size).
    fn expert_bytes(&self) -> usize;

    /// Whether `load` returns owned bytes (true) or zero-copy mmap views
    /// (false). Caching mmap views gains nothing: the OS page cache already
    /// is the cache, so the tiered cache is bypassed for such sources.
    fn loads_into_ram(&self) -> bool;

    /// Short human-readable description for logs.
    fn describe(&self) -> String;
}

/// What the MoE layer depends on to obtain expert weights
/// (Dependency Inversion: the model never knows about tiers or files).
pub trait ExpertProvider: Send + Sync {
    /// Returns the experts for `keys`, in the same order.
    fn fetch(&self, keys: &[ExpertKey]) -> Result<Vec<Arc<Expert>>>;

    /// Cache counters, if the provider keeps any.
    fn stats(&self) -> CacheStats;

    /// Routing counters observed so far.
    fn routing_stats(&self) -> RoutingStats;
}
