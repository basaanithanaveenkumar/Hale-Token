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
mod expert_key;
mod expert_provider;
mod expert_source;
mod lru;
mod pack;
pub mod planner;
mod stats;

pub use cache::{CachePolicy, CacheStats, ExpertCache};
pub use checkpoint_source::CheckpointExpertSource;
pub use expert_key::ExpertKey;
pub use expert_provider::ExpertProvider;
pub use expert_source::ExpertSource;
pub use lru::LruCache;
pub use pack::{PackExpertSource, PackHeader, PackWriter, PACK_FILE_NAME};
pub use stats::RoutingStats;

use crate::ops::silu;
use crate::tensor::WeightMatrix;

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
