//! The per-layer MLP: dense SwiGLU or routed MoE.

use super::super::MoeLayer;
use crate::error::Result;
use crate::expert::Expert;

/// The feed-forward half of a decoder layer.
pub enum FeedForward {
    /// A plain SwiGLU MLP that every token uses.
    Dense(Expert),
    /// A routed Mixture-of-Experts block.
    Moe(MoeLayer),
}

impl FeedForward {
    pub(super) fn forward(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            FeedForward::Dense(mlp) => {
                mlp.forward(x, out);
                Ok(())
            }
            FeedForward::Moe(moe) => moe.forward(x, out),
        }
    }
}
