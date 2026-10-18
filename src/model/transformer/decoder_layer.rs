//! One pre-norm decoder layer: attention + feed-forward.

use super::super::Attention;
use super::FeedForward;

/// Attention + feed-forward with pre-normalisation and residuals.
pub struct DecoderLayer {
    /// Pre-attention RMSNorm weight.
    pub input_norm: Vec<f32>,
    /// Self-attention block.
    pub attention: Attention,
    /// Pre-MLP RMSNorm weight.
    pub post_attention_norm: Vec<f32>,
    /// Dense or MoE feed-forward block.
    pub feed_forward: FeedForward,
}
