//! One pre-norm decoder layer: attention + feed-forward.

use super::super::Attention;
use super::FeedForward;

/// Attention + feed-forward with pre-normalisation and residuals.
pub struct DecoderLayer {
    pub input_norm: Vec<f32>,
    pub attention: Attention,
    pub post_attention_norm: Vec<f32>,
    pub feed_forward: FeedForward,
}
