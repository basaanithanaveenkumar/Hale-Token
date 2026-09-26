//! The transformer: attention, MoE feed-forward, and the full forward pass.
//!
//! ```text
//! token ─► embedding ─► [ DecoderLayer x N ] ─► RMSNorm ─► lm_head ─► logits
//!
//! DecoderLayer:
//!   x = x + Attention( RMSNorm(x) )          (GQA + RoPE, KV cache)
//!   x = x + FeedForward( RMSNorm(x) )        (MoE router + top-k experts,
//!                                             or a dense SwiGLU MLP)
//! ```

pub mod arch;
mod attention;
mod kv_cache;
mod moe;
mod transformer;

pub use attention::Attention;
pub use kv_cache::KvCache;
pub use moe::{MoeLayer, Route};
pub use transformer::{DecoderLayer, FeedForward, Transformer};
