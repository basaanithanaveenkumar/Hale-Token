//! Small numeric building blocks of a transformer layer.
//!
//! Each function works on plain `&[f32]` slices so it can be read, tested
//! and benchmarked in isolation.

mod activation;
mod norm;
mod rope;
mod topk;

pub use activation::{argmax, silu, softmax_in_place};
pub use norm::{rms_norm, rms_norm_in_place};
pub use rope::Rope;
pub use topk::{top_k, Scored};
