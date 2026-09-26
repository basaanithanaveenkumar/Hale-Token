//! Weight storage and the numeric kernels that read it.
//!
//! * [`DType`]        - which encoding a row uses (f32, bf16, q8_0, ...).
//! * [`ByteBuf`]      - owned or memory-mapped bytes, cheap to clone.
//! * [`WeightMatrix`] - a row-major matrix plus the parallel `matvec`.
//! * [`quant`]        - per-format encode/decode/dot kernels.

mod buffer;
mod dtype;
mod matrix;
pub mod quant;

pub use buffer::ByteBuf;
pub use dtype::{DType, QK};
pub use matrix::WeightMatrix;
