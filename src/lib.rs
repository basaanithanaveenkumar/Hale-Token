//! # Hale-Token
//!
//! Run data-center scale Mixture-of-Experts (MoE) language models on Apple
//! Silicon Macs (M1 - M5) by keeping the hot experts in unified memory and
//! streaming the rest from the SSD.
//!
//! ## Crate map
//!
//! | Module         | Responsibility                                             |
//! |----------------|------------------------------------------------------------|
//! | [`config`]     | Parse `config.json` into a family-independent [`ModelConfig`] |
//! | [`tensor`]     | Weight storage formats and the mat-vec kernels            |
//! | [`ops`]        | RMSNorm, RoPE, softmax, top-k                             |
//! | [`loader`]     | Zero-copy safetensors reader/writer                       |
//! | [`expert`]     | Tiered expert cache (RAM pinned / RAM LRU / SSD) + planner |
//! | [`model`]      | Attention, MoE layer, transformer forward pass            |
//! | [`generate`]   | Samplers and the decode loop                              |
//! | [`tokenizer`]  | Text <-> ids, chat prompt formats                         |
//! | [`hardware`]   | Apple chip detection and bandwidth benchmarks             |
//! | [`convert`]    | `hale convert`: checkpoint -> quantized expert pack       |
//! | [`engine`]     | [`Engine`]: loads everything and generates text           |
//!
//! Start reading at [`engine::Engine::load`]; it wires the other modules.

pub mod config;
pub mod convert;
pub mod engine;
pub mod error;
pub mod expert;
pub mod generate;
pub mod hardware;
pub mod loader;
pub mod model;
pub mod ops;
pub mod tensor;
pub mod tokenizer;

pub use config::{Architecture, ModelConfig};
pub use engine::{Engine, EngineOptions};
pub use error::{HaleError, Result};
