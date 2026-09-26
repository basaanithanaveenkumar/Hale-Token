//! Turning next-token logits into text: sampling and the decode loop.

mod generator;
mod sampler;

pub use generator::{generate, GenerationOptions, GenerationStats};
pub use sampler::{Greedy, Sampler, SamplingParams, TopPSampler};
