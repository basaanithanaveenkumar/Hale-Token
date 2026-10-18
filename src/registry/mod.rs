//! Name-keyed registries for implementations chosen from configuration.
//!
//! A [`Registry`] maps a string such as `"MixtralForCausalLM"` or `"greedy"`
//! to a factory, so supporting a new implementation means registering one
//! more entry instead of editing a `match` (Open/Closed principle).

mod registry_map;

pub use self::registry_map::Registry;
