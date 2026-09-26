//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use hale::engine::{Engine, EngineOptions};
use serde::Deserialize;

/// Golden outputs produced by `scripts/reference/make_fixtures.py`.
#[derive(Deserialize)]
pub struct Expected {
    pub prompt: Vec<u32>,
    pub greedy_continuation: Vec<u32>,
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
}

pub const FIXTURES: [&str; 2] = ["tiny-qwen3-moe", "tiny-mixtral"];

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn expected(name: &str) -> Expected {
    let text = std::fs::read_to_string(fixture(name).join("expected.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Loads a model with an explicit expert RAM budget and a private stats file.
pub fn load(dir: &Path, expert_ram_bytes: usize, stats_dir: &Path) -> Engine {
    let options = EngineOptions {
        expert_ram_bytes: Some(expert_ram_bytes),
        routing_stats_path: Some(stats_dir.join("routing.json")),
        ..EngineOptions::default()
    };
    Engine::load(dir, options).unwrap()
}

/// Feeds `tokens` one by one (teacher forcing) and returns every step's logits.
pub fn teacher_forced_logits(engine: &Engine, tokens: &[u32]) -> Vec<Vec<f32>> {
    let model = engine.model();
    let mut cache = model.new_cache(tokens.len());
    tokens
        .iter()
        .map(|&t| model.forward(t, &mut cache).unwrap())
        .collect()
}

/// Largest absolute difference between two logit matrices.
pub fn max_abs_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()))
        .fold(0.0, f32::max)
}
