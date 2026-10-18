//! The model quantities the planner needs.

use crate::config::ModelConfig;
use serde::Serialize;

/// The sizes that matter for planning, independent of the architecture.
#[derive(Debug, Clone, Serialize)]
pub struct ModelShape {
    /// Model name.
    pub name: String,
    /// Parameters that are not routed experts.
    pub dense_params: f64,
    /// Parameters in one routed expert.
    pub params_per_expert: f64,
    /// Number of layers that contain routed experts.
    pub moe_layers: usize,
    /// Routed experts per MoE layer.
    pub experts_per_layer: usize,
    /// Experts activated per token (top-k).
    pub experts_per_token: usize,
    /// Whether Hale-Token can execute this architecture today.
    pub runnable: bool,
}

impl ModelShape {
    /// Shape of a model we can load.
    pub fn from_config(name: &str, c: &ModelConfig) -> Self {
        ModelShape {
            name: name.to_string(),
            dense_params: c.dense_params() as f64,
            params_per_expert: c.params_per_expert() as f64,
            moe_layers: c.num_moe_layers(),
            experts_per_layer: c.num_experts,
            experts_per_token: c.experts_per_token,
            runnable: true,
        }
    }

    /// Total number of routed experts across all layers.
    pub fn total_experts(&self) -> usize {
        self.moe_layers * self.experts_per_layer
    }
    /// Total parameter count.
    pub fn total_params(&self) -> f64 {
        self.dense_params + self.total_experts() as f64 * self.params_per_expert
    }
    /// Parameters touched per token.
    pub fn active_params(&self) -> f64 {
        self.dense_params
            + (self.moe_layers * self.experts_per_token) as f64 * self.params_per_expert
    }

    /// Well-known MoE models for comparison. Figures derive from each
    /// model's published `config.json`.
    pub fn presets() -> Vec<ModelShape> {
        let shape =
            |name: &str, dense: f64, per_expert: f64, layers, experts, k, runnable| ModelShape {
                name: name.to_string(),
                dense_params: dense,
                params_per_expert: per_expert,
                moe_layers: layers,
                experts_per_layer: experts,
                experts_per_token: k,
                runnable,
            };
        vec![
            // hidden 2048, expert inter 768, 48 layers x 128 experts, top-8
            shape(
                "Qwen3-30B-A3B",
                1.53e9,
                3.0 * 2048.0 * 768.0,
                48,
                128,
                8,
                true,
            ),
            // hidden 4096, expert inter 1536, 94 layers x 128 experts, top-8
            shape(
                "Qwen3-235B-A22B",
                7.98e9,
                3.0 * 4096.0 * 1536.0,
                94,
                128,
                8,
                true,
            ),
            // hidden 4096, inter 14336, 32 layers x 8 experts, top-2
            shape(
                "Mixtral-8x7B",
                1.60e9,
                3.0 * 4096.0 * 14336.0,
                32,
                8,
                2,
                true,
            ),
            // hidden 6144, inter 16384, 56 layers x 8 experts, top-2
            shape(
                "Mixtral-8x22B",
                5.34e9,
                3.0 * 6144.0 * 16384.0,
                56,
                8,
                2,
                true,
            ),
            // hidden 7168, expert inter 2048, 58 MoE layers x 256 experts, top-8
            shape(
                "DeepSeek-V3 (671B)",
                17.1e9,
                3.0 * 7168.0 * 2048.0,
                58,
                256,
                8,
                false,
            ),
            // hidden 7168, expert inter 2048, 60 MoE layers x 384 experts, top-8
            shape(
                "Kimi-K2 (1T)",
                11.5e9,
                3.0 * 7168.0 * 2048.0,
                60,
                384,
                8,
                false,
            ),
        ]
    }

    /// Finds a preset by case-insensitive prefix, e.g. `"qwen3-235b"`.
    pub fn preset(name: &str) -> Option<ModelShape> {
        let wanted = name.to_ascii_lowercase();
        Self::presets()
            .into_iter()
            .find(|p| p.name.to_ascii_lowercase().starts_with(&wanted))
    }
}
