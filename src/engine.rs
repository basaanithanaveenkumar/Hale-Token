//! [`Engine`]: the one-stop API that wires everything together (Facade).
//!
//! ```no_run
//! use hale::engine::{Engine, EngineOptions};
//! use hale::generate::{GenerationOptions, SamplingParams};
//! use hale::tokenizer::PromptFormat;
//!
//! let engine = Engine::load("models/Qwen3-30B-A3B-q4".as_ref(), EngineOptions::default())?;
//! let options = GenerationOptions { max_new_tokens: 64, sampling: SamplingParams::default(), stop_tokens: vec![] };
//! let output = engine.generate_text("Why is the sky blue?", PromptFormat::ChatMl, &options, |piece| print!("{piece}"))?;
//! println!("\n{:.1} tok/s", output.stats.decode_tokens_per_second());
//! # Ok::<(), hale::HaleError>(())
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::config::ModelConfig;
use crate::error::{HaleError, Result};
use crate::expert::planner::{self, HardwareProfile, ModelShape, Plan, PlanRequest};
use crate::expert::{
    CachePolicy, CheckpointExpertSource, ExpertCache, ExpertKey, ExpertSource, PackExpertSource,
    RoutingStats, PACK_FILE_NAME,
};
use crate::generate::{generate, GenerationOptions, GenerationStats};
use crate::hardware;
use crate::loader::{SafetensorsCheckpoint, TensorSource};
use crate::model::Transformer;
use crate::tokenizer::{PromptFormat, StreamDecoder, Tokenizer};

/// File (inside the model directory) where routing counters persist.
pub const ROUTING_STATS_FILE: &str = "hale-routing.json";

/// Knobs for loading a model.
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// RAM for experts in bytes. `None` = let the planner decide from the
    /// machine's memory.
    pub expert_ram_bytes: Option<usize>,
    /// Share of total RAM the planner may use.
    pub ram_fraction: f64,
    /// Where routing counters are loaded from / saved to.
    /// `None` = `<model>/hale-routing.json`.
    pub routing_stats_path: Option<PathBuf>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            expert_ram_bytes: None,
            ram_fraction: planner::DEFAULT_RAM_FRACTION,
            routing_stats_path: None,
        }
    }
}

/// Result of [`Engine::generate_text`].
#[derive(Debug, Clone)]
pub struct TextOutput {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stats: GenerationStats,
}

/// A loaded model, its tokenizer and its expert cache.
pub struct Engine {
    model: Transformer,
    tokenizer: Option<Tokenizer>,
    plan: Plan,
    source_description: String,
    expert_bytes: usize,
    stats_path: PathBuf,
}

impl Engine {
    /// Loads a Hugging Face checkpoint or a `hale convert` output directory.
    pub fn load(dir: &Path, options: EngineOptions) -> Result<Self> {
        let config = ModelConfig::from_dir(dir)?;
        let tensors = Arc::new(SafetensorsCheckpoint::open_dir(dir)?);

        let source: Arc<dyn ExpertSource> = if dir.join(PACK_FILE_NAME).exists() {
            Arc::new(PackExpertSource::open(&dir.join(PACK_FILE_NAME))?)
        } else {
            Arc::new(CheckpointExpertSource::new(
                tensors.clone() as Arc<dyn TensorSource>,
                &config,
            )?)
        };

        let stats_path = options
            .routing_stats_path
            .clone()
            .unwrap_or_else(|| dir.join(ROUTING_STATS_FILE));
        let routing = load_routing_stats(&stats_path, &config);
        let have_stats = routing.total() > 0;

        let plan = make_plan(&config, &*tensors, source.as_ref(), &options, have_stats)?;
        let policy = cache_policy(&config, source.as_ref(), &plan, &options, &routing);
        let cache = ExpertCache::new(source.clone(), routing, policy)?;

        let model = Transformer::load(config, &*tensors, Arc::new(cache))?;
        let tokenizer = Tokenizer::from_dir(dir).ok();
        Ok(Engine {
            model,
            tokenizer,
            plan,
            source_description: source.describe(),
            expert_bytes: source.expert_bytes(),
            stats_path,
        })
    }

    pub fn model(&self) -> &Transformer {
        &self.model
    }
    pub fn config(&self) -> &ModelConfig {
        &self.model.config
    }
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }
    /// The placement plan the cache was built from.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    /// Where experts are read from.
    pub fn source_description(&self) -> &str {
        &self.source_description
    }

    /// Bytes of one routed expert as stored by the expert source.
    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }

    /// Fraction of all experts the RAM tiers can hold - the hit-rate floor
    /// under uniform routing.
    pub fn ram_expert_fraction(&self) -> f64 {
        let c = self.config();
        let total = (c.num_moe_layers() * c.num_experts).max(1);
        let s = self.model.experts().stats();
        let slots = s.pinned_experts + s.lru_capacity_bytes / self.expert_bytes.max(1);
        (slots as f64 / total as f64).min(1.0)
    }

    /// Tokenizes `prompt` (wrapped in `format`), generates, and streams text
    /// pieces to `on_text` as they are decoded.
    pub fn generate_text(
        &self,
        prompt: &str,
        format: PromptFormat,
        options: &GenerationOptions,
        mut on_text: impl FnMut(&str),
    ) -> Result<TextOutput> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| HaleError::Tokenizer("model directory has no tokenizer.json".into()))?;
        let prompt_ids = tokenizer.encode(&format.apply(prompt), true)?;

        let mut options = options.clone();
        options
            .stop_tokens
            .extend(self.config().eos_token_ids.iter().copied());
        options.stop_tokens.extend(
            format
                .stop_strings()
                .iter()
                .filter_map(|s| tokenizer.token_id(s)),
        );

        let mut decoder = StreamDecoder::new(tokenizer);
        let mut decode_error = None;
        let (tokens, stats) = generate(&self.model, &prompt_ids, &options, |id| {
            if options.stop_tokens.contains(&id) {
                return false;
            }
            match decoder.push(id) {
                Ok(piece) => {
                    on_text(&piece);
                    true
                }
                Err(e) => {
                    decode_error = Some(e);
                    false
                }
            }
        })?;
        if let Some(e) = decode_error {
            return Err(e);
        }
        let visible: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|t| !options.stop_tokens.contains(t))
            .collect();
        Ok(TextOutput {
            text: tokenizer.decode(&visible)?,
            tokens,
            stats,
        })
    }

    /// Saves routing counters so the next start pins the right experts.
    pub fn save_routing_stats(&self) -> Result<PathBuf> {
        self.model
            .experts()
            .routing_stats()
            .save(&self.stats_path)?;
        Ok(self.stats_path.clone())
    }
}

/// Loads saved counters if they exist and match the model's shape.
fn load_routing_stats(path: &Path, config: &ModelConfig) -> RoutingStats {
    RoutingStats::load(path)
        .ok()
        .filter(|s| s.num_layers() == config.num_layers && s.num_experts() == config.num_experts)
        .unwrap_or_else(|| RoutingStats::new(config.num_layers, config.num_experts))
}

fn make_plan(
    config: &ModelConfig,
    tensors: &dyn TensorSource,
    source: &dyn ExpertSource,
    options: &EngineOptions,
    have_routing_stats: bool,
) -> Result<Plan> {
    let sys = hardware::detect();
    let spec = sys.chip.copied().unwrap_or(hardware::ChipSpec {
        name: "generic",
        memory_bandwidth_gbps: sys.memory_bandwidth_gbps(),
        performance_cores: sys.logical_cpus as u32,
        efficiency_cores: 0,
        max_memory_gb: 0,
    });
    let hardware = HardwareProfile::from_chip(&spec, sys.total_memory_bytes as f64 / 1e9);
    let dense_dtype = tensors
        .matrix(&crate::model::arch::naming_for(config.architecture).embeddings())?
        .dtype();
    let mut expert_dtype = dense_dtype;
    if let Ok(e) = source.load(first_expert(config)) {
        expert_dtype = e.gate.dtype();
    }
    Ok(planner::plan(&PlanRequest {
        model: ModelShape::from_config("model", config),
        hardware,
        dense_dtype,
        expert_dtype,
        ram_fraction: options.ram_fraction,
        have_routing_stats,
    }))
}

fn first_expert(config: &ModelConfig) -> ExpertKey {
    ExpertKey::new(
        (0..config.num_layers)
            .find(|l| config.is_moe_layer(*l))
            .unwrap_or(0),
        0,
    )
}

/// Picks `n` experts to pin: the hottest ones from past runs first, then
/// (with no or too little history) experts spread evenly over the layers,
/// so every layer gets the same share of hits.
fn choose_pinned(
    routing: &RoutingStats,
    all: &[ExpertKey],
    num_experts: usize,
    n: usize,
) -> Vec<ExpertKey> {
    if n >= all.len() {
        return all.to_vec();
    }
    let mut chosen = routing.hottest(n);
    let mut taken: std::collections::HashSet<ExpertKey> = chosen.iter().copied().collect();
    // `all` is layer-major; visiting it expert-major interleaves the layers.
    let layers = all.len() / num_experts.max(1);
    for e in 0..num_experts {
        for l in 0..layers {
            if chosen.len() == n {
                return chosen;
            }
            let key = all[l * num_experts + e];
            if taken.insert(key) {
                chosen.push(key);
            }
        }
    }
    chosen
}

/// Turns the plan (or an explicit RAM budget) into concrete tier contents.
fn cache_policy(
    config: &ModelConfig,
    source: &dyn ExpertSource,
    plan: &Plan,
    options: &EngineOptions,
    routing: &RoutingStats,
) -> CachePolicy {
    let expert_bytes = source.expert_bytes().max(1);
    let budget = options
        .expert_ram_bytes
        .unwrap_or(plan.expert_ram_bytes as usize);
    let all: Vec<ExpertKey> = (0..config.num_layers)
        .filter(|l| config.is_moe_layer(*l))
        .flat_map(|l| (0..config.num_experts).map(move |e| ExpertKey::new(l, e)))
        .collect();
    let per_token = config.num_moe_layers() * config.experts_per_token;
    let split = planner::split_budget(budget / expert_bytes, all.len(), per_token);
    let pinned = choose_pinned(routing, &all, config.num_experts, split.pinned);
    CachePolicy {
        pinned,
        lru_bytes: split.lru * expert_bytes,
    }
}
