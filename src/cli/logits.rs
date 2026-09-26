//! `hale logits`: dump raw model outputs so they can be compared with other
//! implementations (see `scripts/reference/compare_with_transformers.py`).

use std::path::PathBuf;

use clap::Args;
use hale::engine::{Engine, EngineOptions};
use hale::HaleError;

#[derive(Args)]
pub struct LogitsArgs {
    /// Model directory.
    model: PathBuf,
    /// Comma-separated token ids, e.g. "1,415,5565".
    #[arg(long, default_value = "")]
    tokens: String,
    /// Text to tokenize instead of --tokens (uses the model's tokenizer).
    #[arg(long, conflicts_with = "tokens")]
    text: Option<String>,
    /// RAM for experts in GB (default: planned).
    #[arg(long)]
    expert_ram_gb: Option<f64>,
}

pub fn execute(args: LogitsArgs) -> hale::Result<()> {
    let options = EngineOptions {
        expert_ram_bytes: args.expert_ram_gb.map(|gb| (gb * 1e9) as usize),
        ..EngineOptions::default()
    };
    let engine = Engine::load(&args.model, options)?;
    let tokens: Vec<u32> = match &args.text {
        Some(text) => engine
            .tokenizer()
            .ok_or_else(|| HaleError::Tokenizer("no tokenizer.json".into()))?
            .encode(text, true)?,
        None => args
            .tokens
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|e| HaleError::InvalidArgument(format!("--tokens: {e}")))?,
    };
    let model = engine.model();
    let mut cache = model.new_cache(tokens.len());
    let mut logits = Vec::with_capacity(tokens.len());
    for &t in &tokens {
        logits.push(model.forward(t, &mut cache)?);
    }
    let json = serde_json::json!({ "tokens": tokens, "logits": logits });
    println!("{json}");
    Ok(())
}
