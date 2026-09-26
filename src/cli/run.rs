//! `hale run`.

use std::io::Write;
use std::path::PathBuf;

use clap::{Args, ValueEnum};
use hale::engine::{Engine, EngineOptions};
use hale::expert::planner::DEFAULT_RAM_FRACTION;
use hale::generate::{GenerationOptions, SamplingParams};
use hale::tokenizer::PromptFormat;

use super::format;

#[derive(Clone, Copy, ValueEnum)]
enum FormatArg {
    /// Pick the chat format from the model architecture.
    Auto,
    /// No chat markup.
    Raw,
    /// Qwen-style `<|im_start|>` markup.
    Chatml,
    /// Mistral-style `[INST]` markup.
    Mistral,
}

#[derive(Args)]
pub struct RunArgs {
    /// Model directory (Hugging Face checkpoint or `hale convert` output).
    model: PathBuf,
    /// The prompt.
    #[arg(short, long)]
    prompt: String,
    /// Maximum new tokens.
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,
    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// Nucleus sampling threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    /// Keep only the k most likely tokens (0 = off).
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Prompt markup.
    #[arg(long, value_enum, default_value_t = FormatArg::Auto)]
    format: FormatArg,
    /// RAM for experts in GB (default: planned from installed memory).
    #[arg(long)]
    expert_ram_gb: Option<f64>,
    /// Share of installed RAM the planner may use.
    #[arg(long, default_value_t = DEFAULT_RAM_FRACTION)]
    ram_fraction: f64,
    /// Do not update the routing statistics file after the run.
    #[arg(long)]
    no_save_stats: bool,
    /// Print final statistics as JSON (for scripts and CI).
    #[arg(long)]
    json_stats: bool,
}

pub fn execute(args: RunArgs) -> hale::Result<()> {
    let options = EngineOptions {
        expert_ram_bytes: args.expert_ram_gb.map(|gb| (gb * 1e9) as usize),
        ram_fraction: args.ram_fraction,
        routing_stats_path: None,
    };
    let engine = Engine::load(&args.model, options)?;
    let plan = engine.plan();
    eprintln!(
        "[hale] experts from {} | pinned {} | LRU {} | planned hit rate >= {:.0}%",
        engine.source_description(),
        plan.pinned_experts,
        format::bytes(plan.lru_bytes),
        plan.expected_hit_rate * 100.0
    );

    let prompt_format = match args.format {
        FormatArg::Auto => PromptFormat::for_architecture(engine.config().architecture),
        FormatArg::Raw => PromptFormat::Raw,
        FormatArg::Chatml => PromptFormat::ChatMl,
        FormatArg::Mistral => PromptFormat::MistralInstruct,
    };
    let generation = GenerationOptions {
        max_new_tokens: args.max_tokens,
        sampling: SamplingParams {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            seed: args.seed,
        },
        stop_tokens: Vec::new(),
    };

    let mut stdout = std::io::stdout();
    let output = engine.generate_text(&args.prompt, prompt_format, &generation, |piece| {
        print!("{piece}");
        let _ = stdout.flush();
    })?;
    println!();

    let s = &output.stats;
    eprintln!(
        "[hale] prompt {} tok @ {:.1} tok/s | generated {} tok @ {:.2} tok/s | expert hit rate {:.1}% | SSD {} @ {:.2} GB/s",
        s.prompt_tokens,
        s.prefill_tokens_per_second(),
        s.generated_tokens,
        s.decode_tokens_per_second(),
        s.cache.hit_rate() * 100.0,
        format::bytes(s.cache.bytes_loaded as f64),
        s.cache.load_gbps()
    );
    if args.json_stats {
        println!("{}", serde_json::to_string(s).expect("stats serialise"));
    }
    if !args.no_save_stats {
        match engine.save_routing_stats() {
            Ok(path) => eprintln!("[hale] routing statistics saved to {}", path.display()),
            Err(e) => eprintln!("[hale] could not save routing statistics: {e}"),
        }
    }
    Ok(())
}
