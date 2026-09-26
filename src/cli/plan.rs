//! `hale plan`.

use std::path::PathBuf;

use clap::Args;
use hale::expert::planner::{self, HardwareProfile, ModelShape, PlanRequest};
use hale::hardware::{lookup_chip, APPLE_CHIPS};
use hale::tensor::DType;
use hale::{HaleError, ModelConfig};

use super::format;

#[derive(Args)]
pub struct PlanArgs {
    /// Plan for a local model directory...
    #[arg(long, conflicts_with = "preset")]
    model: Option<PathBuf>,
    /// ...or for a well-known model (qwen3-30b, qwen3-235b, mixtral-8x7b,
    /// mixtral-8x22b, deepseek-v3, kimi-k2).
    #[arg(long)]
    preset: Option<String>,
    /// Chip name such as "M4 Max" (default: this machine).
    #[arg(long)]
    chip: Option<String>,
    /// Unified memory in GB (default: this machine, or the chip's maximum).
    #[arg(long)]
    ram_gb: Option<f64>,
    /// Measured SSD read speed in GB/s (see `hale bench ssd`).
    #[arg(long)]
    ssd_gbps: Option<f64>,
    /// Expert storage format.
    #[arg(long, default_value = "q4_0")]
    expert_dtype: String,
    /// Print a table of every preset model on a range of Macs.
    #[arg(long)]
    compare: bool,
}

pub fn execute(args: PlanArgs) -> hale::Result<()> {
    let expert_dtype = DType::parse(&args.expert_dtype)?;
    if args.compare {
        print_comparison(expert_dtype);
        return Ok(());
    }

    let model = match (&args.model, &args.preset) {
        (Some(dir), _) => {
            ModelShape::from_config(&dir.display().to_string(), &ModelConfig::from_dir(dir)?)
        }
        (None, Some(name)) => ModelShape::preset(name)
            .ok_or_else(|| HaleError::InvalidArgument(format!("unknown preset `{name}`")))?,
        (None, None) => {
            return Err(HaleError::InvalidArgument(
                "pass --model, --preset or --compare".into(),
            ))
        }
    };

    let sys = hale::hardware::detect();
    let chip = match &args.chip {
        Some(name) => lookup_chip(name)
            .ok_or_else(|| HaleError::InvalidArgument(format!("unknown chip `{name}`")))?,
        None => sys.chip.unwrap_or(&APPLE_CHIPS[0]),
    };
    let ram_gb = args.ram_gb.unwrap_or_else(|| {
        if args.chip.is_none() && sys.total_memory_bytes > 0 {
            sys.total_memory_bytes as f64 / 1e9
        } else {
            chip.max_memory_gb as f64
        }
    });
    let mut hardware = HardwareProfile::from_chip(chip, ram_gb);
    if let Some(ssd) = args.ssd_gbps {
        hardware.ssd_bandwidth_gbps = ssd;
    }

    let p = planner::plan(&PlanRequest {
        model: model.clone(),
        hardware: hardware.clone(),
        dense_dtype: DType::BF16,
        expert_dtype,
        ram_fraction: planner::DEFAULT_RAM_FRACTION,
        have_routing_stats: true,
    });

    println!(
        "model        {} ({} total, {} active)",
        model.name,
        format::params(model.total_params()),
        format::params(model.active_params())
    );
    println!(
        "machine      {} ({} GB/s memory, {} GB/s SSD)",
        hardware.name, hardware.memory_bandwidth_gbps, hardware.ssd_bandwidth_gbps
    );
    println!(
        "dense        {} (bf16, always in RAM)",
        format::bytes(p.dense_bytes)
    );
    println!(
        "experts      {} x {} = {} ({expert_dtype})",
        model.total_experts(),
        format::bytes(p.expert_bytes),
        format::bytes(p.total_expert_bytes)
    );
    println!(
        "RAM budget   {} ({:.0}% of RAM), {} left for experts",
        format::bytes(p.ram_budget_bytes),
        planner::DEFAULT_RAM_FRACTION * 100.0,
        format::bytes(p.expert_ram_bytes)
    );
    if !p.feasible {
        println!("verdict      does not fit: dense weights alone exceed the RAM budget");
        return Ok(());
    }
    if p.fully_resident {
        println!("placement    every expert resident in unified memory (no SSD traffic)");
    } else {
        println!(
            "placement    {} pinned + {} LRU in RAM, the rest streamed from SSD",
            p.pinned_experts,
            format::bytes(p.lru_bytes)
        );
    }
    println!(
        "hit rate     >= {:.1}% (uniform-routing floor; real routing is skewed and does better)",
        p.expected_hit_rate * 100.0
    );
    println!(
        "estimate     ~{:.1} tokens/s (bound by {})",
        p.est_tokens_per_second, p.bottleneck
    );
    if !model.runnable {
        println!("note         this architecture is planned but not yet executable by hale run");
    }
    Ok(())
}

/// Estimated decode speed of each preset on representative Macs.
fn print_comparison(expert_dtype: DType) {
    let macs = [
        ("M1", 16.0),
        ("M2 Pro", 32.0),
        ("M3 Max", 128.0),
        ("M4 Pro", 64.0),
        ("M4 Max", 128.0),
        ("M5", 32.0),
        ("M2 Ultra", 192.0),
        ("M3 Ultra", 512.0),
    ];
    print!("{:<20} {:>8} {:>8}", "model", "total", "active");
    for (chip, ram) in macs {
        print!(" {:>12}", format!("{chip} {ram:.0}G"));
    }
    println!();
    for model in ModelShape::presets() {
        print!(
            "{:<20} {:>8} {:>8}",
            model.name,
            format::params(model.total_params()),
            format::params(model.active_params())
        );
        for (chip, ram) in macs {
            let hw = HardwareProfile::from_chip(lookup_chip(chip).expect("known chip"), ram);
            let p = planner::plan(&PlanRequest {
                model: model.clone(),
                hardware: hw,
                dense_dtype: DType::BF16,
                expert_dtype,
                ram_fraction: planner::DEFAULT_RAM_FRACTION,
                have_routing_stats: true,
            });
            let cell = if !p.feasible {
                "-".to_string()
            } else if p.fully_resident {
                format!("{:.1} RAM", p.est_tokens_per_second)
            } else {
                format!("{:.1} SSD", p.est_tokens_per_second)
            };
            print!(" {cell:>12}");
        }
        println!();
    }
    println!("\nestimated decode tokens/s, experts as {expert_dtype}; RAM = fully resident, SSD = streamed, - = does not fit");
}
