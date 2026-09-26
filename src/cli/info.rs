//! `hale info` and `hale sysinfo`.

use std::path::Path;

use hale::expert::{PackExpertSource, PACK_FILE_NAME};
use hale::ModelConfig;

use super::format;

pub fn execute(dir: &Path) -> hale::Result<()> {
    let c = ModelConfig::from_dir(dir)?;
    println!("model          {}", dir.display());
    println!("architecture   {:?}", c.architecture);
    println!(
        "parameters     {} total, {} active per token",
        format::params(c.total_params() as f64),
        format::params(c.active_params() as f64)
    );
    println!(
        "layers         {} ({} MoE, {} dense)",
        c.num_layers,
        c.num_moe_layers(),
        c.dense_layers.len()
    );
    println!(
        "attention      {} query heads, {} KV heads, head dim {}",
        c.num_heads, c.num_kv_heads, c.head_dim
    );
    println!(
        "experts        {} per layer, top-{} routing, {} params each",
        c.num_experts,
        c.experts_per_token,
        format::params(c.params_per_expert() as f64)
    );
    println!("vocabulary     {}", c.vocab_size);
    println!("context        {} tokens", c.max_position_embeddings);

    let pack = dir.join(PACK_FILE_NAME);
    if pack.exists() {
        let p = PackExpertSource::open(&pack)?;
        let h = p.header();
        println!(
            "expert pack    {} ({}), {} per expert, {} total",
            PACK_FILE_NAME,
            h.dtype,
            format::bytes(h.record_bytes as f64),
            format::bytes(h.file_size() as f64)
        );
    } else {
        println!("expert pack    none (run `hale convert` for SSD streaming + quantization)");
    }
    Ok(())
}

pub fn print_sysinfo() {
    let sys = hale::hardware::detect();
    println!("cpu            {}", sys.summary());
    match sys.chip {
        Some(chip) => println!(
            "apple chip     {} - {} GB/s unified memory, {}P + {}E cores (full config)",
            chip.name, chip.memory_bandwidth_gbps, chip.performance_cores, chip.efficiency_cores
        ),
        None => println!("apple chip     not detected (estimates use generic defaults)"),
    }
    println!("apple silicon  {}", sys.is_apple_silicon);
    println!(
        "memory         {}",
        format::bytes(sys.total_memory_bytes as f64)
    );
}
