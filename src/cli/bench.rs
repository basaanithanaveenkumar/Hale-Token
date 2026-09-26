//! `hale bench`.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use hale::engine::{Engine, EngineOptions};
use hale::generate::{generate, GenerationOptions, SamplingParams};
use hale::hardware::bench;
use hale::tensor::DType;

use super::format;

#[derive(Args)]
pub struct BenchArgs {
    #[command(subcommand)]
    what: BenchKind,
}

#[derive(Subcommand)]
enum BenchKind {
    /// Unified-memory read bandwidth.
    Memory {
        /// Buffer size in MB.
        #[arg(long, default_value_t = 1024)]
        size_mb: usize,
    },
    /// SSD read bandwidth using expert-sized parallel reads of FILE.
    Ssd {
        /// A large file on the disk to test (e.g. an experts.hpk).
        file: PathBuf,
        /// Read size in MB (about one expert).
        #[arg(long, default_value_t = 8)]
        chunk_mb: usize,
        /// Parallel readers.
        #[arg(long, default_value_t = 8)]
        threads: usize,
    },
    /// Mat-vec kernel throughput for every storage format.
    Kernels,
    /// End-to-end decode speed of a model (greedy, fixed prompt).
    Model {
        model: PathBuf,
        /// Tokens to generate.
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        /// RAM for experts in GB (default: planned).
        #[arg(long)]
        expert_ram_gb: Option<f64>,
    },
}

pub fn execute(args: BenchArgs) -> hale::Result<()> {
    match args.what {
        BenchKind::Memory { size_mb } => {
            let gbps = bench::memory_bandwidth(size_mb * 1_000_000, 5);
            println!("memory read bandwidth: {gbps:.1} GB/s");
            if let Some(chip) = hale::hardware::detect().chip {
                println!(
                    "advertised for {}: {} GB/s",
                    chip.name, chip.memory_bandwidth_gbps
                );
            }
        }
        BenchKind::Ssd {
            file,
            chunk_mb,
            threads,
        } => {
            let gbps = bench::ssd_read_bandwidth(&file, chunk_mb * 1_000_000, threads)?;
            println!("SSD read bandwidth: {gbps:.2} GB/s ({chunk_mb} MB reads, {threads} threads)");
            println!("pass --ssd-gbps {gbps:.1} to `hale plan` for accurate estimates");
        }
        BenchKind::Kernels => {
            for dtype in [
                DType::F32,
                DType::BF16,
                DType::F16,
                DType::Q8_0,
                DType::Q4_0,
            ] {
                let gflops = bench::matvec_gflops(4096, 4096, dtype, 20)?;
                let gbps = gflops / 2.0 * dtype.bits_per_weight() / 8.0;
                println!("matvec 4096x4096 {dtype:>5}: {gflops:6.1} GFLOP/s  ({gbps:.1} GB/s of weights)");
            }
        }
        BenchKind::Model {
            model,
            tokens,
            expert_ram_gb,
        } => {
            let options = EngineOptions {
                expert_ram_bytes: expert_ram_gb.map(|gb| (gb * 1e9) as usize),
                ..EngineOptions::default()
            };
            let engine = Engine::load(&model, options)?;
            let prompt: Vec<u32> = (1..=16).collect();
            let gen = GenerationOptions {
                max_new_tokens: tokens,
                sampling: SamplingParams::greedy(),
                stop_tokens: Vec::new(),
            };
            let (_, s) = generate(engine.model(), &prompt, &gen, |_| true)?;
            println!(
                "prefill {:.2} tok/s | decode {:.2} tok/s | expert hit rate {:.1}% | loaded {}",
                s.prefill_tokens_per_second(),
                s.decode_tokens_per_second(),
                s.cache.hit_rate() * 100.0,
                format::bytes(s.cache.bytes_loaded as f64)
            );
        }
    }
    Ok(())
}
