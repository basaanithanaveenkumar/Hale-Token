//! Argument parsing and dispatch for the `hale` binary.

mod bench;
mod convert;
mod format;
mod info;
mod logits;
mod plan;
mod run;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Run data-center scale Mixture-of-Experts models on Apple Silicon.
#[derive(Parser)]
#[command(name = "hale", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text from a prompt.
    Run(run::RunArgs),
    /// Convert a Hugging Face checkpoint into a quantized, SSD-optimised Hale model.
    Convert(convert::ConvertArgs),
    /// Show a model's architecture, size and memory needs.
    Info {
        /// Model directory (Hugging Face checkpoint or `hale convert` output).
        model: PathBuf,
    },
    /// Plan RAM/SSD placement and estimate tokens/second for a model on a Mac.
    Plan(plan::PlanArgs),
    /// Measure memory, SSD and kernel throughput on this machine.
    Bench(bench::BenchArgs),
    /// Show the detected chip and memory.
    Sysinfo,
    /// Print next-token logits for a token sequence as JSON (for verification).
    Logits(logits::LogitsArgs),
}

/// Parses `std::env::args` and runs the chosen command.
pub fn run() -> hale::Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run::execute(args),
        Command::Convert(args) => convert::execute(args),
        Command::Info { model } => info::execute(&model),
        Command::Plan(args) => plan::execute(args),
        Command::Bench(args) => bench::execute(args),
        Command::Logits(args) => logits::execute(args),
        Command::Sysinfo => {
            info::print_sysinfo();
            Ok(())
        }
    }
}
