//! `hale convert`.

use std::path::PathBuf;

use clap::Args;
use hale::tensor::DType;

use super::format;

#[derive(Args)]
pub struct ConvertArgs {
    /// Hugging Face checkpoint directory (config.json + *.safetensors).
    input: PathBuf,
    /// Output directory for the Hale model.
    output: PathBuf,
    /// Expert storage format: q4_0 (smallest), q8_0 (near-lossless), bf16, f16 or f32.
    #[arg(long, default_value = "q4_0")]
    expert_dtype: String,
}

pub fn execute(args: ConvertArgs) -> hale::Result<()> {
    let dtype = DType::parse(&args.expert_dtype)?;
    println!(
        "converting {} -> {} (experts as {dtype})",
        args.input.display(),
        args.output.display()
    );
    let report =
        hale::convert::convert_checkpoint(&args.input, &args.output, dtype, |done, total| {
            eprint!("\r  experts: layer {done}/{total}");
        })?;
    eprintln!();
    println!(
        "done in {:.1}s: {} dense tensors ({}), {} experts ({})",
        report.seconds,
        report.dense_tensors,
        format::bytes(report.dense_bytes as f64),
        report.experts,
        format::bytes(report.pack_bytes as f64)
    );
    Ok(())
}
