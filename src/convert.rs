//! `hale convert`: turn a Hugging Face checkpoint into a Hale model directory.
//!
//! ```text
//! <out>/config.json            copied
//! <out>/tokenizer*.json ...    copied
//! <out>/dense.safetensors      every non-expert tensor, original precision
//! <out>/experts.hpk            every routed expert, quantized, page-aligned
//! ```
//!
//! Dense weights stay in full precision because they are small and used by
//! every token; experts are the bulk of the model, so quantizing them to
//! 4 bits cuts both RAM use and SSD traffic by ~3.5x versus bf16.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::config::ModelConfig;
use crate::error::{HaleError, Result};
use crate::expert::{
    CheckpointExpertSource, ExpertKey, ExpertSource, PackHeader, PackWriter, PACK_FILE_NAME,
};
use crate::loader::{write_safetensors, SafetensorsCheckpoint, TensorSource};
use crate::model::arch::{naming_for, MlpProj};
use crate::tensor::DType;

/// Name of the dense-weights file in a converted directory.
pub const DENSE_FILE_NAME: &str = "dense.safetensors";

/// Files copied verbatim from the source directory when present.
const COPIED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
];

/// Summary of a conversion.
#[derive(Debug, Clone)]
pub struct ConvertReport {
    pub dense_tensors: usize,
    pub dense_bytes: usize,
    pub experts: usize,
    pub pack_bytes: usize,
    pub seconds: f64,
}

/// Converts `src` into `dst` with experts stored as `expert_dtype`.
///
/// `progress(done, total)` is called after each MoE layer.
pub fn convert_checkpoint(
    src: &Path,
    dst: &Path,
    expert_dtype: DType,
    progress: impl Fn(usize, usize) + Sync,
) -> Result<ConvertReport> {
    if src == dst {
        return Err(HaleError::InvalidArgument(
            "output directory must differ from the input".into(),
        ));
    }
    let started = Instant::now();
    let config = ModelConfig::from_dir(src)?;
    std::fs::create_dir_all(dst).map_err(|e| HaleError::io(dst, e))?;
    let checkpoint = Arc::new(SafetensorsCheckpoint::open_dir(src)?);
    let naming = naming_for(config.architecture);

    // 1. Everything that is not a routed expert goes to dense.safetensors.
    let mut expert_names = HashSet::new();
    for l in (0..config.num_layers).filter(|l| config.is_moe_layer(*l)) {
        for e in 0..config.num_experts {
            for p in [MlpProj::Gate, MlpProj::Up, MlpProj::Down] {
                expert_names.insert(naming.expert(l, e, p));
            }
        }
    }
    let mut dense = Vec::new();
    for name in checkpoint.tensor_names() {
        if expert_names.contains(name) {
            continue;
        }
        let (info, bytes) = checkpoint.raw(name)?;
        let Some(dtype) = DType::from_safetensors(&info.dtype) else {
            continue; // e.g. integer buffers some exporters include
        };
        dense.push((name.to_string(), info.shape.clone(), dtype, bytes));
    }
    let dense_bytes = dense.iter().map(|t| t.3.len()).sum();
    write_safetensors(&dst.join(DENSE_FILE_NAME), &dense)?;

    // 2. Experts go to the pack, one MoE layer at a time, experts in parallel.
    let source = CheckpointExpertSource::new(checkpoint.clone() as Arc<dyn TensorSource>, &config)?;
    let header = PackHeader::for_model(&config, expert_dtype)?;
    let pack_bytes = header.file_size();
    let writer = PackWriter::create(&dst.join(PACK_FILE_NAME), header)?;
    let layers = writer.header().moe_layers.clone();
    let done = AtomicUsize::new(0);
    for &l in &layers {
        (0..config.num_experts).into_par_iter().try_for_each(|e| {
            let key = ExpertKey::new(l, e);
            writer.write_expert(key, &source.load(key)?)
        })?;
        progress(done.fetch_add(1, Ordering::SeqCst) + 1, layers.len());
    }
    writer.finish()?;

    // 3. Config and tokenizer files.
    for file in COPIED_FILES {
        let from = src.join(file);
        if from.exists() {
            std::fs::copy(&from, dst.join(file)).map_err(|e| HaleError::io(&from, e))?;
        }
    }

    Ok(ConvertReport {
        dense_tensors: dense.len(),
        dense_bytes,
        experts: layers.len() * config.num_experts,
        pack_bytes,
        seconds: started.elapsed().as_secs_f64(),
    })
}
