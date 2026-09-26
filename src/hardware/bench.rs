//! Micro-benchmarks that measure what the planner needs.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::error::{HaleError, Result};
use crate::tensor::{DType, WeightMatrix};

/// Measures achievable read bandwidth of unified memory in GB/s by summing a
/// buffer of `size_bytes` from all cores, `passes` times.
pub fn memory_bandwidth(size_bytes: usize, passes: usize) -> f64 {
    let data = vec![1.0f32; size_bytes / 4];
    let mut best = f64::MAX;
    let mut checksum = 0.0f64;
    for _ in 0..passes.max(1) {
        let start = Instant::now();
        let sum: f64 = data
            .par_chunks(64 * 1024)
            .map(|c| c.iter().map(|v| f64::from(*v)).sum::<f64>())
            .sum();
        best = best.min(start.elapsed().as_secs_f64());
        checksum += sum;
    }
    std::hint::black_box(checksum);
    (data.len() * 4) as f64 / best / 1e9
}

/// Measures random-ish read throughput of `path` in GB/s using `threads`
/// parallel positioned reads of `chunk_bytes` each (like expert loads).
pub fn ssd_read_bandwidth(path: &Path, chunk_bytes: usize, threads: usize) -> Result<f64> {
    let file = File::open(path).map_err(|e| HaleError::io(path, e))?;
    disable_cache(&file);
    let len = file.metadata().map_err(|e| HaleError::io(path, e))?.len() as usize;
    if len < chunk_bytes {
        return Err(HaleError::InvalidArgument(format!(
            "{} is smaller than one {chunk_bytes}-byte chunk",
            path.display()
        )));
    }
    let chunks = len / chunk_bytes;
    // Visit chunks in a scrambled order so read-ahead cannot help.
    let order: Vec<usize> = (0..chunks).map(|i| (i * 7919) % chunks).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| HaleError::InvalidArgument(e.to_string()))?;
    let start = Instant::now();
    let bytes: usize = pool
        .install(|| {
            order
                .par_iter()
                .map_init(
                    || vec![0u8; chunk_bytes],
                    |buf, &i| {
                        file.read_exact_at(buf, (i * chunk_bytes) as u64)
                            .map(|_| chunk_bytes)
                    },
                )
                .collect::<std::io::Result<Vec<usize>>>()
                .map(|v| v.into_iter().sum())
        })
        .map_err(|e| HaleError::io(path, e))?;
    Ok(bytes as f64 / start.elapsed().as_secs_f64() / 1e9)
}

/// Measures matrix-vector throughput in GFLOP/s for a `rows x cols` matrix
/// in `dtype` - the kernel that dominates decoding.
pub fn matvec_gflops(rows: usize, cols: usize, dtype: DType, iterations: usize) -> Result<f64> {
    let values: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 97) as f32 - 48.0) / 97.0)
        .collect();
    let m = WeightMatrix::from_f32(rows, cols, dtype, &values)?;
    let x = vec![0.5f32; cols];
    let mut out = vec![0.0f32; rows];
    m.matvec(&x, &mut out); // warm-up
    let start = Instant::now();
    for _ in 0..iterations.max(1) {
        m.matvec(&x, &mut out);
    }
    std::hint::black_box(&out);
    let flops = 2.0 * (rows * cols) as f64 * iterations.max(1) as f64;
    Ok(flops / start.elapsed().as_secs_f64() / 1e9)
}

#[cfg(target_os = "macos")]
fn disable_cache(file: &File) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: valid descriptor; failure only means reads may be cached.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
}

#[cfg(not(target_os = "macos"))]
fn disable_cache(_file: &File) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmarks_return_positive_numbers() {
        assert!(memory_bandwidth(1 << 20, 2) > 0.0);
        assert!(matvec_gflops(64, 64, DType::Q4_0, 2).unwrap() > 0.0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob");
        std::fs::write(&path, vec![7u8; 1 << 20]).unwrap();
        assert!(ssd_read_bandwidth(&path, 64 * 1024, 2).unwrap() > 0.0);
        assert!(ssd_read_bandwidth(&path, 4 << 20, 2).is_err());
    }
}
