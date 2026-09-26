//! A 2-D weight matrix and the matrix-vector product that drives inference.

use rayon::prelude::*;

use super::quant::{self, Q8Activations};
use super::{ByteBuf, DType};
use crate::error::{HaleError, Result};

/// Rows handed to one rayon task. Small enough to balance well across the
/// performance and efficiency cores, large enough to amortise scheduling.
const ROWS_PER_TASK: usize = 32;

/// A row-major `rows x cols` matrix stored in any [`DType`].
///
/// Linear layers in PyTorch store `weight` as `[out_features, in_features]`,
/// so `y = W x` is one dot product per row. That is exactly the access
/// pattern we want during decoding: every row is read once, sequentially.
#[derive(Clone, Debug)]
pub struct WeightMatrix {
    rows: usize,
    cols: usize,
    dtype: DType,
    row_bytes: usize,
    data: ByteBuf,
}

impl WeightMatrix {
    /// Wraps already-encoded bytes, checking that the size matches the shape.
    pub fn new(rows: usize, cols: usize, dtype: DType, data: ByteBuf) -> Result<Self> {
        let row_bytes = dtype.row_bytes(cols)?;
        if data.len() != rows * row_bytes {
            return Err(HaleError::InvalidArgument(format!(
                "matrix {rows}x{cols} {dtype} needs {} bytes, got {}",
                rows * row_bytes,
                data.len()
            )));
        }
        Ok(WeightMatrix {
            rows,
            cols,
            dtype,
            row_bytes,
            data,
        })
    }

    /// Encodes a row-major `f32` slice into `dtype`.
    pub fn from_f32(rows: usize, cols: usize, dtype: DType, values: &[f32]) -> Result<Self> {
        if values.len() != rows * cols {
            return Err(HaleError::InvalidArgument(format!(
                "expected {} values for a {rows}x{cols} matrix, got {}",
                rows * cols,
                values.len()
            )));
        }
        let mut bytes = Vec::with_capacity(rows * dtype.row_bytes(cols)?);
        for row in values.chunks_exact(cols) {
            quant::encode_row(dtype, row, &mut bytes);
        }
        Self::new(rows, cols, dtype, ByteBuf::owned(bytes))
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    /// Raw encoded bytes (used when writing the expert pack).
    pub fn bytes(&self) -> &ByteBuf {
        &self.data
    }
    /// Storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Encoded bytes of row `r`.
    #[inline]
    pub fn row_bytes(&self, r: usize) -> &[u8] {
        let start = r * self.row_bytes;
        &self.data.as_bytes()[start..start + self.row_bytes]
    }

    /// Decodes row `r` into `out` (used for embedding lookups).
    pub fn row_to_f32(&self, r: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols);
        quant::decode_row(self.dtype, self.row_bytes(r), out);
    }

    /// Decodes the whole matrix into `f32` (used by the pack converter).
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.rows * self.cols];
        for (r, chunk) in out.chunks_exact_mut(self.cols).enumerate() {
            self.row_to_f32(r, chunk);
        }
        out
    }

    /// Re-encodes this matrix in a different `dtype`.
    pub fn convert(&self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return Ok(self.clone());
        }
        Self::from_f32(self.rows, self.cols, dtype, &self.to_f32())
    }

    /// Computes `out = W x` in parallel across rows.
    ///
    /// # Panics
    /// Panics when `x.len() != cols` or `out.len() != rows`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "matvec input length");
        assert_eq!(out.len(), self.rows, "matvec output length");
        let rows = out.par_iter_mut().with_min_len(ROWS_PER_TASK).enumerate();
        if self.dtype.block_len() > 1 {
            // Quantize the input once; every row reuses it.
            let xq = Q8Activations::quantize(x);
            rows.for_each(|(r, o)| *o = quant::dot_quantized(self.dtype, self.row_bytes(r), &xq));
        } else {
            rows.for_each(|(r, o)| *o = quant::dot(self.dtype, self.row_bytes(r), x));
        }
    }

    /// Single-threaded `out = W x`, for callers that already run in parallel
    /// (e.g. several experts evaluated at once).
    pub fn matvec_serial(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "matvec input length");
        assert_eq!(out.len(), self.rows, "matvec output length");
        if self.dtype.block_len() > 1 {
            let xq = Q8Activations::quantize(x);
            for (r, o) in out.iter_mut().enumerate() {
                *o = quant::dot_quantized(self.dtype, self.row_bytes(r), &xq);
            }
        } else {
            for (r, o) in out.iter_mut().enumerate() {
                *o = quant::dot(self.dtype, self.row_bytes(r), x);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_matches_naive_product() {
        let (rows, cols) = (70, 64);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 / cols as f32) - 0.5).collect();
        let expect: Vec<f32> = w
            .chunks(cols)
            .map(|r| r.iter().zip(&x).map(|(a, b)| a * b).sum())
            .collect();

        let m = WeightMatrix::from_f32(rows, cols, DType::F32, &w).unwrap();
        let mut par = vec![0.0; rows];
        let mut ser = vec![0.0; rows];
        m.matvec(&x, &mut par);
        m.matvec_serial(&x, &mut ser);
        assert_eq!(
            par, ser,
            "parallel and serial kernels must agree bit-for-bit"
        );
        for (a, b) in par.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn rejects_wrong_byte_count() {
        let err = WeightMatrix::new(2, 32, DType::BF16, ByteBuf::owned(vec![0; 10]));
        assert!(err.is_err());
    }

    #[test]
    fn convert_quantizes_and_keeps_shape() {
        let w: Vec<f32> = (0..4 * 64).map(|i| (i as f32).sin()).collect();
        let m = WeightMatrix::from_f32(4, 64, DType::F32, &w).unwrap();
        let q = m.convert(DType::Q8_0).unwrap();
        assert_eq!((q.rows(), q.cols(), q.dtype()), (4, 64, DType::Q8_0));
        assert!(q.size_bytes() < m.size_bytes() / 3);
    }
}
