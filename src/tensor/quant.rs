//! Scalar kernels for every [`DType`]: encode, decode and dot product.
//!
//! These are the hottest loops in the engine: decoding a token is dominated
//! by `dot(row, x)` over every weight row that is touched. No `unsafe`
//! intrinsics are used; the loops are shaped so LLVM vectorises them onto
//! Apple Silicon's 128-bit NEON registers:
//!
//! * **Float formats** decode 64 values at a time into a stack buffer (a
//!   shift for bf16, hardware `fcvtl` for f16, nothing for aligned f32) and
//!   then multiply with eight independent accumulators. Floating-point
//!   addition is not associative, so the compiler will not split a single
//!   accumulator into SIMD lanes on its own.
//! * **Quantized formats** first quantize the activation vector `x` to 8-bit
//!   blocks ([`Q8Activations`], once per mat-vec, shared by every row) and
//!   then multiply integers. Integer addition *is* associative, so LLVM
//!   vectorises freely and on M-series chips emits the `sdot` instruction
//!   (four int8 products summed per lane per cycle). This is the same trick
//!   llama.cpp uses; it costs ~0.5% extra error.
//!
//! Quantized layouts follow GGML:
//!
//! ```text
//! Q8_0 block (34 bytes):  [ scale: f16 ][ q0 .. q31 : i8 ]          value = scale * q
//! Q4_0 block (18 bytes):  [ scale: f16 ][ 16 bytes of nibbles ]     value = scale * (nibble - 8)
//!                          byte j holds element j (low nibble) and element j+16 (high nibble)
//! ```

use half::{bf16, f16};

use super::dtype::{DType, QK};

/// Number of parallel accumulators in the dot-product loops.
const LANES: usize = 8;

/// Sums `lanes` into one value (kept separate so every kernel reduces identically).
#[inline(always)]
fn reduce(lanes: [f32; LANES]) -> f32 {
    let a = (lanes[0] + lanes[4]) + (lanes[1] + lanes[5]);
    let b = (lanes[2] + lanes[6]) + (lanes[3] + lanes[7]);
    a + b
}

#[inline(always)]
fn read_f16(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

/// `bf16` is the upper half of an `f32`, so decoding is a shift.
#[inline(always)]
fn bf16_bits_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u32::from(hi) << 24) | (u32::from(lo) << 16))
}

/// Dot product of one encoded row with an `f32` vector.
///
/// `row` must hold exactly `x.len()` elements in `dtype` encoding. For the
/// quantized formats this quantizes `x` on every call; mat-vec code should
/// call [`Q8Activations::quantize`] once and use [`dot_quantized`] instead.
pub fn dot(dtype: DType, row: &[u8], x: &[f32]) -> f32 {
    match dtype {
        DType::F32 => dot_f32(row, x),
        DType::F16 => dot_f16(row, x),
        DType::BF16 => dot_bf16(row, x),
        DType::Q8_0 | DType::Q4_0 => dot_quantized(dtype, row, &Q8Activations::quantize(x)),
    }
}

/// An activation vector quantized to symmetric 8-bit blocks of [`QK`].
#[derive(Debug, Clone)]
pub struct Q8Activations {
    scales: Vec<f32>,
    quants: Vec<i8>,
}

impl Q8Activations {
    /// Quantizes `x` (length must be a multiple of [`QK`]).
    pub fn quantize(x: &[f32]) -> Self {
        assert_eq!(
            x.len() % QK,
            0,
            "activation length must be a multiple of {QK}"
        );
        let mut scales = Vec::with_capacity(x.len() / QK);
        let mut quants = Vec::with_capacity(x.len());
        for block in x.chunks_exact(QK) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales.push(scale);
            quants.extend(block.iter().map(|v| (v * inv).round() as i8));
        }
        Q8Activations { scales, quants }
    }

    fn block(&self, b: usize) -> (f32, &[i8; QK]) {
        let q = self.quants[b * QK..(b + 1) * QK]
            .try_into()
            .expect("QK values");
        (self.scales[b], q)
    }
}

/// Dot product of a quantized row with pre-quantized activations.
///
/// # Panics
/// Panics if `dtype` is not a quantized format.
pub fn dot_quantized(dtype: DType, row: &[u8], x: &Q8Activations) -> f32 {
    match dtype {
        DType::Q8_0 => dot_q8_0(row, x),
        DType::Q4_0 => dot_q4_0(row, x),
        other => panic!("dot_quantized called with float dtype {other}"),
    }
}

fn dot_q8_0(row: &[u8], x: &Q8Activations) -> f32 {
    let mut total = 0.0f32;
    for (b, block) in row.chunks_exact(DType::Q8_0.block_bytes()).enumerate() {
        let (x_scale, xq) = x.block(b);
        let w: &[u8; QK] = block[2..].try_into().expect("QK quants");
        let mut isum = 0i32;
        for j in 0..QK {
            isum += i32::from(w[j] as i8) * i32::from(xq[j]);
        }
        total += read_f16(block) * x_scale * isum as f32;
    }
    total
}

fn dot_q4_0(row: &[u8], x: &Q8Activations) -> f32 {
    const HALF: usize = QK / 2;
    let mut total = 0.0f32;
    for (b, block) in row.chunks_exact(DType::Q4_0.block_bytes()).enumerate() {
        let (x_scale, xq) = x.block(b);
        let nibbles: &[u8; HALF] = block[2..].try_into().expect("QK/2 bytes");
        let mut isum = 0i32;
        for j in 0..HALF {
            let lo = i32::from(nibbles[j] & 0x0F) - 8;
            let hi = i32::from(nibbles[j] >> 4) - 8;
            isum += lo * i32::from(xq[j]) + hi * i32::from(xq[j + HALF]);
        }
        total += read_f16(block) * x_scale * isum as f32;
    }
    total
}

/// Elements decoded per step by the float kernels: small enough to stay in
/// registers/L1 (256 bytes of f32), large enough to amortise the loop.
const DECODE_BLOCK: usize = 64;

/// Dot product of `x` with `f32` values `w` using [`LANES`] accumulators.
#[inline(always)]
fn dot_lanes(w: &[f32], x: &[f32], acc: &mut [f32; LANES]) -> f32 {
    for (ws, xs) in w.chunks_exact(LANES).zip(x.chunks_exact(LANES)) {
        for l in 0..LANES {
            acc[l] += ws[l] * xs[l];
        }
    }
    let done = w.len() / LANES * LANES;
    w[done..].iter().zip(&x[done..]).map(|(a, b)| a * b).sum()
}

/// Shared driver for the float formats: decode a block of the row into a
/// stack buffer, then run the vectorised f32 dot product on it. Two simple
/// loops vectorise far better than one loop that decodes and multiplies.
#[inline(always)]
fn dot_decoded(
    row: &[u8],
    x: &[f32],
    bytes_per_value: usize,
    decode: impl Fn(&[u8], &mut [f32]),
) -> f32 {
    let mut acc = [0.0f32; LANES];
    let mut tail = 0.0f32;
    let mut buf = [0.0f32; DECODE_BLOCK];
    for (wb, xb) in row
        .chunks(DECODE_BLOCK * bytes_per_value)
        .zip(x.chunks(DECODE_BLOCK))
    {
        let w = &mut buf[..xb.len()];
        decode(wb, w);
        tail += dot_lanes(w, xb, &mut acc);
    }
    reduce(acc) + tail
}

fn dot_f32(row: &[u8], x: &[f32]) -> f32 {
    // Zero-copy when the bytes are 4-byte aligned (the normal case).
    if let Ok(w) = bytemuck::try_cast_slice::<u8, f32>(row) {
        let mut acc = [0.0f32; LANES];
        let tail = dot_lanes(w, x, &mut acc);
        return reduce(acc) + tail;
    }
    dot_decoded(row, x, 4, |b, out| decode_row(DType::F32, b, out))
}

fn dot_bf16(row: &[u8], x: &[f32]) -> f32 {
    dot_decoded(row, x, 2, decode_bf16)
}

fn dot_f16(row: &[u8], x: &[f32]) -> f32 {
    dot_decoded(row, x, 2, decode_f16)
}

/// bf16 -> f32 is a 16-bit left shift; on aligned input this compiles to
/// NEON `ushll`/`shll` over eight values at a time.
fn decode_bf16(bytes: &[u8], out: &mut [f32]) {
    match bytemuck::try_cast_slice::<u8, u16>(bytes) {
        Ok(bits) => {
            for (o, &b) in out.iter_mut().zip(bits) {
                *o = f32::from_bits(u32::from(b) << 16);
            }
        }
        Err(_) => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = bf16_bits_to_f32(b[0], b[1]);
            }
        }
    }
}

/// f16 -> f32 via `half`'s slice conversion, which uses the hardware
/// `fcvtl` instruction on Apple Silicon.
fn decode_f16(bytes: &[u8], out: &mut [f32]) {
    use half::slice::HalfFloatSliceExt;
    match bytemuck::try_cast_slice::<u8, f16>(bytes) {
        Ok(halves) => halves.convert_to_f32_slice(out),
        Err(_) => {
            for (o, b) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = read_f16(b);
            }
        }
    }
}

/// Decodes one encoded row into `out` (`out.len()` elements).
pub fn decode_row(dtype: DType, row: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        DType::F16 => decode_f16(row, out),
        DType::BF16 => decode_bf16(row, out),
        DType::Q8_0 => {
            for (os, block) in out
                .chunks_exact_mut(QK)
                .zip(row.chunks_exact(DType::Q8_0.block_bytes()))
            {
                let scale = read_f16(block);
                for (o, &q) in os.iter_mut().zip(&block[2..]) {
                    *o = scale * f32::from(q as i8);
                }
            }
        }
        DType::Q4_0 => {
            for (os, block) in out
                .chunks_exact_mut(QK)
                .zip(row.chunks_exact(DType::Q4_0.block_bytes()))
            {
                let scale = read_f16(block);
                for (j, &byte) in block[2..].iter().enumerate() {
                    os[j] = scale * (f32::from(byte & 0x0F) - 8.0);
                    os[j + QK / 2] = scale * (f32::from(byte >> 4) - 8.0);
                }
            }
        }
    }
}

/// Encodes `values` into `dtype`, appending the bytes to `out`.
///
/// Quantized formats require `values.len()` to be a multiple of [`QK`].
pub fn encode_row(dtype: DType, values: &[f32], out: &mut Vec<u8>) {
    match dtype {
        DType::F32 => values
            .iter()
            .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        DType::F16 => values
            .iter()
            .for_each(|v| out.extend_from_slice(&f16::from_f32(*v).to_le_bytes())),
        DType::BF16 => values
            .iter()
            .for_each(|v| out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes())),
        DType::Q8_0 => values
            .chunks_exact(QK)
            .for_each(|b| encode_block_q8_0(b, out)),
        DType::Q4_0 => values
            .chunks_exact(QK)
            .for_each(|b| encode_block_q4_0(b, out)),
    }
}

/// Symmetric 8-bit: the largest magnitude maps to +/-127.
fn encode_block_q8_0(block: &[f32], out: &mut Vec<u8>) {
    let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
    out.extend(
        block
            .iter()
            .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8),
    );
}

/// 4-bit with offset 8: the value with the largest magnitude maps to -8,
/// which uses the full signed range (GGML's convention).
fn encode_block_q4_0(block: &[f32], out: &mut Vec<u8>) {
    let extreme = block
        .iter()
        .copied()
        .fold(0.0f32, |m, v| if v.abs() > m.abs() { v } else { m });
    let scale = extreme / -8.0;
    let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
    out.extend_from_slice(&f16::from_f32(scale).to_le_bytes());
    let q = |v: f32| ((v * inv + 8.5) as i32).clamp(0, 15) as u8;
    for j in 0..QK / 2 {
        out.push(q(block[j]) | (q(block[j + QK / 2]) << 4));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random values in [-1, 1).
    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn every_dtype_round_trips_within_its_precision() {
        let v = values(96, 7);
        for (dtype, tol) in [
            (DType::F32, 0.0),
            (DType::F16, 1e-3),
            (DType::BF16, 1e-2),
            (DType::Q8_0, 1e-2),
            (DType::Q4_0, 0.15),
        ] {
            let mut bytes = Vec::new();
            encode_row(dtype, &v, &mut bytes);
            assert_eq!(bytes.len(), dtype.row_bytes(v.len()).unwrap());
            let mut back = vec![0.0; v.len()];
            decode_row(dtype, &bytes, &mut back);
            let err = v
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(err <= tol, "{dtype}: max error {err} > {tol}");
        }
    }

    #[test]
    fn dot_equals_dot_of_decoded_row() {
        // Also covers lengths that are not a multiple of LANES for float formats.
        let w = values(64, 3);
        let x = values(64, 11);
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q8_0,
            DType::Q4_0,
        ] {
            let mut bytes = Vec::new();
            encode_row(dtype, &w, &mut bytes);
            let mut decoded = vec![0.0; w.len()];
            decode_row(dtype, &bytes, &mut decoded);
            let expect = naive_dot(&decoded, &x);
            let got = dot(dtype, &bytes, &x);
            // Quantized kernels also quantize `x` to 8 bits (~0.5% error).
            let tol = if dtype.block_len() > 1 {
                2e-2 * expect.abs().max(1.0)
            } else {
                1e-4
            };
            assert!((expect - got).abs() < tol, "{dtype}: {expect} vs {got}");
        }
        let (w, x) = (values(13, 5), values(13, 9));
        let mut bytes = Vec::new();
        encode_row(DType::F32, &w, &mut bytes);
        assert!((dot(DType::F32, &bytes, &x) - naive_dot(&w, &x)).abs() < 1e-5);
    }

    #[test]
    fn unaligned_rows_give_the_same_answer_as_aligned_ones() {
        let (w, x) = (values(100, 4), values(100, 8));
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let mut bytes = Vec::new();
            encode_row(dtype, &w, &mut bytes);
            // Copy into a buffer shifted by one byte to force the fallback path.
            let mut shifted = vec![0u8; bytes.len() + 1];
            shifted[1..].copy_from_slice(&bytes);
            let (aligned, unaligned) = (&bytes[..], &shifted[1..]);
            assert_eq!(
                dot(dtype, aligned, &x),
                dot(dtype, unaligned, &x),
                "{dtype} dot"
            );
            let (mut a, mut b) = (vec![0.0; 100], vec![0.0; 100]);
            decode_row(dtype, aligned, &mut a);
            decode_row(dtype, unaligned, &mut b);
            assert_eq!(a, b, "{dtype} decode");
        }
    }

    #[test]
    fn zero_block_quantizes_to_zero() {
        let zeros = vec![0.0f32; QK];
        for dtype in [DType::Q8_0, DType::Q4_0] {
            let mut bytes = Vec::new();
            encode_row(dtype, &zeros, &mut bytes);
            let mut back = vec![1.0; QK];
            decode_row(dtype, &bytes, &mut back);
            assert!(back.iter().all(|v| *v == 0.0));
        }
    }
}
