//! Element formats a weight matrix can be stored in.

use serde::{Deserialize, Serialize};

use crate::error::{HaleError, Result};

/// Storage format of a weight matrix row.
///
/// Floating-point formats store one value per element. The quantized
/// formats (`Q8_0`, `Q4_0`) group 32 consecutive elements into a *block*
/// that shares one `f16` scale; this is the same layout llama.cpp/GGML uses,
/// so the math is well understood and battle tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 8-bit blocks: `f16` scale + 32 x `i8`  (34 bytes per 32 values, ~8.5 bits/weight).
    #[serde(rename = "q8_0")]
    Q8_0,
    /// 4-bit blocks: `f16` scale + 16 bytes of nibbles (18 bytes per 32 values, ~4.5 bits/weight).
    #[serde(rename = "q4_0")]
    Q4_0,
}

/// Number of elements that share one scale in the quantized formats.
pub const QK: usize = 32;

impl DType {
    /// Elements per storage block (1 for plain floats, [`QK`] for quantized).
    pub fn block_len(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q8_0 | DType::Q4_0 => QK,
        }
    }

    /// Bytes per storage block.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q8_0 => 2 + QK,
            DType::Q4_0 => 2 + QK / 2,
        }
    }

    /// Bytes needed to store `cols` elements (one matrix row).
    ///
    /// Returns an error when a quantized row length is not a multiple of
    /// the block size.
    pub fn row_bytes(self, cols: usize) -> Result<usize> {
        if cols % self.block_len() != 0 {
            return Err(HaleError::InvalidArgument(format!(
                "{self:?} needs row length divisible by {}, got {cols}",
                self.block_len()
            )));
        }
        Ok(cols / self.block_len() * self.block_bytes())
    }

    /// Average storage cost per element, useful for capacity planning.
    pub fn bits_per_weight(self) -> f64 {
        self.block_bytes() as f64 * 8.0 / self.block_len() as f64
    }

    /// Maps a safetensors dtype string (`"BF16"`, `"F32"`, ...) to a [`DType`].
    pub fn from_safetensors(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(DType::F32),
            "F16" => Some(DType::F16),
            "BF16" => Some(DType::BF16),
            _ => None,
        }
    }

    /// The safetensors dtype string, for the float formats safetensors knows.
    pub fn safetensors_name(self) -> Option<&'static str> {
        match self {
            DType::F32 => Some("F32"),
            DType::F16 => Some("F16"),
            DType::BF16 => Some("BF16"),
            DType::Q8_0 | DType::Q4_0 => None,
        }
    }

    /// Parses a user-supplied name such as `q4_0` or `bf16`.
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "f32" => Ok(DType::F32),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "q8_0" | "q8" => Ok(DType::Q8_0),
            "q4_0" | "q4" => Ok(DType::Q4_0),
            other => Err(HaleError::InvalidArgument(format!(
                "unknown dtype `{other}` (expected f32, f16, bf16, q8_0 or q4_0)"
            ))),
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::Q8_0 => "q8_0",
            DType::Q4_0 => "q4_0",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_match_the_block_layout() {
        assert_eq!(DType::F32.row_bytes(64).unwrap(), 256);
        assert_eq!(DType::BF16.row_bytes(64).unwrap(), 128);
        assert_eq!(DType::Q8_0.row_bytes(64).unwrap(), 68);
        assert_eq!(DType::Q4_0.row_bytes(64).unwrap(), 36);
        assert!(DType::Q4_0.row_bytes(33).is_err());
    }

    #[test]
    fn parse_round_trips_display() {
        for d in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q8_0,
            DType::Q4_0,
        ] {
            assert_eq!(DType::parse(&d.to_string()).unwrap(), d);
        }
        assert!((DType::Q4_0.bits_per_weight() - 4.5).abs() < 1e-9);
    }
}
