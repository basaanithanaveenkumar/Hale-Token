//! The JSON header describing an expert pack layout.

use super::super::{Expert, ExpertKey};
use super::{round_up, ALIGN, FORMAT_VERSION};
use crate::config::ModelConfig;
use crate::error::{HaleTokenError, Result};
use crate::tensor::{ByteBuf, DType, WeightMatrix};
use serde::{Deserialize, Serialize};

/// Self-describing header stored at the start of the pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PackHeader {
    pub version: u32,
    pub dtype: DType,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_experts: usize,
    /// Model layer indices that contain experts, in file order.
    pub moe_layers: Vec<usize>,
    /// Bytes of one record before padding.
    pub record_bytes: usize,
    /// Distance between consecutive records (record_bytes rounded up to ALIGN).
    pub record_stride: usize,
    /// Offset of the first record.
    pub data_offset: usize,
}

impl PackHeader {
    /// Header for packing `config`'s experts in `dtype`.
    pub fn for_model(config: &ModelConfig, dtype: DType) -> Result<Self> {
        let (h, i) = (config.hidden_size, config.expert_intermediate_size);
        let record_bytes = 2 * i * dtype.row_bytes(h)? + h * dtype.row_bytes(i)?;
        let mut header = PackHeader {
            version: FORMAT_VERSION,
            dtype,
            hidden_size: h,
            intermediate_size: i,
            num_experts: config.num_experts,
            moe_layers: (0..config.num_layers)
                .filter(|l| config.is_moe_layer(*l))
                .collect(),
            record_bytes,
            record_stride: round_up(record_bytes, ALIGN),
            data_offset: 0,
        };
        // +32 bytes of slack: writing the real data_offset adds a few digits.
        header.data_offset = round_up(16 + header.to_json()?.len() + 32, ALIGN);
        Ok(header)
    }

    pub(super) fn to_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| HaleTokenError::serialization("expert pack header", e))
    }

    /// Total file size.
    pub fn file_size(&self) -> usize {
        self.data_offset + self.moe_layers.len() * self.num_experts * self.record_stride
    }

    /// Byte offset of `key`'s record, or an error if `key` is not in the pack.
    pub fn offset_of(&self, key: ExpertKey) -> Result<u64> {
        let ordinal = self
            .moe_layers
            .iter()
            .position(|&l| l == key.layer as usize)
            .ok_or_else(|| {
                HaleTokenError::InvalidArgument(format!(
                    "layer {} has no experts in the pack",
                    key.layer
                ))
            })?;
        if key.expert as usize >= self.num_experts {
            return Err(HaleTokenError::InvalidArgument(format!(
                "expert {key} out of range"
            )));
        }
        let index = ordinal * self.num_experts + key.expert as usize;
        Ok((self.data_offset + index * self.record_stride) as u64)
    }

    /// Splits one record's bytes into the expert's three matrices (no copy).
    pub(super) fn decode_record(&self, record: ByteBuf) -> Result<Expert> {
        let (h, i, d) = (self.hidden_size, self.intermediate_size, self.dtype);
        let proj_bytes = i * d.row_bytes(h)?;
        Ok(Expert {
            gate: WeightMatrix::new(i, h, d, record.slice(0, proj_bytes))?,
            up: WeightMatrix::new(i, h, d, record.slice(proj_bytes, proj_bytes))?,
            down: WeightMatrix::new(h, i, d, record.slice(2 * proj_bytes, h * d.row_bytes(i)?))?,
        })
    }
}
