//! The Hale expert pack: every routed expert in one SSD-friendly file.
//!
//! # Layout
//!
//! ```text
//! offset 0       "HALEPACK" magic (8 bytes)
//! offset 8       u64 LE: length of the JSON header
//! offset 16      JSON PackHeader
//! ...            zero padding up to `data_offset` (a multiple of ALIGN)
//! data_offset    record (moe_layer 0, expert 0)
//!                record (moe_layer 0, expert 1)
//!                ...
//! ```
//!
//! Each record is `gate | up | down` encoded in the pack's dtype, padded to
//! `record_stride` (a multiple of 16 KiB, the Apple Silicon page size).
//!
//! # Why this layout is fast
//!
//! * **One read per expert.** All three matrices are contiguous, so a cache
//!   miss is a single `pread` - the SSD sees large sequential requests.
//! * **O(1) addressing.** Every record has the same size, so the offset is
//!   `data_offset + (ordinal * num_experts + expert) * stride`; no index to
//!   search.
//! * **No double caching.** On macOS the file is opened with `F_NOCACHE`,
//!   so experts are not also kept in the OS page cache; our own cache
//!   decides what stays in unified memory.
//! * **Parallel reads.** `pread` takes an explicit offset and no shared file
//!   cursor, so several threads can read different experts at once and keep
//!   the NVMe queue busy.

use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::{Expert, ExpertKey, ExpertSource};
use crate::config::ModelConfig;
use crate::error::{HaleError, Result};
use crate::tensor::{ByteBuf, DType, WeightMatrix};

/// File name of the pack inside a converted model directory.
pub const PACK_FILE_NAME: &str = "experts.hpk";

const MAGIC: &[u8; 8] = b"HALEPACK";
const FORMAT_VERSION: u32 = 1;
/// Apple Silicon uses 16 KiB pages; aligning records to them keeps every
/// read page-aligned on macOS (and is also a multiple of 4 KiB elsewhere).
pub const ALIGN: usize = 16 * 1024;

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
        header.data_offset = round_up(16 + header.to_json().len() + 32, ALIGN);
        Ok(header)
    }

    fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("header serialises")
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
                HaleError::InvalidArgument(format!(
                    "layer {} has no experts in the pack",
                    key.layer
                ))
            })?;
        if key.expert as usize >= self.num_experts {
            return Err(HaleError::InvalidArgument(format!(
                "expert {key} out of range"
            )));
        }
        let index = ordinal * self.num_experts + key.expert as usize;
        Ok((self.data_offset + index * self.record_stride) as u64)
    }

    /// Splits one record's bytes into the expert's three matrices (no copy).
    fn decode_record(&self, record: ByteBuf) -> Result<Expert> {
        let (h, i, d) = (self.hidden_size, self.intermediate_size, self.dtype);
        let proj_bytes = i * d.row_bytes(h)?;
        Ok(Expert {
            gate: WeightMatrix::new(i, h, d, record.slice(0, proj_bytes))?,
            up: WeightMatrix::new(i, h, d, record.slice(proj_bytes, proj_bytes))?,
            down: WeightMatrix::new(h, i, d, record.slice(2 * proj_bytes, h * d.row_bytes(i)?))?,
        })
    }
}

fn round_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

/// Writes experts into a new pack file. Experts may be written in any order
/// and from several threads, because each goes to a fixed offset.
pub struct PackWriter {
    file: File,
    path: PathBuf,
    header: PackHeader,
}

impl PackWriter {
    /// Creates `path` and writes the header.
    pub fn create(path: &Path, header: PackHeader) -> Result<Self> {
        let io = |e| HaleError::io(path, e);
        let mut file = File::create(path).map_err(io)?;
        let json = header.to_json();
        file.write_all(MAGIC).map_err(io)?;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .map_err(io)?;
        file.write_all(&json).map_err(io)?;
        file.set_len(header.file_size() as u64).map_err(io)?;
        Ok(PackWriter {
            file,
            path: path.to_path_buf(),
            header,
        })
    }

    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    /// Writes one expert, converting it to the pack dtype if needed.
    pub fn write_expert(&self, key: ExpertKey, expert: &Expert) -> Result<()> {
        let d = self.header.dtype;
        let mut record = Vec::with_capacity(self.header.record_bytes);
        for m in [&expert.gate, &expert.up, &expert.down] {
            let converted = m.convert(d)?;
            record.extend_from_slice(converted.bytes().as_bytes());
        }
        if record.len() != self.header.record_bytes {
            return Err(HaleError::bad_tensor(
                key.to_string(),
                format!(
                    "record is {} bytes, expected {}",
                    record.len(),
                    self.header.record_bytes
                ),
            ));
        }
        let offset = self.header.offset_of(key)?;
        self.file
            .write_all_at(&record, offset)
            .map_err(|e| HaleError::io(&self.path, e))
    }

    /// Flushes the file to disk.
    pub fn finish(self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| HaleError::io(&self.path, e))
    }
}

/// Reads experts from a pack with positioned reads (tier 2 of the cache).
pub struct PackExpertSource {
    file: File,
    path: PathBuf,
    header: PackHeader,
}

impl PackExpertSource {
    /// Opens a pack and validates its header.
    pub fn open(path: &Path) -> Result<Self> {
        let io = |e| HaleError::io(path, e);
        let mut file = File::open(path).map_err(io)?;
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix).map_err(io)?;
        if &prefix[..8] != MAGIC {
            return Err(HaleError::InvalidArgument(format!(
                "{} is not a Hale expert pack",
                path.display()
            )));
        }
        let json_len = u64::from_le_bytes(prefix[8..].try_into().expect("8 bytes")) as usize;
        if json_len > ALIGN * 64 {
            return Err(HaleError::InvalidArgument(format!(
                "{}: header too large",
                path.display()
            )));
        }
        let mut json = vec![0u8; json_len];
        file.read_exact(&mut json).map_err(io)?;
        let header: PackHeader = serde_json::from_slice(&json).map_err(|e| HaleError::Json {
            path: path.to_path_buf(),
            source: e,
        })?;
        if header.version != FORMAT_VERSION {
            return Err(HaleError::InvalidArgument(format!(
                "{}: pack version {} is not supported (expected {FORMAT_VERSION})",
                path.display(),
                header.version
            )));
        }
        let actual = file.metadata().map_err(io)?.len() as usize;
        if actual < header.file_size() {
            return Err(HaleError::InvalidArgument(format!(
                "{}: truncated pack ({actual} bytes, expected {})",
                path.display(),
                header.file_size()
            )));
        }
        disable_os_cache(&file);
        Ok(PackExpertSource {
            file,
            path: path.to_path_buf(),
            header,
        })
    }

    pub fn header(&self) -> &PackHeader {
        &self.header
    }
}

impl ExpertSource for PackExpertSource {
    fn load(&self, key: ExpertKey) -> Result<Expert> {
        let offset = self.header.offset_of(key)?;
        let mut record = vec![0u8; self.header.record_bytes];
        self.file
            .read_exact_at(&mut record, offset)
            .map_err(|e| HaleError::io(&self.path, e))?;
        self.header.decode_record(ByteBuf::owned(record))
    }

    fn expert_bytes(&self) -> usize {
        self.header.record_bytes
    }

    fn loads_into_ram(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        format!("SSD pack {} ({})", self.path.display(), self.header.dtype)
    }
}

/// Asks macOS not to keep this file's pages in the unified buffer cache.
#[cfg(target_os = "macos")]
fn disable_os_cache(file: &File) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: fcntl on a valid, owned descriptor; failure is harmless (we
    // just fall back to cached reads), so the return value is ignored.
    unsafe {
        libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
    }
}

/// Other platforms keep the default page-cache behaviour.
#[cfg(not(target_os = "macos"))]
fn disable_os_cache(_file: &File) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> ModelConfig {
        ModelConfig::from_json_str(
            r#"{"architectures":["MixtralForCausalLM"],"hidden_size":64,"intermediate_size":32,
                "num_hidden_layers":2,"num_attention_heads":4,"vocab_size":10,
                "num_local_experts":3,"num_experts_per_tok":2}"#,
        )
        .unwrap()
    }

    fn expert(seed: f32) -> Expert {
        let m = |r, c, s: f32| {
            let v: Vec<f32> = (0..r * c).map(|i| ((i as f32 + s) * 0.37).sin()).collect();
            WeightMatrix::from_f32(r, c, DType::F32, &v).unwrap()
        };
        Expert {
            gate: m(32, 64, seed),
            up: m(32, 64, seed + 1.0),
            down: m(64, 32, seed + 2.0),
        }
    }

    #[test]
    fn write_then_read_every_expert_in_any_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PACK_FILE_NAME);
        let header = PackHeader::for_model(&tiny_config(), DType::F32).unwrap();
        assert_eq!(header.record_stride % ALIGN, 0);
        assert_eq!(header.data_offset % ALIGN, 0);

        let writer = PackWriter::create(&path, header).unwrap();
        for key in [
            ExpertKey::new(1, 2),
            ExpertKey::new(0, 0),
            ExpertKey::new(1, 0),
        ] {
            writer
                .write_expert(key, &expert(key.layer as f32 * 10.0 + key.expert as f32))
                .unwrap();
        }
        writer.finish().unwrap();

        let pack = PackExpertSource::open(&path).unwrap();
        let got = pack.load(ExpertKey::new(1, 2)).unwrap();
        assert_eq!(got.down.to_f32(), expert(12.0).down.to_f32());
        assert_eq!(pack.expert_bytes(), got.size_bytes());
        assert!(
            pack.load(ExpertKey::new(0, 3)).is_err(),
            "expert out of range"
        );
    }

    #[test]
    fn quantized_pack_is_smaller_and_rejects_garbage() {
        let c = tiny_config();
        let f32_pack = PackHeader::for_model(&c, DType::F32).unwrap();
        let q4_pack = PackHeader::for_model(&c, DType::Q4_0).unwrap();
        assert!(q4_pack.record_bytes * 7 < f32_pack.record_bytes);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("junk.hpk");
        std::fs::write(&path, b"definitely not a pack").unwrap();
        assert!(PackExpertSource::open(&path).is_err());
    }
}
