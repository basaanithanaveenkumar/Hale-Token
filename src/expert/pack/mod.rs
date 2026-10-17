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

mod pack_expert_source;
mod pack_header;
mod pack_writer;

pub use self::pack_expert_source::PackExpertSource;
pub use self::pack_header::PackHeader;
pub use self::pack_writer::PackWriter;

/// File name of the pack inside a converted model directory.
pub const PACK_FILE_NAME: &str = "experts.hpk";

const MAGIC: &[u8; 8] = b"HALEPACK";

const FORMAT_VERSION: u32 = 1;

/// Apple Silicon uses 16 KiB pages; aligning records to them keeps every
/// read page-aligned on macOS (and is also a multiple of 4 KiB elsewhere).
pub const ALIGN: usize = 16 * 1024;

fn round_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;
    use crate::expert::{Expert, ExpertKey, ExpertSource};
    use crate::tensor::{DType, WeightMatrix};

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
