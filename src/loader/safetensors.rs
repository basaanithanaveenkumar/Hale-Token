//! A minimal, zero-copy safetensors reader and writer.
//!
//! The format is deliberately simple, which is why we parse it ourselves
//! instead of pulling in another dependency:
//!
//! ```text
//! [ u64 little-endian: N ][ N bytes of JSON header ][ raw tensor bytes ... ]
//! header = { "name": { "dtype": "BF16", "shape": [rows, cols],
//!                      "data_offsets": [begin, end] }, ... }
//! ```
//!
//! Offsets are relative to the first byte after the header. Large models are
//! split into several *shards*; `model.safetensors.index.json` says which
//! shard holds which tensor, but since every shard's header lists its own
//! tensors we simply read all headers and build one lookup table.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use super::TensorSource;
use crate::error::{HaleError, Result};
use crate::tensor::{quant, ByteBuf, DType, WeightMatrix};

/// Header entry for one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Where a tensor lives: which shard, and at which absolute byte offset.
#[derive(Debug, Clone)]
struct Location {
    shard: usize,
    info: TensorInfo,
    /// Absolute offset of the tensor's first byte inside the shard file.
    start: usize,
}

/// A set of memory-mapped safetensors shards viewed as one checkpoint.
pub struct SafetensorsCheckpoint {
    maps: Vec<Arc<Mmap>>,
    index: HashMap<String, Location>,
}

impl SafetensorsCheckpoint {
    /// Opens every `*.safetensors` file in `dir`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| HaleError::io(dir, e))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(HaleError::InvalidArgument(format!(
                "no .safetensors files in {}",
                dir.display()
            )));
        }
        Self::open_files(&files)
    }

    /// Opens an explicit list of shard files.
    pub fn open_files(files: &[PathBuf]) -> Result<Self> {
        let mut maps = Vec::with_capacity(files.len());
        let mut index = HashMap::new();
        for (shard, path) in files.iter().enumerate() {
            let file = File::open(path).map_err(|e| HaleError::io(path, e))?;
            // SAFETY: the map is read-only. Mutating a checkpoint while it is
            // being served is unsupported, exactly as with any mmap-based loader.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| HaleError::io(path, e))?;
            for (name, info, start) in parse_header(path, &map)? {
                index.insert(name, Location { shard, info, start });
            }
            maps.push(Arc::new(map));
        }
        Ok(SafetensorsCheckpoint { maps, index })
    }

    /// Names of all tensors, sorted (useful for debugging and conversion).
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.index.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Header information for `name`.
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.location(name).map(|l| &l.info)
    }

    /// Header info and raw bytes of `name`, whatever its dtype.
    pub fn raw(&self, name: &str) -> Result<(&TensorInfo, ByteBuf)> {
        let loc = self.location(name)?;
        let len = loc.info.data_offsets[1] - loc.info.data_offsets[0];
        Ok((
            &loc.info,
            ByteBuf::mapped(Arc::clone(&self.maps[loc.shard]), loc.start, len),
        ))
    }

    fn location(&self, name: &str) -> Result<&Location> {
        self.index
            .get(name)
            .ok_or_else(|| HaleError::MissingTensor(name.to_string()))
    }

    fn bytes_and_dtype(&self, name: &str) -> Result<(ByteBuf, DType, &TensorInfo)> {
        let loc = self.location(name)?;
        let dtype = DType::from_safetensors(&loc.info.dtype).ok_or_else(|| {
            HaleError::bad_tensor(name, format!("unsupported dtype {}", loc.info.dtype))
        })?;
        let len = loc.info.data_offsets[1] - loc.info.data_offsets[0];
        let buf = ByteBuf::mapped(Arc::clone(&self.maps[loc.shard]), loc.start, len);
        Ok((buf, dtype, &loc.info))
    }
}

impl TensorSource for SafetensorsCheckpoint {
    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn matrix(&self, name: &str) -> Result<WeightMatrix> {
        let (buf, dtype, info) = self.bytes_and_dtype(name)?;
        let [rows, cols] = info.shape[..] else {
            return Err(HaleError::bad_tensor(
                name,
                format!("expected 2-D, got shape {:?}", info.shape),
            ));
        };
        WeightMatrix::new(rows, cols, dtype, buf)
            .map_err(|e| HaleError::bad_tensor(name, e.to_string()))
    }

    fn vector(&self, name: &str) -> Result<Vec<f32>> {
        let (buf, dtype, info) = self.bytes_and_dtype(name)?;
        let [n] = info.shape[..] else {
            return Err(HaleError::bad_tensor(
                name,
                format!("expected 1-D, got shape {:?}", info.shape),
            ));
        };
        if buf.len() != n * dtype.block_bytes() {
            return Err(HaleError::bad_tensor(
                name,
                "byte length does not match shape",
            ));
        }
        let mut out = vec![0.0; n];
        quant::decode_row(dtype, buf.as_bytes(), &mut out);
        Ok(out)
    }
}

/// Parses and validates the JSON header of one shard.
fn parse_header(path: &Path, bytes: &[u8]) -> Result<Vec<(String, TensorInfo, usize)>> {
    let bad = |message: &str| HaleError::Safetensors {
        path: path.to_path_buf(),
        message: message.to_string(),
    };
    if bytes.len() < 8 {
        return Err(bad("file shorter than the 8-byte header length"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8usize
        .checked_add(header_len)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| bad("header length exceeds file size"))?;
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|e| HaleError::Json {
            path: path.to_path_buf(),
            source: e,
        })?;

    let mut tensors = Vec::with_capacity(header.len());
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(value).map_err(|e| HaleError::Json {
            path: path.to_path_buf(),
            source: e,
        })?;
        let [begin, end] = info.data_offsets;
        if begin > end || data_start + end > bytes.len() {
            return Err(bad(&format!("tensor `{name}` points outside the file")));
        }
        let start = data_start + begin;
        tensors.push((name, info, start));
    }
    Ok(tensors)
}

/// Writes float tensors to a single safetensors file.
///
/// Used by `hale convert` to store the non-expert ("dense") weights.
/// Tensor data is written in the order given; each entry is
/// `(name, shape, dtype, raw little-endian bytes)`.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, DType, ByteBuf)],
) -> Result<()> {
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for (name, shape, dtype, bytes) in tensors {
        let dtype_name = dtype.safetensors_name().ok_or_else(|| {
            HaleError::InvalidArgument(format!("{name}: {dtype} cannot be stored in safetensors"))
        })?;
        let info = TensorInfo {
            dtype: dtype_name.to_string(),
            shape: shape.clone(),
            data_offsets: [offset, offset + bytes.len()],
        };
        header.insert(name.clone(), info);
        offset += bytes.len();
    }
    let mut json = serde_json::to_vec(&header).expect("header serialises");
    // The spec recommends padding the header to 8 bytes so data is aligned.
    while json.len() % 8 != 0 {
        json.push(b' ');
    }

    let mut file = std::io::BufWriter::new(File::create(path).map_err(|e| HaleError::io(path, e))?);
    let io = |e| HaleError::io(path, e);
    file.write_all(&(json.len() as u64).to_le_bytes())
        .map_err(io)?;
    file.write_all(&json).map_err(io)?;
    for (_, _, _, bytes) in tensors {
        file.write_all(bytes.as_bytes()).map_err(io)?;
    }
    file.flush().map_err(io)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let matrix =
            WeightMatrix::from_f32(2, 3, DType::BF16, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut norm = Vec::new();
        quant::encode_row(DType::F32, &[0.5, -0.5], &mut norm);
        write_safetensors(
            &path,
            &[
                ("w".into(), vec![2, 3], DType::BF16, matrix.bytes().clone()),
                ("norm".into(), vec![2], DType::F32, ByteBuf::owned(norm)),
            ],
        )
        .unwrap();

        let ckpt = SafetensorsCheckpoint::open_dir(dir.path()).unwrap();
        assert_eq!(ckpt.tensor_names(), vec!["norm", "w"]);
        assert_eq!(
            ckpt.matrix("w").unwrap().to_f32(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert!(ckpt.matrix("w").unwrap().bytes().is_mapped());
        assert_eq!(ckpt.vector("norm").unwrap(), vec![0.5, -0.5]);
        assert!(matches!(
            ckpt.matrix("missing"),
            Err(HaleError::MissingTensor(_))
        ));
        assert!(ckpt.vector("w").is_err(), "2-D tensor is not a vector");
    }

    #[test]
    fn rejects_truncated_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.safetensors");
        std::fs::write(&path, 1000u64.to_le_bytes()).unwrap();
        assert!(SafetensorsCheckpoint::open_dir(dir.path()).is_err());
    }
}
