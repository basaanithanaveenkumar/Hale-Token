//! Writes experts into a pack file (positional, thread-safe).

use super::super::{Expert, ExpertKey};
use super::{PackHeader, MAGIC};
use crate::error::{HaleTokenError, Result};
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

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
        let io = |e| HaleTokenError::io(path, e);
        let mut file = File::create(path).map_err(io)?;
        let json = header.to_json()?;
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

    /// The header this writer will emit.
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
            return Err(HaleTokenError::bad_tensor(
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
            .map_err(|e| HaleTokenError::io(&self.path, e))
    }

    /// Flushes the file to disk.
    pub fn finish(self) -> Result<()> {
        self.file
            .sync_all()
            .map_err(|e| HaleTokenError::io(&self.path, e))
    }
}
