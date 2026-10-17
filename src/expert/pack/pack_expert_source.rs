//! Reads experts from a pack with direct, page-cache-bypassing I/O.

use super::super::{Expert, ExpertKey, ExpertSource};
use super::{PackHeader, ALIGN, FORMAT_VERSION, MAGIC};
use crate::error::{HaleTokenError, Result};
use crate::loader::read_u64_le;
use crate::tensor::ByteBuf;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Reads experts from a pack with positioned reads (tier 2 of the cache).
pub struct PackExpertSource {
    file: File,
    path: PathBuf,
    header: PackHeader,
}

impl PackExpertSource {
    /// Opens a pack and validates its header.
    pub fn open(path: &Path) -> Result<Self> {
        let io = |e| HaleTokenError::io(path, e);
        let mut file = File::open(path).map_err(io)?;
        let mut prefix = [0u8; 16];
        file.read_exact(&mut prefix).map_err(io)?;
        if &prefix[..8] != MAGIC {
            return Err(HaleTokenError::InvalidArgument(format!(
                "{} is not a Hale expert pack",
                path.display()
            )));
        }
        let json_len = read_u64_le(&prefix[8..]) as usize;
        if json_len > ALIGN * 64 {
            return Err(HaleTokenError::InvalidArgument(format!(
                "{}: header too large",
                path.display()
            )));
        }
        let mut json = vec![0u8; json_len];
        file.read_exact(&mut json).map_err(io)?;
        let header: PackHeader =
            serde_json::from_slice(&json).map_err(|e| HaleTokenError::Json {
                path: path.to_path_buf(),
                source: e,
            })?;
        if header.version != FORMAT_VERSION {
            return Err(HaleTokenError::InvalidArgument(format!(
                "{}: pack version {} is not supported (expected {FORMAT_VERSION})",
                path.display(),
                header.version
            )));
        }
        let actual = file.metadata().map_err(io)?.len() as usize;
        if actual < header.file_size() {
            return Err(HaleTokenError::InvalidArgument(format!(
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
            .map_err(|e| HaleTokenError::io(&self.path, e))?;
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
