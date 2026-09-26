//! Byte storage that is either owned or borrowed from a memory-mapped file.
//!
//! On Apple Silicon the CPU and GPU share one pool of *unified memory*, and
//! the OS page cache lives in that same pool. Memory-mapping a checkpoint
//! therefore gives us weights "for free": pages are pulled from the SSD on
//! first touch and can be evicted by the OS under pressure, without us
//! copying anything. [`ByteBuf`] hides whether bytes came from a map or
//! from an owned allocation, so kernels never have to care.

use std::sync::Arc;

use memmap2::Mmap;

/// Where the bytes physically live.
#[derive(Clone)]
enum Storage {
    Owned(Arc<Vec<u8>>),
    Mapped(Arc<Mmap>),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            Storage::Owned(v) => v,
            Storage::Mapped(m) => m,
        }
    }
}

/// An immutable, cheaply clonable window of bytes.
///
/// Cloning or slicing never copies data; it only bumps a reference count.
#[derive(Clone)]
pub struct ByteBuf {
    storage: Storage,
    offset: usize,
    len: usize,
}

impl ByteBuf {
    /// Wraps an owned vector (e.g. an expert read from the SSD pack).
    pub fn owned(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        ByteBuf {
            storage: Storage::Owned(Arc::new(bytes)),
            offset: 0,
            len,
        }
    }

    /// Borrows `len` bytes starting at `offset` from a shared memory map.
    ///
    /// # Panics
    /// Panics if the window lies outside the map; callers validate offsets
    /// when they parse file headers.
    pub fn mapped(map: Arc<Mmap>, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).is_some_and(|end| end <= map.len()),
            "mapped window out of bounds"
        );
        ByteBuf {
            storage: Storage::Mapped(map),
            offset,
            len,
        }
    }

    /// The bytes of this window.
    pub fn as_bytes(&self) -> &[u8] {
        &self.storage.bytes()[self.offset..self.offset + self.len]
    }

    /// Number of bytes in the window.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the window is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the bytes come from a memory-mapped file.
    pub fn is_mapped(&self) -> bool {
        matches!(self.storage, Storage::Mapped(_))
    }

    /// A sub-window sharing the same storage (no copy).
    ///
    /// # Panics
    /// Panics if `start + len` exceeds this window.
    pub fn slice(&self, start: usize, len: usize) -> ByteBuf {
        assert!(
            start.checked_add(len).is_some_and(|end| end <= self.len),
            "slice out of bounds"
        );
        ByteBuf {
            storage: self.storage.clone(),
            offset: self.offset + start,
            len,
        }
    }
}

impl std::fmt::Debug for ByteBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = if self.is_mapped() { "mapped" } else { "owned" };
        write!(f, "ByteBuf({kind}, {} bytes)", self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slicing_shares_storage_and_offsets_compose() {
        let buf = ByteBuf::owned((0u8..10).collect());
        let a = buf.slice(2, 6);
        let b = a.slice(1, 3);
        assert_eq!(b.as_bytes(), &[3, 4, 5]);
        assert!(!b.is_mapped());
    }
}
