//! Knowing the machine we run on.
//!
//! Token generation speed on Apple Silicon is set almost entirely by two
//! numbers: unified-memory bandwidth (how fast resident weights can be
//! streamed through the cores) and SSD read bandwidth (how fast missing
//! experts arrive). This module detects the chip, looks up its published
//! specs, and ([`bench`]) measures the real values.

pub mod bench;
mod chips;
mod detect;

pub use chips::{lookup_chip, ChipSpec, APPLE_CHIPS};
pub use detect::{detect, SystemInfo};
