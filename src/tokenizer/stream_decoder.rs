//! Incremental detokenisation for streaming output.

use super::Tokenizer;
use crate::error::Result;

/// Turns a growing list of ids into text deltas suitable for streaming.
///
/// Decoding token-by-token is wrong for byte-level BPE: one character can
/// span several tokens, and some tokenizers merge spaces across tokens.
/// So we decode the whole sequence and emit only the new suffix, holding
/// back output that ends in an incomplete UTF-8 character.
pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    ids: Vec<u32>,
    emitted: usize,
}

impl<'a> StreamDecoder<'a> {
    /// Creates a decoder that starts with no pending tokens.
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        StreamDecoder {
            tokenizer,
            ids: Vec::new(),
            emitted: 0,
        }
    }

    /// Adds one token and returns the newly completed text (may be empty).
    pub fn push(&mut self, id: u32) -> Result<String> {
        self.ids.push(id);
        let text = self.tokenizer.decode(&self.ids)?;
        if text.ends_with('\u{FFFD}')
            || text.len() < self.emitted
            || !text.is_char_boundary(self.emitted)
        {
            return Ok(String::new());
        }
        let delta = text[self.emitted..].to_string();
        self.emitted = text.len();
        Ok(delta)
    }
}
