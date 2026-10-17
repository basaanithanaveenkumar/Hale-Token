//! Text <-> token ids, and chat prompt formatting.
//!
//! Wraps Hugging Face's `tokenizers` crate, which reads the `tokenizer.json`
//! shipped with every modern checkpoint.

mod prompt_format;
mod stream_decoder;

pub use self::prompt_format::PromptFormat;
pub use self::stream_decoder::StreamDecoder;

use crate::error::{HaleTokenError, Result};
use std::path::Path;

/// A loaded tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Loads `<dir>/tokenizer.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| HaleTokenError::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Tokenizer { inner })
    }

    /// Text to ids. `add_special_tokens` adds e.g. a BOS token when the
    /// tokenizer is configured to.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| HaleTokenError::Tokenizer(e.to_string()))
    }

    /// Ids to text (special tokens are dropped).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| HaleTokenError::Tokenizer(e.to_string()))
    }

    /// Id of a literal token such as `"<|im_end|>"`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Architecture;

    #[test]
    fn chat_formats_wrap_the_message() {
        assert_eq!(PromptFormat::Raw.apply("hi"), "hi");
        assert!(PromptFormat::ChatMl
            .apply("hi")
            .ends_with("<|im_start|>assistant\n"));
        assert_eq!(
            PromptFormat::MistralInstruct.apply("hi"),
            "[INST] hi [/INST]"
        );
        assert_eq!(
            PromptFormat::for_architecture(Architecture::Qwen3Moe),
            PromptFormat::ChatMl
        );
    }
}
