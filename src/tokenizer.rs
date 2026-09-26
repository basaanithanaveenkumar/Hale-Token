//! Text <-> token ids, and chat prompt formatting.
//!
//! Wraps Hugging Face's `tokenizers` crate, which reads the `tokenizer.json`
//! shipped with every modern checkpoint.

use std::path::Path;

use crate::config::Architecture;
use crate::error::{HaleError, Result};

/// A loaded tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Loads `<dir>/tokenizer.json`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| HaleError::Tokenizer(format!("{}: {e}", path.display())))?;
        Ok(Tokenizer { inner })
    }

    /// Text to ids. `add_special_tokens` adds e.g. a BOS token when the
    /// tokenizer is configured to.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| HaleError::Tokenizer(e.to_string()))
    }

    /// Ids to text (special tokens are dropped).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| HaleError::Tokenizer(e.to_string()))
    }

    /// Id of a literal token such as `"<|im_end|>"`.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

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

/// How a user prompt is wrapped before tokenisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptFormat {
    /// Send the text as-is (base models, or pre-formatted prompts).
    Raw,
    /// `<|im_start|>` chat markup used by Qwen models.
    ChatMl,
    /// `[INST] ... [/INST]` markup used by Mistral/Mixtral instruct models.
    MistralInstruct,
}

impl PromptFormat {
    /// The natural chat format for an architecture.
    pub fn for_architecture(arch: Architecture) -> Self {
        match arch {
            Architecture::Qwen3Moe => PromptFormat::ChatMl,
            Architecture::Mixtral => PromptFormat::MistralInstruct,
        }
    }

    /// Wraps a single user message.
    pub fn apply(self, user_message: &str) -> String {
        match self {
            PromptFormat::Raw => user_message.to_string(),
            PromptFormat::ChatMl => {
                format!("<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n")
            }
            PromptFormat::MistralInstruct => format!("[INST] {user_message} [/INST]"),
        }
    }

    /// Extra stop tokens the format implies.
    pub fn stop_strings(self) -> &'static [&'static str] {
        match self {
            PromptFormat::ChatMl => &["<|im_end|>", "<|endoftext|>"],
            PromptFormat::MistralInstruct | PromptFormat::Raw => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
