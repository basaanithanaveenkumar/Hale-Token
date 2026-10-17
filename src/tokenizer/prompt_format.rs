//! Chat prompt templates per model family.

use crate::config::Architecture;

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
