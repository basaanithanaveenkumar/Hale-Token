//! Chat prompt templates per model family.

use crate::config::Architecture;

/// How a user prompt is wrapped before tokenisation.
///
/// # Examples
///
/// ```
/// use hale::tokenizer::PromptFormat;
///
/// assert_eq!(PromptFormat::MistralInstruct.apply("hi"), "[INST] hi [/INST]");
/// assert_eq!(PromptFormat::Raw.apply("hi"), "hi");
/// ```
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
    fn each_architecture_has_its_native_format() {
        assert_eq!(
            PromptFormat::for_architecture(Architecture::Qwen3Moe),
            PromptFormat::ChatMl
        );
        assert_eq!(
            PromptFormat::for_architecture(Architecture::Mixtral),
            PromptFormat::MistralInstruct
        );
    }

    #[test]
    fn chat_ml_wraps_user_turn_and_opens_assistant_turn() {
        assert_eq!(
            PromptFormat::ChatMl.apply("hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn only_chat_ml_adds_stop_strings() {
        assert!(PromptFormat::ChatMl.stop_strings().contains(&"<|im_end|>"));
        assert!(PromptFormat::Raw.stop_strings().is_empty());
        assert!(PromptFormat::MistralInstruct.stop_strings().is_empty());
    }
}
