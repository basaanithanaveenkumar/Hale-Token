//! Crate-wide error type.
//!
//! Every fallible function in the library returns [`Result<T>`], so callers
//! only ever have to handle one error type. Each variant carries enough
//! context to be printed straight to a user.

use std::path::PathBuf;

/// Everything that can go wrong inside Hale-Token.
#[derive(Debug, thiserror::Error)]
pub enum HaleError {
    /// A filesystem or I/O failure, annotated with the path involved.
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A JSON file (config, manifest, stats) could not be parsed.
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// A safetensors file is malformed.
    #[error("invalid safetensors file {path}: {message}")]
    Safetensors { path: PathBuf, message: String },

    /// A tensor the model needs is not present in the checkpoint.
    #[error("tensor `{0}` not found in checkpoint")]
    MissingTensor(String),

    /// A tensor exists but has an unexpected shape or dtype.
    #[error("tensor `{name}`: {message}")]
    BadTensor { name: String, message: String },

    /// The checkpoint uses a model architecture we do not implement.
    #[error(
        "unsupported model architecture `{0}` (supported: Qwen3MoeForCausalLM, MixtralForCausalLM)"
    )]
    UnsupportedArchitecture(String),

    /// The user asked for something that is not valid (bad flag, bad value).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Tokenizer loading or encoding/decoding failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The prompt plus requested tokens do not fit the model context.
    #[error("context overflow: {needed} tokens needed but the model supports {max}")]
    ContextOverflow { needed: usize, max: usize },
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, HaleError>;

impl HaleError {
    /// Builds an [`HaleError::Io`] from an `io::Error` and the path it concerns.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        HaleError::Io {
            path: path.into(),
            source,
        }
    }

    /// Builds an [`HaleError::BadTensor`].
    pub fn bad_tensor(name: impl Into<String>, message: impl Into<String>) -> Self {
        HaleError::BadTensor {
            name: name.into(),
            message: message.into(),
        }
    }
}
