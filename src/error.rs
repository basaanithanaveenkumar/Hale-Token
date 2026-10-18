//! Crate-wide error type.
//!
//! Every fallible function in the library returns [`Result<T>`], so callers
//! only ever have to handle one error type, [`HaleTokenError`]. Each variant
//! carries enough context to be printed straight to a user.
//!
//! ```text
//! HaleTokenError
//! ├── Io / Json / Safetensors / Serialization   reading and writing files
//! ├── MissingTensor / BadTensor                 checkpoint contents
//! ├── UnsupportedArchitecture                   model family not implemented
//! ├── UnknownRegistryKey / DuplicateRegistryKey  name-selected implementations
//! ├── InvalidArgument / ContextOverflow         bad user input
//! ├── Tokenizer                                 tokenizer load / encode / decode
//! └── Internal                                  a broken invariant (a bug), never user input
//! ```

use std::path::PathBuf;

/// Everything that can go wrong inside Hale-Token.
#[derive(Debug, thiserror::Error)]
pub enum HaleTokenError {
    /// A filesystem or I/O failure, annotated with the path involved.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// File or directory involved.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// A JSON file (config, manifest, stats) could not be parsed.
    #[error("invalid JSON in {path}: {source}")]
    Json {
        /// File that failed to parse.
        path: PathBuf,
        /// The underlying serde error.
        #[source]
        source: serde_json::Error,
    },

    /// A safetensors file is malformed.
    #[error("invalid safetensors file {path}: {message}")]
    Safetensors {
        /// File that is malformed.
        path: PathBuf,
        /// What is wrong with it.
        message: String,
    },

    /// A tensor the model needs is not present in the checkpoint.
    #[error("tensor `{0}` not found in checkpoint")]
    MissingTensor(String),

    /// A tensor exists but has an unexpected shape or dtype.
    #[error("tensor `{name}`: {message}")]
    BadTensor {
        /// Tensor name.
        name: String,
        /// What is wrong with it.
        message: String,
    },

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
    ContextOverflow {
        /// Tokens required (prompt + requested new tokens).
        needed: usize,
        /// Maximum context length of the model.
        max: usize,
    },

    /// An in-memory value could not be serialised (e.g. a header or stats to JSON).
    #[error("cannot serialise {what}: {source}")]
    Serialization {
        /// What was being serialised.
        what: &'static str,
        /// The underlying serde error.
        #[source]
        source: serde_json::Error,
    },

    /// A registry lookup used a name nobody registered.
    #[error("unknown {registry} `{key}`{}", format_suggestions(.suggestions))]
    UnknownRegistryKey {
        /// Kind of registry (e.g. `"tensor naming"`).
        registry: &'static str,
        /// The name that was requested.
        key: String,
        /// Registered names close to `key`.
        suggestions: Vec<String>,
    },

    /// A name was registered twice without `force`.
    #[error("{registry} `{key}` is already registered")]
    DuplicateRegistryKey {
        /// Kind of registry.
        registry: &'static str,
        /// The duplicated name.
        key: String,
    },

    /// An internal invariant was violated. This is a bug in Hale-Token, not a user error.
    #[error("internal error: {0} (please report this as a bug)")]
    Internal(String),
}

fn format_suggestions(suggestions: &[String]) -> String {
    if suggestions.is_empty() {
        String::new()
    } else {
        format!(" (did you mean: {}?)", suggestions.join(", "))
    }
}

/// Former name of [`HaleTokenError`], kept so existing code keeps compiling.
pub type HaleError = HaleTokenError;

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, HaleTokenError>;

impl HaleTokenError {
    /// Builds an [`HaleTokenError::Io`] from an `io::Error` and the path it concerns.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        HaleTokenError::Io {
            path: path.into(),
            source,
        }
    }

    /// Builds a [`HaleTokenError::Serialization`] for `what`.
    pub fn serialization(what: &'static str, source: serde_json::Error) -> Self {
        HaleTokenError::Serialization { what, source }
    }

    /// Builds an [`HaleTokenError::BadTensor`].
    pub fn bad_tensor(name: impl Into<String>, message: impl Into<String>) -> Self {
        HaleTokenError::BadTensor {
            name: name.into(),
            message: message.into(),
        }
    }
}
