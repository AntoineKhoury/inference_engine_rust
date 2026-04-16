//! Unified error type for the inference engine library.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),

    /// GGUF parsing / layout issues (metadata, tensor table, seek/read).
    #[error("GGUF: {0}")]
    Gguf(String),

    #[error("tensor: {0}")]
    Tensor(String),

    #[error("matmul: {0}")]
    MatMul(String),

    #[error("model: {0}")]
    Model(String),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error(transparent)]
    KvCache(#[from] crate::layers::attention::KVCacheError),

    #[error(transparent)]
    Sampling(#[from] crate::sampling::SamplingError),

    /// Invalid arguments to a low-level op (e.g. RoPE dimensions).
    #[error("invalid op: {0}")]
    Op(String),
}
