//! Tokenizer: **SentencePiece** (`.model`) or Hugging Face **`tokenizer.json`** (e.g. Gemma 4).
pub mod backend;

pub use backend::Tokenizer;
