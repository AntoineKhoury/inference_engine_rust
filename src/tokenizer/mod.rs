//! Tokenizer: **SentencePiece** (`.model`) or Hugging Face **`tokenizer.json`** (e.g. Gemma 4).
pub mod sentencepiece;

pub use sentencepiece::Tokenizer;
