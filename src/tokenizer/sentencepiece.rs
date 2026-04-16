use sentencepiece::SentencePieceProcessor;
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

use crate::EngineError;
use crate::model_config::TokenizerPromptConfig;

enum TokenizerBackend {
    SentencePiece(SentencePieceProcessor),
    HuggingFace(HfTokenizer),
}

/// Text tokenizer: **SentencePiece** (`.model`) or Hugging Face **`tokenizer.json`**.
pub struct Tokenizer {
    backend: TokenizerBackend,
    /// SPM-only cache for [`Self::decode`] when pieces were produced by [`Self::encode`].
    id_to_piece: std::collections::HashMap<u32, String>,
}

impl Tokenizer {
    /// Load from path. Uses **`tokenizer.json`** when the extension is `.json` (Gemma 4, etc.);
    /// otherwise loads as SentencePiece (Mistral/Llama **`tokenizer.model`**).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, EngineError> {
        let path = path.as_ref();
        let is_hf_json = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));

        if is_hf_json {
            let inner = HfTokenizer::from_file(path).map_err(|e| {
                EngineError::Tokenizer(format!("failed to load Hugging Face tokenizer.json: {e}"))
            })?;
            return Ok(Self {
                backend: TokenizerBackend::HuggingFace(inner),
                id_to_piece: std::collections::HashMap::new(),
            });
        }

        let inner = SentencePieceProcessor::open(path).map_err(|e| {
            EngineError::Tokenizer(format!("failed to load SentencePiece tokenizer: {e}"))
        })?;

        Ok(Self {
            backend: TokenizerBackend::SentencePiece(inner),
            id_to_piece: std::collections::HashMap::new(),
        })
    }

    pub fn decode_piece_ids(&self, ids: &[u32]) -> Result<String, EngineError> {
        match &self.backend {
            TokenizerBackend::SentencePiece(sp) => sp
                .decode_piece_ids(ids)
                .map_err(|e| EngineError::Tokenizer(format!("decode_piece_ids: {e}"))),
            TokenizerBackend::HuggingFace(hf) => hf
                .decode(ids, false)
                .map_err(|e| EngineError::Tokenizer(format!("decode: {e}"))),
        }
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<u32>, EngineError> {
        match &mut self.backend {
            TokenizerBackend::SentencePiece(sp) => {
                let pieces = sp
                    .encode(text)
                    .map_err(|e| EngineError::Tokenizer(format!("encode: {e}")))?;
                for piece in &pieces {
                    self.id_to_piece.insert(piece.id, piece.piece.clone());
                }
                Ok(pieces.iter().map(|piece| piece.id).collect())
            }
            TokenizerBackend::HuggingFace(hf) => {
                let enc = hf
                    .encode(text, false)
                    .map_err(|e| EngineError::Tokenizer(format!("encode: {e}")))?;
                Ok(enc.get_ids().to_vec())
            }
        }
    }

    pub fn encode_with_prompt_config(
        &mut self,
        text: &str,
        cfg: &TokenizerPromptConfig,
    ) -> Result<Vec<u32>, EngineError> {
        let mut ids = self.encode(text)?;
        if cfg.add_bos_token {
            ids.insert(0, cfg.bos_token_id);
        }
        if cfg.add_eos_token {
            ids.push(cfg.eos_token_id);
        }
        Ok(ids)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, EngineError> {
        match &self.backend {
            TokenizerBackend::HuggingFace(hf) => hf
                .decode(tokens, false)
                .map_err(|e| EngineError::Tokenizer(format!("decode: {e}"))),
            TokenizerBackend::SentencePiece(_) => {
                let piece_strings: Vec<String> = tokens
                    .iter()
                    .filter_map(|&id| self.id_to_piece.get(&id).cloned())
                    .collect();

                if piece_strings.len() == tokens.len() {
                    Ok(piece_strings.join(""))
                } else {
                    Err(EngineError::Tokenizer(format!(
                        "cannot decode: missing piece strings for {} out of {} tokens (decode needs cache from encode)",
                        tokens.len() - piece_strings.len(),
                        tokens.len()
                    )))
                }
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        match &self.backend {
            TokenizerBackend::HuggingFace(hf) => hf.get_vocab_size(true),
            TokenizerBackend::SentencePiece(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires model/mistral-7b-v0.1/tokenizer.model (see model/README.md)"]
    fn test_tokenizer_load() {
        let path = "model/mistral-7b-v0.1/tokenizer.model";
        let result = Tokenizer::load_from_file(path);
        assert!(
            result.is_ok(),
            "Failed to load tokenizer at {path}: {:?}. Place Mistral tokenizer there (see model/README.md).",
            result.err()
        );
    }

    #[test]
    #[ignore = "requires model/mistral-7b-v0.1/tokenizer.model (see model/README.md)"]
    fn test_encode_decode_roundtrip() {
        let path = "model/mistral-7b-v0.1/tokenizer.model";
        let mut tokenizer = Tokenizer::load_from_file(path)
            .expect("Failed to load tokenizer (see model/README.md)");

        let test_text = "Hello, world!";
        let tokens = tokenizer.encode(test_text).expect("Failed to encode text");

        assert!(!tokens.is_empty(), "Encoded tokens should not be empty");

        let decoded = tokenizer.decode(&tokens).expect("Failed to decode tokens");

        assert!(!decoded.is_empty(), "Decoded text should not be empty");
    }

    #[test]
    #[ignore = "requires model/mistral-7b-v0.1/tokenizer.model (see model/README.md)"]
    fn test_encode_empty_string() {
        let path = "model/mistral-7b-v0.1/tokenizer.model";
        let mut tokenizer = Tokenizer::load_from_file(path)
            .expect("Failed to load tokenizer (see model/README.md)");

        let _tokens = tokenizer.encode("").expect("Failed to encode empty string");
    }
}
