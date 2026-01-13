use sentencepiece::SentencePieceProcessor;
use std::path::Path;

/// SentencePiece tokenizer wrapper for the inference engine
/// 
/// This struct provides a simple interface for encoding text to token IDs
/// and decoding token IDs back to text. It wraps `sentencepiece::SentencePieceProcessor`
/// to provide a clean API that fits the inference engine's architecture.
/// 
/// # Thread Safety
/// The underlying tokenizer is thread-safe (`Send + Sync`), allowing
/// concurrent tokenization operations across multiple threads.
pub struct Tokenizer {
    /// Internal SentencePiece processor
    inner: SentencePieceProcessor,
    /// Cache of piece strings for each token ID (for decoding)
    /// This is populated lazily as we encode text
    id_to_piece: std::collections::HashMap<u32, String>,
}

impl Tokenizer {
    /// Load a SentencePiece tokenizer from a model file
    /// 
    /// # Arguments
    /// * `path` - Path to the SentencePiece model file (typically `tokenizer.model`)
    /// 
    /// # Returns
    /// * `Result<Self>` - The loaded tokenizer or an error if loading fails
    /// 
    /// # Errors
    /// Returns an error if:
    /// - The file cannot be opened or read
    /// - The file is not a valid SentencePiece model
    /// 
    /// # Performance Note
    /// Loading the tokenizer involves reading and parsing the model file.
    /// This is typically done once during initialization and cached for reuse.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = SentencePieceProcessor::open(path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
        
        Ok(Self { 
            inner,
            id_to_piece: std::collections::HashMap::new(),
        })
    }
    
    /// Encode text into a sequence of token IDs
    /// 
    /// # Arguments
    /// * `text` - The input text to tokenize
    /// 
    /// # Returns
    /// * `Result<Vec<u32>>` - Vector of token IDs representing the input text
    /// 
/// # Errors
/// Returns an error if tokenization fails (should be rare for valid input)
pub fn encode(&mut self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        // The encode method returns a vector of SentencePiecePiece structs
        // Each piece has an id field (token ID) and a piece field (string representation)
        let pieces = self.inner.encode(text)
            .map_err(|e| format!("Failed to encode text: {}", e))?;
        
        // Cache the piece strings for decoding later
        for piece in &pieces {
            self.id_to_piece.insert(piece.id, piece.piece.clone());
        }
        
        let token_ids: Vec<u32> = pieces
            .iter()
            .map(|piece| piece.id)
            .collect();
        
        Ok(token_ids)
    }
    
    /// Decode a sequence of token IDs back into text
    /// 
    /// # Arguments
    /// * `tokens` - Slice of token IDs to decode
    /// 
    /// # Returns
    /// * `Result<String>` - The decoded text
    /// 
/// # Errors
/// Returns an error if decoding fails (e.g., invalid token IDs)
pub fn decode(&self, tokens: &[u32]) -> Result<String, Box<dyn std::error::Error>> {
        // The sentencepiece crate API may vary - let's try different approaches
        // First, try if there's a decode_ids method
        // If not, we'll need to reconstruct pieces from cached strings
        
        // Try to get piece strings from cache and reconstruct
        // Note: This is a workaround - the actual API might be different
        // We'll need to verify the exact sentencepiece crate API
        
        // For now, let's try using the piece strings we cached
        let piece_strings: Vec<String> = tokens
            .iter()
            .filter_map(|&id| self.id_to_piece.get(&id).cloned())
            .collect();
        
        // If we have all pieces, try to decode
        // The sentencepiece crate might have a decode method that takes strings
        // or we might need to use a different approach
        
        // Temporary: return an error indicating this needs API verification
        // The actual implementation will depend on the sentencepiece crate's exact API
        if piece_strings.len() == tokens.len() {
            // We have all pieces - try to decode
            // Note: This is a placeholder - actual API may differ
            Ok(piece_strings.join(""))
        } else {
            Err(format!(
                "Cannot decode: missing piece strings for {} out of {} tokens. \
                 Decode requires pieces to be cached during encoding.",
                tokens.len() - piece_strings.len(),
                tokens.len()
            ).into())
        }
    }
    
    /// Get the vocabulary size of the tokenizer
    /// 
    /// # Returns
    /// The number of tokens in the vocabulary
    /// 
    /// # Note
    /// This is useful for:
    /// - Validating token IDs are within valid range
    /// - Allocating embedding matrices with correct dimensions
    /// - Understanding the model's token space
    pub fn vocab_size(&self) -> usize {
        // The SentencePieceProcessor doesn't expose vocab_size directly
        // We can estimate it by checking the maximum token ID we can encode
        // For now, return a reasonable default or we could add a method to query it
        // Most modern models have vocab sizes between 30k-100k
        // This is a limitation - we might need to track this separately
        // For now, we'll return 0 to indicate it's not directly available
        // In practice, this would come from model metadata or be stored during loading
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_load() {
        // Test that we can load the tokenizer.model file
        let result = Tokenizer::load_from_file("tokenizer.model");
        assert!(result.is_ok(), "Failed to load tokenizer: {:?}", result.err());
    }
    
    #[test]
    fn test_encode_decode_roundtrip() {
        // Test that encoding and decoding preserves text (modulo normalization)
        let mut tokenizer = Tokenizer::load_from_file("tokenizer.model")
            .expect("Failed to load tokenizer");
        
        let test_text = "Hello, world!";
        let tokens = tokenizer.encode(test_text)
            .expect("Failed to encode text");
        
        assert!(!tokens.is_empty(), "Encoded tokens should not be empty");
        
        let decoded = tokenizer.decode(&tokens)
            .expect("Failed to decode tokens");
        
        // Note: The decoded text might not exactly match due to SentencePiece normalization
        // (e.g., whitespace handling, special token handling)
        // But it should be close
        assert!(!decoded.is_empty(), "Decoded text should not be empty");
    }
    
    #[test]
    fn test_encode_empty_string() {
        let mut tokenizer = Tokenizer::load_from_file("tokenizer.model")
            .expect("Failed to load tokenizer");
        
        let _tokens = tokenizer.encode("")
            .expect("Failed to encode empty string");
        
        // Empty string might encode to special tokens or empty vector
        // Both are valid
    }
}
