use crate::core::types::{GGUFData, TensorType};

/// Embedding lookup operation for converting token IDs to embedding vectors
/// 
/// This module provides efficient embedding lookup, which is the first step
/// in the inference pipeline. Token IDs (from the tokenizer) are converted
/// to dense embedding vectors that can be processed by the transformer layers.
/// 
/// # Architecture Notes
/// - Embeddings are typically stored as F32 (unquantized) for frequent access
/// - Shape: [vocab_size, hidden_dim] (e.g., [32000, 4096] for Mistral-7B)
/// - Lookup is a simple row selection: `embedding = weights[token_id]`
/// - This is a memory-bound operation, so cache locality matters

/// Lookup embeddings for a sequence of token IDs
/// 
/// # Arguments
/// * `gguf_data` - The loaded GGUF model data containing all tensors
/// * `token_ids` - Vector of token IDs to convert to embeddings
/// 
/// # Returns
/// * `Result<Vec<Vec<f32>>>` - Vector of embedding vectors, one per token
///   Each inner vector has length `hidden_dim` (e.g., 4096 for Mistral-7B)
/// 
/// # Errors
/// Returns an error if:
/// - The embedding tensor is not found in the model
/// - Token IDs are out of vocabulary range
/// - The embedding tensor has unexpected shape or type
/// 
/// # Performance
/// This is a simple memory lookup operation. For a sequence of length N:
/// - Memory access: N * hidden_dim floats (sequential, cache-friendly)
/// - No computation, just data movement
pub fn lookup_embeddings(
    gguf_data: &mut GGUFData,
    file_path: &str,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    // Try common embedding tensor names used in GGUF models
    // Check token_embd.weight first (used by Mistral) then others
    let embedding_tensor_names = [
        "token_embd.weight",
        "tok_embeddings.weight",
        "embeddings.weight",
    ];
    
    // Try to find the embedding tensor in metadata first
    let embedding_tensor_name = embedding_tensor_names
        .iter()
        .find(|&&name| {
            gguf_data.tensors_metadata()
                .iter()
                .any(|t| t.name == name)
        });
    
    let embedding_tensor_name = match embedding_tensor_name {
        Some(name) => *name,
        None => return Err("Embedding tensor not found in metadata. Expected one of: token_embd.weight, tok_embeddings.weight, embeddings.weight".into()),
    };
    
    // Load the embedding tensor if not already loaded (lazy loading)
    if gguf_data.get_tensor(embedding_tensor_name).is_none() {
        gguf_data.load_single_tensor(file_path, embedding_tensor_name)?;
    }
    
    let embedding_tensor = gguf_data
        .get_tensor(embedding_tensor_name)
        .expect("Tensor should be loaded now");
    
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err(format!(
            "Expected 2D embedding tensor, got {}D with shape {:?}",
            dims.len(),
            dims
        ).into());
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Mistral uses [hidden_dim, vocab_size] = [4096, 32000]
    let (hidden_dim, vocab_size) = if dims[0] < dims[1] {
        // Likely [hidden_dim, vocab_size] - Mistral format
        (dims[0] as usize, dims[1] as usize)
    } else {
        // Likely [vocab_size, hidden_dim] - standard format
        (dims[1] as usize, dims[0] as usize)
    };
    
    // Validate token IDs are within vocabulary range
    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(format!(
                "Token ID {} is out of vocabulary range [0, {})",
                token_id, vocab_size
            ).into());
        }
    }
    
    // Perform embedding lookup with dequantization if needed
    let mut embeddings = Vec::with_capacity(token_ids.len());
    
    match embedding_tensor.tensor_type {
        TensorType::F32 => {
            // F32: Direct lookup
            let embedding_data = embedding_tensor
                .f32_data()
                .ok_or("F32 embedding tensor has no data")?;
            
            // Layout: [hidden_dim, vocab_size] means row-major: row i = token i
            // Row i starts at index i * hidden_dim
            for &token_id in token_ids {
                let row_start = (token_id as usize) * hidden_dim;
                let row_end = row_start + hidden_dim;
                let embedding = embedding_data[row_start..row_end].to_vec();
                embeddings.push(embedding);
            }
        }
        TensorType::Q4K => {
            // Q4K: Dequantize on-the-fly
            let quantized_data = embedding_tensor
                .quantized_data()
                .ok_or("Q4K embedding tensor missing quantized_data")?;
            let scales = embedding_tensor
                .scales()
                .ok_or("Q4K embedding tensor missing scales")?;
            let mins = embedding_tensor
                .mins()
                .ok_or("Q4K embedding tensor missing mins")?;
            
            const BLOCK_SIZE: usize = 32; // Q4K uses blocks of 32 weights
            
            for &token_id in token_ids {
                let mut embedding = Vec::with_capacity(hidden_dim);
                let row_start = (token_id as usize) * hidden_dim;
                
                // Dequantize each element in the embedding vector
                for dim_idx in 0..hidden_dim {
                    let element_idx = row_start + dim_idx;
                    let block_idx = element_idx / BLOCK_SIZE;
                    let quantized = quantized_data[element_idx] as f32;
                    let scale = scales[block_idx];
                    let min = mins[block_idx];
                    let dequantized = (quantized * scale) + min;
                    embedding.push(dequantized);
                }
                embeddings.push(embedding);
            }
        }
        TensorType::Q6K => {
            // Q6K: Dequantize on-the-fly (similar to Q4K but 6-bit)
            let quantized_data = embedding_tensor
                .quantized_data()
                .ok_or("Q6K embedding tensor missing quantized_data")?;
            let scales = embedding_tensor
                .scales()
                .ok_or("Q6K embedding tensor missing scales")?;
            let mins = embedding_tensor
                .mins()
                .ok_or("Q6K embedding tensor missing mins")?;
            
            const BLOCK_SIZE: usize = 32; // Q6K also uses blocks of 32 weights
            
            for &token_id in token_ids {
                let mut embedding = Vec::with_capacity(hidden_dim);
                let row_start = (token_id as usize) * hidden_dim;
                
                // Dequantize each element in the embedding vector
                for dim_idx in 0..hidden_dim {
                    let element_idx = row_start + dim_idx;
                    let block_idx = element_idx / BLOCK_SIZE;
                    let quantized = quantized_data[element_idx] as f32;
                    let scale = scales[block_idx];
                    let min = mins[block_idx];
                    let dequantized = (quantized * scale) + min;
                    embedding.push(dequantized);
                }
                embeddings.push(embedding);
            }
        }
    }
    
    Ok(embeddings)
}

/// Get the embedding dimension (hidden_dim) from the model
/// 
/// Useful for validating inputs and allocating buffers with correct sizes
/// 
/// Note: This requires the embedding tensor to be loaded. Use `load_single_tensor()`
/// if you haven't loaded all tensors yet.
pub fn get_embedding_dim(gguf_data: &GGUFData) -> Result<usize, Box<dyn std::error::Error>> {
    let embedding_tensor = gguf_data
        .get_tensor("token_embd.weight")
        .or_else(|| gguf_data.get_tensor("tok_embeddings.weight"))
        .or_else(|| gguf_data.get_tensor("embeddings.weight"))
        .ok_or("Embedding tensor not found. Load it first with load_single_tensor()")?;
    
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err("Embedding tensor must be 2D".into());
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Return the smaller dimension as hidden_dim
    Ok(if dims[0] < dims[1] { dims[0] as usize } else { dims[1] as usize })
}

/// Get the vocabulary size from the embedding tensor
/// 
/// Note: This requires the embedding tensor to be loaded. Use `load_single_tensor()`
/// if you haven't loaded all tensors yet.
pub fn get_vocab_size(gguf_data: &GGUFData) -> Result<usize, Box<dyn std::error::Error>> {
    let embedding_tensor = gguf_data
        .get_tensor("token_embd.weight")
        .or_else(|| gguf_data.get_tensor("tok_embeddings.weight"))
        .or_else(|| gguf_data.get_tensor("embeddings.weight"))
        .ok_or("Embedding tensor not found. Load it first with load_single_tensor()")?;
    
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err("Embedding tensor must be 2D".into());
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Return the larger dimension as vocab_size
    Ok(if dims[0] > dims[1] { dims[0] as usize } else { dims[1] as usize })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::file_loader::read_file;
    
    #[test]
    fn test_embedding_lookup() {
        // Load the model
        let path = "model/mistral-7b-v0.1.Q4_K_M.gguf";
        let mut gguf_data = read_file(path).expect("Failed to read GGUF file");
        // Don't load all tensors - just load the embedding tensor on demand
        // gguf_data.load_tensors(path).expect("Failed to load tensors");
        
        // Test with a few token IDs (this will lazy-load the embedding tensor)
        let token_ids = vec![1, 2, 3];
        let embeddings = lookup_embeddings(&mut gguf_data, path, &token_ids)
            .expect("Failed to lookup embeddings");
        
        // Verify we got the right number of embeddings
        assert_eq!(embeddings.len(), token_ids.len());
        
        // Verify each embedding has the correct dimension
        let hidden_dim = get_embedding_dim(&gguf_data)
            .expect("Failed to get embedding dimension");
        
        for embedding in &embeddings {
            assert_eq!(embedding.len(), hidden_dim, "Embedding dimension mismatch");
        }
    }
    
    #[test]
    fn test_get_embedding_dim() {
        let path = "model/mistral-7b-v0.1.Q4_K_M.gguf";
        let mut gguf_data = read_file(path).expect("Failed to read GGUF file");
        // Load just the embedding tensor
        gguf_data.load_single_tensor(path, "token_embd.weight")
            .expect("Failed to load embedding tensor");
        
        let dim = get_embedding_dim(&gguf_data)
            .expect("Failed to get embedding dimension");
        
        // Mistral-7B should have 4096 hidden dimension
        assert_eq!(dim, 4096, "Expected hidden_dim=4096 for Mistral-7B");
    }
    
    #[test]
    fn test_get_vocab_size() {
        let path = "model/mistral-7b-v0.1.Q4_K_M.gguf";
        let mut gguf_data = read_file(path).expect("Failed to read GGUF file");
        // Load just the embedding tensor
        gguf_data.load_single_tensor(path, "token_embd.weight")
            .expect("Failed to load embedding tensor");
        
        let vocab_size = get_vocab_size(&gguf_data)
            .expect("Failed to get vocabulary size");
        
        // Mistral-7B should have vocabulary size around 32000
        assert_eq!(vocab_size, 32000, "Expected vocab_size=32000 for Mistral-7B");
    }
    
    #[test]
    fn test_out_of_vocab_error() {
        let path = "model/mistral-7b-v0.1.Q4_K_M.gguf";
        let mut gguf_data = read_file(path).expect("Failed to read GGUF file");
        // Load just the embedding tensor
        gguf_data.load_single_tensor(path, "token_embd.weight")
            .expect("Failed to load embedding tensor");
        
        let vocab_size = get_vocab_size(&gguf_data)
            .expect("Failed to get vocabulary size");
        
        // Try to lookup an out-of-vocabulary token
        let invalid_token_id = vocab_size as u32 + 100;
        let token_ids = vec![invalid_token_id];
        
        let result = lookup_embeddings(&mut gguf_data, path, &token_ids);
        assert!(result.is_err(), "Should error on out-of-vocabulary token");
    }
}
