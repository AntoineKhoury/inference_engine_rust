use crate::EngineError;
use crate::model_loader::gguf_types::GGUFData;
use crate::core::tensor::{Tensor, TensorType};
use crate::ops::quant::quant_k_handler::{
    dequantize_q4k_block, dequantize_q6k_block, Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE,
};
const BLOCK_ELEMENTS: usize = 256;

/// Flat index into the GGUF tensor buffer for embedding element `h` of `token_id`.
/// Quantized tensors are stored with **reversed** dims vs metadata (see gguf `ReaderTensor` /
/// `quant_shape_to_byte_shape`): logical rows are **vocab** × **contiguous hidden**. So the slice
/// for one token is `token_id * hidden_dim ..` and index = `token_id * hidden_dim + h`.
#[inline]
fn embedding_buffer_index(hidden_dim: usize, token_id: u32, h: usize) -> usize {
    (token_id as usize) * hidden_dim + h
}

/// Lookup embeddings for a sequence of token IDs.
///
/// This is the usual first step in the inference pipeline: tokenizer IDs → dense rows.
///
/// # Arguments
///
/// * `gguf_data` — loaded GGUF model data
/// * `token_ids` — IDs to look up
///
/// # Returns
///
/// One `Vec<f32>` per token, each of length `hidden_dim`.
///
/// # Errors
///
/// Missing embedding tensor, out-of-vocab IDs, or unexpected shape/dtype.
///
/// # Performance
///
/// For sequence length N: about `N * hidden_dim` float reads (memory-bound; locality matters).
///
/// # Architecture notes
///
/// - Embeddings are often F32 for frequent access.
/// - Logical shape is `[vocab_size, hidden_dim]` (or transposed in some GGUFs).
/// - Row selection: `embedding = weights[token_id]`.
pub fn lookup_embeddings(
    gguf_data: &mut GGUFData,
    file_path: &str,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, EngineError> {
    let embedding_tensor_name = resolve_embedding_tensor_name(gguf_data)?;

    if gguf_data.get_tensor(embedding_tensor_name).is_none() {
        gguf_data.load_single_tensor(file_path, embedding_tensor_name)?;
    }

    lookup_embeddings_loaded(gguf_data, token_ids)
}

/// Same as [`lookup_embeddings`], but only reads an **already-loaded** embedding tensor.
///
/// Use this while holding [`crate::model_weights::ModelWeights`] (which borrows `GGUFData`)
/// so you can fetch the next token’s row during decode without `&mut GGUFData`.
pub fn lookup_embeddings_loaded(
    gguf_data: &GGUFData,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, EngineError> {
    let embedding_tensor_name = resolve_embedding_tensor_name(gguf_data)?;
    let embedding_tensor = gguf_data
        .get_tensor(embedding_tensor_name)
               .ok_or_else(|| {
            EngineError::Model(format!(
                "embedding tensor '{embedding_tensor_name}' not loaded; call load_single_tensor or lookup_embeddings first"
            ))
        })?;

    lookup_embedding_rows(embedding_tensor, token_ids)
}

fn resolve_embedding_tensor_name(gguf_data: &GGUFData) -> Result<&'static str, EngineError> {
    const NAMES: [&str; 3] = [
        "token_embd.weight",
        "tok_embeddings.weight",
        "embeddings.weight",
    ];
    for &name in &NAMES {
        if gguf_data
            .tensors_metadata()
            .iter()
            .any(|t| t.name == name)
        {
            return Ok(name);
        }
    }
    Err(EngineError::Model(
        "embedding tensor not found in metadata (expected token_embd.weight, tok_embeddings.weight, or embeddings.weight)".into(),
    ))
}

fn lookup_embedding_rows(
    embedding_tensor: &Tensor,
    token_ids: &[u32],
) -> Result<Vec<Vec<f32>>, EngineError> {
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err(EngineError::Tensor(format!(
            "expected 2D embedding tensor, got {}D with shape {:?}",
            dims.len(),
            dims
        )));
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Mistral uses [hidden_dim, vocab_size] = [4096, 32000]
    let (hidden_dim, vocab_size) = if dims[0] < dims[1] {
        // Likely [hidden_dim, vocab_size] - Mistral format
        (dims[0], dims[1])
    } else {
        // Likely [vocab_size, hidden_dim] - standard format
        (dims[1], dims[0])
    };
    
    // Validate token IDs are within vocabulary range
    for &token_id in token_ids {
        if token_id as usize >= vocab_size {
            return Err(EngineError::Model(format!(
                "token ID {token_id} out of vocabulary range [0, {vocab_size})"
            )));
        }
    }
    
    let buf = embedding_tensor.buffer();
    let mut embeddings = Vec::with_capacity(token_ids.len());

    match embedding_tensor.dtype() {
        TensorType::F32 => {
            for &token_id in token_ids {
                let mut embedding = vec![0.0f32; hidden_dim];
                for (h, slot) in embedding.iter_mut().enumerate() {
                    let idx = embedding_buffer_index(hidden_dim, token_id, h);
                    *slot = embedding_tensor.f32_at(idx)?;
                }
                embeddings.push(embedding);
            }
        }
        TensorType::Q4K => {
            for &token_id in token_ids {
                let mut embedding = vec![0.0f32; hidden_dim];
                let mut cached_block = usize::MAX;
                let mut decoded = [0.0f32; BLOCK_ELEMENTS];
                for (h, slot) in embedding.iter_mut().enumerate() {
                    let idx = embedding_buffer_index(hidden_dim, token_id, h);
                    let block_idx = idx / BLOCK_ELEMENTS;
                    let el = idx % BLOCK_ELEMENTS;
                    if block_idx != cached_block {
                        let start = block_idx * Q4K_BLOCK_SIZE;
                        let block = buf
                            .get(start..start + Q4K_BLOCK_SIZE)
                            .ok_or_else(|| {
                                EngineError::Tensor("Q4K embedding block out of bounds".into())
                            })?;
                        dequantize_q4k_block(block, &mut decoded)?;
                        cached_block = block_idx;
                    }
                    *slot = decoded[el];
                }
                embeddings.push(embedding);
            }
        }
        TensorType::Q6K => {
            for &token_id in token_ids {
                let mut embedding = vec![0.0f32; hidden_dim];
                let mut cached_block = usize::MAX;
                let mut decoded = [0.0f32; BLOCK_ELEMENTS];
                for (h, slot) in embedding.iter_mut().enumerate() {
                    let idx = embedding_buffer_index(hidden_dim, token_id, h);
                    let block_idx = idx / BLOCK_ELEMENTS;
                    let el = idx % BLOCK_ELEMENTS;
                    if block_idx != cached_block {
                        let start = block_idx * Q6K_BLOCK_SIZE;
                        let block = buf
                            .get(start..start + Q6K_BLOCK_SIZE)
                            .ok_or_else(|| {
                                EngineError::Tensor("Q6K embedding block out of bounds".into())
                            })?;
                        dequantize_q6k_block(block, &mut decoded)?;
                        cached_block = block_idx;
                    }
                    *slot = decoded[el];
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
pub fn get_embedding_dim(gguf_data: &GGUFData) -> Result<usize, EngineError> {
    let embedding_tensor = gguf_data
        .get_tensor("token_embd.weight")
        .or_else(|| gguf_data.get_tensor("tok_embeddings.weight"))
        .or_else(|| gguf_data.get_tensor("embeddings.weight"))
        .ok_or_else(|| {
            EngineError::Model(
                "embedding tensor not found; load it first with load_single_tensor()".into(),
            )
        })?;
    
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err(EngineError::Tensor("embedding tensor must be 2D".into()));
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Return the smaller dimension as hidden_dim
    Ok(if dims[0] < dims[1] { dims[0] } else { dims[1] })
}

/// Get the vocabulary size from the embedding tensor
/// 
/// Note: This requires the embedding tensor to be loaded. Use `load_single_tensor()`
/// if you haven't loaded all tensors yet.
pub fn get_vocab_size(gguf_data: &GGUFData) -> Result<usize, EngineError> {
    let embedding_tensor = gguf_data
        .get_tensor("token_embd.weight")
        .or_else(|| gguf_data.get_tensor("tok_embeddings.weight"))
        .or_else(|| gguf_data.get_tensor("embeddings.weight"))
        .ok_or_else(|| {
            EngineError::Model(
                "embedding tensor not found; load it first with load_single_tensor()".into(),
            )
        })?;
    
    let dims = embedding_tensor.dimensions();
    if dims.len() != 2 {
        return Err(EngineError::Tensor("embedding tensor must be 2D".into()));
    }
    
    // Handle both layouts: [vocab_size, hidden_dim] or [hidden_dim, vocab_size]
    // Return the larger dimension as vocab_size
    Ok(if dims[0] > dims[1] { dims[0] } else { dims[1] })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_loader::file_loader::read_file;
    
    #[test]
    #[ignore = "requires model/mistral-7b-v0.1.Q4_K_M.gguf at repo root (cargo test -- --ignored)"]
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
            let n_nan = embedding.iter().filter(|x| x.is_nan()).count();
            assert_eq!(
                n_nan, 0,
                "Q4_K embedding dequant must be finite (ggml-compatible sign/layout)"
            );
        }
    }
    
    #[test]
    #[ignore = "requires model/mistral-7b-v0.1.Q4_K_M.gguf at repo root (cargo test -- --ignored)"]
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
    #[ignore = "requires model/mistral-7b-v0.1.Q4_K_M.gguf at repo root (cargo test -- --ignored)"]
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
    #[ignore = "requires model/mistral-7b-v0.1.Q4_K_M.gguf at repo root (cargo test -- --ignored)"]
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
