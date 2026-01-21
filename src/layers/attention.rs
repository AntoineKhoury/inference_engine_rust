use core::panic;
use thiserror::Error;
pub struct KVCache{
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    current_pos: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

impl KVCache{
    pub fn new(max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self{
        // Calculate the total size of each cache
        let total_size = max_seq_len * num_heads * head_dim; 

        // Return ownership of a newly created element
        Self {
            k_cache: vec![0.0; total_size],
            v_cache: vec![0.0; total_size],
            current_pos: 0,
            max_seq_len,
            num_heads,
            head_dim
        }
    }

    pub fn append_kv(&mut self, k: &[f32], v: &[f32]) -> Result<(), KVCacheError>{
        // Check we have space
        if self.current_pos >= self.max_seq_len{
            return Err(KVCacheError::KVCacheFull{ max_len: self.max_seq_len });
        }
        // Check we have the right input size for our cache
        let expected_len = self.num_heads * self.head_dim;
        if k.len() != expected_len || v.len() != expected_len{
            return Err(KVCacheError::KVDimMismatch { k_size: expected_len });
        }

        let stride = self.num_heads * self.head_dim;
        let start_idx = self.current_pos * stride;

        
        self.k_cache[start_idx .. start_idx + expected_len].copy_from_slice(k);
        self.v_cache[start_idx .. start_idx + expected_len].copy_from_slice(v);

        // Advance pos
        self.current_pos += 1;
        Ok(())
    }

    pub fn get_k_slice(&self, position: usize, head: usize) -> &[f32]{
        if position > self.current_pos{
            panic!("Position out of bounds for k slice")
        }
        if head >= self.num_heads{
            panic!("Head index is out of bound")
        }

        // We need to look for a specific k slice for a certain head at a certain position
        let start_pos = position * self.num_heads * self.head_dim + head * self.head_dim;
        &self.k_cache[start_pos .. start_pos + self.head_dim]
    }

    pub fn get_v_slice(&self, position: usize, head: usize) -> &[f32]{
        if position >= self.current_pos{
            panic!("Position out of bounds for v slice")
        }
        if head >= self.num_heads{
            panic!("Head index is out of bound")
        }
        
        // Same as for get_k_slice
        let start_pos = position * self.num_heads * self.head_dim + head * self.head_dim;
        &self.v_cache[start_pos .. start_pos + self.head_dim]
    }
}

#[derive(Debug, Error)]
pub enum KVCacheError{
    #[error("KVCache is Full: max len is {max_len}.")]
    KVCacheFull{max_len: usize},
    
    #[error("Input size of k or v for KVCache isn't correct, size should be {k_size}")]
    KVDimMismatch{k_size: usize},
}