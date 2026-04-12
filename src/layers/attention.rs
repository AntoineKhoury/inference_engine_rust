use thiserror::Error;
use std::sync::Arc;

use crate::core::tensor::{Tensor, TensorType};
use crate::model_config::ModelConfig;
use crate::model_weights::LayerWeights;
use crate::ops::matmul::matmul;
use crate::ops::rope::rope;
use crate::ops::softmax::softmax;
use crate::prefill::PrefillState;
/// Per-layer KV cache: one `[head_dim]` slice per **KV head** per timestep (GQA/MQA).
pub struct KVCache {
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    current_pos: usize,
    max_seq_len: usize,
    /// Number of key/value heads (≤ query head count; equal for standard MHA).
    n_kv_heads: usize,
    head_dim: usize,
}

impl KVCache {
    pub fn new(max_seq_len: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let stride = n_kv_heads * head_dim;
        let total_size = max_seq_len * stride;

        Self {
            k_cache: vec![0.0; total_size],
            v_cache: vec![0.0; total_size],
            current_pos: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn append_kv(&mut self, k: &[f32], v: &[f32]) -> Result<(), KVCacheError> {
        if self.current_pos >= self.max_seq_len {
            return Err(KVCacheError::KVCacheFull {
                max_len: self.max_seq_len,
            });
        }
        let expected_len = self.n_kv_heads * self.head_dim;
        if k.len() != expected_len || v.len() != expected_len {
            return Err(KVCacheError::KVDimMismatch {
                k_size: expected_len,
            });
        }

        let stride = expected_len;
        let start_idx = self.current_pos * stride;

        self.k_cache[start_idx..start_idx + expected_len].copy_from_slice(k);
        self.v_cache[start_idx..start_idx + expected_len].copy_from_slice(v);

        self.current_pos += 1;
        Ok(())
    }

    pub fn get_k_slice(&self, position: usize, kv_head: usize) -> &[f32] {
        assert!(
            position < self.current_pos,
            "Position out of bounds for k slice"
        );
        assert!(kv_head < self.n_kv_heads, "KV head index out of bounds");

        let start_pos = position * self.n_kv_heads * self.head_dim + kv_head * self.head_dim;
        &self.k_cache[start_pos..start_pos + self.head_dim]
    }

    pub fn get_v_slice(&self, position: usize, kv_head: usize) -> &[f32] {
        assert!(
            position < self.current_pos,
            "Position out of bounds for v slice"
        );
        assert!(kv_head < self.n_kv_heads, "KV head index out of bounds");

        let start_pos = position * self.n_kv_heads * self.head_dim + kv_head * self.head_dim;
        &self.v_cache[start_pos..start_pos + self.head_dim]
    }
}

#[derive(Debug, Error)]
pub enum KVCacheError{
    #[error("KVCache is Full: max len is {max_len}.")]
    KVCacheFull{max_len: usize},
    
    #[error("Input size of k or v for KVCache isn't correct, size should be {k_size}")]
    KVDimMismatch{k_size: usize},
}

pub fn prefill_attention_layer(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    if hidden_dim != config.hidden_dim {
        return Err(format!(
            "prefill attention: hidden_dim {} != config.hidden_dim {}",
            hidden_dim, config.hidden_dim
        )
        .into());
    }
    if hidden_dim != config.n_heads * config.head_dim {
        return Err("prefill attention: hidden_dim != n_heads * head_dim".into());
    }

    let kv_dim = config.n_kv_heads * config.head_dim;
    if kv_cache.n_kv_heads() != config.n_kv_heads || kv_cache.head_dim() != config.head_dim {
        return Err("prefill attention: KVCache n_kv_heads/head_dim does not match config".into());
    }

    let group_size = config.n_heads / config.n_kv_heads;

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, hidden_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    let mut v_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);

    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    matmul(&input_tensor, weights.wk, &mut k_tensor)?;
    matmul(&input_tensor, weights.wv, &mut v_tensor)?;

    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    let v_data = v_tensor.as_f32_slice_mut()?;

    let base = config.rope_theta as u32;
    let head_dim = config.head_dim;

    for pos in 0..seq_len {
        let q_row = pos * hidden_dim;
        for head in 0..config.n_heads {
            let head_start = q_row + head * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut q_data[head_start..head_end],
                base,
                pos as u32,
                head_dim as u32,
                head_dim as u32,
            );
        }
        let k_row = pos * kv_dim;
        for kv_h in 0..config.n_kv_heads {
            let head_start = k_row + kv_h * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut k_data[head_start..head_end],
                base,
                pos as u32,
                head_dim as u32,
                head_dim as u32,
            );
        }
    }

    for pos in 0..seq_len {
        let row_start = pos * kv_dim;
        let row_end = row_start + kv_dim;
        kv_cache.append_kv(&k_data[row_start..row_end], &v_data[row_start..row_end])?;
    }

    let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for pos in 0..seq_len {
        for head in 0..config.n_heads {
            let kv_head = head / group_size;
            let q_start = pos * hidden_dim + head * head_dim;
            let q = &q_data[q_start..q_start + head_dim];

            let mut scores = vec![0.0f32; pos + 1];
            for j in 0..=pos {
                let k_start = j * kv_dim + kv_head * head_dim;
                let k = &k_data[k_start..k_start + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * k[d];
                }
                scores[j] = dot * scale;
            }

            let mut weights_buf = vec![0.0f32; pos + 1];
            softmax(&scores, &mut weights_buf)?;

            let out_start = pos * hidden_dim + head * head_dim;
            let out = &mut attn_out[out_start..out_start + head_dim];
            for j in 0..=pos {
                let v_start = j * kv_dim + kv_head * head_dim;
                let v = &v_data[v_start..v_start + head_dim];
                let w = weights_buf[j];
                for d in 0..head_dim {
                    out[d] += w * v[d];
                }
            }
        }
    }

    let attn_tensor = tensor_from_f32_slice(&attn_out, vec![seq_len, hidden_dim]);
    let mut projected = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&attn_tensor, weights.wo, &mut projected)?;

    Ok(projected.as_f32_slice()?.to_vec())
}

fn tensor_from_f32_slice(data: &[f32], dimensions: Vec<usize>) -> Tensor {
    Tensor::new(
        TensorType::F32,
        Arc::new(f32_bytes(data)),
        dimensions,
    )
}

fn empty_f32_tensor(dimensions: Vec<usize>) -> Tensor {
    let len = dimensions.iter().product::<usize>();
    Tensor::new(TensorType::F32, Arc::new(vec![0u8; len * 4]), dimensions)
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}