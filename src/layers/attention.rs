use thiserror::Error;
use std::sync::Arc;

use crate::EngineError;
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

    /// Number of timesteps stored in the cache (next write index).
    pub fn current_pos(&self) -> usize {
        self.current_pos
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

        let start_idx = self.current_pos * expected_len;

        self.k_cache[start_idx..start_idx + expected_len].copy_from_slice(k);
        self.v_cache[start_idx..start_idx + expected_len].copy_from_slice(v);

        self.current_pos += 1;
        Ok(())
    }

    /// Key vector for timestep `position` and KV head `kv_head` (length `head_dim`).
    pub fn get_k_slice(&self, position: usize, kv_head: usize) -> Result<&[f32], KVCacheError> {
        if position >= self.current_pos {
            return Err(KVCacheError::PositionOutOfBounds {
                position,
                current_pos: self.current_pos,
            });
        }
        if kv_head >= self.n_kv_heads {
            return Err(KVCacheError::KvHeadOutOfBounds {
                kv_head,
                n_kv_heads: self.n_kv_heads,
            });
        }
        let start_pos = position * self.n_kv_heads * self.head_dim + kv_head * self.head_dim;
        Ok(&self.k_cache[start_pos..start_pos + self.head_dim])
    }

    /// Value vector for timestep `position` and KV head `kv_head` (length `head_dim`).
    pub fn get_v_slice(&self, position: usize, kv_head: usize) -> Result<&[f32], KVCacheError> {
        if position >= self.current_pos {
            return Err(KVCacheError::PositionOutOfBounds {
                position,
                current_pos: self.current_pos,
            });
        }
        if kv_head >= self.n_kv_heads {
            return Err(KVCacheError::KvHeadOutOfBounds {
                kv_head,
                n_kv_heads: self.n_kv_heads,
            });
        }
        let start_pos = position * self.n_kv_heads * self.head_dim + kv_head * self.head_dim;
        Ok(&self.v_cache[start_pos..start_pos + self.head_dim])
    }
}

#[derive(Debug, Error)]
pub enum KVCacheError {
    #[error("KVCache is Full: max len is {max_len}.")]
    KVCacheFull { max_len: usize },

    #[error("Input size of k or v for KVCache isn't correct, size should be {k_size}")]
    KVDimMismatch { k_size: usize },

    #[error("KV cache position {position} is out of bounds (current_pos is {current_pos})")]
    PositionOutOfBounds {
        position: usize,
        current_pos: usize,
    },

    #[error("KV head index {kv_head} is out of bounds (n_kv_heads is {n_kv_heads})")]
    KvHeadOutOfBounds {
        kv_head: usize,
        n_kv_heads: usize,
    },
}

/// Undo HF→GGUF `LlamaModel.permute` on **one row** of Q or K activations (Llama-style GGUF only;
/// Mistral exports usually set `ModelConfig.unpack_llama_gguf_qk = false`).
///
/// `row` has length `n_groups * head_dim` (Q: `n_heads`; K: `n_kv_heads`). GGUF stores each head as
/// `reshape(n_g, 2, d/2).swapaxes(1,2).flatten` indices `h*2+b`; logical layout is `b*(d/2)+h`.
pub fn unpack_llama_gguf_qk_row(row: &mut [f32], n_groups: usize, head_dim: usize) {
    assert!(
        head_dim % 2 == 0,
        "unpack_llama_gguf_qk_row: head_dim must be even"
    );
    let half = head_dim / 2;
    assert_eq!(
        row.len(),
        n_groups * head_dim,
        "unpack_llama_gguf_qk_row: row length mismatch"
    );

    let tmp = row.to_vec();
    for g in 0..n_groups {
        let base = g * head_dim;
        for h in 0..half {
            for b in 0..2 {
                let perm_off = base + h * 2 + b;
                let log_off = base + b * half + h;
                row[log_off] = tmp[perm_off];
            }
        }
    }
}

pub fn prefill_attention_layer(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(format!(
            "prefill attention: hidden_dim {hidden_dim} != config.hidden_dim {}",
            config.hidden_dim
        )));
    }
    if hidden_dim != config.n_heads * config.head_dim {
        return Err(EngineError::Model(
            "prefill attention: hidden_dim != n_heads * head_dim".into(),
        ));
    }

    let kv_dim = config.n_kv_heads * config.head_dim;
    if kv_cache.n_kv_heads() != config.n_kv_heads || kv_cache.head_dim() != config.head_dim {
        return Err(EngineError::Model(
            "prefill attention: KVCache n_kv_heads/head_dim does not match config".into(),
        ));
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

    let head_dim = config.head_dim;
    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * hidden_dim..(pos + 1) * hidden_dim],
                config.n_heads,
                head_dim,
            );
            unpack_llama_gguf_qk_row(
                &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                config.n_kv_heads,
                head_dim,
            );
        }
    }

    let rope_base = config.rope_theta;

    for pos in 0..seq_len {
        let q_row = pos * hidden_dim;
        for head in 0..config.n_heads {
            let head_start = q_row + head * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut q_data[head_start..head_end],
                rope_base,
                pos as u32,
                head_dim as u32,
                head_dim as u32,
            )?;
        }
        let k_row = pos * kv_dim;
        for kv_h in 0..config.n_kv_heads {
            let head_start = k_row + kv_h * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut k_data[head_start..head_end],
                rope_base,
                pos as u32,
                head_dim as u32,
                head_dim as u32,
            )?;
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
            for (j, score_slot) in scores.iter_mut().enumerate().take(pos + 1) {
                let k_start = j * kv_dim + kv_head * head_dim;
                let k = &k_data[k_start..k_start + head_dim];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[d] * k[d];
                }
                *score_slot = dot * scale;
            }

            let mut weights_buf = vec![0.0f32; pos + 1];
            softmax(&scores, &mut weights_buf)?;

            let out_start = pos * hidden_dim + head * head_dim;
            let out = &mut attn_out[out_start..out_start + head_dim];
            for (j, &w) in weights_buf.iter().enumerate().take(pos + 1) {
                let v_start = j * kv_dim + kv_head * head_dim;
                let v = &v_data[v_start..v_start + head_dim];
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

/// Single-token attention for autoregressive decode. `input` must have `seq_len == 1`.
///
/// RoPE uses position `kv_cache.current_pos` (0-based index of this token in the full sequence).
/// Past keys/values are read from `kv_cache`; the new K/V are appended after RoPE.
pub fn decode_attention_layer(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    if seq_len != 1 {
        return Err(EngineError::Model(
            "decode attention: expected seq_len == 1".into(),
        ));
    }
    let hidden_dim = input.hidden_dim();
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(format!(
            "decode attention: hidden_dim {hidden_dim} != config.hidden_dim {}",
            config.hidden_dim
        )));
    }
    if hidden_dim != config.n_heads * config.head_dim {
        return Err(EngineError::Model(
            "decode attention: hidden_dim != n_heads * head_dim".into(),
        ));
    }

    let kv_dim = config.n_kv_heads * config.head_dim;
    if kv_cache.n_kv_heads() != config.n_kv_heads || kv_cache.head_dim() != config.head_dim {
        return Err(EngineError::Model(
            "decode attention: KVCache n_kv_heads/head_dim does not match config".into(),
        ));
    }

    let group_size = config.n_heads / config.n_kv_heads;
    let rope_pos = kv_cache.current_pos() as u32;

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![1, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![1, hidden_dim]);
    let mut k_tensor = empty_f32_tensor(vec![1, kv_dim]);
    let mut v_tensor = empty_f32_tensor(vec![1, kv_dim]);

    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    matmul(&input_tensor, weights.wk, &mut k_tensor)?;
    matmul(&input_tensor, weights.wv, &mut v_tensor)?;

    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    let v_data = v_tensor.as_f32_slice_mut()?;

    let head_dim = config.head_dim;
    if config.unpack_llama_gguf_qk {
        unpack_llama_gguf_qk_row(q_data, config.n_heads, head_dim);
        unpack_llama_gguf_qk_row(k_data, config.n_kv_heads, head_dim);
    }

    let rope_base = config.rope_theta;

    for head in 0..config.n_heads {
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        rope(
            &mut q_data[head_start..head_end],
            rope_base,
            rope_pos,
            head_dim as u32,
            head_dim as u32,
        )?;
    }
    for kv_h in 0..config.n_kv_heads {
        let head_start = kv_h * head_dim;
        let head_end = head_start + head_dim;
        rope(
            &mut k_data[head_start..head_end],
            rope_base,
            rope_pos,
            head_dim as u32,
            head_dim as u32,
        )?;
    }

    kv_cache.append_kv(k_data, v_data)?;

    let total_pos = kv_cache.current_pos();
    let mut attn_out = vec![0.0f32; hidden_dim];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for head in 0..config.n_heads {
        let kv_head = head / group_size;
        let q_start = head * head_dim;
        let q = &q_data[q_start..q_start + head_dim];

        let mut scores = vec![0.0f32; total_pos];
        for (j, score_slot) in scores.iter_mut().enumerate().take(total_pos) {
            let k_vec = kv_cache.get_k_slice(j, kv_head)?;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k_vec[d];
            }
            *score_slot = dot * scale;
        }

        let mut weights_buf = vec![0.0f32; total_pos];
        softmax(&scores, &mut weights_buf)?;

        let out = &mut attn_out[head * head_dim..(head + 1) * head_dim];
        for (j, &w) in weights_buf.iter().enumerate().take(total_pos) {
            let v_vec = kv_cache.get_v_slice(j, kv_head)?;
            for d in 0..head_dim {
                out[d] += w * v_vec[d];
            }
        }
    }

    let attn_tensor = tensor_from_f32_slice(&attn_out, vec![1, hidden_dim]);
    let mut projected = empty_f32_tensor(vec![1, hidden_dim]);
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

#[cfg(test)]
mod unpack_tests {
    use super::unpack_llama_gguf_qk_row;

    #[test]
    fn unpack_restores_hf_qk_head_layout() {
        // Two heads × dim 4; simulate GGUF row layout (permute) holding logical channel values.
        let mut row = vec![
            0., 2., 1., 3., // head 0
            4., 6., 5., 7., // head 1
        ];
        unpack_llama_gguf_qk_row(&mut row, 2, 4);
        assert_eq!(row, vec![0., 1., 2., 3., 4., 5., 6., 7.]);
    }
}