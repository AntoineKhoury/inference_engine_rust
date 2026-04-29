use std::sync::Arc;
use thiserror::Error;

use crate::core::tensor::{Tensor, TensorType};
use crate::model_config::{LayerAttentionSpec, LayerDims, ModelConfig, ModelFamily};
use crate::model_weights::LayerWeights;
use crate::ops::matmul::matmul;
use crate::ops::quant::utils::{f16_to_f32, f32_to_f16};
use crate::ops::rmsnorm::{rmsnorm, rmsnorm_inplace_no_scale};
use crate::ops::rope::rope;
use crate::ops::softmax::softmax;
use crate::prefill::PrefillState;
use crate::EngineError;

#[allow(dead_code)]
#[derive(Default)]
struct ScoreStats {
    min: f32,
    max: f32,
    sum: f64,
    count: usize,
}

#[allow(dead_code)]
impl ScoreStats {
    fn update(&mut self, x: f32) {
        if self.count == 0 {
            self.min = x;
            self.max = x;
        } else {
            self.min = self.min.min(x);
            self.max = self.max.max(x);
        }
        self.sum += x as f64;
        self.count += 1;
    }

    fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }
}

#[allow(dead_code)]
fn debug_attn_layers_from_env() -> Option<Vec<usize>> {
    let raw = std::env::var("INFERENCE_ENGINE_DEBUG_ATTN_LAYERS").ok()?;
    let mut out = Vec::new();
    for part in raw.split([',', ' ']).filter(|s| !s.is_empty()) {
        if let Ok(v) = part.parse::<usize>() {
            out.push(v);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[allow(dead_code)]
fn should_debug_layer(layer_idx: usize) -> bool {
    debug_attn_layers_from_env()
        .as_ref()
        .is_some_and(|layers| layers.contains(&layer_idx))
}

#[inline]
fn round_f32_to_f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

#[inline]
fn use_f16_kv_cache() -> bool {
    std::env::var("INFERENCE_ENGINE_F16_KV_CACHE")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

#[inline]
fn use_flash_q_f16_dot() -> bool {
    std::env::var("INFERENCE_ENGINE_FLASH_Q_F16_DOT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[inline]
fn use_flash_v_f16_accum() -> bool {
    std::env::var("INFERENCE_ENGINE_FLASH_V_F16_ACCUM")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

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

        if use_f16_kv_cache() {
            for i in 0..expected_len {
                self.k_cache[start_idx + i] = round_f32_to_f16(k[i]);
                self.v_cache[start_idx + i] = round_f32_to_f16(v[i]);
            }
        } else {
            self.k_cache[start_idx..start_idx + expected_len].copy_from_slice(k);
            self.v_cache[start_idx..start_idx + expected_len].copy_from_slice(v);
        }

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
    PositionOutOfBounds { position: usize, current_pos: usize },

    #[error("KV head index {kv_head} is out of bounds (n_kv_heads is {n_kv_heads})")]
    KvHeadOutOfBounds { kv_head: usize, n_kv_heads: usize },
}

/// One [`KVCache`] per layer, sized from [`ModelConfig::layer_dims`] (per-layer head width).
pub fn kv_caches_for_config(config: &ModelConfig) -> Vec<KVCache> {
    config
        .layer_dims
        .iter()
        .map(|d| KVCache::new(config.context_length, config.n_kv_heads, d.head_dim))
        .collect()
}

/// Undo HF→GGUF `LlamaModel.permute` on **one row** of Q or K activations (Llama-style GGUF only;
/// Mistral exports usually set `ModelConfig.unpack_llama_gguf_qk = false`).
///
/// `row` has length `n_groups * head_dim` (Q: `n_heads`; K: `n_kv_heads`). GGUF stores each head as
/// `reshape(n_g, 2, d/2).swapaxes(1,2).flatten` indices `h*2+b`; logical layout is `b*(d/2)+h`.
fn rope_freq_slice<'a>(weights: &'a LayerWeights<'a>) -> Option<&'a [f32]> {
    weights
        .rope_freqs
        .and_then(|t| t.as_f32_slice().ok())
        .filter(|s| !s.is_empty())
}

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

#[allow(clippy::needless_range_loop)]
pub fn prefill_attention_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    layer_attn: &LayerAttentionSpec,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
    layer_idx: usize,
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(format!(
            "prefill attention: hidden_dim {hidden_dim} != config.hidden_dim {}",
            config.hidden_dim
        )));
    }

    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;
    let head_dim = layer_dims.head_dim;

    if q_dim != config.n_heads * head_dim {
        return Err(EngineError::Model(format!(
            "prefill attention: q_dim {q_dim} != n_heads * head_dim ({})",
            config.n_heads * head_dim
        )));
    }
    if kv_dim != config.n_kv_heads * head_dim {
        return Err(EngineError::Model(format!(
            "prefill attention: kv_dim {kv_dim} != n_kv_heads * head_dim ({})",
            config.n_kv_heads * head_dim
        )));
    }

    let own = kv_caches
        .get(layer_idx)
        .ok_or_else(|| EngineError::Model("prefill attention: kv_caches index".into()))?;
    if own.n_kv_heads() != config.n_kv_heads || own.head_dim() != head_dim {
        return Err(EngineError::Model(
            "prefill attention: KVCache n_kv_heads/head_dim does not match layer_dims".into(),
        ));
    }

    let borrow_src = config
        .gemma4_kv_borrow_from
        .get(layer_idx)
        .copied()
        .flatten();
    if let Some(src) = borrow_src {
        let ksrc = kv_caches.get(src).ok_or_else(|| {
            EngineError::Model(format!(
                "prefill attention: KV borrow source {src} out of range"
            ))
        })?;
        if ksrc.n_kv_heads() != config.n_kv_heads || ksrc.head_dim() != head_dim {
            return Err(EngineError::Model(
                "prefill attention: borrow source KV shape mismatch".into(),
            ));
        }
        if ksrc.current_pos() != seq_len {
            return Err(EngineError::Model(format!(
                "prefill attention: borrow layer {layer_idx} expected source cache len {seq_len}, got {}",
                ksrc.current_pos()
            )));
        }
    }

    let group_size = config.n_heads / config.n_kv_heads;

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    let mut v_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);

    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    if borrow_src.is_none() {
        matmul(&input_tensor, weights.wk, &mut k_tensor)?;
        matmul(&input_tensor, weights.wv, &mut v_tensor)?;
    }

    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    let v_data = v_tensor.as_f32_slice_mut()?;

    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * q_dim..(pos + 1) * q_dim],
                config.n_heads,
                head_dim,
            );
            if borrow_src.is_none() {
                unpack_llama_gguf_qk_row(
                    &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                    config.n_kv_heads,
                    head_dim,
                );
            }
        }
    }

    let mut head_scratch = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        apply_optional_head_rmsnorm(
            &mut q_data[pos * q_dim..(pos + 1) * q_dim],
            config.n_heads,
            head_dim,
            weights.attn_q_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
        if borrow_src.is_none() {
            apply_optional_head_rmsnorm(
                &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                config.n_kv_heads,
                head_dim,
                weights.attn_k_norm,
                config.rms_norm_eps,
                &mut head_scratch,
            )?;
        }
    }

    // HF `Gemma4TextAttention`: `v_norm` is Gemma4RMSNorm(..., with_scale=false) on each value head.
    if borrow_src.is_none() && matches!(config.family, ModelFamily::Gemma4) {
        for pos in 0..seq_len {
            let row = &mut v_data[pos * kv_dim..(pos + 1) * kv_dim];
            for h in 0..config.n_kv_heads {
                let s = h * head_dim;
                rmsnorm_inplace_no_scale(&mut row[s..s + head_dim], config.rms_norm_eps);
            }
        }
    }

    let rope_base = layer_attn.rope_theta;
    let rope_rotary = layer_attn.rope_rotary_dim as u32;
    let rope_ff = rope_freq_slice(weights);

    for pos in 0..seq_len {
        let q_row = pos * q_dim;
        for head in 0..config.n_heads {
            let head_start = q_row + head * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut q_data[head_start..head_end],
                rope_base,
                pos as u32,
                head_dim as u32,
                rope_rotary,
                rope_ff,
            )?;
        }
        if borrow_src.is_none() {
            let k_row = pos * kv_dim;
            for kv_h in 0..config.n_kv_heads {
                let head_start = k_row + kv_h * head_dim;
                let head_end = head_start + head_dim;
                rope(
                    &mut k_data[head_start..head_end],
                    rope_base,
                    pos as u32,
                    head_dim as u32,
                    rope_rotary,
                    rope_ff,
                )?;
            }
        }
    }

    if borrow_src.is_none() {
        for pos in 0..seq_len {
            let row_start = pos * kv_dim;
            let row_end = row_start + kv_dim;
            kv_caches[layer_idx]
                .append_kv(&k_data[row_start..row_end], &v_data[row_start..row_end])?;
        }
    }

    let use_flash = std::env::var("INFERENCE_ENGINE_FLASH_ATTN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    let mut attn_out = vec![0.0f32; seq_len * q_dim];
    let scale = match config.family {
        ModelFamily::Gemma4 => 1.0f32,
        ModelFamily::MistralLlama => 1.0f32 / (head_dim as f32).sqrt(),
    };

    let src_idx = borrow_src.unwrap_or(layer_idx);
    let flash_q_f16_dot = use_flash_q_f16_dot();
    let flash_v_f16_accum = use_flash_v_f16_accum();

    for pos in 0..seq_len {
        let j_min = layer_attn
            .sliding_window
            .map(|w| pos.saturating_sub(w.saturating_sub(1)))
            .unwrap_or(0);
        for head in 0..config.n_heads {
            let kv_head = head / group_size;
            let q_start = pos * q_dim + head * head_dim;
            let q = &q_data[q_start..q_start + head_dim];

            let out_start = pos * q_dim + head * head_dim;
            let out = &mut attn_out[out_start..out_start + head_dim];

            if use_flash {
                let mut m = f32::NEG_INFINITY;
                let mut l = 0.0f32;

                for j in j_min..=pos {
                    let k_vec = kv_caches[src_idx].get_k_slice(j, kv_head)?;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let qd = if flash_q_f16_dot {
                            round_f32_to_f16(q[d])
                        } else {
                            q[d]
                        };
                        dot += qd * k_vec[d];
                    }
                    let score = dot * scale;

                    let m_old = m;
                    let mut ms = 1.0f32;
                    let mut vs = 1.0f32;

                    if score > m {
                        m = score;
                        ms = (m_old - m).exp();
                    } else {
                        vs = (score - m).exp();
                    }

                    let v_vec = kv_caches[src_idx].get_v_slice(j, kv_head)?;
                    for d in 0..head_dim {
                        out[d] = if flash_v_f16_accum {
                            let scaled = round_f32_to_f16(out[d] * ms);
                            round_f32_to_f16(scaled + vs * v_vec[d])
                        } else {
                            out[d] * ms + vs * v_vec[d]
                        };
                    }
                    l = l * ms + vs;
                }

                if l > 0.0 {
                    let inv_l = 1.0 / l;
                    for d in 0..head_dim {
                        out[d] *= inv_l;
                    }
                }
            } else {
                let mut scores = vec![f32::NEG_INFINITY; pos + 1];
                for j in j_min..=pos {
                    let k_vec = kv_caches[src_idx].get_k_slice(j, kv_head)?;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[d] * k_vec[d];
                    }
                    scores[j] = dot * scale;
                }
                let mut weights_buf = vec![0.0f32; pos + 1];
                softmax(&scores, &mut weights_buf)?;
                for j in j_min..=pos {
                    let w = weights_buf[j];
                    let v_vec = kv_caches[src_idx].get_v_slice(j, kv_head)?;
                    for d in 0..head_dim {
                        out[d] += w * v_vec[d];
                    }
                }
            }
        }
    }

    let attn_tensor = tensor_from_f32_slice(&attn_out, vec![seq_len, q_dim]);
    let mut projected = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&attn_tensor, weights.wo, &mut projected)?;

    Ok(projected.as_f32_slice()?.to_vec())
}

/// Same core as [`prefill_attention_layer`] but returns the pre-projection attention output
/// (`[seq_len, q_dim]`, equivalent to llama `kqv_out-*` before `wo` matmul).
pub fn prefill_attention_core_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    layer_attn: &LayerAttentionSpec,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
    layer_idx: usize,
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let head_dim = layer_dims.head_dim;
    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;

    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(format!(
            "prefill attention(core): hidden_dim {hidden_dim} != config.hidden_dim {}",
            config.hidden_dim
        )));
    }
    if q_dim != config.n_heads * head_dim || kv_dim != config.n_kv_heads * head_dim {
        return Err(EngineError::Model(
            "prefill attention(core): layer_dims inconsistent with head counts".into(),
        ));
    }

    let own = kv_caches
        .get(layer_idx)
        .ok_or_else(|| EngineError::Model("prefill attention(core): kv_caches index".into()))?;
    if own.n_kv_heads() != config.n_kv_heads || own.head_dim() != head_dim {
        return Err(EngineError::Model(
            "prefill attention(core): KVCache n_kv_heads/head_dim mismatch".into(),
        ));
    }

    let borrow_src = config
        .gemma4_kv_borrow_from
        .get(layer_idx)
        .copied()
        .flatten();
    if let Some(src) = borrow_src {
        let ksrc = kv_caches.get(src).ok_or_else(|| {
            EngineError::Model(format!(
                "prefill attention(core): KV borrow source {src} out of range"
            ))
        })?;
        if ksrc.n_kv_heads() != config.n_kv_heads || ksrc.head_dim() != head_dim {
            return Err(EngineError::Model(
                "prefill attention(core): borrow source KV shape mismatch".into(),
            ));
        }
        if ksrc.current_pos() != seq_len {
            return Err(EngineError::Model(format!(
                "prefill attention(core): borrow layer {layer_idx} expected source cache len {seq_len}, got {}",
                ksrc.current_pos()
            )));
        }
    }

    let group_size = config.n_heads / config.n_kv_heads;
    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    let mut v_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);

    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    if borrow_src.is_none() {
        matmul(&input_tensor, weights.wk, &mut k_tensor)?;
        matmul(&input_tensor, weights.wv, &mut v_tensor)?;
    }

    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    let v_data = v_tensor.as_f32_slice_mut()?;

    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * q_dim..(pos + 1) * q_dim],
                config.n_heads,
                head_dim,
            );
            if borrow_src.is_none() {
                unpack_llama_gguf_qk_row(
                    &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                    config.n_kv_heads,
                    head_dim,
                );
            }
        }
    }

    let mut head_scratch = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        apply_optional_head_rmsnorm(
            &mut q_data[pos * q_dim..(pos + 1) * q_dim],
            config.n_heads,
            head_dim,
            weights.attn_q_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
        if borrow_src.is_none() {
            apply_optional_head_rmsnorm(
                &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                config.n_kv_heads,
                head_dim,
                weights.attn_k_norm,
                config.rms_norm_eps,
                &mut head_scratch,
            )?;
        }
    }

    if borrow_src.is_none() && matches!(config.family, ModelFamily::Gemma4) {
        for pos in 0..seq_len {
            let row = &mut v_data[pos * kv_dim..(pos + 1) * kv_dim];
            for h in 0..config.n_kv_heads {
                let s = h * head_dim;
                rmsnorm_inplace_no_scale(&mut row[s..s + head_dim], config.rms_norm_eps);
            }
        }
    }

    let rope_base = layer_attn.rope_theta;
    let rope_rotary = layer_attn.rope_rotary_dim as u32;
    let rope_ff = rope_freq_slice(weights);
    for pos in 0..seq_len {
        let q_row = pos * q_dim;
        for head in 0..config.n_heads {
            let head_start = q_row + head * head_dim;
            rope(
                &mut q_data[head_start..head_start + head_dim],
                rope_base,
                pos as u32,
                head_dim as u32,
                rope_rotary,
                rope_ff,
            )?;
        }
        if borrow_src.is_none() {
            let k_row = pos * kv_dim;
            for kv_h in 0..config.n_kv_heads {
                let head_start = k_row + kv_h * head_dim;
                rope(
                    &mut k_data[head_start..head_start + head_dim],
                    rope_base,
                    pos as u32,
                    head_dim as u32,
                    rope_rotary,
                    rope_ff,
                )?;
            }
        }
    }

    if borrow_src.is_none() {
        for pos in 0..seq_len {
            let row_start = pos * kv_dim;
            let row_end = row_start + kv_dim;
            kv_caches[layer_idx]
                .append_kv(&k_data[row_start..row_end], &v_data[row_start..row_end])?;
        }
    }

    let gemma_inv_sqrt = std::env::var("INFERENCE_ENGINE_GEMMA4_ATTN_SCALE_INV_SQRT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let scale = match config.family {
        ModelFamily::Gemma4 if gemma_inv_sqrt => 1.0f32 / (head_dim as f32).sqrt(),
        ModelFamily::Gemma4 => 1.0f32,
        ModelFamily::MistralLlama => 1.0f32 / (head_dim as f32).sqrt(),
    };
    let src_idx = borrow_src.unwrap_or(layer_idx);
    let flash_q_f16_dot = use_flash_q_f16_dot();
    let flash_v_f16_accum = use_flash_v_f16_accum();

    let mut attn_out = vec![0.0f32; seq_len * q_dim];
    for pos in 0..seq_len {
        let j_min = layer_attn
            .sliding_window
            .map(|w| pos.saturating_sub(w.saturating_sub(1)))
            .unwrap_or(0);
        for head in 0..config.n_heads {
            let kv_head = head / group_size;
            let q_start = pos * q_dim + head * head_dim;
            let q = &q_data[q_start..q_start + head_dim];

            let out_start = pos * q_dim + head * head_dim;
            let out = &mut attn_out[out_start..out_start + head_dim];

            let mut m = f32::NEG_INFINITY;
            let mut l = 0.0f32;

            for j in j_min..=pos {
                let k_vec = kv_caches[src_idx].get_k_slice(j, kv_head)?;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let qd = if flash_q_f16_dot {
                        round_f32_to_f16(q[d])
                    } else {
                        q[d]
                    };
                    dot += qd * k_vec[d];
                }
                let score = dot * scale;

                let m_old = m;
                let mut ms = 1.0f32;
                let mut vs = 1.0f32;

                if score > m {
                    m = score;
                    ms = (m_old - m).exp();
                } else {
                    vs = (score - m).exp();
                }

                let v_vec = kv_caches[src_idx].get_v_slice(j, kv_head)?;
                for d in 0..head_dim {
                    out[d] = if flash_v_f16_accum {
                        let scaled = round_f32_to_f16(out[d] * ms);
                        round_f32_to_f16(scaled + vs * v_vec[d])
                    } else {
                        out[d] * ms + vs * v_vec[d]
                    };
                }
                l = l * ms + vs;
            }

            if l > 0.0 {
                let inv_l = 1.0 / l;
                for d in 0..head_dim {
                    out[d] *= inv_l;
                }
            }
        }
    }
    Ok(attn_out)
}

/// Return Q and K tensors after q/k RMSNorm + RoPE, before cache write and attention matmul.
/// Shapes are flattened `[seq_len, q_dim]` and `[seq_len, kv_dim]`.
pub fn prefill_qk_after_rope_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    layer_attn: &LayerAttentionSpec,
    weights: &LayerWeights,
    kv_caches: &[KVCache],
    layer_idx: usize,
) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let head_dim = layer_dims.head_dim;
    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_qk_after_rope: hidden_dim mismatch".into(),
        ));
    }
    if q_dim != config.n_heads * head_dim || kv_dim != config.n_kv_heads * head_dim {
        return Err(EngineError::Model(
            "prefill_qk_after_rope: layer_dims inconsistent with head counts".into(),
        ));
    }
    let borrow_src = config
        .gemma4_kv_borrow_from
        .get(layer_idx)
        .copied()
        .flatten();
    if let Some(src) = borrow_src {
        let ksrc = kv_caches.get(src).ok_or_else(|| {
            EngineError::Model(format!(
                "prefill_qk_after_rope: KV borrow source {src} out of range"
            ))
        })?;
        if ksrc.current_pos() != seq_len {
            return Err(EngineError::Model(
                "prefill_qk_after_rope: borrowed layer unsupported for direct K extraction".into(),
            ));
        }
    }

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    if borrow_src.is_none() {
        matmul(&input_tensor, weights.wk, &mut k_tensor)?;
    }
    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * q_dim..(pos + 1) * q_dim],
                config.n_heads,
                head_dim,
            );
            if borrow_src.is_none() {
                unpack_llama_gguf_qk_row(
                    &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                    config.n_kv_heads,
                    head_dim,
                );
            }
        }
    }
    let mut head_scratch = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        apply_optional_head_rmsnorm(
            &mut q_data[pos * q_dim..(pos + 1) * q_dim],
            config.n_heads,
            head_dim,
            weights.attn_q_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
        if borrow_src.is_none() {
            apply_optional_head_rmsnorm(
                &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
                config.n_kv_heads,
                head_dim,
                weights.attn_k_norm,
                config.rms_norm_eps,
                &mut head_scratch,
            )?;
        }
    }

    let rope_base = layer_attn.rope_theta;
    let rope_rotary = layer_attn.rope_rotary_dim as u32;
    let rope_ff = rope_freq_slice(weights);
    for pos in 0..seq_len {
        let q_row = pos * q_dim;
        for head in 0..config.n_heads {
            let hs = q_row + head * head_dim;
            rope(
                &mut q_data[hs..hs + head_dim],
                rope_base,
                pos as u32,
                head_dim as u32,
                rope_rotary,
                rope_ff,
            )?;
        }
        if borrow_src.is_none() {
            let k_row = pos * kv_dim;
            for kv_h in 0..config.n_kv_heads {
                let hs = k_row + kv_h * head_dim;
                rope(
                    &mut k_data[hs..hs + head_dim],
                    rope_base,
                    pos as u32,
                    head_dim as u32,
                    rope_rotary,
                    rope_ff,
                )?;
            }
        }
    }
    Ok((q_data.to_vec(), k_data.to_vec()))
}

/// Return Q and K tensors after q/k RMSNorm, before RoPE.
pub fn prefill_qk_normed_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    weights: &LayerWeights,
) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let head_dim = layer_dims.head_dim;
    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_qk_normed: hidden_dim mismatch".into(),
        ));
    }

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    matmul(&input_tensor, weights.wk, &mut k_tensor)?;
    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * q_dim..(pos + 1) * q_dim],
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

    let mut head_scratch = vec![0.0f32; head_dim];
    for pos in 0..seq_len {
        apply_optional_head_rmsnorm(
            &mut q_data[pos * q_dim..(pos + 1) * q_dim],
            config.n_heads,
            head_dim,
            weights.attn_q_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
        apply_optional_head_rmsnorm(
            &mut k_data[pos * kv_dim..(pos + 1) * kv_dim],
            config.n_kv_heads,
            head_dim,
            weights.attn_k_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
    }
    Ok((q_data.to_vec(), k_data.to_vec()))
}

/// Return raw Q and K projections before q/k norm and RoPE.
pub fn prefill_qk_raw_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    weights: &LayerWeights,
) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let head_dim = layer_dims.head_dim;
    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_qk_raw: hidden_dim mismatch".into(),
        ));
    }
    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![seq_len, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![seq_len, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![seq_len, kv_dim]);
    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    matmul(&input_tensor, weights.wk, &mut k_tensor)?;
    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    if config.unpack_llama_gguf_qk {
        for pos in 0..seq_len {
            unpack_llama_gguf_qk_row(
                &mut q_data[pos * q_dim..(pos + 1) * q_dim],
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
    Ok((q_data.to_vec(), k_data.to_vec()))
}

fn apply_optional_head_rmsnorm(
    row: &mut [f32],
    n_groups: usize,
    head_dim: usize,
    norm: Option<&Tensor>,
    eps: f32,
    scratch: &mut [f32],
) -> Result<(), EngineError> {
    let Some(t) = norm else {
        return Ok(());
    };
    let w = t.as_f32_slice()?;
    if w.len() != head_dim {
        return Err(EngineError::Model(format!(
            "attn q/k norm weight len {} != head_dim {}",
            w.len(),
            head_dim
        )));
    }
    if scratch.len() < head_dim {
        return Err(EngineError::Model(
            "internal: head norm scratch shorter than head_dim".into(),
        ));
    }
    let tmp = &mut scratch[..head_dim];
    for g in 0..n_groups {
        let s = g * head_dim;
        rmsnorm(&row[s..s + head_dim], w, eps, tmp)?;
        row[s..s + head_dim].copy_from_slice(tmp);
    }
    Ok(())
}

/// Single-token attention for autoregressive decode. `input` must have `seq_len == 1`.
///
/// RoPE uses position `kv_cache.current_pos` (0-based index of this token in the full sequence).
/// Past keys/values are read from `kv_cache`; the new K/V are appended after RoPE.
#[allow(clippy::needless_range_loop)]
pub fn decode_attention_layer(
    input: &PrefillState,
    config: &ModelConfig,
    layer_dims: &LayerDims,
    layer_attn: &LayerAttentionSpec,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
    layer_idx: usize,
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

    let q_dim = layer_dims.q_dim;
    let kv_dim = layer_dims.kv_dim;
    let head_dim = layer_dims.head_dim;

    if q_dim != config.n_heads * head_dim || kv_dim != config.n_kv_heads * head_dim {
        return Err(EngineError::Model(
            "decode attention: layer_dims inconsistent with head counts".into(),
        ));
    }

    let own = kv_caches
        .get(layer_idx)
        .ok_or_else(|| EngineError::Model("decode attention: kv_caches index".into()))?;
    if own.n_kv_heads() != config.n_kv_heads || own.head_dim() != head_dim {
        return Err(EngineError::Model(
            "decode attention: KVCache n_kv_heads/head_dim does not match layer_dims".into(),
        ));
    }

    let borrow_src = config
        .gemma4_kv_borrow_from
        .get(layer_idx)
        .copied()
        .flatten();

    let rope_pos = if let Some(src) = borrow_src {
        let n = kv_caches[src].current_pos();
        if n == 0 {
            return Err(EngineError::Model(
                "decode attention: KV borrow source cache empty (rope_pos)".into(),
            ));
        }
        (n - 1) as u32
    } else {
        own.current_pos() as u32
    };

    let group_size = config.n_heads / config.n_kv_heads;

    let input_tensor = tensor_from_f32_slice(input.hidden(), vec![1, hidden_dim]);
    let mut q_tensor = empty_f32_tensor(vec![1, q_dim]);
    let mut k_tensor = empty_f32_tensor(vec![1, kv_dim]);
    let mut v_tensor = empty_f32_tensor(vec![1, kv_dim]);

    matmul(&input_tensor, weights.wq, &mut q_tensor)?;
    if borrow_src.is_none() {
        matmul(&input_tensor, weights.wk, &mut k_tensor)?;
        matmul(&input_tensor, weights.wv, &mut v_tensor)?;
    }

    let q_data = q_tensor.as_f32_slice_mut()?;
    let k_data = k_tensor.as_f32_slice_mut()?;
    let v_data = v_tensor.as_f32_slice_mut()?;

    if config.unpack_llama_gguf_qk {
        unpack_llama_gguf_qk_row(q_data, config.n_heads, head_dim);
        if borrow_src.is_none() {
            unpack_llama_gguf_qk_row(k_data, config.n_kv_heads, head_dim);
        }
    }

    let mut head_scratch = vec![0.0f32; head_dim];
    apply_optional_head_rmsnorm(
        q_data,
        config.n_heads,
        head_dim,
        weights.attn_q_norm,
        config.rms_norm_eps,
        &mut head_scratch,
    )?;
    if borrow_src.is_none() {
        apply_optional_head_rmsnorm(
            k_data,
            config.n_kv_heads,
            head_dim,
            weights.attn_k_norm,
            config.rms_norm_eps,
            &mut head_scratch,
        )?;
    }

    if borrow_src.is_none() && matches!(config.family, ModelFamily::Gemma4) {
        for h in 0..config.n_kv_heads {
            let s = h * head_dim;
            rmsnorm_inplace_no_scale(&mut v_data[s..s + head_dim], config.rms_norm_eps);
        }
    }

    let rope_base = layer_attn.rope_theta;
    let rope_rotary = layer_attn.rope_rotary_dim as u32;
    let rope_ff = rope_freq_slice(weights);

    for head in 0..config.n_heads {
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        rope(
            &mut q_data[head_start..head_end],
            rope_base,
            rope_pos,
            head_dim as u32,
            rope_rotary,
            rope_ff,
        )?;
    }
    if borrow_src.is_none() {
        for kv_h in 0..config.n_kv_heads {
            let head_start = kv_h * head_dim;
            let head_end = head_start + head_dim;
            rope(
                &mut k_data[head_start..head_end],
                rope_base,
                rope_pos,
                head_dim as u32,
                rope_rotary,
                rope_ff,
            )?;
        }
    }

    if borrow_src.is_none() {
        kv_caches[layer_idx].append_kv(k_data, v_data)?;
    }

    let src_idx = borrow_src.unwrap_or(layer_idx);
    let total_pos = kv_caches[src_idx].current_pos();
    let j_min = layer_attn
        .sliding_window
        .map(|w| total_pos.saturating_sub(w))
        .unwrap_or(0);
    let mut attn_out = vec![0.0f32; q_dim];
    let scale = match config.family {
        ModelFamily::Gemma4 => 1.0f32,
        ModelFamily::MistralLlama => 1.0f32 / (head_dim as f32).sqrt(),
    };
    let flash_q_f16_dot = use_flash_q_f16_dot();
    let flash_v_f16_accum = use_flash_v_f16_accum();

    for head in 0..config.n_heads {
        let kv_head = head / group_size;
        let q_start = head * head_dim;
        let q = &q_data[q_start..q_start + head_dim];

        let out = &mut attn_out[head * head_dim..(head + 1) * head_dim];
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;

        for j in j_min..total_pos {
            let k_vec = kv_caches[src_idx].get_k_slice(j, kv_head)?;
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                let qd = if flash_q_f16_dot {
                    round_f32_to_f16(q[d])
                } else {
                    q[d]
                };
                dot += qd * k_vec[d];
            }
            let score = dot * scale;

            let m_old = m;
            let mut ms = 1.0f32;
            let mut vs = 1.0f32;

            if score > m {
                m = score;
                ms = (m_old - m).exp();
            } else {
                vs = (score - m).exp();
            }

            let v_vec = kv_caches[src_idx].get_v_slice(j, kv_head)?;
            for d in 0..head_dim {
                out[d] = if flash_v_f16_accum {
                    let scaled = round_f32_to_f16(out[d] * ms);
                    round_f32_to_f16(scaled + vs * v_vec[d])
                } else {
                    out[d] * ms + vs * v_vec[d]
                };
            }
            l = l * ms + vs;
        }

        if l > 0.0 {
            let inv_l = 1.0 / l;
            for d in 0..head_dim {
                out[d] *= inv_l;
            }
        }
    }

    let attn_tensor = tensor_from_f32_slice(&attn_out, vec![1, q_dim]);
    let mut projected = empty_f32_tensor(vec![1, hidden_dim]);
    matmul(&attn_tensor, weights.wo, &mut projected)?;

    Ok(projected.as_f32_slice()?.to_vec())
}

fn tensor_from_f32_slice(data: &[f32], dimensions: Vec<usize>) -> Tensor {
    Tensor::new(TensorType::F32, Arc::new(f32_bytes(data)), dimensions)
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
