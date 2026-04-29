use std::sync::Arc;

use crate::EngineError;
use crate::core::tensor::{Tensor, TensorType};
use crate::layers::attention::KVCache;
use crate::layers::prefill_block::{decode_layer_block, prefill_layer_block};
use crate::model_config::ModelConfig;
use crate::model_weights::ModelWeights;
use crate::ops::matmul::matmul;
use crate::ops::rmsnorm::rmsnorm;
use crate::prefill::PrefillState;

/// Run the transformer stack over prompt activations and populate KV caches.
pub fn prefill_forward(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, EngineError> {
    if kv_caches.len() != weights.layers.len() {
        return Err(EngineError::Model(
            "prefill_forward: kv_caches len != number of layers".into(),
        ));
    }

    let mut state = input.replace_hidden(input.hidden().to_vec())?;

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        state = prefill_layer_block(&state, config, layer_idx, layer_weights, kv_caches)?;
    }

    Ok(state)
}

/// One autoregressive step: `input` must be a single token (`seq_len == 1`). Each layer appends
/// K/V to the corresponding cache; RoPE position is the cache length **before** this step.
pub fn decode_forward(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, EngineError> {
    if input.seq_len() != 1 {
        return Err(EngineError::Model(
            "decode_forward: seq_len must be 1".into(),
        ));
    }
    if kv_caches.len() != weights.layers.len() {
        return Err(EngineError::Model(
            "decode_forward: kv_caches len != number of layers".into(),
        ));
    }

    let mut state = input.replace_hidden(input.hidden().to_vec())?;

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        state = decode_layer_block(&state, config, layer_idx, layer_weights, kv_caches)?;
    }

    Ok(state)
}

/// Run [`decode_forward`] from a single-token embedding row (length `config.hidden_dim`).
/// Prefer [`crate::prefill::prefill_state_for_single_token_loaded`] for Gemma 4 so embeddings are scaled and PLE is populated.
pub fn decode_from_embedding_row(
    embedding_row: Vec<f32>,
    config: &ModelConfig,
    weights: &ModelWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, EngineError> {
    let input = PrefillState::from_flat(embedding_row, 1, config.hidden_dim)?;
    decode_forward(&input, config, weights, kv_caches)
}

pub fn final_logits_last_token(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    if seq_len == 0 {
        return Err(EngineError::Model(
            "final_logits_last_token: empty input".into(),
        ));
    }

    let last_start = (seq_len - 1) * hidden_dim;
    let last_end = last_start + hidden_dim;
    let last_hidden = &input.hidden()[last_start..last_end];

    let norm_weights = weights.output_norm.as_f32_slice()?;
    if norm_weights.len() != hidden_dim {
        return Err(EngineError::Model(format!(
            "final_logits_last_token: output_norm len {} != hidden_dim {}",
            norm_weights.len(),
            hidden_dim
        )));
    }

    let mut normed = vec![0.0f32; hidden_dim];
    rmsnorm(last_hidden, norm_weights, config.rms_norm_eps, &mut normed)?;

    let input_tensor = tensor_from_f32_slice(&normed, vec![1, hidden_dim]);
    let mut logits_tensor = empty_f32_tensor(vec![1, config.vocab_size]);
    matmul(&input_tensor, weights.lm_head, &mut logits_tensor)?;

    let mut logits = logits_tensor.as_f32_slice()?.to_vec();
    if let Some(cap) = config.final_logit_softcapping {
        for z in logits.iter_mut() {
            *z = cap * (*z / cap).tanh();
        }
    }
    Ok(logits)
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
