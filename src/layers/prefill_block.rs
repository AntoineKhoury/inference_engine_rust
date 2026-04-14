use std::sync::Arc;

use crate::core::tensor::{Tensor, TensorType};
use crate::layers::attention::{decode_attention_layer, prefill_attention_layer, KVCache};
use crate::model_config::ModelConfig;
use crate::model_weights::LayerWeights;
use crate::ops::matmul::matmul;
use crate::ops::residual_add::residual_add;
use crate::ops::rmsnorm::rmsnorm;
use crate::ops::swiglu::swiglu;
use crate::prefill::PrefillState;

pub fn prefill_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim {
        return Err(format!(
            "attn_norm weights len {} != hidden_dim {}",
            attn_norm_weights.len(),
            hidden_dim
        )
        .into());
    }

    let mut normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rmsnorm(
            &input.hidden()[start..end],
            attn_norm_weights,
            config.rms_norm_eps,
            &mut normed[start..end],
        )?;
    }

    let normed_state = PrefillState::from_flat(normed, seq_len, hidden_dim)?;
    let attn_out = prefill_attention_layer(&normed_state, config, weights, kv_cache)?;

    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

pub fn prefill_ffn(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if input.len() != seq_len * hidden_dim {
        return Err("prefill_ffn: input length does not match shape".into());
    }
    if hidden_dim != config.hidden_dim {
        return Err("prefill_ffn: hidden_dim mismatch with config".into());
    }

    let input_tensor = tensor_from_f32_slice(input, vec![seq_len, hidden_dim]);

    let mut gate_tensor = empty_f32_tensor(vec![seq_len, config.ffn_dim]);
    let mut up_tensor = empty_f32_tensor(vec![seq_len, config.ffn_dim]);

    matmul(&input_tensor, weights.w_gate, &mut gate_tensor)?;
    matmul(&input_tensor, weights.w_up, &mut up_tensor)?;

    let gate = gate_tensor.as_f32_slice()?;
    let up = up_tensor.as_f32_slice()?;
    let mut activated = vec![0.0f32; gate.len()];
    swiglu(gate, up, &mut activated)?;

    let activated_tensor = tensor_from_f32_slice(&activated, vec![seq_len, config.ffn_dim]);
    let mut down_tensor = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&activated_tensor, weights.w_down, &mut down_tensor)?;

    Ok(down_tensor.as_f32_slice()?.to_vec())
}

pub fn prefill_ffn_with_norm(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let ffn_norm_weights = weights.ffn_norm.as_f32_slice()?;
    if ffn_norm_weights.len() != hidden_dim {
        return Err(format!(
            "ffn_norm weights len {} != hidden_dim {}",
            ffn_norm_weights.len(),
            hidden_dim
        )
        .into());
    }

    let mut normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rmsnorm(
            &input[start..end],
            ffn_norm_weights,
            config.rms_norm_eps,
            &mut normed[start..end],
        )?;
    }

    let ffn_out = prefill_ffn(&normed, seq_len, hidden_dim, config, weights)?;
    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input, &ffn_out, &mut residual_out)?;
    Ok(residual_out)
}

pub fn prefill_layer_block(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<PrefillState, Box<dyn std::error::Error>> {
    let attn_out = prefill_attention_with_norm(input, config, weights, kv_cache)?;
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let ffn_out = prefill_ffn_with_norm(&attn_out, seq_len, hidden_dim, config, weights)?;
    PrefillState::from_flat(ffn_out, seq_len, hidden_dim)
}

pub fn decode_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if input.seq_len() != 1 {
        return Err("decode_attention_with_norm: seq_len must be 1".into());
    }
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim {
        return Err(format!(
            "attn_norm weights len {} != hidden_dim {}",
            attn_norm_weights.len(),
            hidden_dim
        )
        .into());
    }

    let mut normed = vec![0.0f32; hidden_dim];
    rmsnorm(
        input.hidden(),
        attn_norm_weights,
        config.rms_norm_eps,
        &mut normed,
    )?;

    let normed_state = PrefillState::from_flat(normed, 1, hidden_dim)?;
    let attn_out = decode_attention_layer(&normed_state, config, weights, kv_cache)?;

    let mut residual_out = vec![0.0f32; hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

pub fn decode_layer_block(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &LayerWeights,
    kv_cache: &mut KVCache,
) -> Result<PrefillState, Box<dyn std::error::Error>> {
    if input.seq_len() != 1 {
        return Err("decode_layer_block: seq_len must be 1".into());
    }
    let hidden_dim = input.hidden_dim();
    let attn_out = decode_attention_with_norm(input, config, weights, kv_cache)?;
    let ffn_out = prefill_ffn_with_norm(&attn_out, 1, hidden_dim, config, weights)?;
    PrefillState::from_flat(ffn_out, 1, hidden_dim)
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
