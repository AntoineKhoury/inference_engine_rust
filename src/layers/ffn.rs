//! FFN sub-layer: gate/up/down projections, activation, norm wrappers, and Gemma 4 PLE tail.

use std::sync::Arc;

use crate::EngineError;
use crate::core::tensor::{Tensor, TensorType};
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_weights::LayerWeights;
use crate::ops::gelu::gelu_tanh;
use crate::ops::matmul::matmul;
use crate::ops::quant::quant_k_handler::{Q8_0_BLOCK_SIZE, dequantize_q8_0_block};
use crate::ops::residual_add::residual_add;
use crate::ops::rmsnorm::rmsnorm;
use crate::ops::swiglu::swiglu;

pub fn prefill_ffn(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    if input.len() != seq_len * hidden_dim {
        return Err(EngineError::Model(
            "prefill_ffn: input length does not match shape".into(),
        ));
    }
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_ffn: hidden_dim mismatch with config".into(),
        ));
    }

    let input_tensor = tensor_from_f32_slice(input, vec![seq_len, hidden_dim]);

    let mut gate_tensor = empty_f32_tensor(vec![seq_len, ffn_dim]);
    let mut up_tensor = empty_f32_tensor(vec![seq_len, ffn_dim]);

    matmul(&input_tensor, weights.w_gate, &mut gate_tensor)?;
    matmul(&input_tensor, weights.w_up, &mut up_tensor)?;

    let gate = gate_tensor.as_f32_slice()?;
    let up = up_tensor.as_f32_slice()?;
    let mut activated = vec![0.0f32; gate.len()];
    match config.family {
        // HF `Gemma4TextMLP`: `down_proj(act_fn(gate_proj(x)) * up_proj(x))` with
        // `hidden_activation="gelu_pytorch_tanh"`.
        ModelFamily::Gemma4 => {
            for i in 0..gate.len() {
                activated[i] = gelu_tanh(gate[i]) * up[i];
            }
        }
        ModelFamily::MistralLlama => {
            swiglu(gate, up, &mut activated)?;
        }
    }

    let activated_tensor = tensor_from_f32_slice(&activated, vec![seq_len, ffn_dim]);
    let mut down_tensor = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&activated_tensor, weights.w_down, &mut down_tensor)?;

    Ok(down_tensor.as_f32_slice()?.to_vec())
}

pub fn prefill_ffn_with_norm(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    match config.family {
        ModelFamily::MistralLlama => {
            mistral_prefill_ffn_with_norm(input, seq_len, hidden_dim, ffn_dim, config, weights)
        }
        ModelFamily::Gemma4 => {
            gemma4_prefill_ffn_with_norm(input, seq_len, hidden_dim, ffn_dim, config, weights)
        }
    }
}

fn mistral_prefill_ffn_with_norm(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    let ffn_norm_weights = weights.ffn_norm.as_f32_slice()?;
    if ffn_norm_weights.len() != hidden_dim {
        return Err(EngineError::Model(format!(
            "ffn_norm weights len {} != hidden_dim {}",
            ffn_norm_weights.len(),
            hidden_dim
        )));
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

    let ffn_out = prefill_ffn(&normed, seq_len, hidden_dim, ffn_dim, config, weights)?;
    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input, &ffn_out, &mut residual_out)?;
    Ok(residual_out)
}

fn gemma4_prefill_ffn_with_norm(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    let ffn_norm_weights = weights.ffn_norm.as_f32_slice()?;
    let post_ffn_w = weights
        .ffn_post_norm
        .ok_or_else(|| EngineError::Model("Gemma 4: missing post_ffw_norm".into()))?
        .as_f32_slice()?;
    if ffn_norm_weights.len() != hidden_dim || post_ffn_w.len() != hidden_dim {
        return Err(EngineError::Model(
            "Gemma 4: ffn norm weight length mismatch".into(),
        ));
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

    let mut ffn_out = prefill_ffn(&normed, seq_len, hidden_dim, ffn_dim, config, weights)?;
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        let mut tmp = vec![0.0f32; hidden_dim];
        rmsnorm(
            &ffn_out[start..end],
            post_ffn_w,
            config.rms_norm_eps,
            &mut tmp,
        )?;
        ffn_out[start..end].copy_from_slice(&tmp);
    }

    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input, &ffn_out, &mut residual_out)?;
    Ok(residual_out)
}

/// Apply Gemma 4 per-layer embedding tail: gate projection → GELU × PLE slice → proj → norm + residual.
pub(super) fn apply_per_layer_tail(
    hidden: &mut [f32],
    seq_len: usize,
    hidden_dim: usize,
    layer_idx: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
    ple_packed: &[f32],
) -> Result<(), EngineError> {
    let ple_dim = config.embedding_length_per_layer;
    if ple_dim == 0 {
        return Ok(());
    }
    let n_layers = config.n_layers;
    let pack = n_layers
        .checked_mul(ple_dim)
        .ok_or_else(|| EngineError::Model("PLE: pack overflow".into()))?;
    if ple_packed.len() != seq_len * pack {
        return Err(EngineError::Model(format!(
            "PLE packed len {} != seq_len * pack {}",
            ple_packed.len(),
            seq_len * pack
        )));
    }

    let gate = weights
        .ple_inp_gate
        .ok_or_else(|| EngineError::Model("PLE: missing inp_gate".into()))?;
    let proj = weights
        .ple_proj
        .ok_or_else(|| EngineError::Model("PLE: missing proj".into()))?;
    let post_n = weights
        .ple_post_norm
        .ok_or_else(|| EngineError::Model("PLE: missing post_norm".into()))?;

    let in_t = tensor_from_f32_slice(hidden, vec![seq_len, hidden_dim]);
    let mut gate_t = empty_f32_tensor(vec![seq_len, ple_dim]);
    matmul(&in_t, gate, &mut gate_t)?;
    let mut go = gate_t.as_f32_slice()?.to_vec();

    for p in 0..seq_len {
        let base = p * ple_dim;
        let pb = p * pack + layer_idx * ple_dim;
        for i in 0..ple_dim {
            let x = go[base + i];
            go[base + i] = gelu_tanh(x) * ple_packed[pb + i];
        }
    }

    let go_t = tensor_from_f32_slice(&go, vec![seq_len, ple_dim]);
    let mut out_t = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&go_t, proj, &mut out_t)?;
    let proj_out = out_t.as_f32_slice()?.to_vec();

    let w_post = post_n.as_f32_slice()?;
    if w_post.len() != hidden_dim {
        return Err(EngineError::Model(
            "PLE post_norm weight len mismatch".into(),
        ));
    }
    let eps = config.rms_norm_eps;
    let mut normed_row = vec![0.0f32; hidden_dim];
    for p in 0..seq_len {
        let h0 = p * hidden_dim;
        rmsnorm(&proj_out[h0..h0 + hidden_dim], w_post, eps, &mut normed_row)?;
        for i in 0..hidden_dim {
            hidden[h0 + i] += normed_row[i];
        }
    }
    Ok(())
}

pub(super) fn apply_gemma_layer_output_scale(
    hidden: &mut [f32],
    scale: Option<&Tensor>,
) -> Result<(), EngineError> {
    let Some(t) = scale else {
        return Ok(());
    };
    let s = layer_output_scale_as_f32(t)?;
    for v in hidden.iter_mut() {
        *v *= s;
    }
    Ok(())
}

fn layer_output_scale_as_f32(t: &Tensor) -> Result<f32, EngineError> {
    match t.dtype() {
        TensorType::F32 => t
            .as_f32_slice()?
            .first()
            .copied()
            .ok_or_else(|| EngineError::Model("layer_output_scale: empty F32 tensor".into())),
        TensorType::Q8_0 => {
            let b = t.buffer();
            if b.len() < Q8_0_BLOCK_SIZE {
                return Err(EngineError::Model(
                    "layer_output_scale: Q8_0 buffer too small".into(),
                ));
            }
            let mut dq = [0f32; 32];
            dequantize_q8_0_block(&b[..Q8_0_BLOCK_SIZE], &mut dq)?;
            Ok(dq[0])
        }
        TensorType::Q4K | TensorType::Q6K => Err(EngineError::Model(
            "layer_output_scale: unsupported dtype for scalar".into(),
        )),
    }
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
