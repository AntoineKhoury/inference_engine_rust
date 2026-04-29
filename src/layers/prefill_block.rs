use std::sync::Arc;

use crate::core::tensor::{Tensor, TensorType};
use crate::layers::attention::{decode_attention_layer, prefill_attention_layer, KVCache};
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_weights::LayerWeights;
use crate::ops::gelu::gelu_tanh;
use crate::ops::matmul::matmul;
use crate::ops::quant::quant_k_handler::{dequantize_q8_0_block, Q8_0_BLOCK_SIZE};
use crate::ops::residual_add::residual_add;
use crate::ops::rmsnorm::rmsnorm;
use crate::ops::swiglu::swiglu;
use crate::prefill::PrefillState;
use crate::EngineError;

#[inline]
fn gemma4_disable_ple_tail() -> bool {
    std::env::var("INFERENCE_ENGINE_GEMMA4_DISABLE_PLE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct Gemma4LayerDebug {
    pub attn_proj: Vec<f32>,
    pub attn_post_norm: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_normed: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_geglu: Vec<f32>,
    pub ffn_down: Vec<f32>,
    /// `ffn_down / rms(ffn_down)` (no learned scale).
    pub ffn_post_rms: Vec<f32>,
    /// `ffn_post_rms * post_ffw_norm.weight` (learned scale only).
    pub ffn_post_mul: Vec<f32>,
    pub ffn_post_norm: Vec<f32>,
    pub pe_in: Vec<f32>,
    pub after_tail: Vec<f32>,
    pub l_out: Vec<f32>,
}

fn rms_only(input: &[f32], epsilon: f32, output: &mut [f32]) -> Result<(), EngineError> {
    if input.len() != output.len() {
        return Err(EngineError::Model("rms_only: len mismatch".into()));
    }
    let dim = input.len();
    if dim == 0 {
        return Ok(());
    }
    let mut sum: f64 = 0.0;
    for &x in input.iter() {
        sum += (x * x) as f64;
    }
    let mean: f32 = (sum / dim as f64) as f32;
    let scale: f32 = 1.0f32 / (mean + epsilon).sqrt();
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x * scale;
    }
    Ok(())
}

pub fn prefill_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    match config.family {
        ModelFamily::MistralLlama => {
            mistral_prefill_attention_with_norm(input, config, layer_idx, weights, kv_caches)
        }
        ModelFamily::Gemma4 => {
            gemma4_prefill_attention_with_norm(input, config, layer_idx, weights, kv_caches)
        }
    }
}

fn mistral_prefill_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim {
        return Err(EngineError::Model(format!(
            "attn_norm weights len {} != hidden_dim {}",
            attn_norm_weights.len(),
            hidden_dim
        )));
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
    let layer_attn = config.layer_attention_for(layer_idx)?;
    let layer_dims = config.layer_dims_for(layer_idx)?;
    let attn_out = prefill_attention_layer(
        &normed_state,
        config,
        layer_dims,
        layer_attn,
        weights,
        kv_caches,
        layer_idx,
    )?;

    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

fn gemma4_prefill_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    let post_attn_w = weights
        .attn_post_norm
        .ok_or_else(|| EngineError::Model("Gemma 4: missing post_attention_norm".into()))?
        .as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim || post_attn_w.len() != hidden_dim {
        return Err(EngineError::Model(
            "Gemma 4: attn norm weight length mismatch".into(),
        ));
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
    let layer_attn = config.layer_attention_for(layer_idx)?;
    let layer_dims = config.layer_dims_for(layer_idx)?;
    let mut attn_out = prefill_attention_layer(
        &normed_state,
        config,
        layer_dims,
        layer_attn,
        weights,
        kv_caches,
        layer_idx,
    )?;

    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        let mut tmp = vec![0.0f32; hidden_dim];
        rmsnorm(
            &attn_out[start..end],
            post_attn_w,
            config.rms_norm_eps,
            &mut tmp,
        )?;
        attn_out[start..end].copy_from_slice(&tmp);
    }

    let mut residual_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

pub fn prefill_ffn(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    let debug = prefill_ffn_debug(input, seq_len, hidden_dim, ffn_dim, config, weights)?;
    prefill_ffn_down_from_activated(
        &debug.activated,
        seq_len,
        hidden_dim,
        ffn_dim,
        config,
        weights,
    )
}

pub struct PrefillFfnDebug {
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub activated: Vec<f32>,
}

pub fn prefill_ffn_debug(
    input: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<PrefillFfnDebug, EngineError> {
    if input.len() != seq_len * hidden_dim {
        return Err(EngineError::Model(
            "prefill_ffn_debug: input length does not match shape".into(),
        ));
    }
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_ffn_debug: hidden_dim mismatch with config".into(),
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
        // `hidden_activation="gelu_pytorch_tanh"` (see `gelu::gelu_tanh`).
        ModelFamily::Gemma4 => {
            for i in 0..gate.len() {
                activated[i] = gelu_tanh(gate[i]) * up[i];
            }
        }
        ModelFamily::MistralLlama => {
            swiglu(gate, up, &mut activated)?;
        }
    }

    Ok(PrefillFfnDebug {
        gate: gate.to_vec(),
        up: up.to_vec(),
        activated,
    })
}

pub fn prefill_ffn_down_from_activated(
    activated: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    config: &ModelConfig,
    weights: &LayerWeights,
) -> Result<Vec<f32>, EngineError> {
    if activated.len() != seq_len * ffn_dim {
        return Err(EngineError::Model(
            "prefill_ffn_down_from_activated: input length does not match shape".into(),
        ));
    }
    if hidden_dim != config.hidden_dim {
        return Err(EngineError::Model(
            "prefill_ffn_down_from_activated: hidden_dim mismatch with config".into(),
        ));
    }

    let activated_tensor = tensor_from_f32_slice(activated, vec![seq_len, ffn_dim]);
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

fn apply_per_layer_tail(
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

/// Gemma 4 per-layer debug breakdown for semantic parity against llama.cpp node dumps.
pub fn gemma4_prefill_layer_debug(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Gemma4LayerDebug, EngineError> {
    if !matches!(config.family, ModelFamily::Gemma4) {
        return Err(EngineError::Model(
            "gemma4_prefill_layer_debug is only valid for Gemma4".into(),
        ));
    }
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    let post_attn_w = weights
        .attn_post_norm
        .ok_or_else(|| EngineError::Model("Gemma 4: missing post_attention_norm".into()))?
        .as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim || post_attn_w.len() != hidden_dim {
        return Err(EngineError::Model(
            "Gemma 4: attn norm weight length mismatch".into(),
        ));
    }

    let mut attn_normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rmsnorm(
            &input.hidden()[start..end],
            attn_norm_weights,
            config.rms_norm_eps,
            &mut attn_normed[start..end],
        )?;
    }
    let normed_state = PrefillState::from_flat(attn_normed, seq_len, hidden_dim)?;
    let layer_attn = config.layer_attention_for(layer_idx)?;
    let layer_dims = config.layer_dims_for(layer_idx)?;
    let attn_proj = prefill_attention_layer(
        &normed_state,
        config,
        layer_dims,
        layer_attn,
        weights,
        kv_caches,
        layer_idx,
    )?;

    let mut attn_post_norm = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rmsnorm(
            &attn_proj[start..end],
            post_attn_w,
            config.rms_norm_eps,
            &mut attn_post_norm[start..end],
        )?;
    }
    let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
    residual_add(input.hidden(), &attn_post_norm, &mut attn_out)?;

    let ffn_dim = layer_dims.ffn_dim;
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

    let mut ffn_normed = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rmsnorm(
            &attn_out[start..end],
            ffn_norm_weights,
            config.rms_norm_eps,
            &mut ffn_normed[start..end],
        )?;
    }

    let input_tensor = tensor_from_f32_slice(&ffn_normed, vec![seq_len, hidden_dim]);
    let mut gate_tensor = empty_f32_tensor(vec![seq_len, ffn_dim]);
    let mut up_tensor = empty_f32_tensor(vec![seq_len, ffn_dim]);
    matmul(&input_tensor, weights.w_gate, &mut gate_tensor)?;
    matmul(&input_tensor, weights.w_up, &mut up_tensor)?;
    let ffn_gate = gate_tensor.as_f32_slice()?.to_vec();
    let ffn_up = up_tensor.as_f32_slice()?.to_vec();
    let mut ffn_geglu = vec![0.0f32; ffn_gate.len()];
    for i in 0..ffn_gate.len() {
        ffn_geglu[i] = gelu_tanh(ffn_gate[i]) * ffn_up[i];
    }

    let activated_tensor = tensor_from_f32_slice(&ffn_geglu, vec![seq_len, ffn_dim]);
    let mut down_tensor = empty_f32_tensor(vec![seq_len, hidden_dim]);
    matmul(&activated_tensor, weights.w_down, &mut down_tensor)?;
    let ffn_down = down_tensor.as_f32_slice()?.to_vec();

    let mut ffn_post_rms = vec![0.0f32; seq_len * hidden_dim];
    let mut ffn_post_mul = vec![0.0f32; seq_len * hidden_dim];
    let mut ffn_post_norm = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let start = pos * hidden_dim;
        let end = start + hidden_dim;
        rms_only(
            &ffn_down[start..end],
            config.rms_norm_eps,
            &mut ffn_post_rms[start..end],
        )?;
        let plus_one = std::env::var("INFERENCE_ENGINE_GEMMA4_POSTFFN_WEIGHT_PLUS_ONE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        for i in 0..hidden_dim {
            let x = ffn_post_rms[start + i];
            let w = post_ffn_w[i];
            let s = if plus_one { 1.0 + w } else { w };
            ffn_post_mul[start + i] = x * s;
        }
        ffn_post_norm[start..end].copy_from_slice(&ffn_post_mul[start..end]);
    }

    let mut pe_in = vec![0.0f32; seq_len * hidden_dim];
    residual_add(&attn_out, &ffn_post_norm, &mut pe_in)?;

    let mut after_tail = pe_in.clone();
    if config.embedding_length_per_layer > 0 && !gemma4_disable_ple_tail() {
        apply_per_layer_tail(
            &mut after_tail,
            seq_len,
            hidden_dim,
            layer_idx,
            config,
            weights,
            input.per_layer_packed(),
        )?;
    }

    let mut l_out = after_tail.clone();
    apply_gemma_layer_output_scale(&mut l_out, weights.layer_output_scale)?;

    Ok(Gemma4LayerDebug {
        attn_proj,
        attn_post_norm,
        attn_out,
        ffn_normed,
        ffn_gate,
        ffn_up,
        ffn_geglu,
        ffn_down,
        ffn_post_rms,
        ffn_post_mul,
        ffn_post_norm,
        pe_in,
        after_tail,
        l_out,
    })
}

pub fn prefill_layer_block(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, EngineError> {
    let attn_out = prefill_attention_with_norm(input, config, layer_idx, weights, kv_caches)?;
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let ffn_dim = config.layer_dims_for(layer_idx)?.ffn_dim;
    let mut ffn_out =
        prefill_ffn_with_norm(&attn_out, seq_len, hidden_dim, ffn_dim, config, weights)?;

    if config.embedding_length_per_layer > 0 && !gemma4_disable_ple_tail() {
        apply_per_layer_tail(
            &mut ffn_out,
            seq_len,
            hidden_dim,
            layer_idx,
            config,
            weights,
            input.per_layer_packed(),
        )?;
    }

    apply_gemma_layer_output_scale(&mut ffn_out, weights.layer_output_scale)?;

    PrefillState::from_flat_with_ple(
        ffn_out,
        seq_len,
        hidden_dim,
        input.per_layer_packed().to_vec(),
        input.ple_n_layers(),
        input.ple_dim(),
    )
}

pub fn decode_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    if input.seq_len() != 1 {
        return Err(EngineError::Model(
            "decode_attention_with_norm: seq_len must be 1".into(),
        ));
    }
    match config.family {
        ModelFamily::MistralLlama => {
            mistral_decode_attention_with_norm(input, config, layer_idx, weights, kv_caches)
        }
        ModelFamily::Gemma4 => {
            gemma4_decode_attention_with_norm(input, config, layer_idx, weights, kv_caches)
        }
    }
}

fn mistral_decode_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim {
        return Err(EngineError::Model(format!(
            "attn_norm weights len {} != hidden_dim {}",
            attn_norm_weights.len(),
            hidden_dim
        )));
    }

    let mut normed = vec![0.0f32; hidden_dim];
    rmsnorm(
        input.hidden(),
        attn_norm_weights,
        config.rms_norm_eps,
        &mut normed,
    )?;

    let normed_state = PrefillState::from_flat(normed, 1, hidden_dim)?;
    let layer_attn = config.layer_attention_for(layer_idx)?;
    let layer_dims = config.layer_dims_for(layer_idx)?;
    let attn_out = decode_attention_layer(
        &normed_state,
        config,
        layer_dims,
        layer_attn,
        weights,
        kv_caches,
        layer_idx,
    )?;

    let mut residual_out = vec![0.0f32; hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

fn gemma4_decode_attention_with_norm(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<Vec<f32>, EngineError> {
    let hidden_dim = input.hidden_dim();

    let attn_norm_weights = weights.attn_norm.as_f32_slice()?;
    let post_attn_w = weights
        .attn_post_norm
        .ok_or_else(|| EngineError::Model("Gemma 4: missing post_attention_norm".into()))?
        .as_f32_slice()?;
    if attn_norm_weights.len() != hidden_dim || post_attn_w.len() != hidden_dim {
        return Err(EngineError::Model(
            "Gemma 4: attn norm weight length mismatch".into(),
        ));
    }

    let mut normed = vec![0.0f32; hidden_dim];
    rmsnorm(
        input.hidden(),
        attn_norm_weights,
        config.rms_norm_eps,
        &mut normed,
    )?;

    let normed_state = PrefillState::from_flat(normed, 1, hidden_dim)?;
    let layer_attn = config.layer_attention_for(layer_idx)?;
    let layer_dims = config.layer_dims_for(layer_idx)?;
    let mut attn_out = decode_attention_layer(
        &normed_state,
        config,
        layer_dims,
        layer_attn,
        weights,
        kv_caches,
        layer_idx,
    )?;

    let mut tmp = vec![0.0f32; hidden_dim];
    rmsnorm(&attn_out, post_attn_w, config.rms_norm_eps, &mut tmp)?;
    attn_out.copy_from_slice(&tmp);

    let mut residual_out = vec![0.0f32; hidden_dim];
    residual_add(input.hidden(), &attn_out, &mut residual_out)?;
    Ok(residual_out)
}

pub fn decode_layer_block(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, EngineError> {
    if input.seq_len() != 1 {
        return Err(EngineError::Model(
            "decode_layer_block: seq_len must be 1".into(),
        ));
    }
    let hidden_dim = input.hidden_dim();
    let attn_out = decode_attention_with_norm(input, config, layer_idx, weights, kv_caches)?;
    let ffn_dim = config.layer_dims_for(layer_idx)?.ffn_dim;
    let mut ffn_out = prefill_ffn_with_norm(&attn_out, 1, hidden_dim, ffn_dim, config, weights)?;

    if config.embedding_length_per_layer > 0 && !gemma4_disable_ple_tail() {
        apply_per_layer_tail(
            &mut ffn_out,
            1,
            hidden_dim,
            layer_idx,
            config,
            weights,
            input.per_layer_packed(),
        )?;
    }

    apply_gemma_layer_output_scale(&mut ffn_out, weights.layer_output_scale)?;

    PrefillState::from_flat_with_ple(
        ffn_out,
        1,
        hidden_dim,
        input.per_layer_packed().to_vec(),
        input.ple_n_layers(),
        input.ple_dim(),
    )
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

fn apply_gemma_layer_output_scale(
    hidden: &mut [f32],
    scale: Option<&Tensor>,
) -> Result<(), EngineError> {
    if std::env::var("INFERENCE_ENGINE_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return Ok(());
    }
    let Some(t) = scale else {
        return Ok(());
    };
    let s = layer_output_scale_as_f32(t)?;
    for v in hidden.iter_mut() {
        *v *= s;
    }
    Ok(())
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
