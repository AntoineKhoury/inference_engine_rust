use crate::EngineError;
use crate::core::tensor::Tensor;
use crate::model_loader::gguf_types::GGUFData;

use super::names::{Gemma4PleNames, LayerNames, ModelWeightNames};

/// Borrowed references to the three global Gemma 4 PLE tensors.
#[derive(Debug, Clone, Copy)]
pub struct Gemma4PleTensors<'a> {
    pub per_layer_token_embd: &'a Tensor,
    pub per_layer_model_proj: &'a Tensor,
    pub per_layer_proj_norm: &'a Tensor,
}

/// Borrowed tensor views for a single transformer block, valid for the lifetime of [`GGUFData`].
#[derive(Debug)]
pub struct LayerWeights<'a> {
    pub attn_norm: &'a Tensor,
    pub ffn_norm: &'a Tensor,
    /// Gemma 4: RMSNorm on attention output before residual (`post_attention_norm`).
    pub attn_post_norm: Option<&'a Tensor>,
    /// Gemma 4: RMSNorm on FFN output before residual (`post_ffw_norm`).
    pub ffn_post_norm: Option<&'a Tensor>,
    /// Per-head RMSNorm on Q before RoPE (Gemma 4); `None` for Llama/Mistral-style blocks.
    pub attn_q_norm: Option<&'a Tensor>,
    pub attn_k_norm: Option<&'a Tensor>,
    pub wq: &'a Tensor,
    pub wk: &'a Tensor,
    pub wv: &'a Tensor,
    pub wo: &'a Tensor,
    pub w_gate: &'a Tensor,
    pub w_up: &'a Tensor,
    pub w_down: &'a Tensor,
    /// PLE: `hidden -> ple_dim` (`blk.*.inp_gate`).
    pub ple_inp_gate: Option<&'a Tensor>,
    /// PLE: `ple_dim -> hidden` (`blk.*.proj`).
    pub ple_proj: Option<&'a Tensor>,
    /// PLE: RMSNorm on projected vector before residual (`blk.*.post_norm`).
    pub ple_post_norm: Option<&'a Tensor>,
    /// Gemma 4 full-attention: `blk.*.rope_freqs.weight` for proportional RoPE (optional).
    pub rope_freqs: Option<&'a Tensor>,
    /// Gemma 4: `blk.*.layer_output_scale.weight` (length 1); applied after PLE.
    pub layer_output_scale: Option<&'a Tensor>,
}

/// Borrowed view of all model tensors needed for a forward pass.
#[derive(Debug)]
pub struct ModelWeights<'a> {
    pub token_embeddings: &'a Tensor,
    pub output_norm: &'a Tensor,
    pub lm_head: &'a Tensor,
    pub layers: Vec<LayerWeights<'a>>,
    pub gemma4_ple: Option<Gemma4PleTensors<'a>>,
}

impl<'a> ModelWeights<'a> {
    pub fn from_loaded(gguf: &'a GGUFData, names: &ModelWeightNames) -> Result<Self, EngineError> {
        let mut layers = Vec::with_capacity(names.layers.len());
        for layer in &names.layers {
            layers.push(build_layer_weights(gguf, layer)?);
        }

        let gemma4_ple = if let Some(ref g) = names.gemma4_ple {
            Some(build_ple_tensors(gguf, g)?)
        } else {
            None
        };

        Ok(Self {
            token_embeddings: get_loaded(gguf, &names.token_embeddings)?,
            output_norm: get_loaded(gguf, &names.output_norm)?,
            lm_head: get_loaded(gguf, &names.lm_head)?,
            layers,
            gemma4_ple,
        })
    }
}

fn build_layer_weights<'a>(
    gguf: &'a GGUFData,
    layer: &LayerNames,
) -> Result<LayerWeights<'a>, EngineError> {
    Ok(LayerWeights {
        attn_norm: get_loaded(gguf, &layer.attn_norm)?,
        ffn_norm: get_loaded(gguf, &layer.ffn_norm)?,
        attn_post_norm: layer
            .attn_post_norm
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        ffn_post_norm: layer
            .ffn_post_norm
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        attn_q_norm: layer
            .attn_q_norm
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        attn_k_norm: layer
            .attn_k_norm
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        wq: get_loaded(gguf, &layer.wq)?,
        wk: get_loaded(gguf, &layer.wk)?,
        wv: get_loaded(gguf, &layer.wv)?,
        wo: get_loaded(gguf, &layer.wo)?,
        w_gate: get_loaded(gguf, &layer.w_gate)?,
        w_up: get_loaded(gguf, &layer.w_up)?,
        w_down: get_loaded(gguf, &layer.w_down)?,
        ple_inp_gate: layer
            .ple_inp_gate
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        ple_proj: layer
            .ple_proj
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        ple_post_norm: layer
            .ple_post_norm
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        rope_freqs: layer
            .rope_freqs
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
        layer_output_scale: layer
            .layer_output_scale
            .as_ref()
            .map(|n| get_loaded(gguf, n))
            .transpose()?,
    })
}

fn build_ple_tensors<'a>(
    gguf: &'a GGUFData,
    g: &Gemma4PleNames,
) -> Result<Gemma4PleTensors<'a>, EngineError> {
    Ok(Gemma4PleTensors {
        per_layer_token_embd: get_loaded(gguf, &g.per_layer_token_embd)?,
        per_layer_model_proj: get_loaded(gguf, &g.per_layer_model_proj)?,
        per_layer_proj_norm: get_loaded(gguf, &g.per_layer_proj_norm)?,
    })
}

fn get_loaded<'a>(gguf: &'a GGUFData, name: &str) -> Result<&'a Tensor, EngineError> {
    gguf.get_tensor(name)
        .ok_or_else(|| EngineError::Model(format!("tensor '{name}' not found after loading")))
}
