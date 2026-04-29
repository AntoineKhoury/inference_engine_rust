//! Transformer block: attention + FFN + optional Gemma 4 PLE tail, for both prefill and decode.

use crate::EngineError;
use crate::engine::state::ForwardState;
use crate::layers::attention::{KVCache, decode_attention_with_norm, prefill_attention_with_norm};
use crate::layers::ffn::{
    apply_gemma_layer_output_scale, apply_per_layer_tail, prefill_ffn_with_norm,
};
use crate::model_config::ModelConfig;
use crate::model_weights::LayerWeights;

pub fn prefill_layer_block(
    input: &ForwardState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<ForwardState, EngineError> {
    let attn_out = prefill_attention_with_norm(input, config, layer_idx, weights, kv_caches)?;
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let ffn_dim = config.layer_dims_for(layer_idx)?.ffn_dim;
    let mut ffn_out =
        prefill_ffn_with_norm(&attn_out, seq_len, hidden_dim, ffn_dim, config, weights)?;

    if config.embedding_length_per_layer > 0 {
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

    ForwardState::from_flat_with_ple(
        ffn_out,
        seq_len,
        hidden_dim,
        input.per_layer_packed().to_vec(),
        input.ple_n_layers(),
        input.ple_dim(),
    )
}

pub fn decode_layer_block(
    input: &ForwardState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &LayerWeights,
    kv_caches: &mut [KVCache],
) -> Result<ForwardState, EngineError> {
    if input.seq_len() != 1 {
        return Err(EngineError::Model(
            "decode_layer_block: seq_len must be 1".into(),
        ));
    }
    let hidden_dim = input.hidden_dim();
    let attn_out = decode_attention_with_norm(input, config, layer_idx, weights, kv_caches)?;
    let ffn_dim = config.layer_dims_for(layer_idx)?.ffn_dim;
    let mut ffn_out = prefill_ffn_with_norm(&attn_out, 1, hidden_dim, ffn_dim, config, weights)?;

    if config.embedding_length_per_layer > 0 {
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

    ForwardState::from_flat_with_ple(
        ffn_out,
        1,
        hidden_dim,
        input.per_layer_packed().to_vec(),
        input.ple_n_layers(),
        input.ple_dim(),
    )
}
