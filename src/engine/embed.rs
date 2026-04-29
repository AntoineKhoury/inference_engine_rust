//! Token embedding lookup and prefill/decode input preparation.
//!
//! Converts token IDs into [`ForwardState`] activation buffers, handling embedding scaling and
//! the Gemma 4 per-layer embedding (PLE) pathway. These are the only functions that should call
//! `lookup_embeddings*` and construct the initial hidden state.

use crate::EngineError;
use crate::engine::state::ForwardState;
use crate::layers::embeddings::lookup_embeddings;
use crate::layers::gemma4_ple::compute_packed_per_layer_inputs;
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_loader::gguf_types::GGUFData;

/// Build a [`ForwardState`] for a prompt. Loads embedding tensor on demand (lazy GGUF path).
pub fn prefill_from_tokens(
    gguf: &mut GGUFData,
    file_path: &str,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Result<ForwardState, EngineError> {
    if token_ids.is_empty() {
        return Err(EngineError::Model("prefill: empty token list".into()));
    }
    if token_ids.len() > config.context_length {
        return Err(EngineError::Model(format!(
            "prefill: token length {} exceeds context length {}",
            token_ids.len(),
            config.context_length
        )));
    }

    let mut embeddings = lookup_embeddings(gguf, file_path, token_ids)?;
    scale_embeddings(&mut embeddings, config.token_embedding_scale);

    let per_layer =
        if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
            let ple = weights_gemma4_ple_tensors(gguf, file_path, config)?;
            Some(compute_packed_per_layer_inputs(&ple, config, token_ids, &embeddings)?)
        } else {
            None
        };

    let (packed, pl, pd) = unpack_ple(per_layer, config);
    ForwardState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
}

/// Build a length-1 [`ForwardState`] for a decode step. Requires weights already resident.
pub fn prefill_state_for_single_token_loaded(
    gguf: &GGUFData,
    config: &ModelConfig,
    token_id: u32,
) -> Result<ForwardState, EngineError> {
    let mut embeddings =
        crate::layers::embeddings::lookup_embeddings_loaded(gguf, &[token_id])?;
    scale_embeddings(&mut embeddings, config.token_embedding_scale);

    let per_layer =
        if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
            let ple = crate::model_weights::Gemma4PleTensors {
                per_layer_token_embd: gguf
                    .get_tensor("per_layer_token_embd.weight")
                    .ok_or_else(|| {
                        EngineError::Model(
                            "per_layer_token_embd.weight must be loaded for Gemma 4 PLE decode"
                                .into(),
                        )
                    })?,
                per_layer_model_proj: gguf
                    .get_tensor("per_layer_model_proj.weight")
                    .ok_or_else(|| {
                        EngineError::Model(
                            "per_layer_model_proj.weight must be loaded for Gemma 4 PLE decode"
                                .into(),
                        )
                    })?,
                per_layer_proj_norm: gguf
                    .get_tensor("per_layer_proj_norm.weight")
                    .ok_or_else(|| {
                        EngineError::Model(
                            "per_layer_proj_norm.weight must be loaded for Gemma 4 PLE decode"
                                .into(),
                        )
                    })?,
            };
            Some(compute_packed_per_layer_inputs(&ple, config, &[token_id], &embeddings)?)
        } else {
            None
        };

    let (packed, pl, pd) = unpack_ple(per_layer, config);
    ForwardState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
}

/// Same as [`prefill_from_tokens`] but only reads already-loaded tensors (`&GGUFData`).
///
/// Use when [`crate::model_weights::ModelWeights`] holds an immutable borrow of `gguf`
/// (e.g. in the chat REPL or bench harness).
pub fn prefill_from_tokens_loaded(
    gguf: &GGUFData,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Result<ForwardState, EngineError> {
    if token_ids.is_empty() {
        return Err(EngineError::Model("prefill: empty token list".into()));
    }
    if token_ids.len() > config.context_length {
        return Err(EngineError::Model(format!(
            "prefill: token length {} exceeds context length {}",
            token_ids.len(),
            config.context_length
        )));
    }

    let mut embeddings =
        crate::layers::embeddings::lookup_embeddings_loaded(gguf, token_ids)?;
    scale_embeddings(&mut embeddings, config.token_embedding_scale);

    let per_layer =
        if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
            let ple = gemma4_ple_tensors_loaded(gguf, config)?;
            Some(compute_packed_per_layer_inputs(&ple, config, token_ids, &embeddings)?)
        } else {
            None
        };

    let (packed, pl, pd) = unpack_ple(per_layer, config);
    ForwardState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
}

fn scale_embeddings(embeddings: &mut [Vec<f32>], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for row in embeddings.iter_mut() {
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

fn unpack_ple(per_layer: Option<Vec<f32>>, config: &ModelConfig) -> (Vec<f32>, usize, usize) {
    match per_layer {
        Some(v) => (v, config.n_layers, config.embedding_length_per_layer),
        None => (Vec::new(), 0, 0),
    }
}

fn gemma4_ple_tensors_loaded<'a>(
    gguf: &'a GGUFData,
    config: &ModelConfig,
) -> Result<crate::model_weights::Gemma4PleTensors<'a>, EngineError> {
    use crate::model_weights::Gemma4PleTensors;
    if config.embedding_length_per_layer == 0 {
        return Err(EngineError::Model(
            "gemma4_ple_tensors_loaded: PLE disabled on config".into(),
        ));
    }
    Ok(Gemma4PleTensors {
        per_layer_token_embd: gguf
            .get_tensor("per_layer_token_embd.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_token_embd.weight not loaded".into())
            })?,
        per_layer_model_proj: gguf
            .get_tensor("per_layer_model_proj.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_model_proj.weight not loaded".into())
            })?,
        per_layer_proj_norm: gguf
            .get_tensor("per_layer_proj_norm.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_proj_norm.weight not loaded".into())
            })?,
    })
}

fn weights_gemma4_ple_tensors<'a>(
    gguf: &'a mut GGUFData,
    file_path: &str,
    config: &ModelConfig,
) -> Result<crate::model_weights::Gemma4PleTensors<'a>, EngineError> {
    use crate::model_weights::Gemma4PleTensors;
    if config.embedding_length_per_layer == 0 {
        return Err(EngineError::Model(
            "weights_gemma4_ple_tensors: PLE disabled on config".into(),
        ));
    }
    for name in [
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        "per_layer_proj_norm.weight",
    ] {
        if gguf.get_tensor(name).is_none() {
            gguf.load_single_tensor(file_path, name)?;
        }
    }
    Ok(Gemma4PleTensors {
        per_layer_token_embd: gguf
            .get_tensor("per_layer_token_embd.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_token_embd.weight missing after load".into())
            })?,
        per_layer_model_proj: gguf
            .get_tensor("per_layer_model_proj.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_model_proj.weight missing after load".into())
            })?,
        per_layer_proj_norm: gguf
            .get_tensor("per_layer_proj_norm.weight")
            .ok_or_else(|| {
                EngineError::Model("per_layer_proj_norm.weight missing after load".into())
            })?,
    })
}
