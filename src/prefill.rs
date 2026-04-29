use crate::EngineError;
use crate::layers::embeddings::lookup_embeddings;
use crate::layers::gemma4_ple::compute_packed_per_layer_inputs;
use crate::layers::prefill_block::{decode_layer_block, prefill_layer_block};
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_loader::gguf_types::GGUFData;
use crate::model_weights::ModelWeights;
use crate::layers::attention::KVCache;
use crate::ops::rmsnorm::rmsnorm;
use crate::ops::matmul::matmul;
use crate::core::tensor::{Tensor, TensorType};
use std::sync::Arc;
/// Prefill buffers use a contiguous [seq, hidden] layout.
#[derive(Debug)]
pub struct PrefillState {
    seq_len: usize,
    hidden_dim: usize,
    hidden: Vec<f32>,
    per_layer_packed: Vec<f32>,
    ple_n_layers: usize,
    ple_dim: usize,
}

impl PrefillState {
    pub fn from_embeddings(
        embeddings: Vec<Vec<f32>>,
        hidden_dim: usize,
    ) -> Result<Self, EngineError> {
        Self::from_embeddings_inner(embeddings, hidden_dim, Vec::new(), 0, 0)
    }

    fn from_embeddings_inner(
        embeddings: Vec<Vec<f32>>,
        hidden_dim: usize,
        per_layer_packed: Vec<f32>,
        ple_n_layers: usize,
        ple_dim: usize,
    ) -> Result<Self, EngineError> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return Err(EngineError::Model("PrefillState: empty embeddings".into()));
        }

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for (idx, row) in embeddings.into_iter().enumerate() {
            if row.len() != hidden_dim {
                return Err(EngineError::Model(format!(
                    "PrefillState: embedding {idx} has dim {}, expected {hidden_dim}",
                    row.len(),
                )));
            }
            hidden.extend_from_slice(&row);
        }

        let stride = ple_n_layers.saturating_mul(ple_dim);
        if stride > 0 && per_layer_packed.len() != seq_len * stride {
            return Err(EngineError::Model(format!(
                "PrefillState: per_layer_packed len {} != seq_len * ple_stride {}",
                per_layer_packed.len(),
                seq_len * stride
            )));
        }

        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
            per_layer_packed,
            ple_n_layers,
            ple_dim,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    pub fn hidden_mut(&mut self) -> &mut [f32] {
        &mut self.hidden
    }

    pub fn per_layer_packed(&self) -> &[f32] {
        &self.per_layer_packed
    }

    pub fn ple_dim(&self) -> usize {
        self.ple_dim
    }

    pub fn ple_n_layers(&self) -> usize {
        self.ple_n_layers
    }

    /// Same activations shape, keep Gemma 4 PLE payload from `self`.
    pub fn replace_hidden(&self, hidden: Vec<f32>) -> Result<Self, EngineError> {
        Self::from_flat_with_ple(
            hidden,
            self.seq_len,
            self.hidden_dim,
            self.per_layer_packed.clone(),
            self.ple_n_layers,
            self.ple_dim,
        )
    }

    pub fn from_flat(
        hidden: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Self, EngineError> {
        Self::from_flat_with_ple(hidden, seq_len, hidden_dim, Vec::new(), 0, 0)
    }

    pub fn from_flat_with_ple(
        hidden: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
        per_layer_packed: Vec<f32>,
        ple_n_layers: usize,
        ple_dim: usize,
    ) -> Result<Self, EngineError> {
        if seq_len == 0 {
            return Err(EngineError::Model(
                "PrefillState: seq_len must be > 0".into(),
            ));
        }
        let expected_len = seq_len
            .checked_mul(hidden_dim)
            .ok_or_else(|| EngineError::Model("PrefillState: seq_len * hidden_dim overflow".into()))?;
        if hidden.len() != expected_len {
            return Err(EngineError::Model(format!(
                "PrefillState: hidden len {} != expected {}",
                hidden.len(),
                expected_len)));
        }
        let stride = ple_n_layers.saturating_mul(ple_dim);
        if stride > 0 && per_layer_packed.len() != seq_len * stride {
            return Err(EngineError::Model(format!(
                "PrefillState: per_layer_packed len {} != expected {}",
                per_layer_packed.len(),
                seq_len * stride
            )));
        }
        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
            per_layer_packed,
            ple_n_layers,
            ple_dim,
        })
    }

    pub fn row(&self, idx: usize) -> Result<&[f32], EngineError> {
        if idx >= self.seq_len {
            return Err(EngineError::Model(
                "PrefillState: row index out of bounds".into(),
            ));
        }
        let start = idx * self.hidden_dim;
        Ok(&self.hidden[start..start + self.hidden_dim])
    }

    pub fn row_mut(&mut self, idx: usize) -> Result<&mut [f32], EngineError> {
        if idx >= self.seq_len {
            return Err(EngineError::Model(
                "PrefillState: row index out of bounds".into(),
            ));
        }
        let start = idx * self.hidden_dim;
        Ok(&mut self.hidden[start..start + self.hidden_dim])
    }
}

pub fn prefill_from_tokens(
    gguf: &mut GGUFData,
    file_path: &str,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Result<PrefillState, EngineError> {
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
    let s = config.token_embedding_scale;
    if s != 1.0 {
        for row in &mut embeddings {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
    }

    let per_layer = if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
        let ple = weights_gemma4_ple_tensors(gguf, file_path, config)?;
        Some(compute_packed_per_layer_inputs(
            &ple,
            config,
            token_ids,
            &embeddings,
        )?)
    } else {
        None
    };

    let (packed, pl, pd) = match per_layer {
        Some(v) => (v, config.n_layers, config.embedding_length_per_layer),
        None => (Vec::new(), 0, 0),
    };
    PrefillState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
}

/// Build a length-1 [`PrefillState`] for decode after weights are resident (uses
/// [`crate::layers::embeddings::lookup_embeddings_loaded`] and PLE tensors already in `gguf`).
pub fn prefill_state_for_single_token_loaded(
    gguf: &GGUFData,
    config: &ModelConfig,
    token_id: u32,
) -> Result<PrefillState, EngineError> {
    let mut embeddings = crate::layers::embeddings::lookup_embeddings_loaded(gguf, &[token_id])?;
    let s = config.token_embedding_scale;
    if s != 1.0 {
        for row in &mut embeddings {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
    }

    let per_layer = if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
        let ple = crate::model_weights::Gemma4PleTensors {
            per_layer_token_embd: gguf
                .get_tensor("per_layer_token_embd.weight")
                .ok_or_else(|| {
                    EngineError::Model(
                        "per_layer_token_embd.weight must be loaded for Gemma 4 PLE decode".into(),
                    )
                })?,
            per_layer_model_proj: gguf
                .get_tensor("per_layer_model_proj.weight")
                .ok_or_else(|| {
                    EngineError::Model(
                        "per_layer_model_proj.weight must be loaded for Gemma 4 PLE decode".into(),
                    )
                })?,
            per_layer_proj_norm: gguf
                .get_tensor("per_layer_proj_norm.weight")
                .ok_or_else(|| {
                    EngineError::Model(
                        "per_layer_proj_norm.weight must be loaded for Gemma 4 PLE decode".into(),
                    )
                })?,
        };
        Some(compute_packed_per_layer_inputs(
            &ple,
            config,
            &[token_id],
            &embeddings,
        )?)
    } else {
        None
    };

    let (packed, pl, pd) = match per_layer {
        Some(v) => (v, config.n_layers, config.embedding_length_per_layer),
        None => (Vec::new(), 0, 0),
    };
    PrefillState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
}

/// Same as [`prefill_from_tokens`] but only reads **already-loaded** tensors (`&GGUFData`).
/// Use when [`crate::model_weights::ModelWeights`] holds an immutable borrow of `gguf` (e.g. chat REPL).
pub fn prefill_from_tokens_loaded(
    gguf: &GGUFData,
    config: &ModelConfig,
    token_ids: &[u32],
) -> Result<PrefillState, EngineError> {
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

    let mut embeddings = crate::layers::embeddings::lookup_embeddings_loaded(gguf, token_ids)?;
    let s = config.token_embedding_scale;
    if s != 1.0 {
        for row in &mut embeddings {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
    }

    let per_layer = if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
        let ple = gemma4_ple_tensors_loaded(gguf, config)?;
        Some(compute_packed_per_layer_inputs(
            &ple,
            config,
            token_ids,
            &embeddings,
        )?)
    } else {
        None
    };

    let (packed, pl, pd) = match per_layer {
        Some(v) => (v, config.n_layers, config.embedding_length_per_layer),
        None => (Vec::new(), 0, 0),
    };
    PrefillState::from_embeddings_inner(embeddings, config.hidden_dim, packed, pl, pd)
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
            .ok_or_else(|| EngineError::Model("per_layer_token_embd.weight not loaded".into()))?,
        per_layer_model_proj: gguf
            .get_tensor("per_layer_model_proj.weight")
            .ok_or_else(|| EngineError::Model("per_layer_model_proj.weight not loaded".into()))?,
        per_layer_proj_norm: gguf
            .get_tensor("per_layer_proj_norm.weight")
            .ok_or_else(|| EngineError::Model("per_layer_proj_norm.weight not loaded".into()))?,
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
        per_layer_token_embd: gguf.get_tensor("per_layer_token_embd.weight").ok_or_else(|| {
            EngineError::Model("per_layer_token_embd.weight missing after load".into())
        })?,
        per_layer_model_proj: gguf.get_tensor("per_layer_model_proj.weight").ok_or_else(|| {
            EngineError::Model("per_layer_model_proj.weight missing after load".into())
        })?,
        per_layer_proj_norm: gguf.get_tensor("per_layer_proj_norm.weight").ok_or_else(|| {
            EngineError::Model("per_layer_proj_norm.weight missing after load".into())
        })?,
    })
}

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
/// Prefer [`prefill_state_for_single_token_loaded`] for Gemma 4 so embeddings are scaled and PLE is populated.
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
