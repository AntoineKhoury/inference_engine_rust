use crate::layers::embeddings::lookup_embeddings;
use crate::layers::prefill_block::prefill_layer_block;
use crate::model_config::ModelConfig;
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
}

impl PrefillState {
    pub fn from_embeddings(
        embeddings: Vec<Vec<f32>>,
        hidden_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return Err("PrefillState: empty embeddings".into());
        }

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for (idx, row) in embeddings.into_iter().enumerate() {
            if row.len() != hidden_dim {
                return Err(format!(
                    "PrefillState: embedding {} has dim {}, expected {}",
                    idx,
                    row.len(),
                    hidden_dim
                )
                .into());
            }
            hidden.extend_from_slice(&row);
        }

        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
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

    pub fn from_flat(
        hidden: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if seq_len == 0 {
            return Err("PrefillState: seq_len must be > 0".into());
        }
        let expected_len = seq_len
            .checked_mul(hidden_dim)
            .ok_or("PrefillState: seq_len * hidden_dim overflow")?;
        if hidden.len() != expected_len {
            return Err(format!(
                "PrefillState: hidden len {} != expected {}",
                hidden.len(),
                expected_len
            )
            .into());
        }
        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
        })
    }

    pub fn row(&self, idx: usize) -> Result<&[f32], Box<dyn std::error::Error>> {
        if idx >= self.seq_len {
            return Err("PrefillState: row index out of bounds".into());
        }
        let start = idx * self.hidden_dim;
        Ok(&self.hidden[start..start + self.hidden_dim])
    }

    pub fn row_mut(&mut self, idx: usize) -> Result<&mut [f32], Box<dyn std::error::Error>> {
        if idx >= self.seq_len {
            return Err("PrefillState: row index out of bounds".into());
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
) -> Result<PrefillState, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("prefill: empty token list".into());
    }
    if token_ids.len() > config.context_length {
        return Err(format!(
            "prefill: token length {} exceeds context length {}",
            token_ids.len(),
            config.context_length
        )
        .into());
    }

    let embeddings = lookup_embeddings(gguf, file_path, token_ids)?;
    PrefillState::from_embeddings(embeddings, config.hidden_dim)
}

pub fn prefill_forward(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
    kv_caches: &mut [KVCache],
) -> Result<PrefillState, Box<dyn std::error::Error>> {
    if kv_caches.len() != weights.layers.len() {
        return Err("prefill_forward: kv_caches len != number of layers".into());
    }

    let mut state = PrefillState::from_flat(
        input.hidden().to_vec(),
        input.seq_len(),
        input.hidden_dim(),
    )?;

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let cache = &mut kv_caches[layer_idx];
        state = prefill_layer_block(&state, config, layer_weights, cache)?;
    }

    Ok(state)
}

pub fn final_logits_last_token(
    input: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    if seq_len == 0 {
        return Err("final_logits_last_token: empty input".into());
    }

    let last_start = (seq_len - 1) * hidden_dim;
    let last_end = last_start + hidden_dim;
    let last_hidden = &input.hidden()[last_start..last_end];

    let norm_weights = weights.output_norm.as_f32_slice()?;
    if norm_weights.len() != hidden_dim {
        return Err(format!(
            "final_logits_last_token: output_norm len {} != hidden_dim {}",
            norm_weights.len(),
            hidden_dim
        )
        .into());
    }

    let mut normed = vec![0.0f32; hidden_dim];
    rmsnorm(last_hidden, norm_weights, config.rms_norm_eps, &mut normed)?;

    let input_tensor = tensor_from_f32_slice(&normed, vec![1, hidden_dim]);
    let mut logits_tensor = empty_f32_tensor(vec![1, config.vocab_size]);
    matmul(&input_tensor, weights.lm_head, &mut logits_tensor)?;

    Ok(logits_tensor.as_f32_slice()?.to_vec())
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
