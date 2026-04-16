//! Gemma 4 per-layer embeddings (PLE): HF `get_per_layer_inputs` + `project_per_layer_inputs`.

use std::sync::Arc;

use crate::EngineError;
use crate::core::tensor::{Tensor, TensorType};
use crate::layers::embeddings::read_token_row_f32;
use crate::model_config::ModelConfig;
use crate::model_weights::Gemma4PleTensors;
use crate::ops::matmul::matmul;
use crate::ops::rmsnorm::rmsnorm;

fn tensor_from_f32_slice(data: &[f32], dimensions: Vec<usize>) -> Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Tensor::new(TensorType::F32, Arc::new(bytes), dimensions)
}

fn empty_f32_tensor(dimensions: Vec<usize>) -> Tensor {
    let len = dimensions.iter().product::<usize>();
    Tensor::new(TensorType::F32, Arc::new(vec![0u8; len * 4]), dimensions)
}

/// Packed layout `[seq, n_layers * ple_dim]` row-major (positions contiguous).
pub fn compute_packed_per_layer_inputs(
    ple: &Gemma4PleTensors<'_>,
    config: &ModelConfig,
    token_ids: &[u32],
    main_scaled_embeddings: &[Vec<f32>],
) -> Result<Vec<f32>, EngineError> {
    let ple_dim = config.embedding_length_per_layer;
    let n_layers = config.n_layers;
    if ple_dim == 0 {
        return Ok(Vec::new());
    }
    let seq_len = token_ids.len();
    if main_scaled_embeddings.len() != seq_len {
        return Err(EngineError::Model(
            "compute_packed_per_layer_inputs: embeddings len != token_ids len".into(),
        ));
    }
    let pack = n_layers
        .checked_mul(ple_dim)
        .ok_or_else(|| EngineError::Model("PLE: n_layers * ple_dim overflow".into()))?;

    let mut stacked = Vec::with_capacity(seq_len * config.hidden_dim);
    for row in main_scaled_embeddings {
        if row.len() != config.hidden_dim {
            return Err(EngineError::Model(
                "compute_packed_per_layer_inputs: bad embedding width".into(),
            ));
        }
        stacked.extend_from_slice(row);
    }

    let input_tensor = tensor_from_f32_slice(&stacked, vec![seq_len, config.hidden_dim]);
    let mut proj_tensor = empty_f32_tensor(vec![seq_len, pack]);
    matmul(&input_tensor, ple.per_layer_model_proj, &mut proj_tensor)?;
    let mut proj = proj_tensor.as_f32_slice()?.to_vec();
    let scale_h = config.ple_model_proj_scale;
    for v in &mut proj {
        *v *= scale_h;
    }

    let sqrt_ple = (ple_dim as f32).sqrt();
    let combine = config.ple_combine_scale;

    let norm_w = ple.per_layer_proj_norm.as_f32_slice()?;
    if norm_w.len() != ple_dim {
        return Err(EngineError::Model(format!(
            "per_layer_proj_norm len {} != ple_dim {}",
            norm_w.len(),
            ple_dim
        )));
    }

    let eps = config.rms_norm_eps;
    let mut out = vec![0.0f32; seq_len * pack];
    let mut normed_chunk = vec![0.0f32; ple_dim];

    for p in 0..seq_len {
        let tid = token_ids[p];
        let mut token_row = read_token_row_f32(ple.per_layer_token_embd, tid)?;
        if token_row.len() != pack {
            return Err(EngineError::Model(format!(
                "per_layer_token_embd row width {} != n_layers*ple_dim {}",
                token_row.len(),
                pack
            )));
        }
        for v in &mut token_row {
            *v *= sqrt_ple;
        }

        let base = p * pack;
        for l in 0..n_layers {
            let off = base + l * ple_dim;
            rmsnorm(
                &proj[off..off + ple_dim],
                norm_w,
                eps,
                &mut normed_chunk,
            )?;
            let tok_off = l * ple_dim;
            for i in 0..ple_dim {
                out[off + i] = (normed_chunk[i] + token_row[tok_off + i]) * combine;
            }
        }
    }

    Ok(out)
}
