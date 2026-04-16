use std::collections::HashSet;

use crate::EngineError;
use crate::core::tensor::Tensor;
use crate::model_config::ModelConfig;
use crate::model_loader::gguf_types::GGUFData;

#[derive(Debug)]
pub struct LayerWeights<'a> {
    pub attn_norm: &'a Tensor,
    pub ffn_norm: &'a Tensor,
    pub wq: &'a Tensor,
    pub wk: &'a Tensor,
    pub wv: &'a Tensor,
    pub wo: &'a Tensor,
    pub w_gate: &'a Tensor,
    pub w_up: &'a Tensor,
    pub w_down: &'a Tensor,
}

#[derive(Debug)]
pub struct ModelWeights<'a> {
    pub token_embeddings: &'a Tensor,
    pub output_norm: &'a Tensor,
    pub lm_head: &'a Tensor,
    pub layers: Vec<LayerWeights<'a>>,
}

impl<'a> ModelWeights<'a> {
    pub fn from_loaded(
        gguf: &'a GGUFData,
        names: &ModelWeightNames,
    ) -> Result<Self, EngineError> {
        let mut layers = Vec::with_capacity(names.layers.len());
        for layer in &names.layers {
            layers.push(LayerWeights {
                attn_norm: get_loaded(gguf, &layer.attn_norm)?,
                ffn_norm: get_loaded(gguf, &layer.ffn_norm)?,
                wq: get_loaded(gguf, &layer.wq)?,
                wk: get_loaded(gguf, &layer.wk)?,
                wv: get_loaded(gguf, &layer.wv)?,
                wo: get_loaded(gguf, &layer.wo)?,
                w_gate: get_loaded(gguf, &layer.w_gate)?,
                w_up: get_loaded(gguf, &layer.w_up)?,
                w_down: get_loaded(gguf, &layer.w_down)?,
            });
        }

        Ok(Self {
            token_embeddings: get_loaded(gguf, &names.token_embeddings)?,
            output_norm: get_loaded(gguf, &names.output_norm)?,
            lm_head: get_loaded(gguf, &names.lm_head)?,
            layers,
        })
    }
}

#[derive(Debug)]
pub struct LayerNames {
    attn_norm: String,
    ffn_norm: String,
    wq: String,
    wk: String,
    wv: String,
    wo: String,
    w_gate: String,
    w_up: String,
    w_down: String,
}

#[derive(Debug)]
pub struct ModelWeightNames {
    token_embeddings: String,
    output_norm: String,
    lm_head: String,
    layers: Vec<LayerNames>,
}

impl ModelWeightNames {
    pub fn resolve(
        gguf: &GGUFData,
        config: &ModelConfig,
    ) -> Result<Self, EngineError> {
        let available = available_tensor_names(gguf);

        let token_embeddings = resolve_name_from_strs(
            &available,
            &[
                "token_embd.weight",
                "tok_embeddings.weight",
                "embeddings.weight",
            ],
        )?;
        let output_norm =
            resolve_name_from_strs(&available, &["output_norm.weight", "norm.weight"])?;
        let lm_head = resolve_name_from_strs(&available, &["output.weight", "lm_head.weight"])?;

        let mut layer_names = Vec::with_capacity(config.n_layers);
        for layer_idx in 0..config.n_layers {
            let prefix = format!("blk.{layer_idx}.");
            layer_names.push(LayerNames {
                attn_norm: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}attn_norm.weight"),
                        format!("{prefix}attention_norm.weight"),
                    ],
                )?,
                ffn_norm: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}ffn_norm.weight"),
                        format!("{prefix}feed_forward_norm.weight"),
                    ],
                )?,
                wq: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}attn_q.weight"), format!("{prefix}wq.weight")],
                )?,
                wk: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}attn_k.weight"), format!("{prefix}wk.weight")],
                )?,
                wv: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}attn_v.weight"), format!("{prefix}wv.weight")],
                )?,
                wo: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}attn_output.weight"),
                        format!("{prefix}wo.weight"),
                    ],
                )?,
                w_gate: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}ffn_gate.weight"), format!("{prefix}w1.weight")],
                )?,
                w_up: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}ffn_up.weight"), format!("{prefix}w3.weight")],
                )?,
                w_down: resolve_name_from_strings(
                    &available,
                    &[format!("{prefix}ffn_down.weight"), format!("{prefix}w2.weight")],
                )?,
            });
        }

        Ok(Self {
            token_embeddings,
            output_norm,
            lm_head,
            layers: layer_names,
        })
    }

    pub fn load_all(
        &self,
        gguf: &mut GGUFData,
        file_path: &str,
    ) -> Result<(), EngineError> {
        let mut names_to_load = Vec::new();
        names_to_load.push(self.token_embeddings.clone());
        names_to_load.push(self.output_norm.clone());
        names_to_load.push(self.lm_head.clone());
        for layer in &self.layers {
            names_to_load.push(layer.attn_norm.clone());
            names_to_load.push(layer.ffn_norm.clone());
            names_to_load.push(layer.wq.clone());
            names_to_load.push(layer.wk.clone());
            names_to_load.push(layer.wv.clone());
            names_to_load.push(layer.wo.clone());
            names_to_load.push(layer.w_gate.clone());
            names_to_load.push(layer.w_up.clone());
            names_to_load.push(layer.w_down.clone());
        }

        gguf.load_named_tensors(file_path, &names_to_load)?;

        Ok(())
    }
}

fn available_tensor_names(gguf: &GGUFData) -> HashSet<String> {
    gguf.tensors_metadata()
        .iter()
        .map(|t| t.name.clone())
        .collect()
}

fn resolve_name_from_strs(
    available: &HashSet<String>,
    candidates: &[&str],
) -> Result<String, EngineError> {
    for candidate in candidates {
        if available.contains(*candidate) {
            return Ok(candidate.to_string());
        }
    }
    Err(EngineError::Model(format!(
        "none of the candidate tensor names were found: {candidates:?}"
    )))
}

fn resolve_name_from_strings(
    available: &HashSet<String>,
    candidates: &[String],
) -> Result<String, EngineError> {
    for candidate in candidates {
        if available.contains(candidate) {
            return Ok(candidate.clone());
        }
    }
    Err(EngineError::Model(format!(
        "none of the candidate tensor names were found: {candidates:?}"
    )))
}

fn get_loaded<'a>(
    gguf: &'a GGUFData,
    name: &str,
) -> Result<&'a Tensor, EngineError> {
    gguf.get_tensor(name).ok_or_else(|| {
        EngineError::Model(format!("tensor '{name}' not found after loading"))
    })
}
