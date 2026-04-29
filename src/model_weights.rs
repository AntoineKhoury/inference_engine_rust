use std::collections::HashSet;

use crate::EngineError;
use crate::core::tensor::Tensor;
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_loader::gguf_types::GGUFData;

#[derive(Debug, Clone, Copy)]
pub struct Gemma4PleTensors<'a> {
    pub per_layer_token_embd: &'a Tensor,
    pub per_layer_model_proj: &'a Tensor,
    pub per_layer_proj_norm: &'a Tensor,
}

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
    /// Gemma 4 full-attention: `blk.*.rope_freqs.weight` for proportional RoPE (optional; sliding layers omit).
    pub rope_freqs: Option<&'a Tensor>,
    /// Gemma 4: HF `layer_scalar` / GGUF `blk.*.layer_output_scale.weight` (length1); applied after PLE.
    pub layer_output_scale: Option<&'a Tensor>,
}

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
            layers.push(LayerWeights {
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
            });
        }

        let gemma4_ple = if let Some(ref g) = names.gemma4_ple {
            Some(Gemma4PleTensors {
                per_layer_token_embd: get_loaded(gguf, &g.per_layer_token_embd)?,
                per_layer_model_proj: get_loaded(gguf, &g.per_layer_model_proj)?,
                per_layer_proj_norm: get_loaded(gguf, &g.per_layer_proj_norm)?,
            })
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

#[derive(Debug)]
pub struct LayerNames {
    attn_norm: String,
    ffn_norm: String,
    attn_post_norm: Option<String>,
    ffn_post_norm: Option<String>,
    attn_q_norm: Option<String>,
    attn_k_norm: Option<String>,
    wq: String,
    wk: String,
    wv: String,
    wo: String,
    w_gate: String,
    w_up: String,
    w_down: String,
    ple_inp_gate: Option<String>,
    ple_proj: Option<String>,
    ple_post_norm: Option<String>,
    rope_freqs: Option<String>,
    layer_output_scale: Option<String>,
}

#[derive(Debug)]
pub struct Gemma4PleNames {
    pub per_layer_token_embd: String,
    pub per_layer_model_proj: String,
    pub per_layer_proj_norm: String,
}

#[derive(Debug)]
pub struct ModelWeightNames {
    token_embeddings: String,
    output_norm: String,
    lm_head: String,
    layers: Vec<LayerNames>,
    gemma4_ple: Option<Gemma4PleNames>,
}

impl ModelWeightNames {
    pub fn resolve(gguf: &GGUFData, config: &ModelConfig) -> Result<Self, EngineError> {
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
        let lm_head = resolve_lm_head(&available)?;

        let mut layer_names = Vec::with_capacity(config.n_layers);
        for layer_idx in 0..config.n_layers {
            let prefix = format!("blk.{layer_idx}.");
            let (attn_post_norm, ffn_post_norm, ple_inp_gate, ple_proj, ple_post_norm) =
                match config.family {
                    ModelFamily::Gemma4 => {
                        let attn_post_norm = Some(resolve_name_from_strings(
                            &available,
                            &[format!("{prefix}post_attention_norm.weight")],
                        )?);
                        let ffn_post_norm = Some(resolve_name_from_strings(
                            &available,
                            &[format!("{prefix}post_ffw_norm.weight")],
                        )?);
                        if config.embedding_length_per_layer > 0 {
                            (
                                attn_post_norm,
                                ffn_post_norm,
                                Some(resolve_name_from_strings(
                                    &available,
                                    &[format!("{prefix}inp_gate.weight")],
                                )?),
                                Some(resolve_name_from_strings(
                                    &available,
                                    &[format!("{prefix}proj.weight")],
                                )?),
                                Some(resolve_name_from_strings(
                                    &available,
                                    &[format!("{prefix}post_norm.weight")],
                                )?),
                            )
                        } else {
                            (attn_post_norm, ffn_post_norm, None, None, None)
                        }
                    }
                    ModelFamily::MistralLlama => (None, None, None, None, None),
                };
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
                attn_post_norm,
                ffn_post_norm,
                wq: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}attn_q.weight"),
                        format!("{prefix}wq.weight"),
                    ],
                )?,
                wk: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}attn_k.weight"),
                        format!("{prefix}wk.weight"),
                    ],
                )?,
                wv: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}attn_v.weight"),
                        format!("{prefix}wv.weight"),
                    ],
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
                    &[
                        format!("{prefix}ffn_gate.weight"),
                        format!("{prefix}w1.weight"),
                    ],
                )?,
                w_up: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}ffn_up.weight"),
                        format!("{prefix}w3.weight"),
                    ],
                )?,
                w_down: resolve_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}ffn_down.weight"),
                        format!("{prefix}w2.weight"),
                    ],
                )?,
                attn_q_norm: optional_name_from_strings(
                    &available,
                    &[format!("{prefix}attn_q_norm.weight")],
                ),
                attn_k_norm: optional_name_from_strings(
                    &available,
                    &[format!("{prefix}attn_k_norm.weight")],
                ),
                ple_inp_gate,
                ple_proj,
                ple_post_norm,
                rope_freqs: optional_name_from_strings(
                    &available,
                    &[
                        format!("{prefix}rope_freqs.weight"),
                        format!("{prefix}rope_freqs"),
                    ],
                ),
                layer_output_scale: optional_name_from_strings(
                    &available,
                    &[format!("{prefix}layer_output_scale.weight")],
                ),
            });
        }

        let gemma4_ple =
            if config.family == ModelFamily::Gemma4 && config.embedding_length_per_layer > 0 {
                Some(Gemma4PleNames {
                    per_layer_token_embd: resolve_name_from_strs(
                        &available,
                        &["per_layer_token_embd.weight"],
                    )?,
                    per_layer_model_proj: resolve_name_from_strs(
                        &available,
                        &["per_layer_model_proj.weight"],
                    )?,
                    per_layer_proj_norm: resolve_name_from_strs(
                        &available,
                        &["per_layer_proj_norm.weight"],
                    )?,
                })
            } else {
                None
            };

        Ok(Self {
            token_embeddings,
            output_norm,
            lm_head,
            layers: layer_names,
            gemma4_ple,
        })
    }

    pub fn load_all(&self, gguf: &mut GGUFData, file_path: &str) -> Result<(), EngineError> {
        let mut names_to_load = Vec::new();
        names_to_load.push(self.token_embeddings.clone());
        names_to_load.push(self.output_norm.clone());
        names_to_load.push(self.lm_head.clone());
        for layer in &self.layers {
            names_to_load.push(layer.attn_norm.clone());
            names_to_load.push(layer.ffn_norm.clone());
            if let Some(ref n) = layer.attn_post_norm {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.ffn_post_norm {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.attn_q_norm {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.attn_k_norm {
                names_to_load.push(n.clone());
            }
            names_to_load.push(layer.wq.clone());
            names_to_load.push(layer.wk.clone());
            names_to_load.push(layer.wv.clone());
            names_to_load.push(layer.wo.clone());
            names_to_load.push(layer.w_gate.clone());
            names_to_load.push(layer.w_up.clone());
            names_to_load.push(layer.w_down.clone());
            if let Some(ref n) = layer.ple_inp_gate {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.ple_proj {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.ple_post_norm {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.rope_freqs {
                names_to_load.push(n.clone());
            }
            if let Some(ref n) = layer.layer_output_scale {
                names_to_load.push(n.clone());
            }
        }
        if let Some(ref g) = self.gemma4_ple {
            names_to_load.push(g.per_layer_token_embd.clone());
            names_to_load.push(g.per_layer_model_proj.clone());
            names_to_load.push(g.per_layer_proj_norm.clone());
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

fn resolve_lm_head(available: &HashSet<String>) -> Result<String, EngineError> {
    if let Ok(n) = resolve_name_from_strs(available, &["output.weight", "lm_head.weight"]) {
        return Ok(n);
    }
    if available.contains("token_embd.weight") {
        return Ok("token_embd.weight".to_string());
    }
    Err(EngineError::Model(
        "LM head: none of output.weight, lm_head.weight, token_embd.weight (tied) found".into(),
    ))
}

fn optional_name_from_strings(
    available: &HashSet<String>,
    candidates: &[String],
) -> Option<String> {
    for candidate in candidates {
        if available.contains(candidate) {
            return Some(candidate.clone());
        }
    }
    None
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

fn get_loaded<'a>(gguf: &'a GGUFData, name: &str) -> Result<&'a Tensor, EngineError> {
    gguf.get_tensor(name)
        .ok_or_else(|| EngineError::Model(format!("tensor '{name}' not found after loading")))
}
