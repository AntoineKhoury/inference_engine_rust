use std::collections::HashSet;

use crate::EngineError;
use crate::model_config::{ModelConfig, ModelFamily};
use crate::model_loader::gguf_types::GGUFData;

/// Resolved GGUF tensor names for a single transformer block.
///
/// Fields are `pub(crate)` so [`super::view`] can build borrowed [`super::view::LayerWeights`]
/// from the same names without exposing the raw strings publicly.
#[derive(Debug)]
pub struct LayerNames {
    pub(crate) attn_norm: String,
    pub(crate) ffn_norm: String,
    pub(crate) attn_post_norm: Option<String>,
    pub(crate) ffn_post_norm: Option<String>,
    pub(crate) attn_q_norm: Option<String>,
    pub(crate) attn_k_norm: Option<String>,
    pub(crate) wq: String,
    pub(crate) wk: String,
    pub(crate) wv: String,
    pub(crate) wo: String,
    pub(crate) w_gate: String,
    pub(crate) w_up: String,
    pub(crate) w_down: String,
    pub(crate) ple_inp_gate: Option<String>,
    pub(crate) ple_proj: Option<String>,
    pub(crate) ple_post_norm: Option<String>,
    pub(crate) rope_freqs: Option<String>,
    pub(crate) layer_output_scale: Option<String>,
}

/// Resolved names for Gemma 4 global PLE tensors (not per-layer).
#[derive(Debug)]
pub struct Gemma4PleNames {
    pub per_layer_token_embd: String,
    pub per_layer_model_proj: String,
    pub per_layer_proj_norm: String,
}

/// Resolved GGUF tensor names for the entire model, ready to be loaded.
#[derive(Debug)]
pub struct ModelWeightNames {
    pub(crate) token_embeddings: String,
    pub(crate) output_norm: String,
    pub(crate) lm_head: String,
    pub(crate) layers: Vec<LayerNames>,
    pub(crate) gemma4_ple: Option<Gemma4PleNames>,
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
