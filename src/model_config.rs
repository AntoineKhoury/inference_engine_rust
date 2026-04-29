use crate::EngineError;
use crate::model_loader::gguf_types::{Data, GGUFData};

/// Tokenizer special-token policy read from GGUF (same keys as llama.cpp / `tokenizer.ggml.*`).
///
/// Use this to align Rust token IDs with [`llama-tokenize`] / `llama-completion` for the same file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerPromptConfig {
    /// If true, prepend [`Self::bos_token_id`] before the SentencePiece-encoded prompt.
    pub add_bos_token: bool,
    /// If true, append [`Self::eos_token_id`] after the encoded prompt (rare for raw completion).
    pub add_eos_token: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

impl Default for TokenizerPromptConfig {
    fn default() -> Self {
        Self {
            add_bos_token: false,
            add_eos_token: false,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

impl TokenizerPromptConfig {
    /// Load from GGUF KV metadata.
    ///
    /// Matches [llama.cpp `llama_vocab::impl::load`](https://github.com/ggml-org/llama.cpp): for
    /// SentencePiece / `tokenizer.ggml.model == "llama"`, the implicit defaults are **add BOS**,
    /// **no EOS** on encode, unless `tokenizer.ggml.add_bos_token` / `add_eos_token` override them.
    /// Many GGUFs (including some Mistral builds) omit those override keys entirely.
    pub fn from_gguf(gguf: &GGUFData) -> Result<Self, EngineError> {
        let tokenizer_model = get_string(gguf, "tokenizer.ggml.model").unwrap_or_default();
        let (default_bos, default_eos) = defaults_for_tokenizer_model(tokenizer_model.as_str());

        let add_bos_token = get_bool(gguf, "tokenizer.ggml.add_bos_token").unwrap_or(default_bos);
        let add_eos_token = get_bool(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(default_eos);
        let bos_token_id = get_u32(gguf, "tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_token_id = get_u32(gguf, "tokenizer.ggml.eos_token_id").unwrap_or(2);
        Ok(Self {
            add_bos_token,
            add_eos_token,
            bos_token_id,
            eos_token_id,
        })
    }
}

/// Implicit `add_bos` / `add_eos` when GGUF omits `tokenizer.ggml.add_*` (see llama.cpp vocab load).
fn defaults_for_tokenizer_model(tokenizer_model: &str) -> (bool, bool) {
    match tokenizer_model {
        // LLAMA_VOCAB_TYPE_SPM: add_bos = true, add_eos = false
        "llama" => (true, false),
        // Other vocab types set different defaults in llama.cpp; keep conservative unless we need them.
        _ => (false, false),
    }
}

/// High-level checkpoint family so loaders and tests can branch without stringly-typed checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelFamily {
    #[default]
    /// Dense Mistral / Llama-style decoder (full attention every layer unless overridden in GGUF).
    MistralLlama,
    /// Gemma 4 hybrid attention: per-layer sliding-window vs full context (see [`LayerAttentionSpec`]).
    Gemma4,
}

/// Per-layer attention / RoPE settings. Dense models use the same spec on every layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerAttentionSpec {
    /// `None`: full causal attention. `Some(w)`: only the last `w` keys (including current).
    pub sliding_window: Option<usize>,
    pub rope_theta: f32,
    /// RoPE applies to the first `rope_rotary_dim` elements of each head (even, ≤ model `head_dim`).
    pub rope_rotary_dim: usize,
}

impl LayerAttentionSpec {
    pub fn full_causal(rope_theta: f32, head_dim: usize) -> Self {
        Self {
            sliding_window: None,
            rope_theta,
            rope_rotary_dim: head_dim,
        }
    }
}

/// Per-layer tensor-derived widths (Q/K/V projections and FFN). Dense models use the same values
/// on every layer; Gemma 4 hybrid checkpoints vary SWA vs global layers.
#[derive(Debug, Clone)]
pub struct LayerDims {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub family: ModelFamily,
    pub context_length: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    /// Representative head width (dense: `hidden_dim / n_heads`; Gemma 4: max across layers).
    pub head_dim: usize,
    /// Representative FFN inner size (dense: global metadata; Gemma 4: max across layers).
    pub ffn_dim: usize,
    /// One entry per `blk.{i}`; authoritative for matmul and KV head width.
    pub layer_dims: Vec<LayerDims>,
    /// Default / “global” RoPE base from GGUF (`llama.rope.theta`). Per-layer values live in
    /// [`Self::layer_attention`]; this stays for diagnostics and Gemma-free checkpoints.
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    /// If true, undo HF→GGUF `LlamaModel.permute` on Q/K **activations** (`convert_hf_to_gguf.py`).
    /// Mistral GGUFs (`MistralModel.undo_permute = false`) default **false** via `general.name` … `mistral`;
    /// Llama permuted checkpoints default **true**. Override: `INFERENCE_ENGINE_GGUF_QK_UNPACK=0|1` or
    /// `INFERENCE_ENGINE_GGUF_NO_QK_UNPACK=1` (off).
    pub unpack_llama_gguf_qk: bool,
    /// One entry per transformer block, aligned with `blk.{i}.*` tensors.
    pub layer_attention: Vec<LayerAttentionSpec>,
    /// HF `Gemma4TextScaledWordEmbedding`: multiply token rows by `sqrt(hidden_dim)` (1.0 for other families).
    pub token_embedding_scale: f32,
    /// Gemma 4 PLE width (`gemma4.embedding_length_per_layer_input`); 0 means disabled.
    pub embedding_length_per_layer: usize,
    /// `1/sqrt(2)` when combining PLE context + token identity; unused when PLE disabled.
    pub ple_combine_scale: f32,
    /// `1/sqrt(hidden_dim)` for PLE projection path; unused when PLE disabled.
    pub ple_model_proj_scale: f32,
    /// Gemma 4: for layer `i`, if `Some(s)`, attention uses K/V from layer `s`'s [`KVCache`] (HF shared KV); `None` = normal layer.
    pub gemma4_kv_borrow_from: Vec<Option<usize>>,
    /// Gemma 4: `gemma4.final_logit_softcapping` — `tanh(x/cap)*cap` on LM logits; `None` if absent.
    pub final_logit_softcapping: Option<f32>,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GGUFData) -> Result<Self, EngineError> {
        let context_length =
            get_usize_alt(gguf, &["llama.context_length", "gemma4.context_length"])?;
        let hidden_dim =
            get_usize_alt(gguf, &["llama.embedding_length", "gemma4.embedding_length"])?;
        let n_layers = get_usize_alt(gguf, &["llama.block_count", "gemma4.block_count"])?;
        let n_heads = get_usize_alt(
            gguf,
            &["llama.attention.head_count", "gemma4.attention.head_count"],
        )?;
        let n_kv_heads = get_usize_opt(gguf, "llama.attention.head_count_kv")
            .or_else(|| get_usize_opt(gguf, "gemma4.attention.head_count_kv"))
            .unwrap_or(n_heads);
        let ffn_dim_meta = get_usize_alt(
            gguf,
            &["llama.feed_forward_length", "gemma4.feed_forward_length"],
        )?;
        let rope_theta = get_f32_opt(gguf, "llama.rope.theta")
            .or_else(|| get_f32_opt(gguf, "gemma4.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_norm_eps = get_f32_alt(
            gguf,
            &[
                "llama.attention.layer_norm_rms_epsilon",
                "gemma4.attention.layer_norm_rms_epsilon",
            ],
        )?;
        let vocab_size = if let Some(v) = get_usize_opt(gguf, "llama.vocab_size")
            .or_else(|| get_usize_opt(gguf, "gemma4.vocab_size"))
        {
            v
        } else {
            get_array_len(gguf, "tokenizer.ggml.tokens")?
        };

        // Gemma 4 may report `gemma4.attention.key_length` for KV heads that do not match
        // `embedding_length / head_count` (hybrid SWA/global). Use the quotient when it matches;
        // otherwise fall back to `hidden_dim / n_heads` for this dense path.
        let head_dim = if let Some(kd) = get_usize_opt(gguf, "gemma4.attention.key_length") {
            if kd * n_heads == hidden_dim {
                kd
            } else if hidden_dim % n_heads != 0 {
                return Err(EngineError::Model(format!(
                    "hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
                )));
            } else {
                hidden_dim / n_heads
            }
        } else if hidden_dim % n_heads != 0 {
            return Err(EngineError::Model(format!(
                "hidden_dim {hidden_dim} not divisible by n_heads {n_heads}"
            )));
        } else {
            hidden_dim / n_heads
        };

        if n_heads % n_kv_heads != 0 {
            return Err(EngineError::Model(format!(
                "GQA requires n_heads ({n_heads}) divisible by n_kv_heads ({n_kv_heads})"
            )));
        }

        // HF→GGUF `LlamaModel.permute` on Q/K weights is **not** applied for `MistralModel`
        // (`convert_hf_to_gguf.py`: `MistralModel.undo_permute = False`). Those checkpoints need
        // **no** activation unpack; Llama-family builds that ran permute need unpack on.
        let family = detect_model_family(gguf);
        let general_name = get_string(gguf, "general.name").unwrap_or_default();
        let lower = general_name.to_ascii_lowercase();
        let mistral_style_gguf = lower.contains("mistral") || lower.contains("gemma");
        let unpack_llama_gguf_qk = match std::env::var("INFERENCE_ENGINE_GGUF_QK_UNPACK") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
            _ => match std::env::var("INFERENCE_ENGINE_GGUF_NO_QK_UNPACK") {
                Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => false,
                _ => match family {
                    // Gemma 4 arch is Mistral-style in GGUF (no Llama Q/K permute); do not rely on
                    // `general.name` containing "gemma" (some exports use neutral names).
                    ModelFamily::Gemma4 => false,
                    ModelFamily::MistralLlama => !mistral_style_gguf,
                },
            },
        };
        let layer_dims = infer_layer_dims(
            gguf,
            family,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            head_dim,
            ffn_dim_meta,
        )?;

        let head_dim = match family {
            ModelFamily::MistralLlama => head_dim,
            ModelFamily::Gemma4 => layer_dims
                .iter()
                .map(|d| d.head_dim)
                .max()
                .ok_or_else(|| EngineError::Model("layer_dims empty".into()))?,
        };
        let ffn_dim = match family {
            ModelFamily::MistralLlama => ffn_dim_meta,
            ModelFamily::Gemma4 => layer_dims
                .iter()
                .map(|d| d.ffn_dim)
                .max()
                .ok_or_else(|| EngineError::Model("layer_dims empty".into()))?,
        };

        let layer_attention =
            build_layer_attention_specs(gguf, family, n_layers, &layer_dims, rope_theta)?;

        let embedding_length_per_layer = match family {
            ModelFamily::Gemma4 => {
                get_usize_opt(gguf, "gemma4.embedding_length_per_layer_input").unwrap_or(0)
            }
            ModelFamily::MistralLlama => 0,
        };
        let token_embedding_scale = match family {
            ModelFamily::Gemma4 => (hidden_dim as f32).sqrt(),
            ModelFamily::MistralLlama => 1.0,
        };
        let (ple_combine_scale, ple_model_proj_scale) = if embedding_length_per_layer > 0 {
            (1.0 / 2.0_f32.sqrt(), 1.0 / (hidden_dim as f32).sqrt())
        } else {
            (0.0, 0.0)
        };

        let gemma4_kv_borrow_from = match family {
            ModelFamily::Gemma4 => {
                let n_shared =
                    get_usize_opt(gguf, "gemma4.attention.shared_kv_layers").unwrap_or(0);
                build_gemma4_kv_borrow_from(n_layers, n_shared, &layer_attention)?
            }
            ModelFamily::MistralLlama => vec![None; n_layers],
        };

        let final_logit_softcapping = get_f32_opt(gguf, "gemma4.final_logit_softcapping")
            .filter(|&x| x > 0.0 && x.is_finite());

        Ok(Self {
            family,
            context_length,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            layer_dims,
            rope_theta,
            rms_norm_eps,
            vocab_size,
            unpack_llama_gguf_qk,
            layer_attention,
            token_embedding_scale,
            embedding_length_per_layer,
            ple_combine_scale,
            ple_model_proj_scale,
            gemma4_kv_borrow_from,
            final_logit_softcapping,
        })
    }

    pub fn layer_dims_for(&self, layer_idx: usize) -> Result<&LayerDims, EngineError> {
        self.layer_dims.get(layer_idx).ok_or_else(|| {
            EngineError::Model(format!(
                "layer_dims: index {layer_idx} out of range (n_layers = {})",
                self.n_layers
            ))
        })
    }

    pub fn layer_attention_for(
        &self,
        layer_idx: usize,
    ) -> Result<&LayerAttentionSpec, EngineError> {
        self.layer_attention.get(layer_idx).ok_or_else(|| {
            EngineError::Model(format!(
                "layer_attention: index {layer_idx} out of range (n_layers = {})",
                self.n_layers
            ))
        })
    }
}

/// HF `Gemma4TextAttention`: last `num_kv_shared` layers reuse K/V from the last earlier layer with the same
/// sliding vs full pattern (`layer_types` match).
fn build_gemma4_kv_borrow_from(
    n_layers: usize,
    num_kv_shared: usize,
    layer_attention: &[LayerAttentionSpec],
) -> Result<Vec<Option<usize>>, EngineError> {
    let mut out = vec![None; n_layers];
    if num_kv_shared == 0 {
        return Ok(out);
    }
    if layer_attention.len() != n_layers {
        return Err(EngineError::Model(format!(
            "build_gemma4_kv_borrow_from: layer_attention len {} != n_layers {n_layers}",
            layer_attention.len()
        )));
    }
    let first_shared = n_layers.checked_sub(num_kv_shared).ok_or_else(|| {
        EngineError::Model("gemma4 attention.shared_kv_layers exceeds block_count".into())
    })?;
    if first_shared == 0 {
        return Err(EngineError::Model(
            "gemma4: shared_kv_layers == block_count is unsupported (no KV source layer)".into(),
        ));
    }
    for layer_idx in first_shared..n_layers {
        let want_swa = layer_attention[layer_idx].sliding_window.is_some();
        let mut src = None;
        for s in (0..first_shared).rev() {
            if layer_attention[s].sliding_window.is_some() == want_swa {
                src = Some(s);
                break;
            }
        }
        out[layer_idx] = src;
        if out[layer_idx].is_none() {
            return Err(EngineError::Model(format!(
                "Gemma 4: no KV-borrow source for layer {layer_idx} (match sliding/full in 0..{first_shared})"
            )));
        }
    }
    Ok(out)
}

fn infer_layer_dims(
    gguf: &GGUFData,
    family: ModelFamily,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    hidden_dim: usize,
    head_dim_uniform: usize,
    ffn_dim_uniform: usize,
) -> Result<Vec<LayerDims>, EngineError> {
    match family {
        ModelFamily::MistralLlama => {
            let q_dim = n_heads.checked_mul(head_dim_uniform).ok_or_else(|| {
                EngineError::Model("layer dims: n_heads * head_dim overflow".into())
            })?;
            let kv_dim = n_kv_heads.checked_mul(head_dim_uniform).ok_or_else(|| {
                EngineError::Model("layer dims: n_kv_heads * head_dim overflow".into())
            })?;
            Ok(vec![
                LayerDims {
                    q_dim,
                    kv_dim,
                    head_dim: head_dim_uniform,
                    ffn_dim: ffn_dim_uniform,
                };
                n_layers
            ])
        }
        ModelFamily::Gemma4 => {
            let mut out = Vec::with_capacity(n_layers);
            for layer_idx in 0..n_layers {
                let prefix = format!("blk.{layer_idx}.");
                let q_name = format!("{prefix}attn_q.weight");
                let k_name = format!("{prefix}attn_k.weight");
                let gate_name = format!("{prefix}ffn_gate.weight");
                let (k_q, q_dim) = tensor_weight_k_n(gguf, &q_name)?;
                let (k_k, kv_dim) = tensor_weight_k_n(gguf, &k_name)?;
                let (k_g, ffn_d) = tensor_weight_k_n(gguf, &gate_name)?;
                if k_q != hidden_dim || k_k != hidden_dim || k_g != hidden_dim {
                    return Err(EngineError::Model(format!(
                        "layer {layer_idx}: expected attn/FFN input dim {hidden_dim}, got q={k_q} k={k_k} gate={k_g}"
                    )));
                }
                if q_dim % n_heads != 0 {
                    return Err(EngineError::Model(format!(
                        "{q_name}: out dim {q_dim} not divisible by n_heads {n_heads}"
                    )));
                }
                if kv_dim % n_kv_heads != 0 {
                    return Err(EngineError::Model(format!(
                        "{k_name}: out dim {kv_dim} not divisible by n_kv_heads {n_kv_heads}"
                    )));
                }
                let hd_q = q_dim / n_heads;
                let hd_kv = kv_dim / n_kv_heads;
                if hd_q != hd_kv {
                    return Err(EngineError::Model(format!(
                        "layer {layer_idx}: Q head dim {hd_q} != KV head dim {hd_kv}"
                    )));
                }
                out.push(LayerDims {
                    q_dim,
                    kv_dim,
                    head_dim: hd_q,
                    ffn_dim: ffn_d,
                });
            }
            Ok(out)
        }
    }
}

fn tensor_weight_k_n(gguf: &GGUFData, name: &str) -> Result<(usize, usize), EngineError> {
    let meta = gguf
        .tensors_metadata()
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| EngineError::Model(format!("missing tensor metadata '{name}'")))?;
    if meta.dimensions.len() < 2 {
        return Err(EngineError::Model(format!(
            "tensor '{name}': expected 2 dimensions, got {}",
            meta.dimensions.len()
        )));
    }
    Ok((meta.dimensions[0], meta.dimensions[1]))
}

fn detect_model_family(gguf: &GGUFData) -> ModelFamily {
    let arch = get_string(gguf, "general.architecture")
        .unwrap_or_default()
        .to_ascii_lowercase();
    let name = get_string(gguf, "general.name")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if arch.contains("gemma4")
        || arch.contains("gemma-4")
        || name.contains("gemma-4")
        || name.contains("gemma4")
    {
        return ModelFamily::Gemma4;
    }
    if gguf
        .get_metadata("gemma4.attention.sliding_window_pattern")
        .is_some()
    {
        return ModelFamily::Gemma4;
    }
    ModelFamily::MistralLlama
}

fn build_layer_attention_specs(
    gguf: &GGUFData,
    family: ModelFamily,
    n_layers: usize,
    layer_dims: &[LayerDims],
    default_rope_theta: f32,
) -> Result<Vec<LayerAttentionSpec>, EngineError> {
    if layer_dims.len() != n_layers {
        return Err(EngineError::Model(format!(
            "layer_attention: layer_dims len {} != n_layers {n_layers}",
            layer_dims.len()
        )));
    }
    let hd0 = layer_dims[0].head_dim;
    if hd0 == 0 || hd0 % 2 != 0 {
        return Err(EngineError::Model(format!(
            "layer attention: head_dim must be positive and even (got {hd0})"
        )));
    }

    match family {
        ModelFamily::MistralLlama => {
            let spec = LayerAttentionSpec::full_causal(default_rope_theta, hd0);
            Ok(vec![spec; n_layers])
        }
        ModelFamily::Gemma4 => {
            build_gemma4_layer_attention(gguf, n_layers, layer_dims, default_rope_theta)
        }
    }
}

fn build_gemma4_layer_attention(
    gguf: &GGUFData,
    n_layers: usize,
    layer_dims: &[LayerDims],
    default_rope_theta: f32,
) -> Result<Vec<LayerAttentionSpec>, EngineError> {
    if layer_dims.len() != n_layers {
        return Err(EngineError::Model(format!(
            "Gemma 4 layer_attention: layer_dims len {} != n_layers {n_layers}",
            layer_dims.len()
        )));
    }
    let pattern = match gguf.get_metadata("gemma4.attention.sliding_window_pattern") {
        Some(Data::Array(items)) => {
            if items.len() != n_layers {
                return Err(EngineError::Model(format!(
                    "gemma4.attention.sliding_window_pattern length {} != n_layers {n_layers}",
                    items.len()
                )));
            }
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Data::Bool(b) => out.push(*b),
                    Data::Uint8(v) => out.push(*v != 0),
                    Data::Int8(v) => out.push(*v != 0),
                    Data::Uint32(v) => out.push(*v != 0),
                    Data::Int32(v) => out.push(*v != 0),
                    _ => {
                        return Err(EngineError::Model(
                            "gemma4.attention.sliding_window_pattern must be bool[] (or integer 0/1)"
                                .into(),
                        ));
                    }
                }
            }
            out
        }
        Some(_) => {
            return Err(EngineError::Model(
                "gemma4.attention.sliding_window_pattern has wrong type".into(),
            ));
        }
        None => {
            return Err(EngineError::Model(
                "Gemma 4 GGUF missing gemma4.attention.sliding_window_pattern".into(),
            ));
        }
    };

    let window = get_usize_opt(gguf, "gemma4.attention.sliding_window")
        .or_else(|| get_usize_opt(gguf, "llama.attention.sliding_window"))
        .ok_or_else(|| {
            EngineError::Model(
                "Gemma 4 GGUF missing sliding window size (gemma4.attention.sliding_window or llama.attention.sliding_window)".into(),
            )
        })?;
    if window == 0 {
        return Err(EngineError::Model(
            "Gemma 4 sliding window size must be > 0".into(),
        ));
    }

    // GGUF uses `gemma4.rope.dimension_count(_swa)`; clamp per layer to that layer's `head_dim`.
    let default_hd = layer_dims.first().map(|d| d.head_dim).unwrap_or(1);
    let rope_rotary_dim_local_meta = get_usize_opt(gguf, "gemma4.rope.dimension_count_swa")
        .or_else(|| get_usize_opt(gguf, "gemma4.attention.rotary_dim_local"))
        .or_else(|| get_usize_opt(gguf, "llama.rope.dimension_count"))
        .unwrap_or(default_hd);
    let rope_rotary_dim_global_meta = get_usize_opt(gguf, "gemma4.rope.dimension_count")
        .or_else(|| get_usize_opt(gguf, "gemma4.attention.rotary_dim_global"))
        .unwrap_or(rope_rotary_dim_local_meta);
    let clamp_rotary = |mut rd: usize, max_hd: usize| -> usize {
        rd = rd.min(max_hd);
        if rd == 0 {
            rd = max_hd;
        }
        if rd % 2 != 0 {
            rd -= 1;
        }
        rd
    };

    let rope_local = get_f32_opt(gguf, "gemma4.rope.freq_base_swa")
        .or_else(|| get_f32_opt(gguf, "gemma4.rope.local_theta"))
        .unwrap_or(default_rope_theta);
    let rope_global = get_f32_opt(gguf, "gemma4.rope.freq_base")
        .or_else(|| get_f32_opt(gguf, "gemma4.rope.global_theta"))
        .unwrap_or(default_rope_theta);

    let mut out = Vec::with_capacity(n_layers);
    for (i, is_swa) in pattern.into_iter().enumerate() {
        let max_hd = layer_dims
            .get(i)
            .map(|d| d.head_dim)
            .ok_or_else(|| EngineError::Model(format!("layer_dims missing entry for layer {i}")))?;
        let raw_rd = if is_swa {
            rope_rotary_dim_local_meta
        } else {
            rope_rotary_dim_global_meta
        };
        let rotary_dim = clamp_rotary(raw_rd, max_hd);
        let (swa, theta) = if is_swa {
            (Some(window), rope_local)
        } else {
            (None, rope_global)
        };
        out.push(LayerAttentionSpec {
            sliding_window: swa,
            rope_theta: theta,
            rope_rotary_dim: rotary_dim,
        });
    }
    Ok(out)
}

fn get_usize_alt(gguf: &GGUFData, keys: &[&str]) -> Result<usize, EngineError> {
    for key in keys {
        if let Some(v) = get_usize_opt(gguf, key) {
            return Ok(v);
        }
    }
    Err(EngineError::Model(format!(
        "missing integer metadata; tried keys {keys:?}"
    )))
}

fn get_f32_alt(gguf: &GGUFData, keys: &[&str]) -> Result<f32, EngineError> {
    for key in keys {
        if let Some(v) = get_f32_opt(gguf, key) {
            return Ok(v);
        }
    }
    Err(EngineError::Model(format!(
        "missing float metadata; tried keys {keys:?}"
    )))
}

fn get_usize_opt(gguf: &GGUFData, key: &str) -> Option<usize> {
    match gguf.get_metadata(key)? {
        Data::Uint32(v) => Some(*v as usize),
        Data::Int32(v) => usize::try_from(*v).ok(),
        Data::Uint64(v) => usize::try_from(*v).ok(),
        Data::Int64(v) => usize::try_from(*v).ok(),
        Data::Uint16(v) => Some(*v as usize),
        Data::Int16(v) => usize::try_from(*v as i32).ok(),
        Data::Uint8(v) => Some(*v as usize),
        Data::Int8(v) => usize::try_from(*v as i32).ok(),
        Data::Array(arr) => {
            // llama.cpp stores `feed_forward_length` etc. as u32[] for some arches (e.g. Gemma 4).
            let first = arr.first()?;
            get_usize_opt_from_data_elem(first)
        }
        _ => None,
    }
}

fn get_usize_opt_from_data_elem(d: &Data) -> Option<usize> {
    match d {
        Data::Uint32(v) => Some(*v as usize),
        Data::Int32(v) => usize::try_from(*v).ok(),
        Data::Uint64(v) => usize::try_from(*v).ok(),
        Data::Int64(v) => usize::try_from(*v).ok(),
        Data::Uint16(v) => Some(*v as usize),
        Data::Int16(v) => usize::try_from(*v as i32).ok(),
        Data::Uint8(v) => Some(*v as usize),
        Data::Int8(v) => usize::try_from(*v as i32).ok(),
        _ => None,
    }
}

fn get_f32_opt(gguf: &GGUFData, key: &str) -> Option<f32> {
    match gguf.get_metadata(key)? {
        Data::Float32(v) => Some(*v),
        Data::Float64(v) => Some(*v as f32),
        _ => None,
    }
}

fn get_array_len(gguf: &GGUFData, key: &str) -> Result<usize, EngineError> {
    match gguf.get_metadata(key) {
        Some(Data::Array(v)) => Ok(v.len()),
        Some(_) => Err(EngineError::Model(format!(
            "metadata key '{key}' is not an array"
        ))),
        None => Err(EngineError::Model(format!("missing metadata key '{key}'"))),
    }
}

fn get_bool(gguf: &GGUFData, key: &str) -> Option<bool> {
    match gguf.get_metadata(key)? {
        Data::Bool(b) => Some(*b),
        _ => None,
    }
}

fn get_u32(gguf: &GGUFData, key: &str) -> Option<u32> {
    match gguf.get_metadata(key)? {
        Data::Uint32(v) => Some(*v),
        Data::Int32(v) => u32::try_from(*v).ok(),
        Data::Uint16(v) => Some(*v as u32),
        Data::Int16(v) => u32::try_from(*v as i32).ok(),
        Data::Uint8(v) => Some(*v as u32),
        Data::Int8(v) => u32::try_from(*v as i32).ok(),
        Data::Uint64(v) => u32::try_from(*v).ok(),
        Data::Int64(v) => u32::try_from(*v).ok(),
        _ => None,
    }
}

fn get_string(gguf: &GGUFData, key: &str) -> Option<String> {
    match gguf.get_metadata(key)? {
        Data::String(s) => Some(s.clone()),
        _ => None,
    }
}
