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
    pub fn from_gguf(gguf: &GGUFData) -> Result<Self, Box<dyn std::error::Error>> {
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

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub context_length: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    /// If true, undo HF→GGUF `LlamaModel.permute` on Q/K **activations** (`convert_hf_to_gguf.py`).
    /// Mistral GGUFs (`MistralModel.undo_permute = false`) default **false** via `general.name` … `mistral`;
    /// Llama permuted checkpoints default **true**. Override: `INFERENCE_ENGINE_GGUF_QK_UNPACK=0|1` or
    /// `INFERENCE_ENGINE_GGUF_NO_QK_UNPACK=1` (off).
    pub unpack_llama_gguf_qk: bool,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GGUFData) -> Result<Self, Box<dyn std::error::Error>> {
        let context_length = get_usize(gguf, "llama.context_length")?;
        let hidden_dim = get_usize(gguf, "llama.embedding_length")?;
        let n_layers = get_usize(gguf, "llama.block_count")?;
        let n_heads = get_usize(gguf, "llama.attention.head_count")?;
        let n_kv_heads =
            get_usize(gguf, "llama.attention.head_count_kv").unwrap_or(n_heads);
        let ffn_dim = get_usize(gguf, "llama.feed_forward_length")?;
        let rope_theta = get_f32(gguf, "llama.rope.theta").unwrap_or(10000.0);
        let rms_norm_eps = get_f32(gguf, "llama.attention.layer_norm_rms_epsilon")?;
        let vocab_size = get_usize(gguf, "llama.vocab_size")
            .or_else(|_| get_array_len(gguf, "tokenizer.ggml.tokens"))?;

        if hidden_dim % n_heads != 0 {
            return Err(format!(
                "hidden_dim {} not divisible by n_heads {}",
                hidden_dim, n_heads
            )
            .into());
        }
        if n_heads % n_kv_heads != 0 {
            return Err(format!(
                "GQA requires n_heads ({}) divisible by n_kv_heads ({})",
                n_heads, n_kv_heads
            )
            .into());
        }
        let head_dim = hidden_dim / n_heads;

        // HF→GGUF `LlamaModel.permute` on Q/K weights is **not** applied for `MistralModel`
        // (`convert_hf_to_gguf.py`: `MistralModel.undo_permute = False`). Those checkpoints need
        // **no** activation unpack; Llama-family builds that ran permute need unpack on.
        let general_name = get_string(gguf, "general.name").unwrap_or_default();
        let mistral_style_gguf = general_name.to_ascii_lowercase().contains("mistral");
        let unpack_llama_gguf_qk = match std::env::var("INFERENCE_ENGINE_GGUF_QK_UNPACK") {
            Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
            _ => match std::env::var("INFERENCE_ENGINE_GGUF_NO_QK_UNPACK") {
                Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") => false,
                _ => !mistral_style_gguf,
            },
        };

        Ok(Self {
            context_length,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            rope_theta,
            rms_norm_eps,
            vocab_size,
            unpack_llama_gguf_qk,
        })
    }
}

fn get_usize(gguf: &GGUFData, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    match gguf.get_metadata(key) {
        Some(Data::Uint32(v)) => Ok(*v as usize),
        Some(Data::Int32(v)) => Ok(*v as usize),
        Some(Data::Uint64(v)) => Ok(*v as usize),
        Some(Data::Int64(v)) => Ok(*v as usize),
        Some(Data::Uint16(v)) => Ok(*v as usize),
        Some(Data::Int16(v)) => Ok(*v as usize),
        Some(Data::Uint8(v)) => Ok(*v as usize),
        Some(Data::Int8(v)) => Ok(*v as usize),
        Some(_) => Err(format!("Metadata key '{}' is not an integer", key).into()),
        None => Err(format!("Missing metadata key '{}'", key).into()),
    }
}

fn get_f32(gguf: &GGUFData, key: &str) -> Result<f32, Box<dyn std::error::Error>> {
    match gguf.get_metadata(key) {
        Some(Data::Float32(v)) => Ok(*v),
        Some(Data::Float64(v)) => Ok(*v as f32),
        Some(_) => Err(format!("Metadata key '{}' is not a float", key).into()),
        None => Err(format!("Missing metadata key '{}'", key).into()),
    }
}

fn get_array_len(gguf: &GGUFData, key: &str) -> Result<usize, Box<dyn std::error::Error>> {
    match gguf.get_metadata(key) {
        Some(Data::Array(v)) => Ok(v.len()),
        Some(_) => Err(format!("Metadata key '{}' is not an array", key).into()),
        None => Err(format!("Missing metadata key '{}'", key).into()),
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
