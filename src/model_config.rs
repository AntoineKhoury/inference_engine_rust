use crate::model_loader::gguf_types::{Data, GGUFData};

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
