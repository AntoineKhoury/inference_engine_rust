//! Shared integration-test constants (GGUF reference model + Gemma paths).
//!
//! Each `tests/*.rs` binary only uses a subset of these; `reference_model.rs` uses most.
//!
//! Weights live under **`model/<model-name>/`** (see repo **`model/README.md`**).
//!
//! Reference floats for `token_embd` token_id=2 were taken from PyPI **`gguf`** `dequantize()`
//! (ggml-compatible), shape `(32000, 4096)` row `token_id`, matching this engine’s
//! `token_id * hidden_dim + h` buffer layout.
#![allow(dead_code)]

pub mod llama_logits_helpers;
pub mod llama_tensor_dump_helpers;

use std::path::{Path, PathBuf};

/// Relative to workspace root (same as `CARGO_MANIFEST_DIR` for integration tests).
pub const REFERENCE_MODEL_REL_PATH: &str = "model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf";

pub const REFERENCE_MODEL_DISPLAY_NAME: &str = "TheBloke Mistral-7B-v0.1 Q4_K_M";

/// Direct HTTP URL for optional `curl` download test (`tests/reference_model.rs`).
pub const REFERENCE_MODEL_DOWNLOAD_URL: &str =
    "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf";

pub const REFERENCE_TOKEN_ID: u32 = 2;

/// First 8 hidden dims of `token_embd` for [`REFERENCE_TOKEN_ID`] (gguf-py reference).
pub const REF_EMB_TOKEN2_HEAD: [f32; 8] = [
    -0.00165278,
    0.00089693,
    0.00016844,
    0.00016844,
    0.00016844,
    -0.00274551,
    0.00162542,
    0.00016844,
];

pub const REF_EMB_TOKEN2_IDX1024: f32 = -0.00292897;
pub const REF_EMB_TOKEN2_IDX4095: f32 = -0.00003505;

pub fn reference_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(REFERENCE_MODEL_REL_PATH)
}

/// [`Tokenizer::load_from_file`] next to the reference GGUF. Must match [`REFERENCE_MODEL_*`]:
/// **`mistralai/Mistral-7B-v0.1`** `tokenizer.model`.
pub const TOKENIZER_REL_PATH: &str = "model/mistral-7b-v0.1/tokenizer.model";

/// Gemma 4 E2B IT (Q8 example); tokenizer is Hugging Face **`tokenizer.json`** from `google/gemma-4-E2B-it`.
pub const GEMMA4_E2B_IT_DIR: &str = "model/gemma-4-e2b-it";
pub const GEMMA4_E2B_Q8_GGUF_REL_PATH: &str = "model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf";
pub const GEMMA4_E2B_TOKENIZER_REL_PATH: &str = "model/gemma-4-e2b-it/tokenizer.json";

/// Official tokenizer file for Mistral-7B-v0.1 (use to refresh `tokenizer.model`).
pub const REFERENCE_TOKENIZER_DOWNLOAD_URL: &str =
    "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model";

pub fn tokenizer_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(TOKENIZER_REL_PATH)
}

pub fn gemma4_e2b_q8_gguf_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(GEMMA4_E2B_Q8_GGUF_REL_PATH)
}

pub fn gemma4_e2b_tokenizer_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(GEMMA4_E2B_TOKENIZER_REL_PATH)
}

#[inline]
pub fn assert_close(a: f32, b: f32, eps: f32) {
    let d = (a - b).abs();
    assert!(d <= eps, "expected |{a} - {b}| <= {eps}, got diff {d}");
}
