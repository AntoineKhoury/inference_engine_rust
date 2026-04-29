//! Gemma 4 E2B: last-token hidden after `output_norm` vs `tools/llama_hidden_ref` (llama.cpp `result_norm`).
//!
//! ```text
//! cargo test --test gemma_hidden_vs_llama gemma4_e2b_last_hidden_vs_llama --release -- --ignored --nocapture
//! GEMMA_LOGITS_TOKEN_IDS=2 cargo test --test gemma_hidden_vs_llama gemma4_e2b_last_hidden_vs_llama --release -- --ignored --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::ops::rmsnorm::rmsnorm;
use inference_engine_rust::prefill::{prefill_forward, prefill_from_tokens};
use inference_engine_rust::tokenizer::Tokenizer;

use common::{
    gemma4_e2b_q8_gguf_path, gemma4_e2b_tokenizer_path, GEMMA4_E2B_Q8_GGUF_REL_PATH,
};

fn llama_hidden_ref_binary() -> PathBuf {
    if let Ok(p) = std::env::var("LLAMA_HIDDEN_REF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("llama_hidden_ref")
}

fn read_reference_hidden(
    bin: &Path,
    model: &Path,
    token_ids: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(bin);
    cmd.arg(model);
    for &t in token_ids {
        cmd.arg(t.to_string());
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "llama_hidden_ref failed: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let bytes = out.stdout;
    if bytes.len() < 4 {
        return Err("reference hidden output too short".into());
    }
    let n_embd = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected = 4 + n_embd * 4;
    if bytes.len() != expected {
        return Err(format!(
            "bad hidden ref size: got {} bytes, expected {} (n_embd={})",
            bytes.len(),
            expected,
            n_embd
        )
        .into());
    }
    let mut v = Vec::with_capacity(n_embd);
    for i in 0..n_embd {
        let off = 4 + i * 4;
        v.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    Ok(v)
}

fn hidden_diff_max_abs(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > m {
            m = d;
        }
    }
    m
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer.json + tools/llama_hidden_ref; slow"]
fn gemma4_e2b_last_hidden_vs_llama() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");

    let ref_bin = llama_hidden_ref_binary();
    assert!(
        ref_bin.is_file(),
        "missing {} — run ./tools/build_llama_logits_ref.sh",
        ref_bin.display()
    );

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = if let Ok(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS") {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        assert!(!out.is_empty(), "GEMMA_LOGITS_TOKEN_IDS empty");
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        let prompt = std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string());
        tokenizer
            .encode_with_prompt_config(&prompt, &tok_prompt)
            .expect("encode")
    };

    eprintln!("prompt ids: {prompt_ids:?}");

    let ref_hidden = read_reference_hidden(&ref_bin, &model_path, &prompt_ids).expect("llama hidden");

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    eprintln!(
        "unpack_llama_gguf_qk={} family={:?}",
        config.unpack_llama_gguf_qk, config.family
    );
    assert_eq!(
        ref_hidden.len(),
        config.hidden_dim,
        "hidden dim mismatch"
    );

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");

    let prefill_in = prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let mut kv_caches = kv_caches_for_config(&config);

    let state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches).expect("prefill");

    let seq_len = state.seq_len();
    let hd = state.hidden_dim();
    let last = &state.hidden()[(seq_len - 1) * hd..seq_len * hd];

    let norm_w = weights.output_norm.as_f32_slice().expect("output norm f32");
    let mut rust_normed = vec![0f32; hd];
    rmsnorm(last, norm_w, config.rms_norm_eps, &mut rust_normed).expect("rmsnorm");

    let max_abs = hidden_diff_max_abs(&ref_hidden, &rust_normed);
    eprintln!("last hidden (post output_norm): max |Δ| vs llama: {max_abs:.6}");

    // Q8 Gemma 4 hybrid attention: allow larger drift than Mistral; wrong FFN/scale gave ~249 here.
    const TOL: f32 = 56.0;
    assert!(
        max_abs <= TOL,
        "post-norm hidden differs from llama beyond tolerance (max_abs {max_abs} > {TOL})"
    );
}
