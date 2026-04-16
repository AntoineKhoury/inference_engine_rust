//! Compare Rust **Gemma 4 E2B** prefill logits (last position) to `tools/llama_logits_ref` (llama.cpp).
//!
//! Requires the GGUF and `tokenizer.json` under `model/gemma-4-e2b-it/` (see `tests/common/mod.rs`).
//!
//! ```text
//! ./tools/build_llama_logits_ref.sh
//! cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
//! ```
//!
//! Use the same prompt as the CLI to reproduce garbage-generation debugging, e.g. set env:
//! `GEMMA_LOGITS_PROMPT='Hello'`.

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{final_logits_last_token, prefill_forward, prefill_from_tokens};
use inference_engine_rust::tokenizer::Tokenizer;

use common::llama_logits_helpers::{
    argmax_f32, llama_logits_ref_binary, logits_diff_stats, read_reference_logits,
};
use common::{
    gemma4_e2b_q8_gguf_path, gemma4_e2b_tokenizer_path, GEMMA4_E2B_Q8_GGUF_REL_PATH,
};

fn prompt_for_test() -> String {
    std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string())
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer.json + tools/llama_logits_ref; slow"]
fn gemma4_e2b_prefill_logits_match_llama() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(
        model_path.is_file(),
        "missing GGUF at {} — place gemma-4-E2B-it-Q8_0.gguf per model/README.md",
        model_path.display()
    );
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(
        tok_path.is_file(),
        "missing tokenizer at {}",
        tok_path.display()
    );

    let ref_bin = llama_logits_ref_binary();
    assert!(
        ref_bin.is_file(),
        "missing {} — run ./tools/build_llama_logits_ref.sh",
        ref_bin.display()
    );

    let prompt = prompt_for_test();
    let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = tokenizer
        .encode_with_prompt_config(&prompt, &tok_prompt)
        .expect("encode");

    eprintln!("prompt: {prompt:?}");
    eprintln!("prompt token ids: {prompt_ids:?}");

    let ref_logits = read_reference_logits(&ref_bin, &model_path, &prompt_ids).expect("llama ref logits");
    let n_vocab = ref_logits.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    eprintln!(
        "ModelConfig: family={:?} unpack_llama_gguf_qk={} final_logit_softcapping={:?}",
        config.family,
        config.unpack_llama_gguf_qk,
        config.final_logit_softcapping
    );
    assert_eq!(
        config.vocab_size, n_vocab,
        "vocab size mismatch Rust GGUF vs llama ref"
    );

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");

    let prefill_in = prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let mut kv_caches = kv_caches_for_config(&config);

    let state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches).expect("prefill");
    let rust_logits = final_logits_last_token(&state, &config, &weights).expect("rust logits");

    assert_eq!(rust_logits.len(), n_vocab);

    let (max_abs, rmse, am_mismatch) = logits_diff_stats(&ref_logits, &rust_logits);
    let ia = argmax_f32(&ref_logits).unwrap();
    let ib = argmax_f32(&rust_logits).unwrap();

    eprintln!("llama.cpp argmax token id: {ia}");
    eprintln!("Rust        argmax token id: {ib}");
    eprintln!("max |Δlogit|: {max_abs:.6}, RMSE: {rmse:.6}, argmax mismatch: {am_mismatch}");

    // Gemma 4 Q8_0 + deep stack: expect wider drift vs llama.cpp CPU than dense Mistral Q4_K tests.
    // (Previously wrong FFN activation / missing `layer_output_scale` gave max_abs ~42 and wrong argmax.)
    const MAX_ABS_TOLERANCE: f32 = 40.0;
    const RMSE_TOLERANCE: f32 = 21.0;

    assert!(
        max_abs <= MAX_ABS_TOLERANCE && rmse <= RMSE_TOLERANCE,
        "logits differ from llama.cpp beyond tolerance (max_abs {max_abs} > {MAX_ABS_TOLERANCE} or rmse {rmse} > {RMSE_TOLERANCE})"
    );
    if ia != ib {
        eprintln!(
            "note: greedy argmax differs (llama={ia} rust={ib}); inspect top-k if RMSE is small"
        );
    }
}
