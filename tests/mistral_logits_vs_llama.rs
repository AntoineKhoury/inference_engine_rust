//! Compare Rust **Mistral** prefill logits (last position) to `tools/llama_logits_ref`.
//!
//! ```text
//! ./tools/build_llama_logits_ref.sh
//! MISTRAL_LOGITS_PROMPT='What is the capital of France?' \
//! MISTRAL_LOGITS_TOP_K=10 \
//!   cargo test --test mistral_logits_vs_llama mistral_prefill_logits_match_llama --release -- --ignored --nocapture
//! ```

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{
    final_logits_last_token, prefill_forward, prefill_from_tokens,
};
use inference_engine_rust::tokenizer::Tokenizer;

use common::llama_logits_helpers::{
    argmax_f32, llama_logits_ref_binary, logits_diff_stats, read_reference_logits,
};
use common::{reference_model_path, tokenizer_model_path, REFERENCE_MODEL_REL_PATH};

fn prompt_for_test() -> String {
    std::env::var("MISTRAL_LOGITS_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France?".to_string())
}

fn optional_token_ids_from_env() -> Option<Vec<u32>> {
    let s = std::env::var("MISTRAL_LOGITS_TOKEN_IDS").ok()?;
    let mut out = Vec::new();
    for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
        out.push(part.parse().ok()?);
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let keep = k.min(idx.len());
    if keep == 0 {
        return Vec::new();
    }
    idx.select_nth_unstable_by(keep - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(keep);
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

fn decode_one(tokenizer: &Tokenizer, token_id: usize) -> String {
    tokenizer
        .decode_piece_ids(&[token_id as u32])
        .unwrap_or_else(|_| "<decode-error>".to_string())
        .replace('\n', "\\n")
}

fn print_top_k(label: &str, logits: &[f32], k: usize, tokenizer: &Tokenizer) {
    if k == 0 {
        return;
    }
    eprintln!("{label} top-{k}:");
    for (rank, id) in top_k_indices(logits, k).into_iter().enumerate() {
        eprintln!(
            "  #{:02} id={:<6} logit={:>10.6} token={:?}",
            rank + 1,
            id,
            logits[id],
            decode_one(tokenizer, id)
        );
    }
}

#[test]
#[ignore = "requires Mistral GGUF + tokenizer.model + tools/llama_logits_ref; slow"]
fn mistral_prefill_logits_match_llama() {
    let model_path = reference_model_path();
    assert!(
        model_path.is_file(),
        "missing GGUF at {}",
        model_path.display()
    );
    let tok_path = tokenizer_model_path();
    assert!(
        tok_path.is_file(),
        "missing tokenizer at {}",
        tok_path.display()
    );
    let ref_bin = llama_logits_ref_binary();
    assert!(
        ref_bin.is_file(),
        "missing {} - run ./tools/build_llama_logits_ref.sh",
        ref_bin.display()
    );

    let path_str = REFERENCE_MODEL_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let (prompt_ids, prompt_label) = if let Some(ids) = optional_token_ids_from_env() {
        (ids, "MISTRAL_LOGITS_TOKEN_IDS".to_string())
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        let prompt = prompt_for_test();
        let ids = tokenizer
            .encode_with_prompt_config(&prompt, &tok_prompt)
            .expect("encode");
        (ids, format!("prompt {prompt:?}"))
    };

    eprintln!("source: {prompt_label}");
    eprintln!("prompt token ids: {prompt_ids:?}");

    let ref_logits =
        read_reference_logits(&ref_bin, &model_path, &prompt_ids).expect("llama ref logits");
    let n_vocab = ref_logits.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    eprintln!(
        "ModelConfig: family={:?} unpack_llama_gguf_qk={} final_logit_softcapping={:?}",
        config.family, config.unpack_llama_gguf_qk, config.final_logit_softcapping
    );
    assert_eq!(
        config.vocab_size, n_vocab,
        "vocab size mismatch Rust GGUF vs llama ref"
    );

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");

    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let mut kv_caches = kv_caches_for_config(&config);

    let state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches).expect("prefill");
    let rust_logits = final_logits_last_token(&state, &config, &weights).expect("rust logits");
    assert_eq!(rust_logits.len(), n_vocab);

    let (max_abs, rmse, am_mismatch) = logits_diff_stats(&ref_logits, &rust_logits);
    let ia = argmax_f32(&ref_logits).unwrap();
    let ib = argmax_f32(&rust_logits).unwrap();
    let tokenizer = Tokenizer::load_from_file(&tok_path).expect("diagnostic tokenizer");

    eprintln!(
        "llama.cpp argmax token id: {ia} token={:?}",
        decode_one(&tokenizer, ia)
    );
    eprintln!(
        "Rust        argmax token id: {ib} token={:?}",
        decode_one(&tokenizer, ib)
    );
    eprintln!("max |Delta logit|: {max_abs:.6}, RMSE: {rmse:.6}, argmax mismatch: {am_mismatch}");

    let top_k = env_usize("MISTRAL_LOGITS_TOP_K", 0);
    print_top_k("llama.cpp", &ref_logits, top_k, &tokenizer);
    print_top_k("Rust", &rust_logits, top_k, &tokenizer);

    let max_abs_tolerance: f32 = std::env::var("MISTRAL_LOGITS_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let rmse_tolerance: f32 = std::env::var("MISTRAL_LOGITS_RMSE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    assert!(
        max_abs <= max_abs_tolerance && rmse <= rmse_tolerance,
        "logits differ from llama.cpp beyond tolerance (max_abs {max_abs} > {max_abs_tolerance} or rmse {rmse} > {rmse_tolerance})"
    );
}
