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
//!
//! For raw id lists (e.g. **single-token** prefill): `GEMMA_LOGITS_TOKEN_IDS=2` or `2,9259`
//! (comma-separated; skips tokenizer). Single-token checks isolate the stack without cross-position attention.

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::layers::prefill_block::prefill_layer_block;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{
    final_logits_last_token, prefill_forward, prefill_from_tokens, PrefillState,
};
use inference_engine_rust::tokenizer::Tokenizer;

use common::llama_logits_helpers::{
    argmax_f32, llama_logits_ref_binary, logits_diff_stats, read_reference_logits,
};
use common::{gemma4_e2b_q8_gguf_path, gemma4_e2b_tokenizer_path, GEMMA4_E2B_Q8_GGUF_REL_PATH};

fn prompt_for_test() -> String {
    std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string())
}

/// When set (e.g. `2` or `2,9259`), skip tokenizer and use these ids for parity with llama_logits_ref.
fn optional_token_ids_from_env() -> Option<Vec<u32>> {
    let s = std::env::var("GEMMA_LOGITS_TOKEN_IDS").ok()?;
    let mut out = Vec::new();
    for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
        out.push(part.parse().ok()?);
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

fn token_ids_from_env(name: &str) -> Option<Vec<usize>> {
    let s = std::env::var(name).ok()?;
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

fn print_watch_tokens(label: &str, logits: &[f32], watch_ids: &[usize], tokenizer: &Tokenizer) {
    if watch_ids.is_empty() {
        return;
    }
    eprintln!("{label} watched tokens:");
    for &id in watch_ids {
        if id >= logits.len() {
            eprintln!("  id={id} out of range for vocab {}", logits.len());
            continue;
        }
        eprintln!(
            "  id={:<6} logit={:>10.6} token={:?}",
            id,
            logits[id],
            decode_one(tokenizer, id)
        );
    }
    if watch_ids.len() == 2 && watch_ids.iter().all(|&id| id < logits.len()) {
        let a = watch_ids[0];
        let b = watch_ids[1];
        eprintln!("  margin id{a}-id{b} = {:.6}", logits[a] - logits[b]);
    }
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

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let (prompt_ids, prompt_label) = if let Some(ids) = optional_token_ids_from_env() {
        (ids, "GEMMA_LOGITS_TOKEN_IDS".to_string())
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

    eprintln!("llama.cpp argmax token id: {ia}");
    eprintln!("Rust        argmax token id: {ib}");
    eprintln!("max |Δlogit|: {max_abs:.6}, RMSE: {rmse:.6}, argmax mismatch: {am_mismatch}");

    let diag_tokenizer = Tokenizer::load_from_file(&tok_path).expect("diagnostic tokenizer");
    let top_k = env_usize("GEMMA_LOGITS_TOP_K", 0);
    let watch_ids = token_ids_from_env("GEMMA_LOGITS_WATCH_TOKEN_IDS").unwrap_or_default();
    print_top_k("llama.cpp", &ref_logits, top_k, &diag_tokenizer);
    print_top_k("Rust", &rust_logits, top_k, &diag_tokenizer);
    print_watch_tokens("llama.cpp", &ref_logits, &watch_ids, &diag_tokenizer);
    print_watch_tokens("Rust", &rust_logits, &watch_ids, &diag_tokenizer);
    for &id in &watch_ids {
        if id < ref_logits.len() && id < rust_logits.len() {
            eprintln!(
                "  Δ token id={id}: rust-ref = {:.6}",
                rust_logits[id] - ref_logits[id]
            );
        }
    }

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

#[test]
#[ignore = "requires Gemma GGUF + tokenizer.json; diagnostic layer-by-layer margin trace"]
fn gemma4_e2b_rust_layer_margin_trace() {
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

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = if let Some(ids) = optional_token_ids_from_env() {
        ids
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        tokenizer
            .encode_with_prompt_config(&prompt_for_test(), &tok_prompt)
            .expect("encode")
    };

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");
    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let tokenizer = Tokenizer::load_from_file(&tok_path).expect("diagnostic tokenizer");

    let watch_ids =
        token_ids_from_env("GEMMA_LOGITS_WATCH_TOKEN_IDS").unwrap_or_else(|| vec![5279, 1018]);
    assert!(
        watch_ids.len() >= 2,
        "set GEMMA_LOGITS_WATCH_TOKEN_IDS to at least two token ids"
    );
    let a = watch_ids[0];
    let b = watch_ids[1];
    eprintln!("prompt token ids: {prompt_ids:?}");
    eprintln!(
        "tracking margin id{a}({:?}) - id{b}({:?})",
        decode_one(&tokenizer, a),
        decode_one(&tokenizer, b)
    );

    let mut kv_caches = kv_caches_for_config(&config);
    let mut state = PrefillState::from_flat_with_ple(
        prefill_in.hidden().to_vec(),
        prefill_in.seq_len(),
        prefill_in.hidden_dim(),
        prefill_in.per_layer_packed().to_vec(),
        prefill_in.ple_n_layers(),
        prefill_in.ple_dim(),
    )
    .expect("clone prefill state");

    print_layer_margin(
        "embedding",
        &state,
        &config,
        &weights,
        &watch_ids,
        &tokenizer,
    );
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        state = prefill_layer_block(&state, &config, layer_idx, layer_weights, &mut kv_caches)
            .expect("prefill_layer_block");
        print_layer_margin(
            &format!("layer {layer_idx:02}"),
            &state,
            &config,
            &weights,
            &watch_ids,
            &tokenizer,
        );
    }
}

fn print_layer_margin(
    label: &str,
    state: &PrefillState,
    config: &ModelConfig,
    weights: &ModelWeights,
    watch_ids: &[usize],
    tokenizer: &Tokenizer,
) {
    let logits = final_logits_last_token(state, config, weights).expect("partial logits");
    let argmax = argmax_f32(&logits).expect("argmax");
    let margin =
        if watch_ids.len() >= 2 && watch_ids[0] < logits.len() && watch_ids[1] < logits.len() {
            logits[watch_ids[0]] - logits[watch_ids[1]]
        } else {
            f32::NAN
        };
    eprintln!(
        "{label:>10}: argmax={:<6} token={:?} margin={:>10.6}",
        argmax,
        decode_one(tokenizer, argmax),
        margin
    );
    for &id in watch_ids {
        if id < logits.len() {
            eprintln!(
                "            id={:<6} logit={:>10.6} token={:?}",
                id,
                logits[id],
                decode_one(tokenizer, id)
            );
        }
    }
}
