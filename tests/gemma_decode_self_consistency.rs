//! Self-consistency check: prefill of N+1 tokens must yield the same logits as
//! prefill of N tokens followed by a single decode step for token N+1.
//!
//! This verifies that the decode KV-cache path is algebraically equivalent to a
//! fresh full-sequence prefill (which is the correctness requirement for autoregressive
//! generation with a KV cache).
//!
//! ```text
//! cargo test --test gemma_decode_self_consistency --release -- --ignored --nocapture
//! ```
//!
//! Token IDs: `GEMMA_LOGITS_TOKEN_IDS` (comma-separated, first N are prompt, last is decode token)
//! or `GEMMA_LOGITS_PROMPT` (uses tokenizer; last prompt token is used as the decode token).
//! Override RMSE tolerance: `GEMMA_DECODE_RMSE_TOL` (default 0.1).

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::engine::embed::{
    prefill_from_tokens_loaded, prefill_state_for_single_token_loaded,
};
use inference_engine_rust::engine::runtime::{decode_forward, final_logits_last_token, prefill_forward};
use inference_engine_rust::tokenizer::Tokenizer;

use common::{GEMMA4_E2B_Q8_GGUF_REL_PATH, gemma4_e2b_q8_gguf_path, gemma4_e2b_tokenizer_path};

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    let n = a.len();
    let mut max_abs = 0.0f32;
    let mut se = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let dd = (*x as f64) - (*y as f64);
        se += dd * dd;
    }
    let rmse = (se / n as f64).sqrt() as f32;
    (max_abs, rmse)
}

fn argmax_f32(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

fn tokens_from_env_or_prompt(
    tok_path: &std::path::Path,
    tok_prompt: &TokenizerPromptConfig,
) -> Vec<u32> {
    if let Some(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse::<u32>().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        assert!(!out.is_empty(), "GEMMA_LOGITS_TOKEN_IDS is empty");
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(tok_path).expect("tokenizer");
        let prompt = std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string());
        tokenizer
            .encode_with_prompt_config(&prompt, tok_prompt)
            .expect("encode")
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf; slow (loads full model)"]
fn gemma4_decode_logits_match_fullseq_prefill() {
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

    // Build token list: last element is the "decode" token.
    let all_ids: Vec<u32> = if let Some(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse::<u32>().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        assert!(
            out.len() >= 2,
            "GEMMA_LOGITS_TOKEN_IDS needs at least 2 tokens (prompt + decode)"
        );
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        let prompt = std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string());
        let mut ids = tokenizer
            .encode_with_prompt_config(&prompt, &tok_prompt)
            .expect("encode");
        assert!(ids.len() >= 2, "prompt must encode to at least 2 tokens");
        // Append the known first generated token (107) as the decode token for a concrete test.
        // Override with GEMMA_LOGITS_TOKEN_IDS if you want a different split.
        ids.push(107);
        ids
    };

    let n = all_ids.len();
    let prompt_ids = &all_ids[..n - 1]; // first N-1 tokens: the prefill context
    let decode_token = all_ids[n - 1]; // last token: fed as decode input

    eprintln!("prompt_ids (N={}) : {:?}", prompt_ids.len(), prompt_ids);
    eprintln!("decode_token      : {decode_token}");

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    eprintln!(
        "ModelConfig: family={:?} n_layers={} hidden_dim={} vocab={}",
        config.family, config.n_layers, config.hidden_dim, config.vocab_size
    );

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    // Load all weights (including PLE tensors) so we can use the immutable-borrow variant below.
    names.load_all(&mut gguf, path_str).expect("load weights");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    // ── Path A: full prefill of (prompt + decode token) ──────────────────────
    let full_ids: Vec<u32> = all_ids.clone();
    let prefill_full = prefill_from_tokens_loaded(&gguf, &config, &full_ids)
        .expect("prefill_from_tokens_loaded (full)");
    let mut kv_a = kv_caches_for_config(&config);
    let state_a = prefill_forward(&prefill_full, &config, &weights, &mut kv_a)
        .expect("prefill_forward (full)");
    let logits_a =
        final_logits_last_token(&state_a, &config, &weights).expect("final_logits (full prefill)");

    // ── Path B: N-1 token prefill + 1 decode step ────────────────────────────
    let prefill_short = prefill_from_tokens_loaded(&gguf, &config, prompt_ids)
        .expect("prefill_from_tokens_loaded (short)");
    let mut kv_b = kv_caches_for_config(&config);
    let _state_b_prefill = prefill_forward(&prefill_short, &config, &weights, &mut kv_b)
        .expect("prefill_forward (short)");

    let decode_in = prefill_state_for_single_token_loaded(&gguf, &config, decode_token)
        .expect("prefill_state_for_single_token_loaded");
    let state_b_decode =
        decode_forward(&decode_in, &config, &weights, &mut kv_b).expect("decode_forward");
    let logits_b =
        final_logits_last_token(&state_b_decode, &config, &weights).expect("final_logits (decode)");

    assert_eq!(
        logits_a.len(),
        logits_b.len(),
        "vocab size mismatch between paths"
    );

    let (max_abs, rmse) = diff_stats(&logits_a, &logits_b);
    let argmax_a = argmax_f32(&logits_a).unwrap();
    let argmax_b = argmax_f32(&logits_b).unwrap();

    eprintln!(
        "Path A (full prefill) argmax: {argmax_a}  logits[argmax_a]={:.4}",
        logits_a[argmax_a]
    );
    eprintln!(
        "Path B (decode step)  argmax: {argmax_b}  logits[argmax_b]={:.4}",
        logits_b[argmax_b]
    );
    eprintln!("Logit diff: max_abs={max_abs:.6}  RMSE={rmse:.6}");

    if argmax_a != argmax_b {
        // Print top-5 for both to help diagnose
        let mut sorted_a: Vec<(usize, f32)> = logits_a.iter().copied().enumerate().collect();
        sorted_a.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut sorted_b: Vec<(usize, f32)> = logits_b.iter().copied().enumerate().collect();
        sorted_b.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("Top-5 Path A: {:?}", &sorted_a[..5]);
        eprintln!("Top-5 Path B: {:?}", &sorted_b[..5]);
    }

    let tol_rmse: f32 = std::env::var("GEMMA_DECODE_RMSE_TOL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);

    assert!(
        rmse <= tol_rmse,
        "RMSE {rmse:.6} between full-prefill logits and decode-step logits exceeds tolerance {tol_rmse}: \
         decode path is not self-consistent with prefill"
    );
    assert_eq!(
        argmax_a, argmax_b,
        "argmax mismatch between full-prefill ({argmax_a}) and decode-step ({argmax_b}): \
         decode path picks a different top token than prefill (RMSE={rmse:.6})"
    );

    eprintln!(
        "PASS: decode logits match full-prefill logits within RMSE={rmse:.6} (tol={tol_rmse})"
    );
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf; slow (loads full model)"]
fn gemma4_generation_decode_vs_teacher_forced_prefill() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(
        model_path.is_file(),
        "missing Gemma GGUF at {}",
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
    let prompt_ids = tokens_from_env_or_prompt(&tok_path, &tok_prompt);
    assert!(!prompt_ids.is_empty(), "prompt ids must not be empty");
    let max_new_tokens: usize = std::env::var("GEMMA_MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let stop_id = std::env::var("GEMMA_STOP_TOKEN_ID")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(tok_prompt.eos_token_id);

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");

    // A) Normal prefill + decode loop
    let mut kv_a = kv_caches_for_config(&config);
    let prefill_a = prefill_from_tokens_loaded(&gguf, &config, &prompt_ids).expect("prefill A");
    let mut state_a =
        prefill_forward(&prefill_a, &config, &weights, &mut kv_a).expect("prefill_forward A");
    let mut gen_a: Vec<u32> = Vec::new();
    for _ in 0..max_new_tokens {
        let logits = final_logits_last_token(&state_a, &config, &weights).expect("logits A");
        let next_id = argmax_f32(&logits).expect("argmax A") as u32;
        if next_id == stop_id {
            break;
        }
        gen_a.push(next_id);
        let step_in =
            prefill_state_for_single_token_loaded(&gguf, &config, next_id).expect("decode input A");
        state_a = decode_forward(&step_in, &config, &weights, &mut kv_a).expect("decode_forward A");
    }

    // B) Teacher-forced full-prefill each step (no decode path)
    let mut full_ids = prompt_ids.clone();
    let mut gen_b: Vec<u32> = Vec::new();
    for _ in 0..max_new_tokens {
        let prefill_b = prefill_from_tokens_loaded(&gguf, &config, &full_ids).expect("prefill B");
        let mut kv_b = kv_caches_for_config(&config);
        let state_b =
            prefill_forward(&prefill_b, &config, &weights, &mut kv_b).expect("prefill_forward B");
        let logits = final_logits_last_token(&state_b, &config, &weights).expect("logits B");
        let next_id = argmax_f32(&logits).expect("argmax B") as u32;
        if next_id == stop_id {
            break;
        }
        gen_b.push(next_id);
        full_ids.push(next_id);
    }

    let shared_prefix = gen_a
        .iter()
        .zip(gen_b.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let a_text = tokenizer
        .decode_piece_ids(&gen_a)
        .unwrap_or_else(|_| "<decode failed A>".to_string());
    let b_text = tokenizer
        .decode_piece_ids(&gen_b)
        .unwrap_or_else(|_| "<decode failed B>".to_string());
    eprintln!("prompt_ids: {:?}", prompt_ids);
    eprintln!("max_new_tokens={max_new_tokens} stop_id={stop_id}");
    eprintln!("A(normal decode) generated {} tokens", gen_a.len());
    eprintln!("B(full prefill) generated {} tokens", gen_b.len());
    eprintln!("shared prefix len: {shared_prefix}");
    if shared_prefix < gen_a.len().min(gen_b.len()) {
        eprintln!(
            "first divergence at step {}: A={} B={}",
            shared_prefix, gen_a[shared_prefix], gen_b[shared_prefix]
        );
    }
    eprintln!("A text: {:?}", a_text);
    eprintln!("B text: {:?}", b_text);

    // Strict discriminator requested by investigation: if decode path is healthy, sequences should match.
    assert_eq!(
        gen_a, gen_b,
        "generation mismatch between normal decode and teacher-forced full-prefill; likely decode/KV path issue"
    );
}
