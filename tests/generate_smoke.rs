//! Greedy continuation after a text prompt: encode → prefill → decode loop → decode tokens to string.
//!
//! Requires the reference GGUF (~4GB) and `tokenizer.model` at the workspace root.
//!
//! ```text
//! cargo test --test generate_smoke greedy_generate_continuation_after_prompt --release -- --ignored --nocapture
//! ```
//!
//! Prints **prompt token ids**, **generated token ids**, and decoded text (`--nocapture` required).

mod common;

use inference_engine_rust::layers::attention::KVCache;
use inference_engine_rust::layers::embeddings::lookup_embeddings_loaded;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{
    decode_forward, final_logits_last_token, prefill_forward, prefill_from_tokens, PrefillState,
};
use inference_engine_rust::sampling::sample_greedy;
use inference_engine_rust::tokenizer::Tokenizer;

use common::{reference_model_path, tokenizer_model_path, REFERENCE_MODEL_REL_PATH};

const PROMPT: &str = "Rust will rule the";
const NEW_TOKENS: usize = 12;

fn logits_top2(logits: &[f32]) -> Option<(usize, f32, usize, f32)> {
    if logits.len() < 2 || logits.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut i1 = 0usize;
    let mut v1 = logits[0];
    let mut i2 = 1usize;
    let mut v2 = logits[1];
    if v2 > v1 {
        std::mem::swap(&mut i1, &mut i2);
        std::mem::swap(&mut v1, &mut v2);
    }
    for i in 2..logits.len() {
        let v = logits[i];
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    Some((i1, v1, i2, v2))
}

#[test]
#[ignore = "requires model/mistral-7b-v0.1.Q4_K_M.gguf + tokenizer.model; slow on CPU"]
fn greedy_generate_continuation_after_prompt() {
    let model_path = reference_model_path();
    assert!(
        model_path.is_file(),
        "missing GGUF at {} — place file or download (see tests/common/mod.rs)",
        model_path.display()
    );

    let tokenizer_path = tokenizer_model_path();
    assert!(
        tokenizer_path.is_file(),
        "missing tokenizer at {} (expected Mistral-style tokenizer.model at repo root)",
        tokenizer_path.display()
    );

    let mut tokenizer = Tokenizer::load_from_file(&tokenizer_path).expect("load tokenizer");

    let path_str = REFERENCE_MODEL_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tokenizer prompt config");
    let prompt_ids = tokenizer
        .encode_with_prompt_config(PROMPT, &tok_prompt)
        .expect("encode prompt");
    assert!(!prompt_ids.is_empty(), "prompt should tokenize to non-empty ids");
    eprintln!("tokenizer prompt config: {tok_prompt:?}");
    eprintln!("prompt token ids ({}): {:?}", prompt_ids.len(), prompt_ids);

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve names");
    names
        .load_all(&mut gguf, path_str)
        .expect("load weights");

    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("model weights");

    let mut kv_caches: Vec<KVCache> = (0..config.n_layers)
        .map(|_| {
            KVCache::new(
                config.context_length,
                config.n_kv_heads,
                config.head_dim,
            )
        })
        .collect();

    let mut state =
        prefill_forward(&prefill_in, &config, &weights, &mut kv_caches).expect("prefill forward");

    eprintln!(
        "config: rope_theta={}, hidden={}, vocab={}",
        config.rope_theta, config.hidden_dim, config.vocab_size
    );

    let mut generated = Vec::with_capacity(NEW_TOKENS);
    let mut prev_decoded_hidden: Option<Vec<f32>> = None;

    for step in 0..NEW_TOKENS {
        let logits = final_logits_last_token(&state, &config, &weights).expect("logits");
        let n_nan = logits.iter().filter(|x| x.is_nan()).count();
        let n_inf = logits
            .iter()
            .filter(|x| x.is_infinite())
            .count();
        if let Some((i1, v1, i2, v2)) = logits_top2(&logits) {
            eprintln!(
                "decode step {step}: logit top2 = ({i1}, {v1:.4}), ({i2}, {v2:.4}), nan={n_nan} inf={n_inf}"
            );
        } else {
            eprintln!("decode step {step}: logits_top2 unavailable (nan/inf/short)");
        }

        let next_id = sample_greedy(&logits).expect("greedy sample");
        generated.push(next_id);

        let rows = lookup_embeddings_loaded(&gguf, &[next_id]).expect("embedding row");
        let step_in = PrefillState::from_embeddings(rows, config.hidden_dim).expect("decode input");
        state = decode_forward(&step_in, &config, &weights, &mut kv_caches).expect("decode forward");

        let h = state.hidden();
        let h_sum: f32 = h.iter().sum();
        let h_l1_prev = prev_decoded_hidden.as_ref().map(|p| {
            h.iter()
                .zip(p.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        });
        eprintln!(
            "  after decode_forward: hidden len={}, sum={h_sum:.4}, L1 diff vs prev decode={h_l1_prev:?}",
            h.len()
        );
        if let Some(l1) = h_l1_prev {
            assert!(
                l1 > 1e-2,
                "decoded hidden state barely changed between steps (L1={l1}); transformer state may be stuck — check RoPE build (see ENGINEERING_LOG) and run `cargo clean`"
            );
        }
        prev_decoded_hidden = Some(h.to_vec());
    }

    let continuation = tokenizer
        .decode_piece_ids(&generated)
        .expect("decode generated token ids");

    eprintln!("prompt text: {PROMPT:?}");
    eprintln!("generated token ids ({NEW_TOKENS}): {generated:?}");
    if generated.len() >= 2 && generated.iter().all(|&id| id == generated[0]) {
        eprintln!(
            "note: all generated ids are the same ({}); greedy is stuck on one token id",
            generated[0]
        );
    }
    eprintln!("generated text: {continuation:?}");

    assert!(
        continuation.chars().any(|c| !c.is_whitespace()),
        "expected some non-whitespace in greedy continuation, got {continuation:?}"
    );
}
