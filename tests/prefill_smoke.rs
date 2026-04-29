//! End-to-end prefill smoke test (loads real GGUF weights; slow on CPU).

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::ModelConfig;
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};

use inference_engine_rust::engine::embed::{
    prefill_from_tokens, prefill_state_for_single_token_loaded,
};
use inference_engine_rust::engine::runtime::{
    decode_forward, final_logits_last_token, prefill_forward,
};
use inference_engine_rust::layers::embeddings::lookup_embeddings;

const MODEL_PATH: &str = common::REFERENCE_MODEL_REL_PATH;

#[test]
fn embedding_lookup_matches_whether_embd_loaded_alone_or_with_all_weights() {
    let mut g1 = read_file(MODEL_PATH).expect("read gguf");
    g1.load_single_tensor(MODEL_PATH, "token_embd.weight")
        .expect("load embd only");
    let e1 = lookup_embeddings(&mut g1, MODEL_PATH, &[1u32]).expect("lookup1");

    let mut g2 = read_file(MODEL_PATH).expect("read gguf");
    let config = ModelConfig::from_gguf(&g2).expect("config");
    let names = ModelWeightNames::resolve(&g2, &config).expect("resolve names");
    names.load_all(&mut g2, MODEL_PATH).expect("load all");
    let e2 = lookup_embeddings(&mut g2, MODEL_PATH, &[1u32]).expect("lookup2");

    assert_eq!(e1[0].len(), e2[0].len());
    let max_delta = e1[0]
        .iter()
        .zip(e2[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_delta < 1e-4, "lookup differs: max_delta={max_delta}");
}

#[test]
fn token_embd_buffer_matches_single_tensor_load() {
    let mut g1 = read_file(MODEL_PATH).expect("read gguf");
    g1.load_single_tensor(MODEL_PATH, "token_embd.weight")
        .expect("load embd only");
    let b1 = g1
        .get_tensor("token_embd.weight")
        .expect("embd")
        .buffer()
        .to_vec();

    let mut g2 = read_file(MODEL_PATH).expect("read gguf");
    let config = ModelConfig::from_gguf(&g2).expect("config");
    let names = ModelWeightNames::resolve(&g2, &config).expect("resolve names");
    names.load_all(&mut g2, MODEL_PATH).expect("load all");
    let b2 = g2
        .get_tensor("token_embd.weight")
        .expect("embd")
        .buffer()
        .to_vec();

    assert_eq!(
        b1.len(),
        b2.len(),
        "token_embd byte length differs between load paths"
    );
    let mismatches = b1.iter().zip(b2.iter()).filter(|(a, b)| a != b).count();
    assert_eq!(
        mismatches, 0,
        "token_embd buffer differs in {mismatches} bytes between single and load_all"
    );
}

/// Full forward on real weights is slow (large GGUF + many layers). Run explicitly:  
/// `cargo test --test prefill_smoke prefill_one_token_end_to_end --release -- --ignored --nocapture`
#[test]
#[ignore]
fn prefill_one_token_end_to_end() {
    let mut gguf = read_file(MODEL_PATH).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve names");
    names
        .load_all(&mut gguf, MODEL_PATH)
        .expect("load weight tensors");

    let token_ids = [2u32];
    let input = prefill_from_tokens(&mut gguf, MODEL_PATH, &config, &token_ids).expect("embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("model weights");

    let mut kv_caches = kv_caches_for_config(&config);

    let out = prefill_forward(&input, &config, &weights, &mut kv_caches).expect("prefill forward");
    let logits = final_logits_last_token(&out, &config, &weights).expect("logits");

    assert_eq!(logits.len(), config.vocab_size);
}

/// Two-token prefill vs one-token prefill + decode for the second token should match (KV + RoPE).
#[test]
#[ignore]
fn prefill_two_tokens_matches_prefill_one_then_decode() {
    let mut gguf = read_file(MODEL_PATH).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve names");
    names
        .load_all(&mut gguf, MODEL_PATH)
        .expect("load weight tensors");

    let t0: u32 = 1;
    let t1: u32 = 2;

    let logits_a = {
        let mut kv_a = kv_caches_for_config(&config);
        let input_a =
            prefill_from_tokens(&mut gguf, MODEL_PATH, &config, &[t0, t1]).expect("embed2");
        let weights = ModelWeights::from_loaded(&gguf, &names).expect("model weights");
        let out_a = prefill_forward(&input_a, &config, &weights, &mut kv_a).expect("prefill2");
        final_logits_last_token(&out_a, &config, &weights).expect("logits_a")
    };

    let logits_b = {
        let mut kv_b = kv_caches_for_config(&config);
        let input_b = prefill_from_tokens(&mut gguf, MODEL_PATH, &config, &[t0]).expect("embed1");
        let decode_in =
            prefill_state_for_single_token_loaded(&gguf, &config, t1).expect("decode state");
        let weights = ModelWeights::from_loaded(&gguf, &names).expect("model weights");
        let _out_b = prefill_forward(&input_b, &config, &weights, &mut kv_b).expect("prefill1");
        let out_dec = decode_forward(&decode_in, &config, &weights, &mut kv_b).expect("decode");
        final_logits_last_token(&out_dec, &config, &weights).expect("logits_b")
    };

    assert_eq!(logits_a.len(), logits_b.len());
    let eps: f32 = 1e-3;
    let max_delta = logits_a
        .iter()
        .zip(logits_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta < eps,
        "prefill [t0,t1] vs prefill [t0] + decode t1: max |Δ| = {max_delta} (eps {eps})"
    );
}
