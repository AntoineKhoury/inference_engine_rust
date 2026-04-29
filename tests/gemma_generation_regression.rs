//! Gemma-only generation regressions that do not require llama.cpp.

mod common;

use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::ModelConfig;
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::engine::embed::prefill_from_tokens;
use inference_engine_rust::engine::runtime::{final_logits_last_token, prefill_forward};

use common::{GEMMA4_E2B_Q8_GGUF_REL_PATH, gemma4_e2b_q8_gguf_path};

fn argmax_f32(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
}

#[test]
#[ignore = "requires Gemma GGUF; slow (loads full model)"]
fn gemma4_e2b_france_continuation_picks_capital() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(
        model_path.is_file(),
        "missing GGUF at {} - place gemma-4-E2B-it-Q8_0.gguf per model/README.md",
        model_path.display()
    );

    // Regression for the GeGLU bug: before fixing tanh-GELU's coefficient,
    // this continuation picked a punctuation/control-style token instead.
    let prompt_ids = [
        2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 818,
    ];
    let expected_next = 5279usize; // " capital"

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");

    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let mut kv_caches = kv_caches_for_config(&config);
    let state =
        prefill_forward(&prefill_in, &config, &weights, &mut kv_caches).expect("prefill forward");
    let logits = final_logits_last_token(&state, &config, &weights).expect("final logits");
    let actual = argmax_f32(&logits).expect("argmax");

    assert_eq!(actual, expected_next);
}
