//! Fast checks against `model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf` (metadata only; no tensor blob read).

mod common;

use inference_engine_rust::model_config::{ModelConfig, ModelFamily};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_loader::gguf_types::Data;
use inference_engine_rust::tokenizer::Tokenizer;

use common::{GEMMA4_E2B_Q8_GGUF_REL_PATH, GEMMA4_E2B_TOKENIZER_REL_PATH};

/// Runs when `model/gemma-4-e2b-it/tokenizer.json` is present (from `google/gemma-4-E2B-it`).
#[test]
fn gemma4_hf_tokenizer_loads_when_present() {
    let p = std::path::Path::new(GEMMA4_E2B_TOKENIZER_REL_PATH);
    if !p.is_file() {
        return;
    }
    let t = Tokenizer::load_from_file(p).expect("Hugging Face tokenizer.json should load");
    assert!(
        t.vocab_size() > 10_000,
        "unexpected vocab size {}",
        t.vocab_size()
    );
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf (see model/README.md)"]
fn gemma4_q8_metadata_loads() {
    let gguf = read_file(GEMMA4_E2B_Q8_GGUF_REL_PATH).expect("read gguf header/metadata");
    let config = ModelConfig::from_gguf(&gguf).expect("model config from gemma gguf");
    assert_eq!(config.family, ModelFamily::Gemma4);
    assert_eq!(config.layer_attention.len(), config.n_layers);
    match gguf.get_metadata("gemma4.final_logit_softcapping") {
        Some(Data::Float32(v)) if *v > 0.0 => assert_eq!(config.final_logit_softcapping, Some(*v)),
        Some(Data::Float64(v)) if *v > 0.0 => {
            assert_eq!(config.final_logit_softcapping, Some(*v as f32));
        }
        _ => assert!(config.final_logit_softcapping.is_none()),
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_tensor_shapes_debug() {
    let gguf = read_file(GEMMA4_E2B_Q8_GGUF_REL_PATH).expect("read gguf");
    for name in [
        "token_embd.weight",
        "per_layer_token_embd.weight",
        "per_layer_model_proj.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.inp_gate.weight",
        "blk.0.proj.weight",
    ] {
        let t = gguf.tensors_metadata().iter().find(|x| x.name == name);
        eprintln!("{name}: {:?}", t.map(|x| x.dimensions.clone()));
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_attention_metadata() {
    let gguf = read_file(GEMMA4_E2B_Q8_GGUF_REL_PATH).expect("read gguf");
    for key in [
        "gemma4.attention.head_count",
        "gemma4.attention.head_count_kv",
        "gemma4.attention.key_length",
        "gemma4.attention.key_length_swa",
        "gemma4.attention.value_length",
        "gemma4.attention.value_length_swa",
        "gemma4.embedding_length",
    ] {
        if let Some(d) = gguf.get_metadata(key) {
            eprintln!("{key}: {d:?}");
        }
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_attn_k_dims_sample_layers() {
    let gguf = read_file(GEMMA4_E2B_Q8_GGUF_REL_PATH).expect("read gguf");
    for idx in [0, 34] {
        for stem in [
            "attn_k.weight",
            "attn_q.weight",
            "attn_output.weight",
            "attn_v.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
        ] {
            let name = format!("blk.{idx}.{stem}");
            let t = gguf.tensors_metadata().iter().find(|x| x.name == name);
            eprintln!("{name}: {:?}", t.map(|x| x.dimensions.clone()));
        }
        for stem in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            let name = format!("blk.{idx}.{stem}");
            let t = gguf.tensors_metadata().iter().find(|x| x.name == name);
            eprintln!("{name}: {:?}", t.map(|x| x.dimensions.clone()));
        }
    }
}
