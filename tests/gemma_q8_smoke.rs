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
        "blk.15.post_ffw_norm.weight",
        "blk.15.post_attention_norm.weight",
        "blk.15.ffn_norm.weight",
        "blk.15.attn_norm.weight",
    ] {
        let t = gguf
            .tensors_metadata()
            .iter()
            .find(|x| x.name == name);
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

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_kv_borrow_map_debug() {
    let gguf = read_file(GEMMA4_E2B_Q8_GGUF_REL_PATH).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("model config from gemma gguf");
    assert_eq!(config.family, ModelFamily::Gemma4);
    eprintln!("n_layers={}", config.n_layers);
    for i in 0..config.n_layers {
        let attn = config.layer_attention_for(i).expect("layer attn");
        let swa = attn.sliding_window.is_some();
        let borrow = config
            .gemma4_kv_borrow_from
            .get(i)
            .copied()
            .flatten()
            .map(|x| x.to_string())
            .unwrap_or_else(|| "-".to_string());
        eprintln!(
            "layer={i:02} swa={} head_dim={} kv_dim={} borrow_from={}",
            swa,
            config.layer_dims_for(i).expect("layer dims").head_dim,
            config.layer_dims_for(i).expect("layer dims").kv_dim,
            borrow
        );
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_post_norm_weight_stats() {
    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = inference_engine_rust::model_weights::ModelWeightNames::resolve(&gguf, &config)
        .expect("resolve names");
    names.load_all(&mut gguf, path_str).expect("load all");
    let weights = inference_engine_rust::model_weights::ModelWeights::from_loaded(&gguf, &names)
        .expect("weights");

    for &l in &[14usize, 15usize] {
        let w = weights.layers[l]
            .attn_post_norm
            .expect("attn_post_norm")
            .as_f32_slice()
            .expect("f32");
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in w {
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
        let mean = (sum / w.len() as f64) as f32;
        eprintln!(
            "layer {l} attn_post_norm.weight stats: len={} min={mn:.6} max={mx:.6} mean={mean:.6} first8={:?}",
            w.len(),
            &w[..8.min(w.len())]
        );
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_all_norm_weight_stats() {
    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = inference_engine_rust::model_weights::ModelWeightNames::resolve(&gguf, &config)
        .expect("resolve names");
    names.load_all(&mut gguf, path_str).expect("load all");
    let weights = inference_engine_rust::model_weights::ModelWeights::from_loaded(&gguf, &names)
        .expect("weights");

    fn stats(label: &str, w: &[f32]) {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for &v in w {
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
        let mean = (sum / w.len() as f64) as f32;
        eprintln!(
            "{label:>40}: len={:<5} min={mn:>10.6} max={mx:>10.6} mean={mean:>10.6}  first4={:?}",
            w.len(),
            &w[..4.min(w.len())]
        );
    }

    eprintln!("\n=== output_norm ===");
    stats("output_norm", weights.output_norm.as_f32_slice().expect("f32"));

    for &l in &[0usize, 1, 14, 15, 29] {
        if l >= config.n_layers { continue; }
        let lw = &weights.layers[l];
        let is_swa = config.layer_attention_for(l).expect("attn").sliding_window.is_some();
        eprintln!("\n=== layer {l} (swa={is_swa}) ===");
        stats(&format!("layer {l} attn_norm"), lw.attn_norm.as_f32_slice().expect("f32"));
        stats(&format!("layer {l} ffn_norm"), lw.ffn_norm.as_f32_slice().expect("f32"));
        if let Some(t) = lw.attn_post_norm {
            stats(&format!("layer {l} post_attn_norm"), t.as_f32_slice().expect("f32"));
        }
        if let Some(t) = lw.ffn_post_norm {
            stats(&format!("layer {l} post_ffn_norm"), t.as_f32_slice().expect("f32"));
        }
        if let Some(t) = lw.attn_q_norm {
            stats(&format!("layer {l} q_norm"), t.as_f32_slice().expect("f32"));
        }
        if let Some(t) = lw.attn_k_norm {
            stats(&format!("layer {l} k_norm"), t.as_f32_slice().expect("f32"));
        }
    }
}

#[test]
#[ignore = "requires model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf"]
fn gemma4_print_qk_norm_weight_stats() {
    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = inference_engine_rust::model_weights::ModelWeightNames::resolve(&gguf, &config)
        .expect("resolve names");
    names.load_all(&mut gguf, path_str).expect("load all");
    let weights = inference_engine_rust::model_weights::ModelWeights::from_loaded(&gguf, &names)
        .expect("weights");

    for &l in &[0usize, 14usize, 15usize, 34usize] {
        let q = weights.layers[l]
            .attn_q_norm
            .expect("attn_q_norm")
            .as_f32_slice()
            .expect("q f32");
        let k = weights.layers[l]
            .attn_k_norm
            .expect("attn_k_norm")
            .as_f32_slice()
            .expect("k f32");
        let mut q_min = f32::INFINITY;
        let mut q_max = f32::NEG_INFINITY;
        let mut q_sum = 0.0f64;
        let mut k_min = f32::INFINITY;
        let mut k_max = f32::NEG_INFINITY;
        let mut k_sum = 0.0f64;
        for &v in q {
            q_min = q_min.min(v);
            q_max = q_max.max(v);
            q_sum += v as f64;
        }
        for &v in k {
            k_min = k_min.min(v);
            k_max = k_max.max(v);
            k_sum += v as f64;
        }
        let q_mean = (q_sum / q.len() as f64) as f32;
        let k_mean = (k_sum / k.len() as f64) as f32;
        eprintln!(
            "layer {l} q_norm: len={} min={q_min:.6} max={q_max:.6} mean={q_mean:.6} first8={:?}",
            q.len(),
            &q[..8.min(q.len())]
        );
        eprintln!(
            "layer {l} k_norm: len={} min={k_min:.6} max={k_max:.6} mean={k_mean:.6} first8={:?}",
            k.len(),
            &k[..8.min(k.len())]
        );
    }
}
