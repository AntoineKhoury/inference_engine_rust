//! Compare a **single GGML node** from `tools/llama_tensor_dump_ref` to this engine’s hidden state.
//!
//! Use this to bisect vs llama.cpp after tracing the graph:
//!
//! ```text
//! LLAMA_TENSOR_DUMP_TRACE=1 ./tools/llama_tensor_dump_ref model.gguf 2 9259 2> /tmp/nodes.tsv
//! ```
//!
//! Pick `node_index` whose `ne0..ne3` match a rank-2 activation (`seq_len` × `hidden_dim`, either axis order).
//! Then:
//!
//! ```text
//! GEMMA_LLAMA_TENSOR_NODE=1234 GEMMA_RUST_NUM_LAYERS=0 \
//!   cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture
//! ```
//!
//! - **`GEMMA_RUST_NUM_LAYERS`**: number of transformer blocks to run (`0` = embeddings + PLE only, same as
//!   `config.n_layers` for full stack **before** `output_norm`).
//! - **`GEMMA_APPLY_OUTPUT_NORM=1`**: apply Rust `output_norm` RMSNorm to the last token (compare to a llama
//!   `result_norm`-like node).
//! - **`GEMMA_TENSOR_MAX_ABS`**: override tolerance (default `80.0`).
//! - Token ids: **`GEMMA_LOGITS_TOKEN_IDS`** or `GEMMA_LOGITS_PROMPT` like other Gemma parity tests.
//!
mod common;

use inference_engine_rust::core::tensor::TensorType;
use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::layers::attention::prefill_attention_core_layer;
use inference_engine_rust::layers::attention::prefill_attention_layer;
use inference_engine_rust::layers::attention::prefill_qk_after_rope_layer;
use inference_engine_rust::layers::attention::prefill_qk_normed_layer;
use inference_engine_rust::layers::attention::prefill_qk_raw_layer;
use inference_engine_rust::layers::prefill_block::{
    gemma4_prefill_layer_debug, prefill_attention_with_norm, prefill_ffn, prefill_ffn_debug,
    prefill_ffn_down_from_activated,
};
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::ops::quant::quant_k_handler::{
    dequantize_q8_0_block, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_SIZE,
};
use inference_engine_rust::ops::rmsnorm::{rmsnorm, rmsnorm_inplace_no_scale};
use inference_engine_rust::prefill::{prefill_forward_layers, prefill_from_tokens, PrefillState};
use inference_engine_rust::tokenizer::Tokenizer;

use common::llama_tensor_dump_helpers::{
    llama_tensor_dump_ref_binary, lmtd_last_token_f32, max_abs_diff, parse_lmtd_v2,
    run_llama_tensor_dump_node,
};
use common::{gemma4_e2b_q8_gguf_path, gemma4_e2b_tokenizer_path, GEMMA4_E2B_Q8_GGUF_REL_PATH};

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max_abs = 0.0f32;
    let mut se = 0.0f64;
    let n = a.len().min(b.len()).max(1);
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y).abs();
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

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn dim_ids_from_env(name: &str) -> Vec<usize> {
    let Some(s) = std::env::var(name).ok() else {
        return Vec::new();
    };
    s.split([',', ' '])
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse().ok())
        .collect()
}

fn print_watch_dims(
    label: &str,
    rust_vec: &[f32],
    llama_vec: &[f32],
    dims: &[usize],
    scale: Option<&[f32]>,
) {
    if dims.is_empty() {
        return;
    }
    eprintln!("{label} watched dims:");
    for &i in dims {
        if i >= rust_vec.len() || i >= llama_vec.len() {
            eprintln!(
                "  dim={i} out of range (rust_len={} llama_len={})",
                rust_vec.len(),
                llama_vec.len()
            );
            continue;
        }
        let scale_text = scale
            .and_then(|s| s.get(i))
            .map(|v| format!(" scale={v:.6}"))
            .unwrap_or_default();
        eprintln!(
            "  dim={i:<5} rust={:>11.6} llama={:>11.6} diff={:>11.6} abs={:>11.6}{}",
            rust_vec[i],
            llama_vec[i],
            rust_vec[i] - llama_vec[i],
            (rust_vec[i] - llama_vec[i]).abs(),
            scale_text
        );
    }
}

fn print_top_diffs(
    label: &str,
    rust_vec: &[f32],
    llama_vec: &[f32],
    top_n: usize,
    scale: Option<&[f32]>,
) {
    if top_n == 0 {
        return;
    }
    let mut idx: Vec<usize> = (0..rust_vec.len().min(llama_vec.len())).collect();
    idx.sort_by(|&a, &b| {
        let da = (rust_vec[a] - llama_vec[a]).abs();
        let db = (rust_vec[b] - llama_vec[b]).abs();
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    eprintln!("{label} top-{top_n} absolute diffs:");
    for &i in idx.iter().take(top_n) {
        let scale_text = scale
            .and_then(|s| s.get(i))
            .map(|v| format!(" scale={v:.6}"))
            .unwrap_or_default();
        eprintln!(
            "  dim={i:<5} rust={:>11.6} llama={:>11.6} diff={:>11.6} abs={:>11.6}{}",
            rust_vec[i],
            llama_vec[i],
            rust_vec[i] - llama_vec[i],
            (rust_vec[i] - llama_vec[i]).abs(),
            scale_text
        );
    }
}

fn matmul_dense_ggml_layout(
    input: &[f32],
    m: usize,
    k: usize,
    weight: &[f32],
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        let in_row = row * k;
        let out_row = row * n;
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += input[in_row + kk] * weight[kk + col * k];
            }
            out[out_row + col] = acc;
        }
    }
    out
}

fn dequantize_q8_0_tensor_to_f32(weight: &inference_engine_rust::core::tensor::Tensor) -> Vec<f32> {
    let dims = weight.dimensions();
    assert_eq!(dims.len(), 2, "expected 2D tensor");
    let k = dims[0];
    let n = dims[1];
    let total = k * n;
    let mut out = vec![0.0f32; total];
    let bytes = weight.buffer();
    let mut decoded = [0.0f32; Q8_0_BLOCK_ELEMENTS];
    for idx in 0..total {
        let block_idx = idx / Q8_0_BLOCK_ELEMENTS;
        let el = idx % Q8_0_BLOCK_ELEMENTS;
        let start = block_idx * Q8_0_BLOCK_SIZE;
        let block = &bytes[start..start + Q8_0_BLOCK_SIZE];
        if el == 0 {
            dequantize_q8_0_block(block, &mut decoded).expect("dequantize q8 block");
        }
        out[idx] = decoded[el];
    }
    out
}

fn lmtd_last_token_f32_any_seq(
    rec: &common::llama_tensor_dump_helpers::LmtdRecordV2,
    seq_len: usize,
    feature_len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // First try existing 1D/2D extractor.
    if let Ok(v) = lmtd_last_token_f32(rec, seq_len, feature_len) {
        return Ok(v);
    }
    // Fallback for 3D activations like [head_dim, n_heads, seq].
    let active: Vec<usize> = (0..4).filter(|&i| rec.ne[i] > 1).collect();
    if active.len() != 3 {
        return Err(format!(
            "lmtd_last_token_f32_any_seq: unsupported active dims {:?} ne {:?}",
            active, rec.ne
        )
        .into());
    }
    let seq_i64 = i64::try_from(seq_len).map_err(|_| "seq_len cast")?;
    let seq_axis = active
        .iter()
        .copied()
        .find(|&i| rec.ne[i] == seq_i64)
        .ok_or("no seq axis matches seq_len")?;
    let mut feat_axes = active.clone();
    feat_axes.retain(|&i| i != seq_axis);
    let feat_prod = usize::try_from(rec.ne[feat_axes[0]] * rec.ne[feat_axes[1]])
        .map_err(|_| "feature product cast")?;
    if feat_prod != feature_len {
        return Err(format!(
            "feature len mismatch for 3D tensor: prod {} vs expected {}",
            feat_prod, feature_len
        )
        .into());
    }
    if rec.nb[feat_axes[0]] != 4 {
        return Err("expected first feature axis stride to be 4 bytes".into());
    }
    let base = (seq_len as u64 - 1) * rec.nb[seq_axis];
    let mut out = vec![0f32; feature_len];
    for i in 0..feature_len {
        let off = base as usize + i * 4;
        if off + 4 > rec.data.len() {
            return Err("tensor read out of bounds".into());
        }
        out[i] = f32::from_le_bytes(rec.data[off..off + 4].try_into().unwrap());
    }
    Ok(out)
}

fn lmtd_matrix_f32(
    rec: &common::llama_tensor_dump_helpers::LmtdRecordV2,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Decode a rank-2-ish f32 tensor into row-major [rows, cols], accepting either axis order.
    // LMTD uses ggml strides (nb) and may expose [cols, rows] or [rows, cols].
    let active: Vec<usize> = (0..4).filter(|&i| rec.ne[i] > 1).collect();
    if active.len() != 2 {
        return Err(format!(
            "lmtd_matrix_f32: expected 2 active dims, got {:?} with ne {:?}",
            active, rec.ne
        )
        .into());
    }
    let a0 = active[0];
    let a1 = active[1];
    let d0 = usize::try_from(rec.ne[a0]).map_err(|_| "ne cast d0")?;
    let d1 = usize::try_from(rec.ne[a1]).map_err(|_| "ne cast d1")?;
    let (row_axis, col_axis) = if d0 == rows && d1 == cols {
        (a0, a1)
    } else if d0 == cols && d1 == rows {
        (a1, a0)
    } else {
        return Err(format!(
            "lmtd_matrix_f32: tensor dims [{d0},{d1}] incompatible with expected [{rows},{cols}]",
        )
        .into());
    };

    if rec.nb[row_axis] != 4 && rec.nb[col_axis] != 4 {
        return Err("lmtd_matrix_f32: neither axis has f32 stride-1 layout".into());
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let off = r as u64 * rec.nb[row_axis] + c as u64 * rec.nb[col_axis];
            let off = usize::try_from(off).map_err(|_| "offset cast")?;
            if off + 4 > rec.data.len() {
                return Err("lmtd_matrix_f32: out of bounds read".into());
            }
            out[r * cols + c] = f32::from_le_bytes(rec.data[off..off + 4].try_into().unwrap());
        }
    }
    Ok(out)
}

fn gemma_attn_normed_state(
    input: &PrefillState,
    config: &ModelConfig,
    layer_idx: usize,
    weights: &ModelWeights,
) -> PrefillState {
    let seq_len = input.seq_len();
    let hidden_dim = input.hidden_dim();
    let attn_norm_w = weights.layers[layer_idx]
        .attn_norm
        .as_f32_slice()
        .expect("attn_norm f32");
    let mut out = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        let s = pos * hidden_dim;
        let e = s + hidden_dim;
        rmsnorm(
            &input.hidden()[s..e],
            attn_norm_w,
            config.rms_norm_eps,
            &mut out[s..e],
        )
        .expect("attn rmsnorm");
    }
    PrefillState::from_flat(out, seq_len, hidden_dim).expect("PrefillState::from_flat")
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref; set GEMMA_LLAMA_TENSOR_NODE"]
fn gemma4_e2b_tensor_node_vs_rust() {
    let node: u32 = std::env::var("GEMMA_LLAMA_TENSOR_NODE")
        .expect("set GEMMA_LLAMA_TENSOR_NODE to the llama trace line index")
        .parse()
        .expect("GEMMA_LLAMA_TENSOR_NODE parse");

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");

    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(
        dump_bin.is_file(),
        "missing {} — run ./tools/build_llama_logits_ref.sh",
        dump_bin.display()
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

    let seq_len = prompt_ids.len();
    eprintln!("prompt ids: {prompt_ids:?} GEMMA_LLAMA_TENSOR_NODE={node}");

    let lmtd_bytes =
        run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, node).expect("tensor dump");
    let rec = parse_lmtd_v2(&lmtd_bytes).expect("parse LMTD");
    eprintln!(
        "llama node {}: type={} ne={:?} nbytes={}",
        rec.node_index,
        rec.ggml_type,
        rec.ne,
        rec.data.len()
    );

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let n_layers_total = config.n_layers;
    let num_layers = if let Ok(s) = std::env::var("GEMMA_RUST_NUM_LAYERS") {
        s.parse::<usize>().expect("GEMMA_RUST_NUM_LAYERS parse")
    } else {
        n_layers_total
    };
    assert!(
        num_layers <= n_layers_total,
        "GEMMA_RUST_NUM_LAYERS {num_layers} > n_layers {n_layers_total}"
    );

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load weights");

    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill embed");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let mut kv_caches = kv_caches_for_config(&config);
    let stage = std::env::var("GEMMA_RUST_STAGE").unwrap_or_else(|_| "l_out".to_string());
    let rust_vec: Vec<f32> = if [
        "attn_proj",
        "attn_post_norm",
        "ffn_normed",
        "ffn_gate",
        "ffn_up",
        "ffn_geglu",
        "ffn_down",
        "q_raw",
        "k_raw",
        "q_normed",
        "k_normed",
        "q_after_rope",
        "k_after_rope",
        "ffn_post_rms",
        "ffn_post_mul",
        "ffn_post_norm",
        "pe_in",
        "after_tail",
        "l_out_debug",
    ]
    .contains(&stage.as_str())
    {
        let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
            .expect("set GEMMA_RUST_LAYER_INDEX for GEMMA_RUST_STAGE=<gemma debug stage>")
            .parse()
            .expect("GEMMA_RUST_LAYER_INDEX parse");
        assert!(
            layer_idx < config.n_layers,
            "GEMMA_RUST_LAYER_INDEX {layer_idx} out of range"
        );
        let pre_state =
            prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
                .expect("prefill state before target layer");
        let dbg = gemma4_prefill_layer_debug(
            &pre_state,
            &config,
            layer_idx,
            &weights.layers[layer_idx],
            &mut kv_caches,
        )
        .expect("gemma4_prefill_layer_debug");
        let slice_last = |v: &[f32], width: usize| -> Vec<f32> {
            v[(seq_len - 1) * width..seq_len * width].to_vec()
        };
        match stage.as_str() {
            "attn_proj" => slice_last(&dbg.attn_proj, config.hidden_dim),
            "attn_post_norm" => slice_last(&dbg.attn_post_norm, config.hidden_dim),
            "ffn_normed" => slice_last(&dbg.ffn_normed, config.hidden_dim),
            "ffn_gate" => slice_last(
                &dbg.ffn_gate,
                config
                    .layer_dims_for(layer_idx)
                    .expect("layer dims")
                    .ffn_dim,
            ),
            "ffn_up" => slice_last(
                &dbg.ffn_up,
                config
                    .layer_dims_for(layer_idx)
                    .expect("layer dims")
                    .ffn_dim,
            ),
            "ffn_geglu" => slice_last(
                &dbg.ffn_geglu,
                config
                    .layer_dims_for(layer_idx)
                    .expect("layer dims")
                    .ffn_dim,
            ),
            "ffn_down" => slice_last(&dbg.ffn_down, config.hidden_dim),
            "q_raw" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let (q, _k) = prefill_qk_raw_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    &weights.layers[layer_idx],
                )
                .expect("prefill_qk_raw_layer");
                slice_last(&q, layer_dims.q_dim)
            }
            "k_raw" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let (_q, k) = prefill_qk_raw_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    &weights.layers[layer_idx],
                )
                .expect("prefill_qk_raw_layer");
                slice_last(&k, layer_dims.kv_dim)
            }
            "q_normed" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let (q, _k) = prefill_qk_normed_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    &weights.layers[layer_idx],
                )
                .expect("prefill_qk_normed_layer");
                slice_last(&q, layer_dims.q_dim)
            }
            "k_normed" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let (_q, k) = prefill_qk_normed_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    &weights.layers[layer_idx],
                )
                .expect("prefill_qk_normed_layer");
                slice_last(&k, layer_dims.kv_dim)
            }
            "q_after_rope" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let layer_attn = config.layer_attention_for(layer_idx).expect("layer attn");
                let (q, _k) = prefill_qk_after_rope_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    layer_attn,
                    &weights.layers[layer_idx],
                    &kv_caches,
                    layer_idx,
                )
                .expect("prefill_qk_after_rope_layer");
                slice_last(&q, layer_dims.q_dim)
            }
            "k_after_rope" => {
                let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
                let layer_attn = config.layer_attention_for(layer_idx).expect("layer attn");
                let (_q, k) = prefill_qk_after_rope_layer(
                    &pre_state,
                    &config,
                    layer_dims,
                    layer_attn,
                    &weights.layers[layer_idx],
                    &kv_caches,
                    layer_idx,
                )
                .expect("prefill_qk_after_rope_layer");
                slice_last(&k, layer_dims.kv_dim)
            }
            "ffn_post_rms" => slice_last(&dbg.ffn_post_rms, config.hidden_dim),
            "ffn_post_mul" => slice_last(&dbg.ffn_post_mul, config.hidden_dim),
            "ffn_post_norm" => slice_last(&dbg.ffn_post_norm, config.hidden_dim),
            "pe_in" => slice_last(&dbg.pe_in, config.hidden_dim),
            "after_tail" => slice_last(&dbg.after_tail, config.hidden_dim),
            "l_out_debug" => slice_last(&dbg.l_out, config.hidden_dim),
            _ => unreachable!(),
        }
    } else if stage == "attn_core" {
        let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
            .expect("set GEMMA_RUST_LAYER_INDEX for GEMMA_RUST_STAGE=attn_core")
            .parse()
            .expect("GEMMA_RUST_LAYER_INDEX parse");
        assert!(
            layer_idx < config.n_layers,
            "GEMMA_RUST_LAYER_INDEX {layer_idx} out of range"
        );
        let pre_state =
            prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
                .expect("prefill state before target layer");
        let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
        let layer_attn = config.layer_attention_for(layer_idx).expect("layer attn");
        let attn_core = prefill_attention_core_layer(
            &pre_state,
            &config,
            layer_dims,
            layer_attn,
            &weights.layers[layer_idx],
            &mut kv_caches,
            layer_idx,
        )
        .expect("prefill_attention_core_layer");
        let q_dim = layer_dims.q_dim;
        attn_core[(seq_len - 1) * q_dim..seq_len * q_dim].to_vec()
    } else if stage == "attn_out" {
        let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
            .expect("set GEMMA_RUST_LAYER_INDEX for GEMMA_RUST_STAGE=attn_out")
            .parse()
            .expect("GEMMA_RUST_LAYER_INDEX parse");
        assert!(
            layer_idx < config.n_layers,
            "GEMMA_RUST_LAYER_INDEX {layer_idx} out of range"
        );
        let pre_state =
            prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
                .expect("prefill state before target layer");
        let attn_out = prefill_attention_with_norm(
            &pre_state,
            &config,
            layer_idx,
            &weights.layers[layer_idx],
            &mut kv_caches,
        )
        .expect("prefill_attention_with_norm");
        let hd = pre_state.hidden_dim();
        attn_out[(seq_len - 1) * hd..seq_len * hd].to_vec()
    } else {
        let state =
            prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, num_layers)
                .expect("prefill_forward_layers");

        let hd = state.hidden_dim();
        let last = &state.hidden()[(seq_len - 1) * hd..seq_len * hd];

        let apply_norm = std::env::var("GEMMA_APPLY_OUTPUT_NORM")
            .map(|s| s == "1")
            .unwrap_or(false);
        if apply_norm {
            let norm_w = weights.output_norm.as_f32_slice().expect("output norm f32");
            let mut v = vec![0f32; hd];
            rmsnorm(last, norm_w, config.rms_norm_eps, &mut v).expect("rmsnorm");
            v
        } else {
            last.to_vec()
        }
    };

    let hd = config.hidden_dim;
    let llama_hd = if ["ffn_gate", "ffn_up", "ffn_geglu"].contains(&stage.as_str()) {
        config
            .layer_dims_for(
                std::env::var("GEMMA_RUST_LAYER_INDEX")
                    .expect("set GEMMA_RUST_LAYER_INDEX for FFN-width stage")
                    .parse::<usize>()
                    .expect("GEMMA_RUST_LAYER_INDEX parse"),
            )
            .expect("layer dims for FFN-width stage")
            .ffn_dim
    } else if stage == "q_raw" || stage == "q_normed" || stage == "q_after_rope" {
        config
            .layer_dims_for(
                std::env::var("GEMMA_RUST_LAYER_INDEX")
                    .expect("set GEMMA_RUST_LAYER_INDEX for q_after_rope")
                    .parse::<usize>()
                    .expect("GEMMA_RUST_LAYER_INDEX parse"),
            )
            .expect("layer dims for q stage")
            .q_dim
    } else if stage == "k_raw" || stage == "k_normed" || stage == "k_after_rope" {
        config
            .layer_dims_for(
                std::env::var("GEMMA_RUST_LAYER_INDEX")
                    .expect("set GEMMA_RUST_LAYER_INDEX for k_after_rope")
                    .parse::<usize>()
                    .expect("GEMMA_RUST_LAYER_INDEX parse"),
            )
            .expect("layer dims for k stage")
            .kv_dim
    } else if stage == "attn_core" {
        config
            .layer_dims_for(
                std::env::var("GEMMA_RUST_LAYER_INDEX")
                    .expect("set GEMMA_RUST_LAYER_INDEX for GEMMA_RUST_STAGE=attn_core")
                    .parse::<usize>()
                    .expect("GEMMA_RUST_LAYER_INDEX parse"),
            )
            .expect("layer dims for attn_core")
            .q_dim
    } else {
        hd
    };
    let llama_vec = if stage == "q_after_rope"
        || stage == "k_after_rope"
        || stage == "q_raw"
        || stage == "k_raw"
        || stage == "q_normed"
        || stage == "k_normed"
        || stage == "attn_core"
    {
        lmtd_last_token_f32_any_seq(&rec, seq_len, llama_hd).expect("lmtd last token any-seq")
    } else {
        lmtd_last_token_f32(&rec, seq_len, llama_hd).expect("lmtd last token f32")
    };

    assert_eq!(
        rust_vec.len(),
        llama_vec.len(),
        "vector len mismatch for selected stage"
    );

    let max_abs = max_abs_diff(&rust_vec, &llama_vec);
    let top_diffs = env_usize("GEMMA_TENSOR_TOP_DIFFS", 0);
    let maybe_scale = if stage == "ffn_post_norm" || stage == "ffn_post_mul" {
        std::env::var("GEMMA_RUST_LAYER_INDEX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .and_then(|layer_idx| weights.layers.get(layer_idx))
            .and_then(|layer| layer.ffn_post_norm)
            .and_then(|t| t.as_f32_slice().ok())
    } else {
        None
    };
    print_top_diffs(
        &format!("stage={stage}"),
        &rust_vec,
        &llama_vec,
        top_diffs,
        maybe_scale,
    );
    let watch_dims = dim_ids_from_env("GEMMA_TENSOR_WATCH_DIMS");
    print_watch_dims(
        &format!("stage={stage}"),
        &rust_vec,
        &llama_vec,
        &watch_dims,
        maybe_scale,
    );
    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    eprintln!(
        "compare: rust stage={stage} num_layers={num_layers} max |Δ| = {max_abs:.6} (tol {tol})"
    );
    assert!(
        max_abs <= tol,
        "tensor node {node} vs Rust hidden: max |Δ| {max_abs} > {tol}"
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref"]
fn gemma4_post_ffw_scale_candidates_vs_llama() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let node_rms: u32 = std::env::var("GEMMA_LLAMA_RMS_NODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(776);
    let node_mul: u32 = std::env::var("GEMMA_LLAMA_MUL_NODE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(777);

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let layer = &weights.layers[layer_idx];
    let hidden_dim = config.hidden_dim;

    let rms_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, node_rms)
        .expect("rms dump");
    let mul_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, node_mul)
        .expect("mul dump");
    let rms_rec = parse_lmtd_v2(&rms_bytes).expect("parse rms");
    let mul_rec = parse_lmtd_v2(&mul_bytes).expect("parse mul");
    let llama_rms = lmtd_last_token_f32(&rms_rec, seq_len, hidden_dim).expect("llama rms");
    let llama_mul = lmtd_last_token_f32(&mul_rec, seq_len, hidden_dim).expect("llama mul");

    let mut implied_scale = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let d = llama_rms[i];
        implied_scale[i] = if d.abs() > 1e-6 {
            llama_mul[i] / d
        } else {
            0.0
        };
    }

    let post_ffw = layer
        .ffn_post_norm
        .expect("missing post_ffw_norm")
        .as_f32_slice()
        .expect("post_ffw_norm f32")
        .to_vec();
    let ffn_norm = layer
        .ffn_norm
        .as_f32_slice()
        .expect("ffn_norm f32")
        .to_vec();
    let attn_post = layer
        .attn_post_norm
        .expect("missing post_attention_norm")
        .as_f32_slice()
        .expect("attn_post_norm f32")
        .to_vec();
    let attn_norm = layer
        .attn_norm
        .as_f32_slice()
        .expect("attn_norm f32")
        .to_vec();

    let candidates: [(&str, Vec<f32>); 8] = [
        ("post_ffw_norm.weight", post_ffw.clone()),
        (
            "1+post_ffw_norm.weight",
            post_ffw.iter().map(|v| 1.0 + *v).collect(),
        ),
        ("ffn_norm.weight", ffn_norm.clone()),
        (
            "1+ffn_norm.weight",
            ffn_norm.iter().map(|v| 1.0 + *v).collect(),
        ),
        ("post_attention_norm.weight", attn_post.clone()),
        (
            "1+post_attention_norm.weight",
            attn_post.iter().map(|v| 1.0 + *v).collect(),
        ),
        ("attn_norm.weight", attn_norm.clone()),
        (
            "1+attn_norm.weight",
            attn_norm.iter().map(|v| 1.0 + *v).collect(),
        ),
    ];

    eprintln!(
        "layer={layer_idx} node_rms={node_rms} node_mul={node_mul} prompt_ids={prompt_ids:?}"
    );
    for (name, cand) in candidates {
        let (max_abs, rmse) = diff_stats(&implied_scale, &cand);
        eprintln!("candidate {name:>28}: max_abs={max_abs:.6} rmse={rmse:.6}");
    }
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer; isolates w_down q8 kernel parity"]
fn gemma4_w_down_q8_kernel_vs_dense_reference() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = if let Ok(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS") {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        tokenizer
            .encode_with_prompt_config(
                &std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string()),
                &tok_prompt,
            )
            .expect("encode")
    };
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let mut kv_caches = kv_caches_for_config(&config);
    let pre_state =
        prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
            .expect("prefill to layer");
    let dbg = gemma4_prefill_layer_debug(
        &pre_state,
        &config,
        layer_idx,
        &weights.layers[layer_idx],
        &mut kv_caches,
    )
    .expect("gemma4 debug");

    let w_down = weights.layers[layer_idx].w_down;
    let dims = w_down.dimensions();
    let (k, n) = (dims[0], dims[1]);
    assert_eq!(
        dbg.ffn_geglu.len() % k,
        0,
        "ffn_geglu len must be multiple of K"
    );
    let m = dbg.ffn_geglu.len() / k;
    let q8_out = dbg.ffn_down.clone();
    let dense_w = match w_down.dtype() {
        TensorType::Q8_0 => dequantize_q8_0_tensor_to_f32(w_down),
        TensorType::F32 => w_down.as_f32_slice().expect("w_down f32").to_vec(),
        other => panic!("w_down dtype {other:?} not handled in this test"),
    };
    let dense_out = matmul_dense_ggml_layout(&dbg.ffn_geglu, m, k, &dense_w, n);
    let (max_abs, rmse) = diff_stats(&q8_out, &dense_out);
    eprintln!(
        "layer={layer_idx} w_down dtype={:?} K={k} N={n} M={m} q8_vs_dense: max_abs={max_abs:.6} rmse={rmse:.6}",
        w_down.dtype()
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer; isolates w_gate/w_up q8 kernel parity"]
fn gemma4_w_gate_up_q8_kernel_vs_dense_reference() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = if let Ok(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS") {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        tokenizer
            .encode_with_prompt_config(
                &std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string()),
                &tok_prompt,
            )
            .expect("encode")
    };
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let mut kv_caches = kv_caches_for_config(&config);
    let pre_state =
        prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
            .expect("prefill to layer");
    let dbg = gemma4_prefill_layer_debug(
        &pre_state,
        &config,
        layer_idx,
        &weights.layers[layer_idx],
        &mut kv_caches,
    )
    .expect("gemma4 debug");

    let x = &dbg.ffn_normed;
    let hidden_dim = config.hidden_dim;
    let ffn_dim = config
        .layer_dims_for(layer_idx)
        .expect("layer dims")
        .ffn_dim;
    let m = x.len() / hidden_dim;

    let eval_one = |name: &str, w: &inference_engine_rust::core::tensor::Tensor, got: &[f32]| {
        let dims = w.dimensions();
        let (k, n) = (dims[0], dims[1]);
        assert_eq!(k, hidden_dim, "{name}: K mismatch");
        assert_eq!(n, ffn_dim, "{name}: N mismatch");
        let dense_w = match w.dtype() {
            TensorType::Q8_0 => dequantize_q8_0_tensor_to_f32(w),
            TensorType::F32 => w.as_f32_slice().expect("f32 weight").to_vec(),
            other => panic!("{name}: unsupported dtype {other:?}"),
        };
        let dense_out = matmul_dense_ggml_layout(x, m, k, &dense_w, n);
        let (max_abs, rmse) = diff_stats(got, &dense_out);
        eprintln!(
            "layer={layer_idx} {name} dtype={:?} M={m} K={k} N={n} q8_vs_dense: max_abs={max_abs:.6} rmse={rmse:.6}",
            w.dtype()
        );
    };

    eval_one("w_gate", weights.layers[layer_idx].w_gate, &dbg.ffn_gate);
    eval_one("w_up", weights.layers[layer_idx].w_up, &dbg.ffn_up);
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer; isolates wq/wk projection kernel parity"]
fn gemma4_wq_wk_q8_kernel_vs_dense_reference() {
    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let path_str = GEMMA4_E2B_Q8_GGUF_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf");
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf).expect("tok config");
    let prompt_ids = if let Ok(s) = std::env::var("GEMMA_LOGITS_TOKEN_IDS") {
        let mut out = Vec::new();
        for part in s.split([',', ' ']).filter(|p| !p.is_empty()) {
            out.push(part.parse().expect("GEMMA_LOGITS_TOKEN_IDS parse"));
        }
        out
    } else {
        let mut tokenizer = Tokenizer::load_from_file(&tok_path).expect("tokenizer");
        tokenizer
            .encode_with_prompt_config(
                &std::env::var("GEMMA_LOGITS_PROMPT").unwrap_or_else(|_| "Hello".to_string()),
                &tok_prompt,
            )
            .expect("encode")
    };
    let config = ModelConfig::from_gguf(&gguf).expect("config");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let prefill_in =
        prefill_from_tokens(&mut gguf, path_str, &config, &prompt_ids).expect("prefill");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let mut kv_caches = kv_caches_for_config(&config);
    let pre_state =
        prefill_forward_layers(&prefill_in, &config, &weights, &mut kv_caches, layer_idx)
            .expect("prefill to layer");
    let attn_normed = gemma_attn_normed_state(&pre_state, &config, layer_idx, &weights);
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");

    let (q_raw, k_raw) = prefill_qk_raw_layer(
        &attn_normed,
        &config,
        layer_dims,
        &weights.layers[layer_idx],
    )
    .expect("prefill_qk_raw_layer");
    let seq_len = attn_normed.seq_len();
    let hidden_dim = attn_normed.hidden_dim();

    let eval_one = |name: &str,
                    w: &inference_engine_rust::core::tensor::Tensor,
                    n: usize,
                    got: &[f32]| {
        let dims = w.dimensions();
        let (k, wn) = (dims[0], dims[1]);
        assert_eq!(k, hidden_dim, "{name}: K mismatch");
        assert_eq!(wn, n, "{name}: N mismatch");
        let dense_w = match w.dtype() {
            TensorType::Q8_0 => dequantize_q8_0_tensor_to_f32(w),
            TensorType::F32 => w.as_f32_slice().expect("f32 weight").to_vec(),
            other => panic!("{name}: unsupported dtype {other:?}"),
        };
        let dense_out =
            matmul_dense_ggml_layout(attn_normed.hidden(), seq_len, hidden_dim, &dense_w, n);
        let (max_abs, rmse) = diff_stats(got, &dense_out);
        eprintln!(
            "layer={layer_idx} {name} dtype={:?} M={seq_len} K={k} N={n} q8_vs_dense: max_abs={max_abs:.6} rmse={rmse:.6}",
            w.dtype()
        );
    };

    eval_one("wq", weights.layers[layer_idx].wq, layer_dims.q_dim, &q_raw);
    eval_one(
        "wk",
        weights.layers[layer_idx].wk,
        layer_dims.kv_dim,
        &k_raw,
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + attention node ids"]
fn gemma4_single_layer_attention_llama_io_parity() {
    // Provide node ids from /tmp/gemma_nodes.tsv:
    // - GEMMA_LLAMA_ATTN_IN_NODE: llama node whose tensor is the exact input to this layer's attention op
    // - GEMMA_LLAMA_ATTN_OUT_NODE: llama node whose tensor is attention output after wo (same shape)
    let in_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_IN_NODE")
        .expect("set GEMMA_LLAMA_ATTN_IN_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_IN_NODE parse");
    let out_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_OUT_NODE")
        .expect("set GEMMA_LLAMA_ATTN_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_OUT_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let in_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, in_node)
        .expect("dump in-node");
    let out_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, out_node)
        .expect("dump out-node");
    let in_rec = parse_lmtd_v2(&in_bytes).expect("parse in");
    let out_rec = parse_lmtd_v2(&out_bytes).expect("parse out");

    let hidden_dim = config.hidden_dim;
    let llama_attn_in =
        lmtd_matrix_f32(&in_rec, seq_len, hidden_dim).expect("decode llama attention input matrix");
    let llama_attn_out = lmtd_matrix_f32(&out_rec, seq_len, hidden_dim)
        .expect("decode llama attention output matrix");

    let input_state = PrefillState::from_flat(llama_attn_in, seq_len, hidden_dim)
        .expect("input prefill state from llama");
    let mut kv_caches = kv_caches_for_config(&config);
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
    let layer_attn = config.layer_attention_for(layer_idx).expect("layer attn");
    let rust_attn_out = prefill_attention_layer(
        &input_state,
        &config,
        layer_dims,
        layer_attn,
        &weights.layers[layer_idx],
        &mut kv_caches,
        layer_idx,
    )
    .expect("prefill_attention_layer");

    let (max_abs, rmse) = diff_stats(&rust_attn_out, &llama_attn_out);
    eprintln!(
        "single-layer attention parity: layer={layer_idx} in_node={in_node} out_node={out_node} seq={seq_len} hidden={hidden_dim} max_abs={max_abs:.6} rmse={rmse:.6}"
    );

    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    assert!(
        max_abs <= tol,
        "single-layer attention mismatch: max_abs {max_abs:.6} > tol {tol:.6} (rmse={rmse:.6})"
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + FFN node ids"]
fn gemma4_single_layer_ffn_llama_io_parity() {
    // Feed llama.cpp's FFN input for the last token into Rust's FFN and compare to llama.cpp FFN output.
    // Typical nodes: GEMMA_LLAMA_FFN_IN_NODE=1476, GEMMA_LLAMA_FFN_OUT_NODE=1480 for layer 34.
    let in_node: u32 = std::env::var("GEMMA_LLAMA_FFN_IN_NODE")
        .expect("set GEMMA_LLAMA_FFN_IN_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_IN_NODE parse");
    let out_node: u32 = std::env::var("GEMMA_LLAMA_FFN_OUT_NODE")
        .expect("set GEMMA_LLAMA_FFN_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_OUT_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(34);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let in_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, in_node)
        .expect("dump in-node");
    let out_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, out_node)
        .expect("dump out-node");
    let in_rec = parse_lmtd_v2(&in_bytes).expect("parse in");
    let out_rec = parse_lmtd_v2(&out_bytes).expect("parse out");

    let hidden_dim = config.hidden_dim;
    let llama_ffn_in =
        lmtd_last_token_f32(&in_rec, seq_len, hidden_dim).expect("decode llama FFN input");
    let llama_ffn_out =
        lmtd_last_token_f32(&out_rec, seq_len, hidden_dim).expect("decode llama FFN output");
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
    let rust_ffn_out = prefill_ffn(
        &llama_ffn_in,
        1,
        hidden_dim,
        layer_dims.ffn_dim,
        &config,
        &weights.layers[layer_idx],
    )
    .expect("prefill_ffn on llama input");

    let (max_abs, rmse) = diff_stats(&rust_ffn_out, &llama_ffn_out);
    eprintln!(
        "single-layer FFN parity: layer={layer_idx} in_node={in_node} out_node={out_node} max_abs={max_abs:.6} rmse={rmse:.6}"
    );
    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    assert!(
        max_abs <= tol,
        "single-layer FFN mismatch: max_abs {max_abs:.6} > tol {tol:.6} (rmse={rmse:.6})"
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + FFN down node ids"]
fn gemma4_ffn_down_llama_geglu_input_parity() {
    // Feed llama.cpp's exact GeGLU activation into Rust's final FFN down projection.
    // For layer 23 on the 17-token continuation: GEMMA_LLAMA_FFN_GEGLU_NODE=1070,
    // GEMMA_LLAMA_FFN_OUT_NODE=1071.
    let geglu_node: u32 = std::env::var("GEMMA_LLAMA_FFN_GEGLU_NODE")
        .expect("set GEMMA_LLAMA_FFN_GEGLU_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_GEGLU_NODE parse");
    let out_node: u32 = std::env::var("GEMMA_LLAMA_FFN_OUT_NODE")
        .expect("set GEMMA_LLAMA_FFN_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_OUT_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(23);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let geglu_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, geglu_node)
        .expect("dump geglu node");
    let out_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, out_node)
        .expect("dump out node");
    let geglu_rec = parse_lmtd_v2(&geglu_bytes).expect("parse geglu");
    let out_rec = parse_lmtd_v2(&out_bytes).expect("parse out");

    let llama_geglu =
        lmtd_last_token_f32(&geglu_rec, seq_len, layer_dims.ffn_dim).expect("decode geglu");
    let llama_out = lmtd_last_token_f32(&out_rec, seq_len, config.hidden_dim).expect("decode out");
    let rust_out = prefill_ffn_down_from_activated(
        &llama_geglu,
        1,
        config.hidden_dim,
        layer_dims.ffn_dim,
        &config,
        &weights.layers[layer_idx],
    )
    .expect("prefill_ffn_down_from_activated on llama GeGLU");

    let (max_abs, rmse) = diff_stats(&rust_out, &llama_out);
    eprintln!(
        "FFN down exact-input parity: layer={layer_idx} geglu_node={geglu_node} out_node={out_node} max_abs={max_abs:.6} rmse={rmse:.6}"
    );
    print_top_diffs(
        "ffn_down_exact_input",
        &rust_out,
        &llama_out,
        env_usize("GEMMA_TENSOR_TOP_DIFFS", 0),
        None,
    );
    let watch_dims = dim_ids_from_env("GEMMA_TENSOR_WATCH_DIMS");
    print_watch_dims(
        "ffn_down_exact_input",
        &rust_out,
        &llama_out,
        &watch_dims,
        None,
    );

    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    assert!(
        max_abs <= tol,
        "FFN down exact-input mismatch: max_abs {max_abs:.6} > tol {tol:.6} (rmse={rmse:.6})"
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + FFN intermediate node ids"]
fn gemma4_ffn_intermediates_llama_input_parity() {
    // Feed llama.cpp's exact FFN input into Rust's gate/up/activation path and compare
    // the last-token intermediate tensors to llama.cpp nodes.
    let in_node: u32 = std::env::var("GEMMA_LLAMA_FFN_IN_NODE")
        .expect("set GEMMA_LLAMA_FFN_IN_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_IN_NODE parse");
    let gate_node: u32 = std::env::var("GEMMA_LLAMA_FFN_GATE_NODE")
        .expect("set GEMMA_LLAMA_FFN_GATE_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_GATE_NODE parse");
    let up_node: u32 = std::env::var("GEMMA_LLAMA_FFN_UP_NODE")
        .expect("set GEMMA_LLAMA_FFN_UP_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_UP_NODE parse");
    let geglu_node: u32 = std::env::var("GEMMA_LLAMA_FFN_GEGLU_NODE")
        .expect("set GEMMA_LLAMA_FFN_GEGLU_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_GEGLU_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(23);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let load_last = |node: u32, width: usize| -> Vec<f32> {
        let bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, node)
            .expect("dump node");
        let rec = parse_lmtd_v2(&bytes).expect("parse node");
        lmtd_last_token_f32(&rec, seq_len, width).expect("decode node")
    };

    let llama_in = load_last(in_node, config.hidden_dim);
    let llama_gate = load_last(gate_node, layer_dims.ffn_dim);
    let llama_up = load_last(up_node, layer_dims.ffn_dim);
    let llama_geglu = load_last(geglu_node, layer_dims.ffn_dim);

    let debug = prefill_ffn_debug(
        &llama_in,
        1,
        config.hidden_dim,
        layer_dims.ffn_dim,
        &config,
        &weights.layers[layer_idx],
    )
    .expect("prefill_ffn_debug on llama input");

    for (name, rust, llama) in [
        ("ffn_gate_exact_input", &debug.gate, &llama_gate),
        ("ffn_up_exact_input", &debug.up, &llama_up),
        ("ffn_geglu_exact_input", &debug.activated, &llama_geglu),
    ] {
        let (max_abs, rmse) = diff_stats(rust, llama);
        eprintln!("{name}: layer={layer_idx} max_abs={max_abs:.6} rmse={rmse:.6}");
        print_top_diffs(
            name,
            rust,
            llama,
            env_usize("GEMMA_TENSOR_TOP_DIFFS", 0),
            None,
        );
        let watch_dims = dim_ids_from_env("GEMMA_TENSOR_WATCH_DIMS");
        print_watch_dims(name, rust, llama, &watch_dims, None);
        let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(80.0);
        assert!(
            max_abs <= tol,
            "{name} mismatch: max_abs {max_abs:.6} > tol {tol:.6} (rmse={rmse:.6})"
        );
    }
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + post-FFN node ids"]
fn gemma4_post_ffn_norm_llama_io_parity() {
    // Feed llama.cpp's FFN output into Rust's post-FFN RMS-only and scale path.
    // Typical nodes for layer 34: input=1480, rms=1481, scaled=1482.
    let in_node: u32 = std::env::var("GEMMA_LLAMA_FFN_OUT_NODE")
        .expect("set GEMMA_LLAMA_FFN_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_OUT_NODE parse");
    let rms_node: u32 = std::env::var("GEMMA_LLAMA_FFN_POST_RMS_NODE")
        .expect("set GEMMA_LLAMA_FFN_POST_RMS_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_POST_RMS_NODE parse");
    let scaled_node: u32 = std::env::var("GEMMA_LLAMA_FFN_POST_MUL_NODE")
        .expect("set GEMMA_LLAMA_FFN_POST_MUL_NODE")
        .parse()
        .expect("GEMMA_LLAMA_FFN_POST_MUL_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(34);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");
    let post_w = weights.layers[layer_idx]
        .ffn_post_norm
        .expect("missing post_ffw_norm")
        .as_f32_slice()
        .expect("post_ffw_norm f32");

    let load_last = |node: u32| -> Vec<f32> {
        let bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, node)
            .expect("dump node");
        let rec = parse_lmtd_v2(&bytes).expect("parse node");
        lmtd_last_token_f32(&rec, seq_len, config.hidden_dim).expect("decode node")
    };
    let llama_in = load_last(in_node);
    let llama_rms = load_last(rms_node);
    let llama_scaled = load_last(scaled_node);

    let mut rust_rms = llama_in.clone();
    rmsnorm_inplace_no_scale(&mut rust_rms, config.rms_norm_eps);
    let mut rust_scaled = vec![0.0f32; config.hidden_dim];
    rmsnorm(&llama_in, post_w, config.rms_norm_eps, &mut rust_scaled).expect("post ffn rmsnorm");

    let (rms_max, rms_rmse) = diff_stats(&rust_rms, &llama_rms);
    let (scaled_max, scaled_rmse) = diff_stats(&rust_scaled, &llama_scaled);
    eprintln!(
        "post-FFN norm IO parity: layer={layer_idx} in_node={in_node} rms_node={rms_node} scaled_node={scaled_node} rms_max={rms_max:.6} rms_rmse={rms_rmse:.6} scaled_max={scaled_max:.6} scaled_rmse={scaled_rmse:.6}"
    );
    let top_n = env_usize("GEMMA_TENSOR_TOP_DIFFS", 0);
    print_top_diffs(
        "post_ffn_rms_exact_input",
        &rust_rms,
        &llama_rms,
        top_n,
        None,
    );
    print_top_diffs(
        "post_ffn_scaled_exact_input",
        &rust_scaled,
        &llama_scaled,
        top_n,
        Some(post_w),
    );

    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    assert!(
        rms_max <= tol && scaled_max <= tol,
        "post-FFN norm mismatch: rms_max {rms_max:.6}, scaled_max {scaled_max:.6}, tol {tol:.6}"
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + attention node ids"]
fn gemma4_single_layer_attention_drift_amplification() {
    // Same node-id contract as gemma4_single_layer_attention_llama_io_parity.
    let in_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_IN_NODE")
        .expect("set GEMMA_LLAMA_ATTN_IN_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_IN_NODE parse");
    let out_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_OUT_NODE")
        .expect("set GEMMA_LLAMA_ATTN_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_OUT_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let repeat_layers: usize = std::env::var("GEMMA_REPEAT_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let in_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, in_node)
        .expect("dump in-node");
    let out_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, out_node)
        .expect("dump out-node");
    let in_rec = parse_lmtd_v2(&in_bytes).expect("parse in");
    let out_rec = parse_lmtd_v2(&out_bytes).expect("parse out");

    let hidden_dim = config.hidden_dim;
    let llama_in = lmtd_matrix_f32(&in_rec, seq_len, hidden_dim).expect("decode llama in");
    let llama_out = lmtd_matrix_f32(&out_rec, seq_len, hidden_dim).expect("decode llama out");

    // First, compute Rust attention output on the same llama input.
    let input_state =
        PrefillState::from_flat(llama_in, seq_len, hidden_dim).expect("state from llama in");
    let mut kv0 = kv_caches_for_config(&config);
    let layer_dims = config.layer_dims_for(layer_idx).expect("layer dims");
    let layer_attn = config.layer_attention_for(layer_idx).expect("layer attn");
    let rust_out0 = prefill_attention_layer(
        &input_state,
        &config,
        layer_dims,
        layer_attn,
        &weights.layers[layer_idx],
        &mut kv0,
        layer_idx,
    )
    .expect("prefill_attention_layer");

    let (base_max, base_rmse) = diff_stats(&rust_out0, &llama_out);
    eprintln!(
        "base one-step delta: layer={layer_idx} in_node={in_node} out_node={out_node} seq={seq_len} max_abs={base_max:.6} rmse={base_rmse:.6}"
    );

    // Amplification experiment:
    // A path starts from llama single-step output.
    // B path starts from rust single-step output.
    // Then apply the same Rust attention operator F repeatedly and watch ||A-B|| grow.
    let mut a = llama_out;
    let mut b = rust_out0;
    for step in 1..=repeat_layers {
        let state_a = PrefillState::from_flat(a, seq_len, hidden_dim).expect("state A");
        let state_b = PrefillState::from_flat(b, seq_len, hidden_dim).expect("state B");
        let mut kv_a = kv_caches_for_config(&config);
        let mut kv_b = kv_caches_for_config(&config);
        a = prefill_attention_layer(
            &state_a,
            &config,
            layer_dims,
            layer_attn,
            &weights.layers[layer_idx],
            &mut kv_a,
            layer_idx,
        )
        .expect("iter attention A");
        b = prefill_attention_layer(
            &state_b,
            &config,
            layer_dims,
            layer_attn,
            &weights.layers[layer_idx],
            &mut kv_b,
            layer_idx,
        )
        .expect("iter attention B");
        let (mx, rmse) = diff_stats(&a, &b);
        eprintln!("repeat step {step:02}: max_abs={mx:.6} rmse={rmse:.6}");
    }

    let (final_max, final_rmse) = diff_stats(&a, &b);
    let amp = if base_rmse > 0.0 {
        final_rmse / base_rmse
    } else {
        0.0
    };
    eprintln!(
        "final after {repeat_layers} repeats: max_abs={final_max:.6} rmse={final_rmse:.6} amplification={amp:.3}x"
    );

    // Optional final-token logits check after repeated attention applications.
    let state_a = PrefillState::from_flat(a, seq_len, hidden_dim).expect("state A logits");
    let state_b = PrefillState::from_flat(b, seq_len, hidden_dim).expect("state B logits");
    let logits_a =
        inference_engine_rust::prefill::final_logits_last_token(&state_a, &config, &weights)
            .expect("final logits A");
    let logits_b =
        inference_engine_rust::prefill::final_logits_last_token(&state_b, &config, &weights)
            .expect("final logits B");
    let arg_a = argmax_f32(&logits_a).expect("argmax A logits");
    let arg_b = argmax_f32(&logits_b).expect("argmax B logits");
    let (logits_max, logits_rmse) = diff_stats(&logits_a, &logits_b);
    eprintln!(
        "after {repeat_layers} repeats logits diff: max_abs={logits_max:.6} rmse={logits_rmse:.6} argmax_a={arg_a} argmax_b={arg_b} same_argmax={}",
        arg_a == arg_b
    );
}

#[test]
#[ignore = "requires Gemma GGUF + tokenizer + tools/llama_tensor_dump_ref + wrapper node ids"]
fn gemma4_single_layer_attention_wrapper_llama_io_parity() {
    // Wrapper I/O node IDs:
    // - GEMMA_LLAMA_ATTN_WRAPPER_IN_NODE: typically layer input before attn path (e.g. l_out of previous layer)
    // - GEMMA_LLAMA_ATTN_WRAPPER_OUT_NODE: attention wrapper output after post_attn_norm + residual add (e.g. attn_out-*)
    let in_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_WRAPPER_IN_NODE")
        .expect("set GEMMA_LLAMA_ATTN_WRAPPER_IN_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_WRAPPER_IN_NODE parse");
    let out_node: u32 = std::env::var("GEMMA_LLAMA_ATTN_WRAPPER_OUT_NODE")
        .expect("set GEMMA_LLAMA_ATTN_WRAPPER_OUT_NODE")
        .parse()
        .expect("GEMMA_LLAMA_ATTN_WRAPPER_OUT_NODE parse");
    let layer_idx: usize = std::env::var("GEMMA_RUST_LAYER_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let model_path = gemma4_e2b_q8_gguf_path();
    assert!(model_path.is_file(), "missing Gemma GGUF");
    let tok_path = gemma4_e2b_tokenizer_path();
    assert!(tok_path.is_file(), "missing tokenizer.json");
    let dump_bin = llama_tensor_dump_ref_binary();
    assert!(dump_bin.is_file(), "missing llama_tensor_dump_ref");

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
    let seq_len = prompt_ids.len();

    let in_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, in_node)
        .expect("dump in-node");
    let out_bytes = run_llama_tensor_dump_node(&dump_bin, &model_path, &prompt_ids, out_node)
        .expect("dump out-node");
    let in_rec = parse_lmtd_v2(&in_bytes).expect("parse in");
    let out_rec = parse_lmtd_v2(&out_bytes).expect("parse out");

    let config = ModelConfig::from_gguf(&gguf).expect("config");
    assert!(layer_idx < config.n_layers, "layer_idx out of range");
    let hidden_dim = config.hidden_dim;
    let llama_in =
        lmtd_matrix_f32(&in_rec, seq_len, hidden_dim).expect("decode llama wrapper input matrix");
    let llama_out =
        lmtd_matrix_f32(&out_rec, seq_len, hidden_dim).expect("decode llama wrapper output matrix");

    let names = ModelWeightNames::resolve(&gguf, &config).expect("resolve");
    names.load_all(&mut gguf, path_str).expect("load");
    let weights = ModelWeights::from_loaded(&gguf, &names).expect("weights");

    let input_state = PrefillState::from_flat(llama_in, seq_len, hidden_dim)
        .expect("input prefill state from llama");
    let mut kv_caches = kv_caches_for_config(&config);
    let rust_out = prefill_attention_with_norm(
        &input_state,
        &config,
        layer_idx,
        &weights.layers[layer_idx],
        &mut kv_caches,
    )
    .expect("prefill_attention_with_norm");

    let (max_abs, rmse) = diff_stats(&rust_out, &llama_out);
    eprintln!(
        "single-layer attention-wrapper parity: layer={layer_idx} in_node={in_node} out_node={out_node} seq={seq_len} hidden={hidden_dim} max_abs={max_abs:.6} rmse={rmse:.6}"
    );
    let arg_r = argmax_f32(&rust_out).expect("argmax rust wrapper out");
    let arg_l = argmax_f32(&llama_out).expect("argmax llama wrapper out");
    eprintln!(
        "wrapper output argmax(flat): rust={arg_r} llama={arg_l} same={}",
        arg_r == arg_l
    );

    let tol: f32 = std::env::var("GEMMA_TENSOR_MAX_ABS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80.0);
    assert!(
        max_abs <= tol,
        "single-layer attention-wrapper mismatch: max_abs {max_abs:.6} > tol {tol:.6} (rmse={rmse:.6})"
    );

    let repeat_layers: usize = std::env::var("GEMMA_REPEAT_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let mut a = llama_out;
    let mut b = rust_out;
    for step in 1..=repeat_layers {
        let state_a = PrefillState::from_flat(a, seq_len, hidden_dim).expect("wrapper state A");
        let state_b = PrefillState::from_flat(b, seq_len, hidden_dim).expect("wrapper state B");
        let mut kv_a = kv_caches_for_config(&config);
        let mut kv_b = kv_caches_for_config(&config);
        a = prefill_attention_with_norm(
            &state_a,
            &config,
            layer_idx,
            &weights.layers[layer_idx],
            &mut kv_a,
        )
        .expect("iter wrapper A");
        b = prefill_attention_with_norm(
            &state_b,
            &config,
            layer_idx,
            &weights.layers[layer_idx],
            &mut kv_b,
        )
        .expect("iter wrapper B");
        let (mx, er) = diff_stats(&a, &b);
        eprintln!("wrapper repeat step {step:02}: max_abs={mx:.6} rmse={er:.6}");
    }
    let (final_max, final_rmse) = diff_stats(&a, &b);
    eprintln!(
        "wrapper final after {repeat_layers} repeats: max_abs={final_max:.6} rmse={final_rmse:.6}"
    );

    let state_a = PrefillState::from_flat(a, seq_len, hidden_dim).expect("wrapper logits state A");
    let state_b = PrefillState::from_flat(b, seq_len, hidden_dim).expect("wrapper logits state B");
    let logits_a =
        inference_engine_rust::prefill::final_logits_last_token(&state_a, &config, &weights)
            .expect("wrapper logits A");
    let logits_b =
        inference_engine_rust::prefill::final_logits_last_token(&state_b, &config, &weights)
            .expect("wrapper logits B");
    let tok_a = argmax_f32(&logits_a).expect("wrapper argmax A");
    let tok_b = argmax_f32(&logits_b).expect("wrapper argmax B");
    let (lm_max, lm_rmse) = diff_stats(&logits_a, &logits_b);
    eprintln!(
        "wrapper logits after {repeat_layers} repeats: max_abs={lm_max:.6} rmse={lm_rmse:.6} argmax_a={tok_a} argmax_b={tok_b} same_argmax={}",
        tok_a == tok_b
    );
}
