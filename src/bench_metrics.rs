//! Wall-clock timings for comparing this engine to llama-bench / llama.cpp-style metrics.
//!
//! - **Cold start** — from [`crate::model_loader::file_loader::read_file`] through the first
//!   greedy token id (tokenizer + weight load + prefill + output head + sample).
//! - **Interactive TTFT** — weights already resident; fresh KV cache. Reports
//!   **`prompt_eval_ms`** (full transformer over the prompt; same idea as llama **`prompt eval time`**),
//!   **`lm_head_sample_ms`** (final RMSNorm + LM head + greedy argmax — not a full autoregressive
//!   decode step), and **`ttft_infer_ms`** = prepare + prompt eval + head (tokenizer excluded), for
//!   apples-to-apples with llama-bench-style TTFT without tokenization.
//! - **Decode throughput** — after a warm prefill, times only the greedy decode loop (`n` new tokens).

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::EngineError;
use crate::layers::attention::kv_caches_for_config;
use crate::loaded_model::LoadedModel;
use crate::model_config::{ModelConfig, TokenizerPromptConfig};
use crate::model_loader::file_loader::read_file;
use crate::model_weights::ModelWeightNames;
use crate::engine::embed::prefill_from_tokens_loaded;
use crate::engine::session::InferenceSession;
use crate::tokenizer::Tokenizer;

/// Default prompt (matches `tests/generate_smoke.rs`).
pub const DEFAULT_BENCH_PROMPT: &str = "Rust will rule the";

#[inline]
fn ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1e3
}

#[derive(Debug, Clone, Serialize)]
pub struct ColdStartMetrics {
    pub prompt_token_count: usize,
    pub gguf_metadata_ms: f64,
    pub tokenizer_load_ms: f64,
    pub tokenizer_encode_ms: f64,
    pub config_and_resolve_ms: f64,
    pub tensor_load_ms: f64,
    /// Embedding gather + build prefill input (not the transformer stack).
    pub prefill_prepare_ms: f64,
    pub weights_build_ms: f64,
    pub kv_alloc_ms: f64,
    /// Full forward over prompt positions (llama **`prompt eval time`** analog).
    pub prompt_eval_ms: f64,
    /// Output norm + LM head + greedy sample on last prefill hidden state (not a full decode step).
    pub lm_head_sample_ms: f64,
    /// Wall time from the start of `read_file` until the first greedy token id is sampled.
    pub cold_ttft_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InteractiveTtftMetrics {
    pub prompt_token_count: usize,
    pub tokenizer_encode_ms: f64,
    pub prefill_prepare_ms: f64,
    pub prompt_eval_ms: f64,
    pub lm_head_sample_ms: f64,
    /// `prefill_prepare_ms` + `prompt_eval_ms` + `lm_head_sample_ms` (no tokenization).
    pub ttft_infer_ms: f64,
    /// Same sum plus `tokenizer_encode_ms`.
    pub ttft_with_tokenizer_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeThroughputMetrics {
    pub prompt_token_count: usize,
    pub decode_tokens: usize,
    /// Untimed setup: prepare prefill + weights + KV + `prefill_forward` before the decode loop.
    pub warm_prefill_ms: f64,
    /// Wall time for `decode_tokens` full autoregressive steps only.
    pub decode_elapsed_ms: f64,
    pub decode_tokens_per_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AllMetrics {
    pub cold: ColdStartMetrics,
    pub interactive: InteractiveTtftMetrics,
    pub decode: DecodeThroughputMetrics,
}

/// Loaded model + tokenizer for warm benchmarks (`interactive-ttft`, `decode-throughput`, or after [`run_cold_start`]).
pub struct EngineBench {
    model: LoadedModel,
    tokenizer: Tokenizer,
}

impl EngineBench {
    /// Load GGUF metadata, tokenizer, resolve names, and [`ModelWeightNames::load_all`] (no inference).
    pub fn load(model: &Path, tokenizer_path: &Path) -> Result<Self, EngineError> {
        if !model.is_file() {
            return Err(EngineError::Model(format!(
                "model file not found: {}",
                model.display()
            )));
        }
        if !tokenizer_path.is_file() {
            return Err(EngineError::Model(format!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        let model = LoadedModel::load(model)?;
        let tokenizer = Tokenizer::load_from_file(tokenizer_path)?;

        Ok(Self { model, tokenizer })
    }

    /// Strict interactive: fresh KV; time tokenizer, prefill prep, prompt eval, LM head + sample.
    pub fn run_interactive_ttft(
        &mut self,
        prompt: &str,
    ) -> Result<InteractiveTtftMetrics, EngineError> {
        let t_enc0 = Instant::now();
        let prompt_ids = self
            .tokenizer
            .encode_with_prompt_config(prompt, self.model.tokenizer_prompt())?;
        let tokenizer_encode_ms = ms(t_enc0.elapsed());

        let t_in0 = Instant::now();
        let prefill_in =
            prefill_from_tokens_loaded(self.model.gguf(), self.model.config(), &prompt_ids)?;
        let prefill_prepare_ms = ms(t_in0.elapsed());

        let weights = self.model.weights()?;

        let t_pf0 = Instant::now();
        let kv_caches = kv_caches_for_config(self.model.config());
        let mut session = InferenceSession::from_parts(&self.model, weights, kv_caches);
        let state = session.prefill_prepared(&prefill_in)?;
        let prompt_eval_ms = ms(t_pf0.elapsed());

        let t_tok0 = Instant::now();
        let _first_id = crate::engine::generation::greedy_next_token(&session, &state)?;
        let lm_head_sample_ms = ms(t_tok0.elapsed());

        let ttft_infer_ms = prefill_prepare_ms + prompt_eval_ms + lm_head_sample_ms;
        let ttft_with_tokenizer_ms = tokenizer_encode_ms + ttft_infer_ms;

        Ok(InteractiveTtftMetrics {
            prompt_token_count: prompt_ids.len(),
            tokenizer_encode_ms,
            prefill_prepare_ms,
            prompt_eval_ms,
            lm_head_sample_ms,
            ttft_infer_ms,
            ttft_with_tokenizer_ms,
        })
    }

    /// Greedy decode throughput after a warm prefill. Timed region: `n` full decode steps
    /// (logits → sample → embedding → `decode_forward`), same as `src/main.rs`.
    pub fn run_decode_throughput(
        &mut self,
        prompt: &str,
        decode_tokens: usize,
    ) -> Result<DecodeThroughputMetrics, EngineError> {
        if decode_tokens == 0 {
            return Err(EngineError::Model(
                "decode_throughput: decode_tokens must be > 0".into(),
            ));
        }

        let prompt_ids = self
            .tokenizer
            .encode_with_prompt_config(prompt, self.model.tokenizer_prompt())?;

        let t_setup0 = Instant::now();
        let mut session = InferenceSession::new(&self.model)?;
        let mut state = session.prefill(&prompt_ids)?;
        let warm_prefill_ms = ms(t_setup0.elapsed());

        let t_dec0 = Instant::now();
        for _ in 0..decode_tokens {
            let next_id = crate::engine::generation::greedy_next_token(&session, &state)?;
            state = session.decode_token(next_id)?;
        }
        let decode_elapsed_ms = ms(t_dec0.elapsed());

        let decode_tokens_per_sec =
            (decode_tokens as f64) / (decode_elapsed_ms / 1e3).max(f64::EPSILON);

        Ok(DecodeThroughputMetrics {
            prompt_token_count: prompt_ids.len(),
            decode_tokens,
            warm_prefill_ms,
            decode_elapsed_ms,
            decode_tokens_per_sec,
        })
    }
}

/// Cold start through the first greedy token; returns metrics and a warm [`EngineBench`] handle.
pub fn run_cold_start(
    model: &Path,
    tokenizer_path: &Path,
    prompt: &str,
) -> Result<(ColdStartMetrics, EngineBench), EngineError> {
    if !model.is_file() {
        return Err(EngineError::Model(format!(
            "model file not found: {}",
            model.display()
        )));
    }
    if !tokenizer_path.is_file() {
        return Err(EngineError::Model(format!(
            "tokenizer file not found: {}",
            tokenizer_path.display()
        )));
    }

    let wall0 = Instant::now();

    let model_path = model
        .to_str()
        .ok_or_else(|| EngineError::Model("model path is not valid UTF-8".into()))?
        .to_string();

    let t0 = Instant::now();
    let mut gguf = read_file(model_path.as_str())?;
    let gguf_metadata_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let mut tokenizer = Tokenizer::load_from_file(tokenizer_path)?;
    let tokenizer_load_ms = ms(t0.elapsed());

    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf)?;

    let t0 = Instant::now();
    let prompt_ids = tokenizer.encode_with_prompt_config(prompt, &tok_prompt)?;
    let tokenizer_encode_ms = ms(t0.elapsed());
    let prompt_token_count = prompt_ids.len();

    let t0 = Instant::now();
    let config = ModelConfig::from_gguf(&gguf)?;
    let names = ModelWeightNames::resolve(&gguf, &config)?;
    let config_and_resolve_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    names.load_all(&mut gguf, model_path.as_str())?;
    let tensor_load_ms = ms(t0.elapsed());

    let model = LoadedModel::from_loaded_parts(model_path, gguf, config, names, tok_prompt);

    let t0 = Instant::now();
    let prefill_in = prefill_from_tokens_loaded(model.gguf(), model.config(), &prompt_ids)?;
    let prefill_prepare_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let weights = model.weights()?;
    let weights_build_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let kv_caches = kv_caches_for_config(model.config());
    let kv_alloc_ms = ms(t0.elapsed());

    let (prompt_eval_ms, lm_head_sample_ms) = {
        let mut session = InferenceSession::from_parts(&model, weights, kv_caches);

        let t0 = Instant::now();
        let state = session.prefill_prepared(&prefill_in)?;
        let prompt_eval_ms = ms(t0.elapsed());

        let t0 = Instant::now();
        let _first_id = crate::engine::generation::greedy_next_token(&session, &state)?;
        let lm_head_sample_ms = ms(t0.elapsed());

        (prompt_eval_ms, lm_head_sample_ms)
    };

    let cold_ttft_ms = ms(wall0.elapsed());

    let bench = EngineBench { model, tokenizer };

    let metrics = ColdStartMetrics {
        prompt_token_count,
        gguf_metadata_ms,
        tokenizer_load_ms,
        tokenizer_encode_ms,
        config_and_resolve_ms,
        tensor_load_ms,
        prefill_prepare_ms,
        weights_build_ms,
        kv_alloc_ms,
        prompt_eval_ms,
        lm_head_sample_ms,
        cold_ttft_ms,
    };

    Ok((metrics, bench))
}

pub fn run_all(
    model: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    decode_tokens: usize,
) -> Result<AllMetrics, EngineError> {
    let (cold, mut bench) = run_cold_start(model, tokenizer_path, prompt)?;
    let interactive = bench.run_interactive_ttft(prompt)?;
    let decode = bench.run_decode_throughput(prompt, decode_tokens)?;
    Ok(AllMetrics {
        cold,
        interactive,
        decode,
    })
}

/// Reference timings from `llama-completion --perf` using **CPU-fair** defaults:
/// no layer offload (`-ngl 0`), no device offload (`--device none`), no sneaking matmuls to the GPU
/// (`--no-op-offload`), single thread (`-t 1`), no empty warmup (`--no-warmup`), one generated token (`-n 1`).
///
/// `prompt_eval_ms` here is parsed from llama’s **`prompt eval time`** line (prompt batch only; no tokenization in that figure).
/// Metal may still appear in logs as a loaded backend; with these flags the perf breakdown should not use GPU memory.
#[derive(Debug, Clone, Serialize)]
pub struct LlamaCompletionTtftRef {
    pub load_ms: f64,
    pub prompt_eval_ms: f64,
    pub prompt_tokens: usize,
}

/// Run `llama_completion_bin` and parse `common_perf_print` lines from merged stdout+stderr.
pub fn run_llama_completion_ttft_ref(
    llama_completion_bin: &Path,
    model: &Path,
    prompt: &str,
) -> Result<LlamaCompletionTtftRef, EngineError> {
    use std::process::Command;

    let out = Command::new(llama_completion_bin)
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg(prompt)
        .args([
            "-t",
            "1",
            "-ngl",
            "0",
            "--device",
            "none",
            "--no-op-offload",
            "--no-warmup",
            "--perf",
            "-n",
            "1",
        ])
        .output()?;

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{stdout}\n{stderr}");

    if !out.status.success() {
        let tail = tail_utf8(&combined, 4000);
        return Err(EngineError::Model(format!(
            "llama-completion exited with {}; last bytes of output:\n{tail}",
            out.status
        )));
    }

    parse_llama_completion_perf_text(&combined).ok_or_else(|| {
        let tail = tail_utf8(&combined, 2000);
        EngineError::Model(format!(
            "could not parse llama-completion --perf (expected `load time` + `prompt eval time` lines); tail:\n{tail}"
        ))
    })
}

fn tail_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        let skip = s.len() - max_bytes;
        let start = s
            .char_indices()
            .find(|(i, _)| *i >= skip)
            .map(|(i, _)| i)
            .unwrap_or(skip);
        &s[start..]
    }
}

fn parse_llama_completion_perf_text(text: &str) -> Option<LlamaCompletionTtftRef> {
    let mut load_ms = None;
    let mut prompt_eval_ms = None;
    let mut prompt_tokens = None;

    for line in text.lines() {
        if line.contains("prompt eval time =") {
            prompt_eval_ms = parse_ms_after_key(line, "prompt eval time =");
            prompt_tokens = parse_prompt_tokens_after_slash(line);
        } else if line.contains("common_perf_print:")
            && line.contains("load time =")
            && !line.contains("prompt eval")
        {
            load_ms = parse_ms_after_key(line, "load time =").or(load_ms);
        }
    }

    Some(LlamaCompletionTtftRef {
        load_ms: load_ms?,
        prompt_eval_ms: prompt_eval_ms?,
        prompt_tokens: prompt_tokens?,
    })
}

fn parse_ms_after_key(line: &str, key: &str) -> Option<f64> {
    let idx = line.find(key)?;
    let rest = line[idx + key.len()..].trim_start();
    let end = rest.find(" ms")?;
    rest[..end].trim().parse().ok()
}

fn parse_prompt_tokens_after_slash(line: &str) -> Option<usize> {
    let slash = line.find('/')?;
    let after = line[slash + 1..].trim();
    after.split_whitespace().next()?.parse().ok()
}

#[cfg(test)]
mod llama_perf_parse_tests {
    use super::*;

    #[test]
    fn parses_homebrew_style_perf_block() {
        let sample = r#"
common_perf_print:        load time =    1610.20 ms
common_perf_print: prompt eval time =    1609.65 ms /     6 tokens (  268.28 ms per token,     3.73 tokens per second)
"#;
        let p = parse_llama_completion_perf_text(sample).expect("parse");
        assert!((p.load_ms - 1610.20).abs() < 0.01);
        assert!((p.prompt_eval_ms - 1609.65).abs() < 0.01);
        assert_eq!(p.prompt_tokens, 6);
    }
}
