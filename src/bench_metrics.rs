//! Wall-clock timings for comparing this engine to llama-bench / llama.cpp-style metrics.
//!
//! - **Cold start** — from [`crate::model_loader::file_loader::read_file`] through the first
//!   greedy token after prefill (includes tokenizer load, weight load, prefill, first decode).
//! - **Interactive TTFT** — weights already resident; fresh KV cache; measures tokenizer encode,
//!   prefill input gather, prefill forward, and first decode step. Also reports
//!   `ttft_infer_ms` (encode excluded) to line up with llama-bench, which omits tokenization.
//! - **Decode throughput** — after a warm prefill, times only the greedy decode loop (`n` new tokens).

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::layers::attention::KVCache;
use crate::layers::embeddings::lookup_embeddings_loaded;
use crate::model_config::{ModelConfig, TokenizerPromptConfig};
use crate::model_loader::file_loader::read_file;
use crate::model_loader::gguf_types::GGUFData;
use crate::model_weights::{ModelWeightNames, ModelWeights};
use crate::prefill::{
    decode_forward, final_logits_last_token, prefill_forward, prefill_from_tokens, PrefillState,
};
use crate::sampling::sample_greedy;
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
    pub encode_ms: f64,
    pub config_and_resolve_ms: f64,
    pub tensor_load_ms: f64,
    pub prefill_input_ms: f64,
    pub weights_build_ms: f64,
    pub kv_alloc_ms: f64,
    pub prefill_forward_ms: f64,
    pub first_token_ms: f64,
    /// Wall time from the start of `read_file` until the first greedy token id is sampled.
    pub cold_ttft_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InteractiveTtftMetrics {
    pub prompt_token_count: usize,
    pub encode_ms: f64,
    pub prefill_input_ms: f64,
    pub prefill_forward_ms: f64,
    pub first_token_ms: f64,
    /// Prefill input + forward + first decode (llama-bench-style: no tokenization).
    pub ttft_infer_ms: f64,
    /// Includes tokenizer encode.
    pub ttft_with_tokenizer_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeThroughputMetrics {
    pub prompt_token_count: usize,
    pub decode_tokens: usize,
    pub prefill_setup_ms: f64,
    pub decode_wall_ms: f64,
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
    model_path: String,
    gguf: GGUFData,
    names: ModelWeightNames,
    tokenizer: Tokenizer,
    tok_prompt: TokenizerPromptConfig,
    config: ModelConfig,
}

impl EngineBench {
    /// Load GGUF metadata, tokenizer, resolve names, and [`ModelWeightNames::load_all`] (no inference).
    pub fn load(model: &Path, tokenizer_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !model.is_file() {
            return Err(format!("model file not found: {}", model.display()).into());
        }
        if !tokenizer_path.is_file() {
            return Err(format!("tokenizer file not found: {}", tokenizer_path.display()).into());
        }

        let model_path = model
            .to_str()
            .ok_or("model path is not valid UTF-8")?
            .to_string();

        let mut gguf = read_file(model_path.as_str())?;
        let tokenizer = Tokenizer::load_from_file(tokenizer_path)?;
        let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let names = ModelWeightNames::resolve(&gguf, &config)?;
        names.load_all(&mut gguf, model_path.as_str())?;

        Ok(Self {
            model_path,
            gguf,
            names,
            tokenizer,
            tok_prompt,
            config,
        })
    }

    /// Strict interactive: fresh KV; time encode, prefill, first greedy token.
    pub fn run_interactive_ttft(&mut self, prompt: &str) -> Result<InteractiveTtftMetrics, Box<dyn std::error::Error>> {
        let t_enc0 = Instant::now();
        let prompt_ids = self
            .tokenizer
            .encode_with_prompt_config(prompt, &self.tok_prompt)?;
        let encode_ms = ms(t_enc0.elapsed());

        let t_in0 = Instant::now();
        let prefill_in = prefill_from_tokens(
            &mut self.gguf,
            self.model_path.as_str(),
            &self.config,
            &prompt_ids,
        )?;
        let prefill_input_ms = ms(t_in0.elapsed());

        let weights = ModelWeights::from_loaded(&self.gguf, &self.names)?;

        let t_pf0 = Instant::now();
        let mut kv_caches: Vec<KVCache> = (0..self.config.n_layers)
            .map(|_| {
                KVCache::new(
                    self.config.context_length,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                )
            })
            .collect();
        let state = prefill_forward(&prefill_in, &self.config, &weights, &mut kv_caches)?;
        let prefill_forward_ms = ms(t_pf0.elapsed());

        let t_tok0 = Instant::now();
        let logits = final_logits_last_token(&state, &self.config, &weights)?;
        let _first_id = sample_greedy(&logits)?;
        let first_token_ms = ms(t_tok0.elapsed());

        let ttft_infer_ms = prefill_input_ms + prefill_forward_ms + first_token_ms;
        let ttft_with_tokenizer_ms = encode_ms + ttft_infer_ms;

        Ok(InteractiveTtftMetrics {
            prompt_token_count: prompt_ids.len(),
            encode_ms,
            prefill_input_ms,
            prefill_forward_ms,
            first_token_ms,
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
    ) -> Result<DecodeThroughputMetrics, Box<dyn std::error::Error>> {
        if decode_tokens == 0 {
            return Err("decode_throughput: decode_tokens must be > 0".into());
        }

        let prompt_ids = self
            .tokenizer
            .encode_with_prompt_config(prompt, &self.tok_prompt)?;

        let t_setup0 = Instant::now();
        let prefill_in = prefill_from_tokens(
            &mut self.gguf,
            self.model_path.as_str(),
            &self.config,
            &prompt_ids,
        )?;
        let weights = ModelWeights::from_loaded(&self.gguf, &self.names)?;
        let mut kv_caches: Vec<KVCache> = (0..self.config.n_layers)
            .map(|_| {
                KVCache::new(
                    self.config.context_length,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                )
            })
            .collect();
        let mut state = prefill_forward(&prefill_in, &self.config, &weights, &mut kv_caches)?;
        let prefill_setup_ms = ms(t_setup0.elapsed());

        let t_dec0 = Instant::now();
        for _ in 0..decode_tokens {
            let logits = final_logits_last_token(&state, &self.config, &weights)?;
            let next_id = sample_greedy(&logits)?;
            let rows = lookup_embeddings_loaded(&self.gguf, &[next_id])?;
            let step_in = PrefillState::from_embeddings(rows, self.config.hidden_dim)?;
            state = decode_forward(&step_in, &self.config, &weights, &mut kv_caches)?;
        }
        let decode_wall_ms = ms(t_dec0.elapsed());

        let decode_tokens_per_sec = (decode_tokens as f64) / (decode_wall_ms / 1e3).max(f64::EPSILON);

        Ok(DecodeThroughputMetrics {
            prompt_token_count: prompt_ids.len(),
            decode_tokens,
            prefill_setup_ms,
            decode_wall_ms,
            decode_tokens_per_sec,
        })
    }
}

/// Cold start through the first greedy token; returns metrics and a warm [`EngineBench`] handle.
pub fn run_cold_start(
    model: &Path,
    tokenizer_path: &Path,
    prompt: &str,
) -> Result<(ColdStartMetrics, EngineBench), Box<dyn std::error::Error>> {
    if !model.is_file() {
        return Err(format!("model file not found: {}", model.display()).into());
    }
    if !tokenizer_path.is_file() {
        return Err(format!("tokenizer file not found: {}", tokenizer_path.display()).into());
    }

    let wall0 = Instant::now();

    let model_path = model
        .to_str()
        .ok_or("model path is not valid UTF-8")?
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
    let encode_ms = ms(t0.elapsed());
    let prompt_token_count = prompt_ids.len();

    let t0 = Instant::now();
    let config = ModelConfig::from_gguf(&gguf)?;
    let names = ModelWeightNames::resolve(&gguf, &config)?;
    let config_and_resolve_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    names.load_all(&mut gguf, model_path.as_str())?;
    let tensor_load_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let prefill_in = prefill_from_tokens(&mut gguf, model_path.as_str(), &config, &prompt_ids)?;
    let prefill_input_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let weights = ModelWeights::from_loaded(&gguf, &names)?;
    let weights_build_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let mut kv_caches: Vec<KVCache> = (0..config.n_layers)
        .map(|_| KVCache::new(config.context_length, config.n_kv_heads, config.head_dim))
        .collect();
    let kv_alloc_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches)?;
    let prefill_forward_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let logits = final_logits_last_token(&state, &config, &weights)?;
    let _first_id = sample_greedy(&logits)?;
    let first_token_ms = ms(t0.elapsed());

    let cold_ttft_ms = ms(wall0.elapsed());

    let bench = EngineBench {
        model_path,
        gguf,
        names,
        tokenizer,
        tok_prompt,
        config,
    };

    let metrics = ColdStartMetrics {
        prompt_token_count,
        gguf_metadata_ms,
        tokenizer_load_ms,
        encode_ms,
        config_and_resolve_ms,
        tensor_load_ms,
        prefill_input_ms,
        weights_build_ms,
        kv_alloc_ms,
        prefill_forward_ms,
        first_token_ms,
        cold_ttft_ms,
    };

    Ok((metrics, bench))
}

pub fn run_all(
    model: &Path,
    tokenizer_path: &Path,
    prompt: &str,
    decode_tokens: usize,
) -> Result<AllMetrics, Box<dyn std::error::Error>> {
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
/// `prompt_eval_ms` matches the README / llama-bench notion of TTFT without tokenization (prompt batch only).
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
) -> Result<LlamaCompletionTtftRef, Box<dyn std::error::Error>> {
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
        return Err(format!(
            "llama-completion exited with {}; last bytes of output:\n{tail}",
            out.status
        )
        .into());
    }

    parse_llama_completion_perf_text(&combined).ok_or_else(|| {
        let tail = tail_utf8(&combined, 2000);
        format!("could not parse llama-completion --perf (expected `load time` + `prompt eval time` lines); tail:\n{tail}")
            .into()
    })
}

fn tail_utf8(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        s
    } else {
        let skip = s.len() - max_bytes;
        let start = s.char_indices().find(|(i, _)| *i >= skip).map(|(i, _)| i).unwrap_or(skip);
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
