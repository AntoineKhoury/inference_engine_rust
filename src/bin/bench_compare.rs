//! Compare-style latency/throughput metrics for this engine (cold TTFT, interactive TTFT, decode t/s).
//!
//! ```text
//! cargo run --release --bin bench_compare -- all --decode-tokens 128
//! cargo run --release --bin bench_compare -- cold-start
//! cargo run --release --bin bench_compare -- interactive-ttft
//! cargo run --release --bin bench_compare -- decode-throughput -n 128
//! cargo run --release --bin bench_compare -- all --json
//! cargo run --release --bin bench_compare -- interactive-ttft --compare-llama
//! ```
//!
//! Map results to llama-bench: **pp** = prompt tokens / prefill wall time; **tg** = decode tok/s.
//! **`ttft_infer_ms`** = `prefill_prepare_ms` + `prompt_eval_ms` + `lm_head_sample_ms` (no tokenizer).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use inference_engine_rust::EngineError;
use inference_engine_rust::bench_metrics::{
    ColdStartMetrics, DEFAULT_BENCH_PROMPT, DecodeThroughputMetrics, EngineBench,
    InteractiveTtftMetrics, LlamaCompletionTtftRef, run_all, run_cold_start,
    run_llama_completion_ttft_ref,
};

#[derive(Parser, Debug)]
#[command(name = "bench_compare")]
#[command(
    about = "Latency/throughput metrics for the inference engine (vs llama-bench-style comparisons)"
)]
struct Cli {
    /// GGUF model path
    #[arg(
        short,
        long,
        default_value = "model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf"
    )]
    model: PathBuf,

    /// `tokenizer.model` (SPM) or `tokenizer.json` (HF)
    #[arg(short, long, default_value = "model/mistral-7b-v0.1/tokenizer.model")]
    tokenizer: PathBuf,

    /// Prompt text (token count should match what you pass to llama-bench `-p`)
    #[arg(short, long, default_value = DEFAULT_BENCH_PROMPT)]
    prompt: String,

    /// Emit JSON (one object, or wrapped `{ "suite": "all", ... }`)
    #[arg(long)]
    json: bool,

    /// Exit with failure if measured decode tok/s is strictly below this (decode / all only)
    #[arg(long)]
    min_decode_tps: Option<f64>,

    /// After Rust `interactive-ttft`, run `llama-completion --perf` with CPU-fair flags (see `bench_metrics::run_llama_completion_ttft_ref`).
    #[arg(long)]
    compare_llama: bool,

    /// Path or name of `llama-completion` on `PATH`.
    #[arg(long, default_value = "llama-completion")]
    llama_completion_bin: PathBuf,

    /// Number of benchmark repetitions per command (minimum 5).
    #[arg(long, default_value_t = 5)]
    runs: usize,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Cold start + interactive TTFT + decode throughput (single process)
    All {
        #[arg(long, default_value_t = 128)]
        decode_tokens: usize,
    },
    /// From `read_file` through first greedy token id (includes load + prefill + first logits/sample)
    ColdStart,
    /// Weights loaded; fresh KV; tokenizer + prefill prep + prompt eval + LM head + sample (`ttft_infer_ms` omits tokenizer)
    InteractiveTtft,
    /// After warm prefill, time greedy decode only (`n` full steps, same as main CLI loop)
    DecodeThroughput {
        #[arg(short = 'n', long, default_value_t = 128)]
        decode_tokens: usize,
    },
}

fn finite_pos_ms(x: f64, name: &str) -> Result<(), EngineError> {
    if !x.is_finite() {
        return Err(EngineError::Model(format!("{name} is not finite")));
    }
    if x < 0.0 {
        return Err(EngineError::Model(format!("{name} is negative")));
    }
    Ok(())
}

fn check_cold(m: &ColdStartMetrics) -> Result<(), EngineError> {
    finite_pos_ms(m.gguf_metadata_ms, "cold.gguf_metadata_ms")?;
    finite_pos_ms(m.cold_ttft_ms, "cold.cold_ttft_ms")?;
    finite_pos_ms(m.lm_head_sample_ms, "cold.lm_head_sample_ms")?;
    if m.prompt_token_count == 0 {
        return Err(EngineError::Model("cold.prompt_token_count is zero".into()));
    }
    Ok(())
}

fn check_interactive(m: &InteractiveTtftMetrics) -> Result<(), EngineError> {
    finite_pos_ms(m.ttft_infer_ms, "interactive.ttft_infer_ms")?;
    finite_pos_ms(
        m.ttft_with_tokenizer_ms,
        "interactive.ttft_with_tokenizer_ms",
    )?;
    if m.prompt_token_count == 0 {
        return Err(EngineError::Model(
            "interactive.prompt_token_count is zero".into(),
        ));
    }
    Ok(())
}

fn check_decode(m: &DecodeThroughputMetrics, min_tps: Option<f64>) -> Result<(), EngineError> {
    finite_pos_ms(m.decode_elapsed_ms, "decode.decode_elapsed_ms")?;
    if !m.decode_tokens_per_sec.is_finite() {
        return Err(EngineError::Model(
            "decode.decode_tokens_per_sec is not finite".into(),
        ));
    }
    if m.decode_tokens == 0 {
        return Err(EngineError::Model("decode.decode_tokens is zero".into()));
    }
    if let Some(min) = min_tps {
        if m.decode_tokens_per_sec < min {
            return Err(EngineError::Model(format!(
                "decode.decode_tokens_per_sec {:.3} < --min-decode-tps {:.3}",
                m.decode_tokens_per_sec, min
            )));
        }
    }
    Ok(())
}

fn print_cold_human(m: &ColdStartMetrics) {
    println!("=== cold-start ===");
    println!("  prompt_token_count: {}", m.prompt_token_count);
    println!("  gguf_metadata_ms:   {:.3}", m.gguf_metadata_ms);
    println!("  tokenizer_load_ms:  {:.3}", m.tokenizer_load_ms);
    println!("  tokenizer_encode_ms:{:.3}", m.tokenizer_encode_ms);
    println!("  config_resolve_ms:  {:.3}", m.config_and_resolve_ms);
    println!("  tensor_load_ms:     {:.3}", m.tensor_load_ms);
    println!("  prefill_prepare_ms: {:.3}", m.prefill_prepare_ms);
    println!("  weights_build_ms:   {:.3}", m.weights_build_ms);
    println!("  kv_alloc_ms:        {:.3}", m.kv_alloc_ms);
    println!(
        "  prompt_eval_ms:     {:.3}  (llama: prompt eval time)",
        m.prompt_eval_ms
    );
    println!(
        "  lm_head_sample_ms:  {:.3}  (output norm + LM head + argmax; not full decode)",
        m.lm_head_sample_ms
    );
    println!("  cold_ttft_ms:       {:.3}", m.cold_ttft_ms);
}

fn print_interactive_human(m: &InteractiveTtftMetrics, llama: Option<&LlamaCompletionTtftRef>) {
    println!("=== interactive-ttft (warm weights, fresh KV) ===");
    println!("  prompt_token_count:        {}", m.prompt_token_count);
    println!("  tokenizer_encode_ms:       {:.3}", m.tokenizer_encode_ms);
    println!("  prefill_prepare_ms:        {:.3}", m.prefill_prepare_ms);
    println!(
        "  prompt_eval_ms:            {:.3}  (llama: prompt eval time)",
        m.prompt_eval_ms
    );
    println!(
        "  lm_head_sample_ms:         {:.3}  (output norm + LM head + argmax; not full decode)",
        m.lm_head_sample_ms
    );
    println!(
        "  ttft_infer_ms:             {:.3}  (prepare + prompt_eval + lm_head_sample; no tokenizer)",
        m.ttft_infer_ms
    );
    println!(
        "  ttft_with_tokenizer_ms:    {:.3}",
        m.ttft_with_tokenizer_ms
    );
    if let Some(l) = llama {
        println!();
        println!(
            "=== llama-completion --perf (reference, CPU-fair: -ngl 0, --device none, --no-op-offload, -t 1) ==="
        );
        println!("  load_ms (model): {:.3}", l.load_ms);
        println!("  prompt_eval_ms:            {:.3}", l.prompt_eval_ms);
        println!("  prompt_token_count:        {}", l.prompt_tokens);
        if l.prompt_tokens != m.prompt_token_count {
            eprintln!(
                "warning: llama prompt token count {} != Rust {}; check BOS handling / tokenizer alignment",
                l.prompt_tokens, m.prompt_token_count
            );
        }
        println!(
            "  ratio_ttft_infer (rust/llama): {:.2}  (rust sum includes prepare+head; llama is prompt-eval only)",
            m.ttft_infer_ms / l.prompt_eval_ms
        );
    }
}

fn print_decode_human(m: &DecodeThroughputMetrics) {
    println!("=== decode-throughput ===");
    println!("  prompt_token_count:   {}", m.prompt_token_count);
    println!("  decode_tokens:        {}", m.decode_tokens);
    println!("  warm_prefill_ms:      {:.3}", m.warm_prefill_ms);
    println!("  decode_elapsed_ms:    {:.3}", m.decode_elapsed_ms);
    println!("  decode_tokens_per_sec:{:.3}", m.decode_tokens_per_sec);
    let pps = (m.prompt_token_count as f64) / (m.warm_prefill_ms / 1e3).max(f64::EPSILON);
    println!("  prefill_tokens_per_s: {pps:.3}  (rough pp analog)");
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let len = values.len();
    if len == 0 {
        return 0.0;
    }
    if len % 2 == 1 {
        values[len / 2]
    } else {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    }
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let rank = ((values.len() - 1) as f64 * p).round() as usize;
    values[rank]
}

fn summarize_interactive(
    runs: &[InteractiveTtftMetrics],
) -> Result<serde_json::Value, EngineError> {
    if runs.is_empty() {
        return Err(EngineError::Model("interactive runs are empty".into()));
    }
    let mut ttft: Vec<f64> = runs.iter().map(|m| m.ttft_infer_ms).collect();
    let mut prompt_eval: Vec<f64> = runs.iter().map(|m| m.prompt_eval_ms).collect();
    Ok(serde_json::json!({
        "runs": runs.len(),
        "ttft_infer_ms": {
            "median": median(&mut ttft),
            "min": ttft.iter().copied().fold(f64::INFINITY, f64::min),
            "max": ttft.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        },
        "prompt_eval_ms": {
            "median": median(&mut prompt_eval),
            "min": prompt_eval.iter().copied().fold(f64::INFINITY, f64::min),
            "max": prompt_eval.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }))
}

fn summarize_decode(runs: &[DecodeThroughputMetrics]) -> Result<serde_json::Value, EngineError> {
    if runs.is_empty() {
        return Err(EngineError::Model("decode runs are empty".into()));
    }
    let mut tps: Vec<f64> = runs.iter().map(|m| m.decode_tokens_per_sec).collect();
    let mut elapsed: Vec<f64> = runs.iter().map(|m| m.decode_elapsed_ms).collect();
    Ok(serde_json::json!({
        "runs": runs.len(),
        "decode_tokens_per_sec": {
            "median": median(&mut tps),
            "p95": percentile(tps.clone(), 0.95),
            "min": tps.iter().copied().fold(f64::INFINITY, f64::min),
            "max": tps.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        },
        "decode_elapsed_ms": {
            "median": median(&mut elapsed),
            "min": elapsed.iter().copied().fold(f64::INFINITY, f64::min),
            "max": elapsed.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }))
}

fn main() -> Result<(), EngineError> {
    let cli = Cli::parse();
    if cli.runs < 5 {
        return Err(EngineError::Model(
            "--runs must be at least 5 for stable comparisons".into(),
        ));
    }

    if cli.compare_llama && !matches!(cli.command, Commands::InteractiveTtft) {
        eprintln!(
            "warning: --compare-llama only runs with the interactive-ttft subcommand; ignoring"
        );
    }

    match cli.command {
        Commands::All { decode_tokens } => {
            let mut runs = Vec::with_capacity(cli.runs);
            for _ in 0..cli.runs {
                let m = run_all(&cli.model, &cli.tokenizer, &cli.prompt, decode_tokens)?;
                check_cold(&m.cold)?;
                check_interactive(&m.interactive)?;
                check_decode(&m.decode, cli.min_decode_tps)?;
                runs.push(m);
            }
            let cold_runs: Vec<ColdStartMetrics> = runs.iter().map(|m| m.cold.clone()).collect();
            let interactive_runs: Vec<InteractiveTtftMetrics> =
                runs.iter().map(|m| m.interactive.clone()).collect();
            let decode_runs: Vec<DecodeThroughputMetrics> =
                runs.iter().map(|m| m.decode.clone()).collect();
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "all",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "runs": {
                            "cold": &cold_runs,
                            "interactive": &interactive_runs,
                            "decode": &decode_runs,
                        },
                        "summary": {
                            "interactive": summarize_interactive(&runs.iter().map(|m| m.interactive.clone()).collect::<Vec<_>>())?,
                            "decode": summarize_decode(&runs.iter().map(|m| m.decode.clone()).collect::<Vec<_>>())?,
                        }
                    })
                );
            } else {
                let first = &runs[0];
                print_cold_human(&first.cold);
                println!();
                print_interactive_human(&first.interactive, None);
                println!();
                print_decode_human(&first.decode);
                let summary_i = summarize_interactive(&interactive_runs)?;
                let summary_d = summarize_decode(&decode_runs)?;
                println!();
                println!("=== summary over {} runs ===", cli.runs);
                println!("  interactive: {}", summary_i);
                println!("  decode:      {}", summary_d);
            }
        }
        Commands::ColdStart => {
            let mut runs = Vec::with_capacity(cli.runs);
            for _ in 0..cli.runs {
                let (cold, _bench) = run_cold_start(&cli.model, &cli.tokenizer, &cli.prompt)?;
                check_cold(&cold)?;
                runs.push(cold);
            }
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "cold-start",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "runs": &runs,
                    })
                );
            } else {
                print_cold_human(&runs[0]);
                println!("  (showing run 1 of {}; use --json for all runs)", cli.runs);
            }
            if cli.min_decode_tps.is_some() {
                eprintln!("warning: --min-decode-tps ignored for cold-start");
            }
        }
        Commands::InteractiveTtft => {
            let mut runs = Vec::with_capacity(cli.runs);
            let mut llama_runs: Vec<LlamaCompletionTtftRef> = Vec::new();
            let mut ratio_runs: Vec<f64> = Vec::new();
            for _ in 0..cli.runs {
                let mut bench = EngineBench::load(&cli.model, &cli.tokenizer)?;
                let m = bench.run_interactive_ttft(&cli.prompt)?;
                check_interactive(&m)?;
                if cli.compare_llama {
                    let l = run_llama_completion_ttft_ref(
                        &cli.llama_completion_bin,
                        &cli.model,
                        &cli.prompt,
                    )?;
                    ratio_runs.push(m.ttft_infer_ms / l.prompt_eval_ms);
                    llama_runs.push(l);
                }
                runs.push(m);
            }
            if cli.json {
                let ratio_summary = if ratio_runs.is_empty() {
                    None
                } else {
                    let mut r = ratio_runs.clone();
                    Some(serde_json::json!({
                        "median": median(&mut r),
                        "min": r.iter().copied().fold(f64::INFINITY, f64::min),
                        "max": r.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    }))
                };
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "interactive-ttft",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "runs": &runs,
                        "summary": summarize_interactive(&runs)?,
                        "llama_completion_runs": &llama_runs,
                        "ratio_ttft_infer_summary": ratio_summary,
                    })
                );
            } else {
                print_interactive_human(&runs[0], llama_runs.first());
                let summary = summarize_interactive(&runs)?;
                println!();
                println!("=== summary over {} runs ===", cli.runs);
                println!("  interactive: {}", summary);
                if !ratio_runs.is_empty() {
                    let mut r = ratio_runs.clone();
                    println!(
                        "  ratio_ttft_infer (rust/llama): median {:.2}, min {:.2}, max {:.2}",
                        median(&mut r),
                        r.iter().copied().fold(f64::INFINITY, f64::min),
                        r.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    );
                }
            }
            if cli.min_decode_tps.is_some() {
                eprintln!("warning: --min-decode-tps ignored for interactive-ttft");
            }
        }
        Commands::DecodeThroughput { decode_tokens } => {
            let mut runs = Vec::with_capacity(cli.runs);
            for _ in 0..cli.runs {
                let mut bench = EngineBench::load(&cli.model, &cli.tokenizer)?;
                let m = bench.run_decode_throughput(&cli.prompt, decode_tokens)?;
                check_decode(&m, cli.min_decode_tps)?;
                runs.push(m);
            }
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "decode-throughput",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "runs": &runs,
                        "summary": summarize_decode(&runs)?,
                    })
                );
            } else {
                print_decode_human(&runs[0]);
                let summary = summarize_decode(&runs)?;
                println!();
                println!("=== summary over {} runs ===", cli.runs);
                println!("  decode: {}", summary);
            }
        }
    }

    Ok(())
}
