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
//! This binary includes tokenization unless noted (`ttft_infer_ms` excludes encode).

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use inference_engine_rust::EngineError;
use inference_engine_rust::bench_metrics::{
    run_all, run_cold_start, run_llama_completion_ttft_ref, ColdStartMetrics, DecodeThroughputMetrics,
    EngineBench, InteractiveTtftMetrics, LlamaCompletionTtftRef, DEFAULT_BENCH_PROMPT,
};

#[derive(Parser, Debug)]
#[command(name = "bench_compare")]
#[command(about = "Latency/throughput metrics for the inference engine (vs llama-bench-style comparisons)")]
struct Cli {
    /// GGUF model path
    #[arg(short, long, default_value = "model/mistral-7b-v0.1.Q4_K_M.gguf")]
    model: PathBuf,

    /// SentencePiece `tokenizer.model`
    #[arg(short, long, default_value = "tokenizer.model")]
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
    /// Weights loaded; fresh KV; time encode + prefill + first token (`ttft_infer_ms` excludes encode)
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
    finite_pos_ms(m.first_token_ms, "cold.first_token_ms")?;
    if m.prompt_token_count == 0 {
        return Err(EngineError::Model(
            "cold.prompt_token_count is zero".into(),
        ));
    }
    Ok(())
}

fn check_interactive(m: &InteractiveTtftMetrics) -> Result<(), EngineError> {
    finite_pos_ms(m.ttft_infer_ms, "interactive.ttft_infer_ms")?;
    finite_pos_ms(m.ttft_with_tokenizer_ms, "interactive.ttft_with_tokenizer_ms")?;
    if m.prompt_token_count == 0 {
        return Err(EngineError::Model(
            "interactive.prompt_token_count is zero".into(),
        ));
    }
    Ok(())
}

fn check_decode(
    m: &DecodeThroughputMetrics,
    min_tps: Option<f64>,
) -> Result<(), EngineError> {
    finite_pos_ms(m.decode_wall_ms, "decode.decode_wall_ms")?;
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
    println!("  encode_ms:          {:.3}", m.encode_ms);
    println!("  config_resolve_ms:  {:.3}", m.config_and_resolve_ms);
    println!("  tensor_load_ms:     {:.3}", m.tensor_load_ms);
    println!("  prefill_input_ms:   {:.3}", m.prefill_input_ms);
    println!("  weights_build_ms:   {:.3}", m.weights_build_ms);
    println!("  kv_alloc_ms:        {:.3}", m.kv_alloc_ms);
    println!("  prefill_forward_ms: {:.3}", m.prefill_forward_ms);
    println!("  first_token_ms:     {:.3}", m.first_token_ms);
    println!("  cold_ttft_ms:       {:.3}", m.cold_ttft_ms);
}

fn print_interactive_human(m: &InteractiveTtftMetrics, llama: Option<&LlamaCompletionTtftRef>) {
    println!("=== interactive-ttft (warm weights, fresh KV) ===");
    println!("  prompt_token_count:        {}", m.prompt_token_count);
    println!("  encode_ms:                 {:.3}", m.encode_ms);
    println!("  prefill_input_ms:          {:.3}", m.prefill_input_ms);
    println!("  prefill_forward_ms:        {:.3}", m.prefill_forward_ms);
    println!("  first_token_ms:            {:.3}", m.first_token_ms);
    println!("  ttft_infer_ms:             {:.3}  (no tokenization; closer to llama-bench)", m.ttft_infer_ms);
    println!("  ttft_with_tokenizer_ms:    {:.3}", m.ttft_with_tokenizer_ms);
    if let Some(l) = llama {
        println!();
        println!("=== llama-completion --perf (reference, CPU-fair: -ngl 0, --device none, --no-op-offload, -t 1) ===");
        println!("  load_ms (model): {:.3}", l.load_ms);
        println!("  prompt_eval_ms:            {:.3}  (TTFT infer analog)", l.prompt_eval_ms);
        println!("  prompt_token_count:        {}", l.prompt_tokens);
        if l.prompt_tokens != m.prompt_token_count {
            eprintln!(
                "warning: llama prompt token count {} != Rust {}; check BOS handling / tokenizer alignment",
                l.prompt_tokens, m.prompt_token_count
            );
        }
        println!(
            "  ratio ttft_infer rust/llama: {:.2}",
            m.ttft_infer_ms / l.prompt_eval_ms
        );
    }
}

fn print_decode_human(m: &DecodeThroughputMetrics) {
    println!("=== decode-throughput ===");
    println!("  prompt_token_count:   {}", m.prompt_token_count);
    println!("  decode_tokens:        {}", m.decode_tokens);
    println!("  prefill_setup_ms:     {:.3}", m.prefill_setup_ms);
    println!("  decode_wall_ms:       {:.3}", m.decode_wall_ms);
    println!("  decode_tokens_per_sec:{:.3}", m.decode_tokens_per_sec);
    let pps = (m.prompt_token_count as f64) / (m.prefill_setup_ms / 1e3).max(f64::EPSILON);
    println!("  prefill_tokens_per_s: {:.3}  (rough pp analog)", pps);
}

fn main() -> Result<(), EngineError> {
    let cli = Cli::parse();

    if cli.compare_llama && !matches!(cli.command, Commands::InteractiveTtft) {
        eprintln!("warning: --compare-llama only runs with the interactive-ttft subcommand; ignoring");
    }

    match cli.command {
        Commands::All { decode_tokens } => {
            let m = run_all(&cli.model, &cli.tokenizer, &cli.prompt, decode_tokens)?;
            check_cold(&m.cold)?;
            check_interactive(&m.interactive)?;
            check_decode(&m.decode, cli.min_decode_tps)?;
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "all",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "metrics": m,
                    })
                );
            } else {
                               print_cold_human(&m.cold);
                println!();
                print_interactive_human(&m.interactive, None);
                println!();
                print_decode_human(&m.decode);
            }
        }
        Commands::ColdStart => {
            let (cold, _bench) = run_cold_start(&cli.model, &cli.tokenizer, &cli.prompt)?;
            check_cold(&cold)?;
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "cold-start",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "metrics": cold,
                    })
                );
            } else {
                print_cold_human(&cold);
            }
            if cli.min_decode_tps.is_some() {
                eprintln!("warning: --min-decode-tps ignored for cold-start");
            }
        }
        Commands::InteractiveTtft => {
            let mut bench = EngineBench::load(&cli.model, &cli.tokenizer)?;
            let m = bench.run_interactive_ttft(&cli.prompt)?;
            check_interactive(&m)?;
            let llama = if cli.compare_llama {
                Some(run_llama_completion_ttft_ref(
                    &cli.llama_completion_bin,
                    &cli.model,
                    &cli.prompt,
                )?)
            } else {
                None
            };
            if cli.json {
                let ratio = llama
                    .as_ref()
                    .map(|l| m.ttft_infer_ms / l.prompt_eval_ms);
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "interactive-ttft",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "metrics": m,
                        "llama_completion": llama,
                        "ratio_ttft_infer": ratio,
                    })
                );
            } else {
                print_interactive_human(&m, llama.as_ref());
            }
            if cli.min_decode_tps.is_some() {
                eprintln!("warning: --min-decode-tps ignored for interactive-ttft");
            }
        }
        Commands::DecodeThroughput { decode_tokens } => {
            let mut bench = EngineBench::load(&cli.model, &cli.tokenizer)?;
            let m = bench.run_decode_throughput(&cli.prompt, decode_tokens)?;
            check_decode(&m, cli.min_decode_tps)?;
            if cli.json {
                println!(
                    "{}",
                    serde_json::json!({
                        "suite": "decode-throughput",
                        "model": cli.model,
                        "tokenizer": cli.tokenizer,
                        "prompt": cli.prompt,
                        "metrics": m,
                    })
                );
            } else {
                print_decode_human(&m);
            }
        }
    }

    Ok(())
}
