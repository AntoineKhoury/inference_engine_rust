# inference_engine_rust

Inference engine for LLMs in the **GGUF** format (Rust).

## Requirements

- **Rust:** **1.85 or newer** stable. This crate uses **`edition = "2024"`**, which requires rustc **1.85+**. The declared MSRV is in **`Cargo.toml`** as **`rust-version`** (Cargo will warn if your toolchain is too old).
- **Optional (heavier tests & benchmarks):** per-model directories under **`model/`** (GGUF + tokenizer) — see [`model/README.md`](model/README.md) and [`tests/common/mod.rs`](tests/common/mod.rs).

## Quick test (no model)

Runs unit tests and fast integration checks:

```bash
cargo test
```

This exercises loaders, ops, tokenizer, embeddings, and the [`bench_compare`](src/bin/bench_compare.rs) binary (CLI `--help` only). Several integration tests are **`#[ignore]`** until you add weights; they still compile and show up as ignored.

## Correctness vs reference (optional)

| What | How |
|------|-----|
| **Embedding row vs hardcoded floats** | Place `model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf`, then `cargo test embedding_token2_matches_gguf_reference -- --ignored --nocapture` |
| **Download that GGUF (~4 GB)** | `DOWNLOAD_REFERENCE_GGUF=1 cargo test download_reference_gguf -- --ignored` (needs `curl`) |
| **Tokenizer file** | Mistral: `model/mistral-7b-v0.1/tokenizer.model`. Gemma 4: `model/gemma-4-e2b-it/tokenizer.json` from [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) (see [`model/README.md`](model/README.md)) |
| **Greedy generation smoke** | `cargo test --test generate_smoke greedy_generate_continuation_after_prompt --release -- --ignored --nocapture` |
| **Logits vs llama.cpp** | Build [`tools/llama_logits_ref`](tools/llama_logits_ref.c) via [`tools/build_llama_logits_ref.sh`](tools/build_llama_logits_ref.sh), then run the ignored test in [`tests/logits_vs_llama.rs`](tests/logits_vs_llama.rs) |
| **Hidden vs llama.cpp** | Same script builds `llama_hidden_ref`; see [`tests/hidden_vs_llama.rs`](tests/hidden_vs_llama.rs) |

Details and flags (CPU vs GPU reference, env overrides) are in those test modules and [`LEARNINGS_SYSTEM.md`](LEARNINGS_SYSTEM.md).

## Performance metrics (`bench_compare`)

Compare **your** wall-clock numbers with **llama-bench** on the **same GGUF** (same machine, similar `-t` / `-ngl` / mmap settings on their side). With **`--compare-llama`**, the same binary also runs **`llama-completion --perf`** for a reference column.

```bash
# Full suite in one process: cold TTFT → warm interactive TTFT → decode throughput
cargo run --release --bin bench_compare -- all --decode-tokens 128

# One metric at a time
cargo run --release --bin bench_compare -- cold-start
cargo run --release --bin bench_compare -- interactive-ttft
cargo run --release --bin bench_compare -- decode-throughput -n 128

# Repeat each benchmark (default is --runs 5; minimum allowed is 5)
cargo run --release --bin bench_compare -- --runs 5 interactive-ttft

# vs llama.cpp on the same GGUF / prompt (global flags work after the subcommand too)
cargo run --release --bin bench_compare -- --compare-llama interactive-ttft

# Fast iteration on Gemma, then confirm on Mistral before logging README history
cargo run --release --bin bench_compare -- --runs 5 -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json interactive-ttft
cargo run --release --bin bench_compare -- --runs 5 --compare-llama interactive-ttft

# JSON + fail if decode tok/s is below a floor (useful for CI)
cargo run --release --bin bench_compare -- --runs 5 --json --min-decode-tps 5.0 all
```

Shared flags: **`-m`** / **`--model`**, **`-t`** / **`--tokenizer`**, **`--prompt`**. Field meanings are in [`src/bench_metrics.rs`](src/bench_metrics.rs) and [`LEARNINGS_SYSTEM.md`](LEARNINGS_SYSTEM.md).

**After a run you care about:** add a row to the **Benchmark history** table below (agents: see **`.cursor/rules/benchmark-experiments.mdc`**).

## Benchmark history (Rust vs llama.cpp)

**How to read:** each row is one experiment. **Newest is at the top.** **`delta_vs_previous`** describes what changed vs the row **immediately below** (the earlier point in time). That gives you “before that change I was at …, after I’m at …” by comparing consecutive rows.

| Column | Meaning |
|--------|---------|
| **ratio_ttft** | `rust_ttft_infer_ms / llama_ttft_infer_ms` (higher = slower Rust vs llama on that run). |
| **speed_target** | What you were trying to hit on that iteration. |

**llama.cpp reference flags** (used by `--compare-llama`): `-t 1 -ngl 0 --device none --no-op-offload --no-warmup --perf -n 1`. This means: single thread, no GPU layer offload, no device offload, no op-offload, no warmup pass, perf output, generate 1 token only. Ensures a CPU-only apples-to-apples comparison with Rust (which is also single-threaded, CPU-only).

*Prepend new data rows after the `|------|` separator line; do not delete older rows (append-only).*

| date | exp | machine_id | commit | delta_vs_previous | bench_command | prompt_toks | rust_ttft_infer_ms | llama_ttft_infer_ms | ratio_ttft | rust_decode_tps | llama_decode_tps | llama_t | llama_ngl | speed_target | notes |
|------|-----|------------|--------|-------------------|---------------|-------------|--------------------|--------------------|------------|-----------------|------------------|---------|-----------|--------------|-------|
| 2026-04-29 | exp-004 | ak-mbp-m1 | 09c184c | enabled Rayon parallel matmul + attention-head loops; switched benchmark target back to Mistral from Gemma row below | `bench_compare --json --compare-llama interactive-ttft` + `decode-throughput --decode-tokens 5` | 6 | 24510.4 | 1760.9 | 13.9 | 0.060 | — | 1 | 0 | reduce Mistral TTFT while preserving correctness | Mistral 7B Q4_K_M. TTFT improved strongly vs earlier Mistral baseline (exp-002: 74533.9 -> 24510.4 ms), but short decode throughput regressed (0.083 -> 0.060 tok/s), likely from decode-side parallel overhead at small workloads. llama load 1761 ms excluded. |
| 2026-04-29 | exp-003 | ak-mbp-m1 | ec077b7 | new model baseline (Gemma 4 E2B Q8) | `bench_compare --json -m gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t gemma-4-e2b-it/tokenizer.json interactive-ttft` + `decode-throughput --decode-tokens 5` | 5 | 16882.4 | — | — | 0.322 | — | — | — | establish Gemma decode+TTFT baseline | Gemma 4 E2B IT Q8_0 (2.6B). prompt_eval 15554 ms, lm_head_sample 1087 ms, warm_prefill 14917 ms. Default prompt `Rust will rule the`. |
| 2026-04-29 | exp-002 | ak-mbp-m1 | ec077b7 | baseline re-measure | `bench_compare --json --compare-llama interactive-ttft` + `decode-throughput --decode-tokens 5` | 6 | 74533.9 | 2833.4 | 26.3 | 0.083 | — | 1 | 0 | re-establish Mistral TTFT+decode baseline | Mistral 7B Q4_K_M. llama flags: `-t 1 -ngl 0 --device none --no-op-offload --no-warmup --perf -n 1` (CPU-only, single-thread, no GPU). prompt_eval 74288 ms, lm_head_sample 246 ms. llama load 2840 ms excluded. Decode: 5 toks in 60188 ms. Default prompt `Rust will rule the`. |

### System profile (same machine → same `machine_id`)

Update when hardware or llama install changes.

| machine_id | machine label | CPU / SoC | RAM | OS | llama.cpp install | notes |
|------------|---------------|-----------|-----|----|-------------------|-------|
| ak-mbp-m1 | MacBook Pro (M1) | Apple M1 | *(fill)* | macOS | Homebrew `llama.cpp` | Matches **exp-002/003**; llama called with CPU-fair flags (see above). |

## Greedy CLI (generation)

```bash
cargo run --release -- --help
cargo run --release -- -n 32 \
  -m model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf \
  -t model/mistral-7b-v0.1/tokenizer.model \
  "Hello"
```

## License / credits

**Code in this repository** is licensed under **MIT OR Apache-2.0** (see [`LICENSE`](LICENSE), [`LICENSE-MIT`](LICENSE-MIT), [`LICENSE-APACHE`](LICENSE-APACHE)). You may use it as a library or binary under either license.

GGUF model weights, tokenizer files, and other third-party assets keep their own terms.
