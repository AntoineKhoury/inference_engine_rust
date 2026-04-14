# inference_engine_rust

Inference engine for LLMs in the **GGUF** format (Rust).

## Requirements

- **Rust** (edition 2024 toolchain)
- **Optional (heavier tests & benchmarks):** a Mistral-7B-v0.1–style **GGUF** (e.g. `Q4_K_M`) and **`tokenizer.model`** at the repo root — see [`tests/common/mod.rs`](tests/common/mod.rs) for the pinned filename and HuggingFace URLs.

## Quick test (no model)

Runs unit tests and fast integration checks:

```bash
cargo test
```

This exercises loaders, ops, tokenizer, embeddings, and [`tests/bench_compare_cli.rs`](tests/bench_compare_cli.rs) (CLI `--help` only). Several integration tests are **`#[ignore]`** until you add weights; they still compile and show up as ignored.

## Correctness vs reference (optional)

| What | How |
|------|-----|
| **Embedding row vs hardcoded floats** | Place `model/mistral-7b-v0.1.Q4_K_M.gguf`, then `cargo test embedding_token2_matches_gguf_reference -- --ignored --nocapture` |
| **Download that GGUF (~4 GB)** | `DOWNLOAD_REFERENCE_GGUF=1 cargo test download_reference_gguf -- --ignored` (needs `curl`) |
| **Tokenizer file** | Copy or download `tokenizer.model` (see URL in [`tests/common/mod.rs`](tests/common/mod.rs)) next to the GGUF |
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

# vs llama.cpp on the same GGUF / prompt (global flags work after the subcommand too)
cargo run --release --bin bench_compare -- interactive-ttft --compare-llama

# JSON + fail if decode tok/s is below a floor (useful for CI)
cargo run --release --bin bench_compare -- all --json --min-decode-tps 5.0
```

Shared flags: **`-m`** / **`--model`**, **`-t`** / **`--tokenizer`**, **`--prompt`**. Field meanings are in [`src/bench_metrics.rs`](src/bench_metrics.rs) and [`LEARNINGS_SYSTEM.md`](LEARNINGS_SYSTEM.md).

**After a run you care about:** add a row to the **Benchmark history** table below (agents: see **`.cursor/rules/benchmark-experiments.mdc`**).

## Benchmark history (Rust vs llama.cpp)

**How to read:** each row is one experiment. **Newest is at the top.** **`delta_vs_previous`** describes what changed vs the row **immediately below** (the earlier point in time). That gives you “before that change I was at …, after I’m at …” by comparing consecutive rows.

| Column | Meaning |
|--------|---------|
| **ratio_ttft** | `rust_ttft_infer_ms / llama_ttft_infer_ms` (higher = slower Rust vs llama on that run). |
| **speed_target** | What you were trying to hit on that iteration. |

*Prepend new data rows after the `|------|` separator line; do not delete older rows (append-only).*

| date | exp | machine_id | commit | delta_vs_previous | bench_command | prompt_toks | rust_ttft_infer_ms | llama_ttft_infer_ms | ratio_ttft | rust_decode_tps | llama_decode_tps | llama_t | llama_ngl | speed_target | notes |
|------|-----|------------|--------|-------------------|---------------|-------------|--------------------|--------------------|------------|-----------------|------------------|---------|-----------|--------------|-------|
| 2026-04-14 | exp-001 | ak-mbp-m1 | ab060e0 | baseline | `bench_compare interactive-ttft --compare-llama` (crate defaults: llama `--no-warmup`, `-t 1`, `-ngl 0`) | 6 | 64411.6 | 1592.8 | 40.4 | — | — | 1 | 0 | establish TTFT infer baseline vs Homebrew llama | Rust: ~64.2 s in `prefill_forward`, ~202 ms first token; llama load ~1.59 s excluded from ttft infer. Default prompt `Rust will rule the`, Mistral 7B Q4_K_M. |

### System profile (same machine → same `machine_id`)

Update when hardware or llama install changes.

| machine_id | machine label | CPU / SoC | RAM | OS | llama.cpp install | notes |
|------------|---------------|-----------|-----|----|-------------------|-------|
| ak-mbp-m1 | MacBook Pro (Antoine) | Apple M1 | *(fill)* | macOS | Homebrew `llama.cpp` | Matches **exp-001**; adjust RAM/OS as needed. |

## Greedy CLI (generation)

```bash
cargo run --release -- --help
cargo run --release -- -n 32 -m model/mistral-7b-v0.1.Q4_K_M.gguf "Hello"
```

## License / credits

See repository files for license; GGUF models and tokenizer files are third-party.
