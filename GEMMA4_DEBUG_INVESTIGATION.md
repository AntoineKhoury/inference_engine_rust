# Gemma 4 Garbage Output Investigation

## Goal

Find why Gemma 4 chat output is garbled (repeated markers/special tokens, poor continuation quality), by matching Rust inference semantics against `llama.cpp` node-by-node.

---

## Symptom

- Chat generation for Gemma 4 E2B (`gemma-4-E2B-it-Q8_0.gguf`) produced low-quality/garbled output.
- Typical artifacts included repeated formatting tokens and unstable continuation behavior.
- Full-stack parity tests vs `llama.cpp` showed large drift in hidden/logits and occasional greedy argmax mismatch.

---

## Investigation Strategy

1. Build direct Rust vs `llama.cpp` references for logits/hidden states.
2. Add layer/node bisection tooling (`llama_tensor_dump_ref` + Rust stage probes).
3. Compare intermediate tensors across attention/FFN/tail pipeline.
4. Promote hypotheses to explicit A/B tests (env toggles + focused tests).
5. Rule out confirmed-equal kernels/fields and move upstream/downstream accordingly.

---

## Tooling Added During Investigation

- `tests/gemma_logits_vs_llama.rs` and `tests/gemma_hidden_vs_llama.rs`
  - Direct parity checks for final logits and post-norm hidden.
- `tools/llama_tensor_dump_ref` + node trace flow (`/tmp/gemma_nodes.tsv`)
  - Node dump used as ground truth for graph bisection.
- `tests/common/llama_tensor_dump_helpers.rs`
  - LMTD parsing and runner helpers.
- `tests/gemma_tensor_node_vs_llama.rs`
  - Main stage-level comparer between Rust and specific llama node IDs.
  - Added many `GEMMA_RUST_STAGE` targets (`attn_core`, FFN stages, tail stages, etc.).
  - Added rank-3 tensor extraction support for nodes like `Qcur_pos` / `Kcur_pos`.
- `src/layers/prefill_block.rs`
  - `gemma4_prefill_layer_debug` exposing internal per-stage tensors.
- `src/layers/attention.rs`
  - Attention score diagnostics (`INFERENCE_ENGINE_DEBUG_ATTN_LAYERS`).
  - Optional Gemma attention scaling override for A/B.
  - Focused Q/K probes:
    - `prefill_qk_raw_layer`
    - `prefill_qk_normed_layer`
    - `prefill_qk_after_rope_layer`
    - `prefill_attention_core_layer` (pre-`wo`, i.e. `kqv_out-*`)

---

## Chronological Findings (What Was Tested)

## 1) Prompt/template correctness and decode stop handling

- Fixed Gemma chat turn rendering to better match template structure.
- Added stop-marker handling for template/multimodal control tokens.
- Helped UX but did not fully solve numeric drift.

## 2) Global parity still failed for single-token/full-stack runs

- Even with single-token prompts (to reduce cross-position confounds), hidden/logits parity remained poor.
- Conclusion: mismatch is in transformer semantics, not just prompt formatting.

## 3) Layer bisection localized a major drift jump around shared-KV regime

- Layer-wise `l_out-*` comparisons showed a clear discontinuity near layer 15.
- This lines up with Gemma shared-KV behavior (`shared_kv_layers`).
- A/B with shared-KV disabled reduced mismatch materially.
- Conclusion: shared-KV path is an important contributor, but not necessarily the first source.

## 4) Tested and ruled out specific hypotheses

- **KV cache f16 rounding** as primary culprit: ruled out (minimal effect).
- **Wrong post-FFN norm field**: ruled out.
  - `post_ffw_norm.weight` is the correct field, used as direct multiplicative scale.
- **Q8 FFN kernels (`w_gate`, `w_up`, `w_down`)**: ruled out via q8-vs-dense reference parity.
- **Q8 attention projection kernels (`wq`, `wk`)**: ruled out via q8-vs-dense reference parity.

## 5) Attention-stage narrowing (Q/K probes)

- Added direct stage comparisons against:
  - `Qcur_normed-*`, `Kcur_normed-*`
  - `Qcur_pos-*`, `Kcur_pos-*`
- Observed:
  - Q-path absolute drift is substantially larger than K-path.
  - This begins pre-RoPE and persists post-RoPE.
- However, later inspection showed Q and K norm scales are intentionally very different, so absolute-error asymmetry alone is not sufficient proof of a Q-only bug.

## 6) Norm weight inspection (important context)

- `attn_q_norm.weight` is near ~1.0 constant.
- `attn_k_norm.weight` is much smaller (roughly ~0.06 to ~0.13 constant depending on layer type).
- Implication: Q naturally has larger absolute magnitude; absolute `max |Δ|` must be interpreted with relative error as well.

---

## Current State (What We Know with High Confidence)

- Chat garbage is a downstream consequence of accumulated numeric mismatch, not a pure decoding/display issue.
- The mismatch is real and appears early enough to be autoregressively amplified.
- Several obvious implementation mistakes have been ruled out:
  - wrong FFN post-norm field,
  - wrong Q8 kernels (FFN and Q/K projections),
  - simple KV f16 rounding issue.
- Shared-KV semantics/interactions still matter and can amplify mismatch from mid-stack onward.

---

## Most Likely Remaining Error Surfaces

1. **Semantic mismatch in attention path ordering/parameterization**
   - Especially around Q/K normalization + RoPE conventions + per-layer variants.
2. **Shared-KV borrow semantics edge cases**
   - Exact source mapping, cache view semantics, SWA/global interactions.
3. **Node mapping / comparison interpretation pitfalls**
   - Need relative metrics to avoid over-weighting naturally larger channels (Q path).

---

## Recommended Next Steps (Concrete)

1. Extend `gemma_tensor_node_vs_llama` reporting with:
   - relative max error (`max |Δ| / max(|ref|, eps)`),
   - relative RMSE.
2. Re-run the same Q/K stage set on layers 0 and 14 with relative metrics:
   - `q_normed`, `k_normed`, `q_after_rope`, `k_after_rope`.
3. Add a dedicated shared-KV semantic probe:
   - compare borrowed K/V slices against expected source-layer cache rows for selected layers/positions.
4. Re-evaluate first true divergence point using relative metrics before deepening FFN/tail checks.

---

## Quick Command Examples

```bash
# Q/K projection kernel parity (already added)
GEMMA_RUST_LAYER_INDEX=0 cargo test --test gemma_tensor_node_vs_llama gemma4_wq_wk_q8_kernel_vs_dense_reference --release -- --ignored --nocapture

# Node-vs-stage compare example
GEMMA_RUST_STAGE=q_after_rope GEMMA_RUST_LAYER_INDEX=14 GEMMA_LLAMA_TENSOR_NODE=708 GEMMA_RUST_NUM_LAYERS=15 GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture

# Q/K norm tensor stats
cargo test --test gemma_q8_smoke gemma4_print_qk_norm_weight_stats --release -- --ignored --nocapture
```

---

---

## Session: 2026-04-17 — Fine-Grained Node-by-Node Comparison

### 7) Confirmed llama.cpp produces correct output with same GGUF

Ran `llama-cli` on `gemma-4-E2B-it-Q8_0.gguf` with the prompt "What is the capital of France?". Output was coherent and correct ("Paris"). Confirms the GGUF file is valid and the problem is solely in the Rust engine.

### 8) Confirmed `inp_scaled` (embedding × √hidden_dim) is a perfect match

```
GEMMA_RUST_NUM_LAYERS=0  GEMMA_LLAMA_TENSOR_NODE=1  GEMMA_RUST_STAGE=inp_scaled
→ max |Δ| = 0.000000
```

Embedding lookup and scaling are identical. The divergence starts **inside the layer computations**.

### 9) Layer 0 stage-by-stage comparison (single BOS token, OLD dequantize-F32 matmul)

| Stage              | Node | max \|Δ\| | Notes                                 |
|--------------------|------|-----------|---------------------------------------|
| inp_scaled         | 1    | **0.000** | Perfect match                         |
| attn_proj (wo out) | 30   | 0.115     | Small Q8 matmul error                 |
| attn_post_norm     | 32   | **4.661** | RMSNorm amplifies ~40×                |
| ffn_normed         | 35   | 0.075     | Residual dampens after add+norm       |
| ffn_down           | 39   | 0.662     | FFN matmul accumulates                |
| ffn_post_norm      | 41   | **4.022** | RMSNorm amplifies again               |
| pe_in              | 42   | 3.904     | After residual (similar to post_norm) |
| l_out              | 63   | **0.077** | layer_output_scale dampens            |

**Key insight:** RMSNorm amplifies errors ~40–50× because the attention/FFN outputs have very small RMS values (~0.02). The layer_output_scale (~0.02) then dampens the layer output back down.

### 10) Cross-layer l_out accumulation (OLD matmul, single BOS token)

| Layer | max \|Δ\| l_out | Layer type |
|-------|-----------------|------------|
| 0     | 0.077           | SWA        |
| 1     | 0.107           | SWA        |
| 2     | 0.238           | SWA        |
| 3     | 1.143           | SWA        |
| 4     | 0.531           | Global     |
| 5     | 0.266           | SWA        |
| 9     | 0.562           | Global     |
| 14    | 4.213           | Global     |
| 19    | 10.591          | Borrowed   |
| 24    | 11.092          | Borrowed   |
| 29    | **28.231**      | Borrowed   |
| 34    | 4.479           | Last layer |

Error grows steadily from 0.077 → 28.2 across 35 layers, non-monotonically. No single layer shows a catastrophic jump — the growth is gradual accumulation.

### 11) Root cause hypothesis: Q8_0 matmul approach differs from ggml

Discovered that ggml's CPU `mul_mat` for F32×Q8_0 does **not** dequantize the weights. Instead:
1. **Quantizes the F32 input row to Q8_0** (via `from_float` / `quantize_row_q8_0`)
2. Uses **Q8_0×Q8_0 integer dot product** (`vec_dot_q8_0_q8_0`)

Our Rust code was doing the opposite: dequantize Q8_0 weights to F32, then F32×F32 dot product. These give systematically different rounding patterns.

**Implemented the ggml-matching approach:**
- Added `quantize_row_q8_0()` (F32→Q8_0) matching ggml's `quantize_row_q8_0_ref`
- Added `vec_dot_q8_0()` (Q8×Q8 integer dot) matching ggml's `ggml_vec_dot_q8_0_q8_0`
- Rewrote `matmul_f32_q8_0()` to use quantize-input + Q8×Q8 path

### 12) Results AFTER matmul rewrite — Layer 0 stages

| Stage          | OLD max \|Δ\| | NEW max \|Δ\| | Change     |
|----------------|---------------|---------------|------------|
| attn_proj      | 0.115         | **0.020**     | **5.6× better** |
| attn_post_norm | 4.661         | **0.977**     | **4.8× better** |
| ffn_normed     | 0.075         | **0.018**     | **4.1× better** |
| ffn_down       | 0.662         | 0.714         | ~same      |
| ffn_post_norm  | 4.022         | 3.799         | ~same      |
| pe_in          | 3.904         | 3.952         | ~same      |
| l_out          | 0.077         | 0.082         | ~same      |

**Matmul outputs** (attn_proj, ffn_normed) improved ~5× because they're directly produced by the matmul. But **layer output** stayed the same — the improvement gets washed out by other error sources (FFN path, PLE, RMSNorm amplification).

### 13) Cross-layer l_out AFTER matmul rewrite

| Layer | OLD max \|Δ\| | NEW max \|Δ\| |
|-------|---------------|---------------|
| 0     | 0.077         | 0.082         |
| 3     | 1.143         | 1.068         |
| 9     | 0.562         | 0.572         |
| 14    | 4.213         | 4.342         |
| 29    | 28.231        | 29.073        |
| 34    | 4.479         | 4.938         |

**Conclusion: The matmul approach change did NOT improve the cross-layer error.** The per-layer error is dominated by something other than the Q8_0 matmul rounding differences.

### 14) Final logits comparison AFTER matmul rewrite (single BOS token)

```
llama.cpp argmax: 236761,  Rust argmax: 90696  (MISMATCH)
max |Δlogit| = 40.92,  RMSE = 25.30
```

Previous (old matmul): RMSE was 26.76. Marginal improvement (~5%), still completely wrong. Argmax still mismatches.

### 15) ggml RMSNorm uses double-precision accumulation

Confirmed from ggml source (`ops.cpp`):
```c
ggml_float sum = 0.0;  // ggml_float = double
for (int64_t i00 = 0; i00 < ne00; i00++) {
    sum += (ggml_float)(x[i00] * x[i00]);
}
```

Our Rust RMSNorm uses `f32` accumulation:
```rust
let mut sum_squared: f32 = 0.0;
for &v in x.iter() { sum_squared += v * v; }
```

For dim=1536, the f64 vs f32 accumulation difference in the sum is ~0.004 absolute, leading to ~1.3e-6 relative error in the RMS. This is **tiny** and unlikely to be the primary source, but it's a systematic difference that compounds through every RMSNorm call (2–3 per layer × 35 layers ≈ 100 RMSNorm calls).

### 16) Known test bugs (not model computation bugs)

- `q_raw` and `k_raw` stage comparisons show max |Δ| ~ 340 because the test function `prefill_qk_raw_layer` projects `input.hidden()` through wq/wk WITHOUT first applying attn_norm. The actual model forward pass DOES apply attn_norm correctly. This is a **test-only bug** — it does not affect model output.

---

## Updated Conclusions

### What is confirmed correct:
- Token embedding lookup and scaling (perfect match)
- Attention scaling (1.0 for Gemma4 — matches llama.cpp)
- RoPE rotary dimensions per layer (256 SWA, 512 global)
- Shared-KV layer borrowing semantics
- V normalization (epsilon-only RMSNorm, no weights)
- GEGLU activation (gelu_pytorch_tanh matches ggml_gelu)
- The llama.cpp GGUF file produces correct output

### What has been tested and rules out as root cause:
- Q8_0 matmul approach (quantize-input vs dequantize-weights): both give ~same layer-level error
- KV cache f16 rounding: previously ruled out as primary culprit
- Wrong FFN post-norm weights: ruled out
- Q8 kernel bugs for Q/K/V/FFN projections: ruled out

### What remains as error sources:
1. **RMSNorm f32 vs f64 accumulation** — systematic but small (~1e-6 per call, ~100 calls total). Easy to fix and test.
2. **KV cache f32 vs f16** — not yet directly tested with the new matmul. In llama.cpp, V is stored/read as f16; in Rust it stays f32. For single-token attention, output = V_f16 (llama) vs V_f32 (Rust).
3. **Flash attention vs naive attention** — for single token, both return V[0], so shouldn't differ. For multi-token decode, numerical differences from different accumulation could matter.
4. **PLE (Per-Layer Embedding) computation** — complex sub-network (matmul + GELU + norm + scale). Not yet compared stage-by-stage. Contributes to the pe_in → l_out error.
5. **Compound accumulation through 35 layers** — even small per-layer errors (~0.08) compound to ~29 at layer 29 due to RMSNorm amplification at each layer.

### Key open question:
Is the error purely from cumulative floating-point differences (no bug, just different numerical path), or is there an actual semantic bug hiding in the noise? The fact that longer prompts (8+ tokens) produce matching argmax but single tokens don't suggests the errors are **relative to signal strength**, not an absolute bug.

---

## Bottom Line

The investigation has substantially narrowed the space: major kernel and field-selection bugs are ruled out. The matmul approach was matched to ggml's exact quantize-input + Q8×Q8 path, which improved individual matmul outputs ~5× but did NOT reduce the cross-layer error. The remaining divergence appears to be from **cumulative small differences** (RMSNorm f32 vs f64, KV cache f16 vs f32, and other precision differences) amplified through 35 layers of RMSNorm. The next most impactful test would be switching RMSNorm to f64 accumulation and/or adding f16 KV cache quantization.

---

## Session: 2026-04-17 — Precision Matching (RMSNorm f64, Softmax f64, KV f16 experiment)

### 17) Matched RMSNorm to ggml's exact precision path

ggml's RMSNorm (`ggml_compute_forward_rms_norm_f32`) uses a specific hybrid precision:
1. Product `x*x` computed in **f32**, then cast to **f64** for accumulation
2. Mean truncated back to **f32**: `float mean = sum/ne00`
3. Scale computed in **f32**: `1.0f / sqrtf(mean + eps)`
4. `y = x * scale` in f32 (weight multiplication is a separate `ggml_mul` node)

Applied this exact path to `rmsnorm()`, `rmsnorm_inplace_no_scale()`, and `rms_only()`.

**Important subtlety**: Initially used `(x as f64) * (x as f64)` for the product (computing in f64), which made results **worse** — it's a *different* numerical path from ggml's `(ggml_float)(x[i]*x[i])` which computes the product in f32 first. Fixed to `(x * x) as f64`.

### 18) Matched softmax to ggml's f64 accumulation

ggml's softmax uses `expf()` (f32 exp) with f64 sum accumulation. Applied same approach: `(x - max).exp() as f64` accumulated in f64 sum, then `(1.0 / sum) as f32` for the inverse.

### 19) Layer 0 results AFTER RMSNorm f64 + Softmax f64 (single BOS token)

| Stage          | Before (f32) | After (f64 match) | Change |
|----------------|-------------|-------------------|--------|
| attn_proj      | 0.020       | **0.018**         | 12% better |
| attn_post_norm | 0.977       | **0.712**         | 27% better |
| ffn_normed     | 0.018       | **0.017**         | 6% better |
| ffn_down       | 0.714       | **0.692**         | 3% better |
| ffn_post_norm  | 3.799       | 3.843             | ~same |
| l_out          | 0.082       | 0.081             | ~same |

### 20) Logits AFTER RMSNorm f64 + Softmax f64

| Prompt | Tokens | Old RMSE | New RMSE | Argmax |
|--------|--------|----------|----------|--------|
| BOS only | 1 | 25.30 | **22.87** | mismatch |
| "Hello" | 2 | ~26.76 | **20.06** | mismatch |
| "What is the capital of France?" | 8 | — | **7.41** | **matches** (107) |

~25% RMSE improvement for 2-token prompt. 8-token prompt produces matching argmax with RMSE 7.4.

### 21) Critical discovery: ggml uses Flash Attention + f16 KV cache

Traced the llama.cpp computation graph (node dump for single BOS token) and found:

- **Node 28: `FLASH_ATTN_EXT`** — ggml uses **flash attention**, not naive Q·K^T → softmax → V
- **Nodes 18/20: `SET_ROWS` with dtype `f16`** — KV cache stored as **f16**
- **Nodes 23-26**: Attention reads K/V from the **f16 cache**, not the original f32 tensors

This means even during prefill, ggml:
1. Computes K/V in f32
2. Writes to f16 cache (lossy)
3. Reads back from f16 cache for attention
4. Uses flash attention algorithm (different accumulation order from naive)

### 22) f16 KV rounding experiment (reverted)

Tested adding f16 rounding to K/V before attention (matching ggml's f16 cache):
- **attn_proj improved 9000×** (0.018 → 0.000002) — confirms f16 KV is the dominant attention error
- **BUT overall logits RMSE got worse** (20.06 → 20.36 for "Hello", 7.41 → 10.34 for 8 tokens)
- **Reason**: f16 KV matches ggml's attention *inputs*, but our naive attention *algorithm* differs from flash attention. Matching one half without the other produces a result that's neither path.
- **Reverted** — f16 KV should only be added together with flash attention implementation.

---

## Updated Conclusions (post session 2026-04-17b)

### What is now confirmed correct:
- All items from previous session
- RMSNorm accumulation precision matches ggml (f32 product, f64 sum, f32 mean/scale)
- Softmax accumulation matches ggml (f32 exp, f64 sum)
- RMSNorm weight multiplication order matches ggml: `(x * scale) * w`

### Remaining dominant error sources (ranked):
1. **Flash attention vs naive attention** — ggml uses `FLASH_ATTN_EXT` with f16 K/V. Our naive attention with f32 K/V produces different results. This is the single biggest remaining difference and would require implementing flash attention + f16 KV cache together.
2. **Q8_0 FFN matmul accumulation** — 3 matmuls per layer with large intermediate dim (6144). Individual per-element errors are inherent to Q8_0 quantization but compound through RMSNorm amplification. Not a bug.
3. **Compound floating-point accumulation** — inherent to different code paths (Rust scalar vs ggml vectorized/SIMD). ~0.08 per layer × 35 layers, amplified by RMSNorm.

### Recommended next steps:
1. **Investigate Q8×Q8 matmul** — the new `vec_dot_q8_0` integer dot product is now the dominant error source, accounting for most of the ~20-26 RMSE. Profiling individual matmul errors vs ggml's Q8_0 dot product would identify whether alignment, rounding, or accumulation order differences are responsible.
2. **Accept current precision** for generation quality testing — the model output is structurally correct; the RMSE comes from compound floating-point differences in the FFN path.

---

## Flash attention experiment (2026-04-17c)

### 23) Implemented online softmax (flash attention algorithm)

Replaced naive "compute all scores → softmax → weighted V sum" with ggml-matching online softmax:
```
for each KV position j:
    score = Q · K[j] * scale
    if score > running_max:  rescale previous accumulation
    VKQ += exp(score - max) * V[j]
    sum += exp(score - max)
output = VKQ / sum
```

Applied to all three attention paths: `prefill_attention_layer`, `prefill_attention_core_layer`, `decode_attention_layer`. Toggle via `INFERENCE_ENGINE_FLASH_ATTN=0` for naive softmax fallback.

### 24) Tested multiple flash attention precision configurations

Controlled A/B test within single session (all using same Q8×Q8 matmul):

| Configuration | BOS (1 tok) | "Hello" (2 tok) | 8 tokens |
|---|---|---|---|
| Naive softmax (f64 sum, FLASH_ATTN=0) | RMSE 26.14 | RMSE 7.07 | RMSE 20.98 |
| Online softmax (f32 accum) | RMSE 26.14 | RMSE 7.07 | RMSE 22.05 |
| Online + f16 KV + f32 accum | RMSE 23.93 | RMSE 7.89 | RMSE 19.96 |
| Online + f16 KV + f16 accum + Q round | RMSE 23.93 | RMSE 10.06 | RMSE 20.32 |

### 25) Key finding: flash attention is NOT the dominant error source

- **BOS and "Hello" identical** between naive and online softmax — proves the algorithm difference doesn't affect short sequences.
- **8-token difference is only ~1 RMSE** (20.98 vs 22.05) — the algorithm choice barely matters.
- **f16 VKQ accumulation hurt** rather than helped (10.06 vs 7.07 for "Hello") — our Q·K dot product doesn't bit-match ggml's f16·f16 SIMD, so matching the lossy accumulator adds error without matching the specific rounding pattern.
- **Previous session's 8-token RMSE of 12.22** was due to the old dequantize+f32 matmul, not the attention algorithm. The new Q8×Q8 integer matmul (`vec_dot_q8_0`) changed the FFN path baseline to ~21 RMSE regardless of attention.

### Conclusion: dominant error is Q8_0 FFN matmul, not attention

The ~20-26 RMSE comes almost entirely from compound Q8_0 matmul differences in the FFN path (3 matmuls × 35 layers). Flash attention vs naive attention contributes <1 RMSE unit. The Q8×Q8 integer dot product implementation is the primary target for further parity improvement.

---

## Session: 2026-04-17d — New Baseline Before Further Changes

### 26) Fresh repo baseline (current tree, no code changes)

Ran:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Observed (`GEMMA_LOGITS_PROMPT="Hello"` path):

- prompt token ids: `[2, 9259]`
- llama argmax: `236888`
- rust argmax: `132650` (mismatch)
- `max |Δlogit| = 37.392067`
- `RMSE = 20.056932`

This is the baseline for subsequent A/B experiments in this session.

### 27) A/B: derive Q8 quantization inverse-scale from fp16-rounded `d`

Hypothesis: ggml parity might require deriving `id` from the exact stored fp16 `d` (`f16_to_f32(f32_to_f16(d))`) instead of full-precision `d`.

Implementation: added env-gated branch in `quantize_row_q8_0()`:

- default: `id = 1.0 / d` (existing behavior)
- A/B: `INFERENCE_ENGINE_Q8_ID_FROM_F16=1` uses `id = 1.0 / d_round`

Runs:

```bash
# default path (control)
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture

# A/B variant
INFERENCE_ENGINE_Q8_ID_FROM_F16=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"` prompt ids `[2, 9259]`):

- **Control:** unchanged baseline (`max |Δ| 37.392067`, `RMSE 20.056932`, argmax `132650`)
- **A/B (`id` from fp16):**
  - llama argmax `236888`, rust argmax `236764` (still mismatch)
  - `max |Δ| 41.688454` (**worse**)
  - `RMSE 20.743261` (**worse**)
  - test fails tolerance (`max_abs > 40`)

Conclusion: deriving `id` from fp16-rounded scale is not the fix; it degrades parity. Keep this only as an env-gated diagnostic path.

### 28) A/B: replace custom fp16 conversion with `half::f16`

Hypothesis: the custom `f32_to_f16` / `f16_to_f32` helpers may not exactly match ggml/IEEE rounding behavior in edge cases, introducing systemic Q8 path drift.

Change:

- Added dependency `half` (`cargo add half`)
- Replaced `src/ops/quant/utils.rs` conversions with:
  - `half::f16::from_f32(value).to_bits()`
  - `half::f16::from_bits(bits).to_f32()`

Runs:

```bash
# default control path
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture

# prior diagnostic toggle (kept for A/B)
INFERENCE_ENGINE_Q8_ID_FROM_F16=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"` prompt ids `[2, 9259]`):

- **Default (new fp16 conversion):**
  - llama argmax `236888`, rust argmax `236929` (still mismatch)
  - `max |Δ| = 34.965363` (**better** vs 37.392067 baseline)
  - `RMSE = 19.223484` (**better** vs 20.056932 baseline)
- **With `INFERENCE_ENGINE_Q8_ID_FROM_F16=1`:**
  - unchanged from prior worse result (`max |Δ| 41.688454`, `RMSE 20.743261`)

Conclusion:

- Replacing the fp16 conversion implementation yields a **meaningful improvement** (~4.1% RMSE drop on this check).
- The separate `id-from-rounded-d` branch remains harmful and should not be used.

### 29) Check generalization on 8-token prompt after fp16 conversion swap

Run:

```bash
GEMMA_LOGITS_PROMPT='What is the capital of France?' cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result:

- prompt ids: `[2, 3689, 563, 506, 5279, 529, 7001, 236881]`
- llama argmax: `107`
- rust argmax: `107` (**match**)
- `max |Δlogit| = 27.899334`
- `RMSE = 8.492210`

Interpretation:

- Multi-token behavior remains aligned on argmax for this probe.
- Compared to the previous session note (`RMSE 7.41` for this prompt under a different code state), this exact branch/config is still in the same qualitative regime (single-digit RMSE with matching argmax), but not yet fully parity-matched.

### 30) Hidden-state parity check after fp16 conversion swap

Run:

```bash
cargo test --test gemma_hidden_vs_llama gemma4_e2b_last_hidden_vs_llama --release -- --ignored --nocapture
```

Result (`"Hello"` path):

- prompt ids: `[2, 9259]`
- `post output_norm max |Δhidden| = 62.552689`
- test fails current tolerance (`62.55 > 56`)

Interpretation:

- Despite improved logits RMSE on `"Hello"`, this hidden probe is still far from parity (and now outside tolerance).
- The fp16 conversion change is helpful but insufficient; dominant upstream numerical divergence remains.

### 31) A/B: quantization tie-breaking (`round` vs ties-to-even)

Hypothesis: ggml nearest-int tie behavior might be ties-to-even; if so, quantizing with ties-to-even could reduce Q8 drift.

Implementation (env-gated in `quantize_row_q8_0`):

- default: `x.round()`
- A/B: `INFERENCE_ENGINE_Q8_TIES_EVEN=1` uses `x.round_ties_even()`

Runs:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_Q8_TIES_EVEN=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- Both runs are **bit-identical** at reported precision:
  - llama argmax `236888`, rust argmax `236929`
  - `max |Δ| = 34.965363`
  - `RMSE = 19.223484`

Conclusion: tie-breaking mode is not a meaningful factor for this prompt/path.

### 32) Direct CLI generation check (user-visible symptom) after fp16 conversion swap

First attempted an invalid flag form (`--prompt`), then corrected to positional prompt.

Valid run:

```bash
cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 32 \
  "What is the capital of France?"
```

Observed output remains garbled:

```text
The**
**France**<turn|>There are<|audio>"Cane" is the"<turn|>
ear" is the
ntrade".<turn|><turn|>
was a
```

Interpretation:

- Numeric parity improved in targeted tests, but user-facing continuation quality is still poor.
- Structural markers still leak in this CLI path, so output quality is impacted by both residual numeric drift and decode/render behavior.

### 33) Fix: apply Gemma structural-stop + assistant-visible cleanup in `src/main.rs`

Observation from step 32 showed the non-interactive CLI path still emitted raw structural markers (`<turn|>`, `<|audio...>`), unlike `src/bin/chat.rs`.

Change in `src/main.rs`:

- Added EOS stop (`tok_prompt.eos_token_id`) in generation loop.
- For `--chat gemma4-e2b`, stop generation when `gemma4_e2b_decode_has_structure_marker()` is detected.
- Postprocess final decode with `gemma4_e2b_assistant_visible()` for Gemma mode.

Validation run:

```bash
cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 32 \
  "What is the capital of France?"
```

New output:

```text
The**
**France**
```

Interpretation:

- Structural garbage tokens are now suppressed in this CLI path.
- Semantic answer quality is still poor (numeric parity issue remains), but output hygiene is improved and consistent with the chat binary’s trimming behavior.

### 34) Regression check after CLI cleanup

Run:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result (`"Hello"`):

- unchanged from step 28 default:
  - llama argmax `236888`, rust argmax `236929`
  - `max |Δ| = 34.965363`
  - `RMSE = 19.223484`

Conclusion: CLI cleanup did not change model math/parity behavior.

### 35) A/B: Q8 dot-product accumulation precision (`f32` vs `f64`)

Hypothesis: `vec_dot_q8_0` block accumulation in `f32` might diverge from ggml enough to matter; `f64` accumulation could reduce error.

Implementation (env-gated in `vec_dot_q8_0`):

- default: `f32` accumulation
- A/B: `INFERENCE_ENGINE_Q8_DOT_F64_ACCUM=1` uses `f64` accumulation and casts once at the end

Runs:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_Q8_DOT_F64_ACCUM=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- control: `max |Δ| = 34.965363`, `RMSE = 19.223484`, argmax `236929`
- f64-accum: `max |Δ| = 37.706837`, `RMSE = 20.652384`, argmax `236929`

Conclusion: f64 accumulation in Q8 dot product is worse for parity; reject.

### 36) A/B: Q8 quant clamp range (`[-128,127]` vs `[-127,127]`)

Hypothesis: ggml quantization might effectively avoid `-128`; forcing symmetric clamp to `[-127,127]` could align better.

Implementation (env-gated in `quantize_row_q8_0`):

- default: clamp `[-128,127]`
- A/B: `INFERENCE_ENGINE_Q8_CLAMP_127=1` clamps `[-127,127]`

Runs:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_Q8_CLAMP_127=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- identical metrics in both runs:
  - `max |Δ| = 34.965363`
  - `RMSE = 19.223484`
  - argmax `236929`

Conclusion: clamp endpoint choice is not a factor in this path.

### 37) A/B: re-introduce Q8 dequantize-to-f32 matmul as fallback

Hypothesis: although ggml uses quantize-input + Q8×Q8 integer dot, this repo’s previous dequantize-weight F32 path might still yield better effective parity/quality in current code state.

Implementation:

- Added env-gated fallback in `matmul_f32_q8_0`:
  - default: current Q8×Q8 path
  - `INFERENCE_ENGINE_Q8_DEQUANT_F32=1`: dequantize each Q8 block and accumulate F32 dot products

Runs:

```bash
# "Hello" parity
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_Q8_DEQUANT_F32=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- default: `max |Δ| = 34.965363`, `RMSE = 19.223484`, rust argmax `236929`
- dequant fallback: `max |Δ| = 33.720894`, `RMSE = 19.112051`, rust argmax `236764`

This is a small numeric improvement on this short prompt.

Additional checks:

```bash
INFERENCE_ENGINE_Q8_DEQUANT_F32=1 GEMMA_LOGITS_PROMPT='What is the capital of France?' cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_Q8_DEQUANT_F32=1 cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 32 "What is the capital of France?"
```

Notes:

- First attempt in sandbox failed in `llama_logits_ref` due Metal backend context init; re-ran outside sandbox.
- 8-token parity with fallback:
  - argmax still matches (`107`)
  - `max |Δlogit| = 25.368717` (better)
  - `RMSE = 9.873473` (worse than current default 8.492210)
- CLI generation output with fallback remains:
  - `The**`
  - `**France**`

Conclusion:

- Dequant fallback helps one short-prompt metric but harms 8-token RMSE and does not improve user-visible output quality on the tested prompt.
- Keep as diagnostic toggle only; not a clear default-path win.

### 38) Direct generation A/B (default vs dequant fallback)

Run:

```bash
cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 48 "What is the capital of France?"
INFERENCE_ENGINE_Q8_DEQUANT_F32=1 cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 48 "What is the capital of France?"
```

Result:

- Outputs are identical for this probe (`The**` / `**France**`).

Conclusion: dequant fallback does not improve practical generation quality for this user-facing prompt.

### 39) A/B: disable Gemma PLE tail to test whether PLE is mis-implemented

Hypothesis: residual drift could come from a semantic bug in the per-layer embedding tail; disabling PLE should improve parity if PLE math/wiring is wrong.

Implementation (env-gated in `prefill_block.rs`):

- Added `INFERENCE_ENGINE_GEMMA4_DISABLE_PLE=1` toggle to skip `apply_per_layer_tail` in:
  - `gemma4_prefill_layer_debug`
  - `prefill_layer_block`
  - `decode_layer_block`

Run:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_GEMMA4_DISABLE_PLE=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- control: `max |Δ| = 34.965363`, `RMSE = 19.223484`, argmax `236929`
- disable PLE:
  - rust argmax `66308`
  - `max |Δ| = 43.379738` (**worse**)
  - `RMSE = 19.628275` (**worse**)
  - fails tolerance (`max_abs > 40`)

Conclusion: PLE is not the culprit; removing it makes parity worse.

### 40) A/B: disable `layer_output_scale`

Hypothesis: post-layer scalar application might be mismatched; disabling it could improve parity if scale handling is wrong.

Implementation (env-gated in `apply_gemma_layer_output_scale`):

- Added `INFERENCE_ENGINE_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE=1` to skip scale multiply.

Run:

```bash
cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
INFERENCE_ENGINE_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE=1 cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results (`"Hello"`):

- disable scale:
  - rust argmax `26299`
  - `max |Δ| = 42.736713` (**worse**, fails tolerance)
  - `RMSE = 15.345409` (numerically lower RMSE but dominated by large max-error outliers + wrong argmax)

Interpretation:

- `layer_output_scale` clearly affects distribution shape; removing it is not a viable fix despite lower RMSE.
- Max-error and argmax behavior confirm this path diverges semantically.

### 41) User-visible generation checks for ablations

Run:

```bash
INFERENCE_ENGINE_GEMMA4_DISABLE_PLE=1 cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 48 "What is the capital of France?"
INFERENCE_ENGINE_GEMMA4_DISABLE_LAYER_OUTPUT_SCALE=1 cargo run --release -- -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf -t model/gemma-4-e2b-it/tokenizer.json --chat gemma4-e2b -n 48 "What is the capital of France?"
```

Outputs are significantly worse than baseline:

- Disable PLE: mixed-script gibberish (`و路нгwi‌ایْkr ...`)
- Disable layer scale: heavily corrupted multilingual repetition

Conclusion: both Gemma-specific components (PLE + layer_output_scale) are required; they are not the root bug.

### 42) Decode-path discriminator: normal decode vs teacher-forced full-prefill-per-step

Goal:

- Determine whether bad generation quality comes from decode/KV-cache semantics (`decode_forward`) or from shared core math that is already present in prefill.

Method:

- Added `gemma4_generation_decode_vs_teacher_forced_prefill` in `tests/gemma_decode_self_consistency.rs`.
- Two generation paths with identical greedy settings:
  - **Path A (normal):** one `prefill_forward`, then token-by-token `decode_forward`.
  - **Path B (teacher-forced prefill):** recompute full `prefill_forward` from scratch each step on the growing token list (no decode/KV-cache path).
- Same prompt, same stop token, same max token budget.

Run 1 (`"Hello"`, quick check):

```bash
GEMMA_MAX_NEW_TOKENS=8 GEMMA_LOGITS_PROMPT='Hello' \
  cargo test --test gemma_decode_self_consistency \
  gemma4_generation_decode_vs_teacher_forced_prefill --release -- --ignored --nocapture
```

Result:

- `prompt_ids: [2, 9259]`
- Path A generated 8 tokens
- Path B generated 8 tokens
- shared prefix length: 8 (identical sequence)
- decoded text (both paths): `` `pyrl\nimport numpy as np` ``

Run 2 (problematic prompt, longer check):

```bash
GEMMA_MAX_NEW_TOKENS=32 GEMMA_LOGITS_PROMPT='What is the capital of France?' \
  cargo test --test gemma_decode_self_consistency \
  gemma4_generation_decode_vs_teacher_forced_prefill --release -- --ignored --nocapture
```

Result:

- `prompt_ids: [2, 3689, 563, 506, 5279, 529, 7001, 236881]`
- Path A generated 16 tokens
- Path B generated 16 tokens
- shared prefix length: 16 (identical sequence)
- decoded text (both paths):
  - `\nFrance?\n<turn|>~I<turn|>\nans<turn|><turn|><turn|>UP<turn|>\n`

Conclusion:

- The generation is indeed wrong on the user-facing problematic prompt.
- But Path A and Path B are identical on both probes, so the issue is **not** a decode-only / KV-cache-only divergence.
- Remaining root-cause surface is in shared model semantics (prefill/decode common math), not in decode-vs-prefill path mismatch.

### 43) Lowest-level attention I/O parity and drift amplification

Goal:

- Directly test whether the Rust attention implementation is fundamentally off by feeding it the exact llama.cpp attention input tensor and comparing against llama.cpp attention output.

Method:

- Added `gemma4_single_layer_attention_llama_io_parity` in `tests/gemma_tensor_node_vs_llama.rs`.
- Inputs are taken from llama tensor dumps:
  - `GEMMA_LLAMA_ATTN_IN_NODE`: attention input tensor node (e.g. `attn_norm-*`)
  - `GEMMA_LLAMA_ATTN_OUT_NODE`: attention output tensor node after `wo`
- Rust runs `prefill_attention_layer(...)` on the exact dumped input and compares full `[seq, hidden]` output vs llama output.

Runs:

```bash
# Layer 0, "Hello" token ids [2,9259]
GEMMA_RUST_LAYER_INDEX=0 GEMMA_LOGITS_TOKEN_IDS=2,9259 \
GEMMA_LLAMA_ATTN_IN_NODE=3 GEMMA_LLAMA_ATTN_OUT_NODE=30 GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_attention_llama_io_parity --release -- --ignored --nocapture

# Layer 1, same prompt ids
GEMMA_RUST_LAYER_INDEX=1 GEMMA_LOGITS_TOKEN_IDS=2,9259 \
GEMMA_LLAMA_ATTN_IN_NODE=65 GEMMA_LLAMA_ATTN_OUT_NODE=91 GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_attention_llama_io_parity --release -- --ignored --nocapture
```

Results:

- Layer 0: `max_abs = 0.035418`, `RMSE = 0.007168`
- Layer 1: `max_abs = 0.057748`, `RMSE = 0.008314`

Interpretation:

- Attention block is close to llama.cpp at this low-level I/O boundary; no evidence of a gross semantic bug in attention itself for these probes.

Drift amplification test:

- Added `gemma4_single_layer_attention_drift_amplification`:
  - Seed A with llama one-step attention output.
  - Seed B with rust one-step attention output.
  - Re-apply the same Rust attention layer `N` times to both and track divergence growth.
  - Also project both final tensors to logits (`output_norm + lm_head`) and compare argmax.

Run:

```bash
GEMMA_RUST_LAYER_INDEX=0 GEMMA_LOGITS_TOKEN_IDS=2,9259 \
GEMMA_LLAMA_ATTN_IN_NODE=3 GEMMA_LLAMA_ATTN_OUT_NODE=30 GEMMA_REPEAT_LAYERS=32 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_attention_drift_amplification --release -- --ignored --nocapture
```

Results:

- Base one-step delta: `max_abs = 0.035418`, `RMSE = 0.007168`
- After 32 repeats:
  - hidden diff: `max_abs = 0.231624`, `RMSE = 0.026522` (≈ `3.700x` RMSE amplification)
  - logits diff: `max_abs = 1.333951`, `RMSE = 0.113022`
  - final argmax token: same on both paths (`5668`)

Conclusion:

- Small attention-only drift can amplify over repeated applications, but in this controlled test it still did **not** flip greedy token choice.
- This supports the view that attention drift alone is unlikely to explain the severe gibberish output.
- Practical generation failure is more likely from compounded multi-component drift across the full stack (attention + FFN/Q8 + residual/norm/tail interactions), not a single catastrophic attention mismatch.

---

## Session: 2026-04-27 — Chat Prompt Greedy Divergence

### 44) Exact chat-template prompt has correct first-token argmax

Goal:

- Separate prompt/tokenizer/template issues from numeric model-forward drift on the user-visible failure prompt.

Prompt tested:

```text
<|turn>user
What is the capital of France?<turn|>
<|turn>model
```

Run:

```bash
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result:

- prompt token ids: `[2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107]`
- special tokens are encoded atomically:
  - `105 = <|turn>`
  - `106 = <turn|>`
  - `107 = newline`
- llama.cpp argmax: `818`
- Rust argmax: `818`
- token `818 = "The"`
- `max |Δlogit| = 20.309284`
- `RMSE = 2.330945`

Interpretation:

- Chat template rendering and tokenizer special-token handling are not the primary cause of the first bad continuation.
- The model forward is close enough at the initial assistant position to choose the same first token as llama.cpp.

### 45) Divergence appears immediately after first generated token

Goal:

- Test the next greedy step after the shared first token `The`, because user-visible output begins as `The**`.

Prompt tested:

```text
<|turn>user
What is the capital of France?<turn|>
<|turn>model
The
```

Run:

```bash
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\nThe' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result:

- prompt token ids: `[2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107, 818]`
- llama.cpp argmax: `5279`
- Rust argmax: `1018`
- token `5279 = "▁capital"`
- token `1018 = "**"`
- `max |Δlogit| = 25.003351`
- `RMSE = 1.606843`

A/B with Q8 dequant fallback:

```bash
INFERENCE_ENGINE_Q8_DEQUANT_F32=1 \
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\nThe' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

- llama.cpp argmax: `5279`
- Rust argmax: `1018`
- `max |Δlogit| = 25.095692`
- `RMSE = 1.441077`

Interpretation:

- The bad English starts from an early high-leverage greedy argmax flip, not from a broad catastrophic logits failure.
- RMSE is low, but the top-token ordering is wrong: llama continues `The capital...`, while Rust continues `The**...`.
- The Q8 dequant fallback slightly improves RMSE but does not fix the top-token ordering, so this specific failure is not solved by switching away from the current Q8×Q8 path.

### 46) Decode path is not the discriminator for this prompt

Run:

```bash
GEMMA_MAX_NEW_TOKENS=8 \
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n' \
  cargo test --test gemma_decode_self_consistency \
  gemma4_generation_decode_vs_teacher_forced_prefill --release -- --ignored --nocapture
```

Result:

- prompt token ids: `[2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 7001, 236881, 106, 107, 105, 4368, 107]`
- normal decode and full-prefill-per-step both generated 8 tokens.
- shared prefix length: `8`
- decoded text from both paths: `"The**\n**France**<turn|>There"`

Interpretation:

- The decode/KV-cache path and the full-prefill path agree on the bad continuation.
- Remaining root-cause surface is the shared forward math and early top-k margin, not a decode-only cache bug.

### 47) Added top-k / watch-token diagnostics

Change:

- Extended `tests/gemma_logits_vs_llama.rs` with:
  - `GEMMA_LOGITS_TOP_K=N` to print top-k tokens for llama.cpp and Rust logits.
  - `GEMMA_LOGITS_WATCH_TOKEN_IDS=a,b,...` to print selected token logits and the `a-b` margin.
  - `gemma4_e2b_rust_layer_margin_trace` to project each Rust partial layer through `output_norm + lm_head` and trace the same watched-token margin.

Run:

```bash
GEMMA_LOGITS_TOP_K=8 \
GEMMA_LOGITS_WATCH_TOKEN_IDS=5279,1018 \
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\nThe' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Key result:

- llama.cpp top token: `5279 = " capital"` with logit `11.795767`
- Rust top token: `1018 = "**"` with logit `0.341894`
- llama.cpp watched margin `5279 - 1018 = +36.457222`
- Rust watched margin `5279 - 1018 = -7.009971`
- Per-token errors:
  - `5279`: Rust is `-18.463844` below llama.cpp.
  - `1018`: Rust is `+25.003351` above llama.cpp.

Interpretation:

- This is not a near tie in llama.cpp. The reference strongly prefers `" capital"`.
- Rust both suppresses the correct token and boosts the bad formatting token.

### 48) Rust partial-layer margin trace localizes the late flip

Run:

```bash
GEMMA_LOGITS_WATCH_TOKEN_IDS=5279,1018 \
GEMMA_LOGITS_PROMPT=$'<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\nThe' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_rust_layer_margin_trace --release -- --ignored --nocapture
```

Selected Rust margins (`5279 " capital" - 1018 "**"`):

| stage | margin | argmax |
|---|---:|---|
| embedding | `-2.742407` | `506 " the"` |
| layer 03 | `+39.901390` | `13906 " musical"` |
| layer 10 | `+35.527966` | `75197 " trắng"` |
| layer 11 | `-21.964584` | `29106 "fulness"` |
| layer 16 | `-31.904034` | `47662 " Units"` |
| layer 24 | `-1.442537` | `192127 " belangrijkste"` |
| layer 30 | `+11.996380` | `5279 " capital"` |
| layer 31 | `+12.891524` | `5279 " capital"` |
| layer 32 | `+9.501369` | `256000 "<|audio>"` |
| layer 33 | `-4.478191` | `3710 " heart"` |
| layer 34 | `-7.009971` | `1018 "**"` |

Interpretation:

- Rust's own partial-stack projection recovers the correct token at layers 30-31, then loses it in the final two layers.
- This makes layers 33 and 34 high-priority targets for node/stage comparison on the chat prompt ending with token `818`.
- Because partial-layer logits are artificial diagnostics, the next check should compare actual llama nodes for layer 33/34 stages rather than treating the partial argmax as model semantics.

### Updated next-test priority

1. Trace llama node IDs for the exact prompt ids ending in `818`.
2. Compare layer 33 and 34 stage outputs against llama nodes: `attn_proj`, `attn_post_norm`, `ffn_down`, `ffn_post_norm`, `pe_in`, `after_tail`, and `l_out`.
3. If the largest jump is in FFN stages, add a focused FFN I/O parity test analogous to `gemma4_single_layer_attention_llama_io_parity`.
4. If layer 33/34 stages look individually close, inspect whether the final `output_norm + lm_head + softcap` path is over-amplifying a residual-stream direction tied to formatting tokens.

### 49) Layer 33/34 node trace for the failing chat step

Prompt ids:

```text
2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818
```

Trace command:

```bash
LLAMA_TENSOR_DUMP_TRACE=1 tools/llama_tensor_dump_ref \
  model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  2 105 2364 107 3689 563 506 5279 529 7001 236881 106 107 105 4368 107 818 \
  2> /tmp/gemma_chat_the_nodes.tsv > /tmp/gemma_chat_the_tensor.bin
```

Relevant llama nodes:

| layer | stage | node |
|---:|---|---:|
| 33 | `attn_proj` (`wo`) | `1432` |
| 33 | `attn_post_norm` | `1434` |
| 33 | `attn_out` | `1435` |
| 33 | `ffn_normed` | `1437` |
| 33 | `ffn_gate` | `1438` |
| 33 | `ffn_up` | `1439` |
| 33 | `ffn_geglu` | `1440` |
| 33 | `ffn_out` / Rust `ffn_down` | `1441` |
| 33 | `ffn_post_rms` | `1442` |
| 33 | `ffn_post_norm` | `1443` |
| 33 | `pe_in` | `1444` |
| 33 | `after_tail` | `1452` |
| 33 | `l_out` | `1453` |
| 34 | `attn_proj` (`wo`) | `1469` |
| 34 | `attn_post_norm` | `1472` |
| 34 | `attn_out` | `1474` |
| 34 | `ffn_normed` | `1476` |
| 34 | `ffn_gate` | `1477` |
| 34 | `ffn_up` | `1478` |
| 34 | `ffn_geglu` | `1479` |
| 34 | `ffn_out` / Rust `ffn_down` | `1480` |
| 34 | `ffn_post_rms` | `1481` |
| 34 | `ffn_post_norm` | `1482` |
| 34 | `pe_in` | `1483` |
| 34 | `after_tail` | `1492` |
| 34 | `l_out` | `1493` |

### 50) Layer 33/34 stage comparison against llama nodes

Run pattern:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=<layer> \
GEMMA_RUST_STAGE=<stage> \
GEMMA_LLAMA_TENSOR_NODE=<node> \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture
```

Results:

| layer | stage | node | max \|Δ\| |
|---:|---|---:|---:|
| 33 | `attn_proj` | `1432` | `0.703454` |
| 33 | `attn_post_norm` | `1434` | `0.275717` |
| 33 | `attn_out` | `1435` | `19.450184` |
| 33 | `ffn_down` | `1441` | `4.757412` |
| 33 | `ffn_post_norm` | `1443` | `4.660990` |
| 33 | `pe_in` | `1444` | `18.731850` |
| 33 | `after_tail` | `1452` | `18.731850` |
| 33 | `l_out_debug` | `1453` | `13.024490` |
| 34 | `attn_proj` | `1469` | `1.991157` |
| 34 | `attn_post_norm` | `1472` | `0.932666` |
| 34 | `attn_out` | `1474` | `12.091827` |
| 34 | `ffn_normed` | `1476` | `1.836390` |
| 34 | `ffn_gate` | `1477` | `2.680873` |
| 34 | `ffn_up` | `1478` | `5.554852` |
| 34 | `ffn_geglu` | `1479` | `7.457932` |
| 34 | `ffn_down` | `1480` | `7.474422` |
| 34 | `ffn_post_rms` | `1481` | `3.819342` |
| 34 | `ffn_post_norm` / `ffn_post_mul` | `1482` | `44.638554` |
| 34 | `pe_in` | `1483` | `56.730381` |
| 34 | `after_tail` | `1492` | `56.730438` |
| 34 | `l_out_debug` | `1493` | `9.473540` |

Interpretation:

- Layer 33 attention projection itself is close, but the residual output `attn_out` already carries large upstream/residual drift.
- Layer 34 FFN input/projection errors are moderate until `ffn_post_norm`.
- The large jump is at layer 34 `ffn_post_norm`: `ffn_post_rms` is `3.819342`, but multiplying by `post_ffw_norm.weight` yields `44.638554`.
- This does **not** mean the post-FFN norm weight is wrong; it means a residual-stream direction error is being strongly amplified by a high-gain scale vector.

### 51) Re-verified layer 34 post-FFN scale field

Run:

```bash
GEMMA_RUST_LAYER_INDEX=34 \
GEMMA_LLAMA_RMS_NODE=1481 \
GEMMA_LLAMA_MUL_NODE=1482 \
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_post_ffw_scale_candidates_vs_llama --release -- --ignored --nocapture
```

Result:

- `post_ffw_norm.weight`: `max_abs=0.000001`, `rmse=0.000000`
- `1+post_ffw_norm.weight`: `max_abs=1.000001`, `rmse=1.000000`
- other norm candidates are much worse.

Interpretation:

- Layer 34 `post_ffw_norm.weight` is the correct scale tensor and is applied directly, not as `1 + weight`.

### 52) Focused FFN I/O parity test

Change:

- Added `gemma4_single_layer_ffn_llama_io_parity` in `tests/gemma_tensor_node_vs_llama.rs`.
- It feeds llama.cpp's FFN input tensor for the **last token** into Rust's `prefill_ffn`, then compares directly to llama.cpp's FFN output tensor.

Runs:

```bash
# layer 33
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=33 GEMMA_LLAMA_FFN_IN_NODE=1437 GEMMA_LLAMA_FFN_OUT_NODE=1441 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_ffn_llama_io_parity --release -- --ignored --nocapture

# layer 34
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=34 GEMMA_LLAMA_FFN_IN_NODE=1476 GEMMA_LLAMA_FFN_OUT_NODE=1480 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_ffn_llama_io_parity --release -- --ignored --nocapture
```

Results:

- Layer 33 FFN I/O parity: `max_abs=1.637649`, `rmse=0.056484`
- Layer 34 FFN I/O parity: `max_abs=3.700244`, `rmse=0.161430`

Interpretation:

- The FFN implementation/kernel is not catastrophically wrong when given llama.cpp's exact FFN input.
- The normal full-pipeline layer 34 `ffn_post_norm` drift is mostly accumulated upstream/residual-direction drift being amplified by RMSNorm + `post_ffw_norm.weight`, plus some moderate FFN Q8 drift.

### Updated next-test priority

1. Add a focused post-FFN RMS/scale I/O parity test: feed llama node `1480` into Rust's RMS-only + `post_ffw_norm.weight`, compare nodes `1481` and `1482`. This should confirm the post-norm operator is exact when the input is exact.
2. Track the residual-stream direction that causes the layer 34 `post_ffw_norm.weight` amplification: print the top dimensions contributing to the `1482` diff and their scale weights.
3. Work backward from layer 33 `attn_out` (`max |Δ| = 19.45`): compare `l_out-32` and layer 33 attention residual inputs to see whether the late-layer problem enters before layer 33 or is introduced by the layer 33 residual add.
4. If the high-gain dimensions are consistent, inspect whether quantized matmul errors in earlier layers project specifically into those dimensions.

### 53) Exact-input post-FFN RMS/scale parity

Change:

- Added `gemma4_post_ffn_norm_llama_io_parity` in `tests/gemma_tensor_node_vs_llama.rs`.
- It feeds llama.cpp's exact `ffn_out` into Rust's:
  - RMS-only path (`rmsnorm_inplace_no_scale`) and compares to node `1481`.
  - RMS + `post_ffw_norm.weight` path and compares to node `1482`.

Run:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=34 \
GEMMA_LLAMA_FFN_OUT_NODE=1480 \
GEMMA_LLAMA_FFN_POST_RMS_NODE=1481 \
GEMMA_LLAMA_FFN_POST_MUL_NODE=1482 \
GEMMA_TENSOR_TOP_DIFFS=10 \
GEMMA_TENSOR_MAX_ABS=1e-3 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_post_ffn_norm_llama_io_parity --release -- --ignored --nocapture
```

Result:

- `rms_max=0.000000`, `rms_rmse=0.000000`
- `scaled_max=0.000000`, `scaled_rmse=0.000000`

Interpretation:

- The post-FFN RMS-only and `post_ffw_norm.weight` multiply are exact when the input is exact.
- Therefore the layer 34 `ffn_post_norm` explosion is not an implementation bug in RMSNorm or scale multiplication.

### 54) Top dimensions responsible for layer 34 `ffn_post_norm` drift

Run:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=34 \
GEMMA_RUST_STAGE=ffn_post_norm \
GEMMA_LLAMA_TENSOR_NODE=1482 \
GEMMA_TENSOR_TOP_DIFFS=20 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture
```

Top contributors:

| dim | Rust | llama | diff | abs | `post_ffw_norm.weight` |
|---:|---:|---:|---:|---:|---:|
| 438 | `-78.517334` | `-33.878780` | `-44.638554` | `44.638554` | `11.687500` |
| 522 | `11.197654` | `-2.230171` | `13.427825` | `13.427825` | `6.718750` |
| 65 | `-4.174379` | `8.908162` | `-13.082541` | `13.082541` | `11.687500` |
| 1269 | `11.681285` | `22.543228` | `-10.861943` | `10.861943` | `11.687500` |
| 690 | `-1.694481` | `-12.251740` | `10.557259` | `10.557259` | `11.687500` |
| 920 | `0.870107` | `10.823763` | `-9.953655` | `9.953655` | `11.687500` |
| 1348 | `-1.644345` | `7.717689` | `-9.362034` | `9.362034` | `11.562500` |

Interpretation:

- The largest errors cluster in channels with high post-FFN scale (`~11.5x`).
- Dimension `438` is the dominant single contributor.

### 55) Dimension 438 is already the dominant residual-stream error before layer 34

Runs:

```bash
# layer 34 ffn_down / ffn_post_rms
GEMMA_RUST_LAYER_INDEX=34 GEMMA_RUST_STAGE=ffn_down GEMMA_LLAMA_TENSOR_NODE=1480 ...
GEMMA_RUST_LAYER_INDEX=34 GEMMA_RUST_STAGE=ffn_post_rms GEMMA_LLAMA_TENSOR_NODE=1481 ...

# previous layer outputs
GEMMA_RUST_LAYER_INDEX=30 GEMMA_RUST_STAGE=l_out_debug GEMMA_LLAMA_TENSOR_NODE=1342 ...
GEMMA_RUST_LAYER_INDEX=31 GEMMA_RUST_STAGE=l_out_debug GEMMA_LLAMA_TENSOR_NODE=1379 ...
GEMMA_RUST_LAYER_INDEX=32 GEMMA_RUST_STAGE=l_out_debug GEMMA_LLAMA_TENSOR_NODE=1416 ...
GEMMA_RUST_LAYER_INDEX=33 GEMMA_RUST_STAGE=l_out_debug GEMMA_LLAMA_TENSOR_NODE=1453 ...
```

Selected results for dimension `438`:

| stage | Rust | llama | diff | abs |
|---|---:|---:|---:|---:|
| layer 30 `l_out` | `58.553028` | `73.717209` | `-15.164181` | `15.164181` |
| layer 31 `l_out` | `59.366623` | `76.001755` | `-16.635132` | `16.635132` |
| layer 32 `l_out` | `55.834270` | `75.056450` | `-19.222179` | `19.222179` |
| layer 33 `l_out` | `48.253765` | `61.278255` | `-13.024490` | `13.024490` |
| layer 34 `ffn_down` | `-5.123775` | `-3.006088` | `-2.117687` | `2.117687` |
| layer 34 `ffn_post_rms` | `-6.718061` | `-2.898719` | `-3.819342` | `3.819342` |
| layer 34 `ffn_post_norm` | `-78.517334` | `-33.878780` | `-44.638554` | `44.638554` |

Interpretation:

- Dimension `438` is already the largest residual-stream error at layers 30-33.
- Layer 34 does not create that error from nothing; it converts a few units of RMS-normalized error into a huge activation error because `post_ffw_norm.weight[438] = 11.6875`.
- The root cause should be searched before layer 30, along the residual stream direction represented by dim `438`, not in the final RMSNorm/scale operator itself.

### Updated next-test priority

1. Track dimension `438` backward across earlier `l_out-*` nodes to find where it first becomes dominant.
2. At that first large jump, compare the local layer stages (`attn_out`, `ffn_down`, `ffn_post_norm`, PLE tail) for dim `438` and top diffs.
3. If dim `438` grows gradually rather than jumping, measure whether Q8 FFN matmul drift accumulates into that channel across many layers.
4. Add a small “watched dimensions” diagnostic, analogous to watched token IDs, so `dim=438` can be followed without dumping top-k each time.

### 56) Watched-dimension trace for dim 438 across all `l_out-*`

Change:

- Added `GEMMA_TENSOR_WATCH_DIMS=...` support to `tests/gemma_tensor_node_vs_llama.rs`.
- This prints selected dimensions for any stage comparison without needing them to be in the top-k diff list.

Run pattern:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=<layer> \
GEMMA_RUST_STAGE=l_out_debug \
GEMMA_LLAMA_TENSOR_NODE=<l_out_node> \
GEMMA_TENSOR_WATCH_DIMS=438 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture
```

Selected dim `438` trace:

| layer | node | Rust | llama | diff | max \|Δ\| |
|---:|---:|---:|---:|---:|---:|
| 0 | `63` | `0.635205` | `1.275623` | `-0.640418` | `2.553881` |
| 1 | `112` | `0.017344` | `-0.059028` | `0.076372` | `1.489656` |
| 9 | `505` | `-0.045440` | `-0.046279` | `0.000839` | `5.332256` |
| 14 | `750` | `-1.561090` | `-0.424618` | `-1.136472` | `10.207157` |
| 15 | `787` | `-3.297813` | `-1.235298` | `-2.062515` | `26.806328` |
| 19 | `935` | `-1.186558` | `-1.154358` | `-0.032199` | `20.488174` |
| 21 | `1009` | `1.526343` | `0.986739` | `0.539604` | `11.544056` |
| 22 | `1046` | `3.020917` | `8.132442` | `-5.111526` | `14.821548` |
| 23 | `1083` | `20.341799` | `40.765129` | `-20.423330` | `20.423330` |
| 24 | `1120` | `24.305092` | `47.733906` | `-23.428814` | `23.428814` |
| 25 | `1157` | `21.641487` | `35.858658` | `-14.217171` | `14.217171` |
| 30 | `1342` | `58.553028` | `73.717209` | `-15.164181` | `15.164181` |
| 31 | `1379` | `59.366623` | `76.001755` | `-16.635132` | `16.635132` |
| 32 | `1416` | `55.834270` | `75.056450` | `-19.222179` | `19.222179` |
| 33 | `1453` | `48.253765` | `61.278255` | `-13.024490` | `13.024490` |
| 34 | `1493` | `-3.111752` | `6.361789` | `-9.473540` | `9.473540` |

Interpretation:

- Dim `438` is small or moderate through layer 21.
- It grows at layer 22 (`diff=-5.11`) and jumps at layer 23 (`diff=-20.42`).
- Layer 23 is therefore the first clear local target for the dim-438 error direction.

### 57) Layer 23 stage-level localization for dim 438

Run pattern:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=23 \
GEMMA_RUST_STAGE=<stage> \
GEMMA_LLAMA_TENSOR_NODE=<node> \
GEMMA_TENSOR_WATCH_DIMS=438 \
GEMMA_TENSOR_TOP_DIFFS=5 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_e2b_tensor_node_vs_rust --release -- --ignored --nocapture
```

Layer 23 dim `438` results:

| stage | node | Rust | llama | diff | max \|Δ\| |
|---|---:|---:|---:|---:|---:|
| `attn_proj` | `1062` | `0.253172` | `1.501292` | `-1.248121` | `1.602021` |
| `attn_post_norm` | `1064` | `0.175034` | `0.958857` | `-0.783824` | `0.783824` |
| `attn_out` | `1065` | `3.195951` | `9.091300` | `-5.895349` | `15.011204` |
| `ffn_normed` | `1067` | `0.943615` | `2.623835` | `-1.680220` | `2.630053` |
| `ffn_down` | `1071` | `1.199100` | `6.497092` | `-5.297991` | `7.197575` |
| `ffn_post_rms` | `1072` | `3.354599` | `14.957105` | `-11.602506` | `17.029873` |
| `ffn_post_norm` | `1073` | `12.212838` | `54.453209` | `-42.240372` | `42.240372` |
| `pe_in` | `1074` | `15.408789` | `63.544510` | `-48.135719` | `48.135719` |
| `after_tail` | `1082` | `47.126701` | `94.442291` | `-47.315590` | `47.315590` |
| `l_out` | `1083` | `20.341799` | `40.765129` | `-20.423330` | `20.423330` |

Interpretation:

- Attention is not the main local cause for dim `438` in layer 23.
- The big local jump happens through FFN output normalization/scaling:
  - `ffn_down` dim-438 diff: `-5.30`
  - `ffn_post_rms` dim-438 diff: `-11.60`
  - `ffn_post_norm` dim-438 diff: `-42.24`
- PLE changes the absolute value strongly but does not remove the direction; `layer_output_scale` then dampens it to `l_out`.

### 58) Layer 23 exact-input checks

Runs:

```bash
GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=23 \
GEMMA_LLAMA_FFN_IN_NODE=1067 \
GEMMA_LLAMA_FFN_OUT_NODE=1071 \
GEMMA_TENSOR_MAX_ABS=1e9 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_single_layer_ffn_llama_io_parity --release -- --ignored --nocapture

GEMMA_LOGITS_TOKEN_IDS=2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818 \
GEMMA_RUST_LAYER_INDEX=23 \
GEMMA_LLAMA_FFN_OUT_NODE=1071 \
GEMMA_LLAMA_FFN_POST_RMS_NODE=1072 \
GEMMA_LLAMA_FFN_POST_MUL_NODE=1073 \
GEMMA_TENSOR_MAX_ABS=1e-3 \
  cargo test --test gemma_tensor_node_vs_llama gemma4_post_ffn_norm_llama_io_parity --release -- --ignored --nocapture
```

Results:

- Layer 23 FFN I/O parity on llama exact input: `max_abs=2.846788`, `rmse=0.123847`.
- Layer 23 post-FFN RMS/scale on llama exact input:
  - `rms_max=0.000000`, `rms_rmse=0.000000`
  - `scaled_max=0.000000`, `scaled_rmse=0.000000`

Interpretation:

- As with layer 34, the post-FFN RMS/scale operator is exact when given exact input.
- The FFN implementation has moderate Q8/numeric mismatch, but the severe full-pipeline dim-438 jump is mostly input/residual drift amplified through RMSNorm and the post-FFN scale.

### 59) Mistral generation control vs llama.cpp

Prompt:

```text
What is the capital of France?
```

Rust run:

```bash
cargo run --release -- \
  -m model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf \
  -t model/mistral-7b-v0.1/tokenizer.model \
  -n 32 \
  "What is the capital of France?"
```

Rust output:

```text
Paris is the capital of France.

What is the capital of Italy?

Rome is the capital of Italy.

What
```

llama.cpp raw completion run:

```bash
llama-completion \
  -m model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf \
  -p "What is the capital of France?" \
  -n 32 --temp 0 -ngl 0 --no-display-prompt
```

llama.cpp output:

```text
Paris is the capital of France.

What is the capital of Italy?

Rome is the capital of Italy.

What
```

Interpretation:

- The simple Mistral raw-completion control matches llama.cpp textually for this prompt and is coherent English.
- This argues against a generic "Rust language" or tokenizer/output loop problem. The severe bad-generation behavior remains localized to Gemma 4's architecture/numeric parity path.
- `llama-cli` defaulted to conversation mode for this model and produced a different but still coherent chat-formatted answer (`The capital of France is Paris...`); `llama-completion` is the closer raw-completion comparator for the Rust CLI.

### 60) Mistral logits control vs llama.cpp

Added an ignored direct logits comparator:

```bash
MISTRAL_LOGITS_PROMPT='What is the capital of France?' \
MISTRAL_LOGITS_TOP_K=10 \
  cargo test --test mistral_logits_vs_llama mistral_prefill_logits_match_llama --release -- --ignored --nocapture
```

Results:

| token-id sequence | llama argmax | Rust argmax | max \|Δlogit\| | RMSE | argmax mismatch |
|---|---:|---:|---:|---:|---:|
| prompt ids `[1,1824,349,272,5565,302,4843,28804]` | `13` (`\n`) | `13` (`\n`) | `0.269844` | `0.060145` | `0` |
| + `13` | `13` (`\n`) | `13` (`\n`) | `1.162677` | `0.289214` | `0` |
| + `13,13` | `3916` (`Par`) | `3916` (`Par`) | `0.328444` | `0.071538` | `0` |
| + `13,5465` diagnostic off-trajectory check | `13` (`\n`) | `13` (`\n`) | `0.251510` | `0.059649` | `0` |

Interpretation:

- Mistral is not bit-identical to llama.cpp, but the drift is small and top-token ordering remains stable on the tested path.
- This is the contrast we wanted: generation can match text while logits still drift slightly. For Mistral that drift is sub-logit scale; for Gemma 4 it grows to tens of logits and flips a near-tie early in decode.
- The remaining Gemma issue is therefore much more likely in Gemma-specific architecture/numeric surfaces than in the shared Llama/Mistral stack.

### 61) Q8 is an execution path, not just smaller weights

Key point:

- GGUF `Q8_0` stores each block as one fp16 scale plus 32 signed int8 quants (`block_q8_0`).
- A mathematically reasonable implementation can dequantize those weights to f32 and do a normal f32 dot.
- llama.cpp's hot path usually does something more specific: quantize the activation row to Q8_0, then run a Q8×Q8 block dot (`ggml_vec_dot_q8_0_q8_0`) with the same block scales, integer inner sums, and accumulation order used by ggml.

Why this matters:

- The effective model is not only "weights rounded to 8 bits"; it is `quantized weights + activation quantization + block dot + accumulation/rounding order`.
- Mistral shows that small implementation drift can be benign (`RMSE < 0.3`, same argmax).
- Gemma 4 shows that the same class of drift can be amplified by high-gain norms/scales/PLE/residual paths until it flips a near-tie.

Repo implication:

- `src/ops/matmul.rs::matmul_f32_q8_0` is the central place to match llama.cpp's Q8 path.
- The remaining work was to make the Q8 quantize/dot details in `src/ops/quant/quant_K_handler.rs` match ggml exactly, then apply that one kernel consistently through all Q8 linear projections.

### 62) Q8 path hardened to one ggml-like execution mode

Code change:

- Removed the `INFERENCE_ENGINE_Q8_DEQUANT_F32` branch from `src/ops/matmul.rs::matmul_f32_q8_0`; Q8 linear layers now always quantize the activation row and run Q8×Q8 block dot.
- Removed env-gated A/B variants from `src/ops/quant/quant_K_handler.rs`:
  - `INFERENCE_ENGINE_Q8_ID_FROM_F16`
  - `INFERENCE_ENGINE_Q8_TIES_EVEN`
  - `INFERENCE_ENGINE_Q8_CLAMP_127`
  - `INFERENCE_ENGINE_Q8_DOT_F64_ACCUM`
- The remaining scalar path mirrors upstream ggml's generic behavior:
  - `d = amax / 127`
  - store `d` as fp16
  - compute `id = 1 / d` from the original f32 scale
  - quantize with round-away-from-zero semantics (`roundf` / Rust `round`)
  - dot with per-block integer sums accumulated in f32

Validation:

```bash
cargo test --lib test_matmul
```

Result: `4 passed`.

Mistral logits control after the cleanup:

```bash
MISTRAL_LOGITS_PROMPT='What is the capital of France?' \
MISTRAL_LOGITS_TOP_K=5 \
  cargo test --test mistral_logits_vs_llama mistral_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result unchanged:

- `max |Delta logit| = 0.269844`
- `RMSE = 0.060145`
- argmax match: `13` (`\n`)

Gemma problematic continuation after the cleanup:

```bash
GEMMA_LOGITS_TOKEN_IDS='2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,818' \
GEMMA_LOGITS_TOP_K=10 \
GEMMA_LOGITS_WATCH_TOKEN_IDS='5279,1018' \
  cargo test --test gemma_logits_vs_llama gemma4_e2b_prefill_logits_match_llama --release -- --ignored --nocapture
```

Result:

- llama.cpp argmax: `5279` (` capital`)
- Rust argmax: `1018` (`**`)
- `max |Δlogit| = 25.003351`
- `RMSE = 1.606843`
- watched margin `5279 - 1018`:
  - llama.cpp: `36.457222`
  - Rust: `-7.009971`

Interpretation:

- The Q8 code path is now less ambiguous and closer to the intended ggml scalar semantics.
- This cleanup did not fix the Gemma argmax flip, so the remaining error is not caused by accidentally using the old f32-dequantized Q8 matmul branch.
- Continue with the layer 22 → 23 / dim-438 bisection; Gemma-specific amplification remains the leading hypothesis.

### 63) f16 KV cache + flash accumulation precision A/B

Code change:

- `KVCache::append_kv` now rounds K/V values through fp16 by default before storing them, matching the value precision llama.cpp exposes when attention reads from its f16 KV cache.
- The cache still stores rounded values in `Vec<f32>` internally so existing `get_k_slice` / `get_v_slice` readers do not need to allocate or change signatures.
- Added `INFERENCE_ENGINE_F16_KV_CACHE=0` to disable this rounding for A/B tests.
- Added an opt-in `INFERENCE_ENGINE_FLASH_F16_ACCUM=1` approximation that rounds Q values and the online value accumulator through fp16. This stays disabled by default because it is not a faithful enough match for ggml's backend-specific flash kernel.

Validation:

```bash
cargo test --lib
```

Result: `31 passed`, `10 ignored`.

Gemma problematic continuation, previous Q8-hardened baseline:

- Rust argmax: `1018` (`**`)
- `max |Δlogit| = 25.003351`
- `RMSE = 1.606843`
- watched margin `5279 - 1018 = -7.009971`

With f16 KV cache enabled and f16 flash accumulator disabled (new default):

- Rust argmax: `1018` (`**`)
- `max |Δlogit| = 23.662477`
- `RMSE = 1.494872`
- watched margin `5279 - 1018 = -1.390570`

With f16 KV cache + f16 flash accumulator approximation:

- Rust argmax: `1018` (`**`)
- `max |Δlogit| = 25.985569`
- `RMSE = 1.710122`
- watched margin `5279 - 1018 = -10.029787`

With f32 KV cache + f16 flash accumulator approximation:

- Rust argmax: `1346` (` most`)
- `max |Δlogit| = 26.387863`
- `RMSE = 1.460960`
- watched margin `5279 - 1018 = -6.338950`

Mistral control with f16 KV + f16 accumulator approximation:

- `max |Delta logit| = 0.269780`
- `RMSE = 0.060121`
- same argmax (`13`, `\n`)

Additional split A/B after inspecting ggml `FLASH_ATTN_EXT`:

- ggml CPU flash has two separable precision choices:
  - convert Q to the K vec-dot type for KQ dot (`q_to_vec_dot`, f16 when K is f16)
  - use an f16 VKQ accumulator only in the non-tiled one-chunk path when V is f16
- Split the previous combined switch into:
  - `INFERENCE_ENGINE_FLASH_Q_F16_DOT=1`
  - `INFERENCE_ENGINE_FLASH_V_F16_ACCUM=1`

Results with f16 KV cache:

| mode | Rust argmax | max \|Δlogit\| | RMSE | watched margin `5279-1018` |
|---|---:|---:|---:|---:|
| default: f32 Q dot + f32 V accum | `1018` (`**`) | `23.662477` | `1.494872` | `-1.390570` |
| Q f16 dot only | `1018` (`**`) | `26.198515` | `1.464520` | `-11.661889` |
| V f16 accum only | `1346` (` most`) | `26.692537` | `1.591781` | `-0.853662` |
| Q f16 dot + V f16 accum | `1018` (`**`) | `25.985569` | `1.710122` | `-10.029787` |

Interpretation:

- f16 KV cache rounding is a real improvement for the Gemma continuation, especially the watched margin (`-7.01` → `-1.39`), but not enough to recover the llama argmax.
- The split flash precision switches show there is no simple fp16 round-trip that recovers llama here:
  - Q f16 dot hurts the watched margin.
  - V f16 accumulation improves that specific margin but worsens argmax and RMSE.
- These remain opt-in diagnostics, not defaults. The default should stay f16 KV + f32 online accumulation until a closer ggml flash implementation is added.
- Continue localizing layer 22 → 23; f16 KV reduced the symptom but did not remove the underlying Gemma-specific drift.

### 64) Confirmed llama.cpp reference attention mode

Code change:

- Updated `tools/llama_logits_ref.c`, `tools/llama_hidden_ref.c`, and `tools/llama_tensor_dump_ref.c` to:
  - print `flash_attn`, `type_k`, `type_v`, `offload_kqv`, and `op_offload`
  - accept `LLAMA_FLASH_ATTN=auto|enabled|disabled|1|0`
- Rebuilt with:

```bash
./tools/build_llama_logits_ref.sh
```

Probe:

```bash
LLAMA_FLASH_ATTN=auto ./tools/llama_logits_ref \
  model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf 2 105 >/tmp/llama_ref_probe.bin
```

Reported context:

```text
llama_logits_ref: flash_attn=auto type_k=f16 type_v=f16 offload_kqv=0 op_offload=0
llama_context: flash_attn    = auto
sched_reserve: Flash Attention was auto, set to enabled
```

Trace probe:

```bash
LLAMA_FLASH_ATTN=auto LLAMA_TENSOR_DUMP_TRACE=1 ./tools/llama_tensor_dump_ref \
  model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf 2 105 \
  >/tmp/gemma_trace_probe.bin 2>/tmp/gemma_trace_probe.err
```

Trace result:

- One `FLASH_ATTN_EXT` node appears per layer, named `__fattn__-0` through `__fattn__-34`.
- The alternating output widths match Gemma 4 sliding/global attention head dimensions:
  - most layers: `f32 256 8 <seq> 1`
  - global layers: `f32 512 8 <seq> 1`

Flash mode comparison on the problematic 17-token continuation, with Rust at current default (`f16 KV`, f32 online accumulation):

| llama ref mode | llama argmax | llama watched margin `5279-1018` | Rust argmax | max \|Δlogit\| | RMSE |
|---|---:|---:|---:|---:|---:|
| `LLAMA_FLASH_ATTN=auto` | `5279` (` capital`) | `36.457222` | `1018` (`**`) | `23.662477` | `1.494872` |
| `LLAMA_FLASH_ATTN=enabled` | `5279` (` capital`) | `36.457222` | `1018` (`**`) | `23.662477` | `1.494872` |
| `LLAMA_FLASH_ATTN=disabled` | `5279` (` capital`) | `35.578697` | `1018` (`**`) | `23.654140` | `1.489349` |

Interpretation:

- The reference target is definitely f16 K/V cache + `FLASH_ATTN_EXT`; `auto` resolves to enabled.
- Disabling flash in llama.cpp changes logits only slightly on this case and does not explain the Rust argmax flip.
- The remaining mismatch is not high-level "flash on/off"; it is either exact ggml flash kernel numerics/order, an upstream layer drift before attention, or Gemma-specific layer behavior around shared KV/PLE/post-FFN amplification.

### 65) Layer 23 refreshed stage table after f16 KV

After enabling f16 KV cache rounding by default, reran the layer-23 node-aligned checks on the problematic 17-token continuation.

Trace nodes:

```text
1060 FLASH_ATTN_EXT __fattn__-23
1061 RESHAPE        kqv_out-23
1062 MUL_MAT        attn_proj
1064 MUL            attn_post_norm-23
1065 ADD            attn_out-23
1067 MUL            ffn_norm-23
1068 MUL_MAT        ffn_gate-23
1069 MUL_MAT        ffn_up-23
1070 GEGLU          ffn_geglu-23
1071 MUL_MAT        ffn_out-23
1073 MUL            ffn_post_norm-23
1074 ADD            pe_in-23
1082 ADD            after_tail
1083 MUL            l_out-23
```

Key results:

| stage | node | max \|Δ\| | dim 438 Rust | dim 438 llama | dim 438 diff |
|---|---:|---:|---:|---:|---:|
| `attn_core` (`__fattn__`) | `1060` | `11.293030` | `0.016864` | `-0.124573` | `0.141437` |
| `attn_proj` | `1062` | `1.558852` | `0.257400` | `1.501292` | `-1.243892` |
| `ffn_normed` | `1067` | `2.671978` | `1.023650` | `2.623835` | `-1.600185` |
| `ffn_gate` | `1068` | `2.319129` | `-0.501369` | `-0.594968` | `0.093599` |
| `ffn_up` | `1069` | `2.672210` | `0.014475` | `-0.130447` | `0.144921` |
| `ffn_geglu` | `1070` | `2.717242` | `-0.001753` | `0.021417` | `-0.023170` |
| `ffn_down` | `1071` | `7.169326` | `1.210544` | `6.497092` | `-5.286548` |
| `ffn_post_norm` | `1073` | `42.067539` | `12.385671` | `54.453209` | `-42.067539` |
| `pe_in` | `1074` | `47.694328` | `15.850183` | `63.544510` | `-47.694328` |
| `after_tail` | `1082` | `46.961411` | `47.480881` | `94.442291` | `-46.961411` |
| `l_out_debug` | `1083` | `20.270452` | `20.494677` | `40.765129` | `-20.270452` |

Interpretation:

- The `FLASH_ATTN_EXT` output is not perfect (`max |Δ| ~11.3`), but dim `438` is still close at attention core (`0.14`) and only modestly off after `attn_proj` (`-1.24`).
- FFN input drift is moderate (`ffn_normed` dim-438 diff `-1.60`).
- Gate/up/GeGLU do not directly create the dim-438 issue (`dim 438` is near-zero at `ffn_geglu`).
- `w_down` concentrates the upstream FFN-vector drift into hidden dim `438` (`-5.29`), and post-FFN RMS/scale amplifies it to `-42.07`.
- Next highest-signal llama-mimic test: feed llama's exact `ffn_geglu-23` into Rust `w_down` and compare to llama `ffn_out-23`. That isolates the Q8 `w_down` matmul from upstream activation drift.

### 66) Root cause found: Gemma GeGLU used the wrong tanh-GELU constant

The exact-input `w_down` isolation was decisive:

```text
FFN down exact-input parity:
  layer=23 geglu_node=1070 out_node=1071
  max_abs=0.000001 rmse=0.000000
  dim 438 Rust=6.497092 llama=6.497092
```

So `w_down` Q8 matmul is not the root cause. Feeding llama's exact `ffn_normed-23` into Rust and comparing FFN intermediates then showed:

```text
ffn_gate_exact_input:  max_abs=0.000000 rmse=0.000000
ffn_up_exact_input:    max_abs=0.000000 rmse=0.000000
ffn_geglu_exact_input: max_abs=0.234705 rmse=0.026214
```

Watched top-diff dimensions made the bug obvious:

```text
dim=7148 gate=-1.008506 up=-3.232495
Rust GeGLU=0.276292 llama GeGLU=0.510996
```

Rust's `gelu_tanh` used `FRAC_2_SQRT_PI` (`2/sqrt(pi)`) as the tanh coefficient. PyTorch/ggml tanh-GELU uses `sqrt(2/pi)`, which is `FRAC_2_SQRT_PI * FRAC_1_SQRT_2`. The old constant made negative gate values too small in magnitude before multiplication by `up`, creating a broad GeGLU-vector error. `w_down` then projected that broad vector error into hidden dim `438`, and post-FFN RMS/scale amplified it.

After fixing `src/ops/gelu.rs`:

```text
ffn_geglu_exact_input: max_abs=0.001404 rmse=0.000041
single-layer FFN parity: layer=23 in_node=1067 out_node=1071 max_abs=0.001885 rmse=0.000551
stage=ffn_down node=1071:
  max |Δ| = 0.958935
  dim 438 Rust=6.587685 llama=6.497092 diff=0.090593

Gemma problematic continuation logits:
  llama.cpp argmax token id: 5279 (" capital")
  Rust        argmax token id: 5279 (" capital")
  max |Δlogit|: 9.827078
  RMSE: 1.494955
  watched margin id5279-id1018:
    llama = 36.457222
    Rust  = 28.358944
```

Short generation smoke test:

```text
User: What is the capital of France?
Assistant: The capital of France is **Paris**.
```

Interpretation:

- The primary garbled-output root cause was not Rust, not the tokenizer, and not Q8 `w_down`; it was a one-constant GELU bug that only Gemma exposed strongly because Gemma uses GeGLU here.
- Remaining logit drift is still non-trivial (`RMSE ~1.49`), but the decisive argmax flip on the problematic continuation is fixed.
- The remaining drift is now likely the already-known exact attention/shared-KV/PLE accumulation differences, not this FFN semantic bug.

### Updated next-test priority

1. Rerun a few Gemma generation prompts to confirm visible output quality is broadly improved, not just the France continuation.
2. Refresh final layer/logit parity after the GELU fix; the old layer-23 `ffn_down`/dim-438 diagnosis is now mostly resolved.
3. Continue attention/shared-KV parity only if remaining prompts still show bad argmax flips.
4. Keep exact-input FFN intermediate tests as regression tooling for future Gemma numeric work.
