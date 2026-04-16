# Local model checkpoints (not committed)

Place each GGUF and its tokenizer in a **per-model directory** under `model/`.

| Directory | Checkpoint | Tokenizer | Tokenizer source |
|-----------|------------|-----------|------------------|
| `model/mistral-7b-v0.1/` | `mistral-7b-v0.1.Q4_K_M.gguf` | `tokenizer.model` (SentencePiece) | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer.model) |
| `model/gemma-4-e2b-it/` | `gemma-4-E2B-it-Q8_0.gguf` (or other quant) | `tokenizer.json` (Hugging Face) | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it/blob/main/tokenizer.json) |

**Chat templates (Jinja):** tracked under each model dir for reference. **Gemma:** `model/gemma-4-e2b-it/chat_template.jinja` (same as Hub). **Mistral-7B-v0.1 base** has no official chat format; `model/mistral-7b-v0.1/chat_template.jinja` is the template from **[Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)** (`tokenizer_config.json` → `chat_template`) for use with **Instruct** checkpoints and `cargo run -- --chat mistral-instruct`.

The inference CLI loads **`.model`** as SentencePiece and **`.json`** as Hugging Face `tokenizer.json` (same format as Transformers).

## Example layout

```text
model/
  README.md                 ← this file (tracked in git)
  mistral-7b-v0.1/
    mistral-7b-v0.1.Q4_K_M.gguf
    tokenizer.model
    chat_template.jinja   # Instruct v0.2 (optional; base model is completion-only)
  gemma-4-e2b-it/
    gemma-4-E2B-it-Q8_0.gguf
    tokenizer.json
    chat_template.jinja   # from google/gemma-4-E2B-it on Hugging Face
```

## Download snippets

**Mistral GGUF** (example quant):

```bash
mkdir -p model/mistral-7b-v0.1
curl -fL -o model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
curl -fL -o model/mistral-7b-v0.1/tokenizer.model \
  "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/tokenizer.model"
```

**Gemma 4 E2B IT** (Q8 GGUF + HF tokenizer):

```bash
mkdir -p model/gemma-4-e2b-it
curl -fL -o model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q8_0.gguf"
curl -fL -o model/gemma-4-e2b-it/tokenizer.json \
  "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer.json"
curl -fL -o model/gemma-4-e2b-it/chat_template.jinja \
  "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/chat_template.jinja"
```

**Mistral Instruct chat template** (for reference; also committed as `model/mistral-7b-v0.1/chat_template.jinja`): extract `chat_template` from [tokenizer_config.json](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/raw/main/tokenizer_config.json) or copy from this repo.

## Run

```bash
# Mistral
cargo run --release -- \
  -m model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf \
  -t model/mistral-7b-v0.1/tokenizer.model \
  "Hello"

# Gemma 4 E2B IT (instruct-style prompt)
cargo run --release -- \
  --chat gemma4-e2b \
  -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  -t model/gemma-4-e2b-it/tokenizer.json \
  "Hello"
```

## Interactive chat (multi-turn REPL)

Binary: **`chat`** (`cargo run --release --bin chat`). Uses the same formatting as `model/*/chat_template.jinja` (subset: user/assistant turns only).

```bash
# Gemma 4 E2B IT
cargo run --release --bin chat -- \
  --style gemma4-e2b \
  -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
  -t model/gemma-4-e2b-it/tokenizer.json

# Mistral **Instruct** checkpoint (not base v0.1)
cargo run --release --bin chat -- \
  --style mistral-instruct \
  -m path/to/mistral-instruct.Q4_K_M.gguf \
  -t path/to/tokenizer.model
```

Type `/quit` or `/exit` to leave. Optional: `--max-reply-tokens 512`, `--stop-token <id>`.
