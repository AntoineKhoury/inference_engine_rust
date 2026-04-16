//! CLI: greedy continuation from a text prompt (same pipeline as `tests/generate_smoke.rs`).
//!
//! ```text
//! cargo run --release -- --help
//! cargo run --release -- "Rust will rule the"
//! cargo run --release -- -n 32 -m model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf "Hello"
//! cargo run --release -- --chat gemma4-e2b -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
//!   -t model/gemma-4-e2b-it/tokenizer.json "Hello"
//! ```

use std::path::PathBuf;

use clap::Parser;
use inference_engine_rust::EngineError;
use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{
    decode_forward, final_logits_last_token, prefill_forward, prefill_from_tokens,
    prefill_state_for_single_token_loaded,
};
use inference_engine_rust::sampling::sample_greedy;
use inference_engine_rust::chat_prompt::ChatPromptStyle;
use inference_engine_rust::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "inference_engine_rust")]
#[command(about = "Greedy LM generation (GGUF + tokenizer .model or .json)", long_about = None)]
struct Args {
    /// GGUF model path (relative to cwd is fine)
    #[arg(
        short,
        long,
        default_value = "model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf"
    )]
    model: PathBuf,

    /// Tokenizer: SentencePiece `tokenizer.model` or Hugging Face `tokenizer.json`
    #[arg(short, long, default_value = "model/mistral-7b-v0.1/tokenizer.model")]
    tokenizer: PathBuf,

    /// How many new tokens to append after the prompt
    #[arg(short = 'n', long, default_value_t = 20)]
    new_tokens: usize,

    /// Wrap PROMPT for instruct/chat: `raw` (default), `mistral-instruct`, `gemma4-e2b`
    #[arg(long, default_value = "raw")]
    chat: String,

    /// Prompt text. If omitted, one line is read from stdin
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,
}

fn main() -> Result<(), EngineError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args = Args::parse();

    let prompt = match args.prompt {
        Some(p) if !p.trim().is_empty() => p,
        Some(_) => {
            return Err(EngineError::Model("prompt is empty".into()));
        }
        None => {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            line.trim_end_matches(['\n', '\r']).to_string()
        }
    };

    if prompt.is_empty() {
        return Err(EngineError::Model(
            "no prompt: pass PROMPT or pipe a line on stdin".into(),
        ));
    }

    let chat_style = ChatPromptStyle::parse(&args.chat).ok_or_else(|| {
        EngineError::Model(format!(
            "unknown --chat {:?}: use raw | mistral-instruct | gemma4-e2b",
            args.chat
        ))
    })?;
    let prompt = chat_style.wrap(&prompt);

    if !args.model.is_file() {
        return Err(EngineError::Model(format!(
            "model file not found: {}",
            args.model.display()
        )));
    }
    if !args.tokenizer.is_file() {
        return Err(EngineError::Model(format!(
            "tokenizer file not found: {}",
            args.tokenizer.display()
        )));
    }

    let model_path = args
        .model
        .to_str()
        .ok_or_else(|| EngineError::Model("model path is not valid UTF-8".into()))?;
    let mut gguf = read_file(model_path)?;

    let mut tokenizer = Tokenizer::load_from_file(&args.tokenizer)?;
    let tok_prompt = TokenizerPromptConfig::from_gguf(&gguf)?;
    let prompt_ids = tokenizer.encode_with_prompt_config(&prompt, &tok_prompt)?;

    let config = ModelConfig::from_gguf(&gguf)?;
    let names = ModelWeightNames::resolve(&gguf, &config)?;
    names.load_all(&mut gguf, model_path)?;

    let prefill_in = prefill_from_tokens(&mut gguf, model_path, &config, &prompt_ids)?;
    let weights = ModelWeights::from_loaded(&gguf, &names)?;

    let mut kv_caches = kv_caches_for_config(&config);

    let mut state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches)?;

    let mut generated = Vec::with_capacity(args.new_tokens);
    for _ in 0..args.new_tokens {
        let logits = final_logits_last_token(&state, &config, &weights)?;
        let next_id = sample_greedy(&logits)?;
        generated.push(next_id);

        let step_in = prefill_state_for_single_token_loaded(&gguf, &config, next_id)?;
        state = decode_forward(&step_in, &config, &weights, &mut kv_caches)?;
    }

    let continuation = tokenizer.decode_piece_ids(&generated)?;

    println!("{continuation}");
    Ok(())
}
