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
use inference_engine_rust::chat_prompt::{
    ChatPromptStyle, gemma4_e2b_assistant_visible, gemma4_e2b_decode_has_structure_marker,
};
use inference_engine_rust::engine::generation::greedy_next_token;
use inference_engine_rust::engine::session::InferenceSession;
use inference_engine_rust::loaded_model::LoadedModel;
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

    if !args.tokenizer.is_file() {
        return Err(EngineError::Model(format!(
            "tokenizer file not found: {}",
            args.tokenizer.display()
        )));
    }

    let model = LoadedModel::load(&args.model)?;
    let mut tokenizer = Tokenizer::load_from_file(&args.tokenizer)?;
    let tok_prompt = model.tokenizer_prompt();

    let prompt_ids = tokenizer.encode_with_prompt_config(&prompt, tok_prompt)?;
    let mut session = InferenceSession::new(&model)?;
    let mut state = session.prefill(&prompt_ids)?;

    let stop_id = tok_prompt.eos_token_id;
    let mut generated = Vec::with_capacity(args.new_tokens);
    for _ in 0..args.new_tokens {
        let next_id = greedy_next_token(&session, &state)?;
        if next_id == stop_id {
            break;
        }
        generated.push(next_id);
        if matches!(chat_style, ChatPromptStyle::Gemma4E2b) {
            let full = tokenizer.decode_piece_ids(&generated)?;
            if gemma4_e2b_decode_has_structure_marker(&full) {
                break;
            }
        }

        state = session.decode_token(next_id)?;
    }

    let raw = tokenizer.decode_piece_ids(&generated)?;
    let continuation = if matches!(chat_style, ChatPromptStyle::Gemma4E2b) {
        gemma4_e2b_assistant_visible(&raw)
    } else {
        raw.trim_end().to_string()
    };

    println!("{continuation}");
    Ok(())
}
