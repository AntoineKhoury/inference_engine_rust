//! Interactive multi-turn chat (instruct-style prompts only).
//!
//! ```text
//! cargo run --release --bin chat -- --help
//! cargo run --release --bin chat -- --style gemma4-e2b \
//!   -m model/gemma-4-e2b-it/gemma-4-E2B-it-Q8_0.gguf \
//!   -t model/gemma-4-e2b-it/tokenizer.json
//! cargo run --release --bin chat -- --style mistral-instruct \
//!   -m model/mistral-7b-v0.1/mistral-instruct.Q4_K_M.gguf \
//!   -t model/mistral-7b-v0.1/tokenizer.model
//! ```
//!
//! Uses **full-context prefill each turn** (simple, correct; slower on very long chats).
//! There is **no output until prefill finishes** unless you watch stderr (`… prompt tokens…`).
//! Type `/quit` or Ctrl-D to exit.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use clap::Parser;
use inference_engine_rust::EngineError;
use inference_engine_rust::chat_prompt::{
    ChatMessage, ChatPromptStyle, gemma4_e2b_assistant_visible,
    gemma4_e2b_decode_has_structure_marker,
};
use inference_engine_rust::layers::attention::kv_caches_for_config;
use inference_engine_rust::model_config::{ModelConfig, TokenizerPromptConfig};
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_weights::{ModelWeightNames, ModelWeights};
use inference_engine_rust::prefill::{
    prefill_from_tokens_loaded, prefill_state_for_single_token_loaded,
};
use inference_engine_rust::runtime::{decode_forward, final_logits_last_token, prefill_forward};
use inference_engine_rust::sampling::sample_greedy;
use inference_engine_rust::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "chat")]
#[command(about = "Multi-turn instruct chat (Gemma 4 E2B or Mistral Instruct formatting)", long_about = None)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    tokenizer: PathBuf,

    /// `gemma4-e2b` or `mistral-instruct`
    #[arg(long)]
    style: String,

    /// Max new tokens per assistant reply
    #[arg(long, default_value_t = 256)]
    max_reply_tokens: usize,

    /// Stop if model emits this token id (overrides GGUF eos when set)
    #[arg(long)]
    stop_token: Option<u32>,

    /// Print the assistant reply only after the full decode (no token-by-token streaming)
    #[arg(long)]
    no_stream: bool,
}

fn main() -> Result<(), EngineError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let args = Args::parse();
    let style = ChatPromptStyle::parse(&args.style).ok_or_else(|| {
        EngineError::Model(format!(
            "unknown --style {:?}: use gemma4-e2b | mistral-instruct",
            args.style
        ))
    })?;
    if matches!(style, ChatPromptStyle::Raw) {
        return Err(EngineError::Model(
            "use --style gemma4-e2b or mistral-instruct for chat".into(),
        ));
    }

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

    let config = ModelConfig::from_gguf(&gguf)?;
    let names = ModelWeightNames::resolve(&gguf, &config)?;
    names.load_all(&mut gguf, model_path)?;
    let weights = ModelWeights::from_loaded(&gguf, &names)?;

    let stop_id = args.stop_token.unwrap_or(tok_prompt.eos_token_id);

    eprintln!(
        "Chat ({:?}). Commands: /quit /exit. EOS stop id: {}",
        style, stop_id
    );
    eprintln!("— — —");

    let mut history: Vec<ChatMessage> = Vec::new();
    let stdin = std::io::stdin();

    loop {
        eprint!("You> ");
        std::io::stderr().flush().ok();

        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            eprintln!("\nbye");
            break;
        }
        let line = line.trim_end_matches(['\n', '\r']);
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            eprintln!("bye");
            break;
        }

        history.push(ChatMessage::user(line.to_string()));

        let prompt_text = style
            .render_conversation(&history)
            .map_err(|e| EngineError::Model(e.to_string()))?;

        let prompt_ids = tokenizer.encode_with_prompt_config(&prompt_text, &tok_prompt)?;

        eprintln!(
            "… running (prefill {} prompt tokens, up to {} new); CPU can take a while …",
            prompt_ids.len(),
            args.max_reply_tokens
        );
        std::io::stderr().flush().ok();

        let mut kv_caches = kv_caches_for_config(&config);
        let prefill_in = prefill_from_tokens_loaded(&gguf, &config, &prompt_ids)?;
        let mut state = prefill_forward(&prefill_in, &config, &weights, &mut kv_caches)?;

        let stream = !args.no_stream;
        let mut generated: Vec<u32> = Vec::new();
        let mut decoded_prefix = String::new();

        if stream {
            print!("Assistant> ");
            std::io::stdout().flush().ok();
        }

        for _ in 0..args.max_reply_tokens {
            let logits = final_logits_last_token(&state, &config, &weights)?;
            let next_id = sample_greedy(&logits)?;
            if next_id == stop_id {
                break;
            }
            generated.push(next_id);

            let mut stop_on_gemma_structure = false;
            if stream {
                let full = tokenizer.decode_piece_ids(&generated)?;
                let safe = if matches!(style, ChatPromptStyle::Gemma4E2b) {
                    gemma4_e2b_assistant_visible(&full)
                } else {
                    full.trim_end().to_string()
                };
                if safe.starts_with(&decoded_prefix) {
                    print!("{}", &safe[decoded_prefix.len()..]);
                } else {
                    // Rare: decoded string is not a prefix extension; show the new token only.
                    let piece = tokenizer.decode_piece_ids(std::slice::from_ref(&next_id))?;
                    print!("{}", piece);
                }
                std::io::stdout().flush().ok();
                decoded_prefix = safe;
                if matches!(style, ChatPromptStyle::Gemma4E2b)
                    && gemma4_e2b_decode_has_structure_marker(&full)
                {
                    stop_on_gemma_structure = true;
                }
            } else if matches!(style, ChatPromptStyle::Gemma4E2b) {
                let full = tokenizer.decode_piece_ids(&generated)?;
                if gemma4_e2b_decode_has_structure_marker(&full) {
                    stop_on_gemma_structure = true;
                }
            }

            if stop_on_gemma_structure {
                break;
            }

            let step_in = prefill_state_for_single_token_loaded(&gguf, &config, next_id)?;
            state = decode_forward(&step_in, &config, &weights, &mut kv_caches)?;
        }

        let raw = tokenizer.decode_piece_ids(&generated)?;
        let reply = if matches!(style, ChatPromptStyle::Gemma4E2b) {
            gemma4_e2b_assistant_visible(&raw)
        } else if stream {
            decoded_prefix.trim_end().to_string()
        } else {
            raw.trim_end().to_string()
        };

        if stream {
            println!();
        } else {
            println!("Assistant> {}", reply.trim_end());
        }
        println!();

        history.push(ChatMessage::assistant(reply));
    }

    Ok(())
}
