//! Chat formatting for instruct models (subset of HF `chat_template` / Jinja behavior).
//!
//! Full Jinja templates live on Hugging Face; see `model/*/chat_template.jinja` in this repo.

/// Substrings that should not appear in **assistant-visible** chat text. The Jinja template injects
/// `<turn|>` **after** assistant content when serializing training/inference prompts; multimodal
/// and tool blocks use `<|audio|>`, `<|image|>`, etc. Greedy decode often keeps predicting those
/// tokens even when [`TokenizerPromptConfig::eos_token_id`] does not fire first—trim and stop.
pub const GEMMA4_E2B_ASSISTANT_STOP_MARKERS: &[&str] = &[
    "<turn|>",
    "<turn|", // incomplete closer before final `>`
    "<|turn>", // new role header (hallucinated continuation)
    "<|turn",
    "<channel|>",
    "<|channel>",
    "<|channel",
    "<|think|>",
    "<|think",
    "<|tool_call",
    "<|tool_response",
    "<|tool>",
    "<tool|>",
    "<|tool",
    "<|audio",
    "<|image",
    "<|video",
];

/// Cut `s` before the first Gemma E2B structure / modality marker (if any).
pub fn gemma4_e2b_truncate_assistant_at_structure(s: &str) -> &str {
    let mut end = s.len();
    for m in GEMMA4_E2B_ASSISTANT_STOP_MARKERS {
        if let Some(i) = s.find(m) {
            end = end.min(i);
        }
    }
    &s[..end]
}

/// `true` if [`gemma4_e2b_truncate_assistant_at_structure`] shortens `s`.
pub fn gemma4_e2b_decode_has_structure_marker(s: &str) -> bool {
    gemma4_e2b_truncate_assistant_at_structure(s).len() < s.len()
}

/// After cutting at a hallucinated `<|turn…` marker, decode often ends with a **partial contraction**
/// (`… I'`) that was meant to continue as `I'll` but merged with `<|turn`. Drop a trailing
/// **single ASCII letter + apostrophe** when it is its own final “word”.
pub fn gemma4_e2b_strip_dangling_contraction_tail(s: &str) -> &str {
    let s = s.trim_end();
    let Some(no_quote) = s.strip_suffix('\'') else {
        return s;
    };
    let no_quote = no_quote.trim_end();
    let tail_start = no_quote
        .char_indices()
        .rfind(|(_, c)| c.is_whitespace())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let tail = &no_quote[tail_start..];
    if tail.chars().count() == 1
        && tail
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
    {
        return no_quote[..tail_start].trim_end();
    }
    s
}

/// Truncate at template markers, then tidy partial contractions **only if** a marker was present.
pub fn gemma4_e2b_assistant_visible(s: &str) -> String {
    let had_marker = gemma4_e2b_decode_has_structure_marker(s);
    let t = gemma4_e2b_truncate_assistant_at_structure(s);
    let t = if had_marker {
        gemma4_e2b_strip_dangling_contraction_tail(t)
    } else {
        t
    };
    t.trim_end().to_string()
}

/// Role in a back-and-forth transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// How to turn the CLI “prompt” string into model input text before tokenization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChatPromptStyle {
    /// Pass the prompt through unchanged (raw completion).
    #[default]
    Raw,
    /// Mistral-7B-Instruct-v0.2-style (see `model/mistral-7b-v0.1/chat_template.jinja`).
    MistralInstruct,
    /// Gemma 4 E2B IT (see `model/gemma-4-e2b-it/chat_template.jinja`, single-user subset).
    Gemma4E2b,
}

impl ChatPromptStyle {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "raw" | "none" => Some(Self::Raw),
            "mistral" | "mistral-instruct" | "instruct" => Some(Self::MistralInstruct),
            "gemma" | "gemma4" | "gemma4-e2b" | "gemma-e2b" => Some(Self::Gemma4E2b),
            _ => None,
        }
    }

    /// Single-turn: wrap one user utterance (no prior assistant text).
    pub fn wrap(self, user: &str) -> String {
        let user = user.trim();
        match self {
            ChatPromptStyle::Raw => user.to_string(),
            ChatPromptStyle::MistralInstruct => {
                format!("[INST] {user} [/INST]")
            }
            ChatPromptStyle::Gemma4E2b => {
                // Matches HF `chat_template.jinja`: user turn ends with `<turn|>\n`, then `add_generation_prompt`.
                format!("<|turn>user\n{user}<turn|>\n<|turn>model\n")
            }
        }
    }

    /// Multi-turn transcript for chat REPL: must be non-empty, start with **User**, alternate roles,
    /// and end with **User** (ready for assistant generation).
    pub fn render_conversation(self, messages: &[ChatMessage]) -> Result<String, &'static str> {
        match self {
            ChatPromptStyle::Raw => Err("raw prompt style does not support chat transcripts"),
            ChatPromptStyle::MistralInstruct => render_mistral_instruct_multiturn(messages),
            ChatPromptStyle::Gemma4E2b => render_gemma4_e2b_multiturn(messages),
        }
    }
}

fn validate_chat_slice(messages: &[ChatMessage]) -> Result<(), &'static str> {
    if messages.is_empty() {
        return Err("conversation is empty");
    }
    if messages[0].role != ChatRole::User {
        return Err("conversation must start with a user message");
    }
    if messages.last().unwrap().role != ChatRole::User {
        return Err("conversation must end with a user message (assistant reply not generated yet)");
    }
    for w in messages.windows(2) {
        if w[0].role == w[1].role {
            return Err("user and assistant turns must alternate");
        }
    }
    Ok(())
}

/// Mistral Instruct v0.2: leading space + `[INST] ... [/INST]` per user; leading space + assistant text per turn.
fn render_mistral_instruct_multiturn(messages: &[ChatMessage]) -> Result<String, &'static str> {
    validate_chat_slice(messages)?;
    let mut s = String::new();
    for m in messages {
        match m.role {
            ChatRole::User => {
                s.push_str(" [INST] ");
                s.push_str(m.content.trim());
                s.push_str(" [/INST]");
            }
            ChatRole::Assistant => {
                s.push(' ');
                s.push_str(&m.content);
            }
        }
    }
    Ok(s)
}

/// Gemma 4 E2B: same turn markers as `model/gemma-4-e2b-it/chat_template.jinja` (no system/tools subset).
/// Each message uses `<|turn>{role}\n` … `<turn|>\n`; then `add_generation_prompt` appends `<|turn>model\n`.
fn render_gemma4_e2b_multiturn(messages: &[ChatMessage]) -> Result<String, &'static str> {
    validate_chat_slice(messages)?;
    let mut out = String::new();
    for m in messages {
        match m.role {
            ChatRole::User => {
                out.push_str("<|turn>user\n");
                out.push_str(m.content.trim());
                out.push_str("<turn|>\n");
            }
            ChatRole::Assistant => {
                out.push_str("<|turn>model\n");
                out.push_str(m.content.trim_end());
                out.push_str("<turn|>\n");
            }
        }
    }
    out.push_str("<|turn>model\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_wrap_single_turn() {
        let s = ChatPromptStyle::Gemma4E2b.wrap("Hello");
        assert!(s.starts_with("<|turn>user\n"));
        assert!(s.contains("Hello"));
        assert!(s.contains("<turn|>\n"));
        assert!(s.ends_with("<|turn>model\n"));
    }

    #[test]
    fn mistral_wrap() {
        let s = ChatPromptStyle::MistralInstruct.wrap("Hi");
        assert_eq!(s, "[INST] Hi [/INST]");
    }

    #[test]
    fn gemma4_two_turns() {
        let msgs = vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hello."),
            ChatMessage::user("Bye"),
        ];
        let s = ChatPromptStyle::Gemma4E2b
            .render_conversation(&msgs)
            .unwrap();
        assert!(s.contains("<|turn>user\nHi<turn|>\n"));
        assert!(s.contains("<|turn>model\nHello.<turn|>\n"));
        assert!(s.contains("<|turn>user\nBye<turn|>\n"));
        assert!(s.ends_with("<|turn>model\n"));
    }

    #[test]
    fn mistral_two_turns() {
        let msgs = vec![
            ChatMessage::user("Hi"),
            ChatMessage::assistant("Hey."),
            ChatMessage::user("Ok"),
        ];
        let s = ChatPromptStyle::MistralInstruct
            .render_conversation(&msgs)
            .unwrap();
        assert!(s.contains("[INST] Hi [/INST]"));
        assert!(s.contains(" Hey."));
        assert!(s.contains("[INST] Ok [/INST]"));
    }

    #[test]
    fn gemma4_truncate_stops_at_turn_close() {
        let s = "Hello.<turn|>junk";
        assert_eq!(gemma4_e2b_truncate_assistant_at_structure(s), "Hello.");
    }

    #[test]
    fn gemma4_truncate_stops_at_hallucinated_turn_header() {
        let s = "Ok.<|turn>user";
        assert_eq!(gemma4_e2b_truncate_assistant_at_structure(s), "Ok.");
    }

    #[test]
    fn gemma4_strip_dangling_contraction_after_turn_cut() {
        let s = "That's a very short message! I'";
        assert_eq!(gemma4_e2b_strip_dangling_contraction_tail(s), "That's a very short message!");
    }

    #[test]
    fn gemma4_assistant_visible_truncates_then_strips() {
        let s = "That's a very short message! I'<|turn>user";
        assert_eq!(gemma4_e2b_assistant_visible(s), "That's a very short message!");
    }

    #[test]
    fn gemma4_no_strip_when_no_marker_and_trailing_quote() {
        // Mid-generation "I'll" has no apostrophe at end; if we ever had lone I' without marker, do not strip.
        let s = "Still typing I'";
        assert_eq!(gemma4_e2b_assistant_visible(s), s);
    }
}
