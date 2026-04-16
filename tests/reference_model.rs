//! Reference checks against a known GGUF file (optional download).
//!
//! **Hardcoded floats** — see `tests/common/mod.rs` (source: gguf-py `dequantize` on `token_embd.weight`).
//!
//! Run the numeric check when the file is present:
//! ```text
//! cargo test embedding_token2_matches_gguf_reference -- --ignored --nocapture
//! ```
//!
//! Optional: download ~4GB with `curl` (requires `curl` on `PATH`):
//! ```text
//! DOWNLOAD_REFERENCE_GGUF=1 cargo test download_reference_gguf -- --ignored
//! ```

mod common;

use inference_engine_rust::layers::embeddings::lookup_embeddings;
use inference_engine_rust::model_loader::file_loader::read_file;
use inference_engine_rust::model_config::TokenizerPromptConfig;

use common::{
    assert_close, reference_model_path, REF_EMB_TOKEN2_HEAD, REF_EMB_TOKEN2_IDX1024,
    REF_EMB_TOKEN2_IDX4095, REFERENCE_MODEL_DISPLAY_NAME, REFERENCE_MODEL_DOWNLOAD_URL,
    REFERENCE_MODEL_REL_PATH, REFERENCE_TOKEN_ID,
};

const EPS: f32 = 2e-5;

/// Inspect GGUF tokenizer flags (run with `--ignored --nocapture` when the reference GGUF is present).
#[test]
#[ignore = "requires model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf"]
fn mistral_tokenizer_prompt_config_from_gguf() {
    let path = reference_model_path();
    if !path.is_file() {
        eprintln!("skip: missing {}", path.display());
        return;
    }
    let gguf = read_file(REFERENCE_MODEL_REL_PATH).expect("read gguf");
    let cfg = TokenizerPromptConfig::from_gguf(&gguf).expect("tokenizer prompt config");
    eprintln!("TokenizerPromptConfig: {cfg:?}");
    // TheBloke Mistral GGUF: no add_bos_token key — llama.cpp uses SPM defaults (BOS on, EOS off).
    assert!(
        cfg.add_bos_token && !cfg.add_eos_token,
        "expected llama-SPM defaults for reference Mistral GGUF, got {cfg:?}"
    );
    assert_eq!(cfg.bos_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
}

/// Compare Rust `lookup_embeddings` to hardcoded gguf-py reference for [`REFERENCE_TOKEN_ID`].
#[test]
#[ignore = "requires model/mistral-7b-v0.1/mistral-7b-v0.1.Q4_K_M.gguf (see tests/common/mod.rs)"]
fn embedding_token2_matches_gguf_reference() {
    let path = reference_model_path();
    assert!(
        path.is_file(),
        "missing reference GGUF at {} — place file or run `download_reference_gguf` test",
        path.display()
    );

    let path_str = REFERENCE_MODEL_REL_PATH;
    let mut gguf = read_file(path_str).expect("read gguf metadata");
    let rows = lookup_embeddings(&mut gguf, path_str, &[REFERENCE_TOKEN_ID]).expect("lookup");
    let row = &rows[0];
    assert_eq!(row.len(), 4096, "{REFERENCE_MODEL_DISPLAY_NAME} hidden dim");

    for (&exp, &got) in REF_EMB_TOKEN2_HEAD.iter().zip(row[..8].iter()) {
        assert_close(exp, got, EPS);
    }
    assert_close(REF_EMB_TOKEN2_IDX1024, row[1024], EPS);
    assert_close(REF_EMB_TOKEN2_IDX4095, row[4095], EPS);

    assert!(
        !row.iter().any(|x| x.is_nan()),
        "embedding must be finite (NaNs usually mean wrong GGUF seek/layout)"
    );
}

/// Download the reference GGUF via `curl` if missing. **Large (~4GB).**
#[test]
#[ignore = "network: downloads ~4GB; set DOWNLOAD_REFERENCE_GGUF=1"]
fn download_reference_gguf() {
    if std::env::var("DOWNLOAD_REFERENCE_GGUF").ok().as_deref() != Some("1") {
        eprintln!(
            "skip: set DOWNLOAD_REFERENCE_GGUF=1 to download from {REFERENCE_MODEL_DOWNLOAD_URL}"
        );
        return;
    }

    let path = reference_model_path();
    if path.is_file() {
        eprintln!("already present: {}", path.display());
        return;
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create model dir");
    }

    let status = std::process::Command::new("curl")
        .args([
            "-fL",
            "--retry",
            "3",
            "-o",
            path.to_str().expect("utf8 path"),
            REFERENCE_MODEL_DOWNLOAD_URL,
        ])
        .status()
        .expect("spawn curl (install curl or download manually)");

    assert!(status.success(), "curl failed: {status}");

    let meta = std::fs::metadata(&path).expect("stat");
    assert!(
        meta.len() > 1_000_000_000,
        "downloaded file suspiciously small ({} bytes); wrong URL?",
        meta.len()
    );
    eprintln!("downloaded {} ({:.2} GB)", path.display(), meta.len() as f64 / 1e9);
}
