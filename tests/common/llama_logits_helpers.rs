//! Shared helpers for comparing Rust prefill logits to `tools/llama_logits_ref`.

use std::path::{Path, PathBuf};
use std::process::Command;

pub fn llama_logits_ref_binary() -> PathBuf {
    if let Ok(p) = std::env::var("LLAMA_LOGITS_REF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("llama_logits_ref")
}

/// Read `u32` LE `n_vocab` then `n_vocab` little-endian `f32` logits from llama.cpp helper stdout.
pub fn read_reference_logits(
    bin: &Path,
    model: &Path,
    token_ids: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(bin);
    cmd.arg(model);
    for &t in token_ids {
        cmd.arg(t.to_string());
    }
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "llama_logits_ref failed: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    let bytes = out.stdout;
    if bytes.len() < 4 {
        return Err("reference output too short".into());
    }
    let n_vocab = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected = 4 + n_vocab * 4;
    if bytes.len() != expected {
        return Err(format!(
            "bad ref size: got {} bytes, expected {} (n_vocab={})",
            bytes.len(),
            expected,
            n_vocab
        )
        .into());
    }
    let mut logits = Vec::with_capacity(n_vocab);
    for i in 0..n_vocab {
        let off = 4 + i * 4;
        logits.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    Ok(logits)
}

pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    let mut best_i = 0usize;
    let mut best_v = s.first().copied()?;
    for (i, &v) in s.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

pub fn logits_diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    let mut max_abs = 0f32;
    let mut sum_sq = 0f32;
    let mut argmax_mismatch = 0usize;
    let ia = argmax_f32(a);
    let ib = argmax_f32(b);
    if ia != ib {
        argmax_mismatch = 1;
    }
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let t = x - y;
        sum_sq += t * t;
    }
    let rmse = (sum_sq / a.len() as f32).sqrt();
    (max_abs, rmse, argmax_mismatch)
}
