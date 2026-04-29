//! Parse `tools/llama_tensor_dump_ref` binary output (LMTD format v2).

use std::path::{Path, PathBuf};
use std::process::Command;

pub const LMTD_MAGIC: &[u8; 4] = b"LMTD";
pub const GGML_TYPE_F32: u32 = 0;

#[derive(Debug, Clone)]
pub struct LmtdRecordV2 {
    pub node_index: u32,
    pub ggml_type: u32,
    pub is_view: u32,
    pub ne: [i64; 4],
    pub nb: [u64; 4],
    pub data: Vec<u8>,
}

pub fn llama_tensor_dump_ref_binary() -> PathBuf {
    if let Ok(p) = std::env::var("LLAMA_TENSOR_DUMP_REF") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("llama_tensor_dump_ref")
}

/// Run `llama_tensor_dump_ref` with `LLAMA_TENSOR_DUMP_NODE` set; returns raw stdout bytes (one LMTD v2 record).
pub fn run_llama_tensor_dump_node(
    bin: &Path,
    model: &Path,
    token_ids: &[u32],
    node_index: u32,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(bin);
    cmd.arg(model);
    for &t in token_ids {
        cmd.arg(t.to_string());
    }
    cmd.env("LLAMA_TENSOR_DUMP_NODE", node_index.to_string());
    // Ensure trace is off so stdout is only the dump.
    cmd.env_remove("LLAMA_TENSOR_DUMP_TRACE");
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(format!(
            "llama_tensor_dump_ref failed: {}\nstderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(out.stdout)
}

fn read_u32_le(b: &[u8], off: &mut usize) -> Result<u32, String> {
    if *off + 4 > b.len() {
        return Err("truncated u32".into());
    }
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
    *off += 4;
    Ok(v)
}

fn read_i64_le(b: &[u8], off: &mut usize) -> Result<i64, String> {
    if *off + 8 > b.len() {
        return Err("truncated i64".into());
    }
    let v = i64::from_le_bytes(b[*off..*off + 8].try_into().unwrap());
    *off += 8;
    Ok(v)
}

fn read_u64_le(b: &[u8], off: &mut usize) -> Result<u64, String> {
    if *off + 8 > b.len() {
        return Err("truncated u64".into());
    }
    let v = u64::from_le_bytes(b[*off..*off + 8].try_into().unwrap());
    *off += 8;
    Ok(v)
}

/// Parse a single v2 record from the start of `bytes`.
pub fn parse_lmtd_v2(bytes: &[u8]) -> Result<LmtdRecordV2, Box<dyn std::error::Error>> {
    if bytes.len() < 4 + 4 + 4 + 4 + 4 + 32 + 32 + 8 {
        return Err("LMTD buffer too short".into());
    }
    if bytes[0..4] != LMTD_MAGIC[..] {
        return Err("missing LMTD magic".into());
    }
    let mut off = 4usize;
    let format_ver = read_u32_le(bytes, &mut off).map_err(|e| e.to_string())?;
    if format_ver != 2 {
        return Err(format!("unsupported LMTD format_ver {format_ver} (need 2)").into());
    }
    let node_index = read_u32_le(bytes, &mut off).map_err(|e| e.to_string())?;
    let ggml_type = read_u32_le(bytes, &mut off).map_err(|e| e.to_string())?;
    let is_view = read_u32_le(bytes, &mut off).map_err(|e| e.to_string())?;
    let mut ne = [0i64; 4];
    for ne_i in &mut ne {
        *ne_i = read_i64_le(bytes, &mut off).map_err(|e| e.to_string())?;
    }
    let mut nb = [0u64; 4];
    for nb_i in &mut nb {
        *nb_i = read_u64_le(bytes, &mut off).map_err(|e| e.to_string())?;
    }
    let nbytes = read_u64_le(bytes, &mut off).map_err(|e| e.to_string())?;
    let n_usize = usize::try_from(nbytes).map_err(|_| "nbytes too large for usize")?;
    if off + n_usize > bytes.len() {
        return Err(format!(
            "LMTD payload truncated: need {} bytes from offset {}",
            n_usize, off
        )
        .into());
    }
    let data = bytes[off..off + n_usize].to_vec();
    Ok(LmtdRecordV2 {
        node_index,
        ggml_type,
        is_view,
        ne,
        nb,
        data,
    })
}

/// Last-token row as `f32`, using GGML strides. `seq_len` / `hidden_dim` must match the two non-1 axes.
pub fn lmtd_last_token_f32(
    rec: &LmtdRecordV2,
    seq_len: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if rec.ggml_type != GGML_TYPE_F32 {
        return Err(format!(
            "ggml_type {} is not F32 (0); dequant not implemented",
            rec.ggml_type
        )
        .into());
    }
    let elts: usize = rec
        .ne
        .iter()
        .map(|&n| if n < 0 { 0usize } else { n as usize })
        .product();
    if elts * 4 != rec.data.len() {
        return Err(format!(
            "F32 byte len {} != ne product *4 ({elts} elts)",
            rec.data.len()
        )
        .into());
    }

    let seq_i64 = i64::try_from(seq_len).map_err(|_| "seq_len")?;
    let hid_i64 = i64::try_from(hidden_dim).map_err(|_| "hidden_dim")?;

    let active: Vec<usize> = (0..4).filter(|&i| rec.ne[i] > 1).collect();

    if active.len() == 1 {
        let i = active[0];
        // llama often keeps last-token-only tensors as [hidden, 1, 1, 1] even when `seq_len > 1`
        // (e.g. `result_norm` after prefill).
        if rec.ne[i] == hid_i64 {
            return f32_vec_from_bytes(&rec.data);
        }
        return Err(format!(
            "1D activation ne[{}]={} does not match hidden_dim={} (seq_len={})",
            i, rec.ne[i], hidden_dim, seq_len
        )
        .into());
    }

    if active.len() != 2 {
        return Err(format!(
            "expected 1 or 2 non-1 dims for prefill activations, got active indices {:?} ne {:?}",
            active, rec.ne
        )
        .into());
    }

    let i0 = active[0];
    let i1 = active[1];
    let (seq_axis, hid_axis) = if rec.ne[i0] == seq_i64 && rec.ne[i1] == hid_i64 {
        (i0, i1)
    } else if rec.ne[i0] == hid_i64 && rec.ne[i1] == seq_i64 {
        (i1, i0)
    } else {
        return Err(format!(
            "cannot map ne {:?} to seq_len={seq_len} hidden_dim={hidden_dim}",
            rec.ne
        )
        .into());
    };

    let last = seq_len - 1;
    let esz = 4u64;
    if rec.nb[hid_axis] != esz {
        return Err(format!("expected nb[hidden_axis]={esz}, got {}", rec.nb[hid_axis]).into());
    }

    let mut out = vec![0f32; hidden_dim];
    let base = (last as u64).saturating_mul(rec.nb[seq_axis]);
    for h in 0..hidden_dim {
        let byte_off = (base + (h as u64) * rec.nb[hid_axis]) as usize;
        if byte_off + 4 > rec.data.len() {
            return Err("byte offset past tensor data".into());
        }
        out[h] = f32::from_le_bytes(rec.data[byte_off..byte_off + 4].try_into().unwrap());
    }
    Ok(out)
}

fn f32_vec_from_bytes(data: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if data.len() % 4 != 0 {
        return Err("odd f32 buffer".into());
    }
    let n = data.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        v.push(f32::from_le_bytes(data[off..off + 4].try_into().unwrap()));
    }
    Ok(v)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y).abs();
        if d > m {
            m = d;
        }
    }
    m
}
