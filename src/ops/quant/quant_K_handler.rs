use crate::EngineError;
use crate::ops::quant::utils::f16_to_f32;

/// `scale * q` with the convention that `0 * ∞` is `0` (IEEE would yield NaN).
#[inline]
fn scale_times_quant_f64(scale: f64, q: f64) -> f64 {
    if q == 0.0 { 0.0 } else { scale * q }
}

/// `block_q4_K` in ggml: d[2] + dmin[2] + scales[12] + qs[128] — 144 bytes.
pub const Q4K_BLOCK_SIZE: usize = 144;
/// `block_q6_K` in ggml: ql[128] + qh[64] + scales[16] + d[2] — see ggml-common.h
pub const Q6K_BLOCK_SIZE: usize = 210;
const BLOCK_ELEMENTS: usize = 256;

/// `block_q8_0` in ggml: fp16 scale `d` + `int8[QK8_0]` with `QK8_0 = 32`.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;
pub const Q8_0_BLOCK_SIZE: usize = 2 + Q8_0_BLOCK_ELEMENTS;

/// Dequantize one Q8_0 block (32 weights). Layout matches ggml `block_q8_0`.
pub fn dequantize_q8_0_block(block: &[u8], out: &mut [f32]) -> Result<(), EngineError> {
    if block.len() < Q8_0_BLOCK_SIZE {
        return Err(EngineError::Tensor("Q8_0 block buffer too small".into()));
    }
    if out.len() < Q8_0_BLOCK_ELEMENTS {
        return Err(EngineError::Tensor(
            "Q8_0 block output buffer too small".into(),
        ));
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..Q8_0_BLOCK_ELEMENTS {
        out[i] = d * (block[2 + i] as i8 as f32);
    }
    Ok(())
}

/// One Q4_K superblock (256 weights). Port of ggml `dequantize_row_q4_K` for a single `block_q4_K`.
pub fn dequantize_q4k_block(block: &[u8], out: &mut [f32]) -> Result<(), EngineError> {
    if out.len() < BLOCK_ELEMENTS {
        return Err(EngineError::Tensor(
            "Q4K block output buffer too small".into(),
        ));
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let d64 = d as f64;
    let dmin64 = dmin as f64;
    let scales = &block[4..16];
    let q = &block[16..144];

    let mut y = 0usize;
    let mut q_ptr = 0usize;
    let mut is = 0i32;
    for _ in 0..4 {
        // j += 64 in ggml: four iterations cover 256 outputs
        let (sc, m) = extract_scale_min_k4(is as usize, scales);
        let (sc_b, m_b) = extract_scale_min_k4((is + 1) as usize, scales);
        let sc0 = sc as f64;
        let m0 = m as f64;
        let sc1 = sc_b as f64;
        let m1b = m_b as f64;

        for l in 0..32 {
            let v = (q[q_ptr + l] & 0xF) as f64;
            let dq = scale_times_quant_f64(d64 * sc0, v);
            let mq = dmin64 * m0;
            out[y + l] = (dq - mq) as f32;
        }
        for l in 0..32 {
            let v = ((q[q_ptr + l] >> 4) & 0x0F) as f64;
            let dq = scale_times_quant_f64(d64 * sc1, v);
            let mq = dmin64 * m1b;
            out[y + 32 + l] = (dq - mq) as f32;
        }
        y += 64;
        q_ptr += 32;
        is += 2;
    }

    Ok(())
}

/// Dequantize one Q6_K superblock (256 weights). Layout matches ggml `block_q6_K` / `dequantize_row_q6_K`.
pub fn dequantize_q6k_block(block: &[u8], out: &mut [f32]) -> Result<(), EngineError> {
    if block.len() < Q6K_BLOCK_SIZE {
        return Err(EngineError::Tensor("Q6K block buffer too small".into()));
    }
    if out.len() < BLOCK_ELEMENTS {
        return Err(EngineError::Tensor(
            "Q6K block output buffer too small".into(),
        ));
    }

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let d64 = d as f64;

    // Two halves of 128 outputs each (see ggml `dequantize_row_q6_K`).
    for half in 0..2 {
        let y_base = half * 128;
        let ql_off = half * 64;
        let qh_off = half * 32;
        let sc_off = half * 8;
        let sc_slice = &scales[sc_off..sc_off + 8];

        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l] & 0xF) as i32 | ((qh[qh_off + l] & 3) as i32) << 4) - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) as i32
                | (((qh[qh_off + l] >> 2) & 3) as i32) << 4)
                - 32;
            let q3 =
                ((ql[ql_off + l] >> 4) as i32 | (((qh[qh_off + l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) as i32
                | (((qh[qh_off + l] >> 6) & 3) as i32) << 4)
                - 32;

            // `block_q6_K.scales` is16× int8 in ggml (`dequantize_row_q6_K`); must not decode as u8.
            let s0 = (sc_slice[is] as i8) as f64;
            let s2 = (sc_slice[is + 2] as i8) as f64;
            let s4 = (sc_slice[is + 4] as i8) as f64;
            let s6 = (sc_slice[is + 6] as i8) as f64;

            out[y_base + l] = scale_times_quant_f64(d64 * s0, q1 as f64) as f32;
            out[y_base + l + 32] = scale_times_quant_f64(d64 * s2, q2 as f64) as f32;
            out[y_base + l + 64] = scale_times_quant_f64(d64 * s4, q3 as f64) as f32;
            out[y_base + l + 96] = scale_times_quant_f64(d64 * s6, q4 as f64) as f32;
        }
    }

    Ok(())
}

// Function to return the scale and min for a sub block, within a superblock
pub fn extract_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let scale = scales[j] & 0x3F;
        let min_val = scales[j + 4] & 0x3F;
        (scale, min_val)
    } else {
        let low_bits = scales[j + 4];
        let scale_low = low_bits & 0x0F;
        let min_low = (low_bits >> 4) & 0x0F;

        let scale_high = (scales[j - 4] >> 6) & 0x03;
        let min_high = (scales[j] >> 6) & 0x03;

        let scale = scale_low | (scale_high << 4);
        let min_val = min_low | (min_high << 4);
        (scale, min_val)
    }
}
