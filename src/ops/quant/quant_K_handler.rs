use crate::ops::quant::utils::f16_to_f32;

const Q4K_BLOCK_SIZE: usize = 144;
const Q6K_BLOCK_SIZE: usize = 208;
const BLOCK_ELEMENTS: usize = 256;
const SUB_BLOCK_SIZE: usize = 32;
const NBR_SUB_BLOCKS: usize = BLOCK_ELEMENTS / SUB_BLOCK_SIZE;

// Function to modify a buffer of quantized q4K weights, into a f32 output buffer
pub fn dequantize_q4k_block(
    block: &[u8],
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if out.len() < BLOCK_ELEMENTS {
        return Err("Q4K block output buffer too small".into());
    }

    // Global scale factor for all blocks 
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    // Global scale factor for the block's offsets, scales the min value for each block
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales_bytes = &block[4..16];
    let qs = &block[16..144];

    for sub_block in 0..NBR_SUB_BLOCKS{ 
        let base = sub_block * SUB_BLOCK_SIZE;
        let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
        let scale = d * scale_6bit as f32;
        let min = dmin * min_6bit as f32;
        let group = base / 64;
        let offset = base % 64;
        let byte_start = group * 32;
        let nibble = offset / 32;

        for i in 0..SUB_BLOCK_SIZE {
            let byte = *qs
                .get(byte_start + i)
                .ok_or("Q4K quantized index out of bounds")?;
            let value = if nibble == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            out[base + i] = (value as f32 * scale) + min;
        }
    }

    Ok(())
}

pub fn dequantize_q6k_block(
    block: &[u8],
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if out.len() < BLOCK_ELEMENTS {
        return Err("Q6K block output buffer too small".into());
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales_bytes = &block[4..16];
    let qs = &block[16..208];

    for sub_block in 0..NBR_SUB_BLOCKS {
        let base = sub_block * SUB_BLOCK_SIZE;
        let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
        let scale = d * scale_6bit as f32;
        let min = dmin * min_6bit as f32;
        let group_base = (base / 4) * 3;

        for group in 0..(SUB_BLOCK_SIZE / 4) {
            let byte_idx = group_base + group * 3;
            let bytes = qs
                .get(byte_idx..byte_idx + 3)
                .ok_or("Q6K quantized index out of bounds")?;
            let byte0 = bytes[0];
            let byte1 = bytes[1];
            let byte2 = bytes[2];

            let value0 = byte0 & 0x3F;
            let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);
            let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);
            let value3 = byte2 >> 2;

            let out_base = base + group * 4;
            out[out_base] = (value0 as f32 * scale) + min;
            out[out_base + 1] = (value1 as f32 * scale) + min;
            out[out_base + 2] = (value2 as f32 * scale) + min;
            out[out_base + 3] = (value3 as f32 * scale) + min;
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
