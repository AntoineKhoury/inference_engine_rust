pub mod block_iterator;

const Q4K_BLOCK_SIZE: usize = 144;
const Q6K_BLOCK_SIZE: usize = 208;
const BLOCK_ELEMENTS: usize = 256;

pub fn dequantize_q4k(buffer: &[u8], element_idx: usize) -> Result<f32, Box<dyn std::error::Error>> {
    let block_idx = element_idx / BLOCK_ELEMENTS;
    let in_block_idx = element_idx % BLOCK_ELEMENTS;
    let block_start = block_idx
        .checked_mul(Q4K_BLOCK_SIZE)
        .ok_or("Q4K block offset overflow")?;
    let block_end = block_start + Q4K_BLOCK_SIZE;
    let block = buffer.get(block_start..block_end).ok_or("Q4K block out of bounds")?;

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales_bytes = &block[4..16];
    let sub_block = in_block_idx / 32;
    let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
    let scale = d * scale_6bit as f32;
    let min = dmin * min_6bit as f32;

    let qs = &block[16..144];
    let quantized = get_quantized_value_q4k(in_block_idx, qs)? as f32;
    Ok((quantized * scale) + min)
}

pub fn dequantize_q6k(buffer: &[u8], element_idx: usize) -> Result<f32, Box<dyn std::error::Error>> {
    let block_idx = element_idx / BLOCK_ELEMENTS;
    let in_block_idx = element_idx % BLOCK_ELEMENTS;
    let block_start = block_idx
        .checked_mul(Q6K_BLOCK_SIZE)
        .ok_or("Q6K block offset overflow")?;
    let block_end = block_start + Q6K_BLOCK_SIZE;
    let block = buffer.get(block_start..block_end).ok_or("Q6K block out of bounds")?;

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales_bytes = &block[4..16];
    let sub_block = in_block_idx / 32;
    let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
    let scale = d * scale_6bit as f32;
    let min = dmin * min_6bit as f32;

    let qs = &block[16..208];
    let quantized = get_quantized_value_q6k(in_block_idx, qs)? as f32;
    Ok((quantized * scale) + min)
}

pub fn dequantize_q4k_range(
    buffer: &[u8],
    start_element: usize,
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if out.is_empty() {
        return Ok(());
    }

    let first_block = start_element / BLOCK_ELEMENTS;
    let last_element = start_element + out.len() - 1;
    let last_block = last_element / BLOCK_ELEMENTS;

    let mut block_out = vec![0.0f32; BLOCK_ELEMENTS];

    for block_idx in first_block..=last_block {
        let block_start = block_idx
            .checked_mul(Q4K_BLOCK_SIZE)
            .ok_or("Q4K block offset overflow")?;
        let block_end = block_start + Q4K_BLOCK_SIZE;
        let block = buffer.get(block_start..block_end).ok_or("Q4K block out of bounds")?;

        dequantize_q4k_block(block, &mut block_out)?;

        let block_base_element = block_idx * BLOCK_ELEMENTS;
        let copy_start = start_element.max(block_base_element);
        let copy_end = (start_element + out.len()).min(block_base_element + BLOCK_ELEMENTS);

        let src_start = copy_start - block_base_element;
        let dst_start = copy_start - start_element;
        let len = copy_end - copy_start;
        out[dst_start..dst_start + len].copy_from_slice(&block_out[src_start..src_start + len]);
    }

    Ok(())
}

pub fn dequantize_q6k_range(
    buffer: &[u8],
    start_element: usize,
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if out.is_empty() {
        return Ok(());
    }

    let first_block = start_element / BLOCK_ELEMENTS;
    let last_element = start_element + out.len() - 1;
    let last_block = last_element / BLOCK_ELEMENTS;

    let mut block_out = vec![0.0f32; BLOCK_ELEMENTS];

    for block_idx in first_block..=last_block {
        let block_start = block_idx
            .checked_mul(Q6K_BLOCK_SIZE)
            .ok_or("Q6K block offset overflow")?;
        let block_end = block_start + Q6K_BLOCK_SIZE;
        let block = buffer.get(block_start..block_end).ok_or("Q6K block out of bounds")?;

        dequantize_q6k_block(block, &mut block_out)?;

        let block_base_element = block_idx * BLOCK_ELEMENTS;
        let copy_start = start_element.max(block_base_element);
        let copy_end = (start_element + out.len()).min(block_base_element + BLOCK_ELEMENTS);

        let src_start = copy_start - block_base_element;
        let dst_start = copy_start - start_element;
        let len = copy_end - copy_start;
        out[dst_start..dst_start + len].copy_from_slice(&block_out[src_start..src_start + len]);
    }

    Ok(())
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    if exponent == 0 {
        if mantissa == 0 {
            if sign == 0 { 0.0 } else { -0.0 }
        } else {
            // Subnormal
            let value = (mantissa as f32) / 1024.0 * 2.0_f32.powi(-14);
            if sign == 0 { value } else { -value }
        }
    } else if exponent == 0x1F {
        if mantissa == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        }
    } else {
        // Normalized
        let exp = (exponent as i32) - 15;
        let mant = 1.0 + (mantissa as f32) / 1024.0;
        let value = mant * 2.0_f32.powi(exp);
        if sign == 0 { value } else { -value }
    }
}

fn dequantize_q4k_block(
    block: &[u8],
    out: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if out.len() < BLOCK_ELEMENTS {
        return Err("Q4K block output buffer too small".into());
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    let scales_bytes = &block[4..16];
    let qs = &block[16..144];

    for element_idx in 0..BLOCK_ELEMENTS {
        let sub_block = element_idx / 32;
        let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
        let scale = d * scale_6bit as f32;
        let min = dmin * min_6bit as f32;
        let quantized = get_quantized_value_q4k(element_idx, qs)? as f32;
        out[element_idx] = (quantized * scale) + min;
    }

    Ok(())
}

fn dequantize_q6k_block(
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

    for element_idx in 0..BLOCK_ELEMENTS {
        let sub_block = element_idx / 32;
        let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block, scales_bytes);
        let scale = d * scale_6bit as f32;
        let min = dmin * min_6bit as f32;
        let quantized = get_quantized_value_q6k(element_idx, qs)? as f32;
        out[element_idx] = (quantized * scale) + min;
    }

    Ok(())
}

fn extract_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
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

fn get_quantized_value_q4k(
    pos: usize,
    qs: &[u8],
) -> Result<u8, Box<dyn std::error::Error>> {
    let group = pos / 64;
    let offset_in_group = pos % 64;
    let byte_idx = group * 32 + (offset_in_group % 32);
    let nibble = offset_in_group / 32;

    let byte = *qs.get(byte_idx).ok_or("Q4K quantized index out of bounds")?;
    let value = if nibble == 0 {
        byte & 0x0F
    } else {
        (byte >> 4) & 0x0F
    };
    Ok(value)
}

fn get_quantized_value_q6k(
    pos: usize,
    qs: &[u8],
) -> Result<u8, Box<dyn std::error::Error>> {
    let group = pos / 4;
    let offset = pos % 4;
    let byte_idx = group * 3;

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

    let value = match offset {
        0 => value0,
        1 => value1,
        2 => value2,
        _ => value3,
    };
    Ok(value)
}
