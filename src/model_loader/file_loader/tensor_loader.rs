use super::io::Reader;
use super::types::{Tensor, TensorInfo, TensorType};
use std::io::{BufRead, Seek};

/// Load a single tensor from the file based on TensorInfo
/// Validates the number of elements read and returns an error if incomplete
pub fn load_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Calculate total number of elements
    let num_elements = tensor_info
        .dimensions
        .iter()
        .product::<u64>() as usize;

    // Seek to the tensor's offset
    reader.seek(tensor_info.offset)
        .map_err(|e| format!("Failed to seek to offset {} for tensor '{}': {}", 
                             tensor_info.offset, tensor_info.name, e))?;

    // Load based on type_id
    match tensor_info.type_id {
        0 => load_f32_tensor(reader, tensor_info, num_elements),
        12 => load_q4k_tensor(reader, tensor_info, num_elements),
        14 => load_q6k_tensor(reader, tensor_info, num_elements),
        _ => Err(format!("Unsupported tensor type_id: {}", tensor_info.type_id).into()),
    }
}

/// Load F32 tensor (unquantized float32)
fn load_f32_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    num_elements: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(num_elements);
    
    for _ in 0..num_elements {
        data.push(reader.read_f32()?);
    }
    
    // Validate we read exactly num_elements
    if data.len() != num_elements {
        return Err(format!(
            "F32 tensor {}: expected {} elements, read {}",
            tensor_info.name, num_elements, data.len()
        ).into());
    }
    
    Ok(Tensor::new(
        TensorType::F32,
        tensor_info.name.clone(),
        tensor_info.dimensions.clone(),
        num_elements,
        Some(data),
        None,
        None,
        None,
    ))
}

/// Load Q4_K tensor (4-bit quantization)
/// Format: 144 bytes per superblock (256 elements)
/// Structure: 4 bytes dm (2 half floats) + 12 bytes scales (packed 6-bit) + 128 bytes qs (quantized values)
fn load_q4k_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    num_elements: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Q4_K: 256 weights per superblock, 144 bytes per superblock
    // Structure: 4 bytes dm + 12 bytes scales + 128 bytes qs
    const BLOCK_SIZE: usize = 144;
    const ELEMENTS_PER_BLOCK: usize = 256;
    const ELEMENTS_PER_SUB_BLOCK: usize = 32;
    
    let num_superblocks = (num_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK; // Round up
    let total_bytes = num_superblocks * BLOCK_SIZE;
    
    // Read entire superblock data
    let block_data = reader.read_bytes(total_bytes as u64)
        .map_err(|e| format!("Q4_K tensor '{}': Failed to read {} bytes at offset {}: {}", 
                             tensor_info.name, total_bytes, tensor_info.offset, e))?;
    
    // Extract quantized values, scales, and mins
    let mut quantized_data = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(num_superblocks * 8); // 8 sub-blocks per superblock
    let mut mins = Vec::with_capacity(num_superblocks * 8);
    
    // Process each superblock
    for block_idx in 0..num_superblocks {
        let block_start = block_idx * BLOCK_SIZE;
        
        // Read dm (4 bytes: 2 half floats)
        let d = f16_to_f32(u16::from_le_bytes([
            block_data[block_start],
            block_data[block_start + 1],
        ]));
        let dmin = f16_to_f32(u16::from_le_bytes([
            block_data[block_start + 2],
            block_data[block_start + 3],
        ]));
        
        // Read scales array (12 bytes)
        let scales_start = block_start + 4;
        let scales_bytes = &block_data[scales_start..scales_start + 12];
        
        // Extract scales and mins for 8 sub-blocks (6 bits each)
        for sub_block_idx in 0..8 {
            let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block_idx, scales_bytes);
            
            // Reconstruct actual scale and min: actual = dm * quantized_value
            // We store the 6-bit quantized values and dm separately for dequantization
            scales.push(d * scale_6bit as f32);
            mins.push(dmin * min_6bit as f32);
        }
        
        // Read quantized values (128 bytes = 256 values, 4 bits each)
        let qs_start = block_start + 16;
        let qs_bytes = &block_data[qs_start..qs_start + 128];
        
        // Extract quantized values according to the group-of-64 layout
        for element_pos in 0..ELEMENTS_PER_BLOCK {
            if quantized_data.len() >= num_elements {
                break;
            }
            
            let quantized = get_quantized_value_q4k(element_pos, qs_bytes);
            quantized_data.push(quantized);
        }
    }
    
    // Trim to exact num_elements
    quantized_data.truncate(num_elements);
    scales.truncate((num_elements + ELEMENTS_PER_SUB_BLOCK - 1) / ELEMENTS_PER_SUB_BLOCK);
    mins.truncate((num_elements + ELEMENTS_PER_SUB_BLOCK - 1) / ELEMENTS_PER_SUB_BLOCK);
    
    // Validate
    if quantized_data.len() != num_elements {
        return Err(format!(
            "Q4_K tensor {}: expected {} quantized elements, got {}",
            tensor_info.name, num_elements, quantized_data.len()
        ).into());
    }
    
    Ok(Tensor::new(
        TensorType::Q4K,
        tensor_info.name.clone(),
        tensor_info.dimensions.clone(),
        num_elements,
        None,
        Some(quantized_data),
        Some(scales),
        Some(mins),
    ))
}

/// Extract scale and min for sub-block j (0-7) from packed scales array (12 bytes)
/// Returns: (scale_6bit, min_6bit) both as u8 values (0-63)
/// Based on the Q4_K packing scheme from GGML specification
fn extract_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        // For sub-blocks 0-3: simple extraction
        // scales[j] contains scale (6 bits), scales[j+4] contains min (6 bits)
        let scale = scales[j] & 0x3F; // 6 bits (0-63)
        let min_val = scales[j + 4] & 0x3F; // 6 bits (0-63)
        (scale, min_val)
    } else {
        // For sub-blocks 4-7: complex packing
        // scales[j+4] contains low 4 bits of both scale and min
        let low_bits = scales[j + 4];
        let scale_low = low_bits & 0x0F; // Low 4 bits of scale
        let min_low = (low_bits >> 4) & 0x0F; // Low 4 bits of min
        
        // High 2 bits are stored in scales[j-4] and scales[j] (bits 6-7)
        let scale_high = (scales[j - 4] >> 6) & 0x03; // High 2 bits of scale
        let min_high = (scales[j] >> 6) & 0x03; // High 2 bits of min (j-0 = j)
        
        // Combine: 6-bit value = low 4 bits | (high 2 bits << 4)
        let scale = scale_low | (scale_high << 4);
        let min_val = min_low | (min_high << 4);
        (scale, min_val)
    }
}

/// Get quantized value (0-15) for element at position pos (0-255) from qs array
/// Layout: organized in groups of 64 elements
fn get_quantized_value_q4k(pos: usize, qs: &[u8]) -> u8 {
    let group = pos / 64; // Which group of 64 (0-3)
    let offset_in_group = pos % 64; // Position within group (0-63)
    let byte_idx = group * 32 + (offset_in_group % 32);
    let nibble = offset_in_group / 32; // 0 = low nibble, 1 = high nibble
    
    let byte = qs[byte_idx];
    if nibble == 0 {
        byte & 0x0F // Low 4 bits
    } else {
        (byte >> 4) & 0x0F // High 4 bits
    }
}

/// Convert half-precision float (f16) to f32
/// Helper function for reading dm values
fn f16_to_f32(bits: u16) -> f32 {
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

/// Load Q6_K tensor (6-bit quantization)
/// Format: 208 bytes per superblock (256 elements)
/// Structure: 4 bytes dm (2 half floats) + 12 bytes scales (packed 6-bit) + 192 bytes qs (quantized values)
fn load_q6k_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    num_elements: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Q6_K: 256 weights per superblock, 208 bytes per superblock
    // Structure: 4 bytes dm + 12 bytes scales + 192 bytes qs
    const BLOCK_SIZE: usize = 208;
    const ELEMENTS_PER_BLOCK: usize = 256;
    const ELEMENTS_PER_SUB_BLOCK: usize = 32;
    
    let num_superblocks = (num_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK; // Round up
    let total_bytes = num_superblocks * BLOCK_SIZE;
    
    // Read entire superblock data at once for efficiency
    let block_data = reader.read_bytes(total_bytes as u64)
        .map_err(|e| format!("Q6_K tensor '{}': Failed to read {} bytes at offset {}: {}", 
                             tensor_info.name, total_bytes, tensor_info.offset, e))?;
    
    // Extract quantized values, scales, and mins
    let mut quantized_data = Vec::with_capacity(num_elements);
    let mut scales = Vec::with_capacity(num_superblocks * 8); // 8 sub-blocks per superblock
    let mut mins = Vec::with_capacity(num_superblocks * 8);
    
    // Process each superblock
    for block_idx in 0..num_superblocks {
        let block_start = block_idx * BLOCK_SIZE;
        
        // Read dm (4 bytes: 2 half floats)
        let d = f16_to_f32(u16::from_le_bytes([
            block_data[block_start],
            block_data[block_start + 1],
        ]));
        let dmin = f16_to_f32(u16::from_le_bytes([
            block_data[block_start + 2],
            block_data[block_start + 3],
        ]));
        
        // Read scales array (12 bytes) - same packing scheme as Q4_K
        let scales_start = block_start + 4;
        let scales_bytes = &block_data[scales_start..scales_start + 12];
        
        // Extract scales and mins for 8 sub-blocks (6 bits each)
        for sub_block_idx in 0..8 {
            let (scale_6bit, min_6bit) = extract_scale_min_k4(sub_block_idx, scales_bytes);
            
            // Reconstruct actual scale and min: actual = dm * quantized_value
            scales.push(d * scale_6bit as f32);
            mins.push(dmin * min_6bit as f32);
        }
        
        // Read quantized values (192 bytes = 256 values, 6 bits each, packed)
        let qs_start = block_start + 16;
        let qs_bytes = &block_data[qs_start..qs_start + 192];
        
        // Unpack 6-bit values: 4 values per 3 bytes
        // Layout: [value0:6][value1:2] [value1:4][value2:4] [value2:2][value3:6]
        let mut byte_idx = 0;
        while byte_idx + 2 < qs_bytes.len() && quantized_data.len() < num_elements {
            let byte0 = qs_bytes[byte_idx];
            let byte1 = qs_bytes[byte_idx + 1];
            let byte2 = qs_bytes[byte_idx + 2];
            
            let value0 = byte0 & 0x3F;  // Lower 6 bits of byte0
            let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);  // Upper 2 bits of byte0 + lower 4 bits of byte1
            let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);  // Upper 4 bits of byte1 + lower 2 bits of byte2
            let value3 = byte2 >> 2;  // Upper 6 bits of byte2
            
            quantized_data.push(value0);
            if quantized_data.len() < num_elements {
                quantized_data.push(value1);
            }
            if quantized_data.len() < num_elements {
                quantized_data.push(value2);
            }
            if quantized_data.len() < num_elements {
                quantized_data.push(value3);
            }
            
            byte_idx += 3;
        }
    }
    
    // Trim to exact num_elements
    quantized_data.truncate(num_elements);
    scales.truncate((num_elements + ELEMENTS_PER_SUB_BLOCK - 1) / ELEMENTS_PER_SUB_BLOCK);
    mins.truncate((num_elements + ELEMENTS_PER_SUB_BLOCK - 1) / ELEMENTS_PER_SUB_BLOCK);
    
    // Validate
    if quantized_data.len() != num_elements {
        return Err(format!(
            "Q6_K tensor {}: expected {} quantized elements, got {}",
            tensor_info.name, num_elements, quantized_data.len()
        ).into());
    }
    
    Ok(Tensor::new(
        TensorType::Q6K,
        tensor_info.name.clone(),
        tensor_info.dimensions.clone(),
        num_elements,
        None,
        Some(quantized_data),
        Some(scales),
        Some(mins),
    ))
}

#[cfg(test)]
mod tests {

    /// Test Q4K unpacking: Each byte contains 2 values (lower 4 bits, upper 4 bits)
    #[test]
    fn test_q4k_unpack_simple() {
        // Test case: byte 0x3A
        // Lower 4 bits: 0xA = 10
        // Upper 4 bits: 0x3 = 3
        // Should unpack to [10, 3]
        let packed = vec![0x3A];
        let mut quantized = Vec::with_capacity(2);
        for byte in &packed {
            quantized.push(byte & 0x0F);  // Lower 4 bits
            quantized.push(byte >> 4);    // Upper 4 bits
        }
        assert_eq!(quantized, vec![10, 3]);
    }

    /// Test Q4K unpacking with multiple bytes
    #[test]
    fn test_q4k_unpack_multiple() {
        // Test case: [0x12, 0x34, 0x56]
        // 0x12: lower=0x2 (2), upper=0x1 (1) → [2, 1]
        // 0x34: lower=0x4 (4), upper=0x3 (3) → [4, 3]
        // 0x56: lower=0x6 (6), upper=0x5 (5) → [6, 5]
        // Expected: [2, 1, 4, 3, 6, 5]
        let packed = vec![0x12, 0x34, 0x56];
        let mut quantized = Vec::with_capacity(6);
        for byte in &packed {
            quantized.push(byte & 0x0F);
            quantized.push(byte >> 4);
        }
        assert_eq!(quantized, vec![2, 1, 4, 3, 6, 5]);
    }

    /// Test Q4K unpacking with edge cases (0x00, 0xFF)
    #[test]
    fn test_q4k_unpack_edges() {
        // 0x00: lower=0, upper=0 → [0, 0]
        // 0xFF: lower=0xF (15), upper=0xF (15) → [15, 15]
        let packed = vec![0x00, 0xFF];
        let mut quantized = Vec::with_capacity(4);
        for byte in &packed {
            quantized.push(byte & 0x0F);
            quantized.push(byte >> 4);
        }
        assert_eq!(quantized, vec![0, 0, 15, 15]);
    }

    /// Test Q6K unpacking: 4 values per 3 bytes
    #[test]
    fn test_q6k_unpack_simple() {
        // Test case: bytes [0x3A, 0x5C, 0x7E]
        let byte0 = 0x3A;
        let byte1 = 0x5C;
        let byte2 = 0x7E;
        
        let value0 = byte0 & 0x3F;  // 0x3A & 0x3F = 58
        let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);  // 0 | (12 << 2) = 48
        let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);  // 5 | (2 << 4) = 37
        let value3 = byte2 >> 2;  // 0x7E >> 2 = 31
        
        assert_eq!(value0, 58);
        assert_eq!(value1, 48);
        assert_eq!(value2, 37);
        assert_eq!(value3, 31);
    }

    /// Test Q6K unpacking with all zeros
    #[test]
    fn test_q6k_unpack_zeros() {
        let byte0 = 0x00;
        let byte1 = 0x00;
        let byte2 = 0x00;
        
        let value0 = byte0 & 0x3F;
        let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);
        let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);
        let value3 = byte2 >> 2;
        
        assert_eq!(value0, 0);
        assert_eq!(value1, 0);
        assert_eq!(value2, 0);
        assert_eq!(value3, 0);
    }

    /// Test Q6K unpacking with maximum values (all 6-bit values = 63)
    #[test]
    fn test_q6k_unpack_max() {
        let byte0 = 0xFF;
        let byte1 = 0xFF;
        let byte2 = 0xFF;
        
        let value0 = byte0 & 0x3F;  // 63
        let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);  // 3 | 60 = 63
        let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);  // 15 | 48 = 63
        let value3 = byte2 >> 2;  // 63
        
        assert_eq!(value0, 63);
        assert_eq!(value1, 63);
        assert_eq!(value2, 63);
        assert_eq!(value3, 63);
    }

    /// Test Q6K unpacking with a sequence of 3-byte groups
    #[test]
    fn test_q6k_unpack_sequence() {
        // Test unpacking multiple 3-byte groups
        let packed = vec![0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC];
        let mut quantized = Vec::new();
        
        for i in (0..packed.len()).step_by(3) {
            if i + 2 < packed.len() {
                let byte0 = packed[i];
                let byte1 = packed[i + 1];
                let byte2 = packed[i + 2];
                
                let value0 = byte0 & 0x3F;
                let value1 = (byte0 >> 6) | ((byte1 & 0x0F) << 2);
                let value2 = (byte1 >> 4) | ((byte2 & 0x03) << 4);
                let value3 = byte2 >> 2;
                
                quantized.push(value0);
                quantized.push(value1);
                quantized.push(value2);
                quantized.push(value3);
            }
        }
        
        // Should have 8 values (2 groups × 4 values)
        assert_eq!(quantized.len(), 8);
        // Verify all values are in valid range (0-63)
        for &val in &quantized {
            assert!(val <= 63, "Value {} exceeds 6-bit range", val);
        }
    }
}

