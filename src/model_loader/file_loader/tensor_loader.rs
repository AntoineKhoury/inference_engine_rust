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
    reader.seek(tensor_info.offset)?;

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
/// Unpacks 4-bit values to u8 (0-15) and converts f16 scales/mins to f32
fn load_q4k_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    num_elements: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Q4_K: 256 weights per superblock, 160 bytes per superblock
    // 128 bytes quantized data + 16 bytes scales + 16 bytes mins
    let num_superblocks = (num_elements + 255) / 256; // Round up
    let num_blocks = (num_elements + 31) / 32; // Round up for scales/mins
    
    // Read all packed data
    let packed_data_size = num_superblocks * 128;
    let packed_data = reader.read_bytes(packed_data_size as u64)?;
    
    // Unpack 4-bit values to u8
    // Each byte contains 2 values: lower 4 bits and upper 4 bits
    let mut quantized_data = Vec::with_capacity(num_elements);
    for byte in &packed_data {
        quantized_data.push(byte & 0x0F);  // Lower 4 bits (0-15)
        quantized_data.push(byte >> 4);    // Upper 4 bits (0-15)
    }
    
    // Trim to exact num_elements (in case of rounding)
    quantized_data.truncate(num_elements);
    
    // Read scales (f16, one per block of 32 weights)
    let mut scales = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        scales.push(reader.read_f16()?);
    }
    
    // Read mins (f16, one per block of 32 weights)
    let mut mins = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        mins.push(reader.read_f16()?);
    }
    
    // Validate
    if quantized_data.len() != num_elements {
        return Err(format!(
            "Q4_K tensor {}: expected {} quantized elements, got {}",
            tensor_info.name, num_elements, quantized_data.len()
        ).into());
    }
    
    if scales.len() != num_blocks || mins.len() != num_blocks {
        return Err(format!(
            "Q4_K tensor {}: expected {} blocks for scales/mins, got {}/{}",
            tensor_info.name, num_blocks, scales.len(), mins.len()
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

/// Load Q6_K tensor (6-bit quantization)
/// Unpacks 6-bit values to u8 (0-63) and converts f16 scales/mins to f32
fn load_q6k_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    num_elements: usize,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Q6_K: 256 weights per superblock, 208 bytes per superblock
    // 192 bytes quantized data + 16 bytes scales + 16 bytes mins
    let num_superblocks = (num_elements + 255) / 256; // Round up
    let num_blocks = (num_elements + 31) / 32; // Round up for scales/mins
    
    // Read all packed data
    let packed_data_size = num_superblocks * 192;
    let packed_data = reader.read_bytes(packed_data_size as u64)?;
    
    // Unpack 6-bit values to u8
    // 4 values per 3 bytes: [value0:6][value1:2] [value1:4][value2:4] [value2:2][value3:6]
    let mut quantized_data = Vec::with_capacity(num_elements);
    let mut i = 0;
    while i + 2 < packed_data.len() && quantized_data.len() < num_elements {
        let byte0 = packed_data[i];
        let byte1 = packed_data[i + 1];
        let byte2 = packed_data[i + 2];
        
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
        
        i += 3;
    }
    
    // Read scales (f16, one per block of 32 weights)
    let mut scales = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        scales.push(reader.read_f16()?);
    }
    
    // Read mins (f16, one per block of 32 weights)
    let mut mins = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        mins.push(reader.read_f16()?);
    }
    
    // Validate
    if quantized_data.len() != num_elements {
        return Err(format!(
            "Q6_K tensor {}: expected {} quantized elements, got {}",
            tensor_info.name, num_elements, quantized_data.len()
        ).into());
    }
    
    if scales.len() != num_blocks || mins.len() != num_blocks {
        return Err(format!(
            "Q6_K tensor {}: expected {} blocks for scales/mins, got {}/{}",
            tensor_info.name, num_blocks, scales.len(), mins.len()
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
    use super::*;

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

