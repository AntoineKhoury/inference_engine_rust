use std::io::{BufRead, Seek};
use std::sync::Arc;

use crate::core::tensor::Tensor;
use crate::EngineError;
use crate::model_loader::gguf_types::TensorInfo;
use crate::model_loader::reader::Reader;
use crate::model_loader::tensor::GgmlType;
use crate::ops::quant::quant_k_handler::{
    Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_SIZE,
};
const BLOCK_ELEMENTS: usize = 256;

/// Load a single tensor from the file based on TensorInfo.
/// This reads raw bytes into the tensor buffer without decoding.
///
/// `tensor_data_base` is the absolute file offset where the GGUF **tensor data blob** begins
/// (immediately after the tensor metadata array). Per GGUF, each `TensorInfo.offset` is relative
/// to that base, not to the start of the file.
/// GGUF BF16 → `f32` (upper 16 bits of IEEE-754 `f32`; little-endian u16 in file).
#[inline]
fn bf16_le_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    f32::from_bits((bits as u32) << 16)
}

pub fn load_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    tensor_data_base: u64,
) -> Result<Tensor, EngineError> {
    let ggml_type = GgmlType::try_from(tensor_info.type_id)?;
    let num_elements = tensor_info
        .dimensions
        .iter()
        .product::<usize>();

    let abs_offset = tensor_data_base
        .checked_add(tensor_info.offset as u64)
        .ok_or_else(|| EngineError::Gguf("tensor offset overflow".into()))?;

    reader.seek(abs_offset)?;

    if ggml_type == GgmlType::BF16 {
        let byte_len = num_elements
            .checked_mul(2)
            .ok_or_else(|| EngineError::Gguf("BF16 tensor byte length overflow".into()))?;
        let raw = reader.read_bytes(byte_len as u64)?;
        let mut f32_bytes = Vec::with_capacity(num_elements * 4);
        for chunk in raw.chunks_exact(2) {
            let f = bf16_le_to_f32([chunk[0], chunk[1]]);
            f32_bytes.extend_from_slice(&f.to_le_bytes());
        }
        return Ok(Tensor::new(
            crate::core::tensor::TensorType::F32,
            Arc::new(f32_bytes),
            tensor_info.dimensions.clone(),
        ));
    }

    let tensor_type = ggml_type.to_tensor_type()?;
    let byte_len = expected_byte_len(tensor_type, num_elements)?;
    let buffer = reader.read_bytes(byte_len as u64)?;

    Ok(Tensor::new(
        tensor_type,
        Arc::new(buffer),
        tensor_info.dimensions.clone(),
    ))
}

fn expected_byte_len(
    tensor_type: crate::core::tensor::TensorType,
    num_elements: usize,
) -> Result<usize, EngineError> {
    match tensor_type {
        crate::core::tensor::TensorType::F32 => Ok(num_elements * 4),
        crate::core::tensor::TensorType::Q4K => {
            let num_blocks = num_elements.div_ceil(BLOCK_ELEMENTS);
            Ok(num_blocks * Q4K_BLOCK_SIZE)
        }
        crate::core::tensor::TensorType::Q6K => {
            let num_blocks = num_elements.div_ceil(BLOCK_ELEMENTS);
            Ok(num_blocks * Q6K_BLOCK_SIZE)
        }
        crate::core::tensor::TensorType::Q8_0 => {
            let num_blocks = num_elements.div_ceil(Q8_0_BLOCK_ELEMENTS);
            Ok(num_blocks * Q8_0_BLOCK_SIZE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bf16_le_to_f32;

    #[test]
    fn bf16_one_roundtrip_bits() {
        //1.0 as BF16 is 0x3f80 (LE bytes80 3f)
        let f = bf16_le_to_f32([0x80, 0x3f]);
        assert!((f - 1.0).abs() < 1e-6, "got {f}");
    }
}
