use std::io::{BufRead, Seek};
use std::sync::Arc;

use crate::core::tensor::Tensor;
use crate::EngineError;
use crate::model_loader::gguf_types::TensorInfo;
use crate::model_loader::reader::Reader;
use crate::model_loader::tensor::GgmlType;
use crate::ops::quant::quant_k_handler::{Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE};
const BLOCK_ELEMENTS: usize = 256;

/// Load a single tensor from the file based on TensorInfo.
/// This reads raw bytes into the tensor buffer without decoding.
///
/// `tensor_data_base` is the absolute file offset where the GGUF **tensor data blob** begins
/// (immediately after the tensor metadata array). Per GGUF, each `TensorInfo.offset` is relative
/// to that base, not to the start of the file.
pub fn load_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
    tensor_data_base: u64,
) -> Result<Tensor, EngineError> {
    let ggml_type = GgmlType::try_from(tensor_info.type_id)?;
    let tensor_type = ggml_type.to_tensor_type()?;
    let num_elements = tensor_info
        .dimensions
        .iter()
        .product::<usize>();

    let byte_len = expected_byte_len(tensor_type, num_elements)?;

    let abs_offset = tensor_data_base
        .checked_add(tensor_info.offset as u64)
        .ok_or_else(|| EngineError::Gguf("tensor offset overflow".into()))?;

    // Seek to the tensor's data (GGUF: offset is relative to tensor data section)
    reader.seek(abs_offset)?;

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
    }
}
