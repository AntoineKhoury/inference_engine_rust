use std::io::{BufRead, Seek};
use std::sync::Arc;

use crate::core::tensor::Tensor;
use crate::model_loader::gguf_types::TensorInfo;
use crate::model_loader::reader::Reader;
use crate::model_loader::tensor::GgmlType;

const Q4K_BLOCK_SIZE: usize = 144;
const Q6K_BLOCK_SIZE: usize = 208;
const BLOCK_ELEMENTS: usize = 256;

/// Load a single tensor from the file based on TensorInfo.
/// This reads raw bytes into the tensor buffer without decoding.
pub fn load_tensor<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_info: &TensorInfo,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let ggml_type = GgmlType::try_from(tensor_info.type_id)?;
    let tensor_type = ggml_type.to_tensor_type()?;
    let num_elements = tensor_info
        .dimensions
        .iter()
        .product::<usize>();

    let byte_len = expected_byte_len(tensor_type, num_elements)?;

    // Seek to the tensor's offset
    reader.seek(tensor_info.offset as u64).map_err(|e| {
        format!(
            "Failed to seek to offset {} for tensor '{}': {}",
            tensor_info.offset, tensor_info.name, e
        )
    })?;

    let buffer = reader.read_bytes(byte_len as u64).map_err(|e| {
        format!(
            "Failed to read {} bytes for tensor '{}': {}",
            byte_len, tensor_info.name, e
        )
    })?;

    Ok(Tensor::new(
        tensor_type,
        Arc::new(buffer),
        tensor_info.dimensions.clone(),
    ))
}

fn expected_byte_len(
    tensor_type: crate::core::tensor::TensorType,
    num_elements: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    match tensor_type {
        crate::core::tensor::TensorType::F32 => Ok(num_elements * 4),
        crate::core::tensor::TensorType::Q4K => {
            let num_blocks = (num_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
            Ok(num_blocks * Q4K_BLOCK_SIZE)
        }
        crate::core::tensor::TensorType::Q6K => {
            let num_blocks = (num_elements + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
            Ok(num_blocks * Q6K_BLOCK_SIZE)
        }
    }
}
