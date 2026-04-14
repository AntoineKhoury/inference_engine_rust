/// 2D weight layout matches **ggml / GGUF** (same as llama.cpp): for `ne = [ne0, ne1]`,
/// element `(i0, i1)` is at **`i0 + i1 * ne0`** (first dimension stride-1), **not** C row-major
/// `i0 * ne1 + i1`. Matmul uses `W(input_kk, out_col)` at `kk + col * K` with `K = ne0`.

use crate::core::tensor::{Tensor, TensorType};
use crate::ops::quant::quant_K_handler::{
    dequantize_q4k_block, dequantize_q6k_block, Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE,
};

const BLOCK_ELEMENTS: usize = 256;

pub fn matmul(
    a: &Tensor,
    b: &Tensor,
    output: &mut Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate dimensions
    let b_dims = b.dimensions();
    let a_dims = a.dimensions();
    let output_dims = output.dimensions();

    
    if a_dims[1] != b_dims[0] {
        return Err(format!(
            "Input size {} doesn't match weight input dimension {}",
            a_dims[1], b_dims[0]
        ).into());
    }
    
    if (output_dims[0] * output_dims[1]) != (a_dims[0] * b_dims[1]) {
        panic!(
            "Dimensions of output tensor {:?},{:?} don't match the input tensors dimensions of matmul inputs {:?},{:?}.", 
            output_dims[0], 
            output_dims[1],
            a_dims[0],
            b_dims[1]
        );
    }

    // Dispatch to appropriate kernel based on weight tensor type
    match (a.dtype(), b.dtype()) {
        (TensorType::F32, TensorType::F32) => matmul_f32_f32(a,b, output),
        (TensorType::F32,TensorType::Q4K) => matmul_f32_q4k(a,b, output),
        (TensorType::F32,TensorType::Q6K) => matmul_f32_q6k(a, b, output),
        _ => panic!("Type combination {:?} and {:?} for matmul isn't implemented", a.dtype(), b.dtype())
    }
}

/// F32 × F32 matrix multiplication  
/// `output[row, col] = sum_kk input[row, kk] * W(kk, col)` with ggml `W` indexing.
fn matmul_f32_f32(input: &Tensor, weight: &Tensor, output: &mut Tensor) -> Result<(), Box<dyn std::error::Error>> {
    // Expect input: [M, K], weight: [K, N], output: [M, N]
    if input.dimensions().len() != 2 || weight.dimensions().len() != 2 || output.dimensions().len() != 2 {
        return Err("F32 matmul expects 2D tensors for input, weight, and output".into());
    }
    let m = input.dimensions()[0];
    let k = input.dimensions()[1];
    let n = weight.dimensions()[1];

    if weight.dimensions()[0] != k {
        return Err("Input K dimension does not match weight K dimension".into());
    }
    if output.dimensions()[0] != m || output.dimensions()[1] != n {
        return Err("Output dimensions do not match MxN of matmul".into());
    }

    let input_data = input.as_f32_slice()?;
    let weight_data = weight.as_f32_slice()?;
    let output_data = output.as_f32_slice_mut()?;

    for row in 0..m {
        let input_row_start = row * k;
        let output_row_start = row * n;
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a = input_data[input_row_start + kk];
                let w = weight_data[kk + col * k];
                acc += a * w;
            }
            output_data[output_row_start + col] = acc;
        }
    }

    Ok(())
}

/// F32 × Q4K matrix multiplication with fused dequantization
/// 
/// Scalar implementation with on-the-fly dequantization:
/// - Dequantize: weight = (quantized * scale) + min
/// - Scales/mins are per block of 32 weights
/// - Q4K: quantized values are in range 0-15
/// 
/// This avoids writing dequantized weights to memory, improving cache locality
fn matmul_f32_q4k(
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    if input.dimensions().len() != 2 || weight.dimensions().len() != 2 || output.dimensions().len() != 2 {
        return Err("Q4K matmul expects 2D tensors for input, weight, and output".into());
    }
    if input.dtype() != TensorType::F32 || output.dtype() != TensorType::F32 || weight.dtype() != TensorType::Q4K {
        return Err("Q4K matmul expects F32 input/output and Q4K weights".into());
    }

    let m = input.dimensions()[0];
    let k = input.dimensions()[1];
    let n = weight.dimensions()[1];

    if weight.dimensions()[0] != k {
        return Err("Input K dimension does not match weight K dimension".into());
    }
    if output.dimensions()[0] != m || output.dimensions()[1] != n {
        return Err("Output dimensions do not match MxN of matmul".into());
    }

    let input_data = input.as_f32_slice()?;
    let output_data = output.as_f32_slice_mut()?;
    let weight_bytes = weight.buffer();

    let total_weights = k * n;
    let total_blocks = (total_weights + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    let expected_bytes = total_blocks * Q4K_BLOCK_SIZE;
    if weight_bytes.len() < expected_bytes {
        return Err("Q4K weight buffer is smaller than expected".into());
    }

    let mut decoded_block = [0.0f32; BLOCK_ELEMENTS];
    let mut current_block_idx = usize::MAX;

    for row in 0..m {
        let input_row_start = row * k;
        let output_row_start = row * n;

        // `col` outer, `kk` inner: `weight_idx = kk + col * k` runs contiguous in `kk` (stride-1 in ggml
        // buffer). The old `kk` outer loop stepped by `k` in `col` and was catastrophically slow on CPU.
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a = input_data[input_row_start + kk];
                if a == 0.0 {
                    continue;
                }
                let weight_idx = kk + col * k;
                let block_idx = weight_idx / BLOCK_ELEMENTS;
                if block_idx != current_block_idx {
                    let block_start = block_idx * Q4K_BLOCK_SIZE;
                    let block_end = block_start + Q4K_BLOCK_SIZE;
                    let block = weight_bytes
                        .get(block_start..block_end)
                        .ok_or("Q4K block out of bounds")?;
                    dequantize_q4k_block(block, &mut decoded_block)?;
                    current_block_idx = block_idx;
                }
                let w = decoded_block[weight_idx % BLOCK_ELEMENTS];
                acc += a * w;
            }
            output_data[output_row_start + col] = acc;
        }
    }

    Ok(())
}

/// F32 × Q6K matrix multiplication with fused dequantization
/// Similar to Q4K but handles 6-bit quantization (values 0-63)
/// 
/// Scalar implementation with on-the-fly dequantization:
/// - Dequantize: weight = (quantized * scale) + min
/// - Scales/mins are per block of 32 weights
/// - Q6K: quantized values are in range 0-63
fn matmul_f32_q6k(
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    if input.dimensions().len() != 2 || weight.dimensions().len() != 2 || output.dimensions().len() != 2 {
        return Err("Q6K matmul expects 2D tensors for input, weight, and output".into());
    }
    if input.dtype() != TensorType::F32 || output.dtype() != TensorType::F32 || weight.dtype() != TensorType::Q6K {
        return Err("Q6K matmul expects F32 input/output and Q6K weights".into());
    }

    let m = input.dimensions()[0];
    let k = input.dimensions()[1];
    let n = weight.dimensions()[1];

    if weight.dimensions()[0] != k {
        return Err("Input K dimension does not match weight K dimension".into());
    }
    if output.dimensions()[0] != m || output.dimensions()[1] != n {
        return Err("Output dimensions do not match MxN of matmul".into());
    }

    let input_data = input.as_f32_slice()?;
    let output_data = output.as_f32_slice_mut()?;
    let weight_bytes = weight.buffer();

    let total_weights = k * n;
    let total_blocks = (total_weights + BLOCK_ELEMENTS - 1) / BLOCK_ELEMENTS;
    let expected_bytes = total_blocks * Q6K_BLOCK_SIZE;
    if weight_bytes.len() < expected_bytes {
        return Err("Q6K weight buffer is smaller than expected".into());
    }

    let mut decoded_block = [0.0f32; BLOCK_ELEMENTS];
    let mut current_block_idx = usize::MAX;

    for row in 0..m {
        let input_row_start = row * k;
        let output_row_start = row * n;

        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a = input_data[input_row_start + kk];
                if a == 0.0 {
                    continue;
                }
                let weight_idx = kk + col * k;
                let block_idx = weight_idx / BLOCK_ELEMENTS;
                if block_idx != current_block_idx {
                    let block_start = block_idx * Q6K_BLOCK_SIZE;
                    let block_end = block_start + Q6K_BLOCK_SIZE;
                    let block = weight_bytes
                        .get(block_start..block_end)
                        .ok_or("Q6K block out of bounds")?;
                    dequantize_q6k_block(block, &mut decoded_block)?;
                    current_block_idx = block_idx;
                }
                let w = decoded_block[weight_idx % BLOCK_ELEMENTS];
                acc += a * w;
            }
            output_data[output_row_start + col] = acc;
        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{Tensor, TensorType};
    use std::sync::Arc;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn create_f32_tensor(data: Vec<f32>, dimensions: Vec<usize>) -> Tensor {
        Tensor::new(TensorType::F32, Arc::new(f32_bytes(&data)), dimensions)
    }

    fn create_q4k_tensor(buffer: Vec<u8>, dimensions: Vec<usize>) -> Tensor {
        Tensor::new(TensorType::Q4K, Arc::new(buffer), dimensions)
    }

    fn create_q6k_tensor(buffer: Vec<u8>, dimensions: Vec<usize>) -> Tensor {
        Tensor::new(TensorType::Q6K, Arc::new(buffer), dimensions)
    }

    fn create_zero_f32_tensor(dimensions: Vec<usize>) -> Tensor {
        let len = dimensions.iter().product::<usize>();
        let data = vec![0.0f32; len];
        create_f32_tensor(data, dimensions)
    }

    #[test]
    fn test_matmul_f32_f32_simple() {
        let input = create_f32_tensor(vec![1.0, 2.0], vec![1, 2]);
        // ggml [2,2]: (kk,col) -> kk + col*2; logical W = [[1,3],[2,4]] -> [1,2,3,4]
        let weight = create_f32_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut output = create_zero_f32_tensor(vec![1, 2]);
        matmul(&input, &weight, &mut output).unwrap();
        let out = output.as_f32_slice().unwrap();
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_q4k_simple() {
        let buffer = vec![0u8; Q4K_BLOCK_SIZE];
        let weight = create_q4k_tensor(buffer, vec![2, 2]);
        let input = create_f32_tensor(vec![1.0, 2.0], vec![1, 2]);
        let mut output = create_zero_f32_tensor(vec![1, 2]);
        matmul(&input, &weight, &mut output).unwrap();
        let out = output.as_f32_slice().unwrap();
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!((out[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_q6k_simple() {
        let buffer = vec![0u8; Q6K_BLOCK_SIZE];
        let weight = create_q6k_tensor(buffer, vec![1, 1]);
        let input = create_f32_tensor(vec![2.0], vec![1, 1]);
        let mut output = create_zero_f32_tensor(vec![1, 1]);
        matmul(&input, &weight, &mut output).unwrap();
        let out = output.as_f32_slice().unwrap();
        assert!((out[0] - 0.0).abs() < 1e-5);
    }
}
