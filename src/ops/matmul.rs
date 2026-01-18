/// Matrix multiplication kernels
/// 
/// Architecture:
/// - Kernel dispatch layer selects appropriate implementation based on tensor types
/// - Scalar implementations with on-the-fly dequantization for quantized weights (Q4K, Q6K)
/// - Standard matmul for F32×F32 operations
/// 
/// Note: SIMD-optimized versions can be added later for performance
/// 
/// Matrix Layout:
/// - All tensors stored in row-major order
/// - Matmul: output = input × weight^T (weight is transposed conceptually)
///   For row-major: C[i,j] = sum_k(A[i,k] * B[k,j])
///   We compute: output[i] = sum_j(input[j] * weight[j,i])
///   This means we iterate over weight columns (which are contiguous in row-major)

use crate::core::types::{Tensor, TensorType};
use super::cpu_features::CpuFeatures;

/// Matrix multiplication: output = input × weight
/// 
/// # Arguments
/// * `input` - Input vector (1D tensor, typically activation from previous layer)
/// * `weight` - Weight matrix (2D tensor, can be F32, Q4K, or Q6K)
/// * `output` - Output buffer (must be pre-allocated with correct size)
/// * `cpu_features` - Detected CPU capabilities for kernel selection
/// 
/// # Panics
/// Panics if dimensions don't match or output buffer is wrong size
pub fn matmul(
    input: &[f32],
    weight: &Tensor,
    output: &mut [f32],
    cpu_features: &CpuFeatures,
) -> Result<(), Box<dyn std::error::Error>> {
    // Validate dimensions
    let weight_dims = weight.dimensions();
    if weight_dims.len() != 2 {
        return Err("Weight tensor must be 2D".into());
    }
    
    let in_features = weight_dims[0] as usize;
    let out_features = weight_dims[1] as usize;
    
    if input.len() != in_features {
        return Err(format!(
            "Input size {} doesn't match weight input dimension {}",
            input.len(), in_features
        ).into());
    }
    
    if output.len() != out_features {
        return Err(format!(
            "Output buffer size {} doesn't match weight output dimension {}",
            output.len(), out_features
        ).into());
    }
    
    // Dispatch to appropriate kernel based on weight tensor type
    match weight.tensor_type {
        TensorType::F32 => {
            matmul_f32_f32(input, weight, output, cpu_features)
        }
        TensorType::Q4K => {
            matmul_f32_q4k(input, weight, output, cpu_features)
        }
        TensorType::Q6K => {
            matmul_f32_q6k(input, weight, output, cpu_features)
        }
    }
}

/// F32 × F32 matrix multiplication
/// Scalar implementation: output[i] = sum_j(input[j] * weight[j, i])
/// 
/// Matrix layout: weight is stored in row-major order
/// weight[j * out_features + i] = weight[j, i]
fn matmul_f32_f32(
    input: &[f32],
    weight: &Tensor,
    output: &mut [f32],
    _cpu_features: &CpuFeatures,
) -> Result<(), Box<dyn std::error::Error>> {
    let weight_data = weight.f32_data()
        .ok_or("F32 tensor missing f32_data")?;
    
    let in_features = input.len();
    let out_features = output.len();
    
    // Initialize output to zero
    output.fill(0.0);
    
    // For each output feature
    for out_idx in 0..out_features {
        // Accumulate: output[out_idx] = sum(input[in_idx] * weight[in_idx, out_idx])
        for in_idx in 0..in_features {
            let weight_idx = in_idx * out_features + out_idx;
            output[out_idx] += input[in_idx] * weight_data[weight_idx];
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
    input: &[f32],
    weight: &Tensor,
    output: &mut [f32],
    _cpu_features: &CpuFeatures,
) -> Result<(), Box<dyn std::error::Error>> {
    let quantized_data = weight.quantized_data()
        .ok_or("Q4K tensor missing quantized_data")?;
    let scales = weight.scales()
        .ok_or("Q4K tensor missing scales")?;
    let mins = weight.mins()
        .ok_or("Q4K tensor missing mins")?;
    
    let in_features = input.len();
    let out_features = output.len();
    const BLOCK_SIZE: usize = 32; // 32 weights per block
    
    // Initialize output to zero
    output.fill(0.0);
    
    // For each output feature
    for out_idx in 0..out_features {
        // Accumulate: output[out_idx] = sum(input[in_idx] * dequantized_weight[in_idx, out_idx])
        for in_idx in 0..in_features {
            let weight_idx = in_idx * out_features + out_idx;
            let block_idx = weight_idx / BLOCK_SIZE;
            
            // Dequantize: weight = (quantized * scale) + min
            let quantized = quantized_data[weight_idx] as f32;
            let scale = scales[block_idx];
            let min = mins[block_idx];
            let dequantized_weight = (quantized * scale) + min;
            
            output[out_idx] += input[in_idx] * dequantized_weight;
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
    input: &[f32],
    weight: &Tensor,
    output: &mut [f32],
    _cpu_features: &CpuFeatures,
) -> Result<(), Box<dyn std::error::Error>> {
    let quantized_data = weight.quantized_data()
        .ok_or("Q6K tensor missing quantized_data")?;
    let scales = weight.scales()
        .ok_or("Q6K tensor missing scales")?;
    let mins = weight.mins()
        .ok_or("Q6K tensor missing mins")?;
    
    let in_features = input.len();
    let out_features = output.len();
    const BLOCK_SIZE: usize = 32; // 32 weights per block
    
    // Initialize output to zero
    output.fill(0.0);
    
    // For each output feature
    for out_idx in 0..out_features {
        // Accumulate: output[out_idx] = sum(input[in_idx] * dequantized_weight[in_idx, out_idx])
        for in_idx in 0..in_features {
            let weight_idx = in_idx * out_features + out_idx;
            let block_idx = weight_idx / BLOCK_SIZE;
            
            // Dequantize: weight = (quantized * scale) + min
            let quantized = quantized_data[weight_idx] as f32;
            let scale = scales[block_idx];
            let min = mins[block_idx];
            let dequantized_weight = (quantized * scale) + min;
            
            output[out_idx] += input[in_idx] * dequantized_weight;
        }
    }
    
    Ok(())
}


mod tests {
    use super::*;
    use crate::core::types::{Tensor, TensorType};
    use crate::ops::cpu_features::CpuFeatures;

    fn create_f32_tensor(data: Vec<f32>, dimensions: Vec<u64>) -> Tensor {
        Tensor::new(
            TensorType::F32,
            "test".to_string(),
            dimensions,
            data.len(),
            Some(data),
            None,
            None,
            None,
        )
    }

    fn create_q4k_tensor(quantized: Vec<u8>, scales: Vec<f32>, mins: Vec<f32>, dimensions: Vec<u64>) -> Tensor {
        Tensor::new(
            TensorType::Q4K,
            "test".to_string(),
            dimensions,
            quantized.len(),
            None,
            Some(quantized),
            Some(scales),
            Some(mins),
        )
    }

    fn create_q6k_tensor(quantized: Vec<u8>, scales: Vec<f32>, mins: Vec<f32>, dimensions: Vec<u64>) -> Tensor {
        Tensor::new(
            TensorType::Q6K,
            "test".to_string(),
            dimensions,
            quantized.len(),
            None,
            Some(quantized),
            Some(scales),
            Some(mins),
        )
    }

    #[test]
    fn test_matmul_f32_f32_simple() {
        let input = vec![1.0, 2.0];
        let weight = create_f32_tensor(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
        let mut output = vec![0.0; 2];
        matmul(&input, &weight, &mut output, &CpuFeatures::detect()).unwrap();
        assert!((output[0] - 5.0).abs() < 1e-5);
        assert!((output[1] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_q4k_simple() {
        let mut quantized = vec![0u8; 32];
        quantized[0] = 10;
        quantized[1] = 15;
        quantized[2] = 5;
        quantized[3] = 8;
        let weight = create_q4k_tensor(quantized, vec![0.1], vec![0.0], vec![2, 2]);
        let input = vec![1.0, 2.0];
        let mut output = vec![0.0; 2];
        matmul(&input, &weight, &mut output, &CpuFeatures::detect()).unwrap();
        assert!((output[0] - 4.0).abs() < 1e-5);
        assert!((output[1] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_f32_q6k_simple() {
        let mut quantized = vec![0u8; 32];
        quantized[0] = 50;
        let weight = create_q6k_tensor(quantized, vec![0.1], vec![0.0], vec![1, 1]);
        let input = vec![2.0];
        let mut output = vec![0.0; 1];
        matmul(&input, &weight, &mut output, &CpuFeatures::detect()).unwrap();
        assert!((output[0] - 10.0).abs() < 1e-5);
    }
}
