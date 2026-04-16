// This is the implementation of RMSNorm, over inputs
// The input should already be dequantized, and the learned weights of the RMSNorm shouldnt be quantized, because their precision matters

use crate::EngineError;

pub fn rmsnorm(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    output: &mut [f32]
) -> Result<(), EngineError> {

    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), weights.len(), "Dimension missmatch for RMSNorm");

    let mut sum_squared: f32 = 0.0;
    let dim: usize = input.len();

    for &x in input.iter() {
        sum_squared += x.powi(2);
    }
    let mean_squared: f32 = sum_squared / (dim as f32);
    let rms = (mean_squared + epsilon).sqrt();
    for ((out_slot, &x), &w) in output.iter_mut().zip(input.iter()).zip(weights.iter()) {
        *out_slot = x * w / rms;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::rmsnorm;

    #[test]
    fn test_simple_rms() {
        let input: Vec<f32> = vec![0.5, 1.0, 1.5];
        let weights: Vec<f32> = vec![0.2, 0.3, 0.4];
        let epsilon: f32 = 1e-6;
        let mut output: Vec<f32> = vec![0.0; input.len()];

        rmsnorm(&input, &weights, epsilon, &mut output).unwrap();

        let expected = [0.092_582, 0.277_746, 0.555_492];

        for i in 0..input.len() {
            assert!((output[i] - expected[i]).abs() < 1e-3);
        }
    }
}