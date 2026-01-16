// This is the implementation of RMSNorm, over inputs
// The input should already be dequantized, and the learned weights of the RMSNorm shouldnt be quantized, because their precision matters

pub fn rmsnorm(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
    output: &mut [f32]
) -> Result<(), Box<dyn std::error::Error>> {

    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), weights.len(), "Dimension missmatch for RMSNorm");

    let mut sum_squared: f32 = 0.0;
    let dim: usize  = input.len();

    for i in 0..dim{
        sum_squared += input[i].powi(2)
    }
    let mean_squared: f32 = sum_squared/(dim as f32);
    let rms = (mean_squared + epsilon).sqrt();
    for i in 0..dim{
        output[i] = input[i] * weights[i]/rms;
    }
    Ok(())
}

mod test{
    use super::*;
    #[test]
    fn test_simple_rms(){
        let input: Vec<f32>= vec![0.5, 1.0, 1.5];
        let weights: Vec<f32> = vec![0.2, 0.3, 0.4];
        let epsilon: f32 = 1e-6;
        let mut output: Vec<f32>  = vec![0.0; input.len()];
        
        rmsnorm(&input, &weights, epsilon, &mut output).unwrap();
        
        let expected = vec![0.092582, 0.277746, 0.555492];
        
        for i in 0..input.len(){
            assert!((output[i]-expected[i]).abs() < 1e-3)
        }
    }
}