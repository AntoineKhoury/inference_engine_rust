pub fn softmax(
    input: &[f32],
    output: &mut [f32]
) -> Result<(), Box<dyn std::error::Error>>{
    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), output.len(), "Dimenssion mismatch at softmax");

    // Check if we have at least one element
    if input.is_empty(){
        panic!("Empty input")
    }

    // Find max value for numerical stability
    let mut max = input[0];
    for i in 0..input.len(){
        if input[i] > max{
            max = input[i];
        }
    }
    let mut sum_exp = 0.0;
    for i in 0..input.len(){
        sum_exp += (input[i]-max).exp();
    }

    for i in 0..input.len(){
        output[i] = (input[i]-max).exp()/sum_exp;
    }

    Ok(())
}


mod test{
    use super::*;
    #[test]
    fn simple_softmax_test(){
        let input = vec![0.0, 1.0];
        let mut output = vec![0.0; input.len()];
        
        softmax(&input, &mut output).unwrap();
        
        // Check that outputs sum to 1.0 (probability distribution)
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax outputs should sum to 1.0");
        
        // Check individual values
        assert!((output[0] - 0.26894142).abs() < 1e-5);
        assert!((output[1] - 0.73105858).abs() < 1e-5);
    }
}