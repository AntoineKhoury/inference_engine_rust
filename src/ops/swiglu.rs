pub fn sigmoid(
    input: &[f32],
    output: &mut [f32]
) -> Result<(), Box<dyn std::error::Error>>{
    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), output.len(), "Dimension mismatch for sigmoid");
    
    for i in 0..input.len(){
        if input[i] >0.0{
            output[i] = 1.0/(1.0+(-input[i]).exp())
        }
        else {
            output[i] = input[i].exp()/(1.0/input[i].exp())
        }
    }

    Ok(())
}

pub fn swiglu(
    x: &[f32],
    gate: &[f32],
    output: &mut [f32]
) -> Result<(), Box<dyn std::error::Error>>{

    #[cfg(debug_assertions)]
    debug_assert_eq!(x.len(), gate.len(), "Dimension mismatch for SwiGLU");

    let mut sigmoid_x = vec![0.0; x.len()];
    sigmoid(&x, &mut sigmoid_x)?;

    for i in 0..x.len(){
        output[i] = x[i] * sigmoid_x[i] * gate[i]
    }

    Ok(())
}

mod test{
    use super::*;
    #[test]
    fn simple_swiglu(){
    let x = vec![0.0, 1.0];
    let gate = vec![1.0, 1.0];
    let mut output = vec![0.0; x.len()];
    
    swiglu(&x, &gate, &mut output).unwrap();
    
    // Expected: [0.0, ~0.731]
    assert!((output[0] - 0.0).abs() < 1e-5);
    assert!((output[1] - 0.7310585786).abs() < 1e-3);
    }
}