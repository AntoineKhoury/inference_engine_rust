use crate::EngineError;

pub fn softmax(
    input: &[f32],
    output: &mut [f32]
) -> Result<(), EngineError> {
    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), output.len(), "Dimenssion mismatch at softmax");

    if input.is_empty() {
        return Err(EngineError::Op("softmax: empty input".into()));
    }

    // Find max value for numerical stability
    let mut max = input[0];
    for &x in input.iter() {
        if x > max {
            max = x;
        }
    }
    let mut sum_exp = 0.0f32;
    for &x in input.iter() {
        sum_exp += (x - max).exp();
    }

    for (out_slot, &x) in output.iter_mut().zip(input.iter()) {
        *out_slot = (x - max).exp() / sum_exp;
    }

    Ok(())
}


#[cfg(test)]
mod test {
    use super::softmax;

    #[test]
    fn simple_softmax_test() {
        let input = vec![0.0, 1.0];
        let mut output = vec![0.0; input.len()];

        softmax(&input, &mut output).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax outputs should sum to 1.0");

        assert!((output[0] - 0.268_941_4).abs() < 1e-5);
        assert!((output[1] - 0.731_058_6).abs() < 1e-5);
    }
}