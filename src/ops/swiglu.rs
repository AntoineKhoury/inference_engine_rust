use crate::EngineError;

pub fn sigmoid(input: &[f32], output: &mut [f32]) -> Result<(), EngineError> {
    #[cfg(debug_assertions)]
    debug_assert_eq!(input.len(), output.len(), "Dimension mismatch for sigmoid");

    for i in 0..input.len() {
        let x = input[i];
        // Numerically stable sigmoid: 1 / (1 + exp(-x))
        if x >= 0.0 {
            let z = (-x).exp();
            output[i] = 1.0 / (1.0 + z);
        } else {
            let z = x.exp();
            output[i] = z / (1.0 + z);
        }
    }

    Ok(())
}

/// Llama/Mistral FFN gated activation: **SiLU(gate) × up** (same as `silu(gate) * up` in HF / llama.cpp).
/// `gate` is the gate projection row; `up` is the up projection row (same length).
pub fn swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) -> Result<(), EngineError> {
    #[cfg(debug_assertions)]
    debug_assert_eq!(gate.len(), up.len(), "Dimension mismatch for SwiGLU");

    let mut sigmoid_gate = vec![0.0; gate.len()];
    sigmoid(gate, &mut sigmoid_gate)?;

    for i in 0..gate.len() {
        output[i] = gate[i] * sigmoid_gate[i] * up[i];
    }

    Ok(())
}

mod test {
    #[test]
    fn simple_swiglu() {
        let gate = vec![0.0, 1.0];
        let up = vec![1.0, 1.0];
        let mut output = vec![0.0; gate.len()];

        super::swiglu(&gate, &up, &mut output).unwrap();

        // SiLU(0)*1 = 0; SiLU(1)*1 ≈ 0.731
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 0.731_058_6).abs() < 1e-3);
    }
}
