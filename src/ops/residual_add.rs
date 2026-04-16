use crate::EngineError;

pub fn residual_add (
    input: &[f32],
    residual: &[f32],
    output: &mut [f32],
) -> Result<(), EngineError> {
    for i in 0..input.len(){
        output[i] = input[i] + residual[i];
    }
    Ok(())
}