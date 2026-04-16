//! GELU with tanh approximation (matches PyTorch `GELU(approximate="tanh")`).

use std::f32::consts::FRAC_2_SQRT_PI;

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    0.5 * x * (1.0 + (FRAC_2_SQRT_PI * (x + 0.044715 * x3)).tanh())
}

pub fn gelu_tanh_inplace(out: &mut [f32], input: &[f32]) {
    debug_assert_eq!(out.len(), input.len());
    for (o, &x) in out.iter_mut().zip(input.iter()) {
        *o = gelu_tanh(x);
    }
}
