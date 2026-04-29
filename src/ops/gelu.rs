//! GELU with tanh approximation (matches PyTorch `GELU(approximate="tanh")`).

use std::f32::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI};

const SQRT_2_OVER_PI: f32 = FRAC_2_SQRT_PI * FRAC_1_SQRT_2;

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x3)).tanh())
}

pub fn gelu_tanh_inplace(out: &mut [f32], input: &[f32]) {
    debug_assert_eq!(out.len(), input.len());
    for (o, &x) in out.iter_mut().zip(input.iter()) {
        *o = gelu_tanh(x);
    }
}

#[cfg(test)]
mod tests {
    use super::gelu_tanh;

    #[test]
    fn gelu_tanh_matches_pytorch_approximate_tanh_reference_points() {
        let cases = [
            (-1.0, -0.15880801),
            (0.0, 0.0),
            (1.0, 0.841192),
            (2.0, 1.9545977),
        ];

        for (x, expected) in cases {
            let got = gelu_tanh(x);
            assert!(
                (got - expected).abs() < 1e-6,
                "gelu_tanh({x}) = {got}, expected {expected}"
            );
        }
    }
}
