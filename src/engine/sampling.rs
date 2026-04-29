//! Map vocabulary logits to the next token id (greedy or stochastic).

use rand::Rng;
use thiserror::Error;

use crate::ops::softmax::softmax;

#[derive(Debug, Error)]
pub enum SamplingError {
    #[error("logits slice is empty")]
    EmptyLogits,

    #[error("invalid logits for argmax (empty or non-finite)")]
    InvalidLogits,

    #[error("temperature must be positive, got {0}")]
    InvalidTemperature(f32),

    #[error("softmax failed")]
    SoftmaxFailed,
}

/// Index of the largest logit. `None` if `logits` is empty or any entry is non-finite.
/// Prefers the **first** index among ties (same as a stable greedy tie-break).
pub fn argmax_index(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut best = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    Some(best)
}

/// Greedy choice: token id = argmax over logits.
pub fn sample_greedy(logits: &[f32]) -> Result<u32, SamplingError> {
    argmax_index(logits).map(|i| i as u32).ok_or({
        if logits.is_empty() {
            SamplingError::EmptyLogits
        } else {
            SamplingError::InvalidLogits
        }
    })
}

/// Stochastic choice: softmax(logits / `temperature`) then sample one index with `rng`.
///
/// As `temperature → 0`, behavior approaches greedy (use [`sample_greedy`] for exact argmax).
pub fn sample_temperature<R: Rng + ?Sized>(
    logits: &[f32],
    temperature: f32,
    rng: &mut R,
) -> Result<u32, SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::EmptyLogits);
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(SamplingError::InvalidTemperature(temperature));
    }

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let mut probs = vec![0.0f32; scaled.len()];
    softmax(&scaled, &mut probs).map_err(|_| SamplingError::SoftmaxFailed)?;

    let r: f32 = rng.gen_range(0.0f32..1.0f32);
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r < cum || i + 1 == probs.len() {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn argmax_picks_largest() {
        let logits = [0.0f32, 2.0, 1.0];
        assert_eq!(argmax_index(&logits), Some(1));
        assert_eq!(sample_greedy(&logits).unwrap(), 1);
    }

    #[test]
    fn greedy_empty_errors() {
        assert!(sample_greedy(&[]).is_err());
    }

    #[test]
    fn temperature_deterministic_with_seed() {
        let logits = [0.0f32, 1.0, 0.0];
        let mut rng = StdRng::seed_from_u64(12345);
        let t = sample_temperature(&logits, 1.0, &mut rng).unwrap();
        let mut rng2 = StdRng::seed_from_u64(12345);
        let t2 = sample_temperature(&logits, 1.0, &mut rng2).unwrap();
        assert_eq!(t, t2);
    }

    #[test]
    fn temperature_rejects_non_positive() {
        let logits = [1.0f32, 2.0];
        let mut rng = StdRng::seed_from_u64(0);
        assert!(sample_temperature(&logits, 0.0, &mut rng).is_err());
        assert!(sample_temperature(&logits, -1.0, &mut rng).is_err());
    }
}
