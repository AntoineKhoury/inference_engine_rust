use crate::EngineError;

/// RoPE on `vec` (one head): rotate the first `rotary_dim` dimensions in non-overlapping pairs.
///
/// Matches ggml `GGML_OP_ROPE` / `ggml_rope_cache_init` when `freq_factors` is set: per pair `k`,
/// angle = `theta / ff[k]` where `theta` starts at `pos` and each step `theta *= base^(-2/n_rot)`
/// with `n_rot = rotary_dim` (llama.cpp `n_dims` passed to `ggml_rope_ext`).
///
/// Gemma 4 **full-attention** layers store `blk.*.rope_freqs` (proportional RoPE); pass that slice
/// (length ≥ `rotary_dim/2`, typically `head_dim/2`). Sliding / Mistral: use `freq_factors: None`.
pub fn rope(
    vec: &mut [f32],
    base: f32,
    pos: u32,
    head_dim: u32,
    rotary_dim: u32,
    freq_factors: Option<&[f32]>,
) -> Result<(), EngineError> {
    if rotary_dim > head_dim {
        return Err(EngineError::Op(format!(
            "RoPE rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }

    let end = (rotary_dim as usize).min(vec.len());
    if end % 2 != 0 {
        return Err(EngineError::Op("RoPE rotary span must be even".into()));
    }
    let num_pairs = end / 2;
    if let Some(ff) = freq_factors {
        if ff.len() < num_pairs {
            return Err(EngineError::Op(format!(
                "RoPE freq_factors len {} < num_pairs {}",
                ff.len(),
                num_pairs
            )));
        }
    }

    let n_rot = rotary_dim as f32;
    let theta_scale = base.powf(-2.0 / n_rot);
    let mut theta = pos as f32;

    for k in 0..num_pairs {
        let ff = freq_factors
            .and_then(|f| f.get(k))
            .copied()
            .filter(|x| *x != 0.0)
            .unwrap_or(1.0);
        let angle = theta / ff;
        let p = 2 * k;
        let temp_0 = vec[p];
        let temp_1 = vec[p + 1];
        vec[p] = temp_0 * angle.cos() - temp_1 * angle.sin();
        vec[p + 1] = temp_0 * angle.sin() + temp_1 * angle.cos();
        theta *= theta_scale;
    }
    Ok(())
}

mod test {
    #[test]
    fn test_rope_dim2() {
        let mut v = [1.0, 2.0];
        super::rope(&mut v[..], 1.0, 1, 2, 2, None).unwrap();
        assert!((v[0] + 1.142_639_6).abs() < 1e-5);
        assert!((v[1] - 1.922_075_6).abs() < 1e-5);
    }

    /// Pairs must be (0,1) and (2,3), not overlapping (0,1),(1,2),(2,3).
    #[test]
    fn test_rope_dim4_pairs_non_overlapping() {
        let mut v = [1.0f32, 0.0, 1.0, 0.0];
        super::rope(&mut v[..], 10000.0, 0, 4, 4, None).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5 && (v[1] - 0.0).abs() < 1e-5);
        assert!((v[2] - 1.0).abs() < 1e-5 && (v[3] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn freq_factor_doubles_effective_angle_for_pair0() {
        let mut a = [1.0f32, 0.0];
        let mut b = [1.0f32, 0.0];
        let ff = [2.0f32];
        super::rope(&mut a, 10000.0, 1, 2, 2, None).unwrap();
        super::rope(&mut b, 10000.0, 1, 2, 2, Some(&ff)).unwrap();
        assert!(a != b);
        assert!((a[0] - b[0]).abs() > 1e-3);
    }
}
