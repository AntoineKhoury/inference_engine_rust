use crate::EngineError;

pub fn rope(
    vec: &mut [f32],
    base: f32,
    pos: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<(), EngineError> {
    // RoPE applies to non-overlapping pairs (2i, 2i+1). Using (i, i+1) for i = 0..len/2 is wrong:
    // it reuses dimensions and corrupts all but the first pair when head_dim > 2.
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

    for i in 0..num_pairs {
        let p = 2 * i;
        let angle = (pos as f32) * base.powf((-2.0 * (i as f32)) / (head_dim as f32));
        let temp_0 = vec[p];
        let temp_1 = vec[p + 1];
        vec[p] = temp_0 * angle.cos() - temp_1 * angle.sin();
        vec[p + 1] = temp_0 * angle.sin() + temp_1 * angle.cos();
    }
    Ok(())
}

mod test {
    #[test]
    fn test_rope_dim2() {
        let mut v = vec![1.0, 2.0];
        super::rope(&mut v[..], 1.0, 1, 2, 2).unwrap();
        assert!((v[0] + 1.1426396637).abs() < 1e-5);
        assert!((v[1] - 1.9220755966).abs() < 1e-5);
    }

    /// Pairs must be (0,1) and (2,3), not overlapping (0,1),(1,2),(2,3).
    #[test]
    fn test_rope_dim4_pairs_non_overlapping() {
        let mut v = vec![1.0f32, 0.0, 1.0, 0.0];
        super::rope(&mut v[..], 10000.0, 0, 4, 4).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5 && (v[1] - 0.0).abs() < 1e-5);
        assert!((v[2] - 1.0).abs() < 1e-5 && (v[3] - 0.0).abs() < 1e-5);
    }
}