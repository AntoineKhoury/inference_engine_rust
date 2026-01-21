pub fn rope(vec: &mut [f32], base: u32, pos: u32, head_dim: u32, rotary_dim: u32) -> (){
    // Rope can be applied to a subset of the vec elements, but it still needs to be applied on an even
    // number of elements
    if rotary_dim > head_dim{
        panic!("Trying to rotate more elements than available in the vector for RoPE.")
    }

    let num_pairs = vec.len()/2;
    for i in 0..num_pairs{
        let angle = (pos as f32) * (base as f32).powf((-2.0*(i as f32))/(head_dim as f32));
        let temp_0 = vec[i];
        let temp_1 = vec[i+1];
        vec[i] = temp_0 * angle.cos() - temp_1 * angle.sin();
        vec[i+1] = temp_0 * angle.sin() + temp_1 * angle.cos();
    }
}

mod test{
    use super::*;
    #[test]
    fn test_rope(){
        let mut v = vec![1.0, 2.0];
        rope(&mut v[..], 1, 1, 2, 2);
        assert!((v[0] + 1.1426396637 ).abs() < 1e-5);
        assert!((v[1] - 1.9220755966 ).abs() < 1e-5);
    }
}