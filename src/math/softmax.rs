use ndarray::Array1;

pub fn softmax(vec: Array1<f32>) -> Array1<f32>{
    let result = vec.exp()/vec.exp().sum();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_1_d_vector() {
        let array:  Array1<f32> = Array1::linspace(0.0, 1.0, 5);
        let softmax_result = softmax(array);
        let true_softmax: Array1<f32> = Array1::from_vec(vec![0.11405072, 0.14644404, 0.18803786, 0.2414454, 0.310022]);
        assert!((softmax_result - true_softmax).abs().into_iter().all(|x| x < 0.0001));
    }
}