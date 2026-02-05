use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor {
    dtype: TensorType,
    buffer: Arc<Vec<u8>>,
    dimensions: Vec<usize>,
    stride: Vec<usize>,
    offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// Unquantized float32 tensors (used for layer normalization weights)
    F32,
    /// 4-bit quantization, unpacked to u8 (values 0-15)
    Q4K,
    /// 6-bit quantization, unpacked to u8 (values 0-63)
    Q6K,
}

impl Tensor {
    /// Create a new Tensor that owns a raw byte buffer.
    pub(crate) fn new(dtype: TensorType, buffer: Arc<Vec<u8>>, dimensions: Vec<usize>) -> Self {
        let stride = compute_row_major_stride(&dimensions);
        Self {
            dtype,
            buffer,
            dimensions,
            stride,
            offset: 0,
        }
    }

    /// Read a single F32 value from the buffer (little-endian).
    pub fn f32_at(&self, index: usize) -> Result<f32, Box<dyn std::error::Error>> {
        if self.dtype != TensorType::F32 {
            return Err("Tensor dtype is not F32".into());
        }
        let start = index
            .checked_mul(4)
            .ok_or("F32 index overflow")?;
        let end = start + 4;
        let bytes = self
            .buffer
            .get(start..end)
            .ok_or("F32 index out of bounds")?;
        Ok(f32::from_le_bytes(bytes.try_into()?))
    }

    /// Return a contiguous F32 slice for row-major tensors.
    pub fn as_f32_slice(&self) -> Result<&[f32], Box<dyn std::error::Error>> {
        if self.dtype != TensorType::F32 {
            return Err("Tensor dtype is not F32".into());
        }
        let (prefix, words, suffix) = unsafe { self.buffer.as_slice().align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("Tensor buffer is not aligned for F32".into());
        }
        Ok(words)
    }

    /// Return a contiguous mutable F32 slice for row-major tensors.
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], Box<dyn std::error::Error>> {
        if self.dtype != TensorType::F32 {
            return Err("Tensor dtype is not F32".into());
        }
        let buffer = Arc::get_mut(&mut self.buffer).ok_or("Tensor buffer is shared")?;
        let (prefix, words, suffix) = unsafe { buffer.as_mut_slice().align_to_mut::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("Tensor buffer is not aligned for F32".into());
        }
        Ok(words)
    }

    /// Access the raw byte buffer.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
    /// Get tensor dimensions
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn stride(&self) -> &[usize]{
        &self.stride
    }
    pub fn dtype(&self) -> TensorType{
        self.dtype
    }

}

fn compute_row_major_stride(dimensions: &[usize]) -> Vec<usize> {
    if dimensions.is_empty() {
        return Vec::new();
    }
    let mut stride = vec![0usize; dimensions.len()];
    let mut acc = 1usize;
    for (i, dim) in dimensions.iter().rev().enumerate() {
        let idx = dimensions.len() - 1 - i;
        stride[idx] = acc;
        acc = acc.saturating_mul(*dim);
    }
    stride
}