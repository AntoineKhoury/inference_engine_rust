use std::sync::Arc;

use crate::EngineError;

#[derive(Debug)]
pub struct Tensor {
    dtype: TensorType,
    buffer: Arc<Vec<u8>>,
    dimensions: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// Unquantized float32 tensors (used for layer normalization weights)
    F32,
    /// 4-bit quantization, unpacked to u8 (values 0-15)
    Q4K,
    /// 6-bit quantization, unpacked to u8 (values 0-63)
    Q6K,
    /// Q8_0: blocks of 32 int8 values with one fp16 scale per block (ggml `block_q8_0`).
    Q8_0,
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
        }
    }

    /// Read a single F32 value from the buffer (little-endian).
    pub fn f32_at(&self, index: usize) -> Result<f32, EngineError> {
        if self.dtype != TensorType::F32 {
            return Err(EngineError::Tensor("dtype is not F32".into()));
        }
        let start = index
            .checked_mul(4)
            .ok_or_else(|| EngineError::Tensor("F32 index overflow".into()))?;
        let end = start + 4;
        let bytes = self
            .buffer
            .get(start..end)
            .ok_or_else(|| EngineError::Tensor("F32 index out of bounds".into()))?;
        let arr: [u8; 4] = bytes
            .try_into()
            .map_err(|_| EngineError::Tensor("F32 read: expected 4 bytes".into()))?;
        Ok(f32::from_le_bytes(arr))
    }

    /// Return a contiguous F32 slice for row-major tensors.
    pub fn as_f32_slice(&self) -> Result<&[f32], EngineError> {
        if self.dtype != TensorType::F32 {
            return Err(EngineError::Tensor("dtype is not F32".into()));
        }
        // SAFETY: `words` is only returned if there is no unaligned prefix/suffix.
        let (prefix, words, suffix) = unsafe { self.buffer.as_slice().align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(EngineError::Tensor("buffer not aligned for F32".into()));
        }
        Ok(words)
    }

    /// Return a contiguous mutable F32 slice for row-major tensors.
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], EngineError> {
        if self.dtype != TensorType::F32 {
            return Err(EngineError::Tensor("dtype is not F32".into()));
        }
        let buffer = Arc::get_mut(&mut self.buffer)
            .ok_or_else(|| EngineError::Tensor("buffer is shared (Arc)".into()))?;
        let (prefix, words, suffix) = unsafe { buffer.as_mut_slice().align_to_mut::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(EngineError::Tensor("buffer not aligned for F32".into()));
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