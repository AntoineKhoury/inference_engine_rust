use std::sync::Arc;

#[derive(Debug)]
pub struct Tensor {
    pub dtype: TensorType,
    pub buffer: Arc<Vec<u8>>,
    pub dimensions: Vec<u64>,
    pub stride: Vec<u64>,
    pub offset: u64,
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
    pub(crate) fn new(dtype: TensorType, buffer: Arc<Vec<u8>>, dimensions: Vec<u64>) -> Self {
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

    /// Access the raw byte buffer.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
    /// Get tensor dimensions
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
}

fn compute_row_major_stride(dimensions: &[u64]) -> Vec<u64> {
    if dimensions.is_empty() {
        return Vec::new();
    }
    let mut stride = vec![0; dimensions.len()];
    let mut acc = 1u64;
    for (i, dim) in dimensions.iter().rev().enumerate() {
        let idx = dimensions.len() - 1 - i;
        stride[idx] = acc;
        acc = acc.saturating_mul(*dim);
    }
    stride
}