// Block iterator for quantized tensors stored in blocks
// Functional for Q4_K and Q6_K quantization types

use crate::core::tensor::TensorType;
use crate::ops::quant::quant_k_handler::{Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE, extract_scale_min_k4};
use crate::ops::quant::utils::f16_to_f32;

impl<'a> Iterator for BlockIter<'a> {
    type Item = BlockRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_index >= self.total_blocks {
            return None;
        }

        let (block_size, qs_len, qbits) = match self.dtype {
            TensorType::Q4K => (Q4K_BLOCK_SIZE, 128usize, 4u8),
            TensorType::Q6K => (Q6K_BLOCK_SIZE, 192usize, 6u8),
            _ => return None,
        };

        let block_start = self.block_index * block_size;
        let block_end = block_start + block_size;
        if block_end > self.buffer.len() {
            return None;
        }

        let block = &self.buffer[block_start..block_end];

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let scales_bytes = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for sub in 0..8 {
            let (s, m) = extract_scale_min_k4(sub, scales_bytes);
            scales[sub] = s;
            mins[sub] = m;
        }

        let qs_start = 16;
        let qs_end = qs_start + qs_len;
        let qs = &block[qs_start..qs_end];

        let out = BlockRef {
            block_index: self.block_index,
            d,
            dmin,
            scales,
            mins,
            qs,
            qbits,
        };

        self.block_index += 1;
        Some(out)
    }
}

#[derive(Debug, Clone)]
pub struct BlockIter<'a> {
    buffer: &'a [u8],
    dtype: TensorType,
    block_index: usize,
    total_blocks: usize,
}

#[derive(Debug, Clone)]
pub struct BlockRef<'a> {
    pub block_index: usize,
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 8],
    pub mins: [u8; 8],
    pub qs: &'a [u8],
    pub qbits: u8,
}
