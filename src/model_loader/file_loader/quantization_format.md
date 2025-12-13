# GGML Quantization Format Reference

This document explains how quantized tensors are stored in GGUF files.

## Key Concepts

**Blocks**: Quantized data is organized into blocks (typically 16 or 32 weights per block)
**Superblocks**: Some formats group blocks into superblocks (e.g., 8 blocks = 256 weights)
**Scales**: Each block has scaling factors to convert quantized values back to floats
**Mins**: Some formats store minimum values per block for asymmetric quantization

## Common Formats

### F32 (Type ID: 0)
- **Block size**: N/A (no blocks)
- **Bytes per weight**: 4 bytes
- **Layout**: Just raw f32 values in little-endian
- **Total size**: `num_elements * 4`

### F16 (Type ID: 1)
- **Block size**: N/A
- **Bytes per weight**: 2 bytes
- **Layout**: Just raw f16 values (half-precision float)
- **Total size**: `num_elements * 2`

### Q4_0 (Type ID: 2)
- **Block size**: 32 weights per block
- **Bytes per block**: 18 bytes
  - 16 bytes: 32 weights × 4 bits = 128 bits = 16 bytes (packed)
  - 2 bytes: scale factor (f16)
- **Layout per block**:
  ```
  [16 bytes: quantized weights (4 bits each, packed)]
  [2 bytes: scale (f16, little-endian)]
  ```
- **Total size**: `(num_elements / 32) * 18` bytes
- **Dequantization**: `weight = (quantized_value - 8) * scale`

### Q4_1 (Type ID: 3)
- **Block size**: 32 weights per block
- **Bytes per block**: 20 bytes
  - 16 bytes: 32 weights × 4 bits = 16 bytes (packed)
  - 2 bytes: scale (f16)
  - 2 bytes: minimum value (f16)
- **Layout per block**:
  ```
  [16 bytes: quantized weights (4 bits each, packed)]
  [2 bytes: scale (f16)]
  [2 bytes: min (f16)]
  ```
- **Total size**: `(num_elements / 32) * 20` bytes
- **Dequantization**: `weight = quantized_value * scale + min`

### Q8_0 (Type ID: 8)
- **Block size**: 32 weights per block
- **Bytes per block**: 34 bytes
  - 32 bytes: 32 weights × 8 bits = 32 bytes
  - 2 bytes: scale (f16)
- **Layout per block**:
  ```
  [32 bytes: quantized weights (8 bits each)]
  [2 bytes: scale (f16)]
  ```
- **Total size**: `(num_elements / 32) * 34` bytes
- **Dequantization**: `weight = (quantized_value - 128) * scale`

### Q4_K (Type ID: 12) - Your model uses this!
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: 160 bytes
  - 128 bytes: 256 weights × 4 bits = 128 bytes (packed)
  - 16 bytes: 8 scales (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
  - 16 bytes: 8 minimums (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
- **Layout per superblock**:
  ```
  [128 bytes: quantized weights for all 8 blocks (4 bits each, packed)]
  [16 bytes: 8 scales (f16 each, 2 bytes per scale)]
  [16 bytes: 8 minimums (f16 each, 2 bytes per min)]
  ```
- **Total size**: `(num_elements / 256) * 160` bytes
- **Bits per weight**: (160 bytes × 8 bits) / 256 weights = 1280 bits / 256 = 5 bits per weight
- **Dequantization per block**: `weight = quantized_value * scale[block_idx] + min[block_idx]`

### Q5_0 (Type ID: 6)
- **Block size**: 32 weights per block
- **Bytes per block**: 22 bytes
  - 20 bytes: 32 weights × 5 bits = 160 bits = 20 bytes (packed)
  - 2 bytes: scale (f16)
- **Total size**: `(num_elements / 32) * 22` bytes

### Q5_1 (Type ID: 7)
- **Block size**: 32 weights per block
- **Bytes per block**: 24 bytes
  - 20 bytes: 32 weights × 5 bits = 20 bytes (packed)
  - 2 bytes: scale (f16)
  - 2 bytes: min (f16)
- **Layout per block**:
  ```
  [20 bytes: quantized weights (5 bits each, packed)]
  [2 bytes: scale (f16)]
  [2 bytes: min (f16)]
  ```
- **Total size**: `(num_elements / 32) * 24` bytes
- **Dequantization**: `weight = quantized_value * scale + min`

### Q8_1 (Type ID: 9)
- **Block size**: 32 weights per block
- **Bytes per block**: 36 bytes
  - 32 bytes: 32 weights × 8 bits = 32 bytes
  - 2 bytes: scale (f16)
  - 2 bytes: min (f16)
- **Layout per block**:
  ```
  [32 bytes: quantized weights (8 bits each)]
  [2 bytes: scale (f16)]
  [2 bytes: min (f16)]
  ```
- **Total size**: `(num_elements / 32) * 36` bytes
- **Dequantization**: `weight = quantized_value * scale + min`

### Q2_K (Type ID: 10)
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: ~144 bytes (exact structure may vary)
  - 64 bytes: 256 weights × 2 bits = 64 bytes (packed)
  - Scales and minimums per block (structure similar to Q4_K)
- **Layout per superblock**: Similar to Q4_K but with 2-bit quantization
- **Total size**: `(num_elements / 256) * ~144` bytes
- **Note**: Exact byte layout may need verification from GGML source

### Q3_K (Type ID: 11)
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: ~152 bytes (exact structure may vary)
  - 96 bytes: 256 weights × 3 bits = 96 bytes (packed)
  - Scales and minimums per block (structure similar to Q4_K)
- **Layout per superblock**: Similar to Q4_K but with 3-bit quantization
- **Total size**: `(num_elements / 256) * ~152` bytes
- **Note**: Exact byte layout may need verification from GGML source

### Q5_K (Type ID: 13)
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: ~176 bytes (exact structure may vary)
  - 160 bytes: 256 weights × 5 bits = 160 bytes (packed)
  - Scales and minimums per block (structure similar to Q4_K)
- **Layout per superblock**: Similar to Q4_K but with 5-bit quantization
- **Total size**: `(num_elements / 256) * ~176` bytes
- **Note**: Exact byte layout may need verification from GGML source

### Q6_K (Type ID: 14) - Your model uses this!
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: 208 bytes
  - 192 bytes: 256 weights × 6 bits = 192 bytes (packed)
  - 16 bytes: 8 scales (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
  - 16 bytes: 8 minimums (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
- **Layout per superblock**:
  ```
  [192 bytes: quantized weights for all 8 blocks (6 bits each, packed)]
  [16 bytes: 8 scales (f16 each, 2 bytes per scale)]
  [16 bytes: 8 minimums (f16 each, 2 bytes per min)]
  ```
- **Total size**: `(num_elements / 256) * 208` bytes
- **Bits per weight**: (208 bytes × 8 bits) / 256 weights = 1664 bits / 256 = 6.5 bits per weight
- **Dequantization per block**: `weight = quantized_value * scale[block_idx] + min[block_idx]`

### Q8_K (Type ID: 15)
- **Superblock size**: 256 weights (8 blocks × 32 weights)
- **Bytes per superblock**: 272 bytes
  - 256 bytes: 256 weights × 8 bits = 256 bytes
  - 16 bytes: 8 scales (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
  - 16 bytes: 8 minimums (one per block, each is f16 = 2 bytes) → 8 × 2 = 16 bytes
- **Layout per superblock**:
  ```
  [256 bytes: quantized weights for all 8 blocks (8 bits each)]
  [16 bytes: 8 scales (f16 each, 2 bytes per scale)]
  [16 bytes: 8 minimums (f16 each, 2 bytes per min)]
  ```
- **Total size**: `(num_elements / 256) * 272` bytes
- **Bits per weight**: (272 bytes × 8 bits) / 256 weights = 2176 bits / 256 = 8.5 bits per weight
- **Dequantization per block**: `weight = quantized_value * scale[block_idx] + min[block_idx]`

### BF16 (Type ID: 30)
- **Block size**: N/A (no blocks)
- **Bytes per weight**: 2 bytes
- **Layout**: Just raw bf16 values (Brain Float 16)
- **Total size**: `num_elements * 2`

## Reading Strategy

1. **Calculate total elements**: Multiply all dimensions together
2. **Determine block/superblock size**: Based on type_id
3. **Calculate number of blocks/superblocks**: `total_elements / block_size`
4. **Calculate total bytes**: `num_blocks * bytes_per_block`
5. **Seek to offset**: Use the `offset` from TensorInfo
6. **Read sequentially**: Process each block/superblock in order

## Important Notes

- All multi-byte values are **little-endian**
- **Bit packing**:
  - 2-bit values: 4 values per byte
  - 3-bit values: Packed (10 values per ~4 bytes, exact packing may vary)
  - 4-bit values: Two 4-bit values per byte
    - Lower 4 bits = first weight
    - Upper 4 bits = second weight
  - 5-bit values: Packed (8 values per 5 bytes = 40 bits)
  - 6-bit values: Packed (4 values per 3 bytes = 24 bits, or similar)
  - 8-bit values: One value per byte (no packing)
- **f16 values**: Stored as 2-byte half-precision floats (IEEE 754 binary16)
- **bf16 values**: Stored as 2-byte Brain Float 16 format

## Format Summary by Type ID

| Type ID | Format | Block/Superblock Size | Bytes per Block/Superblock |
|---------|--------|----------------------|---------------------------|
| 0 | F32 | N/A | 4 bytes per weight |
| 1 | F16 | N/A | 2 bytes per weight |
| 2 | Q4_0 | 32 weights | 18 bytes |
| 3 | Q4_1 | 32 weights | 20 bytes |
| 6 | Q5_0 | 32 weights | 22 bytes |
| 7 | Q5_1 | 32 weights | 24 bytes |
| 8 | Q8_0 | 32 weights | 34 bytes |
| 9 | Q8_1 | 32 weights | 36 bytes |
| 10 | Q2_K | 256 weights (superblock) | ~144 bytes |
| 11 | Q3_K | 256 weights (superblock) | ~152 bytes |
| 12 | Q4_K | 256 weights (superblock) | 160 bytes |
| 13 | Q5_K | 256 weights (superblock) | ~176 bytes |
| 14 | Q6_K | 256 weights (superblock) | 208 bytes |
| 15 | Q8_K | 256 weights (superblock) | 272 bytes |
| 30 | BF16 | N/A | 2 bytes per weight |

