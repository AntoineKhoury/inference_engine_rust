/// Convert an `f32` to IEEE 754 binary16 bits.
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// Convert IEEE 754 binary16 bits to `f32`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}
