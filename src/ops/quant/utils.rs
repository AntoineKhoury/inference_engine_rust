// Needed because rust doesnt have a built in f16 type
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;

    if exponent == 0 {
        if mantissa == 0 {
            if sign == 0 { 0.0 } else { -0.0 }
        } else {
            // Subnormal
            let value = (mantissa as f32) / 1024.0 * 2.0_f32.powi(-14);
            if sign == 0 { value } else { -value }
        }
    } else if exponent == 0x1F {
        if mantissa == 0 {
            if sign == 0 { f32::INFINITY } else { f32::NEG_INFINITY }
        } else {
            f32::NAN
        }
    } else {
        // Normalized
        let exp = (exponent as i32) - 15;
        let mant = 1.0 + (mantissa as f32) / 1024.0;
        let value = mant * 2.0_f32.powi(exp);
        if sign == 0 { value } else { -value }
    }
}
