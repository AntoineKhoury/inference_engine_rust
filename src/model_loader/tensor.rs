use crate::EngineError;
use crate::core::tensor::TensorType;

#[allow(non_camel_case_types)]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    COUNT = 40,
}

impl TryFrom<u32> for GgmlType {
    type Error = EngineError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GgmlType::F32),
            1 => Ok(GgmlType::F16),
            2 => Ok(GgmlType::Q4_0),
            3 => Ok(GgmlType::Q4_1),
            6 => Ok(GgmlType::Q5_0),
            7 => Ok(GgmlType::Q5_1),
            8 => Ok(GgmlType::Q8_0),
            9 => Ok(GgmlType::Q8_1),
            10 => Ok(GgmlType::Q2_K),
            11 => Ok(GgmlType::Q3_K),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            15 => Ok(GgmlType::Q8_K),
            16 => Ok(GgmlType::IQ2_XXS),
            17 => Ok(GgmlType::IQ2_XS),
            18 => Ok(GgmlType::IQ3_XXS),
            19 => Ok(GgmlType::IQ1_S),
            20 => Ok(GgmlType::IQ4_NL),
            21 => Ok(GgmlType::IQ3_S),
            22 => Ok(GgmlType::IQ2_S),
            23 => Ok(GgmlType::IQ4_XS),
            24 => Ok(GgmlType::I8),
            25 => Ok(GgmlType::I16),
            26 => Ok(GgmlType::I32),
            27 => Ok(GgmlType::I64),
            28 => Ok(GgmlType::F64),
            29 => Ok(GgmlType::IQ1_M),
            30 => Ok(GgmlType::BF16),
            34 => Ok(GgmlType::TQ1_0),
            35 => Ok(GgmlType::TQ2_0),
            39 => Ok(GgmlType::MXFP4),
            40 => Ok(GgmlType::COUNT),
            _ => Err(EngineError::Gguf(format!("unknown GGML type id: {value}"))),
        }
    }
}

impl GgmlType {
    pub fn to_tensor_type(self) -> Result<TensorType, EngineError> {
        match self {
            GgmlType::F32 => Ok(TensorType::F32),
            GgmlType::Q4_K => Ok(TensorType::Q4K),
            GgmlType::Q6_K => Ok(TensorType::Q6K),
            GgmlType::Q8_0 => Ok(TensorType::Q8_0),
            _ => Err(EngineError::Tensor(format!(
                "unsupported GGML type for inference: {self:?}"
            ))),
        }
    }
}
