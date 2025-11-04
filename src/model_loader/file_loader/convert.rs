use super::types::{Data, DataType};
use super::io::Reader;
use std::io::{BufRead, Seek};

// Helper function to convert u32 to DataType
pub fn u32_to_data_type(value: u32) -> Result<DataType, Box<dyn std::error::Error>> {
    match value {
        0 => Ok(DataType::Uint8),
        1 => Ok(DataType::Int8),
        2 => Ok(DataType::Uint16),
        3 => Ok(DataType::Int16),
        4 => Ok(DataType::Uint32),
        5 => Ok(DataType::Int32),
        6 => Ok(DataType::Float32),
        7 => Ok(DataType::Bool),
        8 => Ok(DataType::String),
        9 => Ok(DataType::Array),
        10 => Ok(DataType::Uint64),
        11 => Ok(DataType::Int64),
        12 => Ok(DataType::Float64),
        _ => Err("Unknown value type code".into()),
    }
}
