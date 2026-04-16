use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, Seek};

use crate::EngineError;
use crate::model_loader::gguf_types::{Data, DataType, ReadingInfo, TensorInfo};
use crate::model_loader::reader::Reader;

pub fn get_tensors_metadata<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    tensor_count: u64,
) -> Result<Vec<TensorInfo>, EngineError> {
    let mut all_tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);
    let mut unique_types: HashSet<u32> = HashSet::new();
    for _ in 0..tensor_count{
        let curr_tensor: TensorInfo = get_tensor_metadata(reader)?;
        if !unique_types.contains(&curr_tensor.type_id){
            unique_types.insert(curr_tensor.type_id);
        }
        all_tensors.push(curr_tensor);
    }
    log::debug!("GGUF unique tensor type_ids: {unique_types:?}");
    Ok(all_tensors)
}

pub fn get_tensor_metadata<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<TensorInfo, EngineError> {
    let name = reader.read_string()?;
    let n_dimensions = reader.read_u32()? as usize;
    let mut dimensions = Vec::with_capacity(n_dimensions);
    for _ in 0..n_dimensions{
        dimensions.push(reader.read_u64()? as usize);
    }
    let type_id = reader.read_u32()?;
    let offset = reader.read_u64()? as usize;
    Ok(TensorInfo { name, n_dimensions, dimensions, type_id, offset })
}

pub fn get_kv_metadata<R: BufRead + Seek>(
    reader: &mut Reader<R>,
    kv_count: u64,
) -> Result<BTreeMap<String, Data>, EngineError> {
    let mut kv: std::collections::BTreeMap<String, Data> = std::collections::BTreeMap::new();

    for _i in 0..kv_count {
        let (key, val) = get_kv_pair(reader)?;
        kv.insert(key, val);
    }
    Ok(kv)
}

pub fn get_kv_pair<R: BufRead + Seek>(
    reader: &mut Reader<R>,
) -> Result<(String, Data), EngineError> {
    // Read bits of the key, then read bits of the Data with the type, so that you can read properly the coming type
    let key = get_k(reader)?;
    let value_type = get_value_type(reader)?;
    let mut reading_info = ReadingInfo {
        data_type: value_type,
    };
    let value: Data = reading_info.read_bytes_as(reader)?;
    Ok((key, value))
}

pub fn get_k<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<String, EngineError> {
    let key_len = reader.read_u64()?;
    let key_as_bytes = reader.read_bytes(key_len)?;
    let key = String::from_utf8(key_as_bytes)?;
    Ok(key)
}

pub fn get_value_type<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<DataType, EngineError> {
    // Value type is stored as 4 bytes
    let value_type_bytes = reader.read_bytes(4)?;
    let code = u32::from_le_bytes(
        value_type_bytes
            .try_into()
            .map_err(|v: Vec<u8>| EngineError::Gguf(format!("value type: expected 4 bytes, got {}", v.len())))?,
    );
    let value_type: DataType = match code {
        0 => DataType::Uint8,
        1 => DataType::Int8,
        2 => DataType::Uint16,
        3 => DataType::Int16,
        4 => DataType::Uint32,
        5 => DataType::Int32,
        6 => DataType::Float32,
        7 => DataType::Bool,
        8 => DataType::String,
        9 => DataType::Array,
        10 => DataType::Uint64,
        11 => DataType::Int64,
        12 => DataType::Float64,
        _ => return Err(EngineError::Gguf(format!("unknown value type code {code}"))),
    };
    Ok(value_type)
}

pub trait ReadBytesAsType {
    fn read_bytes_as<R: BufRead + Seek>(
        &mut self,
        reader: &mut Reader<R>,
    ) -> Result<Data, EngineError>;
}

impl ReadBytesAsType for ReadingInfo {
    fn read_bytes_as<R: BufRead + Seek>(
        &mut self,
        reader: &mut Reader<R>,
    ) -> Result<Data, EngineError> {
        let data = match self.data_type {
            DataType::Uint8  => Data::Uint8(reader.read_u8()?),
            DataType::Int8   => Data::Int8(reader.read_i8()?),
            DataType::Uint16 => Data::Uint16(reader.read_u16()?),
            DataType::Int16  => Data::Int16(reader.read_i16()?),
            DataType::Uint32 => Data::Uint32(reader.read_u32()?),
            DataType::Int32  => Data::Int32(reader.read_i32()?),
            DataType::Float32=> Data::Float32(reader.read_f32()?),
            DataType::Bool   => Data::Bool(reader.read_bool()?),
            DataType::Uint64 => Data::Uint64(reader.read_u64()?),
            DataType::Int64  => Data::Int64(reader.read_i64()?),
            DataType::Float64=> Data::Float64(reader.read_f64()?),
            DataType::String => Data::String(reader.read_string()?),
            DataType::Array => Data::Array(reader.read_array()?),
        };
        Ok(data)
    }
}

pub fn u32_to_data_type(value: u32) -> Result<DataType, EngineError> {
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
        _ => Err(EngineError::Gguf(format!("unknown value type code {value}"))),
    }
}