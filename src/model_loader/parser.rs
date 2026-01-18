use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, Seek};
use crate::core::types::{Data, DataType, ReadingInfo, TensorInfo};

use super::io::Reader;


pub fn get_tensors_metadata<R: BufRead + Seek>(reader: &mut Reader<R>, tensor_count: u64) -> Result<Vec<TensorInfo>, Box<dyn std::error::Error>> {
    let mut all_tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count as usize);
    let mut unique_types: HashSet<u32> = HashSet::new();
    for _ in 0..tensor_count{
        let curr_tensor: TensorInfo = get_tensor_metadata(reader)?;
        if !unique_types.contains(&curr_tensor.type_id){
            unique_types.insert(curr_tensor.type_id);
        }
        all_tensors.push(curr_tensor);
    }
    println!("Unique tensor types found: {:?}", unique_types);
    Ok(all_tensors)
}

pub fn get_tensor_metadata<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<TensorInfo, Box<dyn std::error::Error>>{
    let name = reader.read_string()?;
    let n_dimensions = reader.read_u32()?;
    let mut dimensions = Vec::with_capacity(n_dimensions as usize);
    for _ in 0..n_dimensions{
        dimensions.push(reader.read_u64()?);
    }
    let type_id = reader.read_u32()?;
    let offset = reader.read_u64()?;
    Ok(TensorInfo { name, n_dimensions, dimensions, type_id, offset })
}

pub fn get_kv_metadata<R: BufRead + Seek>(reader: &mut Reader<R>, kv_count: u64) -> Result<BTreeMap<String, Data>, Box<dyn std::error::Error>> {
    let mut kv: std::collections::BTreeMap<String, Data> = std::collections::BTreeMap::new();

    for _i in 0..kv_count {
        let (key, val) = get_kv_pair(reader)?;
        kv.insert(key, val);
    }
    Ok(kv)
}

pub fn get_kv_pair<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<(String, Data), Box<dyn std::error::Error>> {
    // Read bits of the key, then read bits of the Data with the type, so that you can read properly the coming type
    let key = get_k(reader)?;
    let value_type = get_value_type(reader)?;
    let mut reading_info = ReadingInfo {
        data_type: value_type,
    };
    let value: Data = reading_info.read_bytes_as(reader)?;
    Ok((key, value))
}

pub fn get_k<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<String, Box<dyn std::error::Error>> {
    let key_len = reader.read_u64()?;
    let key_as_bytes = reader.read_bytes(key_len.try_into().expect("Couldnt convert vec of bytes into array"))?;
    let key = String::from_utf8(key_as_bytes)?;
    Ok(key)
}

pub fn get_value_type<R: BufRead + Seek>(reader: &mut Reader<R>) -> Result<DataType, Box<dyn std::error::Error>> {
    // Value type is stored as 4 bytes
    let value_type_bytes = reader.read_bytes(4)?;
    let value_type: DataType = match u32::from_le_bytes(value_type_bytes.try_into().unwrap()) {
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
        _ => return Err("Unknown value type code".into()),
    };
    Ok(value_type)
}

pub trait ReadBytesAsType {
    fn read_bytes_as<R: BufRead + Seek>(&mut self, reader: &mut Reader<R>) -> Result<Data, Box<dyn std::error::Error>>;
}

impl ReadBytesAsType for ReadingInfo {
    fn read_bytes_as<R: BufRead + Seek>(
        &mut self,
        reader: &mut Reader<R>
    ) -> Result<Data, Box<dyn std::error::Error>> {
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
            _ => return Err("Data type not available for easy conversion.".into()),
        };
        Ok(data)
    }
}