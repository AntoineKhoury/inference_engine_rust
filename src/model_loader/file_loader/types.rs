use std::collections::BTreeMap;


#[derive(Debug, Clone)]
pub enum Data {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Data>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug)]
pub enum DataType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

pub struct ReadingInfo{
    pub data_type: DataType,
}

#[derive(Debug)]
pub struct TensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub type_id: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GGUFData {
    version: u32,
    nb_tensors: u64,
    nb_key_vals: u64,
    kv: BTreeMap<String, Data>,
    tensors_metadata: Vec<TensorInfo>
}

impl GGUFData {
    pub fn new(
        version: u32,
        nb_tensors: u64,
        nb_key_vals: u64,
        kv: BTreeMap<String, Data>,
        tensors_metadata: Vec<TensorInfo>
    ) -> Self {
        Self {
            version,
            nb_tensors,
            nb_key_vals,
            kv,
            tensors_metadata,
        }
    }
}