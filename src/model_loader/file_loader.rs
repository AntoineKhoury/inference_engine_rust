use std::{ffi::os_str::Display, fmt::Debug, fs::File, io::{BufRead, BufReader, Read, Seek, SeekFrom}}; // To use a trait, implemented by another method, you still need to import the trait in scope in rust
use std::collections::BTreeMap;

pub fn read_file(path: &str) -> Result<(), Box<dyn std::error::Error>>{
    let mut file =  File::open(path)?;

    let mut start: u64 = 0;
    let mut size: usize = 4;

    let bytes_vec = extract_bytes_from_file(&file, start, size)?;

    // Read as GGUF
    let header: String = String::from_utf8(bytes_vec)?;

    // Read version
    start = 4;
    size = 4;
    let version_bytes = extract_bytes_from_file(&file, start, size)?;
    let version_as_number = u32::from_le_bytes(version_bytes.try_into().unwrap()); // Here, unwrap consumes the Data in the Ok, and will panic if error, try_into is used to convert from a vec to an array
    println!("Version is: {}", version_as_number);

    // Read Tensor Count
    start = 8;
    size = 8;
    let tensor_count_bytes = extract_bytes_from_file(&file, start, size)?;
    let tensor_count = u64::from_le_bytes(tensor_count_bytes.try_into().unwrap());
    println!("Tensor count is: {}", tensor_count);

    // Read Metadata Count
    start = 16;
    size = 8;
    let metadata_count_bytes = extract_bytes_from_file(&file, start, size)?;
    let metadata_count = u64::from_le_bytes(metadata_count_bytes.try_into().unwrap());
    println!("Metadata count is: {}", metadata_count);

    // Read Metadata info
    let mut reader = BufReader::new(file);
    let metadata_start: u64 = 24;
    let mut kv = get_kv_metadata(&mut reader, &metadata_start, metadata_count).unwrap();
    let first_entry = kv.first_entry();
    println!("First Metadata Data is: {:?}", &first_entry);

    Ok(())
}

#[derive(Debug)]
enum Data {
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
enum DataType {
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

struct ReadingInfo{
    data_type: DataType,
    raw_bytes: Vec<u8>
}

// Implement a method to convert a vector of bytes into the correct target type
// This method will be used to convert to the correct type everytime
impl TryFrom<ReadingInfo> for Data{
    fn try_from(reading_info: ReadingInfo) -> Result<Self, Self::Error> {
        // Based on the data_type, define the right policy to read the bytes
        match reading_info.data_type{
            DataType::Uint8 =>,
            DataType::Int8 =>,
            DataType::Uint16 => ,
            DataType::Int16 => ,
            DataType::Uint32 => ,
            DataType::Int32 => ,
            DataType::Float32 =>,
            DataType::Bool =>,
            DataType::String =>,
            DataType::Array => ,
            DataType::Uint64 =>,
            DataType::Int64 =>,
            DataType::Float64 =>,
            _ => panic!("Data type not available yet.")
        }
    }
}


pub fn get_kv_metadata<R: BufRead + Seek>(reader: &mut R, pos: &u64, kv_count: u64) -> Result<BTreeMap<String, Data>, Box<dyn std::error::Error>>{
    let mut kv:std::collections::BTreeMap<String, Data> = std::collections::BTreeMap::new();
    let start_bit = 24;

    for i in 0..1{
        let (key, val) = get_kv_pair(reader,pos).expect("Couldnt get the key val pair");
        println!("Key val pair is: {:?}", (&key,&val));
        kv.insert(key, val);
    }
    Ok(kv)
}

pub fn get_kv_pair<R: BufRead + Seek>(reader: &mut R, pos: &u64) -> Result<(String, Data), Box<dyn std::error::Error>>{
    // Read bits ok the key, then read bits of the Data with the type, so that you can read properly the coming type
    let (key,next_start) = get_k(reader, pos).expect("Couldn't get the k value");
    let value_type = get_value_type(reader, &next_start).expect("Couldnt get the value type");
    println!("Value type is: {:?}", value_type);
    Ok((key,Data::Bool(false)))
}

pub fn get_k<R: BufRead + Seek>(reader: &mut R, pos: &u64) -> Result<(String,u64), Box< dyn std::error::Error>>{
    let key_len_bytes = extract_bytes_from_reader(reader, pos, 8)?;
    println!("Key len bytes: {:?}", key_len_bytes);
    let key_len = u64::from_le_bytes(key_len_bytes.try_into().expect("Couldnt read key length"));
    let key_pos = pos + 8;
    let key_as_bytes = extract_bytes_from_reader(reader, &key_pos, key_len.try_into().expect("Couldnt convert vec of bytes into array"))?;
    let key = String::from_utf8(key_as_bytes)?;
    let next_position = key_pos + key_len;
    Ok((key,next_position))
}

pub fn get_value_type<R: BufRead + Seek>(reader: &mut R, pos: &u64) -> Result<DataType, Box<dyn std::error::Error>>{
    let value_type_bytes = extract_bytes_from_reader(reader, pos, 4).expect("Couldnt get value_type bytes.");
    let value_type: DataType = match u32::from_le_bytes(value_type_bytes.try_into().unwrap()){
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

/*
pub fn get_value<R: BufRead + Seek>(reader: &mut R, pos: u64) -> Result<Data, Box< dyn std::error::Error>>{
    Ok(())
} */


pub fn extract_bytes_from_reader<R: BufRead + Seek>(reader: &mut R, start_pos: &u64, size: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>>{
    let mut vec = vec![0u8; size];
    reader.seek(SeekFrom::Start(*start_pos))?;
    reader.read_exact(&mut vec);
    Ok(vec)
}

pub fn extract_bytes_from_file(file: &File, start_pos: u64, size: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>>{
    let mut reader = BufReader::new(file);
    let mut vec = vec![0u8; size];
    reader.seek(SeekFrom::Start(start_pos))?;
    reader.read(&mut vec);
    Ok(vec)
}


#[cfg(test)]
mod test{
    use super::*; // This is used to have access to functions outside the module
    #[test]
    fn test_file_read(){ // Test functions can't have parameters
        let result = read_file("./model/qwen2.5-7b-instruct-q4_0.gguf");
    }
}