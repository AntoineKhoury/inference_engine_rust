use std::{fs::File, io::{BufRead, BufReader, Read, Seek, SeekFrom}}; // To use a trait, implemented by another method, you still need to import the trait in scope in rust

use std::collections::HashMap
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
    let version_as_number = u32::from_le_bytes(version_bytes.try_into().unwrap()); // Here, unwrap consumes the value in the Ok, and will panic if error, try_into is used to convert from a vec to an array
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


    Ok(())
}

enum ValueType{
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}


pub fn get_kv_metadata<R: BufRead>(reader: &R, pos: u64, kv_count: u64) -> Result<BTreeMap<&str, ValueType>, Box<dyn std::error::Error>>{
    let mut kv:std::collections::BTreeMap<&str, ValueType> = std::collections::BTreeMap::new();

    Ok(())
}

pub fn get_kv_pair<R: BufRead>(reader: &R, pos: u64) -> Result<(&str, ValueType), Box<dyn std::error::Error>>{
    // Read bits ok the key, then read bits of the value with the type, so that you can read properly the coming type
    
    Ok(())
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