use std::fs::File;
use std::io::BufReader;
use super::io::{extract_bytes_from_file, Reader};
use super::parser::*;
use super::types::GGUFData;

pub fn read_file(path: &str) -> Result<(), Box<dyn std::error::Error>>{
    let file = File::open(path)?;
    let mut reader = Reader::new(BufReader::new(file), 0);
    
    // GGUF Header is 4 bytes, so u32
    let header: String = String::from_utf8(reader.read_bytes(4).unwrap())?;

    // Read version, 4 bytes, so u32
    let version = reader.read_u32()?;
    println!("Version is: {}", version);

    // Read Tensor Count, 8 bytes long
    let tensor_count = reader.read_u64()?;
    println!("Tensor count is: {}", tensor_count);

    // Read Metadata Count, 8 bytes long
    let metadata_count = reader.read_u64()?;
    println!("Metadata count is: {}", metadata_count);

    // Read metadata tree
    let mut kv = get_kv_metadata(&mut reader, metadata_count).unwrap();

    // Read tensors
    let tensors_metadata = get_tensors_metadata(&mut reader, tensor_count)?;
    println!("Red all tensors metadata: {:?}", tensors_metadata);
    let loaded_data = GGUFData::new(
        version,
        tensor_count,
        metadata_count,
        kv,
        tensors_metadata
    );
    Ok(())
}

#[cfg(test)]
mod test{
    use super::*; // This is used to have access to functions outside the module
    #[test]
    fn test_file_read(){ // Test functions can't have parameters
        let _result = read_file("./model/qwen2.5-7b-instruct-q4_0.gguf");
    }
}