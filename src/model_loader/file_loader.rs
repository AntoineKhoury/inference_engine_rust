use std::fs::File;
use std::io::BufReader;
use super::io::{extract_bytes_from_file, Reader};
use super::parser::*;
use crate::core::types::GGUFData;

/// Read GGUF file metadata and return GGUFData structure
/// Note: This only reads metadata, not tensor data. Call load_tensors() to load actual tensor weights.
pub fn read_file(path: &str) -> Result<GGUFData, Box<dyn std::error::Error>>{
    let file = File::open(path)?;
    let mut reader = Reader::new(BufReader::new(file), 0);
    
    // GGUF Header is 4 bytes, so u32
    let _header: String = String::from_utf8(reader.read_bytes(4).unwrap())?;

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
    let kv = get_kv_metadata(&mut reader, metadata_count).unwrap();
    //println!("Metadata: {:?}", kv);

    // Read tensors metadata
    let tensors_metadata = get_tensors_metadata(&mut reader, tensor_count)?;
    println!("Read all tensors metadata: {} tensors", tensors_metadata.len());
    
    let loaded_data = GGUFData::new(
        version,
        tensor_count,
        metadata_count,
        kv,
        tensors_metadata
    );
    Ok(loaded_data)
}

#[cfg(test)]
mod test{
    use super::*; // This is used to have access to functions outside the module
    
    #[test]
    fn test_file_read_metadata(){ 
        // Test reading metadata only
        let result = read_file("./model/mistral-7b-v0.1.Q4_K_M.gguf");
        assert!(result.is_ok(), "Failed to read file: {:?}", result.err());
    }
    
    #[test]
    fn test_load_tensors(){
        // Test loading actual tensor data
        let mut gguf_data = read_file("./model/mistral-7b-v0.1.Q4_K_M.gguf").unwrap();
        
        // Initially, no tensors should be loaded
        assert_eq!(gguf_data.num_tensors(), 0);
        
        // Load all tensors
        let load_result = gguf_data.load_tensors("./model/mistral-7b-v0.1.Q4_K_M.gguf");
        assert!(load_result.is_ok(), "Failed to load tensors: {:?}", load_result.err());
        
        // Verify tensors were loaded
        assert!(gguf_data.num_tensors() > 0, "No tensors were loaded");
        
        // Verify we can access a tensor by name
        // Check for a known tensor (e.g., first norm weight which is F32 and small)
        use crate::core::types::TensorType;
        if let Some(tensor) = gguf_data.get_tensor("blk.0.attn_norm.weight") {
            assert_eq!(tensor.tensor_type, TensorType::F32);
        }
    }
    
    #[test]
    fn test_load_single_tensor() {
        // Test loading just one small tensor to verify the loading logic works
        let mut gguf_data = read_file("./model/mistral-7b-v0.1.Q4_K_M.gguf").unwrap();
        
        // Find a small F32 tensor (norm weights are small - 4096 elements)
        let tensor_info = gguf_data.tensors_metadata().iter()
            .find(|t| t.name == "blk.0.attn_norm.weight" && t.type_id == 0)
            .expect("Should have blk.0.attn_norm.weight tensor");
        
        // Load just this one tensor manually
        use crate::model_loader::tensor_loader::load_tensor;
        use std::fs::File;
        use std::io::BufReader;
        
        let file = File::open("./model/mistral-7b-v0.1.Q4_K_M.gguf").unwrap();
        let buf_reader = BufReader::with_capacity(1024 * 1024, file);
        let mut reader = crate::model_loader::io::Reader::new(buf_reader, 0);
        
        let tensor = load_tensor(&mut reader, tensor_info).unwrap();
        
        // Verify it's the right type and has data
        use crate::core::types::TensorType;
        assert_eq!(tensor.tensor_type, TensorType::F32);
        assert_eq!(tensor.dimensions(), &[4096]);
    }
}