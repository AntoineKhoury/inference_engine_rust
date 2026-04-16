use std::fs::File;
use std::io::BufReader;

use crate::EngineError;
use crate::model_loader::gguf_types::{Data, GGUFData};
use crate::model_loader::reader::Reader;

use super::parser::*;

/// After the tensor info table, GGUF pads to `general.alignment` (default 32) before tensor bytes.
fn tensor_data_section_offset(kv: &std::collections::BTreeMap<String, Data>, pos_after_tensor_info: u64) -> u64 {
    const DEFAULT_ALIGNMENT: u32 = 32;
    let align = match kv.get("general.alignment") {
        Some(Data::Uint32(a)) if *a > 0 && (*a).is_power_of_two() => *a,
        _ => DEFAULT_ALIGNMENT,
    };
    let a = u64::from(align);
    (pos_after_tensor_info + a - 1) & !(a - 1)
}

/// Read GGUF file metadata and return GGUFData structure
/// Note: This only reads metadata, not tensor data. Call load_tensors() to load actual tensor weights.
pub fn read_file(path: &str) -> Result<GGUFData, EngineError> {
    let file = File::open(path)?;
    let mut reader = Reader::new(BufReader::new(file), 0);
    
    // GGUF Header is 4 bytes, so u32
    let _header: String = String::from_utf8(reader.read_bytes(4)?)?;

    // Read version, 4 bytes, so u32
    let version = reader.read_u32()?;
    log::debug!("GGUF version: {version}");

    // Read Tensor Count, 8 bytes long
    let tensor_count = reader.read_u64()?;
    log::debug!("GGUF tensor count: {tensor_count}");

    // Read Metadata Count, 8 bytes long
    let metadata_count = reader.read_u64()?;
    log::debug!("GGUF metadata count: {metadata_count}");
    

    // Read metadata tree
    let kv = get_kv_metadata(&mut reader, metadata_count)?;
    //println!("Metadata: {:?}", kv);

    // Read tensors metadata
    let tensors_metadata = get_tensors_metadata(&mut reader, tensor_count)?;
    log::debug!("GGUF tensors metadata: {} tensors", tensors_metadata.len());

    // GGUF: tensor offsets are relative to the aligned start of the tensor data blob (see gguf.cpp).
    let tensor_data_offset = tensor_data_section_offset(&kv, reader.position());

    let loaded_data = GGUFData::new(
        version,
        tensor_count,
        metadata_count,
        kv,
        tensors_metadata,
        tensor_data_offset,
    );
    Ok(loaded_data)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    #[ignore = "requires ./model/mistral-7b-v0.1.Q4_K_M.gguf (cargo test -- --ignored)"]
    fn test_file_read_metadata() { 
        // Test reading metadata only
        let result = read_file("./model/mistral-7b-v0.1.Q4_K_M.gguf");
        assert!(result.is_ok(), "Failed to read file: {:?}", result.err());
    }
    
    #[test]
    #[ignore = "requires ./model/mistral-7b-v0.1.Q4_K_M.gguf (cargo test -- --ignored)"]
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
        use crate::core::tensor::TensorType;
        if let Some(tensor) = gguf_data.get_tensor("blk.0.attn_norm.weight") {
            assert_eq!(tensor.dtype(), TensorType::F32);
        }
    }
    
    #[test]
    #[ignore = "requires ./model/mistral-7b-v0.1.Q4_K_M.gguf (cargo test -- --ignored)"]
    fn test_load_single_tensor() {
        // Test loading just one small tensor to verify the loading logic works
        let gguf_data = read_file("./model/mistral-7b-v0.1.Q4_K_M.gguf").unwrap();
        
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
        let mut reader = crate::model_loader::reader::Reader::new(buf_reader, 0);
        
        let tensor = load_tensor(&mut reader, tensor_info, gguf_data.tensor_data_offset()).unwrap();
        
        // Verify it's the right type and has data
        use crate::core::tensor::TensorType;
        assert_eq!(tensor.dtype(), TensorType::F32);
        assert_eq!(tensor.dimensions(), &[4096usize]);
    }
}