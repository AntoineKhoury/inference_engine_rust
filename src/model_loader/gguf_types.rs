use std::collections::{BTreeMap, HashMap};
use crate::core::tensor::Tensor;

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

// Struct to define the metadata from the GGUF file, describing where all the tensor information is
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dimensions: usize,
    pub dimensions: Vec<usize>,
    pub type_id: u32,
    pub offset: usize,
}

#[derive(Debug)]
pub struct GGUFData {
    version: u32,
    nb_tensors: u64,
    nb_key_vals: u64,
    kv: BTreeMap<String, Data>,
    /// Absolute file offset of the GGUF tensor data blob (tensor offsets are relative to this).
    tensor_data_offset: u64,
    /// Tensor metadata (offsets, type_ids) - used during loading process
    tensors_metadata: Vec<TensorInfo>,
    /// Loaded tensors: HashMap keyed by tensor name
    /// Populated during tensor loading phase
    tensors: HashMap<String, Tensor>,
}

impl GGUFData {
    pub fn new(
        version: u32,
        nb_tensors: u64,
        nb_key_vals: u64,
        kv: BTreeMap<String, Data>,
        tensors_metadata: Vec<TensorInfo>,
        tensor_data_offset: u64,
    ) -> Self {
        Self {
            version,
            nb_tensors,
            nb_key_vals,
            kv,
            tensor_data_offset,
            tensors_metadata,
            tensors: HashMap::new(),
        }
    }

    /// Byte offset in the GGUF file where tensor data begins (after metadata + tensor info table).
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }
    
    /// Load all tensors from the GGUF file
    /// Opens the file, reads tensor data based on tensors_metadata, and populates the tensors HashMap
    /// Uses a larger buffer (1MB) for better I/O performance
    pub fn load_tensors(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::model_loader::tensor_loader::load_tensor;
        use std::fs::File;
        use std::io::BufReader;
        use log::info;
        
        let file = File::open(file_path)?;
        // For random access (seeking to different tensor offsets), use File directly
        // BufReader is optimized for sequential reads and can cause buffer invalidation issues
        // when seeking frequently. We wrap it in BufReader only for the Reader abstraction.
        // Note: For truly random access, File is more appropriate, but Reader expects BufRead + Seek
        let buf_reader = BufReader::with_capacity(1024 * 1024, file);
        let mut reader = crate::model_loader::reader::Reader::new(buf_reader, 0);
        
        let total_tensors = self.tensors_metadata.len();
        info!("Starting to load {} tensors...", total_tensors);
        
        for (idx, tensor_info) in self.tensors_metadata.iter().enumerate() {
            let progress = ((idx + 1) * 100) / total_tensors;
            info!("Loading tensor {}/{} ({}%): {} (offset: {}, type_id: {})", 
                  idx + 1, total_tensors, progress, tensor_info.name, tensor_info.offset, tensor_info.type_id);
            
            let tensor = match load_tensor(&mut reader, tensor_info, self.tensor_data_offset) {
                Ok(t) => t,
                Err(e) => {
                    return Err(format!(
                        "Failed to load tensor {}/{} '{}' (offset: {}, type_id: {}): {}",
                        idx + 1, total_tensors, tensor_info.name, tensor_info.offset, tensor_info.type_id, e
                    ).into());
                }
            };
            self.tensors.insert(tensor_info.name.clone(), tensor);
        }
        
        info!("Successfully loaded all {} tensors", total_tensors);
        Ok(())
    }
    
    /// Get a tensor by name (only if already loaded)
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }
    
    /// Load a single tensor by name without loading all tensors
    /// 
    /// This is more efficient when you only need specific tensors (e.g., just embeddings).
    /// The tensor metadata must already be loaded (from `read_file()`).
    /// 
    /// # Performance
    /// - Seeks directly to the tensor's offset in the file
    /// - Only reads that one tensor's data
    /// - Much faster than loading all 291 tensors when you only need one
    pub fn load_single_tensor(&mut self, file_path: &str, tensor_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        use crate::model_loader::tensor_loader::load_tensor;
        use std::fs::File;
        use std::io::BufReader;
        
        // Find the tensor in metadata
        let tensor_info = self.tensors_metadata
            .iter()
            .find(|t| t.name == tensor_name)
            .ok_or_else(|| format!("Tensor '{}' not found in model metadata", tensor_name))?;
        
        // Check if already loaded
        if self.tensors.contains_key(tensor_name) {
            return Ok(()); // Already loaded, nothing to do
        }
        
        // Load just this one tensor
        let file = File::open(file_path)?;
        let buf_reader = BufReader::with_capacity(1024 * 1024, file);
        let mut reader = crate::model_loader::reader::Reader::new(buf_reader, 0);
        
        let tensor = load_tensor(&mut reader, tensor_info, self.tensor_data_offset)?;
        self.tensors.insert(tensor_name.to_string(), tensor);
        
        Ok(())
    }

    /// Load many tensors with **one** file open and a single `BufReader`.
    ///
    /// Prefer this over calling [`Self::load_single_tensor`] in a loop: that path opens the file
    /// once per tensor (very slow on large GGUFs). Reads are sorted by on-disk offset to reduce
    /// backward seeks when the name list does not match file order.
    pub fn load_named_tensors(
        &mut self,
        file_path: &str,
        tensor_names: &[String],
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::model_loader::tensor_loader::load_tensor;
        use std::fs::File;
        use std::io::BufReader;

        let mut indices: Vec<usize> = Vec::new();
        for name in tensor_names {
            if self.tensors.contains_key(name.as_str()) {
                continue;
            }
            let idx = self
                .tensors_metadata
                .iter()
                .position(|t| t.name == name.as_str())
                .ok_or_else(|| format!("Tensor '{}' not found in model metadata", name))?;
            indices.push(idx);
        }
        if indices.is_empty() {
            return Ok(());
        }
        indices.sort_by_key(|&i| self.tensors_metadata[i].offset);

        let file = File::open(file_path)?;
        let buf_reader = BufReader::with_capacity(1024 * 1024, file);
        let mut reader = crate::model_loader::reader::Reader::new(buf_reader, 0);

        for idx in indices {
            let name = self.tensors_metadata[idx].name.clone();
            let tensor = {
                let info = &self.tensors_metadata[idx];
                load_tensor(&mut reader, info, self.tensor_data_offset)?
            };
            self.tensors.insert(name, tensor);
        }
        Ok(())
    }
    
    /// Get the number of loaded tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Get GGUF version from the file header
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Total number of tensors listed in metadata
    pub fn total_tensors(&self) -> u64 {
        self.nb_tensors
    }

    /// Total number of key/value metadata entries
    pub fn total_key_vals(&self) -> u64 {
        self.nb_key_vals
    }
    
    /// Get tensor metadata (for testing/debugging)
    pub fn tensors_metadata(&self) -> &[TensorInfo] {
        &self.tensors_metadata
    }
    
    /// Get metadata value by key
    /// Useful for accessing tokenizer information and other model metadata
    pub fn get_metadata(&self, key: &str) -> Option<&Data> {
        self.kv.get(key)
    }
    
    /// Get all metadata keys (for debugging/inspection)
    pub fn metadata_keys(&self) -> Vec<&String> {
        self.kv.keys().collect()
    }
}