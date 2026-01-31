use std::{collections::{BTreeMap, HashMap}, sync::Arc};

/// Tensor type identifier - public for zero-overhead kernel dispatch
/// Used by inference kernels to select the appropriate SIMD operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    /// Unquantized float32 tensors (used for layer normalization weights)
    F32,
    /// 4-bit quantization, unpacked to u8 (values 0-15)
    Q4K,
    /// 6-bit quantization, unpacked to u8 (values 0-63)
    Q6K,
}

/// Loaded tensor with all data unpacked and ready for SIMD operations
/// Immutable by design - all fields except tensor_type are private
#[derive(Debug)]
pub struct Tensor {
    pub dtype: TensorType,
    pub buffer: Arc<Vec<u8>>,
    pub dimensions: Vec<u32>,
    pub stride: Vec<u32>,
    pub offset: u32
}

impl Tensor {
    /// Create a new Tensor (constructor for tensor_loader module)
    pub(crate) fn new(
        dtype: TensorType,
        buffer: Arc<Vec<u8>>,
        dimensions: Vec<u32>,
        stride: Vec<u32>,
    ) -> Self {
        Self {
            dtype: dtype,
            buffer: buffer,
            dimensions: dimensions,
            stride: stride,
            offset: 0
        }
    }
    /// Get tensor dimensions
    pub fn dimensions(&self) -> &[u32] {
        &self.dimensions
    }

    pub fn block_iterator(&self) -> QkBlockIter<'_> {
        let num_elements: u64 = self.dimensions.iter().fold(1u64, |acc, &d| acc.saturating_mul(d as u64));
        let total_blocks = match self.dtype {
            TensorType::Q4K | TensorType::Q6K => ((num_elements + 255) / 256) as usize,
            _ => 0,
        };
    
        QkBlockIter {
            buffer: &self.buffer,
            dtype: self.dtype,
            block_index: 0,
            total_blocks,
        }
    }
}

impl<'a> Iterator for QkBlockIter<'a> {
    type Item = QkBlockRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.block_index >= self.total_blocks {
            return None;
        }

        let (block_size, qs_len, qbits) = match self.dtype {
            TensorType::Q4K => (144usize, 128usize, 4u8),
            TensorType::Q6K => (208usize, 192usize, 6u8),
            _ => return None,
        };

        let block_start = self.block_index * block_size;
        let block_end = block_start + block_size;
        if block_end > self.buffer.len() {
            return None;
        }

        let block = &self.buffer[block_start..block_end];

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

        let scales_bytes = &block[4..16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for sub in 0..8 {
            let (s, m) = extract_scale_min_k4(sub, scales_bytes);
            scales[sub] = s;
            mins[sub] = m;
        }

        let qs_start = 16;
        let qs_end = qs_start + qs_len;
        let qs = &block[qs_start..qs_end];

        let out = QkBlockRef {
            block_index: self.block_index,
            d,
            dmin,
            scales,
            mins,
            qs,
            qbits,
        };

        self.block_index += 1;
        Some(out)
    }
}

#[derive(Debug, Clone)]
pub struct QkBlockIter<'a> {
    buffer: &'a [u8],
    dtype: TensorType,
    block_index: usize,
    total_blocks: usize,
}

#[derive(Debug, Clone)]
pub struct QkBlockRef<'a>{
    pub block_index: usize,
    pub d: f32,
    pub dmin: f32,
    pub scales: [u8; 8],
    pub mins: [u8; 8],
    pub qs: &'a [u8],
    pub qbits: u8
}

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
        tensors_metadata: Vec<TensorInfo>
    ) -> Self {
        Self {
            version,
            nb_tensors,
            nb_key_vals,
            kv,
            tensors_metadata,
            tensors: HashMap::new(),
        }
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
        let mut reader = crate::model_loader::io::Reader::new(buf_reader, 0);
        
        let total_tensors = self.tensors_metadata.len();
        info!("Starting to load {} tensors...", total_tensors);
        
        for (idx, tensor_info) in self.tensors_metadata.iter().enumerate() {
            let progress = ((idx + 1) * 100) / total_tensors;
            info!("Loading tensor {}/{} ({}%): {} (offset: {}, type_id: {})", 
                  idx + 1, total_tensors, progress, tensor_info.name, tensor_info.offset, tensor_info.type_id);
            
            let tensor = match load_tensor(&mut reader, tensor_info) {
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
        let mut reader = crate::model_loader::io::Reader::new(buf_reader, 0);
        
        let tensor = load_tensor(&mut reader, tensor_info)?;
        self.tensors.insert(tensor_name.to_string(), tensor);
        
        Ok(())
    }
    
    /// Get the number of loaded tensors
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
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