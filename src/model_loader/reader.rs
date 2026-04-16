use std::io::{BufRead, Seek, SeekFrom};

use crate::EngineError;
use crate::model_loader::gguf_types::{Data, DataType};
use crate::model_loader::parser::u32_to_data_type;

fn le_array<const N: usize>(bytes: Vec<u8>) -> Result<[u8; N], EngineError> {
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| EngineError::Gguf(format!("expected {N} bytes, got {}", v.len())))
}

pub struct Reader<R: BufRead + Seek> {
    buffer: R,
    pos: u64,
}

impl<R: BufRead + Seek> Reader<R> {
    pub fn new(buffer: R, initial_pos: u64) -> Self {
        Reader { buffer, pos: initial_pos }
    }

    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Seek to a specific position in the file
    /// Verifies the actual position after seeking to catch buffer synchronization issues
    pub fn seek(&mut self, pos: u64) -> Result<(), EngineError> {
        let actual_pos = self.buffer.seek(SeekFrom::Start(pos))?;
        if actual_pos != pos {
            return Err(EngineError::Gguf(format!(
                "seek: requested position {pos}, got {actual_pos}"
            )));
        }
        self.pos = pos;
        Ok(())
    }

    pub fn read_bytes(&mut self, size: u64) -> Result<Vec<u8>, EngineError> {
        let mut vec = vec![0u8; size as usize];
        // Read sequentially - BufReader handles buffering automatically
        // No seek needed for sequential reads (seeking invalidates the buffer!)
        self.buffer.read_exact(&mut vec)?;
        self.pos += size;
        Ok(vec)
    }

    // Type-specific read methods
    pub fn read_u8(&mut self) -> Result<u8, EngineError> {
        let bytes = self.read_bytes(1)?;
        Ok(u8::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_i8(&mut self) -> Result<i8, EngineError> {
        let bytes = self.read_bytes(1)?;
        Ok(i8::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_u16(&mut self) -> Result<u16, EngineError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_i16(&mut self) -> Result<i16, EngineError> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_u32(&mut self) -> Result<u32, EngineError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_i32(&mut self) -> Result<i32, EngineError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_f32(&mut self) -> Result<f32, EngineError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_u64(&mut self) -> Result<u64, EngineError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_i64(&mut self) -> Result<i64, EngineError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_f64(&mut self) -> Result<f64, EngineError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(le_array(bytes)?))
    }

    pub fn read_bool(&mut self) -> Result<bool, EngineError> {
        let bytes = self.read_bytes(1)?;
        let b = match bytes[0] {
            0 => false,
            1 => true,
            v => {
                return Err(EngineError::Gguf(format!("invalid GGUF bool byte (expected 0 or 1, got {v})")));
            }
        };
        Ok(b)
    }

    pub fn read_string(&mut self) -> Result<String, EngineError> {
        let str_len_bytes = self.read_bytes(8)?;
        let str_len = u64::from_le_bytes(le_array(str_len_bytes)?);
        let str_as_bytes = self.read_bytes(str_len)?;
        let str = String::from_utf8(str_as_bytes)?;
        Ok(str)
    }

    pub fn read_array(&mut self) -> Result<Vec<Data>, EngineError> {
        // First, read the type stored in the array, value type is stored as 4 bytes
        let value_type_bytes = self.read_bytes(4)?;
        let value_type: DataType =
            u32_to_data_type(u32::from_le_bytes(le_array(value_type_bytes)?))?;

        // Once you have the type, read the array len
        // Len is u64 so 8 bytes
        let array_len = self.read_u64()?;

        let mut result: Vec<Data> = Vec::with_capacity(array_len as usize);

        for _ in 0..array_len {
            let value = match value_type {
                DataType::Uint8 => Data::Uint8(self.read_u8()?),
                DataType::Int8 => Data::Int8(self.read_i8()?),
                DataType::Uint16 => Data::Uint16(self.read_u16()?),
                DataType::Int16 => Data::Int16(self.read_i16()?),
                DataType::Uint32 => Data::Uint32(self.read_u32()?),
                DataType::Int32 => Data::Int32(self.read_i32()?),
                DataType::Float32 => Data::Float32(self.read_f32()?),
                DataType::Uint64 => Data::Uint64(self.read_u64()?),
                DataType::Int64 => Data::Int64(self.read_i64()?),
                DataType::Float64 => Data::Float64(self.read_f64()?),
                DataType::Bool => Data::Bool(self.read_bool()?),
                DataType::String => Data::String(self.read_string()?),
                DataType::Array => Data::Array(self.read_array()?),
            };
            result.push(value);
        }
        Ok(result)
    }
}
