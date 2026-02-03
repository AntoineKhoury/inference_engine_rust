use std::io::{BufRead, Seek, SeekFrom};

use crate::model_loader::convert::u32_to_data_type;
use crate::model_loader::gguf_types::{Data, DataType};

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
    pub fn seek(&mut self, pos: u64) -> Result<(), Box<dyn std::error::Error>> {
        let actual_pos = self.buffer.seek(SeekFrom::Start(pos))?;
        if actual_pos != pos {
            return Err(format!(
                "Seek failed: requested position {}, but got {}",
                pos, actual_pos
            )
            .into());
        }
        self.pos = pos;
        Ok(())
    }

    pub fn read_bytes(&mut self, size: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut vec = vec![0u8; size as usize];
        // Read sequentially - BufReader handles buffering automatically
        // No seek needed for sequential reads (seeking invalidates the buffer!)
        self.buffer.read_exact(&mut vec)?;
        self.pos += size;
        Ok(vec)
    }

    // Type-specific read methods
    pub fn read_u8(&mut self) -> Result<u8, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(1)?;
        Ok(u8::from_le_bytes(bytes.try_into().expect("Couldnt read u8")))
    }

    pub fn read_i8(&mut self) -> Result<i8, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(1)?;
        Ok(i8::from_le_bytes(bytes.try_into().expect("Couldnt read i8")))
    }

    pub fn read_u16(&mut self) -> Result<u16, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().expect("Couldnt read u16")))
    }

    pub fn read_i16(&mut self) -> Result<i16, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes(bytes.try_into().expect("Couldnt read i16")))
    }

    pub fn read_u32(&mut self) -> Result<u32, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("Couldnt read u32")))
    }

    pub fn read_i32(&mut self) -> Result<i32, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().expect("Couldnt read i32")))
    }

    pub fn read_f32(&mut self) -> Result<f32, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().expect("Couldnt read f32")))
    }

    pub fn read_u64(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("Couldnt read u64")))
    }

    pub fn read_i64(&mut self) -> Result<i64, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().expect("Couldnt read i64")))
    }

    pub fn read_f64(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().expect("Couldnt read f64")))
    }

    pub fn read_bool(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        let bytes = self.read_bytes(1)?;
        let b = match bytes[0] {
            0 => false,
            1 => true,
            _ => panic!("Not a boolean value!"),
        };
        Ok(b)
    }

    pub fn read_string(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let str_len_bytes = self.read_bytes(8)?;
        let str_len = u64::from_le_bytes(str_len_bytes.try_into().expect("Couldnt read str length"));
        let str_as_bytes =
            self.read_bytes(str_len.try_into().expect("Couldnt convert vec of bytes into array"))?;
        let str = String::from_utf8(str_as_bytes)?;
        Ok(str)
    }

    pub fn read_array(&mut self) -> Result<Vec<Data>, Box<dyn std::error::Error>> {
        // First, read the type stored in the array, value type is stored as 4 bytes
        let value_type_bytes = self.read_bytes(4)?;
        let value_type: DataType =
            u32_to_data_type(u32::from_le_bytes(value_type_bytes.try_into().unwrap()))?;

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
