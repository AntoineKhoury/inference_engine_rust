use std::{collections::HashSet, ops::Not};

use gguf_rs::{GGUFModel};
use log::info;


pub fn get_tokenizer_info(loaded_model: GGUFModel){
    let mut unique_kinds: HashSet<u32> = HashSet::new();
    for (_,vector) in  loaded_model.tensors().iter().enumerate() {
        if unique_kinds.contains(&vector.kind).not(){
            unique_kinds.insert(vector.kind);
        }
    }
    info!("All unique kinds:");
    for key in unique_kinds.iter(){
        info!("{}", key);
    }
    let metadata = loaded_model.metadata();

}

#[cfg(test)]
mod tests {
    use crate::model_loader::gguf_loader::loader::read_gguf;
    use super::*;

    #[test]
    fn test_read_model_info(){
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .try_init();

        let path = "./model/qwen2.5-7b-instruct-q4_0.gguf";
        let model = read_gguf(path).expect("Failed to read the model");
        let display_token_info = get_tokenizer_info(model);
    }
}