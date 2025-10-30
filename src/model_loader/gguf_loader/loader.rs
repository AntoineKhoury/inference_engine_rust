use gguf_rs::{GGUFModel, get_gguf_container};
use log::{info, error};


pub fn read_gguf(path: &str) -> Result<GGUFModel, Box<dyn std::error::Error>> {
    let mut container = match get_gguf_container(&path){
        Ok(c) => {
            info!("Container instantiated");
            c
        },
        Err(err) => {
            error!("Failed to read the file");
            return Err(err.into());
        }
    };

    info!("Container instantiated.");
    let model = match container.decode(){
        Ok(c) => {
            info!("File decoded");
            c
        },
        Err(err) => {
            error!("Failed to decode the file");
            return Err(err.into());
        }
    };

    info!("Model Family: {}", model.model_family());
    info!("Number of Parameters: {}", model.model_parameters());
    info!("File Type: {}", model.file_type());
    info!("Number of Tensors: {}", model.num_tensor());
    for (_ ,(key,value)) in model.metadata().iter().enumerate() {
        info!("{}: {}",key,value);
    }

    
    
    for i in 0..20{
        let curr_tensor = &model.tensors()[i];
        info!("\n");
        info!(" Tensor {}", i);
        info!("Name: {}", curr_tensor.name);
        info!("Size: {}", curr_tensor.size);
        info!("Offset: {}", curr_tensor.offset);
        info!("Kind: {}", curr_tensor.kind);
        for j in 0..4{
            info!("Shape {}: {}", j, curr_tensor.shape[j])
        }
    }

    Ok(model)
}