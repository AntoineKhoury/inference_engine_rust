use gguf_rs::get_gguf_container;
use log::{info, error};

pub fn read_gguf(path: &str) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Container instantiated.");
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

    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    for (i ,(key,value)) in model.metadata().iter().enumerate() {
        println!("{}: {}",key,value);
    }

    
    
    for i in 0..5{
        let curr_tensor = &model.tensors()[i];
        println!("\n Tensor {}", i);
        println!("Name: {}", curr_tensor.name);
        println!("Size: {}", curr_tensor.size);
        println!("Offset: {}", curr_tensor.offset);
        println!("Kind: {}", curr_tensor.kind);
        for j in 0..4{
            println!("Shape {}: {}", j, curr_tensor.shape[j])
        }
    }

    Ok(())
}