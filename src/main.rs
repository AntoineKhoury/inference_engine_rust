use env_logger;
use inference_engine_rust::model_loader::file_loader::read_file;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path = "model/mistral-7b-v0.1.Q4_K_M.gguf";
    
    println!("Reading file metadata...");
    let mut gguf_data = read_file(path).expect("Failed to read file");
    
    println!("Loading all tensors...");
    gguf_data.load_tensors(path).expect("Failed to load tensors");
    
    println!("Loaded {} tensors", gguf_data.num_tensors());
}
