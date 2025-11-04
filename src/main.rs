use env_logger;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path= "model/mistral-7b-v0.1.Q4_K_M.gguf";
}
