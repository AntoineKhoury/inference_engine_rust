

// use std::error::Error;
use super::structure::Config;
use std::path::Path;
use std::ffi::OsStr;

impl Config 
{
    pub fn load_file(path : &str) -> Result<(), & 'static str>{
        if let Some(ext) = get_extension(path){
            if ext != "gguf"{
                return Err("Error: file extension is not .gguf");
            }
        }
        else{
                return Err("Error: file extension is missing.")
        }
        // check header
        // strat parsing meta data
        // parse corps du fichier

        Ok(())
    }
}

fn get_extension(str: &str) -> Option<&str> {
    Path::new(str)
        .extension()
        .and_then(OsStr::to_str)
}





#[cfg(test)]
mod tests {
    use super::*;

    fn assert_load(path: &str, should_panic: bool) {
        let result = Config::load_file(path);
        if should_panic {
            if let Err(str) = result{
                println!("{}", str)
            }
            assert!(result.is_err(), "Expected error for path: {}", path);
        } else {
            if let Err(str) = result{
                println!("{}", str)
            }
            assert!(result.is_ok(), "Expected success for path: {}", path);
        }
    }

    #[test]
    #[ignore]
    fn test_load_files() {
        let test_cases = vec![
            ("./model/test", true),
            ("./model/test.gguff", true),
            ("./model/test.gguf", false),
            ("./model/test.txt", true),
            ("./model/missing", true),
        ];

        for (path, should_panic) in test_cases {
            assert_load(path, should_panic);
        }
    }
}

