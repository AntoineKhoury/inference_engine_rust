use std::path::Path;

use crate::EngineError;
use crate::model_config::{ModelConfig, TokenizerPromptConfig};
use crate::model_loader::file_loader::read_file;
use crate::model_loader::gguf_types::GGUFData;
use crate::model_weights::{ModelWeightNames, ModelWeights};

/// Fully loaded model storage plus metadata.
///
/// The tensor bytes live in `gguf`. [`ModelWeights`] is intentionally returned as a
/// borrowed view instead of being stored here, because it borrows tensors owned by
/// this struct.
pub struct LoadedModel {
    model_path: String,
    gguf: GGUFData,
    config: ModelConfig,
    names: ModelWeightNames,
    tokenizer_prompt: TokenizerPromptConfig,
}

impl LoadedModel {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, EngineError> {
        let model_path = model_path.as_ref();
        if !model_path.is_file() {
            return Err(EngineError::Model(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }

        let model_path = model_path
            .to_str()
            .ok_or_else(|| EngineError::Model("model path is not valid UTF-8".into()))?
            .to_string();

        let mut gguf = read_file(model_path.as_str())?;
        let tokenizer_prompt = TokenizerPromptConfig::from_gguf(&gguf)?;
        let config = ModelConfig::from_gguf(&gguf)?;
        let names = ModelWeightNames::resolve(&gguf, &config)?;
        names.load_all(&mut gguf, model_path.as_str())?;

        Ok(Self {
            model_path,
            gguf,
            config,
            names,
            tokenizer_prompt,
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn gguf(&self) -> &GGUFData {
        &self.gguf
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn tokenizer_prompt(&self) -> &TokenizerPromptConfig {
        &self.tokenizer_prompt
    }

    pub fn weights(&self) -> Result<ModelWeights<'_>, EngineError> {
        ModelWeights::from_loaded(&self.gguf, &self.names)
    }
}
