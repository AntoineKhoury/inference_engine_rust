use crate::EngineError;
use crate::layers::attention::{KVCache, kv_caches_for_config};
use crate::loaded_model::LoadedModel;
use crate::prefill::{
    PrefillState, prefill_from_tokens_loaded, prefill_state_for_single_token_loaded,
};
use crate::runtime::{decode_forward, final_logits_last_token, prefill_forward};

/// Mutable inference state for one generation run.
///
/// A session owns KV caches. The model owns immutable tensor storage and metadata.
pub struct InferenceSession<'a> {
    model: &'a LoadedModel,
    kv_caches: Vec<KVCache>,
}

impl<'a> InferenceSession<'a> {
    pub fn new(model: &'a LoadedModel) -> Self {
        Self {
            model,
            kv_caches: kv_caches_for_config(model.config()),
        }
    }

    pub fn reset(&mut self) {
        self.kv_caches = kv_caches_for_config(self.model.config());
    }

    pub fn prefill(&mut self, token_ids: &[u32]) -> Result<PrefillState, EngineError> {
        let input = prefill_from_tokens_loaded(self.model.gguf(), self.model.config(), token_ids)?;
        let weights = self.model.weights()?;
        prefill_forward(
            &input,
            self.model.config(),
            &weights,
            self.kv_caches.as_mut_slice(),
        )
    }

    pub fn decode_token(&mut self, token_id: u32) -> Result<PrefillState, EngineError> {
        let input = prefill_state_for_single_token_loaded(
            self.model.gguf(),
            self.model.config(),
            token_id,
        )?;
        let weights = self.model.weights()?;
        decode_forward(
            &input,
            self.model.config(),
            &weights,
            self.kv_caches.as_mut_slice(),
        )
    }

    pub fn logits_last_token(&self, state: &PrefillState) -> Result<Vec<f32>, EngineError> {
        let weights = self.model.weights()?;
        final_logits_last_token(state, self.model.config(), &weights)
    }
}
