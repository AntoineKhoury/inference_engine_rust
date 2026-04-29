use crate::EngineError;
use crate::prefill::PrefillState;
use crate::sampling::sample_greedy;
use crate::session::InferenceSession;

/// Choose the next token greedily from the session's last-token logits.
///
/// This is intentionally one-step policy only. Callers still own loop behavior,
/// stop criteria, streaming, and text postprocessing.
pub fn greedy_next_token(
    session: &InferenceSession<'_>,
    state: &PrefillState,
) -> Result<u32, EngineError> {
    let logits = session.logits_last_token(state)?;
    sample_greedy(&logits).map_err(EngineError::from)
}
