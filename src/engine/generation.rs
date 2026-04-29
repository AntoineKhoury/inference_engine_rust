use crate::EngineError;
use crate::engine::sampling::sample_greedy;
use crate::engine::session::InferenceSession;
use crate::engine::state::ForwardState;

/// Choose the next token greedily from the session's last-token logits.
///
/// This is intentionally one-step policy only. Callers still own loop behavior,
/// stop criteria, streaming, and text postprocessing.
pub fn greedy_next_token(
    session: &InferenceSession<'_>,
    state: &ForwardState,
) -> Result<u32, EngineError> {
    let logits = session.logits_last_token(state)?;
    sample_greedy(&logits).map_err(EngineError::from)
}
