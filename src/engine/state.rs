use crate::EngineError;

/// Activation carrier for both prefill (seq > 1) and decode (seq == 1).
///
/// Buffers use a contiguous [seq, hidden] layout.
#[derive(Debug)]
pub struct ForwardState {
    seq_len: usize,
    hidden_dim: usize,
    hidden: Vec<f32>,
    per_layer_packed: Vec<f32>,
    ple_n_layers: usize,
    ple_dim: usize,
}

impl ForwardState {
    pub fn from_embeddings(
        embeddings: Vec<Vec<f32>>,
        hidden_dim: usize,
    ) -> Result<Self, EngineError> {
        Self::from_embeddings_inner(embeddings, hidden_dim, Vec::new(), 0, 0)
    }

    /// Internal constructor used by [`crate::embed`] to include PLE payload.
    pub(crate) fn from_embeddings_inner(
        embeddings: Vec<Vec<f32>>,
        hidden_dim: usize,
        per_layer_packed: Vec<f32>,
        ple_n_layers: usize,
        ple_dim: usize,
    ) -> Result<Self, EngineError> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return Err(EngineError::Model("ForwardState: empty embeddings".into()));
        }

        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for (idx, row) in embeddings.into_iter().enumerate() {
            if row.len() != hidden_dim {
                return Err(EngineError::Model(format!(
                    "ForwardState: embedding {idx} has dim {}, expected {hidden_dim}",
                    row.len(),
                )));
            }
            hidden.extend_from_slice(&row);
        }

        let stride = ple_n_layers.saturating_mul(ple_dim);
        if stride > 0 && per_layer_packed.len() != seq_len * stride {
            return Err(EngineError::Model(format!(
                "ForwardState: per_layer_packed len {} != seq_len * ple_stride {}",
                per_layer_packed.len(),
                seq_len * stride
            )));
        }

        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
            per_layer_packed,
            ple_n_layers,
            ple_dim,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    pub fn hidden_mut(&mut self) -> &mut [f32] {
        &mut self.hidden
    }

    pub fn per_layer_packed(&self) -> &[f32] {
        &self.per_layer_packed
    }

    pub fn ple_dim(&self) -> usize {
        self.ple_dim
    }

    pub fn ple_n_layers(&self) -> usize {
        self.ple_n_layers
    }

    /// Return a new state with replaced hidden activations, preserving the Gemma 4 PLE payload.
    pub fn replace_hidden(&self, hidden: Vec<f32>) -> Result<Self, EngineError> {
        Self::from_flat_with_ple(
            hidden,
            self.seq_len,
            self.hidden_dim,
            self.per_layer_packed.clone(),
            self.ple_n_layers,
            self.ple_dim,
        )
    }

    pub fn from_flat(
        hidden: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Result<Self, EngineError> {
        Self::from_flat_with_ple(hidden, seq_len, hidden_dim, Vec::new(), 0, 0)
    }

    pub fn from_flat_with_ple(
        hidden: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
        per_layer_packed: Vec<f32>,
        ple_n_layers: usize,
        ple_dim: usize,
    ) -> Result<Self, EngineError> {
        if seq_len == 0 {
            return Err(EngineError::Model(
                "ForwardState: seq_len must be > 0".into(),
            ));
        }
        let expected_len = seq_len.checked_mul(hidden_dim).ok_or_else(|| {
            EngineError::Model("ForwardState: seq_len * hidden_dim overflow".into())
        })?;
        if hidden.len() != expected_len {
            return Err(EngineError::Model(format!(
                "ForwardState: hidden len {} != expected {}",
                hidden.len(),
                expected_len
            )));
        }
        let stride = ple_n_layers.saturating_mul(ple_dim);
        if stride > 0 && per_layer_packed.len() != seq_len * stride {
            return Err(EngineError::Model(format!(
                "ForwardState: per_layer_packed len {} != expected {}",
                per_layer_packed.len(),
                seq_len * stride
            )));
        }
        Ok(Self {
            seq_len,
            hidden_dim,
            hidden,
            per_layer_packed,
            ple_n_layers,
            ple_dim,
        })
    }

    pub fn row(&self, idx: usize) -> Result<&[f32], EngineError> {
        if idx >= self.seq_len {
            return Err(EngineError::Model(
                "ForwardState: row index out of bounds".into(),
            ));
        }
        let start = idx * self.hidden_dim;
        Ok(&self.hidden[start..start + self.hidden_dim])
    }

    pub fn row_mut(&mut self, idx: usize) -> Result<&mut [f32], EngineError> {
        if idx >= self.seq_len {
            return Err(EngineError::Model(
                "ForwardState: row index out of bounds".into(),
            ));
        }
        let start = idx * self.hidden_dim;
        Ok(&mut self.hidden[start..start + self.hidden_dim])
    }
}
