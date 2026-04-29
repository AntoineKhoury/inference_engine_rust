// Core compute kernels (performance-critical, may need SIMD)
pub mod matmul;

// Activation functions (element-wise transforms)
pub mod gelu;
pub mod softmax;
pub mod swiglu;

// Normalization operations
pub mod rmsnorm;

// Utility functions
pub mod cpu_features;
pub mod residual_add;

// Model specific functions
pub mod rope;

// Quantization helpers
pub mod quant;
