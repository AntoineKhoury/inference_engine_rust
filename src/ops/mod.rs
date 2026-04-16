// Core compute kernels (performance-critical, may need SIMD)
pub mod matmul;

// Activation functions (element-wise transforms)
pub mod swiglu;
pub mod softmax;
pub mod gelu;

// Normalization operations
pub mod rmsnorm;

// Utility functions
pub mod residual_add;
pub mod cpu_features;

// Model specific functions
pub mod rope;

// Quantization helpers
pub mod quant;