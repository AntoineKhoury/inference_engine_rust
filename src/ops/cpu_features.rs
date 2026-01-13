/// CPU feature detection for SIMD capabilities
/// Detects available instruction sets at runtime to enable optimized kernels
/// 
/// Architecture Support:
/// - ARM64 (aarch64): Mac M1, Raspberry Pi 5 - NEON is standard
/// - ARMv7: Older Raspberry Pi - NEON may be optional
/// 
/// This module uses Rust's built-in CPU feature detection macros which are
/// compile-time gated but runtime-checked, ensuring we only call intrinsics
/// on architectures that support them.

#[cfg(target_arch = "aarch64")]
use std::arch::is_aarch64_feature_detected;

#[cfg(target_arch = "arm")]
use std::arch::is_arm_feature_detected;

/// Detected CPU capabilities for SIMD operations
/// This struct is populated at startup and used for kernel dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// ARM NEON (Advanced SIMD) support
    /// NEON provides 128-bit SIMD registers with operations on:
    /// - 16×u8, 8×u16, 4×u32, 4×f32, 2×f64
    /// Required for efficient quantized matmul operations
    pub neon: bool,
    
    /// ARMv8.2+ Dot Product instructions (optional)
    /// Provides specialized instructions for integer dot products
    /// Useful for quantized operations but not strictly required
    pub dotprod: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime
    /// This should be called once at startup and the result cached
    /// 
    /// # Safety
    /// This function is safe - it only queries CPU capabilities, it doesn't
    /// execute any SIMD instructions that might not be supported.
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            // On ARM64 (aarch64), NEON is mandatory per the architecture spec
            // However, we still check for it to be defensive
            let neon = is_aarch64_feature_detected!("neon");
            let dotprod = is_aarch64_feature_detected!("dotprod");
            
            Self { neon, dotprod }
        }
        
        #[cfg(target_arch = "arm")]
        {
            // On ARMv7, NEON is optional (some chips don't have it)
            let neon = is_arm_feature_detected!("neon");
            let dotprod = false; // Dot product requires ARMv8.2+
            
            Self { neon, dotprod }
        }
        
        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
        {
            // Fallback for non-ARM architectures (x86_64, etc.)
            // This would require different SIMD implementations (SSE, AVX)
            Self {
                neon: false,
                dotprod: false,
            }
        }
    }
    
    /// Check if SIMD-optimized kernels can be used
    /// Returns true if at least NEON is available
    pub fn has_simd(&self) -> bool {
        self.neon
    }
    
    /// Get a human-readable description of detected features
    pub fn describe(&self) -> String {
        let mut features = Vec::new();
        
        if self.neon {
            features.push("NEON");
        }
        if self.dotprod {
            features.push("DOTPROD");
        }
        
        if features.is_empty() {
            "None (scalar fallback)".to_string()
        } else {
            features.join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        // On ARM systems, NEON should typically be available
        // This test verifies the detection doesn't panic
        let _ = features.describe();
    }
    
    #[test]
    fn test_cpu_features_describe() {
        let features = CpuFeatures {
            neon: true,
            dotprod: false,
        };
        let desc = features.describe();
        assert!(desc.contains("NEON"));
        
        let features = CpuFeatures {
            neon: false,
            dotprod: false,
        };
        let desc = features.describe();
        assert!(desc.contains("None"));
    }
}

