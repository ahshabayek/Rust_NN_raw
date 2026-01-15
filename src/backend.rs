//! Backend abstraction for matrix operations.
//!
//! This module provides a trait that abstracts away whether operations
//! run on CPU or GPU. This allows neural network code to be written once
//! and run on either backend.
//!
//! # Design Pattern: Strategy
//!
//! The `MatrixBackend` trait is an example of the Strategy pattern:
//! - Define a common interface (the trait)
//! - Multiple implementations (CPU, GPU)
//! - User code works with the trait, not specific implementations
//!
//! # Example
//!
//! ```ignore
//! use rust_nn_raw::backend::{MatrixBackend, CpuBackend, GpuBackend};
//!
//! fn compute<B: MatrixBackend>(backend: &B, a: &Matrix, b: &Matrix) -> Matrix {
//!     backend.add(a, b)  // Works with any backend!
//! }
//!
//! // CPU version
//! let cpu = CpuBackend;
//! let result = compute(&cpu, &a, &b);
//!
//! // GPU version
//! let gpu = GpuBackend::new();
//! let result = compute(&gpu, &a, &b);
//! ```

use crate::gpu::GpuContext;
use crate::matrix::Matrix;

/// Trait for matrix operation backends.
///
/// Implement this trait to add a new backend (e.g., OpenCL, CUDA).
/// All neural network operations should use this trait so they can
/// run on any backend.
pub trait MatrixBackend {
    /// Element-wise addition: result[i] = a[i] + b[i]
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix;

    /// Element-wise subtraction: result[i] = a[i] - b[i]
    fn subtract(&self, a: &Matrix, b: &Matrix) -> Matrix;

    /// Element-wise multiplication: result[i] = a[i] * b[i]
    fn elem_multiply(&self, a: &Matrix, b: &Matrix) -> Matrix;

    /// Matrix multiplication: C = A × B
    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix;

    /// Apply a function to each element (for activation functions)
    /// Note: This is CPU-only for now; custom GPU shaders needed for each function
    fn map(&self, m: &Matrix, f: impl Fn(f64) -> f64) -> Matrix;
}

// =============================================================================
// CPU BACKEND
// =============================================================================

/// CPU-based matrix operations.
///
/// This backend uses your existing Matrix methods. It's simple and works
/// everywhere, but doesn't leverage GPU parallelism.
///
/// # When to use
///
/// - Small matrices (< 100x100)
/// - Debugging (easier to trace)
/// - Systems without a GPU
/// - Operations not yet implemented on GPU
pub struct CpuBackend;

impl MatrixBackend for CpuBackend {
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        a.add(b)
    }

    fn subtract(&self, a: &Matrix, b: &Matrix) -> Matrix {
        a.subtract(b)
    }

    fn elem_multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        a.elem_multiply(b)
    }

    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix {
        a.matmul(b)
    }

    fn map(&self, m: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
        m.map(f)
    }
}

// =============================================================================
// GPU BACKEND
// =============================================================================

/// GPU-accelerated matrix operations using wgpu.
///
/// This backend runs operations on the GPU for massive parallelism.
/// Ideal for large matrices where the overhead of CPU-GPU transfer
/// is worth the parallel speedup.
///
/// # When to use
///
/// - Large matrices (> 1000x1000)
/// - Batch operations (many matrices at once)
/// - Training neural networks
///
/// # Current limitations
///
/// - `map` still runs on CPU (custom shader needed for each function)
/// - f32 precision (GPUs are optimized for f32, not f64)
/// - Overhead for small matrices
pub struct GpuBackend {
    context: GpuContext,
}

impl GpuBackend {
    /// Creates a new GPU backend.
    ///
    /// This initializes the GPU connection. It's relatively expensive,
    /// so create one and reuse it.
    pub fn new() -> Self {
        GpuBackend {
            context: GpuContext::new(),
        }
    }
}

impl MatrixBackend for GpuBackend {
    fn add(&self, a: &Matrix, b: &Matrix) -> Matrix {
        self.context.add(a, b)
    }

    fn subtract(&self, a: &Matrix, b: &Matrix) -> Matrix {
        // TODO: Implement GPU subtract shader
        // For now, fall back to CPU
        a.subtract(b)
    }

    fn elem_multiply(&self, a: &Matrix, b: &Matrix) -> Matrix {
        // TODO: Implement GPU multiply shader
        // For now, fall back to CPU
        a.elem_multiply(b)
    }

    fn matmul(&self, a: &Matrix, b: &Matrix) -> Matrix {
        // TODO: Implement GPU matmul shader (most important one!)
        // For now, fall back to CPU
        a.matmul(b)
    }

    fn map(&self, m: &Matrix, f: impl Fn(f64) -> f64) -> Matrix {
        // map requires a custom shader for each function
        // This is complex, so we always use CPU for now
        m.map(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_add() {
        let backend = CpuBackend;
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);

        let result = backend.add(&a, &b);

        assert_eq!(result.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_gpu_backend_add() {
        let backend = GpuBackend::new();
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from_vec(2, 2, vec![10.0, 20.0, 30.0, 40.0]);

        let result = backend.add(&a, &b);

        // Check with epsilon for f32 precision
        let expected = vec![11.0, 22.0, 33.0, 44.0];
        for (i, (&got, &exp)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.001,
                "Mismatch at {}: {} vs {}",
                i,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_backend_trait_generic() {
        // This test shows the power of the trait - same code, different backends
        fn compute_with_backend<B: MatrixBackend>(backend: &B) -> Matrix {
            let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
            let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
            backend.add(&a, &b)
        }

        let cpu_result = compute_with_backend(&CpuBackend);
        let gpu_result = compute_with_backend(&GpuBackend::new());

        // Both should give the same result
        for i in 0..4 {
            assert!(
                (cpu_result.data[i] - gpu_result.data[i]).abs() < 0.001,
                "CPU and GPU differ at index {}",
                i
            );
        }
    }
}
