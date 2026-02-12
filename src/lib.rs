//! Fast matrix multiplication in Rust, built from scratch.
//!
//! I built this to understand what makes BLAS fast. Turns out it's:
//! cache blocking, SIMD intrinsics, and FMA instructions. This crate
//! implements all of that, achieving ~62% of NumPy/OpenBLAS performance.
//!
//! ## Usage
//!
//! ```
//! use matmul::multiply;
//!
//! let a = vec![1.0f64; 256 * 256];
//! let b = vec![1.0f64; 256 * 256];
//! let mut c = vec![0.0f64; 256 * 256];
//!
//! multiply(&a, &b, &mut c, 256, 256, 256);
//! ```
//!
//! For large matrices, use the multi-threaded version:
//!
//! ```
//! use matmul::multiply_parallel;
//!
//! let a = vec![1.0f64; 1024 * 1024];
//! let b = vec![1.0f64; 1024 * 1024];
//! let mut c = vec![0.0f64; 1024 * 1024];
//!
//! multiply_parallel(&a, &b, &mut c, 1024, 1024, 1024, 4);
//! ```
//!
//! ## What's inside
//!
//! - 4x4, 12x4 AVX2 kernels
//! - 8x8 AVX-512 kernel
//! - Cache blocking tuned for L1/L2
//! - Adaptive multi-threading (scales down for small matrices)

pub mod blocked;
pub mod kernels;
pub mod matrix;
pub mod threaded;

pub use matrix::naive_ijk::matmul_naive_ijk;
pub use matrix::naive_ikj::matmul_naive_ikj;

/// Matrix multiply: C += A * B
///
/// Picks the fastest available kernel for your CPU (AVX-512 > AVX2 > scalar).
/// Matrices are row-major: A is m×k, B is k×n, C is m×n.
///
/// # Panics
///
/// Panics if the slice sizes don't match m, n, k.
pub fn multiply(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "A: expected {}x{}={} elements", m, k, m * k);
    assert_eq!(b.len(), k * n, "B: expected {}x{}={} elements", k, n, k * n);
    assert_eq!(c.len(), m * n, "C: expected {}x{}={} elements", m, n, m * n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            unsafe { blocked::gemm_8x8::matmul_blocked_8x8(a, b, c, m, n, k, None, None) };
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { blocked::gemm_12x4::matmul_blocked_12x4(a, b, c, m, n, k, None, None) };
            return;
        }
    }

    matrix::naive_ikj::matmul_naive_ikj(a, b, c, m, n, k);
}

/// Same as [`multiply`] but uses multiple threads.
///
/// Thread count adapts to matrix size - small matrices use fewer threads
/// because the overhead isn't worth it.
pub fn multiply_parallel(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) {
    assert_eq!(a.len(), m * k, "A: expected {}x{}={} elements", m, k, m * k);
    assert_eq!(b.len(), k * n, "B: expected {}x{}={} elements", k, n, k * n);
    assert_eq!(c.len(), m * n, "C: expected {}x{}={} elements", m, n, m * n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("fma") {
            threaded::gemm_8x8_mt::matmul_blocked_8x8_mt(a, b, c, m, n, k, num_threads);
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            threaded::gemm_12x4_mt::matmul_blocked_12x4_mt(a, b, c, m, n, k, num_threads);
            return;
        }
    }

    matrix::naive_ikj::matmul_naive_ikj(a, b, c, m, n, k);
}
