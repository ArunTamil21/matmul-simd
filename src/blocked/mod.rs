//! Cache-blocked GEMM implementations.
//!
//! These functions break the matrix multiplication into tiles that fit in
//! L1/L2 cache, pack the data for sequential access, then call the SIMD
//! microkernels for the inner computation.
//!
//! Available implementations:
//! - `gemm_4x4`: Uses 4×4 AVX2 kernel
//! - `gemm_12x4`: Uses 12×4 AVX2 kernel (better throughput)
//! - `gemm_8x8`: Uses 8×8 AVX-512 kernel

pub mod gemm_12x4;
pub mod gemm_4x4;
pub mod gemm_8x8;
pub mod simple_simd;
