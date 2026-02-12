//! Multi-threaded GEMM implementations.
//!
//! These wrap the blocked GEMM functions with parallel execution across
//! rows. Thread count adapts to matrix size - small matrices use fewer
//! threads to avoid overhead.
//!
//! Available implementations:
//! - `gemm_4x4_mt`: Multi-threaded 4×4 AVX2
//! - `gemm_12x4_mt`: Multi-threaded 12×4 AVX2
//! - `gemm_8x8_mt`: Multi-threaded 8×8 AVX-512

pub mod gemm_12x4_mt;
pub mod gemm_4x4_mt;
pub mod gemm_8x8_mt;
