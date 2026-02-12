//! SIMD microkernels for the inner loop of matrix multiplication.
//!
//! These kernels compute small tiles of C += A × B using AVX2 or AVX-512
//! intrinsics. They're called by the blocked GEMM implementations after
//! packing the input matrices into cache-friendly layouts.
//!
//! Available kernels:
//! - `kernel_4x4`: 4×4 tile, AVX2 (4 registers)
//! - `kernel_12x4`: 12×4 tile, AVX2 (12 registers, better throughput)
//! - `kernel_8x8`: 8×8 tile, AVX-512 (8 registers, 64 outputs per iteration)

pub mod kernel_12x4;
pub mod kernel_4x4;
pub mod kernel_8x8;
