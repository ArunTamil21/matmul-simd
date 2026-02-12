//! Simple SIMD matmul without cache blocking.
//!
//! This was an early experiment - it uses SIMD but doesn't pack matrices
//! or block for cache. Kept for comparison/educational purposes.

use std::arch::x86_64::*;

/// Simple 4×4 SIMD matmul without packing or blocking.
///
/// Demonstrates basic AVX2 usage but doesn't achieve good performance
/// on large matrices due to poor cache behavior. Use `gemm_4x4` or
/// `gemm_12x4` instead.
///
/// # Safety
///
/// Caller must ensure CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::missing_safety_doc)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn matmul_simple_simd(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
) {
    let m_main = (m / 4) * 4;
    let n_main = (n / 4) * 4;

    for i in (0..m_main).step_by(4) {
        for j in (0..n_main).step_by(4) {
            let mut c0 = _mm256_setzero_pd();
            let mut c1 = _mm256_setzero_pd();
            let mut c2 = _mm256_setzero_pd();
            let mut c3 = _mm256_setzero_pd();

            for p in 0..k {
                let b_vec = _mm256_loadu_pd(b.as_ptr().add(p * n + j));

                let a0 = _mm256_set1_pd(a[i * k + p]);
                let a1 = _mm256_set1_pd(a[(i + 1) * k + p]);
                let a2 = _mm256_set1_pd(a[(i + 2) * k + p]);
                let a3 = _mm256_set1_pd(a[(i + 3) * k + p]);

                c0 = _mm256_fmadd_pd(a0, b_vec, c0);
                c1 = _mm256_fmadd_pd(a1, b_vec, c1);
                c2 = _mm256_fmadd_pd(a2, b_vec, c2);
                c3 = _mm256_fmadd_pd(a3, b_vec, c3);
            }

            let c0_orig = _mm256_loadu_pd(c.as_ptr().add(i * n + j));
            let c1_orig = _mm256_loadu_pd(c.as_ptr().add((i + 1) * n + j));
            let c2_orig = _mm256_loadu_pd(c.as_ptr().add((i + 2) * n + j));
            let c3_orig = _mm256_loadu_pd(c.as_ptr().add((i + 3) * n + j));

            _mm256_storeu_pd(c.as_mut_ptr().add(i * n + j), _mm256_add_pd(c0, c0_orig));
            _mm256_storeu_pd(
                c.as_mut_ptr().add((i + 1) * n + j),
                _mm256_add_pd(c1, c1_orig),
            );
            _mm256_storeu_pd(
                c.as_mut_ptr().add((i + 2) * n + j),
                _mm256_add_pd(c2, c2_orig),
            );
            _mm256_storeu_pd(
                c.as_mut_ptr().add((i + 3) * n + j),
                _mm256_add_pd(c3, c3_orig),
            );
        }
    }

    for i in m_main..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }

    for i in 0..m_main {
        for p in 0..k {
            for j in n_main..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}
