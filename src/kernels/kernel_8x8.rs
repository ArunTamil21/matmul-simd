//! 8×8 AVX-512 microkernel for matrix multiplication.

/// Computes an 8×8 tile: C[0:8, 0:8] += A_packed × B_packed
///
/// Uses 8 ZMM registers (512-bit) as accumulators. AVX-512 processes 8 f64
/// values per instruction, so this kernel handles 64 output elements per
/// iteration with excellent throughput on Skylake-X and later.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F, AVX-512DQ, and FMA (checked via `#[target_feature]`)
/// - `a_pack` points to `k * 8` contiguous f64 values (packed A panel)
/// - `b_pack` points to `k * 8` contiguous f64 values (packed B panel)
/// - `c` points to valid memory with stride `ldc`
/// - `c.add(row * ldc)` is valid for row in 0..8, each allowing read/write of 8 f64s
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[allow(clippy::identity_op)]
#[allow(clippy::erasing_op)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_8x8_avx512(
    a_pack: *const f64,
    b_pack: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    // 8 accumulators, one per output row (512 bits = 8 f64 each)
    let mut c0 = _mm512_loadu_pd(c.add(0 * ldc));
    let mut c1 = _mm512_loadu_pd(c.add(1 * ldc));
    let mut c2 = _mm512_loadu_pd(c.add(2 * ldc));
    let mut c3 = _mm512_loadu_pd(c.add(3 * ldc));
    let mut c4 = _mm512_loadu_pd(c.add(4 * ldc));
    let mut c5 = _mm512_loadu_pd(c.add(5 * ldc));
    let mut c6 = _mm512_loadu_pd(c.add(6 * ldc));
    let mut c7 = _mm512_loadu_pd(c.add(7 * ldc));

    for p in 0..k {
        let b_vec = _mm512_loadu_pd(b_pack.add(p * 8));

        c0 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 0)), b_vec, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 1)), b_vec, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 2)), b_vec, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 3)), b_vec, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 4)), b_vec, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 5)), b_vec, c5);
        c6 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 6)), b_vec, c6);
        c7 = _mm512_fmadd_pd(_mm512_set1_pd(*a_pack.add(p * 8 + 7)), b_vec, c7);
    }

    _mm512_storeu_pd(c.add(0 * ldc), c0);
    _mm512_storeu_pd(c.add(1 * ldc), c1);
    _mm512_storeu_pd(c.add(2 * ldc), c2);
    _mm512_storeu_pd(c.add(3 * ldc), c3);
    _mm512_storeu_pd(c.add(4 * ldc), c4);
    _mm512_storeu_pd(c.add(5 * ldc), c5);
    _mm512_storeu_pd(c.add(6 * ldc), c6);
    _mm512_storeu_pd(c.add(7 * ldc), c7);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_8x8_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping - AVX-512 not available");
            return;
        }

        let k = 16;
        let a: Vec<f64> = (0..8 * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * 8).map(|i| (i % 10) as f64).collect();
        let mut c = vec![0.0; 8 * 8];

        // Pack A
        let mut a_pack = vec![0.0; k * 8];
        for p in 0..k {
            for i in 0..8 {
                a_pack[p * 8 + i] = a[i * k + p];
            }
        }

        // Pack B
        let mut b_pack = vec![0.0; k * 8];
        for p in 0..k {
            for j in 0..8 {
                b_pack[p * 8 + j] = b[p * 8 + j];
            }
        }

        unsafe {
            kernel_8x8_avx512(a_pack.as_ptr(), b_pack.as_ptr(), c.as_mut_ptr(), k, 8);
        }

        // Naive reference
        let mut c_expected = vec![0.0; 8 * 8];
        for i in 0..8 {
            for j in 0..8 {
                for p in 0..k {
                    c_expected[i * 8 + j] += a[i * k + p] * b[p * 8 + j];
                }
            }
        }

        for i in 0..64 {
            assert!(
                (c[i] - c_expected[i]).abs() < 1e-10,
                "Mismatch at {}: got {}, expected {}",
                i,
                c[i],
                c_expected[i]
            );
        }
    }
}
