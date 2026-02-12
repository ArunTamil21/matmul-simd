//! 12×4 AVX2 microkernel for matrix multiplication.

/// Computes a 12×4 tile: C[0:12, 0:4] += A_packed × B_packed
///
/// Uses 12 AVX2 registers as accumulators (one per row of C). This larger
/// kernel amortizes loop overhead and achieves better instruction-level
/// parallelism than the 4×4 kernel, but requires more registers.
///
/// The 12×4 shape is chosen because:
/// - 12 accumulators × 256 bits = uses most of the 16 YMM registers
/// - 4 columns fits one AVX2 register (4 × f64)
/// - Leaves room for A broadcasts and B loads
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA (checked via `#[target_feature]`)
/// - `a_pack` points to `k * 12` contiguous f64 values (packed A panel)
/// - `b_pack` points to `k * 4` contiguous f64 values (packed B panel)
/// - `c` points to valid memory with stride `ldc`
/// - `c.add(row * ldc)` is valid for row in 0..12, each allowing read/write of 4 f64s
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::identity_op)]
#[allow(clippy::erasing_op)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_12x4_avx2(
    a_pack: *const f64,
    b_pack: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    // 12 accumulators, one per output row
    let mut c0 = _mm256_loadu_pd(c.add(0 * ldc));
    let mut c1 = _mm256_loadu_pd(c.add(1 * ldc));
    let mut c2 = _mm256_loadu_pd(c.add(2 * ldc));
    let mut c3 = _mm256_loadu_pd(c.add(3 * ldc));
    let mut c4 = _mm256_loadu_pd(c.add(4 * ldc));
    let mut c5 = _mm256_loadu_pd(c.add(5 * ldc));
    let mut c6 = _mm256_loadu_pd(c.add(6 * ldc));
    let mut c7 = _mm256_loadu_pd(c.add(7 * ldc));
    let mut c8 = _mm256_loadu_pd(c.add(8 * ldc));
    let mut c9 = _mm256_loadu_pd(c.add(9 * ldc));
    let mut c10 = _mm256_loadu_pd(c.add(10 * ldc));
    let mut c11 = _mm256_loadu_pd(c.add(11 * ldc));

    for p in 0..k {
        let b_vec = _mm256_loadu_pd(b_pack.add(p * 4));

        c0 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 0)), b_vec, c0);
        c1 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 1)), b_vec, c1);
        c2 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 2)), b_vec, c2);
        c3 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 3)), b_vec, c3);
        c4 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 4)), b_vec, c4);
        c5 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 5)), b_vec, c5);
        c6 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 6)), b_vec, c6);
        c7 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 7)), b_vec, c7);
        c8 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 8)), b_vec, c8);
        c9 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 9)), b_vec, c9);
        c10 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 10)), b_vec, c10);
        c11 = _mm256_fmadd_pd(_mm256_broadcast_sd(&*a_pack.add(p * 12 + 11)), b_vec, c11);
    }

    _mm256_storeu_pd(c.add(0 * ldc), c0);
    _mm256_storeu_pd(c.add(1 * ldc), c1);
    _mm256_storeu_pd(c.add(2 * ldc), c2);
    _mm256_storeu_pd(c.add(3 * ldc), c3);
    _mm256_storeu_pd(c.add(4 * ldc), c4);
    _mm256_storeu_pd(c.add(5 * ldc), c5);
    _mm256_storeu_pd(c.add(6 * ldc), c6);
    _mm256_storeu_pd(c.add(7 * ldc), c7);
    _mm256_storeu_pd(c.add(8 * ldc), c8);
    _mm256_storeu_pd(c.add(9 * ldc), c9);
    _mm256_storeu_pd(c.add(10 * ldc), c10);
    _mm256_storeu_pd(c.add(11 * ldc), c11);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_12x4_correctness() {
        if !is_x86_feature_detected!("avx2") {
            println!("Skipping - AVX2 not available");
            return;
        }

        let k = 16;
        let a: Vec<f64> = (0..12 * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * 4).map(|i| (i % 10) as f64).collect();
        let mut c = vec![0.0; 12 * 4];

        // Pack A: for each k position, store 12 consecutive row values
        let mut a_pack = vec![0.0; k * 12];
        for p in 0..k {
            for i in 0..12 {
                a_pack[p * 12 + i] = a[i * k + p];
            }
        }

        // Pack B: for each k position, store 4 consecutive column values
        let mut b_pack = vec![0.0; k * 4];
        for p in 0..k {
            for j in 0..4 {
                b_pack[p * 4 + j] = b[p * 4 + j];
            }
        }

        unsafe {
            kernel_12x4_avx2(a_pack.as_ptr(), b_pack.as_ptr(), c.as_mut_ptr(), k, 4);
        }

        // Naive reference
        let mut c_expected = vec![0.0; 12 * 4];
        for i in 0..12 {
            for j in 0..4 {
                for p in 0..k {
                    c_expected[i * 4 + j] += a[i * k + p] * b[p * 4 + j];
                }
            }
        }

        for i in 0..12 * 4 {
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
