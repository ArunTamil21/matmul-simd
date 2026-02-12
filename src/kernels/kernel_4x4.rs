//! 4×4 AVX2 microkernel for matrix multiplication.

/// Computes a 4×4 tile: C[0:4, 0:4] += A_packed × B_packed
///
/// This is the inner kernel called by the blocked GEMM. It keeps 4 AVX2
/// registers as accumulators (one per row of C), loads A values via broadcast,
/// and uses FMA for the multiply-accumulate.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA (checked via `#[target_feature]`)
/// - `a_pack` points to `k * 4` contiguous f64 values (packed A panel)
/// - `b_pack` points to `k * 4` contiguous f64 values (packed B panel)
/// - `c` points to valid memory with stride `ldc`
/// - `c.add(row * ldc)` is valid for row in 0..4, each allowing read/write of 4 f64s
///
#[allow(clippy::identity_op)]
#[allow(clippy::erasing_op)]
#[allow(unsafe_op_in_unsafe_fn)]
pub unsafe fn kernel_4x4_avx2(
    a_pack: *const f64,
    b_pack: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    // Load existing C values (we accumulate, not overwrite)
    let mut c0 = _mm256_loadu_pd(c.add(0 * ldc));
    let mut c1 = _mm256_loadu_pd(c.add(1 * ldc));
    let mut c2 = _mm256_loadu_pd(c.add(2 * ldc));
    let mut c3 = _mm256_loadu_pd(c.add(3 * ldc));

    // Main loop: for each k, load B once, broadcast A values, FMA into C
    for p in 0..k {
        let b_vec = _mm256_loadu_pd(b_pack.add(p * 4));

        let a0 = _mm256_broadcast_sd(&*a_pack.add(p * 4 + 0));
        let a1 = _mm256_broadcast_sd(&*a_pack.add(p * 4 + 1));
        let a2 = _mm256_broadcast_sd(&*a_pack.add(p * 4 + 2));
        let a3 = _mm256_broadcast_sd(&*a_pack.add(p * 4 + 3));

        c0 = _mm256_fmadd_pd(a0, b_vec, c0);
        c1 = _mm256_fmadd_pd(a1, b_vec, c1);
        c2 = _mm256_fmadd_pd(a2, b_vec, c2);
        c3 = _mm256_fmadd_pd(a3, b_vec, c3);
    }

    // Store results back to C
    _mm256_storeu_pd(c.add(0 * ldc), c0);
    _mm256_storeu_pd(c.add(1 * ldc), c1);
    _mm256_storeu_pd(c.add(2 * ldc), c2);
    _mm256_storeu_pd(c.add(3 * ldc), c3);
}
