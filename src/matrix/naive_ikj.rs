/// Cache-friendly matrix multiplication using i-k-j loop order.
///
/// By swapping the j and k loops, the innermost loop now accesses both
/// B and C sequentially (stride 1). This alone gives ~9× speedup over
/// the naive i-j-k order on large matrices.
///
/// This is the scalar baseline that SIMD kernels are compared against.
///
/// # Arguments
///
/// * `a` - Matrix A (m × k), row-major
/// * `b` - Matrix B (k × n), row-major
/// * `c` - Matrix C (m × n), row-major, accumulated into (C += A * B)
/// * `m` - Rows of A and C
/// * `n` - Columns of B and C
/// * `k` - Columns of A, rows of B
pub fn matmul_naive_ikj(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

/// i-k-j multiplication with pre-transposed B matrix.
///
/// When B is already transposed (stored as B^T), accessing `bt[j * k + p]`
/// becomes sequential in j. Useful when multiplying the same B many times.
///
/// # Arguments
///
/// * `bt` - Transposed matrix B^T (n × k), row-major
pub fn matmul_ikj_transposed(a: &[f64], bt: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * bt[j * k + p];
            }
        }
    }
}
