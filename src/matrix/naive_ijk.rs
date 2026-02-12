/// Naive matrix multiplication using i-j-k loop order.
///
/// This is the textbook triple-loop implementation. It's slow because
/// the innermost loop accesses B with stride `n` (column-wise), causing
/// cache misses on every iteration.
///
/// Use this as a correctness baseline, not for performance.
///
/// # Arguments
///
/// * `a` - Matrix A (m × k), row-major
/// * `b` - Matrix B (k × n), row-major
/// * `c` - Matrix C (m × n), row-major, accumulated into (C += A * B)
/// * `m` - Rows of A and C
/// * `n` - Columns of B and C
/// * `k` - Columns of A, rows of B
pub fn matmul_naive_ijk(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}