//! 8×8 blocked GEMM using AVX-512.

use crate::kernels::kernel_8x8::kernel_8x8_avx512;
use crate::matrix::transpose::transpose;

/// Cache-blocked matrix multiplication using 8×8 AVX-512 kernel.
///
/// AVX-512 processes 8 doubles per instruction (vs 4 for AVX2), so this
/// kernel handles 64 output elements per microkernel call. Best performance
/// on Skylake-X and later CPUs.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX-512F, AVX-512DQ, and FMA
/// - All slice lengths match the provided dimensions
///
/// # Arguments
///
/// * `row_start`, `row_end` - Optional row range for multi-threaded use
#[target_feature(enable = "avx512f,avx512dq,fma")]
#[allow(clippy::identity_op)]
#[allow(clippy::erasing_op)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_blocked_8x8(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    row_start: Option<usize>,
    row_end: Option<usize>,
) {
    let start = row_start.unwrap_or(0);
    let end = row_end.unwrap_or(m);

    let mut bt = vec![0.0; k * n];
    transpose(b, &mut bt, k, n);

    let m_start = (start / 8) * 8;
    let m_end = (end / 8) * 8;
    let n_main = (n / 8) * 8;

    let kc = k.min(256);
    let mc = (end - start).min(128);

    let mr: usize = 8;
    let nr = 8;

    let mut a_panel = vec![0.0; mc * kc];
    let mut b_panel = vec![0.0; 8 * kc];

    for kk in (0..k).step_by(kc) {
        let k_block = (kk + kc).min(k) - kk;

        for ii in (m_start..m_end).step_by(mc) {
            let m_block = (ii + mc).min(m_end) - ii;

            pack_big_a_panel(a, &mut a_panel, ii, kk, m_block, k_block, k);

            for j in (0..n_main).step_by(nr) {
                pack_b_panel(&bt, &mut b_panel, j, kk, k_block, k);

                for i in (0..m_block).step_by(mr) {
                    let a_pack_offset = i * k_block;

                    kernel_8x8_avx512(
                        a_panel.as_ptr().add(a_pack_offset),
                        b_panel.as_ptr(),
                        c.as_mut_ptr().add((ii + i) * n + j),
                        k_block,
                        n,
                    );
                }
            }
        }
    }

    if m_end < end {
        edge_case_rows(a, b, c, m_end, end, n, k);
    }
    if n_main < n {
        edge_case_cols(a, b, c, m_start, m_end, n_main, n, k);
    }
}

fn pack_big_a_panel(
    a: &[f64],
    a_panel: &mut [f64],
    i_start: usize,
    k_start: usize,
    m_block: usize,
    k_block: usize,
    k_total: usize,
) {
    for i_offset in (0..m_block).step_by(8) {
        for p in 0..k_block {
            let k_idx = k_start + p;
            let out_base = (i_offset * k_block) + (p * 8);

            for idx in 0..8 {
                a_panel[out_base + idx] = a[(i_start + i_offset + idx) * k_total + k_idx];
            }
        }
    }
}

fn pack_b_panel(
    bt: &[f64],
    b_pack: &mut [f64],
    j_start: usize,
    k_start: usize,
    k_block: usize,
    k_total: usize,
) {
    for p in 0..k_block {
        for idx in 0..8 {
            b_pack[p * 8 + idx] = bt[(j_start + idx) * k_total + (k_start + p)];
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn edge_case_rows(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    i_start: usize,
    i_end: usize,
    n: usize,
    k: usize,
) {
    for i in i_start..i_end {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn edge_case_cols(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    i_start: usize,
    i_end: usize,
    j_start: usize,
    n: usize,
    k: usize,
) {
    for i in i_start..i_end {
        for p in 0..k {
            for j in j_start..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::naive_ikj::matmul_naive_ikj;

    #[test]
    fn test_gemm_8x8_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping - AVX-512 not available");
            return;
        }

        let m = 144;
        let n = 128;
        let k = 256;

        let a: Vec<f64> = (0..m * k).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; m * n];
        matmul_naive_ikj(&a, &b, &mut c_naive, m, n, k);

        let mut c_gemm = vec![0.0; m * n];
        unsafe {
            matmul_blocked_8x8(&a, &b, &mut c_gemm, m, n, k, None, None);
        }

        for i in 0..m * n {
            assert!(
                (c_naive[i] - c_gemm[i]).abs() < 1e-8,
                "Mismatch at {}: naive={}, gemm={}",
                i,
                c_naive[i],
                c_gemm[i]
            );
        }

        println!(" 8×8 GEMM test passed!");
    }
}
