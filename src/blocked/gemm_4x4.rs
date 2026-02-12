//! 4×4 blocked GEMM using AVX2.

use crate::kernels::kernel_4x4::kernel_4x4_avx2;
use crate::matrix::transpose::transpose;

/// Cache-blocked matrix multiplication using 4×4 AVX2 kernel.
///
/// Breaks the computation into tiles, packs A and B for sequential access,
/// and calls the microkernel for each tile. Handles edge cases for matrices
/// not divisible by 4.
///
/// # Safety
///
/// Caller must ensure:
/// - CPU supports AVX2 and FMA
/// - All slice lengths match the provided dimensions
///
/// # Arguments
///
/// * `row_start`, `row_end` - Optional row range for multi-threaded use
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::identity_op)]
#[allow(clippy::erasing_op)]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_blocked_4x4(
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
    // Step 1: Transpose B once at the start
    // This lets us access B's columns as rows, which is way faster
    let mut bt = vec![0.0; k * n];
    transpose(b, &mut bt, k, n);

    // Only process complete 4×4 tiles, handle leftovers separately
    let m_start = (start / 4) * 4;
    let m_end = (end / 4) * 4;
    let n_main = (n / 4) * 4;

    // Cache blocking sizes - tuned to fit in L1/L2 cache
    let kc = k.min(256); // L1 blocking: keep working set small
    let mc = m.min(128); // L2 blocking: reuse A across columns

    // Pre-allocate buffers for packed data
    let mut a_panel = vec![0.0; mc * kc]; // Big panel that stays in L2
    let mut b_pack = vec![0.0; 4 * kc]; // Small panel for L1

    // Three nested loops for cache blocking
    // Outer: K dimension (process k in chunks)
    for kk in (0..k).step_by(kc) {
        let k_block = (kk + kc).min(k) - kk;

        // Middle: M dimension (process rows in chunks)
        for ii in (m_start..m_end).step_by(mc) {
            let m_block = (ii + mc).min(m_end) - ii;

            // Pack a big chunk of A into cache-friendly layout
            // Do this ONCE, then reuse for all columns
            pack_a_panel_large(a, &mut a_panel, ii, kk, m_block, k_block, k);

            // Inner: Loop over columns (process 4 at a time)
            for j in (0..n_main).step_by(4) {
                // Pack 4 columns of B
                pack_b_panel(&bt, &mut b_pack, j, kk, k_block, k);

                // Now call the kernel for each 4-row chunk
                for i in (0..m_block).step_by(4) {
                    // Figure out where this 4×4 tile starts in the big A panel
                    let a_pack_offset = i * k_block;

                    kernel_4x4_avx2(
                        a_panel.as_ptr().add(a_pack_offset),
                        b_pack.as_ptr(),
                        c.as_mut_ptr().add((ii + i) * n + j),
                        k_block,
                        n,
                    );
                }
            }
        }
    }

    // Handle leftover rows and columns that don't fit in 4×4 tiles
    if m_end < end {
        edge_case_rows(a, b, c, m_end, end, n, k);
    }
    if n_main < n {
        edge_case_cols(a, b, c, m_start, m_end, n_main, n, k); // CHANGED
    }
}

// Pack a big chunk of A (many rows × k columns) into a more cache-friendly layout
// Original: row-major (rows are sequential)
// Packed: column-major in groups of 4 (each k-position's 4 values are together)
#[allow(clippy::identity_op)]
fn pack_a_panel_large(
    a: &[f64],
    a_panel: &mut [f64],
    i_start: usize,
    k_start: usize,
    m_block: usize,
    k_block: usize,
    k_total: usize,
) {
    // Process in groups of 4 rows (that's our kernel height)
    for i_offset in (0..m_block).step_by(4) {
        // For each k position in this block
        for p in 0..k_block {
            let k_idx = k_start + p;
            let out_base = (i_offset * k_block) + (p * 4);

            // Copy 4 row values for this k position
            // Now they're right next to each other in memory!
            a_panel[out_base + 0] = a[(i_start + i_offset + 0) * k_total + k_idx];
            a_panel[out_base + 1] = a[(i_start + i_offset + 1) * k_total + k_idx];
            a_panel[out_base + 2] = a[(i_start + i_offset + 2) * k_total + k_idx];
            a_panel[out_base + 3] = a[(i_start + i_offset + 3) * k_total + k_idx];
        }
    }
}

// Pack 4 columns of transposed B into sequential layout
// After transpose, B's columns are rows in bt, so we can read them easily
#[allow(clippy::identity_op)]
fn pack_b_panel(
    bt: &[f64],
    b_pack: &mut [f64],
    j_start: usize,
    k_start: usize,
    k_block: usize,
    k_total: usize,
) {
    for p in 0..k_block {
        let k_idx = k_start + p;
        // Grab 4 values from 4 consecutive rows of bt (which are columns of original B)
        b_pack[p * 4 + 0] = bt[(j_start + 0) * k_total + k_idx];
        b_pack[p * 4 + 1] = bt[(j_start + 1) * k_total + k_idx];
        b_pack[p * 4 + 2] = bt[(j_start + 2) * k_total + k_idx];
        b_pack[p * 4 + 3] = bt[(j_start + 3) * k_total + k_idx];
    }
}

// Handle rows that don't fit in 4×4 tiles (just use simple scalar code)
#[allow(clippy::too_many_arguments)]
fn edge_case_rows(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    i_start: usize,
    m: usize,
    n: usize,
    k: usize,
) {
    for i in i_start..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

// Handle columns that don't fit in 4×4 tiles
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
