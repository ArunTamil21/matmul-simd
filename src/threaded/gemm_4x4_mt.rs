//! Multi-threaded 4×4 blocked GEMM.

use crate::blocked::gemm_4x4::matmul_blocked_4x4;
use std::sync::Arc;
use std::thread;

/// Multi-threaded matrix multiplication using 4×4 AVX2 kernel.
///
/// Splits rows across threads, with each thread running the blocked
/// GEMM on its portion. Thread count adapts based on matrix size:
/// - < 100M FLOPs: 1 thread
/// - < 300M FLOPs: 2 threads
/// - Otherwise: up to `num_threads`
///
/// # Arguments
///
/// * `num_threads` - Maximum threads (actual may be fewer for small matrices)
pub fn matmul_blocked_4x4_mt(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    n: usize,
    k: usize,
    num_threads: usize,
) {
    let effective_threads = choose_thread_count(m, n, k, num_threads);

    if effective_threads == 1 {
        unsafe {
            matmul_blocked_4x4(a, b, c, m, n, k, None, None);
        }
        return;
    }

    let rows_per_thread = m / effective_threads;

    let a_arc = Arc::new(a.to_vec());
    let b_arc = Arc::new(b.to_vec());

    let c_ptr = c.as_mut_ptr() as usize;

    let handles: Vec<_> = (0..effective_threads)
        .map(|tid| {
            let a_clone = Arc::clone(&a_arc);
            let b_clone = Arc::clone(&b_arc);

            thread::spawn(move || {
                let start_row = tid * rows_per_thread;
                let end_row = start_row + rows_per_thread;

                unsafe {
                    let c_base = c_ptr as *mut f64;
                    let full_c = std::slice::from_raw_parts_mut(c_base, m * n);

                    matmul_blocked_4x4(
                        &a_clone,
                        &b_clone,
                        full_c,
                        m,
                        n,
                        k,
                        Some(start_row),
                        Some(end_row),
                    );
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

fn choose_thread_count(m: usize, n: usize, k: usize, max_threads: usize) -> usize {
    let flops = 2.0 * (m * n * k) as f64;

    const SINGLE_THREAD_THRESHOLD: f64 = 100_000_000.0;
    const TWO_THREAD_THRESHOLD: f64 = 300_000_000.0;

    let optimal_threads = if flops < SINGLE_THREAD_THRESHOLD {
        1
    } else if flops < TWO_THREAD_THRESHOLD {
        2
    } else {
        max_threads
    };

    let threads_by_rows = (m / 64).max(1);

    optimal_threads.min(threads_by_rows).min(max_threads)
}
