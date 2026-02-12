//! Multi-threaded 8×8 blocked GEMM using AVX-512.

use crate::blocked::gemm_8x8::matmul_blocked_8x8;
use std::sync::Arc;
use std::thread;

/// Multi-threaded matrix multiplication using 8×8 AVX-512 kernel.
///
/// Splits rows across threads, with each thread running the blocked
/// GEMM on its portion. Best performance on Skylake-X and later with
/// AVX-512 support.
///
/// # Arguments
///
/// * `num_threads` - Maximum threads (actual may be fewer for small matrices)
pub fn matmul_blocked_8x8_mt(
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
            matmul_blocked_8x8(a, b, c, m, n, k, None, None);
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

                    matmul_blocked_8x8(
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
            crate::blocked::gemm_8x8::matmul_blocked_8x8(&a, &b, &mut c_gemm, m, n, k, None, None);
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

    #[test]
    fn test_gemm_8x8_mt_correctness() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping - AVX-512 not available");
            return;
        }

        let m = 256;
        let n = 256;
        let k = 256;

        let a: Vec<f64> = (0..m * k).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; m * n];
        matmul_naive_ikj(&a, &b, &mut c_naive, m, n, k);

        let mut c_mt = vec![0.0; m * n];
        matmul_blocked_8x8_mt(&a, &b, &mut c_mt, m, n, k, 4);

        for i in 0..m * n {
            assert!(
                (c_naive[i] - c_mt[i]).abs() < 1e-6,
                "Mismatch at {}: naive={}, mt={}",
                i,
                c_naive[i],
                c_mt[i]
            );
        }

        println!(" 8×8 Multi-threaded GEMM test passed!");
    }

    #[test]
    fn test_adaptive_threading() {
        // Small matrix should use 1 thread (256×256 = 33M FLOPs)
        assert_eq!(choose_thread_count(256, 256, 256, 4), 1);

        // Medium matrix should use 2 threads (450×450 = 182M FLOPs)
        assert_eq!(choose_thread_count(450, 450, 450, 4), 2);

        // Large matrix should use all threads (1024×1024 = 2.1B FLOPs)
        assert_eq!(choose_thread_count(1024, 1024, 1024, 4), 4);

        // Very small rows should limit threads (only 32 rows = can't use 4 threads)
        assert_eq!(choose_thread_count(32, 1024, 1024, 4), 1);

        println!(" Adaptive threading logic test passed!");
    }
}
