//! Benchmark runner for matmul implementations.

use matmul::blocked::gemm_4x4::matmul_blocked_4x4;
use matmul::blocked::gemm_8x8::matmul_blocked_8x8;
use matmul::blocked::gemm_12x4::matmul_blocked_12x4;
use matmul::matrix::naive_ijk::matmul_naive_ijk;
use matmul::matrix::naive_ikj::matmul_naive_ikj;
use matmul::threaded::gemm_4x4_mt::matmul_blocked_4x4_mt;
use matmul::threaded::gemm_8x8_mt::matmul_blocked_8x8_mt;
use matmul::threaded::gemm_12x4_mt::matmul_blocked_12x4_mt;
use std::time::Instant;

fn main() {
    println!("=== Matrix Multiplication Benchmark ===\n");

    let sizes = [256, 512, 1024];
    let iterations = 3;
    let mut all_results = Vec::new();

    let has_avx2 = is_x86_feature_detected!("avx2");
    let has_avx512 = is_x86_feature_detected!("avx512f");

    println!("CPU Features: AVX2={}, AVX-512={}\n", has_avx2, has_avx512);

    for &size in &sizes {
        println!("Matrix: {}×{}", size, size);
        println!("{}", "-".repeat(50));

        let (m, n, k) = (size, size, size);
        let a: Vec<f64> = (0..m * k).map(|i| (i % 100) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 100) as f64).collect();

        let mut results: Vec<(&str, (f64, f64))> = vec![
            (
                "Naive (i-j-k)",
                bench_fn(&a, &b, m, n, k, iterations, matmul_naive_ijk),
            ),
            (
                "Scalar (i-k-j)",
                bench_fn(&a, &b, m, n, k, iterations, matmul_naive_ikj),
            ),
        ];

        if has_avx2 {
            results.push((
                "4×4 AVX2",
                bench_unsafe(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| unsafe {
                    matmul_blocked_4x4(a, b, c, m, n, k, None, None)
                }),
            ));
            results.push((
                "4×4 AVX2 MT",
                bench_fn(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| {
                    matmul_blocked_4x4_mt(a, b, c, m, n, k, 4)
                }),
            ));
            results.push((
                "12×4 AVX2",
                bench_unsafe(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| unsafe {
                    matmul_blocked_12x4(a, b, c, m, n, k, None, None)
                }),
            ));
            results.push((
                "12×4 AVX2 MT",
                bench_fn(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| {
                    matmul_blocked_12x4_mt(a, b, c, m, n, k, 4)
                }),
            ));
        }

        if has_avx512 {
            results.push((
                "8×8 AVX-512",
                bench_unsafe(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| unsafe {
                    matmul_blocked_8x8(a, b, c, m, n, k, None, None)
                }),
            ));
            results.push((
                "8×8 AVX-512 MT",
                bench_fn(&a, &b, m, n, k, iterations, |a, b, c, m, n, k| {
                    matmul_blocked_8x8_mt(a, b, c, m, n, k, 4)
                }),
            ));
        }

        // Print results
        let baseline_time = results[0].1.0;
        for (i, (name, (time_ms, gflops))) in results.iter().enumerate() {
            let speedup = baseline_time / time_ms;
            println!(
                "{}. {:16} {:8.2} ms  {:6.2} GFLOPS  ({:.1}×)",
                i + 1,
                name,
                time_ms,
                gflops,
                speedup
            );
        }
        println!();

        all_results.push((size, results));
    }

    print_summary_table(&all_results);
}

/// Benchmark a safe matmul function
fn bench_fn<F>(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
    f: F,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64], &mut [f64], usize, usize, usize),
{
    // Warmup
    let mut c = vec![0.0; m * n];
    f(a, b, &mut c, m, n, k);

    // Timed runs
    let mut total = 0.0;
    for _ in 0..iterations {
        let mut c = vec![0.0; m * n];
        let start = Instant::now();
        f(a, b, &mut c, m, n, k);
        total += start.elapsed().as_secs_f64();
    }

    let avg = total / iterations as f64;
    let gflops = 2.0 * (m * n * k) as f64 / avg / 1e9;
    (avg * 1000.0, gflops)
}

/// Benchmark an unsafe matmul function (same logic, different type bounds)
fn bench_unsafe<F>(
    a: &[f64],
    b: &[f64],
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
    f: F,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64], &mut [f64], usize, usize, usize),
{
    bench_fn(a, b, m, n, k, iterations, f)
}
#[allow(clippy::type_complexity)]
fn print_summary_table(all_results: &[(usize, Vec<(&str, (f64, f64))>)]) {
    println!("\n{}", "=".repeat(90));
    println!("SUMMARY");
    println!("{}", "=".repeat(90));

    println!(
        "\n{:<18} {:>14} {:>14} {:>14} {:>12}",
        "Method", "256×256", "512×512", "1024×1024", "Speedup"
    );
    println!("{}", "-".repeat(90));

    let num_methods = all_results[0].1.len();

    for method_idx in 0..num_methods {
        let method_name = all_results[0].1[method_idx].0;

        let mut gflops_list = Vec::new();
        let mut speedups = Vec::new();

        for (_, results) in all_results {
            let (time_ms, gflops) = results[method_idx].1;
            let baseline_time = results[0].1.0;
            gflops_list.push(gflops);
            speedups.push(baseline_time / time_ms);
        }

        let avg_speedup: f64 = speedups.iter().sum::<f64>() / speedups.len() as f64;

        println!(
            "{:<18} {:>10.2} GF  {:>10.2} GF  {:>10.2} GF  {:>10.1}×",
            method_name, gflops_list[0], gflops_list[1], gflops_list[2], avg_speedup
        );
    }

    println!("{}", "=".repeat(90));
    println!("\nGF = GFLOPS (billion floating point operations per second)");
    println!("Speedup relative to Naive (i-j-k). Higher is better.\n");
}
