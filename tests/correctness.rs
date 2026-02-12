use matmul::blocked::gemm_4x4::matmul_blocked_4x4;
use matmul::blocked::gemm_8x8::matmul_blocked_8x8;
use matmul::blocked::gemm_12x4::matmul_blocked_12x4;
use matmul::matrix::naive_ikj::matmul_naive_ikj;
use matmul::threaded::gemm_4x4_mt::matmul_blocked_4x4_mt;
use matmul::threaded::gemm_8x8_mt::matmul_blocked_8x8_mt;
use matmul::threaded::gemm_12x4_mt::matmul_blocked_12x4_mt;
use matmul::{multiply, multiply_parallel};

fn assert_matrices_equal(expected: &[f64], actual: &[f64], name: &str) {
    assert_eq!(expected.len(), actual.len(), "{}: length mismatch", name);
    for i in 0..expected.len() {
        assert!(
            (expected[i] - actual[i]).abs() < 1e-8,
            "{}: mismatch at index {}: expected {}, got {}",
            name,
            i,
            expected[i],
            actual[i]
        );
    }
}

// ============================================================
// Small matrix tests (edge case handling)
// ============================================================

#[test]
fn test_2x2_multiply() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let mut c_naive = vec![0.0; 4];
    let mut c_fast = vec![0.0; 4];

    matmul_naive_ikj(&a, &b, &mut c_naive, 2, 2, 2);
    multiply(&a, &b, &mut c_fast, 2, 2, 2);

    assert_matrices_equal(&c_naive, &c_fast, "2x2");
}

#[test]
fn test_2x3_times_3x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

    let mut c_naive = vec![0.0; 4];
    let mut c_fast = vec![0.0; 4];

    matmul_naive_ikj(&a, &b, &mut c_naive, 2, 2, 3);
    multiply(&a, &b, &mut c_fast, 2, 2, 3);

    assert_eq!(c_naive, vec![58.0, 64.0, 139.0, 154.0]);

    assert_matrices_equal(&c_naive, &c_fast, "2x3 * 3x2");
}

#[test]
fn test_small_odd_sizes() {
    let test_sizes = [
        (3, 3, 3),
        (5, 5, 5),
        (7, 7, 7),
        (3, 5, 7),
        (7, 3, 5),
        (11, 13, 17),
    ];

    for (m, n, k) in test_sizes {
        let a: Vec<f64> = (0..m * k).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; m * n];
        let mut c_fast = vec![0.0; m * n];

        matmul_naive_ikj(&a, &b, &mut c_naive, m, n, k);
        multiply(&a, &b, &mut c_fast, m, n, k);

        assert_matrices_equal(&c_naive, &c_fast, &format!("{}x{}x{}", m, n, k));
    }
}

// ============================================================
// Tile boundary tests
// ============================================================

#[test]
fn test_tile_boundary_4x4() {
    let test_sizes = [3, 4, 5, 7, 8, 9, 15, 16, 17];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_fast = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        multiply(&a, &b, &mut c_fast, size, size, size);

        assert_matrices_equal(&c_naive, &c_fast, &format!("tile_4x4_size_{}", size));
    }
}

#[test]
fn test_tile_boundary_12x4() {
    let test_sizes = [11, 12, 13, 23, 24, 25, 35, 36, 37];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_fast = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        multiply(&a, &b, &mut c_fast, size, size, size);

        assert_matrices_equal(&c_naive, &c_fast, &format!("tile_12x4_size_{}", size));
    }
}

#[test]
fn test_tile_boundary_8x8() {
    let test_sizes = [7, 8, 9, 15, 16, 17, 23, 24, 25];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_fast = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        multiply(&a, &b, &mut c_fast, size, size, size);

        assert_matrices_equal(&c_naive, &c_fast, &format!("tile_8x8_size_{}", size));
    }
}

// ============================================================
// Direct kernel tests (bypassing auto-dispatch)
// ============================================================

#[test]
fn test_gemm_4x4_direct() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping - AVX2 not available");
        return;
    }

    let test_sizes = [4, 8, 16, 17, 31, 32, 33, 64, 65];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_gemm = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        unsafe {
            matmul_blocked_4x4(&a, &b, &mut c_gemm, size, size, size, None, None);
        }

        assert_matrices_equal(&c_naive, &c_gemm, &format!("gemm_4x4_size_{}", size));
    }
}

#[test]
fn test_gemm_12x4_direct() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping - AVX2 not available");
        return;
    }

    let test_sizes = [4, 12, 13, 24, 25, 36, 37, 48, 49];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_gemm = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        unsafe {
            matmul_blocked_12x4(&a, &b, &mut c_gemm, size, size, size, None, None);
        }

        assert_matrices_equal(&c_naive, &c_gemm, &format!("gemm_12x4_size_{}", size));
    }
}

#[test]
fn test_gemm_8x8_direct() {
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping - AVX-512 not available");
        return;
    }

    let test_sizes = [8, 9, 16, 17, 24, 25, 32, 33, 64, 65];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; size * size];
        let mut c_gemm = vec![0.0; size * size];

        matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
        unsafe {
            matmul_blocked_8x8(&a, &b, &mut c_gemm, size, size, size, None, None);
        }

        assert_matrices_equal(&c_naive, &c_gemm, &format!("gemm_8x8_size_{}", size));
    }
}

// ============================================================
// Multi-threaded tests
// ============================================================

#[test]
fn test_parallel_matches_single_threaded() {
    let test_sizes = [64, 128, 256];

    for size in test_sizes {
        let a: Vec<f64> = (0..size * size).map(|i| (i % 17) as f64).collect();
        let b: Vec<f64> = (0..size * size).map(|i| (i % 13) as f64).collect();

        let mut c_single = vec![0.0; size * size];
        let mut c_parallel = vec![0.0; size * size];

        multiply(&a, &b, &mut c_single, size, size, size);
        multiply_parallel(&a, &b, &mut c_parallel, size, size, size, 4);

        assert_matrices_equal(&c_single, &c_parallel, &format!("parallel_size_{}", size));
    }
}

#[test]
fn test_parallel_small_matrix() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let mut c_naive = vec![0.0; 4];
    let mut c_parallel = vec![0.0; 4];

    matmul_naive_ikj(&a, &b, &mut c_naive, 2, 2, 3);
    multiply_parallel(&a, &b, &mut c_parallel, 2, 2, 3, 4);

    assert_matrices_equal(&c_naive, &c_parallel, "parallel_small");
}

#[test]
fn test_mt_4x4_direct() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping - AVX2 not available");
        return;
    }

    let size = 256;
    let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
    let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

    let mut c_naive = vec![0.0; size * size];
    let mut c_mt = vec![0.0; size * size];

    matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
    matmul_blocked_4x4_mt(&a, &b, &mut c_mt, size, size, size, 4);

    assert_matrices_equal(&c_naive, &c_mt, "mt_4x4");
}

#[test]
fn test_mt_12x4_direct() {
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping - AVX2 not available");
        return;
    }

    let size = 256;
    let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
    let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

    let mut c_naive = vec![0.0; size * size];
    let mut c_mt = vec![0.0; size * size];

    matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
    matmul_blocked_12x4_mt(&a, &b, &mut c_mt, size, size, size, 4);

    assert_matrices_equal(&c_naive, &c_mt, "mt_12x4");
}

#[test]
fn test_mt_8x8_direct() {
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping - AVX-512 not available");
        return;
    }

    let size = 256;
    let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
    let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

    let mut c_naive = vec![0.0; size * size];
    let mut c_mt = vec![0.0; size * size];

    matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
    matmul_blocked_8x8_mt(&a, &b, &mut c_mt, size, size, size, 4);

    assert_matrices_equal(&c_naive, &c_mt, "mt_8x8");
}

// ============================================================
// Non-square matrix tests
// ============================================================

#[test]
fn test_non_square_matrices() {
    let test_cases = [
        (32, 64, 48),  // wide result
        (64, 32, 48),  // tall result
        (100, 50, 75), // odd sizes
        (48, 48, 100), // deep k
        (13, 17, 19),  // primes
    ];

    for (m, n, k) in test_cases {
        let a: Vec<f64> = (0..m * k).map(|i| (i % 10) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 10) as f64).collect();

        let mut c_naive = vec![0.0; m * n];
        let mut c_fast = vec![0.0; m * n];

        matmul_naive_ikj(&a, &b, &mut c_naive, m, n, k);
        multiply(&a, &b, &mut c_fast, m, n, k);

        assert_matrices_equal(&c_naive, &c_fast, &format!("non_square_{}x{}x{}", m, n, k));
    }
}

// ============================================================
// Accumulation test (C += A*B, not C = A*B)
// ============================================================

#[test]
fn test_accumulation() {
    let size = 64;
    let a: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();
    let b: Vec<f64> = (0..size * size).map(|i| (i % 10) as f64).collect();

    // Start with non-zero C
    let mut c_naive = vec![5.0; size * size];
    let mut c_fast = vec![5.0; size * size];

    matmul_naive_ikj(&a, &b, &mut c_naive, size, size, size);
    multiply(&a, &b, &mut c_fast, size, size, size);

    assert_matrices_equal(&c_naive, &c_fast, "accumulation");

    // Verify values are actually > 5 (not overwritten)
    assert!(c_fast[0] > 5.0, "Should accumulate, not overwrite");
}
