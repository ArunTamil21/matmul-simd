# matmul

Fast matrix multiplication in Rust, built from scratch. Achieves up to **62% of NumPy/OpenBLAS performance** through SIMD, cache blocking, and adaptive multi-threading.

## Performance

### Results (1024×1024 matrices)

| Platform | CPU | Best Single-Thread | Best Multi-Thread | vs Naive |
|----------|-----|-------------------|-------------------|----------|
| macOS | i5-8257U @ 1.4 GHz | 22.37 GFLOPS | **58.82 GFLOPS** | **126×** |
| WSL2 | i7-1185G7 @ 4.8 GHz | 31.74 GFLOPS | **59.11 GFLOPS** | **100×** |

### vs NumPy (OpenBLAS)

| Matrix Size | This Library | NumPy | Ratio |
|-------------|--------------|-------|-------|
| 512×512 | 49 GFLOPS | 79 GFLOPS | 62% |
| 1024×1024 | 55 GFLOPS | 112 GFLOPS | 49% |

NumPy/OpenBLAS represents 20+ years of hand-tuned assembly. This implementation demonstrates the same techniques built from scratch in Rust.

## Usage
```rust
use matmul::{multiply, multiply_parallel};

// Single-threaded (auto-selects best kernel for your CPU)
let a = vec![1.0f64; 1024 * 1024];
let b = vec![1.0f64; 1024 * 1024];
let mut c = vec![0.0f64; 1024 * 1024];

multiply(&a, &b, &mut c, 1024, 1024, 1024);

// Multi-threaded
multiply_parallel(&a, &b, &mut c, 1024, 1024, 1024, 4);
```

## What's Inside

**SIMD Kernels:**
- 4×4 AVX2 (28× speedup)
- 12×4 AVX2 (33× speedup)  
- 8×8 AVX-512 (39× speedup)

**Optimizations:**
- Cache blocking tuned for L1/L2
- Matrix packing for sequential access
- FMA (fused multiply-add) instructions
- Adaptive threading (scales down for small matrices)

## Project Structure
```
src/
├── kernels/          # SIMD microkernels (4×4, 12×4, 8×8)
├── blocked/          # Cache-blocked GEMM implementations
├── threaded/         # Multi-threaded wrappers
├── matrix/           # Naive implementations, transpose
└── lib.rs            # Public API
```

## Building
```bash
cargo build --release
cargo test
cargo bench
```

## Requirements

- Rust 1.70+
- AVX2 support (Intel Haswell+ / AMD Excavator+)
- AVX-512 for 8×8 kernel (Intel Skylake-X+ / 11th gen+)

## Key Insight

The Mac at 1.4 GHz achieves nearly identical performance to WSL2 at 4.8 GHz:

- **macOS**: 42.0 GFLOPS per GHz
- **WSL2**: 11.5 GFLOPS per GHz

3.6× efficiency difference due to thermal management, native OS vs virtualization, and memory subsystem behavior. Understanding *why* performance differs matters as much as the raw numbers.

## References

- [Optimization Journey](docs/OPTIMIZATION_JOURNEY.md) - Deep dive into each optimization step with performance analysis
- [Goto & van de Geijn - Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_final.pdf)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)

## License

MIT