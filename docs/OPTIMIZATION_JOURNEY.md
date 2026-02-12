# Matrix Multiplication: From Cache Locality to SIMD - A Complete Journey

## Executive Summary

This document chronicles a complete optimization journey from naive implementation to high-performance SIMD matrix multiplication, achieving:
- 10× speedup from loop reordering (cache optimization)
- 126× total speedup with SIMD, blocking, and multi-threading
- 59 GFLOPS on consumer hardware
- 62% of highly-optimized NumPy performance

All optimizations are explained with performance data, hardware metrics, and theoretical analysis.

---

## Part 1: The Cache Locality Discovery

### The Original Experiment

**Simple loop reordering (i-j-k to i-k-j) achieved 9.5× speedup** with zero additional FLOPs or memory. This demonstrated that memory access patterns often matter more than algorithmic complexity on modern CPUs.

#### Implementation Details
- Language: Rust (with --release optimizations)
- Matrix sizes: 256×256, 512×512, 1024×1024
- Hardware: Modern x86-64 CPU (~4 GHz)
- Measurement: Hardware performance counters (perf)

#### Two Implementations

**Version 1: Naive (i-j-k loop ordering)**
```rust
for i in 0..m {
    for j in 0..n {
        for p in 0..k {
            c[i*n + j] += a[i*k + p] * b[p*n + j];
        }
    }
}
```

**Version 2: Cache-aware (i-k-j loop ordering)**
```rust
for i in 0..m {
    for p in 0..k {
        for j in 0..n {
            c[i*n + j] += a[i*k + p] * b[p*n + j];
        }
    }
}
```

**Only difference**: Order of the k and j loops

### Benchmark Results - Phase 1

| Matrix Size | i-j-k Time | i-k-j Time | Speedup | i-j-k GFLOPS | i-k-j GFLOPS |
|-------------|------------|------------|---------|--------------|--------------|
| 256×256     | 33.90 ms   | 4.26 ms    | 7.96×   | 0.99         | 7.88         |
| 512×512     | 301.86 ms  | 33.05 ms   | 9.13×   | 0.89         | 8.12         |
| 1024×1024   | 2812.44 ms | 294.83 ms  | 9.54×   | 0.76         | 7.28         |

#### Key Observations

1. **Speedup increases with matrix size**
   - Larger matrices create more cache pressure
   - Loop ordering matters more at scale

2. **i-j-k performance degrades with size**
   - Cache thrashing gets exponentially worse
   - Performance drops from 0.99 to 0.76 GFLOPS

3. **i-k-j stays consistent**
   - Good cache locality scales
   - Stable ~7-8 GFLOPS across sizes

---

## Part 2: Understanding the Memory Hierarchy

### CPU Cache Structure
```
Level | Size    | Latency | Bandwidth
------|---------|---------|------------
L1    | 32 KB   | ~4 cyc  | ~1 TB/s
L2    | 256 KB  | ~12 cyc | ~500 GB/s
L3    | 6 MB    | ~40 cyc | ~200 GB/s
RAM   | 8-16 GB | ~200 cyc| ~40 GB/s
```

**Key insight**: A RAM access is 50× slower than L1 cache hit.

### Memory Access Patterns Explained

**i-j-k ordering (BAD):**
```
Inner loop (k): accesses B[k,j] where k changes
Result: B[0,j], B[1,j], B[2,j], ...
Stride: n elements (8KB for n=1024)
Cache line usage: 1/8 (load 64 bytes, use 8)
Miss rate: ~45%
```

**i-k-j ordering (GOOD):**
```
Inner loop (j): accesses B[k,j] where j changes  
Result: B[k,0], B[k,1], B[k,2], ...
Stride: 1 element (sequential)
Cache line usage: 8/8 (load 64 bytes, use all)
Miss rate: ~12%
```

### Performance Counter Analysis

| Metric | i-j-k (naive) | i-k-j (cache-aware) | Improvement |
|--------|---------------|---------------------|-------------|
| Time (1024×1024) | 2812 ms | 295 ms | 9.54× faster |
| GFLOPS | 0.76 | 7.28 | 9.6× higher |
| L1 miss rate | ~45% | ~12% | 3.8× fewer misses |
| IPC | ~0.5 | ~2.7 | 5.4× more efficient |

---

## Part 3: Cache Blocking - The Next Level

After achieving 7-8 GFLOPS with loop reordering, the next bottleneck was working set size exceeding cache capacity.

### The Problem: Working Set Too Large

For 1024×1024 matrices:
```
Total working set: 3 × 1024² × 8 bytes = 24 MB
L3 cache: 6 MB
Result: Constant cache eviction
```

### Solution: Multi-Level Cache Blocking

#### KC Blocking (L1 Cache)

**Goal**: Keep working set in L1 cache (32 KB)
```rust
let KC = 256;  // Tuned for L1

for kk in (0..k).step_by(KC) {
    let k_block = min(KC, k - kk);
    // Process only KC elements of k dimension
    // Working set: (M+N) × KC ≈ 16KB
}
```

#### MC Blocking (L2 Cache)

**Goal**: Reuse data from L2 cache (256 KB)
```rust
let MC = 120;  // Tuned for L2

for kk in (0..k).step_by(KC) {
    for ii in (0..m).step_by(MC) {
        // Pack MC × KC chunk of A → stays in L2
        pack_a_panel(...);
        
        for j in (0..n).step_by(4) {
            // Reuse A from L2 for all columns!
        }
    }
}
```

### Data Packing Optimization

**Problem**: Original row-major layout causes strided access

**Solution**: Reorganize into column-major groups for sequential memory access.

---

## Part 4: SIMD Vectorization

After cache blocking achieved ~12 GFLOPS, the next bottleneck was scalar operations.

### The SIMD Advantage

**Scalar code**: 1 operation per instruction
**AVX2**: 4 operations per instruction (256-bit)
**AVX-512**: 8 operations per instruction (512-bit)

### Kernel Implementations

#### 4×4 AVX2 Kernel
```rust
// Process 4 rows × 4 columns per call
let mut c0 = _mm256_loadu_pd(c_ptr);  // Load 4 doubles
let a0 = _mm256_broadcast_sd(a_val);   // Broadcast scalar
c0 = _mm256_fmadd_pd(a0, b_vec, c0);  // 4 FMAs at once
```

#### 12×4 AVX2 Kernel
- Process 12 rows × 4 columns per call
- 12 C accumulators for better instruction pipelining

#### 8×8 AVX-512 Kernel
```rust
// Process 8 rows × 8 columns per call
let mut c0 = _mm512_loadu_pd(c_ptr);  // Load 8 doubles
let a0 = _mm512_set1_pd(a_val);        // Broadcast scalar
c0 = _mm512_fmadd_pd(a0, b_vec, c0);  // 8 FMAs at once
```

### FMA (Fused Multiply-Add)

**Critical discovery**: Must explicitly enable FMA target feature
```rust
#[target_feature(enable = "avx2,fma")]
```

This single change gave **2× improvement** - from separate multiply + add to fused instruction.

---

## Part 5: Multi-Threading

After achieving 37 GFLOPS single-threaded, the final optimization was parallelization.

### Adaptive Threading Strategy

Thread count scales with problem size to avoid overhead on small matrices:
```rust
fn choose_thread_count(m: usize, n: usize, k: usize, max_threads: usize) -> usize {
    let flops = 2.0 * (m * n * k) as f64;

    if flops < 100_000_000.0 {      // < 100M FLOPs
        1  // Threading overhead not worth it
    } else if flops < 300_000_000.0 { // < 300M FLOPs
        2
    } else {
        max_threads
    }
}
```

### Row-Based Parallelization

Each thread processes a subset of rows independently:
- No synchronization needed (disjoint output regions)
- Each thread calls the blocked GEMM on its row range
- B matrix is shared read-only across threads

---

## Part 6: Complete Performance Results

### Intel i7-1185G7 @ 4.8GHz (11th Gen, Tiger Lake, WSL2)

| Implementation | 256×256 | 512×512 | 1024×1024 | Avg Speedup |
|---------------|---------|---------|-----------|-------------|
| Naive (i-j-k) | 0.73 GFLOPS | 0.55 GFLOPS | 0.59 GFLOPS | 1× |
| Scalar (i-k-j) | 6.93 GFLOPS | 5.02 GFLOPS | 5.12 GFLOPS | 9× |
| 4×4 AVX2 | 18.54 GFLOPS | 14.66 GFLOPS | 19.27 GFLOPS | 28× |
| 4×4 AVX2 MT | 14.44 GFLOPS | 22.20 GFLOPS | 42.63 GFLOPS | 44× |
| 12×4 AVX2 | 21.27 GFLOPS | 17.73 GFLOPS | 21.89 GFLOPS | 33× |
| 12×4 AVX2 MT | 19.17 GFLOPS | 25.16 GFLOPS | 46.46 GFLOPS | 50× |
| 8×8 AVX-512 | 15.04 GFLOPS | 24.17 GFLOPS | 31.74 GFLOPS | 39× |
| **8×8 AVX-512 MT** | **21.74 GFLOPS** | **34.18 GFLOPS** | **59.11 GFLOPS** | **64×** |

**Peak Performance**: 59.11 GFLOPS (100× speedup over naive)

**Threading Efficiency (8×8 AVX-512, 1024×1024):**
- Single-threaded: 31.74 GFLOPS
- Multi-threaded (4 cores): 59.11 GFLOPS
- Scaling: 1.86× (47% efficiency) - Good for memory-bound workloads

### Intel i5-8257U @ 1.4GHz (8th Gen, Coffee Lake, macOS)

| Implementation | 256×256 | 512×512 | 1024×1024 | Avg Speedup |
|---------------|---------|---------|-----------|-------------|
| Naive (i-j-k) | 0.79 GFLOPS | 0.77 GFLOPS | 0.47 GFLOPS | 1× |
| Scalar (i-k-j) | 6.95 GFLOPS | 6.77 GFLOPS | 5.06 GFLOPS | 9× |
| 4×4 AVX2 | 19.43 GFLOPS | 19.22 GFLOPS | 16.43 GFLOPS | 28× |
| 4×4 AVX2 MT | 18.71 GFLOPS | 29.95 GFLOPS | 52.48 GFLOPS | 58× |
| 12×4 AVX2 | 21.33 GFLOPS | 20.77 GFLOPS | 22.37 GFLOPS | 34× |
| **12×4 AVX2 MT** | **20.45 GFLOPS** | **32.80 GFLOPS** | **58.82 GFLOPS** | **65×** |

**Peak Performance**: 58.82 GFLOPS (126× speedup over naive!)

**Threading Efficiency (12×4 AVX2, 1024×1024):**
- Single-threaded: 22.37 GFLOPS
- Multi-threaded (4 cores): 58.82 GFLOPS
- Scaling: 2.63× (66% efficiency) - Excellent!

### Cross-Platform Comparison

| Platform | CPU | Clock | Best ST | Best MT | vs Naive |
|----------|-----|-------|---------|---------|----------|
| macOS | i5-8257U | 1.4 GHz | 22.37 GFLOPS | **58.82 GFLOPS** | **126×** |
| WSL2 | i7-1185G7 | 4.8 GHz | 31.74 GFLOPS | **59.11 GFLOPS** | **100×** |

**Key Insight**: The Mac at 1.4 GHz achieves nearly identical performance to WSL2 at 4.8 GHz!

**Performance per GHz:**
- macOS: 58.82 / 1.4 = **42.0 GFLOPS/GHz**
- WSL2: 59.11 / 4.8 = **12.3 GFLOPS/GHz**

3.4× efficiency difference due to:
- Superior thermal management (sustained turbo)
- Native OS vs virtualization overhead
- Better memory subsystem utilization

### Comparison with NumPy (OpenBLAS)

| Matrix Size | NumPy (ST) | Rust (MT) | Ratio |
|-------------|------------|-----------|-------|
| 256×256 | ~84 GFLOPS | ~32 GFLOPS | 38% |
| 512×512 | ~79 GFLOPS | ~49 GFLOPS | **62%** |
| 1024×1024 | ~112 GFLOPS | ~55 GFLOPS | **49%** |

Achieving **49-62% of production BLAS performance** built entirely from scratch.

---

## Part 7: Optimization Journey Summary

| Stage | Technique | GFLOPS | Speedup | Bottleneck Removed |
|-------|-----------|--------|---------|-------------------|
| 1 | Naive i-j-k | 0.59 | 1× | - |
| 2 | Loop reorder (i-k-j) | 5.12 | 9× | Strided B access |
| 3 | Cache blocking | 12.09 | 20× | Cache thrashing |
| 4 | 4×4 AVX2 | 19.27 | 33× | Scalar operations |
| 5 | 12×4 AVX2 | 21.89 | 37× | Small kernel |
| 6 | 8×8 AVX-512 | 31.74 | 54× | AVX2 width |
| 7 | **+ Multi-threading** | **59.11** | **100×** | **Single core** |

---

## Part 8: Key Lessons Learned

### 1. Optimization Priority
```
Cache optimization:     0.59 → 12.09 GFLOPS (20× gain)
SIMD vectorization:    12.09 → 31.74 GFLOPS (2.6× gain)
Multi-threading:       31.74 → 59.11 GFLOPS (1.9× gain)

Memory > Compute > Parallelism
```

### 2. Threading Efficiency is Memory-Bound

Multi-threading achieves 47-66% efficiency on 4 cores:
- Theoretical max: 4× (perfect scaling)
- Achieved: 1.86-2.63× 
- **Expected for memory-bound workloads**

### 3. Platform Matters More Than Clock Speed

The 1.4 GHz Mac matched the 4.8 GHz WSL2 system because:
- Sustained thermal performance
- Native OS efficiency
- Memory subsystem optimization

### 4. Measure Everything

Used throughout:
- `perf stat` for cache miss rates
- Manual timing for GFLOPS
- Hardware counters for IPC
- NumPy comparison baseline

---

## References

1. Goto & van de Geijn (2008) - "Anatomy of High-Performance Matrix Multiplication"
2. Van Zee & van de Geijn (2015) - "BLIS: A Framework for Rapidly Instantiating BLAS Functionality"
3. Intel Intrinsics Guide
4. Agner Fog's Optimization Manuals

---

**Date**: February 2026  
**Author**: Arun  
**Repository**: [github.com/ArunTamil21/matmul-simd](https://github.com/ArunTamil21/matmul-simd)