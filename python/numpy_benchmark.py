import numpy as np
import time
import os

# Force single thread
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

print("=== NumPy Single-Threaded Benchmark ===\n")

sizes = [256, 512, 1024]

for size in sizes:
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # Warmup
    C = A @ B
    
    # Timed runs
    times = []
    for _ in range(5):
        start = time.time()
        C = A @ B
        times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    ops = 2 * size**3
    gflops = ops / avg_time / 1e9
    
    print(f"{size}x{size}: {avg_time*1000:.2f} ms ({gflops:.2f} GFLOPS)")