# Set shell to bash
set shell := ["bash", "-c"]

# Run all benchmarks
bench: bench-numpy bench-rust

# NumPy baseline
bench-numpy:
    @echo "NumPy Baseline:"
    @source .venv/bin/activate && python /home/arun/matmul-public/python/numpy_benchmark.py

# Rust implementation
bench-rust:
    @echo "Rust Implementation:"
    @cargo run --release

# Compare side-by-side
compare: bench-numpy bench-rust

# Run comparison 3 times to check variance
compare-stable:
    @echo "=== RUN 1 ==="
    @just compare
    @sleep 30
    @echo ""
    @echo "=== RUN 2 ==="
    @just compare
    @sleep 30
    @echo ""
    @echo "=== RUN 3 ==="
    @just compare

# Quick single comparison
quick: compare