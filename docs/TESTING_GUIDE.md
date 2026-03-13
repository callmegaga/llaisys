# Testing and Comparing CPU Optimization Changes

## Quick Test Commands

### 1. Verify Correctness (Must Pass!)

```bash
cd F:/Project/llaisys

# Test linear operator specifically
python test/ops/linear.py

# Test all operators
python test/ops/linear.py
python test/ops/embedding.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py

# Test full inference pipeline
python test/test_infer.py --test
```

### 2. Performance Benchmarks

```bash
# Operator-level benchmark (linear operator only)
OMP_NUM_THREADS=20 python test/benchmark_operators.py --operators linear

# Full inference benchmark (end-to-end)
OMP_NUM_THREADS=20 python test/benchmark_inference.py
```

---

## Compare Before/After Results

### View All Benchmark Results

```bash
cd F:/Project/llaisys

# Operator benchmarks
echo "=== BASELINE ==="
cat benchmark_baseline_operator.txt

echo "=== PHASE 1 (OpenMP) ==="
cat benchmark_phase1_operator.txt

echo "=== PHASE 2 (OpenMP + AVX2) ==="
cat benchmark_phase2_operator.txt

echo "=== PHASE 3 (OpenMP + AVX2 + Intel MKL) ==="
cat benchmark_phase3_operator.txt
```

### Quick Comparison Table

| Phase | Linear Time (ms) | Speedup vs Baseline | vs PyTorch |
|-------|------------------|---------------------|------------|
| Baseline | 18,529 | 1.00x | 889x slower |
| Phase 1 (OpenMP) | 17,292 | 1.07x | 914x slower |
| Phase 2 (AVX2) | 13,880 | 1.33x | 611x slower |
| Phase 3 (MKL) | ??? | ???x | ???x slower |

---

## Detailed Testing Steps

### Step 1: Verify Build

```bash
cd F:/Project/llaisys
xmake
```

Expected: Clean build with no errors

### Step 2: Test Correctness

```bash
# This MUST pass - verifies numerical correctness
python test/ops/linear.py
```

Expected output:
```
Testing Ops.linear on cpu
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f32>
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <f16>
   out (2, 3), x (2, 4), w (3, 4), bias True, dtype <bf16>
   ...
Test passed!
```

### Step 3: Test Full Inference

```bash
# Verify the model can generate text correctly
python test/test_infer.py --test
```

Expected: Model generates text matching PyTorch output

### Step 4: Benchmark Performance

```bash
# Set thread count for reproducible results
export OMP_NUM_THREADS=20  # or: set OMP_NUM_THREADS=20 on Windows

# Operator benchmark
python test/benchmark_operators.py --operators linear

# Inference benchmark
python test/benchmark_inference.py
```

---

## What Each Phase Changed

### Phase 1: OpenMP Parallelization
- **Files:** `xmake/cpu.lua`, `src/ops/linear/cpu/linear_cpu.cpp`
- **Change:** Added `#pragma omp parallel for` to use 20 CPU cores
- **Expected:** 2-4x speedup (actual: 1.07x due to overhead)

### Phase 2: AVX2 SIMD Vectorization
- **Files:** Same as Phase 1
- **Change:** Added float specialization with `_mm256_fmadd_ps` (8 floats/cycle)
- **Expected:** 5-10x total speedup (actual: 1.33x)

### Phase 3: Intel MKL Integration
- **Files:** Same as Phase 1
- **Change:** Replaced manual loops with `cblas_sgemm` from Intel MKL
- **Expected:** 50-200x speedup (highly optimized BLAS)
- **Target:** 9-13 tokens/s end-to-end

---

## Troubleshooting

### If Tests Fail

1. **Numerical errors:** Check if MKL is linked correctly
   ```bash
   ldd bin/llaisys.dll  # Linux
   # or check dependencies on Windows
   ```

2. **Build errors:** Verify MKL paths in `xmake/cpu.lua`
   ```bash
   ls D:/ProgramData/miniconda3/Library/include/mkl.h
   ls D:/ProgramData/miniconda3/Library/lib/mkl_core_dll.lib
   ```

3. **Performance regression:** Check `OMP_NUM_THREADS` is set
   ```bash
   echo $OMP_NUM_THREADS  # Should be 20
   ```

---

## View Complete Results

```bash
# See the summary document
cat CPU_OPTIMIZATION_RESULTS.md

# Or view in your editor
code CPU_OPTIMIZATION_RESULTS.md
```

---

## Expected Final Results (Phase 3)

With Intel MKL, you should see:
- **Linear operator:** ~30-100ms (down from 18,529ms)
- **Speedup:** 185-617x faster than baseline
- **End-to-end:** 9-13 tokens/s (close to PyTorch's 13.74 tokens/s)
- **vs PyTorch:** Within 1.5-2x of PyTorch performance

If you see these numbers, the optimization was successful! 🎉
