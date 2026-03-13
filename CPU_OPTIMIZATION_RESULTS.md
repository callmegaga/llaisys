# CPU Optimization Results Summary

**Date:** 2026-03-13
**Hardware:** Intel Core Ultra 7 265K (20 cores, 3.9 GHz, AVX2)
**Model:** DeepSeek-R1-Distill-Qwen-1.5B

---

## Linear Operator Performance

| Phase | LLAISYS (ms) | PyTorch (ms) | Speedup vs Baseline | Slowdown vs PyTorch |
|-------|--------------|--------------|---------------------|---------------------|
| **Baseline** | 18,529 | 20.8 | 1.00x | 889x slower |
| **Phase 1: OpenMP** | 17,292 | 18.9 | 1.07x | 914x slower |
| **Phase 2: OpenMP + AVX2** | 13,880 | 22.7 | **1.33x** | 611x slower |

---

## Key Findings

### ✅ Successful Optimizations

**Phase 1 - OpenMP Parallelization:**
- Added `#pragma omp parallel for` to parallelize computation across 20 cores
- Result: 7% improvement (18,529ms → 17,292ms)
- Files modified: `xmake/cpu.lua`, `src/ops/linear/cpu/linear_cpu.cpp`

**Phase 2 - AVX2 SIMD Vectorization:**
- Implemented float specialization with `_mm256_fmadd_ps` (8 floats/cycle)
- Proper horizontal reduction for accumulator
- Result: 25% improvement over baseline (18,529ms → 13,880ms)
- **Total speedup: 1.33x**

### 📊 Analysis

1. **OpenMP alone provided modest gains** - Thread overhead nearly canceled out parallelism benefits
2. **AVX2 SIMD was more effective** - Vectorization provided consistent 20-25% speedup
3. **Still 611x slower than PyTorch** - The naive implementation is fundamentally limited

### ⚠️ Phase 3 (OpenBLAS) - Not Completed

OpenBLAS installation encountered package dependency issues with xmake. This phase would have provided:
- Expected: 100-500x speedup on linear operator
- Target: 9-13 tokens/s end-to-end performance
- Method: Replace manual loops with highly optimized `cblas_sgemm`

---

## Code Changes

### Committed Changes

1. **Phase 1 Commit:** `perf(linear): add OpenMP parallelism to linear operator (Phase 1)`
   - Added OpenMP flags to `xmake/cpu.lua`
   - Parallelized linear operator with flattened loop

2. **Phase 2 Commit:** `perf(linear): add AVX2 SIMD float specialization (Phase 2)`
   - Added AVX2/FMA flags to build config
   - Implemented vectorized float specialization

### Files Modified

- `xmake/cpu.lua` - Build configuration (OpenMP + AVX2 flags)
- `src/ops/linear/cpu/linear_cpu.cpp` - Linear operator implementation

---

## Recommendations

To achieve the target 9-13 tokens/s performance:

1. **Complete OpenBLAS integration** - This is the critical missing piece
   - Alternative: Use Intel MKL or manually implement cache-blocked matrix multiplication
   - Expected impact: 10-15x additional speedup

2. **Optimize other operators** - Linear is not the only bottleneck
   - Profile attention, RMS norm, and other operators
   - Apply similar SIMD + threading optimizations

3. **Consider memory layout optimizations** - Current row-major layout may not be optimal
   - Experiment with different data layouts
   - Reduce memory bandwidth pressure

---

## Conclusion

The optimization achieved a **1.33x speedup** through OpenMP and AVX2, demonstrating that:
- Manual SIMD vectorization works and provides measurable gains
- Multi-threading has diminishing returns without proper workload distribution
- **OpenBLAS/MKL integration is essential** for competitive performance

The current implementation is still 611x slower than PyTorch, confirming that highly optimized BLAS libraries are necessary for production-grade performance.
