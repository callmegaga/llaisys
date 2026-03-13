# CPU Optimization Design: Linear Operator

**Date:** 2026-03-13
**Project:** LLAISYS — Project #1: Optimize LLAISYS for CPU
**Target:** `src/ops/linear/cpu/linear_cpu.cpp`

---

## Problem Statement

The LLAISYS linear operator is 960x slower than PyTorch's equivalent, causing a 16x
end-to-end inference slowdown (0.85 tokens/s vs 13.74 tokens/s). The root cause is a
naive triple-nested loop with no parallelism or vectorization.

Hardware: Intel Core Ultra 7 265K — 20 cores, 3.9 GHz, AVX2 (Arrow Lake, no AVX-512).

---

## Goals

- Optimize `linear_cpu.cpp` using three sequential techniques
- Measure speedup after each phase with `benchmark_operators.py`
- Maintain correctness: all `test/ops/linear.py` tests must pass at every phase
- Final target: 9–13 tokens/s end-to-end (close to PyTorch's 13.74 tokens/s)

---

## Architecture

### Why the `j` loop?

During autoregressive generation, `input_rows` (M) = 1. Parallelizing over M gives no
benefit. The `weight_rows` (N) dimension is 1536–8960 for Qwen2-1.5B and each output
element `out[i][j]` is independent — making it the correct loop to parallelize and
vectorize.

### Data flow per phase

```
Phase 1 (OpenMP):   j loop → 20 threads, each computing one output row
Phase 2 (AVX2):     k loop → 8 floats/cycle via _mm256_fmadd_ps
Phase 3 (OpenBLAS): F32 path → cblas_sgemm replaces manual loops
                    BF16/F16 path → keeps Phase 1+2 (no native BLAS support)
```

---

## Phase 1: OpenMP Parallelism

### Change: `linear_cpu.cpp`

Add `#pragma omp parallel for schedule(static)` on the `j` loop inside `linear_<T>`.
Each thread computes an independent subset of output rows. No shared mutable state —
the accumulator `acc` is thread-local.

### Change: `xmake/cpu.lua` (inside `llaisys-ops-cpu` target block only)

```lua
-- Phase 1: OpenMP
if is_plat("windows") then
    add_cxflags("/openmp")  -- MSVC links OpenMP automatically; no ldflags needed
else
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
    -- -Wno-unknown-pragmas can be removed after this phase (omp pragmas now recognized)
end
```

### Expected outcome

- 8–15x speedup on the linear operator (20 cores, realistic efficiency ~50–75%)
- End-to-end: ~1.7–3.4 tokens/s

---

## Phase 2: AVX2 SIMD

### Change: `linear_cpu.cpp` (F32 specialization)

Vectorize the inner `k` loop using AVX2 intrinsics:
- Load 8 floats from `in` and `weight` with `_mm256_loadu_ps`
- Accumulate with `_mm256_fmadd_ps` (fused multiply-add)
- Reduce the 8-lane accumulator: `_mm256_hadd_ps` twice → extract low 128-bit lane → `_mm_hadd_ps` → `_mm_cvtss_f32`
- Scalar fallback for remainder elements when `input_cols % 8 != 0`

BF16/F16 paths: keep existing cast-to-float loop (no native AVX2 BF16 on Arrow Lake),
but benefit from Phase 1 OpenMP parallelism.

### Change: `xmake/cpu.lua` (inside `llaisys-ops-cpu` target block only)

```lua
-- Phase 2: AVX2 + FMA
if is_plat("windows") then
    add_cxflags("/arch:AVX2")  -- /arch:AVX2 on MSVC implicitly enables FMA; no separate flag needed
else
    add_cxflags("-mavx2", "-mfma")
end
```

### Expected outcome

- Additional 4–6x on top of Phase 1 (AVX2 = 8 floats/cycle, realistic with memory BW)
- End-to-end: ~4.3–8.5 tokens/s

---

## Phase 3: OpenBLAS

### Change: `linear_cpu.cpp` (F32 path only)

Replace the manual F32 loops with a single BLAS call:

```cpp
// M=input_rows, N=weight_rows, K=input_cols
// weight is stored as [N, K] (row-major), so ldb=K even with CblasTrans
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            M, N, K, 1.0f, in, K, weight, K, 0.0f, out, N);
// bias addition: separate loop (cblas_sgemm does not handle bias)
for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++)
        out[i * N + j] += bias[j];
```

BF16/F16 paths remain on the Phase 1+2 implementation.

### Change: `xmake/cpu.lua`

```lua
add_packages("openblas")
```

Installation: `xmake require openblas` (xmake pulls prebuilt binaries on Windows).

### Expected outcome

- Additional 2–4x on top of Phase 2 (OpenBLAS uses cache-blocking + micro-kernels)
- End-to-end: ~9–13 tokens/s

---

## Edge Cases

| Case | Handling |
|---|---|
| `bias == nullptr` | Out of scope — `op.cpp` calls `bias->data()` unconditionally; null bias is not supported by the current API |
| `input_cols % 8 != 0` | AVX2 path: scalar loop handles remainder |
| BF16/F16 with OpenBLAS | Falls through to Phase 1+2 path |
| M=1 (single token, typical) | OpenMP parallelizes over N, not M — no wasted threads |

---

## Benchmarking Plan

| Checkpoint | Command |
|---|---|
| Baseline | `python test/benchmark_operators.py --operators linear` |
| After Phase 1 | same + `python test/benchmark_inference.py` |
| After Phase 2 | same + `python test/benchmark_inference.py` |
| After Phase 3 | same + `python test/benchmark_inference.py` |

Set `OMP_NUM_THREADS=20` when running benchmarks for reproducible results across all phases.

Correctness check after each phase: `python test/ops/linear.py`

---

## Files Changed

| File | Change |
|---|---|
| `src/ops/linear/cpu/linear_cpu.cpp` | OpenMP + AVX2 + OpenBLAS |
| `xmake/cpu.lua` | OpenMP flags, AVX2 flags, OpenBLAS package |

No other files are modified.
