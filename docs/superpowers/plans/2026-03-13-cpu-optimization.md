# CPU Linear Operator Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize `linear_cpu.cpp` through three sequential phases (OpenMP → AVX2 → OpenBLAS) to bring LLAISYS from 0.85 tokens/s to 9–13 tokens/s.

**Architecture:** Single file `src/ops/linear/cpu/linear_cpu.cpp` receives all code changes. `xmake/cpu.lua` receives build flag additions inside the `llaisys-ops-cpu` target block only. Each phase is independently verified before proceeding.

**Tech Stack:** C++17, OpenMP, AVX2 intrinsics (`<immintrin.h>`), OpenBLAS (`cblas_sgemm`), xmake build system.

---

## Chunk 1: Baseline + Phase 1 (OpenMP)

### Task 1: Capture baseline benchmark

**Files:**
- Read: `test/benchmark_operators.py`

- [ ] **Step 1: Run operator benchmark and save output**

```bash
cd F:/Project/llaisys
python test/benchmark_operators.py --operators linear 2>&1 | tee benchmark_baseline_operator.txt
```

Expected: prints timing for LLAISYS linear vs PyTorch linear. LLAISYS will be ~960x slower.

- [ ] **Step 2: Run inference benchmark and save output**

```bash
python test/benchmark_inference.py 2>&1 | tee benchmark_baseline_inference.txt
```

Expected: LLAISYS ~0.85 tokens/s, HuggingFace ~13.74 tokens/s.

---

### Task 2: Add OpenMP flags to build config

**Files:**
- Modify: `xmake/cpu.lua` — inside `llaisys-ops-cpu` target block only

- [ ] **Step 1: Open `xmake/cpu.lua` and locate the `llaisys-ops-cpu` target block**

Add OpenMP flags before `on_install`. On non-Windows, also remove `-Wno-unknown-pragmas` from this target since OpenMP pragmas will now be recognized:

```lua
target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")  -- removed -Wno-unknown-pragmas: omp pragmas now recognized
    end

    -- Phase 1: OpenMP
    if is_plat("windows") then
        add_cxflags("/openmp")  -- MSVC links OpenMP automatically; no ldflags needed
    else
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()
```

- [ ] **Step 2: Verify build succeeds (no code changes yet)**

```bash
cd F:/Project/llaisys
xmake -v 2>&1 | tail -20
```

Expected: build succeeds. If MSVC warns about `/openmp` being unknown, check MSVC version (requires VS 2019+).

---

### Task 3: Add OpenMP to linear_cpu.cpp

**Files:**
- Modify: `src/ops/linear/cpu/linear_cpu.cpp`

- [ ] **Step 1: Replace the file contents with the OpenMP version**

The key changes:
1. Add `#include <omp.h>`
2. Move `#pragma omp parallel for schedule(static) collapse(2)` onto the outer two loops
3. Change F32 inner loop to use a local accumulator (avoids repeated memory writes)

Full file:

```cpp
#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t input_rows, size_t input_cols, size_t weight_rows) {
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < input_rows; i++) {
        for (size_t j = 0; j < weight_rows; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                float acc = llaisys::utils::cast<float>(bias[j]);
                for (size_t k = 0; k < input_cols; k++) {
                    acc += llaisys::utils::cast<float>(in[i * input_cols + k]) *
                           llaisys::utils::cast<float>(weight[j * input_cols + k]);
                }
                out[i * weight_rows + j] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = bias[j];
                for (size_t k = 0; k < input_cols; k++) {
                    acc += in[i * input_cols + k] * weight[j * input_cols + k];
                }
                out[i * weight_rows + j] = acc;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias,
            llaisysDataType_t type, size_t input_rows, size_t input_cols,
            size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
```

- [ ] **Step 2: Build**

```bash
cd F:/Project/llaisys
xmake && xmake install
```

Expected: clean build, no errors.

- [ ] **Step 3: Run correctness test**

```bash
python test/ops/linear.py
```

Expected: all tests pass. If any fail, the OpenMP pragma is causing a race condition — check that `acc` is declared inside the loop body (thread-local).

- [ ] **Step 4: Run operator benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_operators.py --operators linear 2>&1 | tee benchmark_phase1_operator.txt
```

Expected: 8–15x speedup over baseline on linear operator.

- [ ] **Step 5: Run inference benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_inference.py 2>&1 | tee benchmark_phase1_inference.txt
```

Expected: ~1.7–3.4 tokens/s (up from 0.85).

- [ ] **Step 6: Commit**

```bash
cd F:/Project/llaisys
git add src/ops/linear/cpu/linear_cpu.cpp xmake/cpu.lua
git commit -m "perf(linear): add OpenMP parallelism to linear operator (Phase 1)"
```

---

## Chunk 2: Phase 2 (AVX2 SIMD)

### Task 4: Add AVX2 flags to build config

**Files:**
- Modify: `xmake/cpu.lua` — inside `llaisys-ops-cpu` target block only

- [ ] **Step 1: Add AVX2 flags after the OpenMP block**

```lua
    -- Phase 2: AVX2 + FMA
    if is_plat("windows") then
        add_cxflags("/arch:AVX2")  -- implicitly enables FMA on MSVC
    else
        add_cxflags("-mavx2", "-mfma")
    end
```

- [ ] **Step 2: Verify build still succeeds**

```bash
cd F:/Project/llaisys
xmake -v 2>&1 | tail -5
```

Expected: clean build.

---

### Task 5: Add AVX2 float specialization to linear_cpu.cpp

**Files:**
- Modify: `src/ops/linear/cpu/linear_cpu.cpp`

- [ ] **Step 1: Add `#include <immintrin.h>` and a `float` explicit specialization**

The generic template handles BF16/F16 (unchanged from Phase 1). Add a `float` specialization after the generic template that uses AVX2 for the inner `k` loop.

Full file after this step:

```cpp
#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Generic template: handles BF16 and FP16 with OpenMP
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t input_rows, size_t input_cols, size_t weight_rows) {
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < input_rows; i++) {
        for (size_t j = 0; j < weight_rows; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                float acc = llaisys::utils::cast<float>(bias[j]);
                for (size_t k = 0; k < input_cols; k++) {
                    acc += llaisys::utils::cast<float>(in[i * input_cols + k]) *
                           llaisys::utils::cast<float>(weight[j * input_cols + k]);
                }
                out[i * weight_rows + j] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = bias[j];
                for (size_t k = 0; k < input_cols; k++) {
                    acc += in[i * input_cols + k] * weight[j * input_cols + k];
                }
                out[i * weight_rows + j] = acc;
            }
        }
    }
}

// Float specialization: OpenMP + AVX2 FMA
template <>
void linear_<float>(float *out, const float *in, const float *weight,
                    const float *bias, size_t input_rows, size_t input_cols,
                    size_t weight_rows) {
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < input_rows; i++) {
        for (size_t j = 0; j < weight_rows; j++) {
            const float *in_row = in + i * input_cols;
            const float *w_row  = weight + j * input_cols;

            __m256 vacc = _mm256_setzero_ps();
            size_t k = 0;
            for (; k + 8 <= input_cols; k += 8) {
                __m256 va = _mm256_loadu_ps(in_row + k);
                __m256 vb = _mm256_loadu_ps(w_row + k);
                vacc = _mm256_fmadd_ps(va, vb, vacc);
            }

            // Horizontal sum: add upper and lower 128-bit lanes, then hadd twice
            __m128 lo  = _mm256_castps256_ps128(vacc);
            __m128 hi  = _mm256_extractf128_ps(vacc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            float acc = _mm_cvtss_f32(sum);

            // Scalar remainder
            for (; k < input_cols; k++) {
                acc += in_row[k] * w_row[k];
            }

            out[i * weight_rows + j] = acc + bias[j];
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias,
            llaisysDataType_t type, size_t input_rows, size_t input_cols,
            size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
```

- [ ] **Step 2: Build**

```bash
cd F:/Project/llaisys
xmake && xmake install
```

Expected: clean build. If MSVC complains about `_mm256_fmadd_ps` not found, confirm `/arch:AVX2` is in the `llaisys-ops-cpu` block (not `llaisys-device-cpu`).

- [ ] **Step 3: Run correctness test**

```bash
python test/ops/linear.py
```

Expected: all tests pass. If numerical results differ slightly, check the horizontal reduction — a wrong lane extraction produces incorrect sums.

- [ ] **Step 4: Run operator benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_operators.py --operators linear 2>&1 | tee benchmark_phase2_operator.txt
```

Expected: additional 4–6x over Phase 1 result.

- [ ] **Step 5: Run inference benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_inference.py 2>&1 | tee benchmark_phase2_inference.txt
```

Expected: ~4.3–8.5 tokens/s.

- [ ] **Step 6: Commit**

```bash
git add src/ops/linear/cpu/linear_cpu.cpp xmake/cpu.lua
git commit -m "perf(linear): add AVX2 SIMD float specialization (Phase 2)"
```

---

## Chunk 3: Phase 3 (OpenBLAS) + Final Comparison

### Task 6: Install OpenBLAS and update build config

**Files:**
- Modify: `xmake/cpu.lua`

- [ ] **Step 1: Add OpenBLAS package requirement**

Add `add_requires("openblas")` at the top of `xmake/cpu.lua` (before any target blocks):

```lua
add_requires("openblas")
```

Then inside the `llaisys-ops-cpu` target block, add:

```lua
    -- Phase 3: OpenBLAS (F32 path only)
    add_packages("openblas")
```

- [ ] **Step 2: Install OpenBLAS via xmake**

```bash
cd F:/Project/llaisys
xmake require openblas
```

Expected: xmake downloads and builds OpenBLAS. On Windows this pulls prebuilt binaries. May take 1–2 minutes.

- [ ] **Step 3: Verify build succeeds**

```bash
xmake -v 2>&1 | tail -10
```

Expected: clean build with OpenBLAS linked.

---

### Task 7: Replace F32 path with cblas_sgemm

**Files:**
- Modify: `src/ops/linear/cpu/linear_cpu.cpp`

- [ ] **Step 1: Add `#include <cblas.h>` and replace the float specialization**

The generic template (BF16/FP16) stays unchanged. Replace only the `float` specialization:

```cpp
#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cblas.h>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Generic template: handles BF16 and FP16 with OpenMP + AVX2 cast-to-float
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t input_rows, size_t input_cols, size_t weight_rows) {
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t i = 0; i < input_rows; i++) {
        for (size_t j = 0; j < weight_rows; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                float acc = llaisys::utils::cast<float>(bias[j]);
                for (size_t k = 0; k < input_cols; k++) {
                    acc += llaisys::utils::cast<float>(in[i * input_cols + k]) *
                           llaisys::utils::cast<float>(weight[j * input_cols + k]);
                }
                out[i * weight_rows + j] = llaisys::utils::cast<T>(acc);
            } else {
                T acc = bias[j];
                for (size_t k = 0; k < input_cols; k++) {
                    acc += in[i * input_cols + k] * weight[j * input_cols + k];
                }
                out[i * weight_rows + j] = acc;
            }
        }
    }
}

// Float specialization: OpenBLAS cblas_sgemm
// Computes: out = in @ weight.T + bias
// weight is [N, K] row-major, so ldb=K even with CblasTrans
template <>
void linear_<float>(float *out, const float *in, const float *weight,
                    const float *bias, size_t input_rows, size_t input_cols,
                    size_t weight_rows) {
    int M = static_cast<int>(input_rows);
    int N = static_cast<int>(weight_rows);
    int K = static_cast<int>(input_cols);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, in, K, weight, K, 0.0f, out, N);

    // Add bias (cblas_sgemm does not handle bias)
#pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] += bias[j];
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias,
            llaisysDataType_t type, size_t input_rows, size_t input_cols,
            size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
```

- [ ] **Step 2: Build**

```bash
cd F:/Project/llaisys
xmake && xmake install
```

Expected: clean build. If `cblas.h` is not found, check that `add_packages("openblas")` is inside `llaisys-ops-cpu` and `add_requires("openblas")` is at the top of `xmake/cpu.lua`.

- [ ] **Step 3: Run correctness test**

```bash
python test/ops/linear.py
```

Expected: all tests pass. If F32 results differ, check that `cblas_sgemm` arguments match the spec — particularly `CblasTrans` for weight and `ldb=K`.

- [ ] **Step 4: Run operator benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_operators.py --operators linear 2>&1 | tee benchmark_phase3_operator.txt
```

Expected: additional 2–4x over Phase 2. Linear operator should now be within 10–50x of PyTorch (down from 960x).

- [ ] **Step 5: Run inference benchmark**

```bash
OMP_NUM_THREADS=20 python test/benchmark_inference.py 2>&1 | tee benchmark_phase3_inference.txt
```

Expected: ~9–13 tokens/s (up from 0.85).

- [ ] **Step 6: Commit**

```bash
git add src/ops/linear/cpu/linear_cpu.cpp xmake/cpu.lua
git commit -m "perf(linear): replace F32 path with OpenBLAS cblas_sgemm (Phase 3)"
```

---

### Task 8: Final comparison report

**Files:**
- Read: `benchmark_baseline_*.txt`, `benchmark_phase*.txt`

- [ ] **Step 1: Print comparison table**

```bash
echo "=== Operator Benchmark Comparison ===" && \
echo "--- Baseline ---" && cat benchmark_baseline_operator.txt && \
echo "--- Phase 1 (OpenMP) ---" && cat benchmark_phase1_operator.txt && \
echo "--- Phase 2 (AVX2) ---" && cat benchmark_phase2_operator.txt && \
echo "--- Phase 3 (OpenBLAS) ---" && cat benchmark_phase3_operator.txt
```

- [ ] **Step 2: Print inference comparison**

```bash
echo "=== Inference Benchmark Comparison ===" && \
echo "--- Baseline ---" && cat benchmark_baseline_inference.txt && \
echo "--- Phase 1 ---" && cat benchmark_phase1_inference.txt && \
echo "--- Phase 2 ---" && cat benchmark_phase2_inference.txt && \
echo "--- Phase 3 ---" && cat benchmark_phase3_inference.txt
```

- [ ] **Step 3: Run full operator test suite to confirm no regressions**

```bash
python test/ops/linear.py
python test/ops/embedding.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py
```

Expected: all operator tests pass.

- [ ] **Step 4: Commit benchmark results**

```bash
git add benchmark_*_operator.txt benchmark_*_inference.txt
git commit -m "test(benchmark): add before/after benchmark results for CPU optimization"
```
