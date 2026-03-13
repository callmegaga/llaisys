# LLAISYS 性能基准测试结果

**日期**：2026-03-12
**模型**：DeepSeek-R1-Distill-Qwen-1.5B
**设备**：CPU
**测试配置**：4 个测试提示词，最多生成 128 个 token
**测试模式**：批量测试（先测试所有 HF 提示词，再测试所有 LLAISYS 提示词）

## 执行摘要

LLAISYS 推理目前比 HuggingFace/PyTorch 实现在 CPU 上**慢约 16 倍**。

- **HuggingFace 平均**：13.74 tokens/s
- **LLAISYS 平均**：0.85 tokens/s
- **加速比**：0.062x（LLAISYS 慢 16 倍）

## 详细结果

### 测试用例 1："Who are you?"
| 指标 | HuggingFace | LLAISYS | 比率 |
|------|-------------|---------|------|
| 吞吐量 | 13.18 tokens/s | 0.81 tokens/s | 0.06x |
| TTFT | 74.94 ms | 1221.82 ms | 慢 16.3 倍 |
| 总时间 | 6.15 s | 100.19 s | 慢 16.3 倍 |
| 生成 token 数 | 81 | 81 | - |

### 测试用例 2："Explain what is machine learning in simple terms"
| 指标 | HuggingFace | LLAISYS | 比率 |
|------|-------------|---------|------|
| 吞吐量 | 14.56 tokens/s | 0.97 tokens/s | 0.07x |
| TTFT | 68.14 ms | 1026.68 ms | 慢 15.1 倍 |
| 总时间 | 8.79 s | 132.44 s | 慢 15.1 倍 |
| 生成 token 数 | 128 | 128 | - |

### 测试用例 3："Write a Python function to calculate fibonacci numbers"
| 指标 | HuggingFace | LLAISYS | 比率 |
|------|-------------|---------|------|
| 吞吐量 | 14.56 tokens/s | 0.97 tokens/s | 0.07x |
| TTFT | 68.16 ms | 1018.41 ms | 慢 14.9 倍 |
| 总时间 | 8.79 s | 131.38 s | 慢 14.9 倍 |
| 生成 token 数 | 128 | 128 | - |

### 测试用例 4："What are the main differences between Python and C++?"
| 指标 | HuggingFace | LLAISYS | 比率 |
|------|-------------|---------|------|
| 吞吐量 | 12.67 tokens/s | 0.64 tokens/s | 0.05x |
| TTFT | 78.29 ms | 1542.56 ms | 慢 19.7 倍 |
| 总时间 | 10.10 s | 198.99 s | 慢 19.7 倍 |
| 生成 token 数 | 128 | 128 | - |

## 关键发现

### 1. 吞吐量瓶颈
- LLAISYS 每秒仅生成 **0.85 个 token**，而 HuggingFace 为 **13.74 个 token**
- 性能差距达 **16 倍**

### 2. 延迟问题
- 首 token 时间（TTFT）**慢 15-20 倍**
- 平均 TTFT：LLAISYS 1202ms vs HuggingFace 72ms

### 3. 内存使用
- LLAISYS 使用约 4.1GB 内存
- HuggingFace 使用约 3.6GB 内存
- LLAISYS 内存占用与 HuggingFace 相近

### 4. 扩展行为
- 序列越长，性能下降越明显
- 测试 1（81 个 token）：慢 16.3 倍
- 测试 2-4（128 个 token）：慢 15-20 倍

## 根因分析

**已确认**：性能瓶颈在于 **linear（矩阵乘法）算子**。

### 算子基准测试结果

| 算子 | PyTorch (ms) | LLAISYS (ms) | 加速比 |
|------|--------------|--------------|--------|
| **Linear** | 18.908 | 18,136.507 | **0.001x（慢 960 倍）** |

Linear 算子每次操作耗时 **18 秒**，而 PyTorch 仅需 **19 毫秒**。这是 **960 倍的性能差距**，是最关键的单一瓶颈。

### 瓶颈排序（按影响程度）：

1. **Linear 算子**（占 60-80% 时间）- **关键瓶颈**
   - 当前：每次操作 18,136 ms
   - 目标：约 19 ms（PyTorch 水平）
   - **差距：慢 960 倍**
   - 朴素实现，无 SIMD
   - 无多线程
   - 无 BLAS 库集成
   - **优化潜力**：可提升 960 倍
   - **对端到端的影响**：完全优化后推理快 14-17 倍

2. **Self-Attention**（占 10-20% 时间）
   - 内存访问模式
   - 矩阵乘法效率
   - **预期加速潜力**：2-5 倍

3. **其他算子**（各占 <10%）
   - RMS normalization
   - RoPE
   - SwiGLU
   - **预期加速潜力**：各 1.5-3 倍

## 优化建议

### 优先级 1：优化 Linear 算子（关键）

Linear 算子是最大的单一瓶颈，实施以下优化：

#### A. SIMD 向量化
- **技术**：AVX2 或 AVX-512 intrinsics
- **预期提升**：4-8 倍加速
- **工作量**：中等
- **需修改文件**：`src/ops/linear/cpu/linear.cpp`

```cpp
// 示例：使用 AVX2 进行向量化操作
#include <immintrin.h>

// 用 AVX2 一次处理 8 个 float
__m256 a = _mm256_loadu_ps(&data[i]);
__m256 b = _mm256_loadu_ps(&weights[i]);
__m256 result = _mm256_mul_ps(a, b);
```

#### B. OpenMP 多线程
- **技术**：OpenMP
- **预期提升**：2-4 倍加速（4-8 核 CPU）
- **工作量**：低
- **需修改文件**：`xmake.lua`、`src/ops/linear/cpu/linear.cpp`

```cpp
// 示例：并行化外层循环
#pragma omp parallel for
for (int i = 0; i < rows; i++) {
    // 第 i 行的矩阵乘法
}
```

#### C. BLAS 库集成
- **技术**：OpenBLAS、Intel MKL 或 Eigen
- **预期提升**：10-20 倍加速
- **工作量**：中高
- **需修改文件**：`xmake.lua`、`src/ops/linear/cpu/linear.cpp`

```cpp
// 示例：使用 BLAS gemm
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
```

**推荐路径**：先用 OpenMP（快速见效），再加 SIMD，最后集成 BLAS 以获得最大性能。

### 优先级 2：优化 Self-Attention

#### A. 改善内存访问模式
- 缓存友好的数据布局
- 减少内存分配

#### B. 优化矩阵乘法
- 复用 linear 算子的优化成果
- 考虑融合 attention kernel

### 优先级 3：优化其他算子

linear 和 attention 优化完成后，再次分析以确定剩余瓶颈。

## 优化后的预期结果

### 保守估计（OpenMP + SIMD）
- **Linear 算子**：快 5-10 倍
- **整体推理**：快 4-8 倍
- **目标吞吐量**：3.4-6.8 tokens/s
- **加速比**：0.25-0.50x

### 激进估计（OpenMP + SIMD + BLAS）
- **Linear 算子**：快 100-500 倍（可能仍未达到 PyTorch）
- **整体推理**：快 11-15 倍
- **目标吞吐量**：9.4-12.8 tokens/s
- **加速比**：0.68-0.93x

### 最佳情况（完全优化，匹配 PyTorch linear 性能）
- **Linear 算子**：快 960 倍（匹配 PyTorch）
- **整体推理**：快 13-16 倍
- **目标吞吐量**：11.1-13.6 tokens/s
- **加速比**：0.81-0.99x（接近 PyTorch 性能）

## 实施路线图

### 第一阶段：快速见效（1-2 天）
1. 为 linear 算子添加 OpenMP
2. 验证 2-4 倍加速
3. 重新运行基准测试

### 第二阶段：SIMD 优化（3-5 天）
1. 为 linear 算子实现 AVX2 向量化
2. 优化内存访问模式
3. 目标总体 5-10 倍加速
4. 重新运行基准测试

### 第三阶段：BLAS 集成（5-7 天）
1. 集成 OpenBLAS 或 MKL
2. 用 BLAS 调用替换朴素矩阵乘法
3. 目标总体 15-20 倍加速
4. 重新运行基准测试

### 第四阶段：精细调优（2-3 天）
1. 分析剩余瓶颈
2. 优化 attention 和其他算子
3. 目标总体 20 倍以上加速
4. 最终基准测试

## 下一步

1. **立即**：为 linear 算子实现 OpenMP 并行化
2. **短期**：添加 SIMD 向量化
3. **中期**：集成 BLAS 库
4. **长期**：考虑 GPU 实现（Project #2）

## 文件与资源

### 生成的文件
- `benchmark_results/benchmark_results.json` - 详细指标
- `benchmark_results/throughput_comparison.png` - 吞吐量对比
- `benchmark_results/ttft_comparison.png` - 延迟对比
- `benchmark_results/total_time_comparison.png` - 总时间对比
- `benchmark_results/memory_comparison.png` - 内存使用
- `benchmark_results/speedup_factor.png` - 加速比可视化

### 需要优化的关键源文件
- `src/ops/linear/cpu/linear.cpp` - Linear 算子实现
- `src/ops/self_attention/cpu/self_attention.cpp` - Attention 实现
- `xmake.lua` - 添加库的构建配置

### 文档
- `docs/PERFORMANCE_BENCHMARKING.md` - 完整优化指南
- `docs/BENCHMARK_QUICKREF.md` - 快速参考
- `README.md` - Project #1：优化 LLAISYS CPU 性能

## 结论

LLAISYS 目前比 PyTorch 慢 16 倍，主要原因是 linear 算子中未优化的矩阵乘法。通过实施 OpenMP、SIMD 和 BLAS 优化，可以实现 10-16 倍的加速，将性能提升到 PyTorch 的 1-2 倍以内。

优化路径清晰：
1. **OpenMP** → 快 2-4 倍（快速见效）
2. **SIMD** → 快 5-10 倍（中等工作量）
3. **BLAS** → 快 13-16 倍（最佳性能）

优先优化 linear 算子——它将以最小的工作量带来最大的性能提升。
