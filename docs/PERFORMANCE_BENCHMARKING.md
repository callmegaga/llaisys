# LLAISYS 性能基准测试指南

本指南介绍如何对 LLAISYS 推理框架进行基准测试和性能优化。

## 概述

提供两个基准测试脚本：

1. **`benchmark_inference.py`**：端到端推理性能对比
2. **`benchmark_operators.py`**：单个算子性能分析

## 安装

### 安装依赖

```bash
# 安装基准测试依赖
pip install -r benchmark_requirements.txt

# 确保 LLAISYS 已安装
pip install ./python/
```

## 1. 端到端推理基准测试

### 目的
将 LLAISYS 推理性能与 HuggingFace 实现进行对比，识别整体系统性能差距。

### 快速开始

```bash
# CPU 基础基准测试
python test/benchmark_inference.py --model [模型路径]

# GPU 基准测试
python test/benchmark_inference.py --model [模型路径] --device nvidia

# 自定义测试用例
python test/benchmark_inference.py --model [模型路径] \
    --prompts "什么是 AI？" "解释量子计算" \
    --max_tokens 256
```

### 输出文件

所有结果保存到 `benchmark_results/`（或通过 `--output_dir` 指定目录）：

- `benchmark_results.json`：JSON 格式的详细指标
- `throughput_comparison.png`：Tokens/秒对比图
- `ttft_comparison.png`：首 token 时间对比
- `total_time_comparison.png`：总推理时间对比
- `memory_comparison.png`：内存使用对比
- `speedup_factor.png`：加速比可视化

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **Tokens/秒** | 生成吞吐量 | 越高越好 |
| **TTFT** | 首 token 时间（延迟） | 越低越好 |
| **总时间** | 端到端推理时间 | 越低越好 |
| **内存使用** | 峰值内存消耗 | 越低越好 |
| **加速比** | LLAISYS 时间 / HF 时间 | >1.0 表示 LLAISYS 更快 |

## 2. 算子级别基准测试

### 目的
通过对比单个算子性能，识别哪些算子是性能瓶颈。

### 快速开始

```bash
# 测试所有算子
python test/benchmark_operators.py --device cpu

# 测试指定算子
python test/benchmark_operators.py --device cpu \
    --operators linear self_attention rms_norm

# 更多迭代以获得精确测量
python test/benchmark_operators.py --device cpu \
    --warmup 20 --repeat 200
```

### 支持的算子

- `linear`：矩阵乘法（对性能最关键）
- `rms_norm`：RMS 归一化
- `self_attention`：自注意力机制
- `rope`：旋转位置编码
- `swiglu`：SwiGLU 激活函数

### 输出

结果保存到 `operator_benchmark.json` 并打印到控制台：

```
================================================================================
SUMMARY
================================================================================
Operator             PyTorch (ms)    LLAISYS (ms)    Speedup
--------------------------------------------------------------------------------
linear               12.345          45.678          0.27x
rms_norm             2.345           3.456           0.68x
self_attention       8.901           12.345          0.72x
rope                 1.234           2.345           0.53x
swiglu               3.456           4.567           0.76x
--------------------------------------------------------------------------------
Average Speedup:                                     0.59x
================================================================================
```

## 优化工作流

### 第一步：运行端到端基准测试

```bash
python test/benchmark_inference.py --model [模型路径] --device cpu
```

**分析**：检查加速比。如果 LLAISYS 明显更慢（如 <0.5x），进入算子级别分析。

### 第二步：识别瓶颈算子

```bash
python test/benchmark_operators.py --device cpu --repeat 200
```

**分析**：找出加速比最低的算子。通常 `linear`（矩阵乘法）是最大瓶颈。

### 第三步：分析特定算子

对特定算子进行详细分析：

```bash
# 带性能分析运行算子测试
python test/ops/linear.py --profile --device cpu
```

### 第四步：优化

根据识别的瓶颈，应用以下优化：

#### CPU 优化：
- **SIMD 指令**：使用 AVX2/AVX-512 intrinsics
- **多线程**：添加 OpenMP 并行化
- **BLAS 库**：集成 OpenBLAS、MKL 或 Eigen
- **缓存优化**：改善内存访问模式

#### GPU 优化：
- **CUDA Kernel**：优化 kernel 启动配置
- **cuBLAS**：使用 cuBLAS 进行矩阵运算
- **内存合并**：改善内存访问模式
- **共享内存**：有效利用共享内存

### 第五步：测量提升效果

优化后重新运行基准测试：

```bash
# 重新运行算子基准测试
python test/benchmark_operators.py --device cpu --output operator_benchmark_optimized.json

# 重新运行端到端基准测试
python test/benchmark_inference.py --model [模型路径] --output_dir benchmark_results_optimized
```

**对比**：检查算子级别和端到端基准测试中的加速比提升。

## 优化示例流程

```bash
# 1. 基线测量
python test/benchmark_inference.py --model ~/models/qwen-1.5b
# 结果：0.3x 加速比（LLAISYS 慢 3 倍）

# 2. 识别瓶颈
python test/benchmark_operators.py
# 结果：linear 算子 0.2x（慢 5 倍）

# 3. 优化 linear 算子（实现 SIMD + OpenMP）
# ... 修改代码 ...

# 4. 验证算子提升
python test/benchmark_operators.py --operators linear
# 结果：linear 算子现在 0.8x（慢 1.25 倍）- 提升 4 倍！

# 5. 验证端到端提升
python test/benchmark_inference.py --model ~/models/qwen-1.5b --output_dir results_optimized
# 结果：0.7x 加速比（慢 1.4 倍）- 整体提升 2.3 倍！
```

## 精确基准测试的技巧

1. **关闭其他应用程序**以减少系统噪声
2. **多次运行**并取平均值
3. **使用一致的测试用例**进行前后对比
4. **监控 CPU/GPU 温度** - 热降频会影响结果
5. **禁用节能模式**
6. **使用 release 构建**（`xmake f -m release`）

## 结果解读

### 性能良好
- 加速比 > 0.8x：与 PyTorch 性能差距在 20% 以内
- 加速比 > 1.0x：比 PyTorch 更快（优秀！）

### 需要优化
- 加速比 < 0.5x：慢 2 倍以上（需要显著优化）
- 加速比 < 0.3x：慢 3 倍以上（关键瓶颈）

### 关注重点
1. **Linear 算子**：通常占总计算时间的 60-80%
2. **Self-attention**：占 10-20%
3. **其他算子**：各占 <10%

## 进阶：持续性能追踪

创建脚本追踪性能变化：

```bash
#!/bin/bash
# track_performance.sh

DATE=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="performance_history/$DATE"

python test/benchmark_inference.py \
    --model ~/models/qwen-1.5b \
    --output_dir "$OUTPUT_DIR"

python test/benchmark_operators.py \
    --output "$OUTPUT_DIR/operators.json"

echo "结果已保存到 $OUTPUT_DIR"
```

## 故障排除

### 问题：基准测试崩溃或卡住
- **解决方案**：减少 `--max_tokens` 或使用更短的测试提示词
- **检查**：确保模型正确加载

### 问题：结果不稳定
- **解决方案**：增加 `--warmup` 和 `--repeat` 迭代次数
- **检查**：系统负载和后台进程

### 问题：内存不足
- **解决方案**：使用更小的 batch size 或更短的序列
- **检查**：可用系统/GPU 内存

## 下一步

基准测试完成后：

1. 查看 [README.md 中的 Project #1](../README.md#project-1-optimize-llaisys-for-cpu) 了解优化策略
2. 为瓶颈算子实施优化
3. 重新运行基准测试以衡量提升效果
4. 记录你的优化方案和性能收益

## 有问题？

- 查看主 [README.md](../README.md) 了解项目文档
- 查看 `src/ops/` 中的算子实现
- 参考 PyTorch 实现作为对比
