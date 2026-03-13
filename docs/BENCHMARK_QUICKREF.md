# 性能基准测试快速参考

## 安装
```bash
pip install -r test/benchmark_requirements.txt
```

## 命令

### 端到端推理基准测试
```bash
# CPU 基准测试
python test/benchmark_inference.py --model [模型路径]

# GPU 基准测试
python test/benchmark_inference.py --model [模型路径] --device nvidia

# 自定义提示词
python test/benchmark_inference.py --model [模型路径] \
    --prompts "提示词1" "提示词2" "提示词3"

# 生成更多 token
python test/benchmark_inference.py --model [模型路径] --max_tokens 256
```

### 算子基准测试
```bash
# 所有算子
python test/benchmark_operators.py --device cpu

# 指定算子
python test/benchmark_operators.py --device cpu --operators linear rms_norm

# 更精确的测量（更多迭代次数）
python test/benchmark_operators.py --device cpu --warmup 20 --repeat 200
```

## 输出文件

### 推理基准测试
- `benchmark_results/benchmark_results.json` - 详细指标
- `benchmark_results/*.png` - 对比图表

### 算子基准测试
- `operator_benchmark.json` - 算子耗时

## 关键指标

| 指标 | 良好 | 需要优化 |
|------|------|---------|
| 加速比 | >0.8x | <0.5x |
| Tokens/s | 越高越好 | 越低越差 |
| TTFT | 越低越好 | 越高越差 |

## 优化优先级

1. **Linear 算子**（占 60-80% 时间）- 最高优先级
2. **Self-attention**（占 10-20% 时间）
3. **其他算子**（各占 <10%）

## 快速优化对比

```bash
# 优化前
python test/benchmark_operators.py --operators linear > before.txt

# ... 进行优化 ...

# 优化后
python test/benchmark_operators.py --operators linear > after.txt

# 对比结果
diff before.txt after.txt
```

## 常用优化方法

### CPU
- 添加 SIMD（AVX2/AVX-512）
- 添加 OpenMP 多线程
- 使用 BLAS 库（OpenBLAS、MKL）

### GPU
- 优化 CUDA kernel
- 使用 cuBLAS 做矩阵乘法
- 改善内存合并访问

## 完整文档

参见 [docs/PERFORMANCE_BENCHMARKING.md](PERFORMANCE_BENCHMARKING.md)。
