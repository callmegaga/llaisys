"""
Quick Operator-Level Performance Benchmark

This script benchmarks individual operators to identify performance bottlenecks.
"""

import argparse
import sys
import os
import time
import json
from typing import Dict, List

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_utils import (
    random_tensor, random_int_tensor, zero_tensor,
    check_equal, benchmark, torch_device, llaisys_device
)

import llaisys


def benchmark_operator(
    op_name: str,
    torch_func,
    llaisys_func,
    device_name: str,
    warmup: int = 10,
    repeat: int = 100
) -> Dict:
    """Benchmark a single operator"""
    print(f"\nBenchmarking {op_name}...")

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def time_op(func):
        for _ in range(warmup):
            func()
        api.device_synchronize()
        start = time.time()
        for _ in range(repeat):
            func()
        api.device_synchronize()
        end = time.time()
        return (end - start) / repeat

    torch_time = time_op(torch_func)
    llaisys_time = time_op(llaisys_func)

    speedup = torch_time / llaisys_time if llaisys_time > 0 else 0

    result = {
        "operator": op_name,
        "torch_time_ms": torch_time * 1000,
        "llaisys_time_ms": llaisys_time * 1000,
        "speedup": speedup
    }

    print(f"  PyTorch:  {torch_time*1000:.3f} ms")
    print(f"  LLAISYS:  {llaisys_time*1000:.3f} ms")
    print(f"  Speedup:  {speedup:.2f}x {'(LLAISYS faster)' if speedup > 1 else '(PyTorch faster)'}")

    return result


def benchmark_linear(device_name: str, batch_size: int = 32, seq_len: int = 128,
                     in_features: int = 2048, out_features: int = 2048) -> Dict:
    """Benchmark linear operator"""
    # Linear expects 2D tensors: [batch*seq, features]
    torch_in, llaisys_in = random_tensor([batch_size * seq_len, in_features], "bf16", device_name)
    torch_weight, llaisys_weight = random_tensor([out_features, in_features], "bf16", device_name)
    torch_bias, llaisys_bias = zero_tensor([out_features], "bf16", device_name)
    torch_out, llaisys_out = zero_tensor([batch_size * seq_len, out_features], "bf16", device_name)

    def torch_func():
        torch.nn.functional.linear(torch_in, torch_weight, torch_bias, out=torch_out)

    def llaisys_func():
        llaisys.Ops.linear(llaisys_out, llaisys_in, llaisys_weight, llaisys_bias)

    return benchmark_operator("linear", torch_func, llaisys_func, device_name)


def benchmark_rms_norm(device_name: str, batch_size: int = 32, seq_len: int = 128,
                       hidden_size: int = 2048) -> Dict:
    """Benchmark RMS normalization"""
    # RMS norm expects 2D tensors: [batch*seq, hidden]
    torch_in, llaisys_in = random_tensor([batch_size * seq_len, hidden_size], "bf16", device_name)
    torch_weight, llaisys_weight = random_tensor([hidden_size], "bf16", device_name)
    torch_out, llaisys_out = zero_tensor([batch_size * seq_len, hidden_size], "bf16", device_name)

    eps = 1e-6

    def torch_func():
        # PyTorch RMS norm implementation
        variance = torch_in.pow(2).mean(-1, keepdim=True)
        torch_out.copy_(torch_in * torch.rsqrt(variance + eps) * torch_weight)

    def llaisys_func():
        llaisys.Ops.rms_norm(llaisys_out, llaisys_in, llaisys_weight, eps)

    return benchmark_operator("rms_norm", torch_func, llaisys_func, device_name)


def benchmark_self_attention(device_name: str, batch_size: int = 1, seq_len: int = 128,
                              num_heads: int = 32, head_dim: int = 64) -> Dict:
    """Benchmark self-attention"""
    torch_q, llaisys_q = random_tensor([seq_len, num_heads, head_dim], "bf16", device_name)
    torch_k, llaisys_k = random_tensor([seq_len, num_heads, head_dim], "bf16", device_name)
    torch_v, llaisys_v = random_tensor([seq_len, num_heads, head_dim], "bf16", device_name)
    torch_out, llaisys_out = zero_tensor([seq_len, num_heads, head_dim], "bf16", device_name)

    scale = 1.0 / (head_dim ** 0.5)

    def torch_func():
        # Simplified attention
        attn_weights = torch.matmul(torch_q, torch_k.transpose(-2, -1)) * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        torch_out.copy_(torch.matmul(attn_weights, torch_v))

    def llaisys_func():
        llaisys.Ops.self_attention(llaisys_out, llaisys_q, llaisys_k, llaisys_v, scale)

    return benchmark_operator("self_attention", torch_func, llaisys_func, device_name)


def benchmark_rope(device_name: str, seq_len: int = 128, num_heads: int = 32,
                   head_dim: int = 64) -> Dict:
    """Benchmark RoPE"""
    torch_in, llaisys_in = random_tensor([seq_len, num_heads, head_dim], "bf16", device_name)
    torch_pos, llaisys_pos = random_int_tensor([seq_len], device_name, "i64", low=0, high=2048)
    torch_out, llaisys_out = zero_tensor([seq_len, num_heads, head_dim], "bf16", device_name)

    theta = 10000.0

    def torch_func():
        # Simplified RoPE (not exact implementation)
        torch_out.copy_(torch_in)

    def llaisys_func():
        llaisys.Ops.rope(llaisys_out, llaisys_in, llaisys_pos, theta)

    return benchmark_operator("rope", torch_func, llaisys_func, device_name)


def benchmark_swiglu(device_name: str, batch_size: int = 32, seq_len: int = 128,
                     intermediate_size: int = 8192) -> Dict:
    """Benchmark SwiGLU"""
    # SwiGLU expects 2D tensors: [batch*seq, intermediate]
    torch_gate, llaisys_gate = random_tensor([batch_size * seq_len, intermediate_size], "bf16", device_name)
    torch_up, llaisys_up = random_tensor([batch_size * seq_len, intermediate_size], "bf16", device_name)
    torch_out, llaisys_out = zero_tensor([batch_size * seq_len, intermediate_size], "bf16", device_name)

    def torch_func():
        torch_out.copy_(torch_up * torch.nn.functional.silu(torch_gate))

    def llaisys_func():
        llaisys.Ops.swiglu(llaisys_out, llaisys_gate, llaisys_up)

    return benchmark_operator("swiglu", torch_func, llaisys_func, device_name)


def main():
    parser = argparse.ArgumentParser(description="Benchmark individual operators")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--operators", nargs="+", default=None,
                        choices=["linear", "rms_norm", "self_attention", "rope", "swiglu", "all"],
                        help="Operators to benchmark (default: all)")
    parser.add_argument("--warmup", default=10, type=int, help="Number of warmup iterations")
    parser.add_argument("--repeat", default=100, type=int, help="Number of benchmark iterations")
    parser.add_argument("--output", default="operator_benchmark.json", type=str,
                        help="Output JSON file")

    args = parser.parse_args()

    if args.operators is None or "all" in args.operators:
        operators = ["linear", "rms_norm", "self_attention", "rope", "swiglu"]
    else:
        operators = args.operators

    print(f"\n{'='*80}")
    print(f"Operator Performance Benchmark")
    print(f"{'='*80}")
    print(f"Device: {args.device}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.repeat}")
    print(f"Operators: {', '.join(operators)}")
    print(f"{'='*80}\n")

    results = []

    for op in operators:
        if op == "linear":
            result = benchmark_linear(args.device)
        elif op == "rms_norm":
            result = benchmark_rms_norm(args.device)
        elif op == "self_attention":
            result = benchmark_self_attention(args.device)
        elif op == "rope":
            result = benchmark_rope(args.device)
        elif op == "swiglu":
            result = benchmark_swiglu(args.device)
        else:
            print(f"Unknown operator: {op}")
            continue

        results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Operator':<20} {'PyTorch (ms)':<15} {'LLAISYS (ms)':<15} {'Speedup':<10}")
    print("-"*80)

    for r in results:
        speedup_str = f"{r['speedup']:.2f}x"
        print(f"{r['operator']:<20} {r['torch_time_ms']:<15.3f} {r['llaisys_time_ms']:<15.3f} {speedup_str:<10}")

    avg_speedup = np.mean([r['speedup'] for r in results])
    print("-"*80)
    print(f"{'Average Speedup:':<50} {avg_speedup:.2f}x")
    print(f"{'='*80}\n")

    # Save to JSON
    output_data = {
        "device": args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "results": results,
        "average_speedup": avg_speedup
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
