"""
Performance Benchmark for LLAISYS vs HuggingFace Inference

This script benchmarks inference performance and generates comparison charts.
Metrics collected:
- Time to First Token (TTFT)
- Tokens per Second (throughput)
- Total inference time
- Per-token latency
- Memory usage
"""

import gc
import time
import argparse
import os
import sys
import io
from typing import Dict, List, Tuple
import json
from datetime import datetime

import torch
import psutil
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Charts will not be generated.")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add test directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_utils import torch_device, llaisys_device

import llaisys


class PerformanceMetrics:
    """Container for performance metrics"""
    def __init__(self, name: str):
        self.name = name
        self.ttft = 0.0  # Time to first token
        self.total_time = 0.0
        self.num_tokens = 0
        self.tokens_per_second = 0.0
        self.per_token_latency = []  # List of per-token latencies
        self.prefill_time = 0.0  # Time for processing prompt
        self.decode_time = 0.0  # Time for generating tokens
        self.memory_peak_mb = 0.0
        self.memory_allocated_mb = 0.0

    def compute_derived_metrics(self):
        """Compute derived metrics from raw measurements"""
        if self.num_tokens > 0:
            self.tokens_per_second = self.num_tokens / self.total_time if self.total_time > 0 else 0
            self.decode_time = self.total_time - self.prefill_time

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            "name": self.name,
            "ttft_ms": self.ttft * 1000,
            "total_time_s": self.total_time,
            "num_tokens": self.num_tokens,
            "tokens_per_second": self.tokens_per_second,
            "prefill_time_ms": self.prefill_time * 1000,
            "decode_time_s": self.decode_time,
            "avg_per_token_latency_ms": np.mean(self.per_token_latency) * 1000 if self.per_token_latency else 0,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
        }

    def __str__(self) -> str:
        """Pretty print metrics"""
        return f"""
=== {self.name} Performance Metrics ===
Time to First Token (TTFT): {self.ttft*1000:.2f} ms
Total Time: {self.total_time:.2f} s
Tokens Generated: {self.num_tokens}
Throughput: {self.tokens_per_second:.2f} tokens/s
Prefill Time: {self.prefill_time*1000:.2f} ms
Decode Time: {self.decode_time:.2f} s
Avg Per-Token Latency: {np.mean(self.per_token_latency)*1000 if self.per_token_latency else 0:.2f} ms
Memory Peak: {self.memory_peak_mb:.2f} MB
Memory Allocated: {self.memory_allocated_mb:.2f} MB
"""


def get_memory_usage() -> Tuple[float, float]:
    """Get current memory usage in MB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024, mem_info.vms / 1024 / 1024


def load_hf_model(model_path=None, device_name="cpu"):
    """Load HuggingFace model"""
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def benchmark_hf_inference(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 128,
) -> Tuple[PerformanceMetrics, List[int], str]:
    """Benchmark HuggingFace inference with detailed metrics"""
    metrics = PerformanceMetrics("HuggingFace")

    # Prepare input
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)

    # Memory before inference
    mem_before, _ = get_memory_usage()

    # Start timing
    start_time = time.time()
    first_token_time = None
    token_times = []

    with torch.no_grad():
        # Use generate with output_scores to track per-token generation
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistency
            return_dict_in_generate=True,
            output_scores=False,
        )

    end_time = time.time()

    # Memory after inference
    mem_after, _ = get_memory_usage()

    # Extract metrics
    output_ids = outputs.sequences[0].tolist()
    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    metrics.total_time = end_time - start_time
    metrics.num_tokens = len(output_ids) - len(inputs[0])
    metrics.memory_peak_mb = mem_after - mem_before
    metrics.memory_allocated_mb = mem_after

    # Estimate TTFT and per-token latency (rough approximation)
    # For more accurate measurements, we'd need to modify the generation loop
    metrics.ttft = metrics.total_time / (metrics.num_tokens + 1) if metrics.num_tokens > 0 else 0
    metrics.prefill_time = metrics.ttft

    # Approximate per-token latency
    if metrics.num_tokens > 0:
        avg_decode_latency = (metrics.total_time - metrics.prefill_time) / metrics.num_tokens
        metrics.per_token_latency = [avg_decode_latency] * metrics.num_tokens

    metrics.compute_derived_metrics()

    return metrics, output_ids, result


def benchmark_llaisys_inference(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 128,
) -> Tuple[PerformanceMetrics, List[int], str]:
    """Benchmark LLAISYS inference with detailed metrics"""
    metrics = PerformanceMetrics("LLAISYS")

    # Prepare input
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)

    # Memory before inference
    mem_before, _ = get_memory_usage()

    # Start timing
    start_time = time.time()

    # Generate tokens
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=1,  # Greedy decoding
        top_p=1.0,
        temperature=1.0,
    )

    end_time = time.time()

    # Memory after inference
    mem_after, _ = get_memory_usage()

    # Decode result
    result = tokenizer.decode(outputs, skip_special_tokens=True)

    # Extract metrics
    metrics.total_time = end_time - start_time
    metrics.num_tokens = len(outputs) - len(inputs)
    metrics.memory_peak_mb = mem_after - mem_before
    metrics.memory_allocated_mb = mem_after

    # Estimate TTFT and per-token latency
    metrics.ttft = metrics.total_time / (metrics.num_tokens + 1) if metrics.num_tokens > 0 else 0
    metrics.prefill_time = metrics.ttft

    # Approximate per-token latency
    if metrics.num_tokens > 0:
        avg_decode_latency = (metrics.total_time - metrics.prefill_time) / metrics.num_tokens
        metrics.per_token_latency = [avg_decode_latency] * metrics.num_tokens

    metrics.compute_derived_metrics()

    return metrics, outputs, result


def load_llaisys_model(model_path, device_name):
    """Load LLAISYS model"""
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def generate_comparison_charts(
    all_metrics: List[Tuple[str, PerformanceMetrics, PerformanceMetrics]],
    output_dir: str
):
    """Generate comparison charts for all benchmarks"""
    if not HAS_MATPLOTLIB:
        print("Skipping chart generation (matplotlib not available)")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    prompts = [m[0] for m in all_metrics]
    hf_metrics = [m[1] for m in all_metrics]
    llaisys_metrics = [m[2] for m in all_metrics]

    # 1. Throughput Comparison (Tokens/Second)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(prompts))
    width = 0.35

    hf_throughput = [m.tokens_per_second for m in hf_metrics]
    llaisys_throughput = [m.tokens_per_second for m in llaisys_metrics]

    ax.bar(x - width/2, hf_throughput, width, label='HuggingFace', alpha=0.8)
    ax.bar(x + width/2, llaisys_throughput, width, label='LLAISYS', alpha=0.8)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Tokens per Second')
    ax.set_title('Throughput Comparison: HuggingFace vs LLAISYS')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(len(prompts))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=150)
    plt.close()

    # 2. Time to First Token (TTFT) Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    hf_ttft = [m.ttft * 1000 for m in hf_metrics]  # Convert to ms
    llaisys_ttft = [m.ttft * 1000 for m in llaisys_metrics]

    ax.bar(x - width/2, hf_ttft, width, label='HuggingFace', alpha=0.8)
    ax.bar(x + width/2, llaisys_ttft, width, label='LLAISYS', alpha=0.8)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Time to First Token (ms)')
    ax.set_title('TTFT Comparison: HuggingFace vs LLAISYS')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(len(prompts))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_comparison.png'), dpi=150)
    plt.close()

    # 3. Total Inference Time Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    hf_total = [m.total_time for m in hf_metrics]
    llaisys_total = [m.total_time for m in llaisys_metrics]

    ax.bar(x - width/2, hf_total, width, label='HuggingFace', alpha=0.8)
    ax.bar(x + width/2, llaisys_total, width, label='LLAISYS', alpha=0.8)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Total Inference Time: HuggingFace vs LLAISYS')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(len(prompts))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_time_comparison.png'), dpi=150)
    plt.close()

    # 4. Memory Usage Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    hf_memory = [m.memory_peak_mb for m in hf_metrics]
    llaisys_memory = [m.memory_peak_mb for m in llaisys_metrics]

    ax.bar(x - width/2, hf_memory, width, label='HuggingFace', alpha=0.8)
    ax.bar(x + width/2, llaisys_memory, width, label='LLAISYS', alpha=0.8)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Peak Memory Usage (MB)')
    ax.set_title('Memory Usage: HuggingFace vs LLAISYS')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(len(prompts))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=150)
    plt.close()

    # 5. Speedup Factor
    fig, ax = plt.subplots(figsize=(12, 6))

    speedup = [hf_metrics[i].total_time / llaisys_metrics[i].total_time
               if llaisys_metrics[i].total_time > 0 else 0
               for i in range(len(prompts))]

    colors = ['green' if s > 1 else 'red' for s in speedup]
    ax.bar(x, speedup, color=colors, alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (1x)')

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Speedup Factor (HF time / LLAISYS time)')
    ax.set_title('LLAISYS Speedup vs HuggingFace (>1 = LLAISYS faster)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test {i+1}" for i in range(len(prompts))], rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_factor.png'), dpi=150)
    plt.close()

    print(f"\nCharts saved to: {output_dir}")


def save_results_json(
    all_metrics: List[Tuple[str, PerformanceMetrics, PerformanceMetrics]],
    output_dir: str
):
    """Save benchmark results to JSON"""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": []
    }

    for prompt, hf_metrics, llaisys_metrics in all_metrics:
        results["benchmarks"].append({
            "prompt": prompt,
            "huggingface": hf_metrics.to_dict(),
            "llaisys": llaisys_metrics.to_dict(),
            "speedup": hf_metrics.total_time / llaisys_metrics.total_time if llaisys_metrics.total_time > 0 else 0
        })

    output_file = os.path.join(output_dir, 'benchmark_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")


def print_summary_table(all_metrics: List[Tuple[str, PerformanceMetrics, PerformanceMetrics]]):
    """Print a summary table of all benchmarks"""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'Test':<8} {'Framework':<15} {'Tokens/s':<12} {'TTFT (ms)':<12} {'Total (s)':<12} {'Memory (MB)':<12}")
    print("-"*100)

    for i, (prompt, hf_metrics, llaisys_metrics) in enumerate(all_metrics):
        print(f"{i+1:<8} {'HuggingFace':<15} {hf_metrics.tokens_per_second:<12.2f} "
              f"{hf_metrics.ttft*1000:<12.2f} {hf_metrics.total_time:<12.2f} {hf_metrics.memory_peak_mb:<12.2f}")
        print(f"{'':<8} {'LLAISYS':<15} {llaisys_metrics.tokens_per_second:<12.2f} "
              f"{llaisys_metrics.ttft*1000:<12.2f} {llaisys_metrics.total_time:<12.2f} {llaisys_metrics.memory_peak_mb:<12.2f}")

        speedup = hf_metrics.total_time / llaisys_metrics.total_time if llaisys_metrics.total_time > 0 else 0
        speedup_str = f"{'LLAISYS ' + ('faster' if speedup > 1 else 'slower')}: {abs(speedup):.2f}x"
        print(f"{'':<8} {speedup_str}")
        print("-"*100)

    # Calculate averages
    avg_hf_throughput = np.mean([m[1].tokens_per_second for m in all_metrics])
    avg_llaisys_throughput = np.mean([m[2].tokens_per_second for m in all_metrics])
    avg_speedup = np.mean([m[1].total_time / m[2].total_time if m[2].total_time > 0 else 0 for m in all_metrics])

    print(f"\n{'AVERAGE':<8} {'HuggingFace':<15} {avg_hf_throughput:<12.2f}")
    print(f"{'':<8} {'LLAISYS':<15} {avg_llaisys_throughput:<12.2f}")
    print(f"{'':<8} Average Speedup: {avg_speedup:.2f}x")
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLAISYS vs HuggingFace inference performance")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str,
                        help="Device to run inference on")
    parser.add_argument("--model", default=None, type=str,
                        help="Path to model directory")
    parser.add_argument("--max_tokens", default=128, type=int,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", default="benchmark_results", type=str,
                        help="Directory to save results and charts")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Custom prompts to benchmark (space-separated)")

    args = parser.parse_args()

    # Default test prompts with varying complexity
    if args.prompts:
        test_prompts = args.prompts
    else:
        test_prompts = [
            "Who are you?",
            "Explain what is machine learning in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the main differences between Python and C++?",
        ]

    print(f"\n{'='*100}")
    print(f"LLAISYS vs HuggingFace Inference Benchmark")
    print(f"{'='*100}")
    print(f"Device: {args.device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Number of test cases: {len(test_prompts)}")
    print(f"{'='*100}\n")

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)
    print("HuggingFace model loaded.\n")

    # Benchmark all prompts with HuggingFace
    hf_results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*100}")
        print(f"HuggingFace Test Case {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        print(f"{'='*100}\n")

        hf_metrics, hf_tokens, hf_output = benchmark_hf_inference(
            prompt, tokenizer, hf_model, args.max_tokens
        )
        print(hf_metrics)
        hf_results.append((prompt, hf_metrics, hf_tokens, hf_output))

    # Clean up HuggingFace model
    print("\nCleaning up HuggingFace model...")
    del hf_model
    gc.collect()
    if args.device == "nvidia":
        torch.cuda.empty_cache()

    # Load LLAISYS model
    print("\nLoading LLAISYS model...")
    llaisys_model = load_llaisys_model(model_path, args.device)
    print("LLAISYS model loaded.\n")

    # Benchmark all prompts with LLAISYS
    all_metrics = []
    for i, (prompt, hf_metrics, hf_tokens, hf_output) in enumerate(hf_results):
        print(f"\n{'='*100}")
        print(f"LLAISYS Test Case {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        print(f"{'='*100}\n")

        llaisys_metrics, llaisys_tokens, llaisys_output = benchmark_llaisys_inference(
            prompt, tokenizer, llaisys_model, args.max_tokens
        )
        print(llaisys_metrics)

        # Store combined results
        all_metrics.append((prompt, hf_metrics, llaisys_metrics))

    # Clean up LLAISYS model
    print("\nCleaning up LLAISYS model...")
    del llaisys_model
    gc.collect()
    if args.device == "nvidia":
        torch.cuda.empty_cache()

    # Print summary
    print_summary_table(all_metrics)

    # Save results
    save_results_json(all_metrics, args.output_dir)

    # Generate charts
    generate_comparison_charts(all_metrics, args.output_dir)

    print(f"\nBenchmark complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

