import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, zero_tensor


def torch_sample_with_temperature(logits, temperature):
    """Apply temperature scaling and return argmax (for now)"""
    scaled_logits = logits / temperature
    return torch.argmax(scaled_logits, dim=-1, keepdim=True)


def test_op_sample_temperature(
    vocab_size=100,
    temperature=1.0,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   vocab_size {vocab_size} temperature {temperature} dtype <{dtype_name}>")

    # Create input logits and output token tensors
    logits, logits_ = random_tensor((vocab_size,), dtype_name, device_name, scale=10.0, bias=-5.0)
    sampled_token, sampled_token_ = zero_tensor((1,), "i64", device_name)

    # For now, we test with temperature scaling + argmax
    # Top-K and Top-P are disabled (top_k=0, top_p=1.0)
    torch_result = torch_sample_with_temperature(logits, temperature)
    llaisys.Ops.sample(sampled_token_, logits_, temperature, 0, 1.0)

    # Copy torch result to sampled_token for comparison
    sampled_token.copy_(torch_result)

    assert check_equal(sampled_token_, sampled_token, strict=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    testCases = [
        (100, 1.0),   # No temperature scaling
        (100, 0.5),   # Lower temperature (more confident)
        (100, 2.0),   # Higher temperature (more random)
        (4096, 1.0),  # Larger vocabulary
    ]
    testDtype = ["f32", "f16", "bf16"]

    print(f"Testing Ops.sample on {args.device}")
    for vocab_size, temperature in testCases:
        for dtype_name in testDtype:
            test_op_sample_temperature(vocab_size, temperature, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
