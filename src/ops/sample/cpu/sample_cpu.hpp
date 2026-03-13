#ifndef LLAISYS_OPS_SAMPLE_CPU_HPP
#define LLAISYS_OPS_SAMPLE_CPU_HPP

#include "../../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {
namespace sample {
namespace cpu {

void sample_cpu(
    tensor_t sampled_token,  // Output: [1] int64 tensor
    tensor_t logits,         // Input: [vocab_size] float tensor
    float temperature,       // Sampling temperature (0.1-2.0)
    int top_k,              // Top-K filtering (0 = disabled)
    float top_p             // Top-P (nucleus) filtering (0.0-1.0)
);

}  // namespace cpu
}  // namespace sample
}  // namespace ops
}  // namespace llaisys

#endif  // LLAISYS_OPS_SAMPLE_CPU_HPP
