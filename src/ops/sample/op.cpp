#include "op.hpp"
#include "cpu/sample_cpu.hpp"

namespace llaisys {
namespace ops {
namespace sample {

void sample(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
) {
    // Dispatch to CPU implementation
    cpu::sample_cpu(sampled_token, logits, temperature, top_k, top_p);
}

}  // namespace sample
}  // namespace ops
}  // namespace llaisys
