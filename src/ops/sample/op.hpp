#ifndef LLAISYS_OPS_SAMPLE_OP_HPP
#define LLAISYS_OPS_SAMPLE_OP_HPP

#include "../../tensor/tensor.hpp"

namespace llaisys {
namespace ops {
namespace sample {

void sample(
    tensor_t sampled_token,
    tensor_t logits,
    float temperature,
    int top_k,
    float top_p
);

}  // namespace sample
}  // namespace ops
}  // namespace llaisys

#endif  // LLAISYS_OPS_SAMPLE_OP_HPP
