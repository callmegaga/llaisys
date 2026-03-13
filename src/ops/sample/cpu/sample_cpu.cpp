#include "sample_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <algorithm>

template <typename T>
void sample_(int64_t *sampled_token, const T *logits, size_t vocab_size, float temperature, int top_k, float top_p) {
    if (vocab_size == 0) {
        return;
    }

    // For now: Apply temperature scaling and return argmax
    // Top-K and Top-P filtering will be implemented in Task 1.3
    size_t max_idx = 0;
    float max_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < vocab_size; i++) {
        float scaled_logit;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            scaled_logit = llaisys::utils::cast<float>(logits[i]) / temperature;
        } else {
            scaled_logit = static_cast<float>(logits[i]) / temperature;
        }

        if (scaled_logit > max_val) {
            max_val = scaled_logit;
            max_idx = i;
        }
    }

    *sampled_token = static_cast<int64_t>(max_idx);
}

namespace llaisys::ops::cpu {
void sample(std::byte *sampled_token, const std::byte *logits, llaisysDataType_t logits_dtype,
            size_t vocab_size, float temperature, int top_k, float top_p) {
    int64_t *token_ptr = reinterpret_cast<int64_t *>(sampled_token);

    switch (logits_dtype) {
    case LLAISYS_DTYPE_F32:
        sample_(token_ptr, reinterpret_cast<const float *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_BF16:
        sample_(token_ptr, reinterpret_cast<const llaisys::bf16_t *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    case LLAISYS_DTYPE_F16:
        sample_(token_ptr, reinterpret_cast<const llaisys::fp16_t *>(logits), vocab_size, temperature, top_k, top_p);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(logits_dtype);
    }
}
} // namespace llaisys::ops::cpu
