#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <omp.h>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t input_rows, size_t input_cols, size_t weight_rows) {
    const int M = static_cast<int>(input_rows);
    const int N = static_cast<int>(weight_rows);
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < M * N; ij++) {
        int i = ij / N;
        int j = ij % N;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                      std::is_same_v<T, llaisys::fp16_t>) {
            float acc = llaisys::utils::cast<float>(bias[j]);
            for (size_t k = 0; k < input_cols; k++) {
                acc += llaisys::utils::cast<float>(in[i * input_cols + k]) *
                       llaisys::utils::cast<float>(weight[j * input_cols + k]);
            }
            out[i * weight_rows + j] = llaisys::utils::cast<T>(acc);
        } else {
            T acc = bias[j];
            for (size_t k = 0; k < input_cols; k++) {
                acc += in[i * input_cols + k] * weight[j * input_cols + k];
            }
            out[i * weight_rows + j] = acc;
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias,
            llaisysDataType_t type, size_t input_rows, size_t input_cols,
            size_t weight_rows) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       input_rows, input_cols, weight_rows);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
