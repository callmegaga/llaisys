#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <immintrin.h>
#include <omp.h>

// Generic template: handles BF16 and FP16 with OpenMP
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

// Float specialization: OpenMP + AVX2 FMA
template <>
void linear_<float>(float *out, const float *in, const float *weight,
                    const float *bias, size_t input_rows, size_t input_cols,
                    size_t weight_rows) {
    const int M = static_cast<int>(input_rows);
    const int N = static_cast<int>(weight_rows);
#pragma omp parallel for schedule(static)
    for (int ij = 0; ij < M * N; ij++) {
        int i = ij / N;
        int j = ij % N;
        const float *in_row = in + i * input_cols;
        const float *w_row  = weight + j * input_cols;

        __m256 vacc = _mm256_setzero_ps();
        size_t k = 0;
        for (; k + 8 <= input_cols; k += 8) {
            __m256 va = _mm256_loadu_ps(in_row + k);
            __m256 vb = _mm256_loadu_ps(w_row + k);
            vacc = _mm256_fmadd_ps(va, vb, vacc);
        }

        // Horizontal sum: add upper and lower 128-bit lanes, then hadd twice
        __m128 lo  = _mm256_castps256_ps128(vacc);
        __m128 hi  = _mm256_extractf128_ps(vacc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        float acc = _mm_cvtss_f32(sum);

        // Scalar remainder
        for (; k < input_cols; k++) {
            acc += in_row[k] * w_row[k];
        }

        out[i * weight_rows + j] = acc + bias[j];
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
