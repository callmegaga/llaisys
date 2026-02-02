#include "llaisys/models/qwen2.h"
#include "../../models/qwen2.hpp"

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
		return nullptr;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
		return nullptr;
    }


    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken) {
		return 1;
    }
}