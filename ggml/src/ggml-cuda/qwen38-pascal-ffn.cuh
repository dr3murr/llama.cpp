#pragma once

#include "common.cuh"

bool ggml_cuda_qwen38_pascal_ffn_registered(int device, int layer);
void ggml_cuda_op_qwen38_pascal_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
