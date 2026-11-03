#pragma once

#include "common.cuh"

// GP100 (sm_60) fp16 matrix-vector path for batch-size-1 quantized matmuls and MUL_MAT_ID. See mmvq-gp100.cu.
// fusion: nullptr, or a plain gate/up/SWIGLU fusion (src0 = ffn_up, fusion->gate = ffn_gate, dst = the GLU output)
bool ggml_cuda_gp100_mmvq_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_gp100_mmvq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                          ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

// BF16 x one f32 column (plain MUL_MAT, no fusion) with 16-byte loads; see mmvq-gp100.cu
bool ggml_cuda_gp100_mmvf_supported(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                    const ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_gp100_mmvf(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Fused RMS_NORM * weight for a single row that also fills the GP100 activation cache for the matmuls that
// consume it. Returns false (nothing launched) when not applicable.
bool ggml_cuda_gp100_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul);

// types with a GP100 kernel, so MMVQ gate/up/GLU fusion can be allowed on GP100 for them
// (MUL_MAT_ID only has the 32-weight-unit kernel: IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0)
static inline bool ggml_cuda_gp100_mmvq_fusable_type(const ggml_type type, const bool mul_mat_id) {
    const bool unit = type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S ||
                      type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q2_0;
    return unit || (!mul_mat_id && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
                                    type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q3_K));
}
