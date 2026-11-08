#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "qwen38-streamed-attn.cuh"
#include "qwen38-streamed-attn-runtime.cuh"
#include "qwen38-sage-attention.cuh"

#include <algorithm>

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASE( 64, GGML_TYPE_Q8_KV64,  GGML_TYPE_Q8_KV64)
    FATTN_VEC_CASE(128, GGML_TYPE_Q8_KV64,  GGML_TYPE_Q8_KV64)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_KV64,  GGML_TYPE_Q8_KV64)
    FATTN_VEC_CASE(128, GGML_TYPE_Q8_KV128, GGML_TYPE_Q8_KV128)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_KV128, GGML_TYPE_Q8_KV128)
    FATTN_VEC_CASE(256, GGML_TYPE_Q8_KV256, GGML_TYPE_Q8_KV256)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
    BEST_FATTN_KERNEL_MMA_Q8_KV256 = 450,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_KV64:
        case GGML_TYPE_Q8_KV128:
        case GGML_TYPE_Q8_KV256:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    if (K->type != V->type) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    const bool direct_q8_kv = K->type == V->type &&
        (K->type == GGML_TYPE_Q8_KV64 ||
         K->type == GGML_TYPE_Q8_KV128 ||
         K->type == GGML_TYPE_Q8_KV256);
    int64_t direct_q8_max_q = 64;
    if (cc >= GGML_CUDA_CC_BLACKWELL) {
        if (const char * value = getenv("QWEN38_CUDA_DIRECT_Q8_MAX_Q")) {
            const int64_t requested = atoll(value);
            if (requested > 0) {
                direct_q8_max_q = requested;
            }
        }
    }
    const char * direct_q8_mma_value = getenv("QWEN38_CUDA_DIRECT_Q8_MMA");
    const size_t q8_kv256_row_size = sizeof(block_q8_kv256);
    const bool q8_kv256_token_major =
        K->nb[1] == 4*q8_kv256_row_size && K->nb[2] == q8_kv256_row_size &&
        V->nb[1] == 4*q8_kv256_row_size && V->nb[2] == q8_kv256_row_size;
    const bool q8_kv256_head_major =
        ggml_cuda_q8_kv256_is_head_major(K) && ggml_cuda_q8_kv256_is_head_major(V);
    const int64_t direct_q8_mma_max_q = q8_kv256_head_major ? 512 : 256;
    const bool direct_q8_mma = cc >= GGML_CUDA_CC_BLACKWELL &&
        direct_q8_mma_value && atoi(direct_q8_mma_value) != 0 &&
        Q->ne[0] == 256 && Q->ne[1] > 64 && Q->ne[1] <= direct_q8_mma_max_q && gqa_ratio == 6 &&
        K->type == GGML_TYPE_Q8_KV256 && V->type == GGML_TYPE_Q8_KV256 &&
        (q8_kv256_token_major || q8_kv256_head_major) && gqa_opt_applies;
    if (direct_q8_mma) {
        return BEST_FATTN_KERNEL_MMA_Q8_KV256;
    }

    if (can_use_vector_kernel && direct_q8_kv && Q->ne[1] <= direct_q8_max_q) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

namespace {

bool qwen38_sage_attention_enabled() {
    const char * value = getenv("QWEN38_CUDA_SAGE_ATTN_EXPERIMENT");
    if (!value) {
        value = getenv("QWEN38_CUDA_SAGE_ATTN");
    }
    return value && atoi(value) != 0;
}

int qwen38_sage_attention_min_q() {
    const char * value = getenv("QWEN38_CUDA_SAGE_ATTN_MIN_Q");
    return value ? std::max(1, atoi(value)) : 65;
}

int qwen38_sage_attention_min_kv() {
    const char * value = getenv("QWEN38_CUDA_SAGE_ATTN_MIN_KV");
    return value ? std::max(0, atoi(value)) : 65536;
}

int64_t qwen38_sage_attention_capacity(const ggml_tensor * tensor) {
    if (!ggml_cuda_q8_kv256_is_head_major(tensor)) {
        return 0;
    }
    return tensor->nb[2]/sizeof(block_q8_kv256);
}

bool qwen38_sage_attention_supported(const int device, const ggml_tensor * dst) {
    if (!qwen38_sage_attention_enabled() ||
            ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_BLACKWELL ||
            dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!q || !k || !v || !mask || dst->src[4] ||
            q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_KV256 || v->type != GGML_TYPE_Q8_KV256 ||
            mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) ||
            q->ne[0] != 256 || q->ne[2] != 24 || q->ne[3] != 1 ||
            q->ne[1] < qwen38_sage_attention_min_q() || q->ne[1] > 512 ||
            k->ne[0] != 256 || k->ne[2] != 4 || k->ne[3] != 1 ||
            v->ne[0] != 256 || v->ne[2] != 4 || v->ne[3] != 1 ||
            k->ne[1] != v->ne[1] || k->ne[1] < std::max<int64_t>(q->ne[1], qwen38_sage_attention_min_kv()) ||
            k->ne[1] % 64 != 0 ||
            mask->ne[0] != k->ne[1] || mask->ne[1] != q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }

    float scale = 0.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (fabsf(scale - 1.0f/sqrtf(256.0f)) > 1.0e-7f || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    const int64_t k_capacity = qwen38_sage_attention_capacity(k);
    const int64_t v_capacity = qwen38_sage_attention_capacity(v);
    return k_capacity == v_capacity && k_capacity >= k->ne[1] && k_capacity % 64 == 0 &&
        qwen38_cuda_sage_cache_is_active(k->data) &&
        qwen38_cuda_sage_cache_is_active(v->data);
}

struct qwen38_sage_workspace {
    int8_t * q;
    float * q_scale;
    void * output_half;
    int * query_limits;
    uintptr_t end;
};

qwen38_sage_workspace qwen38_sage_attention_workspace(const ggml_tensor * dst) {
    const int q_tokens = dst->src[0]->ne[1];
    uintptr_t end = (uintptr_t) dst->data + ggml_nbytes(dst);
    end = GGML_PAD(end, 128);
    int8_t * q = reinterpret_cast<int8_t *>(end);
    end += qwen38_cuda_sage_q_bytes(q_tokens);
    end = GGML_PAD(end, 128);
    float * q_scale = reinterpret_cast<float *>(end);
    end += qwen38_cuda_sage_q_scale_bytes(q_tokens);
    end = GGML_PAD(end, 128);
    void * output_half = reinterpret_cast<void *>(end);
    end += qwen38_cuda_sage_output_half_bytes(q_tokens);
    end = GGML_PAD(end, 128);
    int * query_limits = reinterpret_cast<int *>(end);
    end += qwen38_cuda_sage_query_limit_bytes(q_tokens);
    return { q, q_scale, output_half, query_limits, end };
}

void qwen38_sage_attention_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int capacity = qwen38_sage_attention_capacity(k);
    const qwen38_sage_workspace workspace = qwen38_sage_attention_workspace(dst);

    const int8_t * const k_data = static_cast<const int8_t *>(k->data);
    const int8_t * const v_data = static_cast<const int8_t *>(v->data);
    const float * const k_scale = reinterpret_cast<const float *>(
        k_data + int64_t(4)*capacity*256);
    const float * const v_scale = reinterpret_cast<const float *>(
        v_data + int64_t(4)*256*capacity);

    CUDA_CHECK(qwen38_cuda_sage_attention(
        static_cast<const float *>(q->data),
        k_data,
        v_data,
        k_scale,
        v_scale,
        static_cast<float *>(dst->data),
        workspace.q,
        workspace.q_scale,
        workspace.output_half,
        workspace.query_limits,
        q->ne[1],
        k->ne[1],
        capacity,
        false,
        mask->data,
        mask->nb[1],
        capacity/64*4,
        ctx.stream()));
}

} // namespace

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    if (qwen38_sage_attention_supported(device, dst)) {
        return qwen38_sage_attention_workspace(dst).end - (uintptr_t) dst->data;
    }

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_MMA_Q8_KV256:
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    // A head-major Q8 cache only removes the F16 materialization allocation
    // when dispatch selected the direct-Q8 MMA kernel.  A test/cache view can
    // have the same strides while the ordinary MMA_F16 kernel is selected;
    // omitting K/V scratch in that case makes the conversion overrun dst.
    const bool persistent_head_major = kernel == BEST_FATTN_KERNEL_MMA_Q8_KV256 &&
        ggml_cuda_q8_kv256_is_head_major(K) && ggml_cuda_q8_kv256_is_head_major(V);
    const bool bounded_local = ggml_cuda_qwen38_bounded_local_attn_supported(device, dst);
    const bool direct_q8_strided = kernel == BEST_FATTN_KERNEL_MMA_Q8_KV256 &&
        getenv("QWEN38_CUDA_DIRECT_Q8_STRIDED") &&
        atoi(getenv("QWEN38_CUDA_DIRECT_Q8_STRIDED")) != 0;
    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        (bounded_local || direct_q8_strided || persistent_head_major) ?
            ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, false, false) :
        kernel == BEST_FATTN_KERNEL_MMA_Q8_KV256 ?
            ggml_cuda_flash_attn_ext_get_q8_head_major_extra_data(dst) :
            ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    const size_t alloc_size = f16_extra.end - (uintptr_t) dst->data;
    if (getenv("QWEN38_CUDA_LOG_FATTN_ALLOC")) {
        fprintf(stderr,
                "qwen38 fattn alloc: device=%d kernel=%d q=%lld kv=%lld K=%s V=%s bytes=%zu "
                "K_data=%p K_nb=[%zu,%zu,%zu,%zu] V_data=%p V_nb=[%zu,%zu,%zu,%zu]\n",
                device, (int) kernel, (long long) dst->src[0]->ne[1], (long long) K->ne[1],
                ggml_type_name(K->type), ggml_type_name(V->type), alloc_size,
                K->data, K->nb[0], K->nb[1], K->nb[2], K->nb[3],
                V->data, V->nb[0], V->nb[1], V->nb[2], V->nb[3]);
    }
    return alloc_size;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    if (qwen38_sage_attention_supported(ctx.device, dst)) {
        qwen38_sage_attention_launch(ctx, dst);
        return;
    }
    if (ggml_cuda_qwen38_streamed_attn(ctx, dst)) {
        return;
    }
    if (ggml_cuda_qwen38_bounded_local_attn(ctx, dst)) {
        return;
    }
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_Q8_KV256:
            {
            const char * value = getenv("QWEN38_CUDA_DIRECT_Q8_SHAPE_EXPERIMENT");
            if (!value) {
                value = getenv("QWEN38_CUDA_DIRECT_Q8_SHAPE");
            }
            if (value) {
                switch (atoi(value)) {
                    case 3202:
                        ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2, true>(ctx, dst);
                        break;
                    default:
                        ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8, true>(ctx, dst);
                        break;
                }
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8, true>(ctx, dst);
            }
            }
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

// --------------------------------------------------------------------------
// Qwen3.8 streamed-attention experiment
//
// This is deliberately a raw fixed-shape interface rather than a graph op.
// It reuses flash_attn_ext_f16_process_tile in the same direct-Q8 mode as the
// production kernel, but asks it to emit an unnormalized fixup result for one
// bounded K/V tile.  The scatter kernel turns the internal padded 8x8 layout
// into [query][24 heads][part][256], after which the existing FP32 log-sum-exp
// combiner can merge independently streamed tiles exactly as it merges
// parallel resident-cache partitions.

namespace {

constexpr int qwen38_sa_d          = 256;
constexpr int qwen38_sa_q_heads    = 24;
constexpr int qwen38_sa_kv_heads   = 4;
constexpr int qwen38_sa_gqa        = 6;
constexpr int qwen38_sa_ncols1     = 8;
constexpr int qwen38_sa_ncols2     = 8;
constexpr int qwen38_sa_ncols      = qwen38_sa_ncols1*qwen38_sa_ncols2;
constexpr int qwen38_sa_q8_bytes   = sizeof(block_q8_kv256);

static int qwen38_sa_blocks(const int q_tokens) {
    return ((q_tokens + qwen38_sa_ncols1 - 1)/qwen38_sa_ncols1)*qwen38_sa_kv_heads;
}

template<int nwarps>
__launch_bounds__(32*nwarps, 1)
static __global__ void qwen38_streamed_attn_partial_kernel(
        const float * __restrict__ q,
        const char  * __restrict__ k,
        const char  * __restrict__ v,
        const half  * __restrict__ mask,
        float2      * __restrict__ raw,
        const float scale,
        const uint3 ne01,
        const int q_tokens,
        const int kv_tokens,
        const int kv_head_stride_tokens,
        const int mask_stride) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE)
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(
        qwen38_sa_d, qwen38_sa_d, qwen38_sa_ncols);

    const int ntiles_q = (q_tokens + qwen38_sa_ncols1 - 1)/qwen38_sa_ncols1;
    const int z_kv = blockIdx.x/ntiles_q;
    const int jt   = blockIdx.x - z_kv*ntiles_q;
    if (z_kv >= qwen38_sa_kv_heads) {
        return;
    }

    const int zt_q = z_kv*qwen38_sa_gqa;
    const float2 * q_f2 = reinterpret_cast<const float2 *>(q) +
        int64_t(zt_q)*q_tokens*(qwen38_sa_d/2);
    const char * k_head = k + int64_t(z_kv)*kv_head_stride_tokens*qwen38_sa_q8_bytes;
    const char * v_head = v + int64_t(z_kv)*kv_head_stride_tokens*qwen38_sa_q8_bytes;

    constexpr bool use_logit_softcap = false;
    constexpr bool V_is_K_view       = false;
    constexpr bool direct_q8_kv256   = true;
    constexpr bool needs_fixup       = false;
    constexpr bool is_fixup          = true;

    flash_attn_ext_f16_process_tile<
        qwen38_sa_d, qwen38_sa_d, qwen38_sa_ncols1, qwen38_sa_ncols2, nwarps,
        use_logit_softcap, V_is_K_view, direct_q8_kv256, needs_fixup, is_fixup>(
            q_f2, k_head, v_head, mask, nullptr, nullptr, raw,
            scale, 1.0f, 0.0f,
            ne01, qwen38_sa_q_heads, qwen38_sa_gqa, kv_tokens,
            qwen38_sa_d/2, q_tokens*(qwen38_sa_d/2),
            qwen38_sa_q8_bytes, qwen38_sa_q8_bytes, mask_stride,
            jt, 0, 0, (kv_tokens + nbatch_fa - 1)/nbatch_fa);
#else
    GGML_UNUSED_VARS(q, k, v, mask, raw, scale, ne01, q_tokens, kv_tokens, kv_head_stride_tokens, mask_stride);
    NO_DEVICE_CODE;
#endif
}

static __global__ void qwen38_streamed_attn_scatter(
        const float2 * __restrict__ raw,
        float * __restrict__ parts,
        float2 * __restrict__ meta,
        const int q_tokens,
        const int part,
        const int n_parts) {
    const int64_t n_vectors = int64_t(q_tokens)*qwen38_sa_q_heads;
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= n_vectors*qwen38_sa_d) {
        return;
    }

    const int d = index % qwen38_sa_d;
    const int64_t vector = index/qwen38_sa_d;
    const int head = vector % qwen38_sa_q_heads;
    const int token = vector/qwen38_sa_q_heads;

    const int ntiles_q = (q_tokens + qwen38_sa_ncols1 - 1)/qwen38_sa_ncols1;
    const int z_kv = head/qwen38_sa_gqa;
    const int c = head - z_kv*qwen38_sa_gqa;
    const int block = z_kv*ntiles_q + token/qwen38_sa_ncols1;
    const int jc = (token % qwen38_sa_ncols1)*qwen38_sa_ncols2 + c;
    const int nblocks = ntiles_q*qwen38_sa_kv_heads;

    const float * raw_data = reinterpret_cast<const float *>(
        raw + int64_t(nblocks)*(2*qwen38_sa_ncols));
    parts[(vector*n_parts + part)*qwen38_sa_d + d] =
        raw_data[(int64_t(block)*qwen38_sa_ncols + jc)*qwen38_sa_d + d];

    if (d == 0) {
        meta[vector*n_parts + part] =
            raw[(int64_t(nblocks) + block)*qwen38_sa_ncols + jc];
    }
}

static __global__ void qwen38_streamed_attn_merge_raw(
        const float2 * __restrict__ raw,
        float * __restrict__ running,
        float2 * __restrict__ running_meta,
        const int q_tokens,
        const bool first) {
    const int64_t vector = blockIdx.x;
    if (vector >= int64_t(q_tokens)*qwen38_sa_q_heads) {
        return;
    }

    const int head = vector % qwen38_sa_q_heads;
    const int token = vector/qwen38_sa_q_heads;
    const int ntiles_q = (q_tokens + qwen38_sa_ncols1 - 1)/qwen38_sa_ncols1;
    const int z_kv = head/qwen38_sa_gqa;
    const int c = head - z_kv*qwen38_sa_gqa;
    const int block = z_kv*ntiles_q + token/qwen38_sa_ncols1;
    const int jc = (token % qwen38_sa_ncols1)*qwen38_sa_ncols2 + c;
    const int nblocks = ntiles_q*qwen38_sa_kv_heads;

    __shared__ float2 scale_and_meta[2];
    if (threadIdx.x == 0) {
        const float2 tile_meta = raw[(int64_t(nblocks) + block)*qwen38_sa_ncols + jc];
        if (first) {
            scale_and_meta[0] = make_float2(0.0f, 1.0f);
            scale_and_meta[1] = tile_meta;
        } else {
            const float2 previous = running_meta[vector];
            const float maximum = fmaxf(previous.x, tile_meta.x);
            const float scale_previous = expf(previous.x - maximum);
            const float scale_tile = expf(tile_meta.x - maximum);
            scale_and_meta[0] = make_float2(scale_previous, scale_tile);
            scale_and_meta[1] = make_float2(
                maximum, scale_previous*previous.y + scale_tile*tile_meta.y);
        }
    }
    __syncthreads();

    const float * raw_data = reinterpret_cast<const float *>(
        raw + int64_t(nblocks)*(2*qwen38_sa_ncols));
    const float partial = raw_data[(int64_t(block)*qwen38_sa_ncols + jc)*qwen38_sa_d + threadIdx.x];
    const int64_t output_index = vector*qwen38_sa_d + threadIdx.x;
    running[output_index] = first ? partial :
        scale_and_meta[0].x*running[output_index] + scale_and_meta[0].y*partial;

    __syncthreads();
    if (threadIdx.x == 0) {
        running_meta[vector] = scale_and_meta[1];
    }
}

static __global__ void qwen38_streamed_attn_finalize_kernel(
        const float * __restrict__ running,
        const float2 * __restrict__ running_meta,
        float * __restrict__ output,
        const int64_t n_vectors) {
    const int64_t vector = blockIdx.x;
    if (vector >= n_vectors) {
        return;
    }
    output[vector*qwen38_sa_d + threadIdx.x] =
        running[vector*qwen38_sa_d + threadIdx.x]/running_meta[vector].y;
}

static __global__ void qwen38_streamed_attn_mask_limits_kernel(
        const half * __restrict__ remote_mask,
        const int mask_stride,
        const int q_tokens,
        const int kv_tokens,
        int * __restrict__ limits) {
    const int query = blockIdx.x*blockDim.x + threadIdx.x;
    if (query >= q_tokens) {
        return;
    }
    const half * row = remote_mask + int64_t(query)*mask_stride;
    int low = 0;
    int high = kv_tokens;
    while (low < high) {
        const int middle = low + (high - low)/2;
        if (__half2float(row[middle]) > -10000.0f) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    limits[query] = low;
}

static __global__ void qwen38_streamed_attn_make_mask_kernel(
        const int * __restrict__ limits,
        half * __restrict__ tile_mask,
        const int q_tokens,
        const int tile_start,
        const int tile_tokens) {
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t n = int64_t(q_tokens)*tile_tokens;
    if (index >= n) {
        return;
    }
    const int query = index/tile_tokens;
    const int key = index - int64_t(query)*tile_tokens;
    tile_mask[index] = tile_start + key < limits[query] ?
        __float2half(0.0f) : __float2half(-65504.0f);
}

static size_t qwen38_streamed_attn_shared_bytes(const int cc) {
    constexpr int DKQ = qwen38_sa_d;
    constexpr int DV  = qwen38_sa_d;
    constexpr int ncols = qwen38_sa_ncols;
    const int nthreads       = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, cc);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, cc);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, cc);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, ncols, cc);
    const int nwarps         = nthreads/32;
    const int cols_per_warp  = get_cols_per_warp(cc);

    const size_t shared_kv = nbatch_fa*(nbatch_K2 + 4)*sizeof(half2) +
        q8_kv256_async_alignment + nbatch_fa*q8_kv256_async_row_stride;
    const size_t shared_q = ncols*(DKQ/2 + 4)*sizeof(half2);
    const size_t shared_mask = qwen38_sa_ncols1*(nbatch_fa/2 + 4)*sizeof(half2);
    const size_t shared_combine = nwarps*cols_per_warp*(nbatch_combine + 4)*sizeof(half2);
    return std::max(shared_combine,
        Q_in_reg ? std::max(shared_q, shared_kv + shared_mask) : shared_q + shared_kv + shared_mask);
}

static cudaError_t qwen38_streamed_attn_launch_raw(
        const float * q,
        const void  * k_head_major,
        const void  * v_head_major,
        const void  * mask,
        const int mask_stride,
        const int mask_key_offset,
        const int q_tokens,
        const int kv_tokens,
        const int kv_head_stride_tokens,
        void * raw,
        const float scale,
        cudaStream_t stream) {
    if (!q || !k_head_major || !v_head_major || !mask || !raw ||
            q_tokens <= 0 ||
            kv_tokens <= 0 || kv_tokens % FATTN_KQ_STRIDE != 0 ||
            kv_head_stride_tokens < kv_tokens ||
            mask_stride < mask_key_offset + kv_tokens || mask_key_offset < 0) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return err;
    }
    cudaDeviceProp prop = {};
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return err;
    }
    if (prop.major < 12) {
        return cudaErrorNotSupported;
    }
    const int cc = 100*prop.major + 10*prop.minor;
    const int nthreads = ggml_cuda_fattn_mma_get_nthreads(
        qwen38_sa_d, qwen38_sa_d, qwen38_sa_ncols, cc);
    if (nthreads <= 0 || nthreads % 32 != 0) {
        return cudaErrorInvalidConfiguration;
    }
    const int nwarps = nthreads/32;
    const size_t nbytes_shared = qwen38_streamed_attn_shared_bytes(cc);
    const int nblocks = qwen38_sa_blocks(q_tokens);
    const uint3 ne01 = init_fastdiv_values(q_tokens);
    const half * mask_tile = static_cast<const half *>(mask) + mask_key_offset;

    if (nwarps == 4) {
        err = cudaFuncSetAttribute(qwen38_streamed_attn_partial_kernel<4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbytes_shared));
        if (err != cudaSuccess) {
            return err;
        }
        qwen38_streamed_attn_partial_kernel<4><<<nblocks, dim3(32, 4, 1), nbytes_shared, stream>>>(
            q, static_cast<const char *>(k_head_major), static_cast<const char *>(v_head_major),
            mask_tile, static_cast<float2 *>(raw), scale, ne01, q_tokens, kv_tokens,
            kv_head_stride_tokens, mask_stride);
    } else if (nwarps == 8) {
        err = cudaFuncSetAttribute(qwen38_streamed_attn_partial_kernel<8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbytes_shared));
        if (err != cudaSuccess) {
            return err;
        }
        qwen38_streamed_attn_partial_kernel<8><<<nblocks, dim3(32, 8, 1), nbytes_shared, stream>>>(
            q, static_cast<const char *>(k_head_major), static_cast<const char *>(v_head_major),
            mask_tile, static_cast<float2 *>(raw), scale, ne01, q_tokens, kv_tokens,
            kv_head_stride_tokens, mask_stride);
    } else {
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

} // namespace

extern "C" size_t qwen38_cuda_streamed_attn_raw_bytes(const int q_tokens) {
    if (q_tokens <= 0) {
        return 0;
    }
    const size_t nblocks = qwen38_sa_blocks(q_tokens);
    const size_t nfloat2 = nblocks*(2*qwen38_sa_ncols + qwen38_sa_ncols*(qwen38_sa_d/2));
    return nfloat2*sizeof(float2);
}

extern "C" size_t qwen38_cuda_streamed_attn_parts_bytes(const int q_tokens, const int n_parts) {
    if (q_tokens <= 0 || n_parts <= 0) {
        return 0;
    }
    return size_t(q_tokens)*qwen38_sa_q_heads*n_parts*qwen38_sa_d*sizeof(float);
}

extern "C" size_t qwen38_cuda_streamed_attn_meta_bytes(const int q_tokens, const int n_parts) {
    if (q_tokens <= 0 || n_parts <= 0) {
        return 0;
    }
    return size_t(q_tokens)*qwen38_sa_q_heads*n_parts*sizeof(float2);
}

extern "C" size_t qwen38_cuda_streamed_attn_output_bytes(const int q_tokens) {
    if (q_tokens <= 0) {
        return 0;
    }
    return size_t(q_tokens)*qwen38_sa_q_heads*qwen38_sa_d*sizeof(float);
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_partial(
        const float * q,
        const void  * k_head_major,
        const void  * v_head_major,
        const void  * mask,
        const int mask_stride,
        const int mask_key_offset,
        const int q_tokens,
        const int kv_tokens,
        const int kv_head_stride_tokens,
        const int part,
        const int n_parts,
        void * raw,
        float * parts,
        void * meta,
        const float scale,
        cudaStream_t stream) {
    if (!parts || !meta || part < 0 || part >= n_parts) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = qwen38_streamed_attn_launch_raw(
        q, k_head_major, v_head_major, mask, mask_stride, mask_key_offset,
        q_tokens, kv_tokens, kv_head_stride_tokens, raw, scale, stream);
    if (err != cudaSuccess) {
        return err;
    }

    const int64_t n = int64_t(q_tokens)*qwen38_sa_q_heads*qwen38_sa_d;
    constexpr int scatter_threads = 256;
    qwen38_streamed_attn_scatter<<<(n + scatter_threads - 1)/scatter_threads,
        scatter_threads, 0, stream>>>(
            static_cast<const float2 *>(raw), parts, static_cast<float2 *>(meta),
            q_tokens, part, n_parts);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_partial_merge(
        const float * q,
        const void  * k_head_major,
        const void  * v_head_major,
        const void  * mask,
        const int mask_stride,
        const int mask_key_offset,
        const int q_tokens,
        const int kv_tokens,
        const int kv_head_stride_tokens,
        const bool first,
        void * raw,
        float * running,
        void * running_meta,
        const float scale,
        cudaStream_t stream) {
    if (!running || !running_meta) {
        return cudaErrorInvalidValue;
    }
    cudaError_t err = qwen38_streamed_attn_launch_raw(
        q, k_head_major, v_head_major, mask, mask_stride, mask_key_offset,
        q_tokens, kv_tokens, kv_head_stride_tokens, raw, scale, stream);
    if (err != cudaSuccess) {
        return err;
    }
    const int64_t n_vectors = int64_t(q_tokens)*qwen38_sa_q_heads;
    qwen38_streamed_attn_merge_raw<<<n_vectors, qwen38_sa_d, 0, stream>>>(
        static_cast<const float2 *>(raw), running, static_cast<float2 *>(running_meta),
        q_tokens, first);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_finalize(
        const float * running,
        const void  * running_meta,
        float * output,
        const int q_tokens,
        cudaStream_t stream) {
    if (!running || !running_meta || !output || q_tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t n_vectors = int64_t(q_tokens)*qwen38_sa_q_heads;
    qwen38_streamed_attn_finalize_kernel<<<n_vectors, qwen38_sa_d, 0, stream>>>(
        running, static_cast<const float2 *>(running_meta), output, n_vectors);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_mask_limits(
        const void * remote_mask,
        const int mask_stride,
        const int q_tokens,
        const int kv_tokens,
        int * limits,
        cudaStream_t stream) {
    if (!remote_mask || !limits || mask_stride < kv_tokens || q_tokens <= 0 || kv_tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    constexpr int threads = 128;
    qwen38_streamed_attn_mask_limits_kernel<<<(q_tokens + threads - 1)/threads,
        threads, 0, stream>>>(
            static_cast<const half *>(remote_mask), mask_stride, q_tokens, kv_tokens, limits);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_make_mask(
        const int * limits,
        void * tile_mask,
        const int q_tokens,
        const int tile_start,
        const int tile_tokens,
        cudaStream_t stream) {
    if (!limits || !tile_mask || q_tokens <= 0 || tile_start < 0 || tile_tokens <= 0) {
        return cudaErrorInvalidValue;
    }
    const int64_t n = int64_t(q_tokens)*tile_tokens;
    constexpr int threads = 256;
    qwen38_streamed_attn_make_mask_kernel<<<(n + threads - 1)/threads,
        threads, 0, stream>>>(
            limits, static_cast<half *>(tile_mask), q_tokens, tile_start, tile_tokens);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_streamed_attn_combine(
        const float * parts,
        const void  * meta,
        float * output,
        const int q_tokens,
        const int n_parts,
        cudaStream_t stream) {
    if (!parts || !meta || !output || q_tokens <= 0 || n_parts <= 0 || n_parts > 256) {
        return cudaErrorInvalidValue;
    }
    const dim3 blocks(q_tokens, qwen38_sa_q_heads, 1);
    flash_attn_combine_results<qwen38_sa_d><<<blocks, qwen38_sa_d,
        size_t(n_parts)*sizeof(float2), stream>>>(
            parts, static_cast<const float2 *>(meta), output, n_parts);
    return cudaGetLastError();
}
