#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// Raw fixed-shape interface for the opt-in Qwen3.8 SageAttention backend.
// K is signed INT8 [4, capacity, 256], V is E4M3 [4, 256, padded_capacity],
// and scales are FP32. Q and output use ggml's F32 [24, q, 256] layout.

extern "C" size_t qwen38_cuda_sage_q_bytes(int q_tokens);
extern "C" size_t qwen38_cuda_sage_q_scale_bytes(int q_tokens);
extern "C" size_t qwen38_cuda_sage_output_half_bytes(int q_tokens);
extern "C" size_t qwen38_cuda_sage_query_limit_bytes(int q_tokens);

// True only while SET_ROWS is maintaining this allocation in the Sage-native
// representation.  FLASH_ATTN_EXT checks this instead of inferring the cache
// representation from thresholds, which also makes per-layer experiments
// safe.
extern "C" bool qwen38_cuda_sage_cache_is_active(const void * storage);

// Incrementally maintain the resident cache representation.  The allocation
// for each cache is the ordinary head-major Q8_KV256 allocation
// (4*capacity*258 bytes); the compact payload and its metadata live inside it.
// Source rows and indices are token-major: index = head*capacity + token.
extern "C" cudaError_t qwen38_cuda_sage_append_k(
        const float * src,
        const int64_t * indices,
        void * storage,
        int rows,
        int capacity,
        size_t src_row_stride,
        cudaStream_t stream);

extern "C" cudaError_t qwen38_cuda_sage_append_v(
        const float * src,
        const int64_t * indices,
        void * storage,
        int rows,
        int capacity,
        int valid_tokens,
        size_t src_row_stride,
        cudaStream_t stream);

// Convert the same allocation between ordinary head-major Q8_KV256 rows and
// the Sage resident representation.  This permits Sage prefill without
// sacrificing the much faster existing Q8 decode kernel.
extern "C" cudaError_t qwen38_cuda_sage_convert_cache(
        void * storage,
        int capacity,
        int valid_tokens,
        int kind,
        bool to_sage,
        cudaStream_t stream);

extern "C" cudaError_t qwen38_cuda_sage_attention(
        const float * q,
        const int8_t * k,
        const int8_t * v,
        const float * k_scale,
        const float * v_scale,
        float * output,
        int8_t * q_quant,
        float * q_scale,
        void * output_half,
        int * query_limits,
        int q_tokens,
        int kv_tokens,
        int kv_capacity,
        bool causal,
        const void * mask,
        size_t mask_row_stride,
        int k_scale_head_stride,
        cudaStream_t stream);
