#pragma once

#include <cuda_runtime.h>

#include <cstddef>

// Experimental fixed-shape entry points used by the Qwen3.8 two-GPU
// streamed-attention proof.  K and V are RTX-local, head-major Q8_KV256
// tiles: [4 KV heads][kv_tokens][258 bytes].  Q is F32 in ggml's
// [256][q_tokens][24 heads] physical layout.  The mask is F16 with one
// full-cache row per query; mask_key_offset selects the current tile.
//
// These entry points intentionally sit outside normal graph dispatch.  They
// let the hardware experiment reuse the production SM120 direct-Q8 MMA body
// and the existing numerically stable FP32 partial-result combiner without
// committing the ring scheduler to the server hot path.

extern "C" size_t qwen38_cuda_streamed_attn_raw_bytes(int q_tokens);
extern "C" size_t qwen38_cuda_streamed_attn_parts_bytes(int q_tokens, int n_parts);
extern "C" size_t qwen38_cuda_streamed_attn_meta_bytes(int q_tokens, int n_parts);
extern "C" size_t qwen38_cuda_streamed_attn_output_bytes(int q_tokens);

extern "C" cudaError_t qwen38_cuda_streamed_attn_partial(
        const float * q,
        const void  * k_head_major,
        const void  * v_head_major,
        const void  * mask,
        int mask_stride,
        int mask_key_offset,
        int q_tokens,
        int kv_tokens,
        int kv_head_stride_tokens,
        int part,
        int n_parts,
        void * raw,
        float * parts,
        void * meta,
        float scale,
        cudaStream_t stream);

extern "C" cudaError_t qwen38_cuda_streamed_attn_combine(
        const float * parts,
        const void  * meta,
        float * output,
        int q_tokens,
        int n_parts,
        cudaStream_t stream);

// Memory-bounded production candidate.  Each call merges one raw tile into a
// single running FP32 numerator/meta pair in increasing KV-position order.
// `first` initializes the state.  Finalize divides the running numerator by
// its accumulated softmax denominator and may operate in place.
extern "C" cudaError_t qwen38_cuda_streamed_attn_partial_merge(
        const float * q,
        const void  * k_head_major,
        const void  * v_head_major,
        const void  * mask,
        int mask_stride,
        int mask_key_offset,
        int q_tokens,
        int kv_tokens,
        int kv_head_stride_tokens,
        bool first,
        void * raw,
        float * running,
        void * running_meta,
        float scale,
        cudaStream_t stream);

extern "C" cudaError_t qwen38_cuda_streamed_attn_finalize(
        const float * running,
        const void  * running_meta,
        float * output,
        int q_tokens,
        cudaStream_t stream);

// Extract the monotonic causal/prefix boundary from each source mask row with
// logarithmic remote reads, then materialize only the current RTX-local tile.
// `remote_mask` may point at peer-visible P100 memory.
extern "C" cudaError_t qwen38_cuda_streamed_attn_mask_limits(
        const void * remote_mask,
        int mask_stride,
        int q_tokens,
        int kv_tokens,
        int * limits,
        cudaStream_t stream);

extern "C" cudaError_t qwen38_cuda_streamed_attn_make_mask(
        const int * limits,
        void * tile_mask,
        int q_tokens,
        int tile_start,
        int tile_tokens,
        cudaStream_t stream);
