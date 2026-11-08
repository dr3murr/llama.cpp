#pragma once

#include "common.cuh"

// Returns true when an opt-in P100 attention operation was submitted to the
// bounded RTX ring and the source stream was made dependent on its result.
// False means normal local dispatch should continue.
bool ggml_cuda_qwen38_streamed_attn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Diagnostic Blackwell-local counterpart.  It reuses the already-qualified
// bounded ring created by the P100 producer, avoiding the full-cache
// head-major Q8 scratch used by the ordinary direct-Q8 MMA path.
bool ggml_cuda_qwen38_bounded_local_attn_supported(
    int device, const ggml_tensor * dst);

bool ggml_cuda_qwen38_bounded_local_attn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst);
