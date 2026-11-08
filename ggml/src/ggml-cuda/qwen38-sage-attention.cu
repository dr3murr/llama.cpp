#include "qwen38-sage-attention.cuh"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>

// SageAttention is kept as a pinned sibling source tree. These headers are
// Apache-2.0 and retain their upstream copyright/license notices. The
// SAGEATTN_STANDALONE guard removes only the PyTorch binding dependency.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
#define SAGEATTN_STANDALONE 1
#include "../../../../SageAttention-reference/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"

namespace {

constexpr int qwen38_sage_q_heads = 24;
constexpr int qwen38_sage_kv_heads = 4;
constexpr int qwen38_sage_gqa = 6;
constexpr int qwen38_sage_head_dim = 256;
constexpr int qwen38_sage_cta_q = 128;
constexpr int qwen38_sage_cta_k = 64;
// Sage's per-thread SM89/SM120 FP8 path is defined around 32 query rows per
// warp.  The earlier per-warp prototype used 16, but that scale-to-MMA lane
// mapping is not valid for the per-thread kernel.
constexpr int qwen38_sage_warp_q = 32;
constexpr int qwen38_sage_warp_k = 64;
constexpr int qwen38_sage_q_groups_per_cta = qwen38_sage_cta_q/qwen38_sage_warp_q;
constexpr int qwen38_sage_q_scales_per_group = 8;
constexpr int qwen38_sage_k_scales_per_block = 4;
constexpr int qwen38_sage_storage_row_bytes = 258;

int qwen38_sage_padded_q(const int q_tokens) {
    return (q_tokens + qwen38_sage_cta_q - 1)/qwen38_sage_cta_q*qwen38_sage_cta_q;
}

// Q8_KV256 contributes eight spare bytes per token to the fixed cache
// allocation. Per-thread K scales consume one of those bytes per token; the
// remaining seven bytes/token must also fit the fixed 4x256 FP32 K mean.
// Production contexts easily satisfy this. Tiny validation contexts do not,
// so they use a zero mean (which is mathematically exact for attention logits,
// with only a small quantization-accuracy cost) rather than accessing beyond
// the cache allocation.
__host__ __device__ constexpr bool qwen38_sage_k_mean_fits(const int capacity) {
    return int64_t(7)*capacity >=
        int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*int64_t(sizeof(float));
}

__host__ __device__ constexpr bool qwen38_sage_v_mean_fits(const int capacity) {
    return int64_t(8)*capacity >=
        int64_t(2)*qwen38_sage_kv_heads*qwen38_sage_head_dim*int64_t(sizeof(float));
}

// A full 262k Q8_KV256 layer occupies roughly 270 MiB.  The production RTX
// layout intentionally leaves much less free VRAM than that, so cache-format
// transitions cannot use a same-sized device temporary.  Keep one reusable,
// mapped pinned-host image instead.  Conversion happens only at the
// prefill/decode phase boundary; preserving the production VRAM budget is much
// more valuable than avoiding this one PCIe round trip.
//
// The event also makes reuse safe if independent graph streams reach different
// attention layers.  The mutex protects enqueue order; the event protects the
// asynchronous use of the shared staging image after the mutex is released.
struct qwen38_sage_host_stage {
    std::mutex mutex;
    void * host = nullptr;
    void * device = nullptr;
    size_t size = 0;
    cudaEvent_t ready = nullptr;
    bool recorded = false;
};

qwen38_sage_host_stage qwen38_sage_stage;

static __device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

static __device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

static __global__ void qwen38_sage_quantize_q_kernel(
        const float * __restrict__ q,
        int8_t * __restrict__ q_quant,
        float * __restrict__ q_scale,
        const int q_tokens,
        const int q_padded,
        const int q_pad,
        const int q_scale_stride) {
    const int scale_group = blockIdx.x % qwen38_sage_q_scales_per_group;
    const int q_group = blockIdx.x / qwen38_sage_q_scales_per_group;
    const int head = blockIdx.y;
    const int token_start = q_group*qwen38_sage_warp_q;
    const int d = threadIdx.x;

    float maximum = 0.0f;
#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_warp_q; ++local_token) {
        if (local_token % qwen38_sage_q_scales_per_group != scale_group) {
            continue;
        }
        const int padded_token = token_start + local_token;
        const int token = padded_token - q_pad;
        if (padded_token < q_padded && token >= 0 && token < q_tokens) {
            const int64_t index = (int64_t(head)*q_tokens + token)*qwen38_sage_head_dim + d;
            maximum = fmaxf(maximum, fabsf(q[index]));
        }
    }

    maximum = warp_max(maximum);
    __shared__ float warp_maximum[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_maximum[warp] = maximum;
    }
    __syncthreads();

    if (warp == 0) {
        maximum = lane < 8 ? warp_maximum[lane] : 0.0f;
        maximum = warp_max(maximum);
        if (lane == 0) {
            warp_maximum[0] = maximum;
        }
    }
    __syncthreads();

    const float scale = fmaxf(warp_maximum[0]/127.0f, 1.0e-8f);
    if (threadIdx.x == 0) {
        q_scale[head*q_scale_stride +
            q_group*qwen38_sage_q_scales_per_group + scale_group] = scale;
    }

#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_warp_q; ++local_token) {
        if (local_token % qwen38_sage_q_scales_per_group != scale_group) {
            continue;
        }
        const int padded_token = token_start + local_token;
        if (padded_token < q_padded) {
            const int token = padded_token - q_pad;
            const int64_t output_index =
                (int64_t(head)*q_padded + padded_token)*qwen38_sage_head_dim + d;
            if (token >= 0 && token < q_tokens) {
                const int64_t input_index =
                    (int64_t(head)*q_tokens + token)*qwen38_sage_head_dim + d;
                const int quantized = max(-127, min(127, __float2int_rn(q[input_index]/scale)));
                q_quant[output_index] = int8_t(quantized);
            } else {
                q_quant[output_index] = 0;
            }
        }
    }
}

static __global__ void qwen38_sage_half_to_float_kernel(
        const half * __restrict__ input,
        float * __restrict__ output,
        const int q_tokens,
        const int q_padded,
        const int q_pad) {
    const int64_t count = int64_t(qwen38_sage_q_heads)*q_tokens*qwen38_sage_head_dim;
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < count) {
        const int d = index % qwen38_sage_head_dim;
        const int64_t vector = index/qwen38_sage_head_dim;
        const int token = vector % q_tokens;
        const int head = vector/q_tokens;
        const int64_t input_index =
            (int64_t(head)*q_padded + token + q_pad)*qwen38_sage_head_dim + d;
        output[index] = __half2float(input[input_index]);
    }
}

static __global__ void qwen38_sage_count_nonfinite_kernel(
        const float * __restrict__ values,
        const int64_t count,
        unsigned int * __restrict__ nonfinite) {
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < count && !isfinite(values[index])) {
        atomicAdd(nonfinite, 1U);
    }
}

static __global__ void qwen38_sage_query_limits_kernel(
        const half * __restrict__ mask,
        const int mask_stride,
        const int q_tokens,
        const int q_padded,
        const int q_pad,
        const int kv_tokens,
        int * __restrict__ limits) {
    const int query = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (query >= q_padded) {
        return;
    }
    const int source_query = min(q_tokens - 1, max(0, query - q_pad));
    const half * const row = mask + int64_t(source_query)*mask_stride;
    int low = 0;
    int high = kv_tokens;
    while (low < high) {
        const int middle = low + (high - low)/2;
        const float value = __half2float(row[middle]);
        if (isfinite(value) && value > -10000.0f) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    limits[query] = query < q_pad ? max(0, low - (q_pad - query)) : low;
}

// Per-thread Sage K quantization uses four interleaved scales per 64-token
// block. Updating a partial block therefore dequantizes and requantizes the
// corresponding 16-token lane group. Only the first update targeting a block
// performs the work for each scale group.
static __global__ void qwen38_sage_append_k_kernel(
        const float * __restrict__ src,
        const int64_t * __restrict__ indices,
        int8_t * __restrict__ storage,
        const int rows,
        const int capacity,
        const size_t src_row_stride) {
    const int update = blockIdx.x;
    const int d = threadIdx.x;
    const int64_t destination = indices[update];
    const int64_t block_key = destination/qwen38_sage_cta_k;
    const int scale_group = blockIdx.y;

    __shared__ int owner;
    __shared__ int replacement[qwen38_sage_cta_k];
    __shared__ float warp_maximum[8];

    if (threadIdx.x == 0) {
        owner = 1;
        for (int i = 0; i < update; ++i) {
            if (indices[i]/qwen38_sage_cta_k == block_key) {
                owner = 0;
                break;
            }
        }
    }
    if (threadIdx.x < qwen38_sage_cta_k) {
        replacement[threadIdx.x] = -1;
    }
    __syncthreads();
    if (!owner) {
        return;
    }

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        if (indices[i]/qwen38_sage_cta_k == block_key) {
            replacement[indices[i] % qwen38_sage_cta_k] = i;
        }
    }
    __syncthreads();

    const int head = int(destination/capacity);
    const int token_block = int((destination % capacity)/qwen38_sage_cta_k);
    const int blocks_per_head = capacity/qwen38_sage_cta_k;
    float * const scale = reinterpret_cast<float *>(
        storage + int64_t(qwen38_sage_kv_heads)*capacity*qwen38_sage_head_dim);
    const float * const mean = qwen38_sage_k_mean_fits(capacity) ? scale +
        qwen38_sage_kv_heads*blocks_per_head*qwen38_sage_k_scales_per_block : nullptr;
    const float mean_value = mean ? mean[head*qwen38_sage_head_dim + d] : 0.0f;
    const int scale_index = head*blocks_per_head*qwen38_sage_k_scales_per_block +
        token_block*qwen38_sage_k_scales_per_block + scale_group;
    const float old_scale = scale[scale_index];
    const int64_t row_base = int64_t(head)*capacity + token_block*qwen38_sage_cta_k;

    float maximum = 0.0f;
#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_cta_k; ++local_token) {
        if ((local_token % 8)/2 != scale_group) {
            continue;
        }
        const int source_row = replacement[local_token];
        const float value = source_row >= 0 ?
            *reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(src) + int64_t(source_row)*src_row_stride + d*sizeof(float)) -
                    mean_value :
            float(storage[(row_base + local_token)*qwen38_sage_head_dim + d])*old_scale;
        maximum = fmaxf(maximum, fabsf(value));
    }

    maximum = warp_max(maximum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_maximum[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < 8 ? warp_maximum[lane] : 0.0f;
        maximum = warp_max(maximum);
        if (lane == 0) {
            warp_maximum[0] = maximum;
        }
    }
    __syncthreads();

    const float new_scale = fmaxf(warp_maximum[0]/127.0f, 1.0e-8f);
    if (threadIdx.x == 0) {
        scale[scale_index] = new_scale;
    }

#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_cta_k; ++local_token) {
        if ((local_token % 8)/2 != scale_group) {
            continue;
        }
        const int source_row = replacement[local_token];
        const int64_t index = (row_base + local_token)*qwen38_sage_head_dim + d;
        const float value = source_row >= 0 ?
            *reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(src) + int64_t(source_row)*src_row_stride + d*sizeof(float)) -
                    mean_value :
            float(storage[index])*old_scale;
        storage[index] = int8_t(max(-127, min(127, __float2int_rn(value/new_scale))));
    }
}

static __device__ __forceinline__ int qwen38_sage_v_permuted_token(const int token) {
    const int block = token & ~63;
    const int local = token & 63;
    const int group = local & ~15;
    const int row = local & 15;
    return block + group + (row/8)*2 + ((row/2) % 4)*4 + row % 2;
}

static __device__ __forceinline__ float qwen38_sage_load_fp8(int8_t value);
static __device__ __forceinline__ float qwen38_sage_block_max(float value, float * warp_values);
static __device__ __forceinline__ float qwen38_sage_block_sum(float value, float * warp_values);

static __global__ void qwen38_sage_append_v_kernel(
        const float * __restrict__ src,
        const int64_t * __restrict__ indices,
        int8_t * __restrict__ storage,
        float * __restrict__ scale,
        const float * __restrict__ mean,
        const int rows,
        const int capacity,
        const int valid_tokens,
        const size_t src_row_stride) {
    const int head = blockIdx.x;
    const int d = blockIdx.y;
    const int scale_index = head*qwen38_sage_head_dim + d;
    const float old_scale = scale[scale_index];
    const float mean_value = mean ? mean[scale_index] : 0.0f;
    __shared__ float warp_values[8];

    // A per-channel FP8 scale is global to the resident prefix.  Recompute it
    // from both the old cache and this append, then requantize the old prefix.
    // This is O(KV) per channel but only O(1/Q) of the following attention
    // work for a Q-token prefill batch, and avoids clipping a new chunk to a
    // scale learned from an earlier prefix.
    float maximum = 0.0f;
    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        const int8_t quantized = storage[
            (int64_t(head)*qwen38_sage_head_dim + d)*capacity +
            qwen38_sage_v_permuted_token(token)];
        maximum = fmaxf(maximum, fabsf(qwen38_sage_load_fp8(quantized)*old_scale));
    }
    for (int update = threadIdx.x; update < rows; update += blockDim.x) {
        if (indices[update]/capacity == head) {
            const float value = *reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(src) + int64_t(update)*src_row_stride + d*sizeof(float));
            maximum = fmaxf(maximum, fabsf(value - mean_value));
        }
    }
    maximum = qwen38_sage_block_max(maximum, warp_values);
    const float new_scale = fmaxf(maximum/448.0f, 1.0e-8f);

    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        const int64_t output_index =
            (int64_t(head)*qwen38_sage_head_dim + d)*capacity +
            qwen38_sage_v_permuted_token(token);
        const float value = qwen38_sage_load_fp8(storage[output_index])*old_scale/new_scale;
        const __nv_fp8_e4m3 quantized(fminf(448.0f, fmaxf(-448.0f, value)));
        storage[output_index] = *reinterpret_cast<const int8_t *>(&quantized);
    }
    for (int update = threadIdx.x; update < rows; update += blockDim.x) {
        const int64_t destination = indices[update];
        if (destination/capacity == head) {
            const int token = int(destination % capacity);
            const float value = *reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(src) + int64_t(update)*src_row_stride + d*sizeof(float));
            const float scaled = isfinite(value) ?
                fminf(448.0f, fmaxf(-448.0f, (value - mean_value)/new_scale)) : 0.0f;
            const __nv_fp8_e4m3 quantized(scaled);
            storage[(int64_t(head)*qwen38_sage_head_dim + d)*capacity +
                qwen38_sage_v_permuted_token(token)] = *reinterpret_cast<const int8_t *>(&quantized);
        }
    }
    if (threadIdx.x == 0) {
        scale[scale_index] = new_scale;
    }
}

static __global__ void qwen38_sage_initialize_v_scale_kernel(
        float * __restrict__ scale) {
    const int index = int(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index < qwen38_sage_kv_heads*qwen38_sage_head_dim) {
        scale[index] = 1.0f;
    }
}

static __device__ __forceinline__ float qwen38_sage_load_q8_row(
        const int8_t * storage,
        const int64_t row,
        const int d) {
    const char * const source = reinterpret_cast<const char *>(storage) +
        row*qwen38_sage_storage_row_bytes;
    return __half2float(*reinterpret_cast<const half *>(source))*
        float(*reinterpret_cast<const int8_t *>(source + sizeof(half) + d));
}

static __device__ __forceinline__ float qwen38_sage_load_fp8(const int8_t value) {
    __nv_fp8_e4m3 converted;
    *reinterpret_cast<int8_t *>(&converted) = value;
    return float(converted);
}

static __device__ __forceinline__ float qwen38_sage_block_max(
        float value,
        float * warp_values) {
    value = warp_max(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? warp_values[lane] : 0.0f;
        value = warp_max(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

static __device__ __forceinline__ float qwen38_sage_block_sum(
        float value,
        float * warp_values) {
    value = warp_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < 8 ? warp_values[lane] : 0.0f;
        value = warp_sum(value);
        if (lane == 0) {
            warp_values[0] = value;
        }
    }
    __syncthreads();
    return warp_values[0];
}

static __global__ void qwen38_q8_to_sage_k_kernel(
        const int8_t * __restrict__ source,
        int8_t * __restrict__ destination,
        const float * __restrict__ mean,
        const int capacity,
        const int valid_tokens) {
    const int token_block = blockIdx.x;
    const int head = blockIdx.y;
    const int scale_group = blockIdx.z;
    const int d = threadIdx.x;
    const int64_t row_base = int64_t(head)*capacity + token_block*qwen38_sage_cta_k;
    __shared__ float warp_values[8];

    float maximum = 0.0f;
#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_cta_k; ++local_token) {
        if ((local_token % 8)/2 != scale_group) {
            continue;
        }
        const int token = token_block*qwen38_sage_cta_k + local_token;
        const float value = token < valid_tokens ?
            qwen38_sage_load_q8_row(source, row_base + local_token, d) -
                (mean ? mean[head*qwen38_sage_head_dim + d] : 0.0f) : 0.0f;
        maximum = fmaxf(maximum, fabsf(value));
    }
    maximum = qwen38_sage_block_max(maximum, warp_values);
    const float scale = fmaxf(maximum/127.0f, 1.0e-8f);
    float * const scales = reinterpret_cast<float *>(
        destination + int64_t(qwen38_sage_kv_heads)*capacity*qwen38_sage_head_dim);
    const int scale_index = head*(capacity/qwen38_sage_cta_k)*qwen38_sage_k_scales_per_block +
        token_block*qwen38_sage_k_scales_per_block + scale_group;
    if (threadIdx.x == 0) {
        scales[scale_index] = scale;
    }
#pragma unroll
    for (int local_token = 0; local_token < qwen38_sage_cta_k; ++local_token) {
        if ((local_token % 8)/2 != scale_group) {
            continue;
        }
        const int token = token_block*qwen38_sage_cta_k + local_token;
        const float value = token < valid_tokens ?
            qwen38_sage_load_q8_row(source, row_base + local_token, d) -
                (mean ? mean[head*qwen38_sage_head_dim + d] : 0.0f) : 0.0f;
        destination[(row_base + local_token)*qwen38_sage_head_dim + d] =
            int8_t(max(-127, min(127, __float2int_rn(value/scale))));
    }
}

static __global__ void qwen38_q8_k_mean_kernel(
        const int8_t * __restrict__ source,
        float * __restrict__ mean,
        const int capacity,
        const int valid_tokens) {
    const int head = blockIdx.x;
    const int d = blockIdx.y;
    __shared__ float warp_values[8];

    float sum = 0.0f;
    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        sum += qwen38_sage_load_q8_row(source, int64_t(head)*capacity + token, d);
    }
    sum = warp_sum(sum);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_values[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? warp_values[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) {
            mean[head*qwen38_sage_head_dim + d] = sum/float(valid_tokens);
        }
    }
}

static __global__ void qwen38_q8_to_sage_v_kernel(
        const int8_t * __restrict__ source,
        int8_t * __restrict__ destination,
        float * __restrict__ mean,
        const int capacity,
        const int valid_tokens) {
    const int head = blockIdx.x;
    const int d = blockIdx.y;
    __shared__ float warp_values[8];

    float sum = 0.0f;
    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        sum += qwen38_sage_load_q8_row(source, int64_t(head)*capacity + token, d);
    }
    const float mean_value = mean ? qwen38_sage_block_sum(sum, warp_values)/float(valid_tokens) : 0.0f;
    if (mean && threadIdx.x == 0) {
        mean[head*qwen38_sage_head_dim + d] = mean_value;
    }
    __syncthreads();

    float maximum = 0.0f;
    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(qwen38_sage_load_q8_row(
            source, int64_t(head)*capacity + token, d) - mean_value));
    }
    maximum = qwen38_sage_block_max(maximum, warp_values);
    const float scale = fmaxf(maximum/448.0f, 1.0e-8f);
    float * const scales = reinterpret_cast<float *>(
        destination + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
    if (threadIdx.x == 0) {
        scales[head*qwen38_sage_head_dim + d] = scale;
    }
    __syncthreads();

    for (int token = threadIdx.x; token < valid_tokens; token += blockDim.x) {
        const float value = qwen38_sage_load_q8_row(
            source, int64_t(head)*capacity + token, d);
        const float centered = (value - mean_value)/scale;
        const float finite_value = isfinite(centered) ?
            fminf(448.0f, fmaxf(-448.0f, centered)) : 0.0f;
        const __nv_fp8_e4m3 quantized(finite_value);
        destination[(int64_t(head)*qwen38_sage_head_dim + d)*capacity +
            qwen38_sage_v_permuted_token(token)] = *reinterpret_cast<const int8_t *>(&quantized);
    }
}

template<bool is_k>
static __global__ void qwen38_sage_to_q8_kernel(
        const int8_t * __restrict__ source,
        int8_t * __restrict__ destination,
        const int capacity) {
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    float value;
    if constexpr (is_k) {
        const float * const scales = reinterpret_cast<const float *>(
            source + int64_t(qwen38_sage_kv_heads)*capacity*qwen38_sage_head_dim);
        const float * const mean = qwen38_sage_k_mean_fits(capacity) ? scales +
            qwen38_sage_kv_heads*(capacity/qwen38_sage_cta_k)*qwen38_sage_k_scales_per_block : nullptr;
        const int scale_index = head*(capacity/qwen38_sage_cta_k)*qwen38_sage_k_scales_per_block +
            (token/qwen38_sage_cta_k)*qwen38_sage_k_scales_per_block + (token % 8)/2;
        value = float(source[(int64_t(head)*capacity + token)*qwen38_sage_head_dim + d])*
            scales[scale_index] +
            (mean ? mean[head*qwen38_sage_head_dim + d] : 0.0f);
    } else {
        const float * const scales = reinterpret_cast<const float *>(
            source + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
        const float * const mean = qwen38_sage_v_mean_fits(capacity) ? scales +
            qwen38_sage_kv_heads*qwen38_sage_head_dim : nullptr;
        value = qwen38_sage_load_fp8(source[
            (int64_t(head)*qwen38_sage_head_dim + d)*capacity +
            qwen38_sage_v_permuted_token(token)])*
            scales[head*qwen38_sage_head_dim + d] +
            (mean ? mean[head*qwen38_sage_head_dim + d] : 0.0f);
    }

    __shared__ float warp_values[8];
    const float maximum = qwen38_sage_block_max(fabsf(value), warp_values);
    const float scale = fmaxf(maximum/127.0f, 1.0e-8f);
    char * const row = reinterpret_cast<char *>(destination) +
        (int64_t(head)*capacity + token)*qwen38_sage_storage_row_bytes;
    if (threadIdx.x == 0) {
        *reinterpret_cast<half *>(row) = __float2half_rn(scale);
    }
    *reinterpret_cast<int8_t *>(row + sizeof(half) + d) =
        int8_t(max(-127, min(127, __float2int_rn(value/scale))));
}

template<MaskMode mask_mode, bool use_query_limits, bool use_v_mean>
static cudaError_t qwen38_sage_launch_attention(
        int8_t * q,
        const int8_t * k,
        const int8_t * v,
        half * output,
        float * q_scale,
        const float * k_scale,
        const float * v_scale,
        const float * v_mean,
        const int q_tokens,
        const int kv_tokens,
        const int kv_capacity,
        const half * mask,
        const int mask_stride_q,
        const int * query_limits,
        const int k_scale_head_stride,
        cudaStream_t stream) {
    constexpr int smem_bytes = qwen38_sage_cta_q*qwen38_sage_head_dim*sizeof(int8_t) +
        qwen38_sage_cta_k*qwen38_sage_head_dim*sizeof(int8_t) +
        qwen38_sage_cta_k*qwen38_sage_head_dim*sizeof(int8_t);
    using output_type = half;
    constexpr auto kernel = qk_int_sv_f8_attn_kernel<
        qwen38_sage_cta_q,
        qwen38_sage_cta_k,
        qwen38_sage_warp_q,
        qwen38_sage_warp_k,
        qwen38_sage_head_dim,
        DataType::kInt8,
        QuantGranularity::kPerThread,
        QuantGranularity::kPerThread,
        float,
        true,
        output_type,
        ComputeUnit::kCudaCore,
        mask_mode,
        false,
        true,
        use_v_mean,
        false,
        use_query_limits>;

    cudaError_t error = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (error != cudaSuccess) {
        return error;
    }

    const int q_ctas = (q_tokens + qwen38_sage_cta_q - 1)/qwen38_sage_cta_q;
    const int q_scale_stride = q_ctas*qwen38_sage_q_groups_per_cta*qwen38_sage_q_scales_per_group;
    const int k_scale_stride =
        (kv_tokens + qwen38_sage_cta_k - 1)/qwen38_sage_cta_k*qwen38_sage_k_scales_per_block;
    const int v_padded_capacity = (kv_capacity + qwen38_sage_cta_k - 1)/qwen38_sage_cta_k*qwen38_sage_cta_k;

    // The upstream kernel derives the K-scale head stride from logical KV
    // length. The raw ABI therefore accepts a compact logical scale array.
    const dim3 grid(q_ctas, qwen38_sage_q_heads, 1);
    const dim3 block(32, qwen38_sage_cta_q/qwen38_sage_warp_q, 1);
    kernel<<<grid, block, smem_bytes, stream>>>(
        q,
        const_cast<int8_t *>(k),
        const_cast<int8_t *>(v),
        output,
        nullptr,
        q_scale,
        const_cast<float *>(k_scale),
        const_cast<float *>(v_scale),
        const_cast<float *>(v_mean),
        q_tokens,
        kv_tokens,
        qwen38_sage_gqa,
        qwen38_sage_q_heads*q_tokens*qwen38_sage_head_dim,
        qwen38_sage_head_dim,
        q_tokens*qwen38_sage_head_dim,
        qwen38_sage_kv_heads*kv_capacity*qwen38_sage_head_dim,
        qwen38_sage_head_dim,
        kv_capacity*qwen38_sage_head_dim,
        qwen38_sage_kv_heads*qwen38_sage_head_dim*v_padded_capacity,
        qwen38_sage_head_dim*v_padded_capacity,
        v_padded_capacity,
        qwen38_sage_q_heads*q_tokens*qwen38_sage_head_dim,
        qwen38_sage_head_dim,
        q_tokens*qwen38_sage_head_dim,
        1.0f/sqrtf(float(qwen38_sage_head_dim)),
        use_query_limits ? nullptr : mask,
        use_query_limits ? 0 : mask_stride_q,
        query_limits,
        k_scale_head_stride);
    (void) q_scale_stride;
    (void) k_scale_stride;
    return cudaGetLastError();
}

} // namespace

extern "C" size_t qwen38_cuda_sage_q_bytes(const int q_tokens) {
    return q_tokens > 0 ?
        size_t(qwen38_sage_q_heads)*qwen38_sage_padded_q(q_tokens)*qwen38_sage_head_dim : 0;
}

extern "C" size_t qwen38_cuda_sage_q_scale_bytes(const int q_tokens) {
    if (q_tokens <= 0) {
        return 0;
    }
    const int q_ctas = (q_tokens + qwen38_sage_cta_q - 1)/qwen38_sage_cta_q;
    return size_t(qwen38_sage_q_heads)*q_ctas*qwen38_sage_q_groups_per_cta*
        qwen38_sage_q_scales_per_group*sizeof(float);
}

extern "C" size_t qwen38_cuda_sage_output_half_bytes(const int q_tokens) {
    return q_tokens > 0 ?
        size_t(qwen38_sage_q_heads)*qwen38_sage_padded_q(q_tokens)*qwen38_sage_head_dim*sizeof(half) : 0;
}

extern "C" size_t qwen38_cuda_sage_query_limit_bytes(const int q_tokens) {
    return q_tokens > 0 ? size_t(qwen38_sage_padded_q(q_tokens))*sizeof(int) : 0;
}

extern "C" cudaError_t qwen38_cuda_sage_append_k(
        const float * src,
        const int64_t * indices,
        void * storage,
        const int rows,
        const int capacity,
        const size_t src_row_stride,
        cudaStream_t stream) {
    if (!src || !indices || !storage || rows <= 0 || capacity <= 0 ||
            capacity % qwen38_sage_cta_k != 0 || src_row_stride < qwen38_sage_head_dim*sizeof(float)) {
        return cudaErrorInvalidValue;
    }
    qwen38_sage_append_k_kernel<<<
        dim3(rows, qwen38_sage_k_scales_per_block, 1), qwen38_sage_head_dim, 0, stream>>>(
        src, indices, static_cast<int8_t *>(storage), rows, capacity, src_row_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_sage_append_v(
        const float * src,
        const int64_t * indices,
        void * storage,
        const int rows,
        const int capacity,
        const int valid_tokens,
        const size_t src_row_stride,
        cudaStream_t stream) {
    if (!src || !indices || !storage || rows <= 0 || capacity <= 0 ||
            valid_tokens < 0 || valid_tokens > capacity ||
            capacity % qwen38_sage_cta_k != 0 || src_row_stride < qwen38_sage_head_dim*sizeof(float)) {
        return cudaErrorInvalidValue;
    }
    float * const scale = reinterpret_cast<float *>(
        static_cast<int8_t *>(storage) + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
    const float * const mean = qwen38_sage_v_mean_fits(capacity) ?
        scale + qwen38_sage_kv_heads*qwen38_sage_head_dim : nullptr;
    qwen38_sage_append_v_kernel<<<
        dim3(qwen38_sage_kv_heads, qwen38_sage_head_dim, 1), 256, 0, stream>>>(
            src, indices, static_cast<int8_t *>(storage), scale, mean,
            rows, capacity, valid_tokens, src_row_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t qwen38_cuda_sage_convert_cache(
        void * storage,
        const int capacity,
        const int valid_tokens,
        const int kind,
        const bool to_sage,
        cudaStream_t stream) {
    if (!storage || capacity <= 0 || capacity % qwen38_sage_cta_k != 0 ||
            valid_tokens < 0 || valid_tokens > capacity ||
            (kind != GGML_QWEN38_SAGE_CACHE_K && kind != GGML_QWEN38_SAGE_CACHE_V)) {
        return cudaErrorInvalidValue;
    }

    // A transition marker is emitted before SET_ROWS appends the current
    // query.  With an empty cache there is therefore no old representation to
    // preserve.  Avoid reading an uninitialized Q8 allocation; V still needs
    // its fixed unit-scale metadata before the direct Sage append.
    if (valid_tokens == 0) {
        if (to_sage) {
            float * const metadata = reinterpret_cast<float *>(
                static_cast<int8_t *>(storage) + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
            if (kind == GGML_QWEN38_SAGE_CACHE_V) {
                qwen38_sage_initialize_v_scale_kernel<<<
                    (qwen38_sage_kv_heads*qwen38_sage_head_dim + 255)/256, 256, 0, stream>>>(metadata);
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    return error;
                }
                if (qwen38_sage_v_mean_fits(capacity)) {
                    error = cudaMemsetAsync(
                        metadata + qwen38_sage_kv_heads*qwen38_sage_head_dim,
                        0,
                        qwen38_sage_kv_heads*qwen38_sage_head_dim*sizeof(float),
                        stream);
                }
                return error;
            } else {
                const size_t bytes = size_t(qwen38_sage_kv_heads)*capacity*qwen38_sage_storage_row_bytes;
                return cudaMemsetAsync(storage, 0, bytes, stream);
            }
            return cudaGetLastError();
        }
        return cudaSuccess;
    }

    const size_t storage_bytes = size_t(qwen38_sage_kv_heads)*capacity*qwen38_sage_storage_row_bytes;
    std::lock_guard<std::mutex> lock(qwen38_sage_stage.mutex);

    cudaError_t error = cudaSuccess;
    if (qwen38_sage_stage.size < storage_bytes) {
        if (qwen38_sage_stage.recorded) {
            error = cudaEventSynchronize(qwen38_sage_stage.ready);
            if (error != cudaSuccess) {
                return error;
            }
            qwen38_sage_stage.recorded = false;
        }
        if (qwen38_sage_stage.ready) {
            error = cudaEventDestroy(qwen38_sage_stage.ready);
            if (error != cudaSuccess) {
                return error;
            }
            qwen38_sage_stage.ready = nullptr;
        }
        if (qwen38_sage_stage.host) {
            error = cudaFreeHost(qwen38_sage_stage.host);
            if (error != cudaSuccess) {
                return error;
            }
            qwen38_sage_stage.host = nullptr;
            qwen38_sage_stage.device = nullptr;
            qwen38_sage_stage.size = 0;
        }
        error = cudaHostAlloc(
            &qwen38_sage_stage.host,
            storage_bytes,
            cudaHostAllocPortable | cudaHostAllocMapped);
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaHostGetDevicePointer(
            &qwen38_sage_stage.device, qwen38_sage_stage.host, 0);
        if (error != cudaSuccess) {
            cudaFreeHost(qwen38_sage_stage.host);
            qwen38_sage_stage.host = nullptr;
            return error;
        }
        error = cudaEventCreateWithFlags(
            &qwen38_sage_stage.ready, cudaEventDisableTiming);
        if (error != cudaSuccess) {
            cudaFreeHost(qwen38_sage_stage.host);
            qwen38_sage_stage.host = nullptr;
            qwen38_sage_stage.device = nullptr;
            return error;
        }
        qwen38_sage_stage.size = storage_bytes;
    } else if (qwen38_sage_stage.recorded) {
        error = cudaStreamWaitEvent(stream, qwen38_sage_stage.ready, 0);
        if (error != cudaSuccess) {
            return error;
        }
    }

    error = cudaMemcpyAsync(
        qwen38_sage_stage.host,
        storage,
        storage_bytes,
        cudaMemcpyDeviceToHost,
        stream);
    if (error != cudaSuccess) {
        return error;
    }
    const void * const source = qwen38_sage_stage.device;

    if (to_sage) {
        const dim3 grid(
            (valid_tokens + qwen38_sage_cta_k - 1)/qwen38_sage_cta_k,
            qwen38_sage_kv_heads,
            1);
        if (kind == GGML_QWEN38_SAGE_CACHE_K) {
            float * const scales = reinterpret_cast<float *>(
                static_cast<int8_t *>(storage) + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
            float * const mean = qwen38_sage_k_mean_fits(capacity) ? scales +
                qwen38_sage_kv_heads*(capacity/qwen38_sage_cta_k)*qwen38_sage_k_scales_per_block : nullptr;
            if (mean) {
                qwen38_q8_k_mean_kernel<<<
                    dim3(qwen38_sage_kv_heads, qwen38_sage_head_dim, 1), 256, 0, stream>>>(
                        static_cast<const int8_t *>(source), mean, capacity, valid_tokens);
            }
            qwen38_q8_to_sage_k_kernel<<<
                dim3(grid.x, grid.y, qwen38_sage_k_scales_per_block), qwen38_sage_head_dim, 0, stream>>>(
                static_cast<const int8_t *>(source), static_cast<int8_t *>(storage), mean, capacity, valid_tokens);
        } else {
            float * const scales = reinterpret_cast<float *>(
                static_cast<int8_t *>(storage) + int64_t(qwen38_sage_kv_heads)*qwen38_sage_head_dim*capacity);
            float * const mean = qwen38_sage_v_mean_fits(capacity) ?
                scales + qwen38_sage_kv_heads*qwen38_sage_head_dim : nullptr;
            qwen38_q8_to_sage_v_kernel<<<
                dim3(qwen38_sage_kv_heads, qwen38_sage_head_dim, 1), 256, 0, stream>>>(
                static_cast<const int8_t *>(source), static_cast<int8_t *>(storage), mean,
                capacity, valid_tokens);
        }
    } else if (kind == GGML_QWEN38_SAGE_CACHE_K) {
        qwen38_sage_to_q8_kernel<true><<<
            dim3(valid_tokens, qwen38_sage_kv_heads, 1), qwen38_sage_head_dim, 0, stream>>>(
                static_cast<const int8_t *>(source), static_cast<int8_t *>(storage), capacity);
    } else {
        qwen38_sage_to_q8_kernel<false><<<
            dim3(valid_tokens, qwen38_sage_kv_heads, 1), qwen38_sage_head_dim, 0, stream>>>(
                static_cast<const int8_t *>(source), static_cast<int8_t *>(storage), capacity);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        // The staging buffer cannot safely be reused while its D2H copy might
        // still be active.  This is an error path, where a synchronous cleanup
        // is preferable to concealing a subsequent race.
        cudaStreamSynchronize(stream);
        return error;
    }
    error = cudaEventRecord(qwen38_sage_stage.ready, stream);
    if (error == cudaSuccess) {
        qwen38_sage_stage.recorded = true;
    }
    return error;
}

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
        const int q_tokens,
        const int kv_tokens,
        const int kv_capacity,
        const bool causal,
        const void * mask,
        const size_t mask_row_stride,
        const int k_scale_head_stride,
        cudaStream_t stream) {
    if (!q || !k || !v || !k_scale || !v_scale || !output || !q_quant || !q_scale || !output_half ||
            q_tokens <= 0 || q_tokens > 512 || kv_tokens < q_tokens || kv_capacity < kv_tokens ||
            kv_capacity % qwen38_sage_cta_k != 0 ||
            k_scale_head_stride < 0 ||
            (mask && (!query_limits || mask_row_stride < size_t(kv_tokens)*sizeof(half) ||
                mask_row_stride % sizeof(half) != 0))) {
        return cudaErrorInvalidValue;
    }

    // Sage's causal epilogue is exact for complete CTA_Q tiles.  Left-pad only
    // the quantized representation so a partial llama.cpp ubatch retains the
    // same bottom-right causal positions as the original, unpadded Q tensor.
    // The dummy prefix is zero and discarded when the F16 result is unpacked.
    const int q_padded = qwen38_sage_padded_q(q_tokens);
    const int q_pad = q_padded - q_tokens;
    const int q_groups = q_padded/qwen38_sage_warp_q;
    const int q_scale_stride = q_padded/qwen38_sage_cta_q*qwen38_sage_q_groups_per_cta*
        qwen38_sage_q_scales_per_group;
    const float * const v_mean = qwen38_sage_v_mean_fits(kv_capacity) ?
        v_scale + qwen38_sage_kv_heads*qwen38_sage_head_dim : nullptr;
    qwen38_sage_quantize_q_kernel<<<
        dim3(q_groups*qwen38_sage_q_scales_per_group, qwen38_sage_q_heads, 1),
        qwen38_sage_head_dim, 0, stream>>>(
            q, q_quant, q_scale, q_tokens, q_padded, q_pad, q_scale_stride);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    if (mask) {
        qwen38_sage_query_limits_kernel<<<(q_padded + 255)/256, 256, 0, stream>>>(
            static_cast<const half *>(mask), int(mask_row_stride/sizeof(half)),
            q_tokens, q_padded, q_pad, kv_tokens, query_limits);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
    }

    if (causal) {
        if (mask) {
            if (v_mean) {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, true, true>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, v_mean,
                    q_padded, kv_tokens, kv_capacity, static_cast<const half *>(mask),
                    int(mask_row_stride/sizeof(half)), query_limits,
                    k_scale_head_stride, stream);
            } else {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, true, false>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, nullptr,
                    q_padded, kv_tokens, kv_capacity, static_cast<const half *>(mask),
                    int(mask_row_stride/sizeof(half)), query_limits,
                    k_scale_head_stride, stream);
            }
        } else {
            if (v_mean) {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, false, true>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, v_mean,
                    q_padded, kv_tokens, kv_capacity, nullptr, 0, nullptr,
                    k_scale_head_stride, stream);
            } else {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, false, false>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, nullptr,
                    q_padded, kv_tokens, kv_capacity, nullptr, 0, nullptr,
                    k_scale_head_stride, stream);
            }
        }
    } else {
        if (mask) {
            // llama.cpp's explicit mask supplies causality even when its
            // FLASH_ATTN_EXT op flag is false.  The compact-limit form is a
            // bottom-right causal mask by construction.
            if (v_mean) {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, true, true>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, v_mean,
                    q_padded, kv_tokens, kv_capacity, static_cast<const half *>(mask),
                    int(mask_row_stride/sizeof(half)), query_limits,
                    k_scale_head_stride, stream);
            } else {
                error = qwen38_sage_launch_attention<MaskMode::kCausal, true, false>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, nullptr,
                    q_padded, kv_tokens, kv_capacity, static_cast<const half *>(mask),
                    int(mask_row_stride/sizeof(half)), query_limits,
                    k_scale_head_stride, stream);
            }
        } else {
            if (v_mean) {
                error = qwen38_sage_launch_attention<MaskMode::kNone, false, true>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, v_mean,
                    q_padded, kv_tokens, kv_capacity, nullptr, 0, nullptr,
                    k_scale_head_stride, stream);
            } else {
                error = qwen38_sage_launch_attention<MaskMode::kNone, false, false>(
                    q_quant, k, v, static_cast<half *>(output_half), q_scale, k_scale, v_scale, nullptr,
                    q_padded, kv_tokens, kv_capacity, nullptr, 0, nullptr,
                    k_scale_head_stride, stream);
            }
        }
    }
    if (error != cudaSuccess) {
        return error;
    }

    const int64_t count = int64_t(qwen38_sage_q_heads)*q_tokens*qwen38_sage_head_dim;
    qwen38_sage_half_to_float_kernel<<<(count + 255)/256, 256, 0, stream>>>(
        static_cast<const half *>(output_half), output, q_tokens, q_padded, q_pad);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    const char * check_finite = getenv("QWEN38_CUDA_SAGE_ATTN_CHECK_FINITE");
    if (check_finite && atoi(check_finite) != 0) {
        unsigned int * const nonfinite_device = reinterpret_cast<unsigned int *>(query_limits);
        unsigned int nonfinite_host = 0;
        error = cudaMemsetAsync(nonfinite_device, 0, sizeof(*nonfinite_device), stream);
        if (error != cudaSuccess) {
            return error;
        }
        qwen38_sage_count_nonfinite_kernel<<<(count + 255)/256, 256, 0, stream>>>(
            output, count, nonfinite_device);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaMemcpyAsync(
            &nonfinite_host, nonfinite_device, sizeof(nonfinite_host), cudaMemcpyDeviceToHost, stream);
        if (error != cudaSuccess) {
            return error;
        }
        error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            return error;
        }
        fprintf(stderr,
            "qwen38 Sage finite check: q=%d kv=%d output_nonfinite=%u\n",
            q_tokens, kv_tokens, nonfinite_host);
    }
    return cudaSuccess;
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
