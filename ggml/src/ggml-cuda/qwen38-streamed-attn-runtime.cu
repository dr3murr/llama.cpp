#include "qwen38-streamed-attn-runtime.cuh"

#include "fattn-tile.cuh"
#include "qwen38-streamed-attn.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace {

constexpr int qwen38_rt_d = 256;
constexpr int qwen38_rt_default_max_q = 256;
constexpr int qwen38_rt_hard_max_q = 512;
constexpr int qwen38_rt_q_heads = 24;
constexpr int qwen38_rt_kv_heads = 4;
constexpr int qwen38_rt_q8_bytes = 258;
constexpr int qwen38_rt_max_context = 262144;

bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && value[0] != '0';
}

int env_integer(const char * name, const int fallback) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

int runtime_tile_tokens() {
    static const int value = []() {
        const char * experiment = std::getenv("QWEN38_CUDA_STREAMED_ATTN_TILE_EXPERIMENT");
        return experiment ? std::atoi(experiment) :
            env_integer("QWEN38_CUDA_STREAMED_ATTN_TILE", 8192);
    }();
    return value;
}

int runtime_minimum_kv() {
    static const int value = []() {
        const char * experiment = std::getenv("QWEN38_CUDA_STREAMED_ATTN_MIN_KV_EXPERIMENT");
        return experiment ? std::atoi(experiment) :
            env_integer("QWEN38_CUDA_STREAMED_ATTN_MIN_KV", 65536);
    }();
    return value;
}

int runtime_minimum_q() {
    static const int value = env_integer("QWEN38_CUDA_STREAMED_ATTN_MIN_Q", 128);
    return value;
}

int runtime_maximum_q() {
    static const int value = []() {
        const char * experiment = std::getenv("QWEN38_CUDA_STREAMED_ATTN_MAX_Q_EXPERIMENT");
        const char * selected = experiment ? experiment : std::getenv("QWEN38_CUDA_STREAMED_ATTN_MAX_Q");
        const int parsed = selected ? std::atoi(selected) : qwen38_rt_default_max_q;
        return parsed >= qwen38_rt_default_max_q && parsed <= qwen38_rt_hard_max_q ?
            parsed : qwen38_rt_default_max_q;
    }();
    return value;
}

struct streamed_state {
    int source = -1;
    int destination = -1;
    int tile_tokens = 0;
    int q_capacity = 0;
    char * source_packed = nullptr;
    size_t source_packed_capacity = 0;
    bool source_pack_disabled = false;

    std::array<char *, 2> ring_k = {};
    std::array<char *, 2> ring_v = {};
    float * q = nullptr;
    half * tile_mask = nullptr;
    int * mask_limits = nullptr;
    void * raw = nullptr;
    float * running = nullptr;
    void * running_meta = nullptr;

    cudaStream_t pack_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    std::array<cudaEvent_t, 2> ready = {};
    std::array<cudaEvent_t, 2> consumed = {};
    cudaEvent_t compute_done = nullptr;

    size_t scratch_bytes = 0;
    bool initialized = false;

    ~streamed_state() {
        if (source >= 0) {
            cudaSetDevice(source);
            if (pack_stream) cudaStreamDestroy(pack_stream);
            if (source_packed) cudaFree(source_packed);
        }
        if (destination >= 0) {
            cudaSetDevice(destination);
            if (compute_done) cudaEventDestroy(compute_done);
            for (int slot = 0; slot < 2; ++slot) {
                if (consumed[slot]) cudaEventDestroy(consumed[slot]);
                if (ready[slot]) cudaEventDestroy(ready[slot]);
            }
            if (compute_stream) cudaStreamDestroy(compute_stream);
            if (copy_stream) cudaStreamDestroy(copy_stream);
            if (running_meta) cudaFree(running_meta);
            if (running) cudaFree(running);
            if (raw) cudaFree(raw);
            if (mask_limits) cudaFree(mask_limits);
            if (tile_mask) cudaFree(tile_mask);
            if (q) cudaFree(q);
            for (int slot = 0; slot < 2; ++slot) {
                if (ring_v[slot]) cudaFree(ring_v[slot]);
                if (ring_k[slot]) cudaFree(ring_k[slot]);
            }
        }
    }
};

std::atomic<streamed_state *> published_state { nullptr };

cudaError_t enable_peer(const int peer) {
    const cudaError_t error = cudaDeviceEnablePeerAccess(peer, 0);
    if (error == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return cudaSuccess;
    }
    return error;
}

int find_blackwell_peer(const int source) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return -1;
    }
    for (int device = 0; device < count; ++device) {
        if (device == source) {
            continue;
        }
        cudaDeviceProp property = {};
        int destination_can_access = 0;
        int source_can_access = 0;
        if (cudaGetDeviceProperties(&property, device) != cudaSuccess || property.major < 12 ||
                cudaDeviceCanAccessPeer(&destination_can_access, device, source) != cudaSuccess ||
                cudaDeviceCanAccessPeer(&source_can_access, source, device) != cudaSuccess) {
            continue;
        }
        if (destination_can_access && source_can_access) {
            return device;
        }
    }
    return -1;
}

cudaError_t initialize_state(
        streamed_state & state, const int source, const int tile_tokens, const int q_capacity) {
    state.source = source;
    state.destination = find_blackwell_peer(source);
    state.tile_tokens = tile_tokens;
    state.q_capacity = q_capacity;
    if (state.destination < 0) {
        return cudaErrorNotSupported;
    }

    cudaError_t error = cudaSetDevice(source);
    if (error != cudaSuccess) return error;
    error = enable_peer(state.destination);
    if (error != cudaSuccess) return error;
    error = cudaStreamCreateWithFlags(&state.pack_stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) return error;

    error = cudaSetDevice(state.destination);
    if (error != cudaSuccess) return error;
    error = enable_peer(source);
    if (error != cudaSuccess) return error;

    const size_t ring_bytes = size_t(tile_tokens)*qwen38_rt_kv_heads*qwen38_rt_q8_bytes;
    const size_t q_bytes = qwen38_cuda_streamed_attn_output_bytes(q_capacity);
    const size_t mask_bytes = size_t(q_capacity)*tile_tokens*sizeof(half);
    const size_t limits_bytes = size_t(q_capacity)*sizeof(int);
    const size_t raw_bytes = qwen38_cuda_streamed_attn_raw_bytes(q_capacity);
    const size_t meta_bytes = qwen38_cuda_streamed_attn_meta_bytes(q_capacity, 1);
    state.scratch_bytes = 4*ring_bytes + 2*q_bytes + mask_bytes + limits_bytes + raw_bytes + meta_bytes;

    // CUDA can lazily materialize kernel modules and launch metadata on the
    // first streamed-attention submission.  Reserving every last reported byte
    // for scratch therefore lets cudaMalloc succeed only for the first kernel
    // launch to fail.  Keep a small explicit margin and use the ordinary Pascal
    // attention path when this optional accelerator cannot fit safely.
    constexpr size_t lazy_launch_reserve = size_t(48)*1024*1024;
    size_t free_before = 0;
    size_t total_before = 0;
    error = cudaMemGetInfo(&free_before, &total_before);
    if (error != cudaSuccess) return error;
    if (state.scratch_bytes > free_before ||
            lazy_launch_reserve > free_before - state.scratch_bytes) {
        GGML_LOG_WARN(
            "qwen38 streamed attention: q=%d scratch=%.3f MiB plus %.3f MiB launch reserve "
            "exceeds destination_free=%.3f MiB; using Pascal fallback\n",
            q_capacity, double(state.scratch_bytes)/(1024.0*1024.0),
            double(lazy_launch_reserve)/(1024.0*1024.0), double(free_before)/(1024.0*1024.0));
        return cudaErrorMemoryAllocation;
    }

    for (int slot = 0; slot < 2; ++slot) {
        error = cudaMalloc(&state.ring_k[slot], ring_bytes);
        if (error != cudaSuccess) return error;
        error = cudaMalloc(&state.ring_v[slot], ring_bytes);
        if (error != cudaSuccess) return error;
    }
    error = cudaMalloc(&state.q, q_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&state.tile_mask, mask_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&state.mask_limits, limits_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&state.raw, raw_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&state.running, q_bytes);
    if (error != cudaSuccess) return error;
    error = cudaMalloc(&state.running_meta, meta_bytes);
    if (error != cudaSuccess) return error;

    // Keep unused head groups deterministic for the diagnostic head-split
    // path.  Production overwrites all of these buffers, so the initialization
    // has no effect on the established all-head result.
    error = cudaMemset(state.q, 0, q_bytes);
    if (error != cudaSuccess) return error;
    for (int slot = 0; slot < 2; ++slot) {
        error = cudaMemset(state.ring_k[slot], 0, ring_bytes);
        if (error != cudaSuccess) return error;
        error = cudaMemset(state.ring_v[slot], 0, ring_bytes);
        if (error != cudaSuccess) return error;
    }

    error = cudaStreamCreateWithFlags(&state.copy_stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) return error;
    error = cudaStreamCreateWithFlags(&state.compute_stream, cudaStreamNonBlocking);
    if (error != cudaSuccess) return error;
    for (int slot = 0; slot < 2; ++slot) {
        error = cudaEventCreateWithFlags(&state.ready[slot], cudaEventDisableTiming);
        if (error != cudaSuccess) return error;
        error = cudaEventCreateWithFlags(&state.consumed[slot], cudaEventDisableTiming);
        if (error != cudaSuccess) return error;
        error = cudaEventRecord(state.consumed[slot], state.compute_stream);
        if (error != cudaSuccess) return error;
    }
    error = cudaEventCreateWithFlags(&state.compute_done, cudaEventDisableTiming);
    if (error != cudaSuccess) return error;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) return error;

    char source_name[128] = {};
    char destination_name[128] = {};
    cudaDeviceProp source_property = {};
    cudaDeviceProp destination_property = {};
    cudaGetDeviceProperties(&source_property, source);
    cudaGetDeviceProperties(&destination_property, state.destination);
    std::snprintf(source_name, sizeof(source_name), "%s", source_property.name);
    std::snprintf(destination_name, sizeof(destination_name), "%s", destination_property.name);
    GGML_LOG_INFO(
        "qwen38 streamed attention: initialized source=%d (%s) destination=%d (%s) "
        "tile=%d q_capacity=%d scratch=%.3f MiB destination_free=%.3f MiB\n",
        source, source_name, state.destination, destination_name, tile_tokens, q_capacity,
        double(state.scratch_bytes)/(1024.0*1024.0), double(free_bytes)/(1024.0*1024.0));

    state.initialized = true;
    return cudaSetDevice(source);
}

streamed_state * get_state(const int source, const int tile_tokens, const int q_tokens) {
    static std::mutex mutex;
    static std::unique_ptr<streamed_state> state;
    static int smallest_failed_capacity = qwen38_rt_hard_max_q + 1;
    static bool capacity_warning_logged = false;
    std::lock_guard<std::mutex> lock(mutex);
    if (state && state->initialized) {
        if (state->source != source || state->tile_tokens != tile_tokens) {
            return nullptr;
        }
        if (q_tokens <= state->q_capacity) {
            return state.get();
        }
        if (!capacity_warning_logged) {
            capacity_warning_logged = true;
            GGML_LOG_WARN(
                "qwen38 streamed attention: q=%d exceeds initialized q_capacity=%d; "
                "using Pascal fallback for wider submissions\n",
                q_tokens, state->q_capacity);
        }
        return nullptr;
    }
    if (q_tokens >= smallest_failed_capacity) {
        return nullptr;
    }
    state = std::make_unique<streamed_state>();
    const cudaError_t error = initialize_state(*state, source, tile_tokens, q_tokens);
    if (error != cudaSuccess) {
        smallest_failed_capacity = std::min(smallest_failed_capacity, q_tokens);
        GGML_LOG_WARN(
            "qwen38 streamed attention: q=%d initialization failed: %s; using Pascal fallback\n",
            q_tokens, cudaGetErrorString(error));
        state.reset();
        cudaGetLastError();
        cudaSetDevice(source);
        return nullptr;
    }
    published_state.store(state.get(), std::memory_order_release);
    return state.get();
}

bool shape_supported(
        const ggml_tensor * dst, const int minimum_kv, const int minimum_q) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const bool k_layout_supported = k && (
        (k->nb[1] == qwen38_rt_q8_bytes && k->nb[2] >= size_t(k->ne[1])*k->nb[1]) ||
        (k->nb[1] == qwen38_rt_kv_heads*qwen38_rt_q8_bytes && k->nb[2] == qwen38_rt_q8_bytes));
    const bool v_layout_supported = v && (
        (v->nb[1] == qwen38_rt_q8_bytes && v->nb[2] >= size_t(v->ne[1])*v->nb[1]) ||
        (v->nb[1] == qwen38_rt_kv_heads*qwen38_rt_q8_bytes && v->nb[2] == qwen38_rt_q8_bytes));
    if (!q || !k || !v || !mask || sinks ||
            q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            k->type != GGML_TYPE_Q8_KV256 || v->type != GGML_TYPE_Q8_KV256 ||
            mask->type != GGML_TYPE_F16 ||
            q->ne[0] != qwen38_rt_d || q->ne[1] < minimum_q || q->ne[1] > runtime_maximum_q() ||
            q->ne[2] != qwen38_rt_q_heads || q->ne[3] != 1 ||
            k->ne[0] != qwen38_rt_d || k->ne[2] != qwen38_rt_kv_heads || k->ne[3] != 1 ||
            v->ne[0] != qwen38_rt_d || v->ne[1] != k->ne[1] ||
            v->ne[2] != qwen38_rt_kv_heads || v->ne[3] != 1 ||
            k->ne[1] < minimum_kv ||
            q->nb[0] != sizeof(float) || q->nb[1] < qwen38_rt_d*sizeof(float) ||
            q->nb[2] < qwen38_rt_d*sizeof(float) ||
            !k_layout_supported || !v_layout_supported ||
            dst->ne[0] != qwen38_rt_d || dst->ne[1] != qwen38_rt_q_heads ||
            dst->ne[2] != q->ne[1] || dst->ne[3] != 1 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] != qwen38_rt_d*sizeof(float) ||
            dst->nb[2] != size_t(qwen38_rt_q_heads)*qwen38_rt_d*sizeof(float) ||
            mask->nb[1] % sizeof(half) != 0 || mask->nb[1]/sizeof(half) < size_t(k->ne[1])) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    const float * op_params = reinterpret_cast<const float *>(dst->op_params);
    std::memcpy(&max_bias,      op_params + 1, sizeof(float));
    std::memcpy(&logit_softcap, op_params + 2, sizeof(float));
    return max_bias == 0.0f && logit_softcap == 0.0f;
}

void enqueue_kv_tile(
        const ggml_tensor * k, const ggml_tensor * v, streamed_state & state,
        const int slot, const int start, const int count, const int ring_stride_tokens,
        const int first_head = 0, const char * packed = nullptr) {
    if (packed) {
        const int part = start/ring_stride_tokens;
        const int remote_heads = qwen38_rt_kv_heads - first_head;
        const size_t part_offset = size_t(part)*2*remote_heads*ring_stride_tokens*qwen38_rt_q8_bytes;
        const size_t value_bytes = size_t(remote_heads)*count*qwen38_rt_q8_bytes;
        for (int value = 0; value < 2; ++value) {
            char * destination = value == 0 ? state.ring_k[slot] : state.ring_v[slot];
            CUDA_CHECK(cudaMemcpyAsync(
                destination + size_t(first_head)*ring_stride_tokens*qwen38_rt_q8_bytes,
                packed + part_offset + size_t(value)*value_bytes,
                value_bytes, cudaMemcpyDeviceToDevice, state.copy_stream));
        }
        return;
    }

    for (int value = 0; value < 2; ++value) {
        const ggml_tensor * source = value == 0 ? k : v;
        char * destination = value == 0 ? state.ring_k[slot] : state.ring_v[slot];
        for (int head = first_head; head < qwen38_rt_kv_heads; ++head) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                destination + size_t(head)*ring_stride_tokens*qwen38_rt_q8_bytes,
                qwen38_rt_q8_bytes,
                static_cast<const char *>(source->data) + size_t(start)*source->nb[1] + size_t(head)*source->nb[2],
                source->nb[1], qwen38_rt_q8_bytes, count,
                cudaMemcpyDeviceToDevice, state.copy_stream));
        }
    }
}

__global__ void qwen38_pack_remote_q8_tile_major(
        const char * __restrict__ k,
        const char * __restrict__ v,
        char * __restrict__ packed,
        const size_t k_token_stride,
        const size_t k_head_stride,
        const size_t v_token_stride,
        const size_t v_head_stride,
        const int kv_tokens,
        const int tile_tokens,
        const int first_head) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const size_t warp = size_t(blockIdx.x)*warps_per_block + size_t(threadIdx.x >> 5);
    const size_t warp_stride = size_t(gridDim.x)*warps_per_block;
    const int remote_heads = qwen38_rt_kv_heads - first_head;
    const size_t rows = size_t(kv_tokens)*2*remote_heads;

    for (size_t row = warp; row < rows; row += warp_stride) {
        size_t index = row;
        const int token = index % kv_tokens;
        index /= kv_tokens;
        const int remote_head = index % remote_heads;
        index /= remote_heads;
        const int value = index & 1;
        const int part = token/tile_tokens;
        const int within = token - part*tile_tokens;
        const int part_count = min(tile_tokens, kv_tokens - part*tile_tokens);
        const size_t packed_row = size_t(part)*2*remote_heads*tile_tokens +
            size_t(value)*remote_heads*part_count + size_t(remote_head)*part_count + within;
        char * destination = packed + packed_row*qwen38_rt_q8_bytes;

        const int head = first_head + remote_head;
        const char * source = value == 0 ?
            k + size_t(token)*k_token_stride + size_t(head)*k_head_stride :
            v + size_t(token)*v_token_stride + size_t(head)*v_head_stride;
        for (int byte = lane; byte < qwen38_rt_q8_bytes; byte += 32) {
            destination[byte] = source[byte];
        }
    }
}

void * source_local_buffer(ggml_tensor * dst, const int local_kv_heads) {
    GGML_ASSERT(local_kv_heads > 0 && local_kv_heads < qwen38_rt_kv_heads);
    // The graph allocator already reserves output + a full four-head FP16 K/V
    // expansion for GP100's ordinary tile-attention fallback.  Streamed
    // attention otherwise leaves that extra-data region unused.  Place the
    // tight local output there; its smaller head-partition FP16 expansion
    // remains inside the existing allocation and requires no persistent
    // cudaMalloc.
    return reinterpret_cast<void *>(GGML_PAD(
        reinterpret_cast<uintptr_t>(dst->data) + ggml_nbytes(dst), 128));
}

size_t source_packed_bytes(
        const ggml_tensor * k, const int local_kv_heads,
        const int tile_tokens, const int n_parts) {
    const int last_count = int(k->ne[1]) - (n_parts - 1)*tile_tokens;
    return (size_t(n_parts - 1)*2*(qwen38_rt_kv_heads - local_kv_heads)*tile_tokens +
        size_t(2)*(qwen38_rt_kv_heads - local_kv_heads)*last_count)*qwen38_rt_q8_bytes;
}

cudaError_t launch_source_pack(
        const ggml_tensor * k, const ggml_tensor * v, char * packed,
        const int kv_tokens, const int tile_tokens,
        const int first_head, cudaStream_t stream) {
    constexpr int blocks = 1024;
    constexpr int threads = 256;
    qwen38_pack_remote_q8_tile_major<<<blocks, threads, 0, stream>>>(
        static_cast<const char *>(k->data), static_cast<const char *>(v->data), packed,
        k->nb[1], k->nb[2], v->nb[1], v->nb[2], kv_tokens, tile_tokens, first_head);
    return cudaGetLastError();
}

void launch_source_head_partition(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * q,
        const ggml_tensor * k,
        const ggml_tensor * v,
        const ggml_tensor * mask,
        ggml_tensor * dst,
        const int local_kv_heads) {
    const int local_q_heads = local_kv_heads*(qwen38_rt_q_heads/qwen38_rt_kv_heads);

    ggml_tensor local_q = *q;
    ggml_tensor local_k = *k;
    ggml_tensor local_v = *v;
    ggml_tensor local_dst = *dst;

    local_q.ne[2] = local_q_heads;
    local_k.ne[2] = local_kv_heads;
    local_v.ne[2] = local_kv_heads;

    local_dst.data = source_local_buffer(dst, local_kv_heads);
    local_dst.ne[1] = local_q_heads;
    local_dst.nb[0] = sizeof(float);
    local_dst.nb[1] = size_t(qwen38_rt_d)*sizeof(float);
    local_dst.nb[2] = size_t(local_q_heads)*local_dst.nb[1];
    local_dst.nb[3] = size_t(q->ne[1])*local_dst.nb[2];
    local_dst.src[0] = &local_q;
    local_dst.src[1] = &local_k;
    local_dst.src[2] = &local_v;
    local_dst.src[3] = const_cast<ggml_tensor *>(mask);

    const ggml_cuda_flash_attn_ext_f16_extra_data local_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(&local_dst, true, true);
    const ggml_cuda_flash_attn_ext_f16_extra_data full_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, true, true);
    GGML_ASSERT(local_extra.end <= full_extra.end);

    ggml_cuda_flash_attn_ext_tile(ctx, &local_dst);
}

} // namespace

bool ggml_cuda_qwen38_streamed_attn(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = env_enabled("QWEN38_CUDA_STREAMED_ATTN");
    if (!enabled) {
        return false;
    }
    const int tile_tokens = runtime_tile_tokens();
    const int minimum_kv = runtime_minimum_kv();
    const int minimum_q = runtime_minimum_q();
    static const int local_kv_heads = []() {
        // Keep a diagnostic override distinct from the manifest-owned policy
        // so the production parser remains strict while the A/B harness can
        // still force the established all-remote control with value zero.
        const char * experiment = std::getenv("QWEN38_CUDA_STREAMED_ATTN_LOCAL_KV_HEADS_EXPERIMENT");
        return experiment ? std::atoi(experiment) :
            env_integer("QWEN38_CUDA_STREAMED_ATTN_LOCAL_KV_HEADS", 0);
    }();
    if (tile_tokens != 4096 && tile_tokens != 8192 && tile_tokens != 16384) {
        return false;
    }
    if (local_kv_heads < 0 || local_kv_heads >= qwen38_rt_kv_heads) {
        return false;
    }

    const int source = ctx.device;
    const int source_cc = ggml_cuda_info().devices[source].cc;
    if (source_cc != GGML_CUDA_CC_PASCAL || !shape_supported(dst, minimum_kv, minimum_q)) {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];
        const ggml_tensor * mask = dst->src[3];
        static bool logged_rejection = false;
        if (!logged_rejection && q && k && v && mask && q->ne[1] >= minimum_q) {
            logged_rejection = true;
            GGML_LOG_INFO(
                "qwen38 streamed attention: rejected source=%d cc=%d "
                "q(type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]) "
                "k(type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]) "
                "v(type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]) "
                "mask(type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]) sinks=%p\n",
                source, source_cc,
                ggml_type_name(q->type),
                (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2], (long long) q->ne[3],
                q->nb[0], q->nb[1], q->nb[2], q->nb[3],
                ggml_type_name(k->type),
                (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2], (long long) k->ne[3],
                k->nb[0], k->nb[1], k->nb[2], k->nb[3],
                ggml_type_name(v->type),
                (long long) v->ne[0], (long long) v->ne[1], (long long) v->ne[2], (long long) v->ne[3],
                v->nb[0], v->nb[1], v->nb[2], v->nb[3],
                ggml_type_name(mask->type),
                (long long) mask->ne[0], (long long) mask->ne[1], (long long) mask->ne[2], (long long) mask->ne[3],
                mask->nb[0], mask->nb[1], mask->nb[2], mask->nb[3], (void *) dst->src[4]);
        }
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int q_tokens = q->ne[1];
    streamed_state * state = get_state(source, tile_tokens, q_tokens);
    if (!state) {
        return false;
    }
    const int local_q_heads = local_kv_heads*(qwen38_rt_q_heads/qwen38_rt_kv_heads);
    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(dst->op_params), sizeof(float));
    const int kv_tokens = k->ne[1];
    const int n_parts = (kv_tokens + tile_tokens - 1)/tile_tokens;
    const size_t q_bytes = qwen38_cuda_streamed_attn_output_bytes(q_tokens);
    const int mask_stride = mask->nb[1]/sizeof(half);
    static const bool source_pack_requested = []() {
        const char * experiment = std::getenv("QWEN38_CUDA_STREAMED_ATTN_SOURCE_PACK_EXPERIMENT");
        return experiment ? std::atoi(experiment) != 0 :
            env_enabled("QWEN38_CUDA_STREAMED_ATTN_SOURCE_PACK");
    }();
    bool source_pack = source_pack_requested && local_kv_heads != 0 && !state->source_pack_disabled;

    // The cross-device event path triggered a P100 PBDMA fault.
    // Keep this boundary host-synchronized until another protocol is qualified.
    CUDA_CHECK(cudaSetDevice(source));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    static std::atomic<int> profile_counter { 0 };
    const bool profile_requested = env_enabled("QWEN38_CUDA_PROFILE_STREAMED_SPLIT") && local_kv_heads != 0;
    const int profile_id = profile_requested ? profile_counter.fetch_add(1, std::memory_order_relaxed) : -1;
    const bool profile = profile_id >= 0 && profile_id < 32;
    std::chrono::steady_clock::time_point profile_wall_begin {};
    cudaEvent_t pack_begin = nullptr;
    cudaEvent_t pack_end = nullptr;
    cudaEvent_t local_begin = nullptr;
    cudaEvent_t local_end = nullptr;
    cudaEvent_t remote_begin = nullptr;
    cudaEvent_t remote_end = nullptr;
    char * packed = nullptr;
    size_t packed_bytes = 0;
    if (profile) {
        profile_wall_begin = std::chrono::steady_clock::now();
    }
    if (source_pack) {
        packed_bytes = source_packed_bytes(k, local_kv_heads, tile_tokens, n_parts);
        if (!state->source_packed) {
            // Do not let the RTX copy engine peer-read the graph allocator's
            // transient VMM workspace.  Repeated full-prefill submissions from
            // that mapping produced a P100 PBDMA/Xid 32 even though a restored
            // one-shot late-context submission passed.  An ordinary, stable
            // cudaMalloc allocation survived the exact same repeated transfer
            // pattern.  Allocate it once at native-context capacity so neither
            // its address nor its size changes during a request.
            const size_t maximum_packed_bytes =
                size_t(2)*(qwen38_rt_kv_heads - local_kv_heads)*
                qwen38_rt_max_context*qwen38_rt_q8_bytes;
            const cudaError_t allocation_error =
                cudaMalloc(&state->source_packed, maximum_packed_bytes);
            if (allocation_error == cudaSuccess) {
                state->source_packed_capacity = maximum_packed_bytes;
                GGML_LOG_INFO(
                    "qwen38 streamed attention: dedicated source pack %.3f MiB on device=%d\n",
                    double(maximum_packed_bytes)/(1024.0*1024.0), source);
            } else {
                GGML_LOG_WARN(
                    "qwen38 streamed attention: dedicated source-pack allocation failed: %s; "
                    "disabling the pack for this process and using unpacked peer copies\n",
                    cudaGetErrorString(allocation_error));
                cudaGetLastError();
                state->source_pack_disabled = true;
                source_pack = false;
            }
        }
        if (source_pack) {
            GGML_ASSERT(packed_bytes <= state->source_packed_capacity);
            packed = state->source_packed;
        }
    }
    if (source_pack) {
        if (profile) {
            CUDA_CHECK(cudaEventCreate(&pack_begin));
            CUDA_CHECK(cudaEventCreate(&pack_end));
            CUDA_CHECK(cudaEventRecord(pack_begin, state->pack_stream));
        }
        CUDA_CHECK(launch_source_pack(
            k, v, packed, kv_tokens, tile_tokens, local_kv_heads, state->pack_stream));
        if (profile) {
            CUDA_CHECK(cudaEventRecord(pack_end, state->pack_stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(state->pack_stream));
    }
    if (local_kv_heads != 0) {
        if (profile) {
            CUDA_CHECK(cudaEventCreate(&local_begin));
            CUDA_CHECK(cudaEventCreate(&local_end));
            CUDA_CHECK(cudaEventRecord(local_begin, ctx.stream()));
        }
        launch_source_head_partition(ctx, q, k, v, mask, dst, local_kv_heads);
        CUDA_CHECK(cudaGetLastError());
        if (profile) {
            CUDA_CHECK(cudaEventRecord(local_end, ctx.stream()));
        }
    }

    CUDA_CHECK(cudaSetDevice(state->destination));
    if (profile) {
        CUDA_CHECK(cudaEventCreate(&remote_begin));
        CUDA_CHECK(cudaEventCreate(&remote_end));
        CUDA_CHECK(cudaEventRecord(remote_begin, state->copy_stream));
    }
    const size_t q_row_bytes = size_t(qwen38_rt_d)*sizeof(float);
    for (int head = local_q_heads; head < qwen38_rt_q_heads; ++head) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            state->q + size_t(head)*q_tokens*qwen38_rt_d, q_row_bytes,
            static_cast<const char *>(q->data) + size_t(head)*q->nb[2], q->nb[1],
            q_row_bytes, q_tokens, cudaMemcpyDeviceToDevice, state->copy_stream));
    }
    CUDA_CHECK(qwen38_cuda_streamed_attn_mask_limits(
        mask->data, mask_stride, q_tokens, kv_tokens,
        state->mask_limits, state->compute_stream));

    for (int part = 0; part < n_parts; ++part) {
        const int slot = part & 1;
        const int start = part*tile_tokens;
        const int count = std::min(tile_tokens, kv_tokens - start);
        CUDA_CHECK(cudaStreamWaitEvent(state->copy_stream, state->consumed[slot]));
        enqueue_kv_tile(k, v, *state, slot, start, count, tile_tokens, local_kv_heads, packed);
        CUDA_CHECK(cudaEventRecord(state->ready[slot], state->copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(state->compute_stream, state->ready[slot]));
        CUDA_CHECK(qwen38_cuda_streamed_attn_make_mask(
            state->mask_limits, state->tile_mask, q_tokens, start, tile_tokens,
            state->compute_stream));
        CUDA_CHECK(qwen38_cuda_streamed_attn_partial_merge(
            state->q, state->ring_k[slot], state->ring_v[slot], state->tile_mask,
            tile_tokens, 0, q_tokens, count, tile_tokens, part == 0,
            state->raw, state->running, state->running_meta, scale, state->compute_stream));
        CUDA_CHECK(cudaEventRecord(state->consumed[slot], state->compute_stream));
    }
    CUDA_CHECK(qwen38_cuda_streamed_attn_finalize(
        state->running, state->running_meta, state->running, q_tokens,
        state->compute_stream));
    CUDA_CHECK(cudaEventRecord(state->compute_done, state->compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(state->copy_stream, state->compute_done));
    if (local_kv_heads == 0) {
        CUDA_CHECK(cudaMemcpyPeerAsync(
            dst->data, source, state->running, state->destination, q_bytes, state->copy_stream));
    } else {
        const size_t full_pitch = size_t(qwen38_rt_q_heads)*qwen38_rt_d*sizeof(float);
        const size_t remote_width = size_t(qwen38_rt_q_heads - local_q_heads)*qwen38_rt_d*sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<char *>(dst->data) + size_t(local_q_heads)*q_row_bytes,
            full_pitch,
            state->running + size_t(local_q_heads)*qwen38_rt_d,
            full_pitch,
            remote_width, q_tokens, cudaMemcpyDeviceToDevice, state->copy_stream));
    }
    if (profile) {
        CUDA_CHECK(cudaEventRecord(remote_end, state->copy_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(state->copy_stream));

    CUDA_CHECK(cudaSetDevice(source));
    if (profile) {
        CUDA_CHECK(cudaEventSynchronize(local_end));
        float pack_ms = 0.0f;
        float local_ms = 0.0f;
        float remote_ms = 0.0f;
        if (pack_end) {
            CUDA_CHECK(cudaEventElapsedTime(&pack_ms, pack_begin, pack_end));
        }
        CUDA_CHECK(cudaEventElapsedTime(&local_ms, local_begin, local_end));
        CUDA_CHECK(cudaSetDevice(state->destination));
        CUDA_CHECK(cudaEventElapsedTime(&remote_ms, remote_begin, remote_end));
        CUDA_CHECK(cudaEventDestroy(remote_end));
        CUDA_CHECK(cudaEventDestroy(remote_begin));
        CUDA_CHECK(cudaSetDevice(source));
        if (pack_end) {
            CUDA_CHECK(cudaEventDestroy(pack_end));
            CUDA_CHECK(cudaEventDestroy(pack_begin));
        }
        CUDA_CHECK(cudaEventDestroy(local_end));
        CUDA_CHECK(cudaEventDestroy(local_begin));
        const double wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - profile_wall_begin).count();
        std::fprintf(stderr,
            "qwen38 streamed attention split profile: call=%d q=%d kv=%d packed=%d packed_mib=%.3f "
            "pack_ms=%.3f local_ms=%.3f remote_ms=%.3f wall_ms=%.3f\n",
            profile_id, q_tokens, kv_tokens, source_pack ? 1 : 0,
            double(packed_bytes)/(1024.0*1024.0), pack_ms, local_ms, remote_ms, wall_ms);
    }
    if (local_kv_heads != 0) {
        const size_t full_pitch = size_t(qwen38_rt_q_heads)*qwen38_rt_d*sizeof(float);
        const size_t local_pitch = size_t(local_q_heads)*qwen38_rt_d*sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(
            dst->data, full_pitch,
            source_local_buffer(dst, local_kv_heads), local_pitch,
            local_pitch, q_tokens, cudaMemcpyDeviceToDevice, ctx.stream()));
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO(
            "qwen38 streamed attention: first offload submitted q=%lld kv=%lld tile=%d parts=%d "
            "local_kv_heads=%d source_pack=%d\n",
            static_cast<long long>(q->ne[1]), static_cast<long long>(kv_tokens),
            tile_tokens, n_parts, local_kv_heads, source_pack ? 1 : 0);
    }
    return true;
}

bool ggml_cuda_qwen38_bounded_local_attn_supported(
        const int device, const ggml_tensor * dst) {
    static const bool enabled = env_enabled("QWEN38_CUDA_BOUNDED_LOCAL_ATTN");
    if (!enabled) {
        return false;
    }
    const int tile_tokens = runtime_tile_tokens();
    const int cc = ggml_cuda_info().devices[device].cc;
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    // A persistent head-major cache can use the faster one-launch dense MMA
    // path directly and no longer needs this scratch-bounded substitute.
    if (ggml_cuda_q8_kv256_is_head_major(k) && ggml_cuda_q8_kv256_is_head_major(v)) {
        return false;
    }
    return cc >= GGML_CUDA_CC_BLACKWELL &&
        (tile_tokens == 4096 || tile_tokens == 8192 || tile_tokens == 16384) &&
        shape_supported(dst, runtime_minimum_kv(), runtime_minimum_q());
}

bool ggml_cuda_qwen38_bounded_local_attn(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_qwen38_bounded_local_attn_supported(ctx.device, dst)) {
        return false;
    }

    streamed_state * state = published_state.load(std::memory_order_acquire);
    const int tile_tokens = runtime_tile_tokens();
    if (!state || !state->initialized || state->destination != ctx.device ||
            state->tile_tokens != tile_tokens || dst->src[0]->ne[1] > state->q_capacity) {
        return false;
    }

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int q_tokens = q->ne[1];
    const int kv_tokens = k->ne[1];
    const int n_parts = (kv_tokens + tile_tokens - 1)/tile_tokens;
    const int mask_stride = mask->nb[1]/sizeof(half);
    const size_t q_row_bytes = size_t(qwen38_rt_d)*sizeof(float);
    const size_t q_bytes = qwen38_cuda_streamed_attn_output_bytes(q_tokens);
    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const float *>(dst->op_params), sizeof(float));

    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    for (int head = 0; head < qwen38_rt_q_heads; ++head) {
        CUDA_CHECK(cudaMemcpy2DAsync(
            state->q + size_t(head)*q_tokens*qwen38_rt_d, q_row_bytes,
            static_cast<const char *>(q->data) + size_t(head)*q->nb[2], q->nb[1],
            q_row_bytes, q_tokens, cudaMemcpyDeviceToDevice, state->copy_stream));
    }
    CUDA_CHECK(qwen38_cuda_streamed_attn_mask_limits(
        mask->data, mask_stride, q_tokens, kv_tokens,
        state->mask_limits, state->compute_stream));

    for (int part = 0; part < n_parts; ++part) {
        const int slot = part & 1;
        const int start = part*tile_tokens;
        const int count = std::min(tile_tokens, kv_tokens - start);
        CUDA_CHECK(cudaStreamWaitEvent(state->copy_stream, state->consumed[slot]));
        enqueue_kv_tile(k, v, *state, slot, start, count, tile_tokens);
        CUDA_CHECK(cudaEventRecord(state->ready[slot], state->copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(state->compute_stream, state->ready[slot]));
        CUDA_CHECK(qwen38_cuda_streamed_attn_make_mask(
            state->mask_limits, state->tile_mask, q_tokens, start, tile_tokens,
            state->compute_stream));
        CUDA_CHECK(qwen38_cuda_streamed_attn_partial_merge(
            state->q, state->ring_k[slot], state->ring_v[slot], state->tile_mask,
            tile_tokens, 0, q_tokens, count, tile_tokens, part == 0,
            state->raw, state->running, state->running_meta, scale, state->compute_stream));
        CUDA_CHECK(cudaEventRecord(state->consumed[slot], state->compute_stream));
    }
    CUDA_CHECK(qwen38_cuda_streamed_attn_finalize(
        state->running, state->running_meta, state->running, q_tokens,
        state->compute_stream));
    CUDA_CHECK(cudaEventRecord(state->compute_done, state->compute_stream));
    CUDA_CHECK(cudaStreamWaitEvent(state->copy_stream, state->compute_done));
    CUDA_CHECK(cudaMemcpyAsync(
        dst->data, state->running, q_bytes, cudaMemcpyDeviceToDevice, state->copy_stream));
    CUDA_CHECK(cudaStreamSynchronize(state->copy_stream));

    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO(
            "qwen38 bounded local attention: first submission device=%d q=%lld kv=%d tile=%d parts=%d\n",
            ctx.device, static_cast<long long>(q_tokens), kv_tokens, tile_tokens, n_parts);
    }
    return true;
}
