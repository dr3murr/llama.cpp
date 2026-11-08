#include "qwen38-pascal-gdn.cuh"

#include "ggml-cuda.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t PASCAL_KEXACT_SINGLE_VERSION = 1;
constexpr char PASCAL_KEXACT_SINGLE_MAGIC[8] = {'P', '2', 'P', 'K', '1', 'A', '1', '6'};

constexpr int QWEN38_EMBED = 5120;
constexpr int QWEN38_GDN_QKV = 10240;
constexpr int QWEN38_GDN_Z = 6144;

#pragma pack(push, 1)
struct pascal_kexact_single_header {
    char magic[8];
    uint32_t version;
    uint32_t k;
    uint32_t n;
    uint32_t source_type;
    uint32_t quant_bits;
    uint32_t group_size;
    uint32_t quant_block;
    uint32_t flags;
    uint64_t q_offset;
    uint64_t high_offset;
    uint64_t metadata_offset;
    uint64_t q_bytes;
    uint64_t high_bytes;
    uint64_t metadata_bytes;
    uint64_t file_bytes;
    uint64_t dm_bytes;
    uint64_t scale_min_bytes;
    char tensor_name[64];
    uint8_t reserved[80];
};
#pragma pack(pop)

static_assert(sizeof(pascal_kexact_single_header) == 256, "packed projection header ABI");

struct packed_file {
    std::ifstream stream;
    uint64_t bytes = 0;

    explicit packed_file(const char * path) : stream(path, std::ios::binary | std::ios::ate) {
        if (!stream) throw std::runtime_error(std::string("failed to open ") + path);
        bytes = static_cast<uint64_t>(stream.tellg());
        stream.seekg(0);
    }

    template<typename T>
    T header() {
        T value{};
        stream.seekg(0);
        stream.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!stream) throw std::runtime_error("failed to read packed header");
        return value;
    }

    std::vector<uint8_t> slice(uint64_t offset, uint64_t size) {
        if (offset > bytes || size > bytes - offset) {
            throw std::runtime_error("packed payload is outside file bounds");
        }
        std::vector<uint8_t> value(static_cast<size_t>(size));
        stream.clear();
        stream.seekg(static_cast<std::streamoff>(offset));
        stream.read(reinterpret_cast<char *>(value.data()), static_cast<std::streamsize>(value.size()));
        if (!stream) throw std::runtime_error("failed to read packed payload");
        return value;
    }
};

static bool has_magic(const char actual[8], const char expected[8]) {
    return std::memcmp(actual, expected, 8) == 0;
}

struct device_allocation {
    void * ptr = nullptr;
    size_t bytes = 0;

    device_allocation() = default;
    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;
    device_allocation(device_allocation &&) = delete;
    device_allocation & operator=(device_allocation &&) = delete;

    ~device_allocation() {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }

    void upload(const std::vector<uint8_t> & host) {
        if (host.empty()) throw std::runtime_error("refusing empty packed allocation");
        bytes = host.size();
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), bytes, cudaMemcpyHostToDevice));
    }

    void allocate(size_t size) {
        bytes = size;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
};

struct packed_matrix {
    int bits = 0;
    size_t dm_bytes = 0;
    device_allocation q;
    device_allocation high;
    device_allocation metadata;
};

struct layer_state {
    int device = -1;
    int layer = -1;
    std::unique_ptr<packed_matrix> qkv;
    std::unique_ptr<packed_matrix> z;
    device_allocation normalized_f16;
};

using registry_key = std::pair<int, int>;
std::mutex registry_mutex;
std::map<registry_key, std::unique_ptr<layer_state>> registry;

static void set_error(char * error, size_t error_size, const std::string & message) {
    if (error && error_size > 0) std::snprintf(error, error_size, "%s", message.c_str());
}

static std::unique_ptr<packed_matrix> load_matrix(const char * path, int expected_n) {
    packed_file file(path);
    const pascal_kexact_single_header header = file.header<pascal_kexact_single_header>();
    const size_t elements = size_t(QWEN38_EMBED) * expected_n;
    const size_t dm_bytes = elements / 256 * sizeof(uint32_t);
    const size_t scale_min_bytes = elements / 32 * sizeof(uint16_t);
    const size_t expected_high_bytes = header.quant_bits == 5 ? elements / 8 : 0;
    if (!has_magic(header.magic, PASCAL_KEXACT_SINGLE_MAGIC) ||
            header.version != PASCAL_KEXACT_SINGLE_VERSION || header.file_bytes != file.bytes ||
            header.k != QWEN38_EMBED || header.n != static_cast<uint32_t>(expected_n) ||
            (header.quant_bits != 4 && header.quant_bits != 5) ||
            (header.quant_bits == 4 && header.source_type != GGML_TYPE_Q4_K) ||
            (header.quant_bits == 5 && header.source_type != GGML_TYPE_Q5_K) ||
            header.group_size != 32 || header.quant_block != 256 ||
            header.q_bytes != elements / 2 || header.high_bytes != expected_high_bytes ||
            header.dm_bytes != dm_bytes || header.scale_min_bytes != scale_min_bytes ||
            header.metadata_bytes != dm_bytes + scale_min_bytes) {
        throw std::runtime_error("projection pack does not match the expected source-exact geometry");
    }

    auto matrix = std::make_unique<packed_matrix>();
    matrix->bits = static_cast<int>(header.quant_bits);
    matrix->dm_bytes = dm_bytes;
    matrix->q.upload(file.slice(header.q_offset, header.q_bytes));
    if (header.high_bytes > 0) matrix->high.upload(file.slice(header.high_offset, header.high_bytes));
    matrix->metadata.upload(file.slice(header.metadata_offset, header.metadata_bytes));
    return matrix;
}

static std::unique_ptr<layer_state> load_layer(
        int device, int layer, const char * qkv_path, const char * z_path) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        throw std::runtime_error("invalid logical CUDA device index");
    }
    if (layer < 0) throw std::runtime_error("invalid layer index");

    ggml_cuda_set_device(device);
    int physical_device = -1;
    CUDA_CHECK(cudaGetDevice(&physical_device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_device));
    if (prop.major != 6 || prop.minor != 0) {
        throw std::runtime_error("Qwen3.8 Pascal GDN projection requires an SM60 device");
    }

    auto state = std::make_unique<layer_state>();
    state->device = device;
    state->layer = layer;
    state->qkv = load_matrix(qkv_path, QWEN38_GDN_QKV);
    if (state->qkv->bits != 5) throw std::runtime_error("QKV projection must preserve Q5_K");
    state->z = load_matrix(z_path, QWEN38_GDN_Z);
    state->normalized_f16.allocate(QWEN38_EMBED * sizeof(half));
    return state;
}

static layer_state * find_state(int device, int layer) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find({device, layer});
    return it == registry.end() ? nullptr : it->second.get();
}

__global__ void cast_f32_to_f16_gdn(const float * input, half * output, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) output[index] = __float2half_rn(input[index]);
}

template<bool HAS_HIGH, int BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ void projection_kexact_a16_half2_multirow(
        const uint32_t * __restrict__ q32,
        const uint32_t * __restrict__ high,
        const uint32_t * __restrict__ dm,
        const uint16_t * __restrict__ scale_min,
        const half2 * __restrict__ input2,
        float * __restrict__ output,
        int n) {
    constexpr int QUANT_BLOCK = 256;
    constexpr int BLOCKS_PER_ROW = QWEN38_EMBED / QUANT_BLOCK;
    constexpr int GROUPS_PER_ROW = QWEN38_EMBED / 32;
    constexpr int WARPS = BLOCK_SIZE / 32;
    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    if (row_base >= n) return;

    __shared__ uint32_t dm_shared[ROWS_PER_BLOCK][BLOCKS_PER_ROW];
    __shared__ uint16_t scale_min_shared[ROWS_PER_BLOCK][GROUPS_PER_ROW];
    for (int index = threadIdx.x; index < ROWS_PER_BLOCK * BLOCKS_PER_ROW; index += BLOCK_SIZE) {
        const int local_row = index / BLOCKS_PER_ROW;
        const int block = index - local_row * BLOCKS_PER_ROW;
        const int row = row_base + local_row;
        if (row < n) {
            dm_shared[local_row][block] = __ldg(dm + size_t(row) * BLOCKS_PER_ROW + block);
        }
    }
    for (int index = threadIdx.x; index < ROWS_PER_BLOCK * GROUPS_PER_ROW; index += BLOCK_SIZE) {
        const int local_row = index / GROUPS_PER_ROW;
        const int group = index - local_row * GROUPS_PER_ROW;
        const int row = row_base + local_row;
        if (row < n) {
            scale_min_shared[local_row][group] =
                __ldg(scale_min + size_t(row) * GROUPS_PER_ROW + group);
        }
    }
    __syncthreads();

    half2 accumulator2[ROWS_PER_BLOCK][4];
#pragma unroll
    for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            accumulator2[local_row][pair] = __float2half2_rn(0.0f);
        }
    }
    constexpr int CHUNKS_PER_ROW = QWEN38_EMBED / 8;
    for (int chunk = threadIdx.x; chunk < CHUNKS_PER_ROW; chunk += BLOCK_SIZE) {
        const int element = chunk * 8;
        const int quant_block = element / QUANT_BLOCK;
        const int group = element / 32;
        uint32_t packed[ROWS_PER_BLOCK];
        uint32_t high_bits[ROWS_PER_BLOCK];
        half2 scale2[ROWS_PER_BLOCK];
        half2 minimum2[ROWS_PER_BLOCK];
#pragma unroll
        for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
            const int row = row_base + local_row;
            if (row < n) {
                packed[local_row] = __ldg(q32 + size_t(row) * CHUNKS_PER_ROW + chunk);
                high_bits[local_row] = HAS_HIGH
                    ? __ldg(high + size_t(row) * GROUPS_PER_ROW + group)
                    : 0;
                const uint32_t dm_bits = dm_shared[local_row][quant_block];
                const uint16_t sm = scale_min_shared[local_row][group];
                const half d = __ushort_as_half(static_cast<uint16_t>(dm_bits));
                const half dmin = __ushort_as_half(static_cast<uint16_t>(dm_bits >> 16));
                scale2[local_row] = __half2half2(__hmul(d, __int2half_rn(sm & 0xff)));
                minimum2[local_row] = __half2half2(__hmul(dmin, __int2half_rn(sm >> 8)));
            }
        }

#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int shift = pair * 8;
            const half2 x2 = __ldg(input2 + element / 2 + pair);
#pragma unroll
            for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
                const int row = row_base + local_row;
                if (row < n) {
                    int q0 = int((packed[local_row] >> shift) & 0x0f);
                    int q1 = int((packed[local_row] >> (shift + 4)) & 0x0f);
                    if (HAS_HIGH) {
                        q0 |= int((high_bits[local_row] >> ((element + pair * 2) & 31)) & 1u) << 4;
                        q1 |= int((high_bits[local_row] >> ((element + pair * 2 + 1) & 31)) & 1u) << 4;
                    }
                    const half2 quant2 = __halves2half2(__int2half_rn(q0), __int2half_rn(q1));
                    const half2 weight2 =
                        __hsub2(__hmul2(quant2, scale2[local_row]), minimum2[local_row]);
                    accumulator2[local_row][pair] =
                        __hfma2(weight2, x2, accumulator2[local_row][pair]);
                }
            }
        }
    }

    float accumulators[ROWS_PER_BLOCK];
#pragma unroll
    for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
        float accumulator = 0.0f;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 values = __half22float2(accumulator2[local_row][pair]);
            accumulator += values.x + values.y;
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
        }
        accumulators[local_row] = accumulator;
    }

    __shared__ float warp_values[ROWS_PER_BLOCK][WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
#pragma unroll
        for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
            warp_values[local_row][warp] = accumulators[local_row];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
            float accumulator = lane < WARPS ? warp_values[local_row][lane] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                accumulator += __shfl_down_sync(0xffffffffu, accumulator, offset);
            }
            const int row = row_base + local_row;
            if (lane == 0 && row < n) output[row] = accumulator;
        }
    }
}

static void launch_projection(
        const packed_matrix & matrix, const half * input, float * output, int n, cudaStream_t stream) {
    const auto * metadata = static_cast<const uint8_t *>(matrix.metadata.ptr);
    constexpr int BLOCK_SIZE = 128;
    constexpr int ROWS_PER_BLOCK = 4;
    const int blocks = (n + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    if (matrix.bits == 5) {
        projection_kexact_a16_half2_multirow<true, BLOCK_SIZE, ROWS_PER_BLOCK>
            <<<blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint32_t *>(matrix.q.ptr),
            static_cast<const uint32_t *>(matrix.high.ptr),
            reinterpret_cast<const uint32_t *>(metadata),
            reinterpret_cast<const uint16_t *>(metadata + matrix.dm_bytes),
            reinterpret_cast<const half2 *>(input), output, n);
    } else {
        projection_kexact_a16_half2_multirow<false, BLOCK_SIZE, ROWS_PER_BLOCK>
            <<<blocks, BLOCK_SIZE, 0, stream>>>(
            static_cast<const uint32_t *>(matrix.q.ptr), nullptr,
            reinterpret_cast<const uint32_t *>(metadata),
            reinterpret_cast<const uint16_t *>(metadata + matrix.dm_bytes),
            reinterpret_cast<const half2 *>(input), output, n);
    }
}

} // namespace

extern "C" bool ggml_backend_cuda_qwen38_pascal_gdn_in_register(
        int device, int layer, const char * qkv_path, const char * z_path,
        char * error, size_t error_size) {
    try {
        if (!qkv_path || !z_path) throw std::runtime_error("packed projection paths are required");
        auto state = load_layer(device, layer, qkv_path, z_path);
        std::lock_guard<std::mutex> lock(registry_mutex);
        ggml_cuda_set_device(device);
        registry[{device, layer}] = std::move(state);
        set_error(error, error_size, "");
        return true;
    } catch (const std::exception & exception) {
        set_error(error, error_size, exception.what());
        return false;
    }
}

extern "C" void ggml_backend_cuda_qwen38_pascal_gdn_in_unregister(int device, int layer) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find({device, layer});
    if (it != registry.end()) {
        ggml_cuda_set_device(device);
        registry.erase(it);
    }
}

extern "C" bool ggml_backend_cuda_qwen38_pascal_gdn_in_is_registered(int device, int layer) {
    return ggml_cuda_qwen38_pascal_gdn_in_registered(device, layer);
}

bool ggml_cuda_qwen38_pascal_gdn_in_registered(int device, int layer) {
    return find_state(device, layer) != nullptr;
}

void ggml_cuda_op_qwen38_pascal_gdn_in(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_nelements(dst) == QWEN38_GDN_QKV + QWEN38_GDN_Z);
    GGML_ASSERT(ggml_nelements(dst->src[0]) == QWEN38_EMBED);

    const int layer = ggml_get_op_params_i32(dst, 0);
    layer_state * state = find_state(ctx.device, layer);
    GGML_ASSERT(state && "Qwen3.8 Pascal GDN input pack was not registered on this CUDA device");

    auto * normalized_half = static_cast<half *>(state->normalized_f16.ptr);
    auto * output = static_cast<float *>(dst->data);
    cudaStream_t stream = ctx.stream();
    cast_f32_to_f16_gdn<<<(QWEN38_EMBED + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(dst->src[0]->data), normalized_half, QWEN38_EMBED);
    launch_projection(*state->qkv, normalized_half, output, QWEN38_GDN_QKV, stream);
    launch_projection(*state->z, normalized_half, output + QWEN38_GDN_QKV, QWEN38_GDN_Z, stream);
}
