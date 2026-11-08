#include "qwen38-pascal-ffn.cuh"

#include "ggml-cuda.h"

#include <cuda_fp16.h>

#include <array>
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

constexpr uint32_t PASCAL_W4_VERSION = 1;
constexpr uint32_t PASCAL_W4_SOURCE_EXACT_K = 1u << 4;
constexpr uint32_t PASCAL_W4_UP_HIGH_BITPLANE = 1u << 5;
constexpr uint32_t PASCAL_W4_SOURCE_EXACT_IQ4_NL = 1u << 6;
constexpr char PASCAL_W4_MAGIC[8] = {'P', '2', 'P', 'W', '4', 'A', '1', '6'};

constexpr uint32_t PASCAL_KEXACT_SINGLE_VERSION = 1;
constexpr char PASCAL_KEXACT_SINGLE_MAGIC[8] = {'P', '2', 'P', 'K', '1', 'A', '1', '6'};

constexpr int QWEN38_EMBED = 5120;
constexpr int QWEN38_FFN   = 17408;

#pragma pack(push, 1)
struct pascal_w4_header {
    char magic[8];
    uint32_t version;
    uint32_t k;
    uint32_t n;
    uint32_t group_size;
    uint32_t gate_source_type;
    uint32_t up_source_type;
    uint32_t flags;
    uint64_t gate_scales_offset;
    uint64_t up_scales_offset;
    uint64_t gate_q_offset;
    uint64_t up_q_offset;
    uint64_t gate_scales_bytes;
    uint64_t up_scales_bytes;
    uint64_t gate_q_bytes;
    uint64_t up_q_bytes;
    uint64_t file_bytes;
    char gate_name[64];
    char up_name[64];
    uint64_t gate_high_offset;
    uint64_t gate_high_bytes;
    uint32_t gate_bits;
};

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

static_assert(sizeof(pascal_w4_header) == 256, "packed gate/up header ABI");
static_assert(sizeof(pascal_kexact_single_header) == 256, "packed down header ABI");

struct packed_file {
    std::ifstream stream;
    uint64_t bytes = 0;

    explicit packed_file(const char * path) : stream(path, std::ios::binary | std::ios::ate) {
        if (!stream) {
            throw std::runtime_error(std::string("failed to open ") + path);
        }
        bytes = static_cast<uint64_t>(stream.tellg());
        stream.seekg(0);
    }

    template<typename T>
    T header() {
        T value{};
        stream.seekg(0);
        stream.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!stream) {
            throw std::runtime_error("failed to read packed header");
        }
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
        if (!stream) {
            throw std::runtime_error("failed to read packed payload");
        }
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
        release();
    }

    void release() {
        if (ptr) {
            // Registration/unregistration selects the owning device. Destruction
            // is never allowed concurrently with graph execution.
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            bytes = 0;
        }
    }

    void upload(const std::vector<uint8_t> & host) {
        if (host.empty()) {
            throw std::runtime_error("refusing empty packed allocation");
        }
        bytes = host.size();
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), bytes, cudaMemcpyHostToDevice));
    }

    void allocate(size_t size) {
        bytes = size;
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
    }
};

struct layer_state {
    enum class gate_up_format {
        kexact_q5_q4,
        kexact_q5_q5,
        iq4_nl,
    };

    int device = -1;
    int layer = -1;
    gate_up_format format = gate_up_format::kexact_q5_q4;
    size_t gate_dm_bytes = 0;
    size_t up_dm_bytes = 0;
    size_t down_dm_bytes = 0;

    device_allocation gate_q;
    device_allocation gate_high;
    device_allocation gate_metadata;
    device_allocation up_q;
    device_allocation up_high;
    device_allocation up_metadata;
    device_allocation down_q;
    device_allocation down_high;
    device_allocation down_metadata;
    device_allocation normalized_f16;
    device_allocation swiglu_f16;
};

using registry_key = std::pair<int, int>;
std::mutex registry_mutex;
std::map<registry_key, std::unique_ptr<layer_state>> registry;

static void set_error(char * error, size_t error_size, const std::string & message) {
    if (!error || error_size == 0) {
        return;
    }
    std::snprintf(error, error_size, "%s", message.c_str());
}

static std::unique_ptr<layer_state> load_layer(
        int device, int layer, const char * gate_up_path, const char * down_path) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        throw std::runtime_error("invalid logical CUDA device index");
    }
    if (layer < 0) {
        throw std::runtime_error("invalid layer index");
    }

    ggml_cuda_set_device(device);
    int physical_device = -1;
    CUDA_CHECK(cudaGetDevice(&physical_device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_device));
    if (prop.major != 6 || prop.minor != 0) {
        throw std::runtime_error("Qwen3.8 Pascal FFN requires an SM60 device");
    }

    packed_file gate_file(gate_up_path);
    const pascal_w4_header gate = gate_file.header<pascal_w4_header>();
    const size_t elements = size_t(QWEN38_FFN) * QWEN38_EMBED;
    const size_t kexact_blocks = elements / 256;
    const size_t kexact_dm_bytes = kexact_blocks * sizeof(uint32_t);
    const size_t kexact_scale_min_bytes = elements / 32 * sizeof(uint16_t);
    const size_t kexact_metadata_bytes = kexact_dm_bytes + kexact_scale_min_bytes;
    const size_t high_bytes = elements / 8;
    const size_t iq4_scale_bytes = elements / 32 * sizeof(uint16_t);
    if (!has_magic(gate.magic, PASCAL_W4_MAGIC) || gate.version != PASCAL_W4_VERSION ||
            gate.file_bytes != gate_file.bytes || gate.k != QWEN38_EMBED || gate.n != QWEN38_FFN ||
            gate.group_size != 32 ||
            (gate.flags & PASCAL_W4_SOURCE_EXACT_K) == 0 ||
            gate.gate_q_bytes != elements / 2 || gate.up_q_bytes != elements / 2) {
        throw std::runtime_error("gate/up pack is not the expected source-exact layer geometry");
    }
    const bool iq4_pair = (gate.flags & PASCAL_W4_SOURCE_EXACT_IQ4_NL) != 0;
    const bool up_has_high = (gate.flags & PASCAL_W4_UP_HIGH_BITPLANE) != 0;
    layer_state::gate_up_format format;
    size_t gate_metadata_bytes = 0;
    size_t up_metadata_bytes = 0;
    if (iq4_pair) {
        if (gate.gate_source_type != GGML_TYPE_IQ4_NL || gate.up_source_type != GGML_TYPE_IQ4_NL ||
                gate.gate_bits != 4 || gate.gate_scales_bytes != iq4_scale_bytes ||
                gate.up_scales_bytes != iq4_scale_bytes || gate.gate_high_bytes != 0 || up_has_high) {
            throw std::runtime_error("invalid source-exact IQ4_NL gate/up payload");
        }
        format = layer_state::gate_up_format::iq4_nl;
        gate_metadata_bytes = iq4_scale_bytes;
        up_metadata_bytes = iq4_scale_bytes;
    } else {
        if (gate.gate_source_type != GGML_TYPE_Q5_K ||
                (gate.up_source_type != GGML_TYPE_Q4_K && gate.up_source_type != GGML_TYPE_Q5_K) ||
                gate.gate_bits != 5 || gate.gate_scales_bytes != kexact_metadata_bytes ||
                gate.gate_high_bytes != high_bytes ||
                gate.up_scales_bytes != kexact_metadata_bytes + (up_has_high ? high_bytes : 0) ||
                up_has_high != (gate.up_source_type == GGML_TYPE_Q5_K)) {
            throw std::runtime_error("invalid source-exact Q5_K/Q4_K-Q5_K gate/up payload");
        }
        format = up_has_high ? layer_state::gate_up_format::kexact_q5_q5
                             : layer_state::gate_up_format::kexact_q5_q4;
        gate_metadata_bytes = kexact_metadata_bytes;
        up_metadata_bytes = kexact_metadata_bytes;
    }

    packed_file down_file(down_path);
    const pascal_kexact_single_header down = down_file.header<pascal_kexact_single_header>();
    if (!has_magic(down.magic, PASCAL_KEXACT_SINGLE_MAGIC) ||
            down.version != PASCAL_KEXACT_SINGLE_VERSION || down.file_bytes != down_file.bytes ||
            down.k != QWEN38_FFN || down.n != QWEN38_EMBED || down.quant_bits != 5 ||
            down.group_size != 32 || down.quant_block != 256 ||
            down.q_bytes != size_t(QWEN38_EMBED) * QWEN38_FFN / 2 ||
            down.high_bytes != size_t(QWEN38_EMBED) * QWEN38_FFN / 8 ||
            down.metadata_bytes != down.dm_bytes + down.scale_min_bytes) {
        throw std::runtime_error("down pack is not the expected source-exact layer geometry");
    }

    auto state = std::make_unique<layer_state>();
    state->device = device;
    state->layer = layer;
    state->format = format;
    state->gate_dm_bytes = iq4_pair ? 0 : kexact_dm_bytes;
    state->up_dm_bytes = iq4_pair ? 0 : kexact_dm_bytes;
    state->down_dm_bytes = static_cast<size_t>(down.dm_bytes);

    state->gate_q.upload(gate_file.slice(gate.gate_q_offset, gate.gate_q_bytes));
    if (gate.gate_high_bytes > 0) {
        state->gate_high.upload(gate_file.slice(gate.gate_high_offset, gate.gate_high_bytes));
    }
    state->gate_metadata.upload(gate_file.slice(gate.gate_scales_offset, gate_metadata_bytes));
    state->up_q.upload(gate_file.slice(gate.up_q_offset, gate.up_q_bytes));
    state->up_metadata.upload(gate_file.slice(gate.up_scales_offset, up_metadata_bytes));
    if (up_has_high) {
        state->up_high.upload(gate_file.slice(gate.up_scales_offset + up_metadata_bytes, high_bytes));
    }
    state->down_q.upload(down_file.slice(down.q_offset, down.q_bytes));
    state->down_high.upload(down_file.slice(down.high_offset, down.high_bytes));
    state->down_metadata.upload(down_file.slice(down.metadata_offset, down.metadata_bytes));
    state->normalized_f16.allocate(QWEN38_EMBED * sizeof(half));
    state->swiglu_f16.allocate(QWEN38_FFN * sizeof(half));
    return state;
}

static layer_state * find_state(int device, int layer) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find({device, layer});
    return it == registry.end() ? nullptr : it->second.get();
}

__global__ void cast_f32_to_f16(const float * input, half * output, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = __float2half_rn(input[index]);
    }
}

template<int BLOCK_SIZE, bool UP_HAS_HIGH>
__global__ void fused_gate_up_kexact_a16_half2(
        const uint32_t * __restrict__ gate_q32,
        const uint32_t * __restrict__ gate_high,
        const uint32_t * __restrict__ up_q32,
        const uint32_t * __restrict__ up_high,
        const uint32_t * __restrict__ gate_dm,
        const uint16_t * __restrict__ gate_scale_min,
        const uint32_t * __restrict__ up_dm,
        const uint16_t * __restrict__ up_scale_min,
        const half2 * __restrict__ input2,
        half * __restrict__ swiglu_output,
        int k,
        int n) {
    constexpr int QUANT_BLOCK = 256;
    constexpr int MAX_BLOCKS = 20;
    constexpr int MAX_GROUPS = MAX_BLOCKS * 8;
    constexpr int WARPS = BLOCK_SIZE / 32;
    const int row = blockIdx.x;
    if (row >= n || k != MAX_BLOCKS * QUANT_BLOCK) return;
    const int blocks_per_row = k / QUANT_BLOCK;
    const int groups_per_row = k / 32;

    __shared__ uint32_t gate_dm_shared[MAX_BLOCKS];
    __shared__ uint32_t up_dm_shared[MAX_BLOCKS];
    __shared__ uint16_t gate_scale_min_shared[MAX_GROUPS];
    __shared__ uint16_t up_scale_min_shared[MAX_GROUPS];
    for (int block = threadIdx.x; block < blocks_per_row; block += BLOCK_SIZE) {
        gate_dm_shared[block] = __ldg(gate_dm + size_t(row) * blocks_per_row + block);
        up_dm_shared[block] = __ldg(up_dm + size_t(row) * blocks_per_row + block);
    }
    for (int group = threadIdx.x; group < groups_per_row; group += BLOCK_SIZE) {
        gate_scale_min_shared[group] = __ldg(gate_scale_min + size_t(row) * groups_per_row + group);
        up_scale_min_shared[group] = __ldg(up_scale_min + size_t(row) * groups_per_row + group);
    }
    __syncthreads();

    half2 gate_acc2[4] = {
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
    };
    half2 up_acc2[4] = {
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
    };
    const int chunks_per_row = k / 8;
    const size_t q_base = size_t(row) * chunks_per_row;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += BLOCK_SIZE) {
        const uint32_t gate_packed = __ldg(gate_q32 + q_base + chunk);
        const uint32_t up_packed = __ldg(up_q32 + q_base + chunk);
        const int element = chunk * 8;
        const int quant_block = element / QUANT_BLOCK;
        const int group = element / 32;
        const uint32_t gate_high_bits = __ldg(gate_high + size_t(row) * groups_per_row + group);
        const uint32_t up_high_bits = UP_HAS_HIGH
            ? __ldg(up_high + size_t(row) * groups_per_row + group)
            : 0;
        const uint32_t gate_dm_bits = gate_dm_shared[quant_block];
        const uint32_t up_dm_bits = up_dm_shared[quant_block];
        const uint16_t gate_sm = gate_scale_min_shared[group];
        const uint16_t up_sm = up_scale_min_shared[group];
        const half gate_d = __ushort_as_half(static_cast<uint16_t>(gate_dm_bits));
        const half gate_dmin = __ushort_as_half(static_cast<uint16_t>(gate_dm_bits >> 16));
        const half up_d = __ushort_as_half(static_cast<uint16_t>(up_dm_bits));
        const half up_dmin = __ushort_as_half(static_cast<uint16_t>(up_dm_bits >> 16));
        const half2 gate_scale2 = __half2half2(__hmul(gate_d, __int2half_rn(gate_sm & 0xff)));
        const half2 gate_minimum2 = __half2half2(__hmul(gate_dmin, __int2half_rn(gate_sm >> 8)));
        const half2 up_scale2 = __half2half2(__hmul(up_d, __int2half_rn(up_sm & 0xff)));
        const half2 up_minimum2 = __half2half2(__hmul(up_dmin, __int2half_rn(up_sm >> 8)));

#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int shift = pair * 8;
            int gate_q0 = int((gate_packed >> shift) & 0x0f);
            int gate_q1 = int((gate_packed >> (shift + 4)) & 0x0f);
            gate_q0 |= int((gate_high_bits >> ((element + pair * 2) & 31)) & 1u) << 4;
            gate_q1 |= int((gate_high_bits >> ((element + pair * 2 + 1) & 31)) & 1u) << 4;
            int up_q0 = int((up_packed >> shift) & 0x0f);
            int up_q1 = int((up_packed >> (shift + 4)) & 0x0f);
            if (UP_HAS_HIGH) {
                up_q0 |= int((up_high_bits >> ((element + pair * 2) & 31)) & 1u) << 4;
                up_q1 |= int((up_high_bits >> ((element + pair * 2 + 1) & 31)) & 1u) << 4;
            }
            const half2 gate_q2 = __halves2half2(__int2half_rn(gate_q0), __int2half_rn(gate_q1));
            const half2 up_q2 = __halves2half2(__int2half_rn(up_q0), __int2half_rn(up_q1));
            const half2 gate_weight2 = __hsub2(__hmul2(gate_q2, gate_scale2), gate_minimum2);
            const half2 up_weight2 = __hsub2(__hmul2(up_q2, up_scale2), up_minimum2);
            const half2 x2 = __ldg(input2 + element / 2 + pair);
            gate_acc2[pair] = __hfma2(gate_weight2, x2, gate_acc2[pair]);
            up_acc2[pair] = __hfma2(up_weight2, x2, up_acc2[pair]);
        }
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 gate_pair = __half22float2(gate_acc2[pair]);
        const float2 up_pair = __half22float2(up_acc2[pair]);
        gate_acc += gate_pair.x + gate_pair.y;
        up_acc += up_pair.x + up_pair.y;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    __shared__ float gate_warp[WARPS];
    __shared__ float up_warp[WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        gate_warp[warp] = gate_acc;
        up_warp[warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < WARPS ? gate_warp[lane] : 0.0f;
        up_acc = lane < WARPS ? up_warp[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
            up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
        }
        if (lane == 0) {
            swiglu_output[row] = __float2half_rn(gate_acc / (1.0f + expf(-gate_acc)) * up_acc);
        }
    }
}

static __device__ __forceinline__ int2 iq4_nl_lookup8(const uint32_t packed) {
    // Resolve all eight nonlinear nibbles with byte permutations.  CUDA has
    // no 4-bit byte lookup, so select from each 8-entry half and then merge on
    // the high index bit.  This mirrors the efficient upstream GGML lookup,
    // but the table words remain immediate/register values here.
    constexpr uint32_t word0 = 0xbfad9881u; // -127, -104, -83, -65
    constexpr uint32_t word1 = 0xf6eaddcfu; //  -49,  -35, -22, -10
    constexpr uint32_t word2 = 0x26190d01u; //    1,   13,  25,  38
    constexpr uint32_t word3 = 0x71594535u; //   53,   69,  89, 113
    const uint32_t selection = 0x32103210u | ((packed & 0x88888888u) >> 1);
    uint32_t values[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int shift = half * 16;
        const uint32_t low = __byte_perm(word0, word1, packed >> shift);
        const uint32_t high = __byte_perm(word2, word3, packed >> shift);
        values[half] = __byte_perm(low, high, selection >> shift);
    }
    return make_int2(
        __byte_perm(values[0], values[1], 0x6420),
        __byte_perm(values[0], values[1], 0x7531));
}

template<int BLOCK_SIZE>
__global__ void fused_gate_up_iq4_nl_a16_half2(
        const uint32_t * __restrict__ gate_q32,
        const uint32_t * __restrict__ up_q32,
        const half * __restrict__ gate_scales,
        const half * __restrict__ up_scales,
        const half2 * __restrict__ input2,
        half * __restrict__ swiglu_output,
        int k,
        int n) {
    constexpr int QUANT_BLOCK = 32;
    constexpr int MAX_GROUPS = QWEN38_EMBED / QUANT_BLOCK;
    constexpr int WARPS = BLOCK_SIZE / 32;
    const int row = blockIdx.x;
    if (row >= n || k != QWEN38_EMBED) return;
    const int groups_per_row = k / QUANT_BLOCK;

    __shared__ half gate_scale_shared[MAX_GROUPS];
    __shared__ half up_scale_shared[MAX_GROUPS];
    for (int group = threadIdx.x; group < groups_per_row; group += BLOCK_SIZE) {
        gate_scale_shared[group] = __ldg(gate_scales + size_t(row) * groups_per_row + group);
        up_scale_shared[group] = __ldg(up_scales + size_t(row) * groups_per_row + group);
    }
    __syncthreads();

    half2 gate_acc2[4] = {
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
    };
    half2 up_acc2[4] = {
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
        __float2half2_rn(0.0f), __float2half2_rn(0.0f),
    };
    const int chunks_per_row = k / 8;
    const size_t q_base = size_t(row) * chunks_per_row;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += BLOCK_SIZE) {
        const uint32_t gate_packed = __ldg(gate_q32 + q_base + chunk);
        const uint32_t up_packed = __ldg(up_q32 + q_base + chunk);
        const int element = chunk * 8;
        const int group = element / QUANT_BLOCK;
        const half2 gate_scale2 = __half2half2(gate_scale_shared[group]);
        const half2 up_scale2 = __half2half2(up_scale_shared[group]);
        const int2 gate_values = iq4_nl_lookup8(gate_packed);
        const int2 up_values = iq4_nl_lookup8(up_packed);

#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int shift = pair * 8;
            const half2 gate_value2 = __halves2half2(
                __int2half_rn(static_cast<int8_t>(gate_values.x >> shift)),
                __int2half_rn(static_cast<int8_t>(gate_values.y >> shift)));
            const half2 up_value2 = __halves2half2(
                __int2half_rn(static_cast<int8_t>(up_values.x >> shift)),
                __int2half_rn(static_cast<int8_t>(up_values.y >> shift)));
            const half2 gate_weight2 = __hmul2(gate_value2, gate_scale2);
            const half2 up_weight2 = __hmul2(up_value2, up_scale2);
            const half2 x2 = __ldg(input2 + element / 2 + pair);
            gate_acc2[pair] = __hfma2(gate_weight2, x2, gate_acc2[pair]);
            up_acc2[pair] = __hfma2(up_weight2, x2, up_acc2[pair]);
        }
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 gate_pair = __half22float2(gate_acc2[pair]);
        const float2 up_pair = __half22float2(up_acc2[pair]);
        gate_acc += gate_pair.x + gate_pair.y;
        up_acc += up_pair.x + up_pair.y;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
    }
    __shared__ float gate_warp[WARPS];
    __shared__ float up_warp[WARPS];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        gate_warp[warp] = gate_acc;
        up_warp[warp] = up_acc;
    }
    __syncthreads();
    if (warp == 0) {
        gate_acc = lane < WARPS ? gate_warp[lane] : 0.0f;
        up_acc = lane < WARPS ? up_warp[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            gate_acc += __shfl_down_sync(0xffffffffu, gate_acc, offset);
            up_acc += __shfl_down_sync(0xffffffffu, up_acc, offset);
        }
        if (lane == 0) {
            swiglu_output[row] = __float2half_rn(gate_acc / (1.0f + expf(-gate_acc)) * up_acc);
        }
    }
}

template<int BLOCK_SIZE, int ROWS_PER_BLOCK>
__global__ void down_kexact_a16_half2_multirow(
        const uint32_t * __restrict__ q32,
        const uint32_t * __restrict__ high,
        const uint32_t * __restrict__ dm,
        const uint16_t * __restrict__ scale_min,
        const half2 * __restrict__ input2,
        const float * __restrict__ residual,
        float * __restrict__ layer_output,
        int k,
        int n) {
    constexpr int QUANT_BLOCK = 256;
    constexpr int MAX_BLOCKS = 68;
    constexpr int MAX_GROUPS = MAX_BLOCKS * 8;
    constexpr int WARPS = BLOCK_SIZE / 32;
    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    if (row_base >= n || k != MAX_BLOCKS * QUANT_BLOCK) return;
    const int blocks_per_row = k / QUANT_BLOCK;
    const int groups_per_row = k / 32;

    __shared__ uint32_t dm_shared[ROWS_PER_BLOCK][MAX_BLOCKS];
    __shared__ uint16_t scale_min_shared[ROWS_PER_BLOCK][MAX_GROUPS];
    for (int index = threadIdx.x; index < ROWS_PER_BLOCK * blocks_per_row; index += BLOCK_SIZE) {
        const int local_row = index / blocks_per_row;
        const int block = index - local_row * blocks_per_row;
        dm_shared[local_row][block] = __ldg(dm + size_t(row_base + local_row) * blocks_per_row + block);
    }
    for (int index = threadIdx.x; index < ROWS_PER_BLOCK * groups_per_row; index += BLOCK_SIZE) {
        const int local_row = index / groups_per_row;
        const int group = index - local_row * groups_per_row;
        scale_min_shared[local_row][group] =
            __ldg(scale_min + size_t(row_base + local_row) * groups_per_row + group);
    }
    __syncthreads();

    half2 accumulator2[ROWS_PER_BLOCK][4];
#pragma unroll
    for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) accumulator2[local_row][pair] = __float2half2_rn(0.0f);
    }

    const int chunks_per_row = k / 8;
    for (int chunk = threadIdx.x; chunk < chunks_per_row; chunk += BLOCK_SIZE) {
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
            packed[local_row] = __ldg(q32 + size_t(row) * chunks_per_row + chunk);
            high_bits[local_row] = __ldg(high + size_t(row) * groups_per_row + group);
            const uint32_t dm_bits = dm_shared[local_row][quant_block];
            const uint16_t sm = scale_min_shared[local_row][group];
            const half d = __ushort_as_half(static_cast<uint16_t>(dm_bits));
            const half dmin = __ushort_as_half(static_cast<uint16_t>(dm_bits >> 16));
            scale2[local_row] = __half2half2(__hmul(d, __int2half_rn(sm & 0xff)));
            minimum2[local_row] = __half2half2(__hmul(dmin, __int2half_rn(sm >> 8)));
        }

#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const int shift = pair * 8;
            const half2 x2 = __ldg(input2 + element / 2 + pair);
#pragma unroll
            for (int local_row = 0; local_row < ROWS_PER_BLOCK; ++local_row) {
                int q0 = int((packed[local_row] >> shift) & 0x0f);
                int q1 = int((packed[local_row] >> (shift + 4)) & 0x0f);
                q0 |= int((high_bits[local_row] >> ((element + pair * 2) & 31)) & 1u) << 4;
                q1 |= int((high_bits[local_row] >> ((element + pair * 2 + 1) & 31)) & 1u) << 4;
                const half2 quant2 = __halves2half2(__int2half_rn(q0), __int2half_rn(q1));
                const half2 weight2 = __hsub2(__hmul2(quant2, scale2[local_row]), minimum2[local_row]);
                accumulator2[local_row][pair] = __hfma2(weight2, x2, accumulator2[local_row][pair]);
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
            if (lane == 0) {
                const int row = row_base + local_row;
                layer_output[row] = accumulator + __ldg(residual + row);
            }
        }
    }
}

} // namespace

extern "C" bool ggml_backend_cuda_qwen38_pascal_ffn_register(
        int device, int layer, const char * gate_up_path, const char * down_path,
        char * error, size_t error_size) {
    try {
        if (!gate_up_path || !down_path) {
            throw std::runtime_error("packed weight paths are required");
        }
        auto state = load_layer(device, layer, gate_up_path, down_path);
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

extern "C" void ggml_backend_cuda_qwen38_pascal_ffn_unregister(int device, int layer) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto it = registry.find({device, layer});
    if (it != registry.end()) {
        ggml_cuda_set_device(device);
        registry.erase(it);
    }
}

extern "C" bool ggml_backend_cuda_qwen38_pascal_ffn_is_registered(int device, int layer) {
    return ggml_cuda_qwen38_pascal_ffn_registered(device, layer);
}

bool ggml_cuda_qwen38_pascal_ffn_registered(int device, int layer) {
    return find_state(device, layer) != nullptr;
}

void ggml_cuda_op_qwen38_pascal_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]);
    GGML_ASSERT(dst->src[1]);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst->src[1]));
    GGML_ASSERT(ggml_nelements(dst) == QWEN38_EMBED);

    const int layer = ggml_get_op_params_i32(dst, 0);
    layer_state * state = find_state(ctx.device, layer);
    GGML_ASSERT(state && "Qwen3.8 Pascal FFN pack was not registered on this CUDA device");

    auto * normalized_half = static_cast<half *>(state->normalized_f16.ptr);
    auto * swiglu_half = static_cast<half *>(state->swiglu_f16.ptr);
    const auto * gate_metadata = static_cast<const uint8_t *>(state->gate_metadata.ptr);
    const auto * up_metadata = static_cast<const uint8_t *>(state->up_metadata.ptr);
    const auto * down_metadata = static_cast<const uint8_t *>(state->down_metadata.ptr);
    cudaStream_t stream = ctx.stream();

    cast_f32_to_f16<<<(QWEN38_EMBED + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(dst->src[0]->data), normalized_half, QWEN38_EMBED);

    switch (state->format) {
        case layer_state::gate_up_format::kexact_q5_q4:
            fused_gate_up_kexact_a16_half2<32, false><<<QWEN38_FFN, 32, 0, stream>>>(
                static_cast<const uint32_t *>(state->gate_q.ptr),
                static_cast<const uint32_t *>(state->gate_high.ptr),
                static_cast<const uint32_t *>(state->up_q.ptr), nullptr,
                reinterpret_cast<const uint32_t *>(gate_metadata),
                reinterpret_cast<const uint16_t *>(gate_metadata + state->gate_dm_bytes),
                reinterpret_cast<const uint32_t *>(up_metadata),
                reinterpret_cast<const uint16_t *>(up_metadata + state->up_dm_bytes),
                reinterpret_cast<const half2 *>(normalized_half),
                swiglu_half, QWEN38_EMBED, QWEN38_FFN);
            break;
        case layer_state::gate_up_format::kexact_q5_q5:
            fused_gate_up_kexact_a16_half2<32, true><<<QWEN38_FFN, 32, 0, stream>>>(
                static_cast<const uint32_t *>(state->gate_q.ptr),
                static_cast<const uint32_t *>(state->gate_high.ptr),
                static_cast<const uint32_t *>(state->up_q.ptr),
                static_cast<const uint32_t *>(state->up_high.ptr),
                reinterpret_cast<const uint32_t *>(gate_metadata),
                reinterpret_cast<const uint16_t *>(gate_metadata + state->gate_dm_bytes),
                reinterpret_cast<const uint32_t *>(up_metadata),
                reinterpret_cast<const uint16_t *>(up_metadata + state->up_dm_bytes),
                reinterpret_cast<const half2 *>(normalized_half),
                swiglu_half, QWEN38_EMBED, QWEN38_FFN);
            break;
        case layer_state::gate_up_format::iq4_nl:
            fused_gate_up_iq4_nl_a16_half2<32><<<QWEN38_FFN, 32, 0, stream>>>(
                static_cast<const uint32_t *>(state->gate_q.ptr),
                static_cast<const uint32_t *>(state->up_q.ptr),
                reinterpret_cast<const half *>(gate_metadata),
                reinterpret_cast<const half *>(up_metadata),
                reinterpret_cast<const half2 *>(normalized_half),
                swiglu_half, QWEN38_EMBED, QWEN38_FFN);
            break;
    }

    down_kexact_a16_half2_multirow<128, 4><<<(QWEN38_EMBED + 3) / 4, 128, 0, stream>>>(
        static_cast<const uint32_t *>(state->down_q.ptr),
        static_cast<const uint32_t *>(state->down_high.ptr),
        reinterpret_cast<const uint32_t *>(down_metadata),
        reinterpret_cast<const uint16_t *>(down_metadata + state->down_dm_bytes),
        reinterpret_cast<const half2 *>(swiglu_half),
        static_cast<const float *>(dst->src[1]->data),
        static_cast<float *>(dst->data), QWEN38_FFN, QWEN38_EMBED);
}
