#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using json = nlohmann::json;

void seek_file(FILE * file, uint64_t offset, const char * what) {
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        throw std::runtime_error(std::string("seek failed for ") + what + ": " + std::strerror(errno));
    }
}

void read_exact(FILE * file, void * data, size_t bytes, const char * what) {
    if (bytes != 0 && std::fread(data, 1, bytes, file) != bytes) {
        throw std::runtime_error(std::string("short read for ") + what);
    }
}

void write_exact(FILE * file, const void * data, size_t bytes, const char * what) {
    if (bytes != 0 && std::fwrite(data, 1, bytes, file) != bytes) {
        throw std::runtime_error(std::string("short write for ") + what);
    }
}

struct file_guard {
    FILE * value = nullptr;
    explicit file_guard(FILE * file) : value(file) {}
    ~file_guard() {
        if (value) {
            std::fclose(value);
        }
    }
};

struct gguf_guard {
    gguf_context * value = nullptr;
    explicit gguf_guard(gguf_context * ctx) : value(ctx) {}
    ~gguf_guard() {
        if (value) {
            gguf_free(value);
        }
    }
};

std::vector<uint16_t> read_bf16_tensor(
        const std::string & path,
        const std::string & name,
        int64_t expected_rows,
        int64_t expected_columns) {
    file_guard input(std::fopen(path.c_str(), "rb"));
    if (!input.value) {
        throw std::runtime_error("cannot open BF16 safetensors shard: " + path);
    }

    uint64_t header_bytes = 0;
    read_exact(input.value, &header_bytes, sizeof(header_bytes), "safetensors header length");
    if (header_bytes == 0 || header_bytes > (1ULL << 30)) {
        throw std::runtime_error("invalid safetensors header length");
    }

    std::string header(header_bytes, '\0');
    read_exact(input.value, header.data(), header.size(), "safetensors header");
    const json metadata = json::parse(header);
    if (!metadata.contains(name)) {
        throw std::runtime_error("safetensors shard does not contain " + name);
    }
    const json & tensor = metadata.at(name);
    if (tensor.at("dtype").get<std::string>() != "BF16") {
        throw std::runtime_error(name + " is not BF16");
    }
    const auto shape = tensor.at("shape").get<std::vector<int64_t>>();
    if (shape != std::vector<int64_t>{expected_rows, expected_columns}) {
        throw std::runtime_error("unexpected " + name + " shape");
    }
    const auto offsets = tensor.at("data_offsets").get<std::vector<uint64_t>>();
    if (offsets.size() != 2 || offsets[1] < offsets[0]) {
        throw std::runtime_error("invalid safetensors data offsets");
    }
    const uint64_t elements = uint64_t(expected_rows) * uint64_t(expected_columns);
    if (offsets[1] - offsets[0] != elements * sizeof(uint16_t)) {
        throw std::runtime_error("BF16 tensor byte count does not match its shape");
    }

    std::vector<uint16_t> result(elements);
    seek_file(input.value, sizeof(uint64_t) + header_bytes + offsets[0], name.c_str());
    read_exact(input.value, result.data(), result.size() * sizeof(uint16_t), name.c_str());
    return result;
}

std::vector<float> bf16_to_float(const std::vector<uint16_t> & input, unsigned threads) {
    std::vector<float> output(input.size());
    const uint64_t chunk = (input.size() + threads - 1) / threads;
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < threads; ++worker) {
        const uint64_t begin = uint64_t(worker) * chunk;
        const uint64_t end = std::min<uint64_t>(input.size(), begin + chunk);
        if (begin >= end) {
            continue;
        }
        workers.emplace_back([&, begin, end]() {
            for (uint64_t index = begin; index < end; ++index) {
                output[index] = std::bit_cast<float>(uint32_t(input[index]) << 16);
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    return output;
}

std::vector<uint8_t> quantize_k_quant(
        const std::vector<float> & input,
        int64_t rows,
        int64_t columns,
        unsigned threads,
        ggml_type output_type) {
    const size_t row_bytes = ggml_row_size(output_type, columns);
    std::vector<uint8_t> output(row_bytes * rows);
    std::atomic<int64_t> next_row{0};
    std::atomic<uint64_t> written{0};
    constexpr int64_t rows_per_chunk = 64;

    ggml_quantize_init(output_type);
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&]() {
            while (true) {
                const int64_t first = next_row.fetch_add(rows_per_chunk);
                if (first >= rows) {
                    return;
                }
                const int64_t count = std::min<int64_t>(rows_per_chunk, rows - first);
                const size_t bytes = ggml_quantize_chunk(
                    output_type,
                    input.data(),
                    output.data(),
                    first * columns,
                    count,
                    columns,
                    nullptr);
                written.fetch_add(bytes);
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
    ggml_quantize_free();

    if (written.load() != output.size()) {
        throw std::runtime_error("Q4_K quantizer returned an unexpected byte count");
    }
    return output;
}

void copy_bytes(FILE * source, FILE * destination, uint64_t bytes, std::vector<uint8_t> & buffer) {
    while (bytes != 0) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(bytes, buffer.size()));
        read_exact(source, buffer.data(), count, "source GGUF tensor");
        write_exact(destination, buffer.data(), count, "destination GGUF tensor");
        bytes -= count;
    }
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: " << argv[0]
                      << " SOURCE_Q4_K_M.gguf OFFICIAL_BF16_SHARD.safetensors OUTPUT.gguf [q4_k|q5_k]\n";
            return 2;
        }
        const std::string source_path = argv[1];
        const std::string bf16_path = argv[2];
        const std::string output_path = argv[3];
        const std::string quant_name = argc == 5 ? argv[4] : "q4_k";
        const ggml_type output_type = quant_name == "q4_k" ? GGML_TYPE_Q4_K :
            quant_name == "q5_k" ? GGML_TYPE_Q5_K : GGML_TYPE_COUNT;
        if (output_type == GGML_TYPE_COUNT) {
            throw std::runtime_error("output quantization must be q4_k or q5_k");
        }
        std::error_code equivalent_error;
        if (std::filesystem::equivalent(source_path, output_path, equivalent_error)) {
            throw std::runtime_error("source and destination GGUF paths must differ");
        }

        ggml_context * tensor_context = nullptr;
        gguf_guard metadata(gguf_init_from_file(
            source_path.c_str(),
            gguf_init_params{ /*.no_alloc =*/ true, /*.ctx =*/ &tensor_context }));
        if (!metadata.value || !tensor_context) {
            throw std::runtime_error("failed to read source GGUF metadata");
        }
        const int64_t output_id = gguf_find_tensor(metadata.value, "output.weight");
        if (output_id < 0 || gguf_get_tensor_type(metadata.value, output_id) != GGML_TYPE_Q6_K) {
            throw std::runtime_error("source output.weight is not the expected Q6_K tensor");
        }
        const int64_t * dimensions = gguf_get_tensor_ne(metadata.value, output_id);
        const int64_t columns = dimensions[0];
        const int64_t rows = dimensions[1];
        if (columns != 5120 || rows != 248320) {
            throw std::runtime_error("source output.weight has an unexpected shape");
        }

        const int64_t n_tensors = gguf_get_n_tensors(metadata.value);
        std::vector<size_t> old_offsets(n_tensors);
        std::vector<size_t> old_sizes(n_tensors);
        for (int64_t index = 0; index < n_tensors; ++index) {
            old_offsets[index] = gguf_get_tensor_offset(metadata.value, index);
            old_sizes[index] = gguf_get_tensor_size(metadata.value, index);
        }
        const size_t old_data_offset = gguf_get_data_offset(metadata.value);

        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::cerr << "reading official BF16 lm_head.weight...\n";
        std::vector<uint16_t> bf16 = read_bf16_tensor(bf16_path, "lm_head.weight", rows, columns);
        std::cerr << "converting " << bf16.size() << " BF16 values to FP32 with " << threads << " threads...\n";
        std::vector<float> fp32 = bf16_to_float(bf16, threads);
        bf16.clear();
        bf16.shrink_to_fit();
        std::cerr << "quantizing official BF16 lm_head.weight directly to "
                  << ggml_type_name(output_type) << "...\n";
        std::vector<uint8_t> quantized = quantize_k_quant(
            fp32, rows, columns, threads, output_type);
        fp32.clear();
        fp32.shrink_to_fit();

        gguf_set_tensor_type(metadata.value, "output.weight", output_type);
        gguf_set_val_str(metadata.value, "qwen38.output_weight.source",
                         "Qwen/Qwen3.8-27B model-00018-of-00018.safetensors BF16 lm_head.weight");
        gguf_set_val_str(metadata.value, "qwen38.output_weight.quantization",
                         ("direct BF16 to " + std::string(ggml_type_name(output_type)) +
                          " without an importance matrix").c_str());
        if (gguf_get_tensor_size(metadata.value, output_id) != quantized.size()) {
            throw std::runtime_error("GGUF output tensor size differs from quantizer output");
        }

        if (!gguf_write_to_file(metadata.value, output_path.c_str(), /* only_meta = */ true)) {
            throw std::runtime_error("failed to write destination GGUF metadata");
        }
        file_guard source(std::fopen(source_path.c_str(), "rb"));
        file_guard destination(std::fopen(output_path.c_str(), "rb+"));
        if (!source.value || !destination.value) {
            throw std::runtime_error("failed to open source or destination GGUF data section");
        }
        const size_t new_data_offset = gguf_get_meta_size(metadata.value);
        std::vector<uint8_t> copy_buffer(64 * 1024 * 1024);

        for (int64_t index = 0; index < n_tensors; ++index) {
            const size_t new_offset = gguf_get_tensor_offset(metadata.value, index);
            seek_file(destination.value, new_data_offset + new_offset, "destination tensor");
            if (index == output_id) {
                write_exact(destination.value, quantized.data(), quantized.size(), "quantized output.weight");
            } else {
                seek_file(source.value, old_data_offset + old_offsets[index], "source tensor");
                copy_bytes(source.value, destination.value, old_sizes[index], copy_buffer);
            }
        }
        uint64_t final_size = new_data_offset;
        for (int64_t index = 0; index < n_tensors; ++index) {
            final_size = std::max<uint64_t>(
                final_size,
                new_data_offset + gguf_get_tensor_offset(metadata.value, index) +
                    gguf_get_tensor_size(metadata.value, index));
        }
        if (fflush(destination.value) != 0 || ftruncate(fileno(destination.value), final_size) != 0) {
            throw std::runtime_error("failed to finalize destination GGUF");
        }

        std::cout << "output=" << output_path << "\n"
                  << "output_type=" << ggml_type_name(output_type) << "\n"
                  << "output_bytes=" << quantized.size() << "\n"
                  << "saved_bytes=" << old_sizes[output_id] - quantized.size() << "\n"
                  << "file_bytes=" << final_size << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen38_rehead: " << error.what() << '\n';
        return 1;
    }
}
