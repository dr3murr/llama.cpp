#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "ggml-cuda.h"
#include "llama.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <clocale>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <signal.h>
#include <string>
#include <thread> // for std::thread::hardware_concurrency
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

namespace {

struct qwen38_hybrid_launch {
    struct pascal_ffn_pack {
        int                   layer;
        std::filesystem::path gate_up;
        std::filesystem::path down;
    };

    std::vector<std::string> arguments;
    std::vector<char *>      argv;
    std::vector<pascal_ffn_pack> pascal_ffn_packs;
};

static std::vector<int> qwen38_registered_pascal_ffn_layers;

static void qwen38_unregister_pascal_ffn_packs() {
    for (auto it = qwen38_registered_pascal_ffn_layers.rbegin();
         it != qwen38_registered_pascal_ffn_layers.rend(); ++it) {
        ggml_backend_cuda_qwen38_pascal_ffn_unregister(0, *it);
    }
    qwen38_registered_pascal_ffn_layers.clear();
}

static void qwen38_backend_free() {
    ggml_backend_cuda_qwen38_pascal_resident_fp16_clear();
    qwen38_unregister_pascal_ffn_packs();
    llama_backend_free();
}

static bool qwen38_set_environment(const char * name, const std::string & value) {
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

static bool qwen38_has_option(
        const std::vector<std::string> & arguments,
        std::initializer_list<const char *> names) {
    for (const std::string & argument : arguments) {
        for (const char * name : names) {
            if (argument == name || argument.rfind(std::string(name) + "=", 0) == 0) {
                return true;
            }
        }
    }
    return false;
}

static void qwen38_append_default(
        std::vector<std::string> & arguments,
        std::initializer_list<const char *> names,
        const char * option,
        const std::string & value) {
    if (!qwen38_has_option(arguments, names)) {
        arguments.emplace_back(option);
        arguments.push_back(value);
    }
}

static std::optional<qwen38_hybrid_launch> qwen38_prepare_hybrid_launch(
        int argc, char ** argv) {
    int model_index = -1;
    std::filesystem::path manifest_path;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if ((argument == "-m" || argument == "--model") && index + 1 < argc) {
            model_index = index + 1;
            manifest_path = argv[model_index];
            break;
        }
        constexpr const char * prefix = "--model=";
        if (argument.rfind(prefix, 0) == 0) {
            model_index = index;
            manifest_path = argument.substr(std::strlen(prefix));
            break;
        }
    }
    if (model_index < 0 || manifest_path.extension() != ".json") {
        return std::nullopt;
    }

    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error("failed to open Qwen3.8 hybrid manifest: " + manifest_path.string());
    }
    nlohmann::json manifest;
    input >> manifest;
    if (manifest.value("schema", std::string()) != "qwen38-hybrid-manifest-v1") {
        throw std::runtime_error(
            "JSON passed through -m is not a qwen38-hybrid-manifest-v1 manifest: " +
            manifest_path.string());
    }

    const auto & model = manifest.at("model");
    const auto & runtime = manifest.at("runtime");
    const std::string model_value = model.at("path").get<std::string>();
    std::filesystem::path model_path(model_value);
    if (model_path.is_relative()) {
        model_path = manifest_path.parent_path() / model_path;
    }
    model_path = std::filesystem::weakly_canonical(model_path);
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("hybrid target model does not exist: " + model_path.string());
    }

    qwen38_hybrid_launch launch;

    const std::string visible_devices = runtime.at("cuda_visible_devices").get<std::string>();
    if (visible_devices.empty() || !qwen38_set_environment("CUDA_VISIBLE_DEVICES", visible_devices)) {
        throw std::runtime_error("failed to set CUDA_VISIBLE_DEVICES from hybrid manifest");
    }
    if (!qwen38_set_environment("GGML_CUDA_P2P", "1")) {
        throw std::runtime_error("failed to enable GGML_CUDA_P2P from hybrid manifest");
    }
    if (runtime.contains("direct_q8_mma") &&
        !qwen38_set_environment(
            "QWEN38_CUDA_DIRECT_Q8_MMA",
            runtime.at("direct_q8_mma").get<bool>() ? "1" : "0")) {
        throw std::runtime_error("failed to set direct-Q8 attention policy from hybrid manifest");
    }
    if (runtime.contains("pascal_kq_hfma2") &&
        !qwen38_set_environment(
            "QWEN38_CUDA_PASCAL_KQ_HFMA2",
            runtime.at("pascal_kq_hfma2").get<bool>() ? "1" : "0")) {
        throw std::runtime_error("failed to set Pascal HFMA2 KQ policy from hybrid manifest");
    }
    if (runtime.contains("pascal_prefill_cublas_algorithm")) {
        const int algorithm = runtime.at("pascal_prefill_cublas_algorithm").get<int>();
        if (algorithm != 6) {
            throw std::runtime_error("hybrid manifest only permits the qualified Pascal prefill cuBLAS algorithm 6");
        }
        if (!qwen38_set_environment(
                "QWEN38_CUDA_PASCAL_PREFILL_CUBLAS_ALGORITHM",
                std::to_string(algorithm))) {
            throw std::runtime_error("failed to set Pascal prefill cuBLAS policy from hybrid manifest");
        }
    }
    if (runtime.contains("pascal_prefill_cublas_gdn_qkv_algorithm")) {
        const int algorithm = runtime.at("pascal_prefill_cublas_gdn_qkv_algorithm").get<int>();
        if (algorithm != 2 ||
            !qwen38_set_environment(
                "QWEN38_CUDA_PASCAL_PREFILL_CUBLAS_GDN_QKV_ALGORITHM",
                std::to_string(algorithm))) {
            throw std::runtime_error("hybrid manifest only permits qualified Pascal GDN-QKV cuBLAS algorithm 2");
        }
    }
    if (runtime.contains("pascal_prefill_cublas_attn_q_algorithm")) {
        const int algorithm = runtime.at("pascal_prefill_cublas_attn_q_algorithm").get<int>();
        if (algorithm != 3 ||
            !qwen38_set_environment(
                "QWEN38_CUDA_PASCAL_PREFILL_CUBLAS_ATTN_Q_ALGORITHM",
                std::to_string(algorithm))) {
            throw std::runtime_error("hybrid manifest only permits qualified Pascal attention-Q cuBLAS algorithm 3");
        }
    }
    if (runtime.contains("pascal_prefill_cublas_gdn_z_algorithm")) {
        const int algorithm = runtime.at("pascal_prefill_cublas_gdn_z_algorithm").get<int>();
        if (algorithm != 6 ||
            !qwen38_set_environment(
                "QWEN38_CUDA_PASCAL_PREFILL_CUBLAS_GDN_Z_ALGORITHM",
                std::to_string(algorithm))) {
            throw std::runtime_error("hybrid manifest only permits qualified Pascal GDN-Z cuBLAS algorithm 6");
        }
    }
    if (runtime.contains("pascal_prefill_resident_fp16")) {
        const std::string selection = runtime.at("pascal_prefill_resident_fp16").get<std::string>();
        if (selection != "both_plus_gdn_z" ||
            !qwen38_set_environment("QWEN38_CUDA_PASCAL_PREFILL_RESIDENT_FP16", selection)) {
            throw std::runtime_error(
                "hybrid manifest only permits the qualified Pascal resident-FP16 selection 'both_plus_gdn_z'");
        }
    }
    if (runtime.contains("streamed_attention")) {
        const auto & streamed_attention = runtime.at("streamed_attention");
        if (!streamed_attention.is_object()) {
            throw std::runtime_error("hybrid streamed-attention policy must be an object");
        }
        const bool enabled = streamed_attention.value("enabled", false);
        const std::string synchronization = streamed_attention.value("synchronization", std::string("host"));
        const int tile_tokens = streamed_attention.value("tile_tokens", 8192);
        const int minimum_kv_tokens = streamed_attention.value("minimum_kv_tokens", 65536);
        const int minimum_q_tokens = streamed_attention.value("minimum_q_tokens", 128);
        const int maximum_q_tokens = streamed_attention.value("maximum_q_tokens", 256);
        const int local_kv_heads = streamed_attention.value("local_kv_heads", 0);
        const bool source_pack = streamed_attention.value("source_pack", false);
        const bool bounded_local = streamed_attention.value("bounded_local", false);
        const bool persistent_head_major_rtx = streamed_attention.value("persistent_head_major_rtx", false);
        const std::string rtx_mma_shape = streamed_attention.value("rtx_mma_shape", std::string("8x8"));
        if (synchronization != "host" ||
            (tile_tokens != 4096 && tile_tokens != 8192 && tile_tokens != 16384) ||
            minimum_kv_tokens <= 0 || minimum_q_tokens <= 0 ||
            (maximum_q_tokens != 256 && maximum_q_tokens != 512) ||
            minimum_q_tokens > maximum_q_tokens ||
            (local_kv_heads != 0 && local_kv_heads != 1) ||
            (source_pack && local_kv_heads != 1) ||
            (rtx_mma_shape != "8x8" && rtx_mma_shape != "32x2") ||
            (rtx_mma_shape == "32x2" && !persistent_head_major_rtx)) {
            throw std::runtime_error("hybrid streamed-attention policy has invalid synchronization or shape limits");
        }
        if (!qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN", enabled ? "1" : "0") ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_TILE", std::to_string(tile_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_MIN_KV", std::to_string(minimum_kv_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_MIN_Q", std::to_string(minimum_q_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_MAX_Q", std::to_string(maximum_q_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_LOCAL_KV_HEADS", std::to_string(local_kv_heads)) ||
            !qwen38_set_environment("QWEN38_CUDA_STREAMED_ATTN_SOURCE_PACK", source_pack ? "1" : "0") ||
            !qwen38_set_environment("QWEN38_CUDA_BOUNDED_LOCAL_ATTN", bounded_local ? "1" : "0") ||
            !qwen38_set_environment("QWEN38_CUDA_PERSISTENT_HEAD_MAJOR_KV", persistent_head_major_rtx ? "1" : "0") ||
            !qwen38_set_environment("QWEN38_CUDA_DIRECT_Q8_SHAPE", rtx_mma_shape == "32x2" ? "3202" : "808")) {
            throw std::runtime_error("failed to set streamed-attention policy from hybrid manifest");
        }
    }
    if (runtime.contains("sage_attention")) {
        const auto & sage_attention = runtime.at("sage_attention");
        if (!sage_attention.is_object()) {
            throw std::runtime_error("hybrid Sage-attention policy must be an object");
        }
        const bool enabled = sage_attention.value("enabled", false);
        const int minimum_kv_tokens = sage_attention.value("minimum_kv_tokens", 65536);
        const int minimum_q_tokens = sage_attention.value("minimum_q_tokens", 65);
        const bool log_transitions = sage_attention.value("log_transitions", false);
        if (minimum_kv_tokens <= 0 || minimum_kv_tokens > 262144 ||
            minimum_q_tokens <= 0 || minimum_q_tokens > 512) {
            throw std::runtime_error("hybrid Sage-attention policy has invalid token thresholds");
        }
        if (!qwen38_set_environment("QWEN38_CUDA_SAGE_ATTN", enabled ? "1" : "0") ||
            !qwen38_set_environment("QWEN38_CUDA_SAGE_ATTN_MIN_KV", std::to_string(minimum_kv_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_SAGE_ATTN_MIN_Q", std::to_string(minimum_q_tokens)) ||
            !qwen38_set_environment("QWEN38_CUDA_SAGE_ATTN_LOG", log_transitions ? "1" : "0")) {
            throw std::runtime_error("failed to set Sage-attention policy from hybrid manifest");
        }
    }
    if (runtime.contains("layer_major_prefill")) {
        const auto & layer_major = runtime.at("layer_major_prefill");
        if (!layer_major.is_object()) {
            throw std::runtime_error("hybrid layer-major prefill policy must be an object");
        }
        const bool enabled = layer_major.value("enabled", false);
        const int minimum_tokens = layer_major.value("minimum_tokens", 2048);
        const int minimum_uncached_tokens = layer_major.value("minimum_uncached_tokens", 2048);
        const int all_rtx_maximum_tokens = layer_major.value("all_rtx_maximum_tokens", 32768);
        const int maximum_tokens = layer_major.value("maximum_tokens", 262144);
        const int chunk_tokens = layer_major.value("chunk_tokens", 512);
        if (minimum_tokens < 2 || minimum_uncached_tokens < 1 ||
                maximum_tokens < minimum_tokens || maximum_tokens > 262144 ||
                all_rtx_maximum_tokens < minimum_tokens || all_rtx_maximum_tokens > maximum_tokens ||
                (chunk_tokens != 128 && chunk_tokens != 256 && chunk_tokens != 512)) {
            throw std::runtime_error("hybrid layer-major prefill policy has invalid token limits");
        }
        if (!qwen38_set_environment("QWEN38_SERVER_LAYER_MAJOR_PREFILL", enabled ? "1" : "0") ||
                !qwen38_set_environment(
                    "QWEN38_SERVER_LAYER_MAJOR_MIN_TOKENS", std::to_string(minimum_tokens)) ||
                !qwen38_set_environment(
                    "QWEN38_SERVER_LAYER_MAJOR_MIN_UNCACHED_TOKENS", std::to_string(minimum_uncached_tokens)) ||
                !qwen38_set_environment(
                    "QWEN38_SERVER_LAYER_MAJOR_MAX_TOKENS", std::to_string(maximum_tokens)) ||
                !qwen38_set_environment(
                    "QWEN38_SERVER_LAYER_MAJOR_ALL_RTX_MAX_TOKENS",
                    std::to_string(all_rtx_maximum_tokens)) ||
                !qwen38_set_environment(
                    "QWEN38_SERVER_LAYER_MAJOR_CHUNK", std::to_string(chunk_tokens))) {
            throw std::runtime_error("failed to set layer-major prefill policy from hybrid manifest");
        }
    }
    if (!qwen38_set_environment(
            "QWEN38_PIPELINE_PARALLEL",
            runtime.value("pipeline_parallel", true) ? "1" : "0")) {
        throw std::runtime_error("failed to set pipeline policy from hybrid manifest");
    }
    if (runtime.contains("graph_reuse")) {
        const bool graph_reuse = runtime.at("graph_reuse").get<bool>();
        if (!qwen38_set_environment("LLAMA_GRAPH_REUSE_DISABLE", graph_reuse ? "0" : "1")) {
            throw std::runtime_error("failed to set graph-reuse policy from hybrid manifest");
        }
    }
    if (runtime.contains("speculative_batch_size")) {
        if (!qwen38_set_environment(
                "QWEN38_SPEC_BATCH",
                std::to_string(runtime.at("speculative_batch_size").get<int>()))) {
            throw std::runtime_error("failed to set speculative batch size from hybrid manifest");
        }
    }

    if (runtime.contains("pascal_ffn_packs")) {
        std::string enabled_layers;
        for (const auto & value : runtime.at("pascal_ffn_packs")) {
            qwen38_hybrid_launch::pascal_ffn_pack pack;
            pack.layer = value.at("layer").get<int>();
            if (pack.layer < 0 || pack.layer >= 64) {
                throw std::runtime_error("hybrid Pascal FFN pack layer is outside 0..63");
            }
            auto resolve_pack = [&](const char * key) {
                std::filesystem::path path(value.at(key).get<std::string>());
                if (path.is_relative()) {
                    path = manifest_path.parent_path() / path;
                }
                path = std::filesystem::weakly_canonical(path);
                if (!std::filesystem::is_regular_file(path)) {
                    throw std::runtime_error(
                        std::string("hybrid Pascal FFN pack does not exist: ") + path.string());
                }
                return path;
            };
            pack.gate_up = resolve_pack("gate_up");
            pack.down = resolve_pack("down");
            launch.pascal_ffn_packs.push_back(std::move(pack));
            if (!enabled_layers.empty()) {
                enabled_layers += ',';
            }
            enabled_layers += std::to_string(value.at("layer").get<int>());
        }
        if (!enabled_layers.empty() &&
            !qwen38_set_environment("QWEN38_PASCAL_FFN_LAYERS", enabled_layers)) {
            throw std::runtime_error("failed to enable manifest Pascal FFN layers");
        }
    }
    if (runtime.contains("speculative_ubatch_size")) {
        if (!qwen38_set_environment(
                "QWEN38_SPEC_UBATCH",
                std::to_string(runtime.at("speculative_ubatch_size").get<int>()))) {
            throw std::runtime_error("failed to set speculative microbatch size from hybrid manifest");
        }
    }

    const auto & split = runtime.at("layer_split");
    const int p100_base_layers = split.at("p100_base_layers").get<int>();
    const int rtx_base_layers = split.at("rtx_base_layers").get<int>();
    const int p100_slots = split.at("p100_schedule_slots").get<int>();
    const int rtx_slots = split.at("rtx_schedule_slots").get<int>();
    if (p100_base_layers < 1 || rtx_base_layers < 1 ||
        p100_base_layers + rtx_base_layers != 64 ||
        p100_slots != p100_base_layers || rtx_slots != rtx_base_layers + 2 ||
        p100_slots + rtx_slots != 66) {
        throw std::runtime_error(
            "v1 hybrid manifest requires a contiguous 64-layer split with all "
            "MTP/output scheduler slots on the RTX (P100 slots = P100 base layers; "
            "RTX slots = RTX base layers + 2)");
    }

    launch.arguments.reserve(static_cast<size_t>(argc) + 48);
    for (int index = 0; index < argc; ++index) {
        launch.arguments.emplace_back(argv[index]);
    }
    if (launch.arguments[model_index].rfind("--model=", 0) == 0) {
        launch.arguments[model_index] = "--model=" + model_path.string();
    } else {
        launch.arguments[model_index] = model_path.string();
    }

    qwen38_append_default(launch.arguments, {"-ngl", "--gpu-layers", "--n-gpu-layers"},
                          "--gpu-layers", "all");
    qwen38_append_default(launch.arguments, {"-sm", "--split-mode"},
                          "--split-mode", "layer");
    qwen38_append_default(launch.arguments, {"-ts", "--tensor-split"},
                          "--tensor-split", std::to_string(p100_slots) + "," + std::to_string(rtx_slots));
    qwen38_append_default(launch.arguments, {"-fit", "--fit"}, "--fit", "off");
    qwen38_append_default(launch.arguments, {"-c", "--ctx-size"}, "--ctx-size",
                          std::to_string(runtime.at("context_size").get<int>()));
    qwen38_append_default(launch.arguments, {"-b", "--batch-size"}, "--batch-size",
                          std::to_string(runtime.value("batch_size", 2048)));
    qwen38_append_default(launch.arguments, {"-ub", "--ubatch-size"}, "--ubatch-size",
                          std::to_string(runtime.value("ubatch_size", 512)));
    qwen38_append_default(launch.arguments, {"-np", "--parallel"}, "--parallel",
                          std::to_string(runtime.value("parallel", 1)));
    qwen38_append_default(launch.arguments, {"-fa", "--flash-attn"}, "--flash-attn", "on");

    if (runtime.contains("sampling")) {
        const auto & sampling = runtime.at("sampling");
        if (!sampling.is_object() ||
            sampling.value("profile", std::string()) != "qwen38-thinking-official" ||
            sampling.value("temperature", -1.0) != 1.0 ||
            sampling.value("top_k", -1) != 20 ||
            sampling.value("top_p", -1.0) != 0.95 ||
            sampling.value("min_p", -1.0) != 0.0 ||
            sampling.value("presence_penalty", -1.0) != 0.0 ||
            sampling.value("repetition_penalty", -1.0) != 1.0) {
            throw std::runtime_error(
                "hybrid sampling policy must match the official Qwen3.8 thinking profile");
        }

        qwen38_append_default(launch.arguments, {"--temp", "--temperature"},
                              "--temp", "1.0");
        qwen38_append_default(launch.arguments, {"--top-k"},
                              "--top-k", "20");
        qwen38_append_default(launch.arguments, {"--top-p"},
                              "--top-p", "0.95");
        qwen38_append_default(launch.arguments, {"--min-p"},
                              "--min-p", "0.0");
        qwen38_append_default(launch.arguments, {"--presence-penalty"},
                              "--presence-penalty", "0.0");
        qwen38_append_default(launch.arguments, {"--repeat-penalty"},
                              "--repeat-penalty", "1.0");
    }

    const auto & kv = runtime.at("kv_cache");
    if (kv.at("bits").get<int>() < 8) {
        throw std::runtime_error("hybrid manifest refuses a KV cache below eight bits");
    }
    qwen38_append_default(launch.arguments, {"-ctk", "--cache-type-k"},
                          "--cache-type-k", kv.at("type_k").get<std::string>());
    qwen38_append_default(launch.arguments, {"-ctv", "--cache-type-v"},
                          "--cache-type-v", kv.at("type_v").get<std::string>());

    if (runtime.contains("speculator")) {
        const auto & spec = runtime.at("speculator");
        const bool enabled = spec.value("enabled", true);
        if (enabled) {
            const std::string type = spec.at("type").get<std::string>();
            const bool uses_draft_context = type == "draft-mtp" || type == "draft-dspark";
            const bool diagnostic_ngram = type == "ngram-simple";
            if (!uses_draft_context && !diagnostic_ngram) {
                throw std::runtime_error(
                    "hybrid manifest only supports the measured draft-mtp and "
                    "draft-dspark speculators, plus diagnostic ngram-simple verification");
            }

            if (spec.contains("context_size")) {
                if (!uses_draft_context) {
                    throw std::runtime_error(
                        "hybrid ngram-simple diagnostics must not define a draft context_size");
                }
                const int spec_context_size = spec.at("context_size").get<int>();
                if (spec_context_size < 1024 || spec_context_size > runtime.at("context_size").get<int>()) {
                    throw std::runtime_error(
                        "hybrid speculator context_size must be between 1024 and the target context size");
                }
                if (!qwen38_set_environment("QWEN38_SPEC_CTX", std::to_string(spec_context_size))) {
                    throw std::runtime_error("failed to set speculative context size from hybrid manifest");
                }
            }

            qwen38_append_default(launch.arguments, {"--spec-type"},
                                  "--spec-type", type);
            if (spec.value("perfect_repeat", false)) {
                if (!diagnostic_ngram) {
                    throw std::runtime_error(
                        "hybrid perfect_repeat is only valid for diagnostic ngram-simple verification");
                }
                if (!qwen38_set_environment("QWEN38_NGRAM_PERFECT_REPEAT", "1")) {
                    throw std::runtime_error("failed to enable the diagnostic perfect-repeat oracle");
                }
            }
            qwen38_append_default(launch.arguments,
                                  {"--spec-draft-n-max", "--draft", "--draft-n", "--draft-max"},
                                  "--spec-draft-n-max", std::to_string(spec.at("n_max").get<int>()));
            if (diagnostic_ngram) {
                // ngram-simple has its own proposal-length parameter.  Keep it
                // equal to the target verifier width so output buffers and the
                // recurrent rollback ring are sized for the same experiment.
                qwen38_append_default(launch.arguments,
                                      {"--spec-ngram-simple-size-m"},
                                      "--spec-ngram-simple-size-m",
                                      std::to_string(spec.at("n_max").get<int>()));
            }
            qwen38_append_default(launch.arguments,
                                  {"--spec-draft-p-min", "--draft-p-min"},
                                  "--spec-draft-p-min", std::to_string(spec.value("p_min", 0.0)));
            if (uses_draft_context) {
                qwen38_append_default(launch.arguments,
                                      {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"},
                                      "--spec-draft-type-k", spec.value("cache_type_k", "q8_kv256"));
                qwen38_append_default(launch.arguments,
                                      {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"},
                                      "--spec-draft-type-v", spec.value("cache_type_v", "q8_kv256"));
                qwen38_append_default(launch.arguments,
                                      {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"},
                                      "--spec-draft-ngl", spec.value("gpu_layers", "all"));
                if (spec.contains("model")) {
                    std::filesystem::path draft_path(spec.at("model").get<std::string>());
                    if (draft_path.is_relative()) {
                        draft_path = manifest_path.parent_path() / draft_path;
                    }
                    draft_path = std::filesystem::weakly_canonical(draft_path);
                    if (!std::filesystem::is_regular_file(draft_path)) {
                        throw std::runtime_error(
                            "hybrid speculator model does not exist: " + draft_path.string());
                    }
                    qwen38_append_default(launch.arguments,
                                          {"--spec-draft-model", "-md", "--model-draft"},
                                          "--spec-draft-model", draft_path.string());
                }

                auto append_tensor_overrides = [&](const char * key,
                                                   std::initializer_list<const char *> names,
                                                   const char * option) {
                    if (!spec.contains(key) || qwen38_has_option(launch.arguments, names)) {
                        return;
                    }
                    const auto overrides = spec.at(key).get<std::vector<std::string>>();
                    if (overrides.empty()) {
                        return;
                    }
                    std::string joined;
                    for (const std::string & value : overrides) {
                        if (value.empty()) {
                            throw std::runtime_error(
                                std::string("empty tensor override in hybrid speculator field '") + key + "'");
                        }
                        if (!joined.empty()) {
                            joined += ',';
                        }
                        joined += value;
                    }
                    launch.arguments.emplace_back(option);
                    launch.arguments.push_back(std::move(joined));
                };

                // Embedded MTP tensors are part of the target GGUF and therefore use
                // the target override list.  Separate DSpark packs use draft overrides.
                append_tensor_overrides("target_tensor_overrides",
                                        {"-ot", "--override-tensor"}, "--override-tensor");
                append_tensor_overrides("draft_tensor_overrides",
                                        {"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"},
                                        "--spec-draft-override-tensor");
            }
        }
    }

    if (runtime.contains("vision")) {
        const auto & vision = runtime.at("vision");
        if (vision.value("enabled", true)) {
            std::filesystem::path mmproj_path(vision.at("model").get<std::string>());
            if (mmproj_path.is_relative()) {
                mmproj_path = manifest_path.parent_path() / mmproj_path;
            }
            mmproj_path = std::filesystem::weakly_canonical(mmproj_path);
            if (!std::filesystem::is_regular_file(mmproj_path)) {
                throw std::runtime_error(
                    "hybrid vision projector does not exist: " + mmproj_path.string());
            }

            qwen38_append_default(launch.arguments, {"-mm", "--mmproj"},
                                  "--mmproj", mmproj_path.string());
            qwen38_append_default(launch.arguments, {"--mmproj-device"},
                                  "--mmproj-device", vision.value("device", "CUDA0"));
            if (vision.contains("image_min_tokens")) {
                qwen38_append_default(launch.arguments, {"--image-min-tokens"},
                                      "--image-min-tokens",
                                      std::to_string(vision.at("image_min_tokens").get<int>()));
            }
            if (vision.contains("image_max_tokens")) {
                qwen38_append_default(launch.arguments, {"--image-max-tokens"},
                                      "--image-max-tokens",
                                      std::to_string(vision.at("image_max_tokens").get<int>()));
            }
        }
    }

    if (model.contains("alias")) {
        qwen38_append_default(launch.arguments, {"-a", "--alias"},
                              "--alias", model.at("alias").get<std::string>());
    }

    launch.argv.reserve(launch.arguments.size() + 1);
    for (std::string & argument : launch.arguments) {
        launch.argv.push_back(argument.data());
    }
    launch.argv.push_back(nullptr);
    return launch;
}

} // namespace

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::optional<qwen38_hybrid_launch> hybrid_launch;
    try {
        hybrid_launch = qwen38_prepare_hybrid_launch(argc, argv);
    } catch (const std::exception & exception) {
        fprintf(stderr, "error: %s\n", exception.what());
        return 1;
    }
    if (hybrid_launch) {
        argc = static_cast<int>(hybrid_launch->arguments.size());
        argv = hybrid_launch->argv.data();
    }

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    if (hybrid_launch) {
        for (const auto & pack : hybrid_launch->pascal_ffn_packs) {
            std::array<char, 512> error{};
            if (!ggml_backend_cuda_qwen38_pascal_ffn_register(
                    0, pack.layer, pack.gate_up.c_str(), pack.down.c_str(),
                    error.data(), error.size())) {
                fprintf(stderr, "error: failed to register Pascal FFN layer %d: %s\n",
                        pack.layer, error.data());
                qwen38_backend_free();
                return 1;
            }
            qwen38_registered_pascal_ffn_layers.push_back(pack.layer);
        }
    }
    llama_numa_init(params.numa);

    return llama_server(params, argc, argv);
}

int llama_server(common_params & params, int argc, char ** argv) {
    bool is_run_by_cli = (argv == nullptr);

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty()
                               && params.model.docker_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_child child; // only used in non-router mode
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS is set to allow all origins ('*') and no API key is set\n");
        SRV_WRN("%s", "this can be a security risk (cross-origin attacks)\n");
        SRV_WRN("%s", "more info: https://github.com/ggml-org/llama.cpp/pull/25655\n");
        SRV_WRN("%s", "-----------------\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (warn_names.size() > 0) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "the following feature(s) are enabled:\n");
        for (const auto & name : warn_names) {
            SRV_WRN("    %s\n", name.c_str());
        }
        SRV_WRN("%s", "do not expose the server to untrusted environments\n");
        SRV_WRN("%s", "-----------------\n");
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            qwen38_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

        try {
            models_routes->models.load_startup_models();
        } catch (const std::exception & e) {
            SRV_ERR("failed to load models on startup: %s\n", e.what());
            ctx_http.stop();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            clean_up();
            return 1;
        }

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            qwen38_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    SRV_INF("listening on %s\n", ctx_http.listening_address.c_str());

    // TODO: remove this in the future
    // check the string to also handle the .sock case
    if (string_ends_with(ctx_http.listening_address, ":8080")) {
        SRV_WRN("%s", "NOTICE: server default port will be changed to :9931 in a future release\n");
        SRV_WRN("%s", "        ref: https://github.com/ggml-org/llama.cpp/pull/26508\n");
    }

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
