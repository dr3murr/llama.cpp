#include "llama-expert-shadow.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {

std::mutex                                      g_mutex;
std::unordered_map<const ggml_tensor *, ggml_tensor *> g_shadow_of;
int64_t                                         g_min_tokens = 64;

} // namespace

llama_expert_shadows::llama_expert_shadows() = default;

llama_expert_shadows::~llama_expert_shadows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const ggml_tensor * t : originals) {
        g_shadow_of.erase(t);
    }
}

size_t llama_expert_shadows::build(llama_model_loader & ml, const std::vector<ggml_tensor *> & cand) {
    if (cand.empty() || !llama_mmap::SUPPORTED) {
        return 0;
    }

    if (const char * e = getenv("LLAMA_EXPERT_SHADOW_MIN_TOKENS")) {
        g_min_tokens = std::max<int64_t>(32, atoll(e)); // below the offload batch the op would run on the CPU
    }

    struct item {
        ggml_tensor * t;
        const llama_model_loader::llama_tensor_weight * w;
    };
    std::vector<item> items;
    std::vector<std::pair<size_t, size_t>> range(ml.files.size(), { SIZE_MAX, 0 });
    for (ggml_tensor * t : cand) {
        const auto * w = ml.get_weight(ggml_get_name(t));
        if (w == nullptr) {
            continue;
        }
        items.push_back({ t, w });
        range[w->idx].first  = std::min(range[w->idx].first, w->offs);
        range[w->idx].second = std::max(range[w->idx].second, w->offs + ggml_nbytes(t));
    }
    if (items.empty()) {
        return 0;
    }

    ggml_init_params params = { items.size() * ggml_tensor_overhead(), nullptr, true };
    ctx.reset(ggml_init(params));

    mmaps.resize(ml.files.size());
    std::vector<ggml_backend_buffer_t> buf_of(ml.files.size(), nullptr);
    for (size_t idx = 0; idx < ml.files.size(); idx++) {
        if (range[idx].second == 0) {
            continue;
        }
        mmaps[idx] = std::make_unique<llama_mmap>(ml.files.at(idx).get(), /*prefetch*/ 0);
        char * base = (char *) mmaps[idx]->addr() + range[idx].first;
        ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(base, range[idx].second - range[idx].first);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);
        buf_of[idx] = buf;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    for (const item & it : items) {
        ggml_tensor * s = ggml_dup_tensor(ctx.get(), it.t);
        ggml_format_name(s, "%s.host", ggml_get_name(it.t));
        ggml_backend_tensor_alloc(buf_of[it.w->idx], s, (char *) mmaps[it.w->idx]->addr() + it.w->offs);
        g_shadow_of[it.t] = s;
        originals.push_back(it.t);
    }
    LLAMA_LOG_INFO("%s: %zu expert tensors on secondary GPUs have host shadows for graphs of >= %lld tokens\n",
            __func__, originals.size(), (long long) g_min_tokens);
    return originals.size();
}

ggml_tensor * llama_expert_shadow_select(ggml_tensor * t, int64_t n_tokens) {
    if (t == nullptr || n_tokens < g_min_tokens) {
        return t;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_shadow_of.find(t);
    return it == g_shadow_of.end() ? t : it->second;
}
