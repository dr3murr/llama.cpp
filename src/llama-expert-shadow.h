#pragma once

#include "ggml.h"
#include "ggml-cpp.h"

#include <memory>
#include <vector>

struct llama_model_loader;
struct llama_mmap;

// Host aliases of the MoE expert tensors that live on a secondary GPU (LLAMA_EXPERT_SHADOW=1).
// A shadow points into a read-only mapping of the model file, so it costs page cache, not a copy.
// Graphs with at least LLAMA_EXPERT_SHADOW_MIN_TOKENS rows (default 64, never below 32) use the shadow:
// the scheduler then offloads those ops to the main GPU, as it does for CPU-resident experts, instead of
// running them on the slower secondary GPU. Smaller graphs (decode, draft verification) keep the GPU copy.
struct llama_expert_shadows {
    std::vector<std::unique_ptr<llama_mmap>> mmaps; // declared first: destroyed after the buffers
    ggml_context_ptr                         ctx;
    std::vector<ggml_backend_buffer_ptr>     bufs;
    std::vector<ggml_tensor *>               originals;

    llama_expert_shadows();
    ~llama_expert_shadows();

    // creates shadows for `cand` (loaded weights of `ml`); returns the number created
    size_t build(llama_model_loader & ml, const std::vector<ggml_tensor *> & cand);
};

// the tensor a MoE graph with n_tokens rows should read for expert weight `t`
ggml_tensor * llama_expert_shadow_select(ggml_tensor * t, int64_t n_tokens);
