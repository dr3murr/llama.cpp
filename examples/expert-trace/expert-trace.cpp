// Router trace: counts how often each routed expert of each MoE layer is selected over a text file.
// Usage: llama-expert-trace -m model.gguf -f input.txt -c 4096 [-ngl ...] ; EXPERT_TRACE_OUT=counts.csv
// Output CSV: layer,expert,count (rows for every layer/expert pair that was selected at least once).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct trace_state {
    std::map<std::pair<int, int>, int64_t> counts; // (layer, expert) -> selections
    std::vector<int32_t> buf;
    int64_t n_rows = 0;                            // token-layer rows seen
};

static bool trace_cb(ggml_tensor * t, bool ask, void * user_data) {
    static const char * prefix = "ffn_moe_topk-";
    if (strncmp(t->name, prefix, strlen(prefix)) != 0) {
        return !ask; // only request the router outputs, let everything else run unobserved
    }
    if (ask) {
        return true;
    }
    auto * st = (trace_state *) user_data;
    // the top-k ids are a view of the argsort result: rows of ne[0] contiguous ids, strided by nb[1]
    GGML_ASSERT(t->type == GGML_TYPE_I32 && t->nb[0] == sizeof(int32_t) && t->ne[2] == 1 && t->ne[3] == 1);
    const int layer = atoi(t->name + strlen(prefix));
    const int64_t n_used = t->ne[0];
    const int64_t n_tok  = t->ne[1];
    st->buf.resize(n_used);
    for (int64_t i = 0; i < n_tok; ++i) {
        ggml_backend_tensor_get(t, st->buf.data(), i*t->nb[1], n_used*sizeof(int32_t));
        for (int64_t j = 0; j < n_used; ++j) {
            st->counts[{layer, st->buf[j]}]++;
        }
    }
    st->n_rows += n_tok;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * out_path = getenv("EXPERT_TRACE_OUT");
    if (!out_path) {
        LOG_ERR("set EXPERT_TRACE_OUT to the output CSV path\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    trace_state st;
    params.cb_eval = trace_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    if (llama_init->model() == nullptr || ctx == nullptr) {
        LOG_ERR("failed to init\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_init->model());
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    const int n_ctx   = (int) llama_n_ctx(ctx);
    const int n_batch = (int) llama_n_batch(ctx);
    LOG_INF("tokens: %zu, window %d, batch %d\n", tokens.size(), n_ctx, n_batch);

    for (size_t w = 0; w < tokens.size(); w += n_ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const size_t w_end = std::min(tokens.size(), w + n_ctx);
        for (size_t b = w; b < w_end; b += n_batch) {
            std::vector<llama_token> chunk(tokens.begin() + b, tokens.begin() + std::min(w_end, b + n_batch));
            common_batch batch = common_batch_get_one(ctx, chunk);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
                LOG_ERR("decode failed at token %zu\n", b);
                return 1;
            }
        }
        LOG_INF("window %zu/%zu done, rows %lld\n", w_end, tokens.size(), (long long) st.n_rows);
    }

    FILE * f = fopen(out_path, "w");
    fprintf(f, "layer,expert,count\n");
    for (const auto & [k, v] : st.counts) {
        fprintf(f, "%d,%d,%lld\n", k.first, k.second, (long long) v);
    }
    fclose(f);
    LOG_INF("wrote %zu rows to %s (tokens %zu)\n", st.counts.size(), out_path, tokens.size());
    llama_backend_free();
    return 0;
}
