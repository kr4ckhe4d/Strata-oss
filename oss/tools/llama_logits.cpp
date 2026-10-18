// oss/tools/llama_logits.cpp - the reference: llama.cpp's logits for every position of a token list.
//   llama_logits MODEL.gguf IDS.txt OUT.bin [n_gpu_layers]
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "llama_logits MODEL IDS OUT [ngl]\n"); return 2; }
    std::vector<llama_token> ids;
    { std::ifstream f(argv[2]); long x; while (f >> x) ids.push_back((llama_token) x); }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 4 ? std::atoi(argv[4]) : 99;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = ids.size() + 16;
    cp.n_batch = ids.size();
    cp.n_ubatch = 512;
    llama_context* ctx = llama_init_from_model(model, cp);
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int step = argc > 5 ? std::atoi(argv[5]) : (int) ids.size();
    llama_batch b = llama_batch_init(ids.size(), 0, 1);
    FILE* f = std::fopen(argv[3], "wb");
    for (size_t i0 = 0; i0 < ids.size(); i0 += step) {
        const size_t n = std::min(ids.size() - i0, (size_t) step);
        for (size_t i = 0; i < n; ++i) {
            b.token[i] = ids[i0 + i]; b.pos[i] = i0 + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) { std::fprintf(stderr, "decode failed\n"); return 1; }
        for (size_t i = 0; i < n; ++i) std::fwrite(llama_get_logits_ith(ctx, i), sizeof(float), V, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "wrote %zu x %d logits\n", ids.size(), V);
    return 0;
}
