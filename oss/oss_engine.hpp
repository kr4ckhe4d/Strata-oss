// oss/oss_engine.hpp - the GPT-OSS engine: Strata's tiering (dense weights + hottest experts in VRAM, the rest in
// RAM computed by the CPU in place, the prompt's missing experts streamed over PCIe, the cache adapting to the
// routing) for the gpt-oss architecture (attention sinks, sliding-window layers, YaRN RoPE, MXFP4 experts with biases).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oss {

struct Options {
    std::string model;               // the GGUF (any shard)
    std::vector<int> gpus;           // HIP device ids, in layer order (empty: all)
    std::vector<int> split;          // layers per GPU (empty: by free VRAM)
    int ctx = 32768;                 // max context
    int swa_slack = 8192;            // a sliding-window layer keeps window + this many positions (rollback room)
    int chunk = 2048;                // prompt chunk
    int threads = 0;                 // CPU workers (0: physical cores)
    double reserve_mib = 768;        // VRAM left free per GPU
    int64_t max_slots = -1;          // cap on expert slots in total (tests)
    std::string profile_in, profile_out;
    int adapt_every = 16;            // decode steps between adaptive swap rounds (0: off)
    int adapt_swaps = 6;             // swaps per round at most
    bool verbose = false;
};

struct Stats {
    uint64_t routed = 0, gpu_hits = 0;       // decode expert routings
    uint64_t prompt_routed = 0, prompt_hits = 0;
    uint64_t swaps = 0;
    double t_prompt = 0, t_decode = 0;
    uint64_t n_prompt = 0, n_decode = 0;
};

class Engine {
public:
    explicit Engine(const Options& o);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    int n_vocab() const;
    int n_ctx() const;
    int bos() const;
    int eos() const;
    const std::vector<std::string>& vocab() const;

    /// Positions [0, n) of the cache hold `tokens[0..n)`; returns how many of `tokens` can be kept (the rest must be
    /// evaluated).  Rolls the cache back to that point.
    int reuse_prefix(const std::vector<int32_t>& tokens);
    /// Evaluates tokens at positions [n_past, n_past + tokens.size()) and returns the logits of the last one
    /// (all of them when all_logits: [tokens.size()][n_vocab]).
    const float* eval(const std::vector<int32_t>& tokens, bool all_logits = false);
    int n_past() const;
    const std::vector<int32_t>& history() const;

    Stats& stats();
    void report(FILE* f) const;
    void save_profile();
    /// Lets the CPU workers sleep (between requests).
    void set_busy(bool busy);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace oss
