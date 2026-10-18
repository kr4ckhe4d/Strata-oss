// oss/oss_cpu.hpp - the CPU half: the experts a token routes to that are not in VRAM, computed in RAM.
//
// As in Strata's expert pool (src/kernels/cpu/pool.cpp): the work is DRAM-bound, so the pool's job is to keep every
// physical core reading expert bytes and to cost less than one expert per dispatch.  Workers spin on an epoch
// counter while a request is running and sleep when idle; each dispatch is a barrier (the GPU needs the sum before
// the layer's residual add).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace oss::cpu {

/// One MXFP4 matrix row (planar: 16 code bytes per block, then the block's E8M0) against int8 activations with a
/// float scale per 32 values.  Rounds like the GPU kernel: per block e8m0/2 * d * integer sum, in float.
float mxfp4_row_dot(const uint8_t* qs, const uint8_t* e, const int8_t* xq, const float* xd, int nb);
/// Two rows at once (same activation).
void mxfp4_row_dot2(const uint8_t* qs0, const uint8_t* e0, const uint8_t* qs1, const uint8_t* e1, const int8_t* xq,
                    const float* xd, int nb, float* out0, float* out1);

/// int8 rounding of 32-value blocks, as the GPU's k_quant_q8 (d = amax / 127, round half away from zero)
void quant_q8(const float* x, int8_t* q, float* d, int n);

float swiglu_oai(float g, float u);

class Pool {
public:
    explicit Pool(int n_workers);   // n_workers includes the calling thread
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int size() const { return n_; }
    /// Runs fn(worker, n_workers) on every worker (the caller is worker 0) and returns when all are done.
    void run(const std::function<void(int, int)>& fn);
    /// A barrier usable inside run()'s function.
    void barrier();
    /// Keeps the workers spinning (a request is running) or lets them sleep after a short idle period.
    void set_hot(bool hot);

private:
    void worker(int id);

    int n_;
    std::vector<std::thread> threads_;
    const std::function<void(int, int)>* fn_ = nullptr;
    alignas(64) std::atomic<uint64_t> epoch_{0};
    alignas(64) std::atomic<int> done_{0};
    alignas(64) std::atomic<int> bar_count_{0};
    alignas(64) std::atomic<uint32_t> bar_gen_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::atomic<bool> hot_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// One routed expert the CPU computes for one column.
struct MissJob {
    const uint8_t* blob = nullptr;  ///< the expert's planar blob (gate, up, down)
    const float* gate_b = nullptr;  ///< n_ff
    const float* up_b = nullptr;    ///< n_ff
    const float* down_b = nullptr;  ///< n_embd
    int col = 0;
    float w = 0.f;                  ///< router weight
};

/// Scratch for compute_misses, sized for `max_jobs` jobs.
struct MissScratch {
    int n_embd = 0, n_ff = 0, max_jobs = 0;
    std::vector<int8_t> aq;   // [max_jobs][n_ff]
    std::vector<float> ad;    // [max_jobs][n_ff / 32]
    std::vector<float> z;     // [max_jobs][n_embd]
    std::atomic<int> next{0};
    void init(int n_embd_, int n_ff_, int max_jobs_);
};

/// y[c] (n_embd floats per column, zeroed here for every column that has a job) += sum of the column's jobs.
/// hq/hd: the columns' quantized FFN inputs (n_embd int8 + n_embd / 32 scales each, `hq_stride` / `hd_stride` apart).
void compute_misses(Pool& pool, const std::vector<MissJob>& jobs, const int8_t* hq, size_t hq_stride, const float* hd,
                    size_t hd_stride, float* y, size_t y_stride, MissScratch& s);

} // namespace oss::cpu
