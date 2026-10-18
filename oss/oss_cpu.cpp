// oss/oss_cpu.cpp - see oss_cpu.hpp.  The AVX2 dot is ggml's ggml_vec_dot_mxfp4_q8_0 (MIT, ggml-cpu/arch/x86/quants.c)
// on the planar layout; with AVX-512 VNNI the byte products go through vpdpbusd instead of maddubs + madd.
#include "oss_cpu.hpp"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace oss::cpu {

namespace {

const int8_t kValues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float e8m0_half(uint8_t e) {
    const uint32_t bits = e ? ((uint32_t) e << 23) : 0x00400000u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f * 0.5f;
}

#if defined(__AVX2__)
inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

inline __m256i expand(const uint8_t* qs, __m128i values, __m128i m4b) {
    const __m128i q4 = _mm_loadu_si128((const __m128i*) qs);
    const __m128i lo = _mm_shuffle_epi8(values, _mm_and_si128(q4, m4b));
    const __m128i hi = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q4, 4), m4b));
    return _mm256_set_m128i(hi, lo);
}

inline __m256i dot_i8(__m256i w, __m256i x) {
    const __m256i aw = _mm256_sign_epi8(w, w);
    const __m256i sx = _mm256_sign_epi8(x, w);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sx);
#else
    return _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1));
#endif
}
#endif

} // namespace

float mxfp4_row_dot(const uint8_t* qs, const uint8_t* e, const int8_t* xq, const float* xd, int nb) {
#if defined(__AVX2__)
    const __m128i values = _mm_loadu_si128((const __m128i*) kValues);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const __m256i p0 = dot_i8(expand(qs + b * 16, values, m4b), _mm256_loadu_si256((const __m256i*) (xq + b * 32)));
        const __m256i p1 =
            dot_i8(expand(qs + b * 16 + 16, values, m4b), _mm256_loadu_si256((const __m256i*) (xq + b * 32 + 32)));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(e8m0_half(e[b]) * xd[b]), _mm256_cvtepi32_ps(p0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(e8m0_half(e[b + 1]) * xd[b + 1]), _mm256_cvtepi32_ps(p1), acc1);
    }
    float sum = hsum8(_mm256_add_ps(acc0, acc1));
#else
    float sum = 0.f;
    int b = 0;
#endif
    for (; b < nb; ++b) {
        int s = 0;
        for (int j = 0; j < 16; ++j) {
            s += xq[b * 32 + j] * kValues[qs[b * 16 + j] & 0xf];
            s += xq[b * 32 + 16 + j] * kValues[qs[b * 16 + j] >> 4];
        }
        sum += e8m0_half(e[b]) * xd[b] * (float) s;
    }
    return sum;
}

void mxfp4_row_dot2(const uint8_t* qs0, const uint8_t* e0, const uint8_t* qs1, const uint8_t* e1, const int8_t* xq,
                    const float* xd, int nb, float* out0, float* out1) {
#if defined(__AVX2__)
    const __m128i values = _mm_loadu_si128((const __m128i*) kValues);
    const __m128i m4b = _mm_set1_epi8(0x0f);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m256i x = _mm256_loadu_si256((const __m256i*) (xq + b * 32));
        const __m256i p0 = dot_i8(expand(qs0 + b * 16, values, m4b), x);
        const __m256i p1 = dot_i8(expand(qs1 + b * 16, values, m4b), x);
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(e8m0_half(e0[b]) * xd[b]), _mm256_cvtepi32_ps(p0), a0);
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(e8m0_half(e1[b]) * xd[b]), _mm256_cvtepi32_ps(p1), a1);
    }
    *out0 = hsum8(a0);
    *out1 = hsum8(a1);
#else
    *out0 = mxfp4_row_dot(qs0, e0, xq, xd, nb);
    *out1 = mxfp4_row_dot(qs1, e1, xq, xd, nb);
#endif
}

void quant_q8(const float* x, int8_t* q, float* d, int n) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float dd = amax / 127.f;
        const float id = dd ? 1.f / dd : 0.f;
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::round(x[b * 32 + j] * id);
        d[b] = dd;
    }
}

float swiglu_oai(float g, float u) {
    constexpr float alpha = 1.702f, limit = 7.0f;
    const float x = std::fmin(g, limit);
    const float y = std::fmin(std::fmax(u, -limit), limit);
    return x / (1.f + std::exp(alpha * (-x))) * (y + 1.f);
}

// ------------------------------------------------------------------------------------------------ pool
Pool::Pool(int n_workers) : n_(n_workers < 1 ? 1 : n_workers) {
    for (int i = 1; i < n_; ++i) threads_.emplace_back([this, i] { worker(i); });
}

Pool::~Pool() {
    stop_.store(true);
    {
        std::lock_guard<std::mutex> lk(mu_);
        epoch_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void Pool::set_hot(bool hot) {
    hot_.store(hot);
    if (hot) {
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    }
}

void Pool::worker(int id) {
    uint64_t seen = 0;
    for (;;) {
        // spin while hot (or for a while after the last job), then sleep
        int spins = 0;
        uint64_t e;
        while ((e = epoch_.load(std::memory_order_acquire)) == seen) {
            if (stop_.load()) return;
            if (++spins > 200000 && !hot_.load(std::memory_order_relaxed)) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                    return epoch_.load() != seen || stop_.load() || hot_.load();
                });
                spins = 0;
            } else {
#if defined(__x86_64__)
                _mm_pause();
#endif
            }
        }
        if (stop_.load()) return;
        seen = e;
        (*fn_)(id, n_);
        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

void Pool::run(const std::function<void(int, int)>& fn) {
    fn_ = &fn;
    done_.store(0, std::memory_order_relaxed);
    epoch_.fetch_add(1, std::memory_order_acq_rel);
    if (!hot_.load()) cv_.notify_all();
    fn(0, n_);
    while (done_.load(std::memory_order_acquire) != n_ - 1) {
#if defined(__x86_64__)
        _mm_pause();
#endif
    }
}

void Pool::barrier() {
    const uint32_t gen = bar_gen_.load(std::memory_order_acquire);
    if (bar_count_.fetch_add(1, std::memory_order_acq_rel) == n_ - 1) {
        bar_count_.store(0, std::memory_order_relaxed);
        bar_gen_.fetch_add(1, std::memory_order_acq_rel);
    } else {
        while (bar_gen_.load(std::memory_order_acquire) == gen) {
#if defined(__x86_64__)
            _mm_pause();
#endif
        }
    }
}

// ------------------------------------------------------------------------------------------------ misses
void MissScratch::init(int n_embd_, int n_ff_, int max_jobs_) {
    n_embd = n_embd_;
    n_ff = n_ff_;
    max_jobs = max_jobs_;
    aq.assign((size_t) max_jobs * n_ff, 0);
    ad.assign((size_t) max_jobs * (n_ff / 32), 0.f);
    z.assign((size_t) max_jobs * n_embd, 0.f);
}

void compute_misses(Pool& pool, const std::vector<MissJob>& jobs, const int8_t* hq, size_t hq_stride, const float* hd,
                    size_t hd_stride, float* y, size_t y_stride, MissScratch& s) {
    const int nj = (int) jobs.size();
    if (nj == 0) return;
    const int H = s.n_embd, F = s.n_ff;
    const int nbh = H / 32, nbf = F / 32;
    const size_t gate_bytes = (size_t) F * nbh * 17;
    const size_t down_off = 2 * gate_bytes;
    // work units of 32 rows: phase 1 the gate/up rows of every job, phase 2 the down rows
    const int u1 = nj * (F / 32), u2 = nj * (H / 32);
    std::atomic<int> next1{0}, next2{0};
    auto fn = [&](int, int) {
        for (;;) {
            const int u = next1.fetch_add(1, std::memory_order_relaxed);
            if (u >= u1) break;
            const int j = u / (F / 32), rb = u % (F / 32);
            const MissJob& job = jobs[j];
            const int8_t* xq = hq + (size_t) job.col * hq_stride;
            const float* xd = hd + (size_t) job.col * hd_stride;
            const uint8_t* g = job.blob;
            const uint8_t* up = job.blob + gate_bytes;
            float act[32];
            for (int i = 0; i < 32; ++i) {
                const int r = rb * 32 + i;
                float gv, uv;
                mxfp4_row_dot2(g + (size_t) r * nbh * 16, g + (size_t) F * nbh * 16 + (size_t) r * nbh,
                               up + (size_t) r * nbh * 16, up + (size_t) F * nbh * 16 + (size_t) r * nbh, xq, xd, nbh,
                               &gv, &uv);
                act[i] = swiglu_oai(gv + job.gate_b[r], uv + job.up_b[r]);
            }
            quant_q8(act, s.aq.data() + (size_t) j * F + rb * 32, s.ad.data() + (size_t) j * nbf + rb, 32);
        }
        pool.barrier();
        for (;;) {
            const int u = next2.fetch_add(1, std::memory_order_relaxed);
            if (u >= u2) break;
            const int j = u / (H / 32), rb = u % (H / 32);
            const MissJob& job = jobs[j];
            const uint8_t* d = job.blob + down_off;
            const int8_t* aq = s.aq.data() + (size_t) j * F;
            const float* ad = s.ad.data() + (size_t) j * nbf;
            float* z = s.z.data() + (size_t) j * H;
            for (int i = 0; i < 32; i += 2) {
                const int r = rb * 32 + i;
                float v0, v1;
                mxfp4_row_dot2(d + (size_t) r * nbf * 16, d + (size_t) H * nbf * 16 + (size_t) r * nbf,
                               d + (size_t) (r + 1) * nbf * 16, d + (size_t) H * nbf * 16 + (size_t) (r + 1) * nbf, aq,
                               ad, nbf, &v0, &v1);
                z[r] = job.w * (v0 + job.down_b[r]);
                z[r + 1] = job.w * (v1 + job.down_b[r + 1]);
            }
        }
    };
    pool.run(fn);
    // the sum per column, in job order (deterministic)
    std::vector<char> seen(64, 0);
    for (int j = 0; j < nj; ++j) {
        float* yc = y + (size_t) jobs[j].col * y_stride;
        const float* z = s.z.data() + (size_t) j * H;
        if (!seen[jobs[j].col]) {
            seen[jobs[j].col] = 1;
            std::memcpy(yc, z, sizeof(float) * H);
        } else {
            for (int i = 0; i < H; ++i) yc[i] += z[i];
        }
    }
}

} // namespace oss::cpu
