// oss/oss_engine.cpp - see oss_engine.hpp.
#include "oss_engine.hpp"

#include "oss_cpu.hpp"
#include "oss_kernels.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#define HIPCHECK(x)                                                                                    \
    do {                                                                                               \
        hipError_t e_ = (x);                                                                           \
        if (e_ != hipSuccess) {                                                                        \
            std::fprintf(stderr, "strata-oss: %s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_),     \
                         __FILE__, __LINE__);                                                          \
            std::abort();                                                                              \
        }                                                                                              \
    } while (0)
#define BLASCHECK(x)                                                                                   \
    do {                                                                                               \
        hipblasStatus_t e_ = (x);                                                                      \
        if (e_ != HIPBLAS_STATUS_SUCCESS) {                                                            \
            std::fprintf(stderr, "strata-oss: %s failed: %d (%s:%d)\n", #x, (int) e_, __FILE__,        \
                         __LINE__);                                                                    \
            std::abort();                                                                              \
        }                                                                                              \
    } while (0)

namespace oss {

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

inline uint32_t load_acquire(const volatile uint32_t* p) { return __atomic_load_n((const uint32_t*) p, __ATOMIC_ACQUIRE); }
inline void store_release(volatile uint32_t* p, uint32_t v) { __atomic_store_n((uint32_t*) p, v, __ATOMIC_RELEASE); }

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; --e; }
            bits = s | (e << 23) | ((m & 0x3ff) << 13);
        }
    } else if (e == 31) bits = s | 0x7f800000 | (m << 13);
    else bits = s | ((e + 127 - 15) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

template <class T>
T* dalloc(size_t n) {
    void* p = nullptr;
    if (n == 0) n = 1;
    HIPCHECK(hipMalloc(&p, n * sizeof(T)));
    return (T*) p;
}

template <class T>
T* dupload(const T* src, size_t n) {
    T* d = dalloc<T>(n);
    HIPCHECK(hipMemcpy(d, src, n * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

constexpr uint32_t kQ8_0 = 8, kF32 = 0, kF16 = 1, kBF16 = 30, kMXFP4 = 39;

} // namespace

struct Hp {
    int L = 0, H = 0, NH = 0, NKV = 0, HD = 0, E = 0, K = 0, F = 0, swa = 0, V = 0, ctx_train = 0;
    float eps = 1e-5f;
    RopeParams rope{};
    int bos = 0, eos = 0;
    int qkv_rows() const { return (NH + 2 * NKV) * HD; }
    bool is_swa(int l) const { return swa > 0 && (l % 2 == 0); }
};

struct DevLayer {
    int8_t* qkv_q = nullptr; uint16_t* qkv_d = nullptr; float* qkv_b = nullptr;
    int8_t* o_q = nullptr; uint16_t* o_d = nullptr; float* o_b = nullptr;
    float *attn_norm = nullptr, *post_norm = nullptr, *sinks = nullptr;
    float *router_w = nullptr, *router_b = nullptr;
    float *gate_b = nullptr, *up_b = nullptr, *down_b = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    int cap = 0, window = 0;
    size_t kv_stride = 0;  // elements between conversations' KV regions
    int32_t* slot_of = nullptr;
};

struct Dev {
    int id = 0;
    int l0 = 0, l1 = 0;
    hipStream_t s = nullptr, cs = nullptr;
    hipblasHandle_t blas = nullptr;
    std::vector<DevLayer> layers;  // [l1 - l0]
    uint8_t* arena = nullptr;
    int64_t n_slots = 0;
    std::vector<std::pair<int, int>> slot_owner;  // slot -> (layer, expert)
    // decode scratch
    StepParams* params = nullptr;
    float *x = nullptr, *qkv = nullptr, *attn_part = nullptr, *attn_out = nullptr, *h = nullptr, *rlog = nullptr;
    int8_t *xq = nullptr, *aoq = nullptr, *hq = nullptr, *actq = nullptr;
    float *xd = nullptr, *aod = nullptr, *hd = nullptr, *actd = nullptr, *act = nullptr, *y = nullptr, *topw = nullptr;
    int32_t* topk = nullptr;
    // head (last device)
    float* out_norm = nullptr;
    int8_t* head_q = nullptr;
    uint16_t* head_d = nullptr;
    float* logits = nullptr;
    // prompt scratch
    float *px = nullptr, *pqkv = nullptr, *py = nullptr, *ph32 = nullptr, *plog = nullptr, *ptopw = nullptr;
    float *ys = nullptr, *pact = nullptr, *pqd = nullptr, *xsd = nullptr, *pad = nullptr;
    int8_t *pq = nullptr, *xsq = nullptr, *paq = nullptr, *aoq_p = nullptr;
    float* aod_p = nullptr;
    int32_t* tiles = nullptr;
    const uint8_t** wptr = nullptr;
    hipEvent_t staged = nullptr;
    uint16_t *xn16 = nullptr, *ao16 = nullptr, *ph16 = nullptr, *w16 = nullptr, *w2 = nullptr;
    int32_t *ptopk = nullptr, *rows = nullptr, *inv = nullptr;
    std::vector<uint8_t*> ring;
    std::vector<hipEvent_t> ring_copied, ring_used;
    hipEvent_t done = nullptr;
    hipGraphExec_t graph[kMaxCols + 1] = {};
};

struct HostLayer {
    std::vector<uint8_t*> blob;     // [E] the expert's RAM copy, or null while it is in a VRAM slot
    std::vector<int32_t> slot;      // [E] its slot on the layer's GPU, or -1
    std::vector<float> gate_b, up_b, down_b;
    std::vector<double> heat;       // [E] routings, decayed
    int dev = 0;
};

struct Eagle {
    bool on = false;
    std::unique_ptr<strata::GgufModel> g;
    int tl[3] = {};
    int R = 0, FF = 0, NS = 0, IN = 0;  // qkv rows, ffn width, draft vocab, attention input (2H)
    int8_t *qkv_q = nullptr, *o_q = nullptr, *gu_q = nullptr, *down_q = nullptr, *fc_q = nullptr, *head_q = nullptr;
    uint16_t *qkv_d = nullptr, *o_d = nullptr, *gu_d = nullptr, *down_d = nullptr, *fc_d = nullptr, *head_d = nullptr;
    float *attn_norm = nullptr, *attn_norm2 = nullptr, *ffn_norm = nullptr, *out_norm = nullptr, *enc_norm = nullptr;
    int32_t* map_d = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    int cap = 0;
    RopeParams rope{};
    float eps = 1e-5f;
    float *feat = nullptr, *pfeat = nullptr, *gring = nullptr, *gtmp = nullptr;
    int G = 0;
    StepParams* params = nullptr;
    float *xe = nullptr, *gin = nullptr, *cat = nullptr, *qkv = nullptr, *ao = nullptr, *res = nullptr, *gu = nullptr,
          *act = nullptr, *out = nullptr, *logits = nullptr;
    int8_t *catq = nullptr, *aoq = nullptr, *hq = nullptr, *actq = nullptr, *gq = nullptr;
    float *catd = nullptr, *aod = nullptr, *hd = nullptr, *actd = nullptr, *gqd = nullptr;
    int32_t* tok_d = nullptr;
    // batched catch-up (prompt rows) through the prompt kernels, kCB rows at a time
    int cb = 256;   // rows per batched catch-up piece (OSS_EAGLE_CB)
    int win = 0;    // attention window of the draft (0: all positions)
    uint16_t* pfeat16 = nullptr;  // the prompt chunk's features, f16
    float *b_g = nullptr, *b_cat = nullptr, *b_qkv = nullptr, *b_res = nullptr, *b_gu = nullptr, *b_act = nullptr,
          *b_out = nullptr, *b_catd = nullptr, *b_hd = nullptr, *b_actd = nullptr, *b_aod = nullptr;
    int8_t *b_catq = nullptr, *b_hq = nullptr, *b_actq = nullptr, *b_aoq = nullptr;
    uint16_t* b_ao16 = nullptr;
    size_t kv_stride = 0;  // per conversation: its draft KV region and its g ring (G x H floats)
    float last_p = 0.f;  // the draft model's probability of its last pick
    float p_min = 0.6f;  // drafting stops below this (OSS_DRAFT_PMIN)
    double t_draft = 0;
    uint64_t n_draft_steps = 0;
};

struct Seq {
    std::vector<int32_t> hist;
    int npast = 0;
    int n_valid = 0;  // EAGLE3: draft KV rows [0, n_valid) hold (token p + 1, g(p))
};

struct Engine::Impl {
    Options o;
    Eagle eg;
    std::vector<Seq> seqs;
    int cur = 0;
    Seq& S() { return seqs[(size_t) cur]; }
    std::unique_ptr<strata::GgufModel> gguf;
    Hp hp;
    std::vector<std::string> vocab;
    std::vector<Dev> devs;
    std::vector<HostLayer> hl;
    size_t blob_bytes = 0;
    std::vector<uint8_t*> host_chunks;      // the pinned RAM arena's allocations
    std::vector<uint8_t*> free_host_blobs;
    LayerMailbox* mb = nullptr;             // [L]
    Handoff* ho = nullptr;                  // [ndev - 1]
    StepParams* params_h = nullptr;
    float* emb_h = nullptr;                 // [kMaxCols][H]
    float* logits_h = nullptr;              // [kMaxCols][V]
    float* prompt_h = nullptr;              // [chunk][H] pinned
    int32_t* ptopk_h = nullptr;             // [chunk][K]
    int32_t* rows_h = nullptr;              // [chunk * K]
    int32_t* inv_h = nullptr;
    int32_t* tiles_h = nullptr;             // [4 * (chunk * K / 32 + E)]
    const uint8_t** wptr_h = nullptr;       // [E]
    uint8_t* swap_h = nullptr;              // one blob, pinned
    const uint8_t* embd = nullptr;          // token_embd (Q8_0, in the mapping)
    uint32_t embd_type = 0;
    std::unique_ptr<cpu::Pool> pool;
    cpu::MissScratch scratch;
    uint32_t seq = 0;
    std::vector<float> all_logits;
    Stats st;
    int steps_since_adapt = 0;

    // ---------------------------------------------------------------- loading
    const strata::TensorInfo& tensor(const std::string& name) const {
        const strata::TensorInfo* t = gguf->find(name);
        if (!t) throw std::runtime_error("strata-oss: tensor " + name + " is missing");
        return *t;
    }
    const uint8_t* tdata(const std::string& name) const {
        size_t shard = 0;
        const strata::TensorInfo* t = gguf->find(name, &shard);
        if (!t) throw std::runtime_error("strata-oss: tensor " + name + " is missing");
        return gguf->shard(shard).tensor_data(*t);
    }
    std::vector<float> f32_tensor(const std::string& name) const {
        const auto& t = tensor(name);
        const uint8_t* d = tdata(name);
        const size_t n = (size_t) t.elements();
        std::vector<float> v(n);
        if (t.type == kF32) std::memcpy(v.data(), d, n * 4);
        else if (t.type == kF16) for (size_t i = 0; i < n; ++i) v[i] = half_to_float(((const uint16_t*) d)[i]);
        else if (t.type == kBF16)
            for (size_t i = 0; i < n; ++i) {
                const uint32_t b = (uint32_t) ((const uint16_t*) d)[i] << 16;
                std::memcpy(&v[i], &b, 4);
            }
        else throw std::runtime_error("strata-oss: " + name + ": unsupported type for a float tensor");
        return v;
    }
    // Q8_0 rows -> (int8 plane, half plane)
    void q8_planes(const std::string& name, int n_in, int n_out, std::vector<int8_t>& q, std::vector<uint16_t>& d,
                   size_t row0, const strata::GgufModel* from = nullptr) const {
        const strata::GgufModel& gg = from ? *from : *gguf;
        size_t shard = 0;
        const strata::TensorInfo* tp = gg.find(name, &shard);
        if (!tp) throw std::runtime_error("strata-oss: tensor " + name + " is missing");
        const auto& t = *tp;
        if (t.type != kQ8_0) throw std::runtime_error("strata-oss: " + name + " is not Q8_0 (only Q8_0 dense weights)");
        if ((int) t.shape[0] != n_in || (int) t.shape[1] != n_out) throw std::runtime_error("strata-oss: " + name + " shape");
        const uint8_t* src = gg.shard(shard).tensor_data(t);
        const int nb = n_in / 32;
        for (int r = 0; r < n_out; ++r)
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = src + ((size_t) r * nb + b) * 34;
                std::memcpy(&d[(row0 + r) * nb + b], blk, 2);
                std::memcpy(&q[(row0 + r) * n_in + (size_t) b * 32], blk + 2, 32);
            }
    }
    int geti(const char* k, int def = -1) const {
        const strata::MetaValue* v = gguf->meta().get(std::string("gpt-oss.") + k);
        if (!v) {
            if (def >= 0) return def;
            throw std::runtime_error(std::string("strata-oss: metadata gpt-oss.") + k + " is missing");
        }
        return (int) v->num();
    }
    float getf(const char* k, float def) const {
        const strata::MetaValue* v = gguf->meta().get(std::string("gpt-oss.") + k);
        return v ? (float) v->num() : def;
    }

    void read_hparams() {
        const strata::MetaValue* arch = gguf->meta().get("general.architecture");
        if (!arch || arch->s != "gpt-oss")
            throw std::runtime_error("strata-oss: not a gpt-oss GGUF (general.architecture = " +
                                     (arch ? arch->s : std::string("?")) + ")");
        hp.L = geti("block_count");
        hp.H = geti("embedding_length");
        hp.NH = geti("attention.head_count");
        hp.NKV = geti("attention.head_count_kv");
        hp.HD = geti("attention.key_length");
        hp.E = geti("expert_count");
        hp.K = geti("expert_used_count");
        hp.F = geti("expert_feed_forward_length");
        hp.swa = geti("attention.sliding_window", 0);
        hp.ctx_train = geti("context_length", 131072);
        hp.eps = getf("attention.layer_norm_rms_epsilon", 1e-5f);
        if (hp.H != 2880 || hp.HD != 64 || hp.K != kTopK || hp.E > 256 || hp.F % 32 || hp.NH % hp.NKV ||
            hp.NH / hp.NKV != 8)
            throw std::runtime_error("strata-oss: this build supports gpt-oss-20b / 120b geometry only");
        const float base = getf("rope.freq_base", 150000.f);
        const float factor = getf("rope.scaling.factor", 1.f);
        const int n_orig = geti("rope.scaling.original_context_length", 4096);
        const float beta_fast = getf("rope.scaling.yarn_beta_fast", 32.f);
        const float beta_slow = getf("rope.scaling.yarn_beta_slow", 1.f);
        hp.rope.freq_base = base;
        hp.rope.freq_scale = 1.f / factor;
        hp.rope.ext_factor = factor > 1.f ? 1.f : 0.f;
        hp.rope.n_dims = hp.HD;
        // llama.cpp's yarn_attn_factor: get_mscale(factor) / (1 + 0.1 ln factor) = 1 (the kernel's mscale stays)
        hp.rope.attn_factor = 1.f;
        auto corr_dim = [&](float n_rot) {
            return hp.HD * std::log((float) n_orig / (n_rot * 2 * (float) M_PI)) / (2 * std::log(base));
        };
        hp.rope.corr0 = std::max(0.f, std::floor(corr_dim(beta_fast)));
        hp.rope.corr1 = std::min((float) (hp.HD - 1), std::ceil(corr_dim(beta_slow)));
        const auto& te = tensor("token_embd.weight");
        hp.V = (int) te.shape[1];
        embd_type = te.type;
        if (embd_type != kQ8_0) throw std::runtime_error("strata-oss: token_embd must be Q8_0");
        embd = tdata("token_embd.weight");
        const strata::MetaValue* b = gguf->meta().get("tokenizer.ggml.bos_token_id");
        const strata::MetaValue* e = gguf->meta().get("tokenizer.ggml.eos_token_id");
        hp.bos = b ? (int) b->num() : 0;
        hp.eos = e ? (int) e->num() : 0;
        const strata::MetaValue* toks = gguf->meta().get("tokenizer.ggml.tokens");
        if (toks)
            for (const auto& it : toks->items) vocab.push_back(it.s);
        const auto& gx = tensor("blk.0.ffn_gate_exps.weight");
        if (gx.type != kMXFP4) throw std::runtime_error("strata-oss: the experts must be MXFP4");
        blob_bytes = 3 * (size_t) hp.F * (hp.H / 32) * 17;
    }

    // the expert e of layer l, repacked into its planar blob
    void make_blob(int l, int e, uint8_t* out) const {
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        size_t off = 0;
        for (int m = 0; m < 3; ++m) {
            const std::string n = "blk." + std::to_string(l) + "." + names[m] + ".weight";
            const auto& t = tensor(n);
            const int n_in = (int) t.shape[0], rows = (int) t.shape[1];
            const int nb = n_in / 32;
            const uint8_t* src = tdata(n) + (size_t) e * rows * nb * 17;
            uint8_t* qs = out + off;
            uint8_t* es = out + off + (size_t) rows * nb * 16;
            for (size_t i = 0; i < (size_t) rows * nb; ++i) {
                es[i] = src[i * 17];
                std::memcpy(qs + i * 16, src + i * 17 + 1, 16);
            }
            off += (size_t) rows * nb * 17;
        }
    }

    void embed(const int32_t* toks, int n, float* out) const {
        const int nb = hp.H / 32;
        for (int c = 0; c < n; ++c) {
            const uint8_t* row = embd + (size_t) toks[c] * nb * 34;
            for (int b = 0; b < nb; ++b) {
                uint16_t dh;
                std::memcpy(&dh, row + b * 34, 2);
                const float d = half_to_float(dh);
                const int8_t* q = (const int8_t*) (row + b * 34 + 2);
                for (int j = 0; j < 32; ++j) out[(size_t) c * hp.H + b * 32 + j] = d * q[j];
            }
        }
    }

    void choose_devices() {
        int n = 0;
        HIPCHECK(hipGetDeviceCount(&n));
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < n; ++i) {
                hipDeviceProp_t pr;
                HIPCHECK(hipGetDeviceProperties(&pr, i));
                if (!pr.integrated) ids.push_back(i);
            }
        if (ids.empty()) throw std::runtime_error("strata-oss: no GPU");
        std::vector<double> freeb;
        for (int id : ids) {
            HIPCHECK(hipSetDevice(id));
            size_t fr = 0, tot = 0;
            HIPCHECK(hipMemGetInfo(&fr, &tot));
            hipDeviceProp_t pr;
            HIPCHECK(hipGetDeviceProperties(&pr, id));
            std::fprintf(stderr, "strata-oss: GPU %d: %s (%s), %.1f of %.1f GiB free\n", id, pr.name, pr.gcnArchName,
                         fr / 1073741824.0, tot / 1073741824.0);
            freeb.push_back((double) fr);
        }
        std::vector<int> split = o.split;
        if (split.empty()) {
            // layers in proportion to the VRAM left once the head (on the last GPU) is counted
            const double head = (double) hp.V * hp.H * 1.07;
            std::vector<double> w(freeb);
            w.back() -= head;
            const double sum = std::accumulate(w.begin(), w.end(), 0.0);
            int left = hp.L;
            for (size_t i = 0; i < ids.size(); ++i) {
                int k = i + 1 == ids.size() ? left : (int) std::lround(hp.L * w[i] / sum);
                k = std::max(1, std::min(k, left - (int) (ids.size() - 1 - i)));
                split.push_back(k);
                left -= k;
            }
        }
        if (split.size() != ids.size() || std::accumulate(split.begin(), split.end(), 0) != hp.L)
            throw std::runtime_error("strata-oss: --split must give every GPU its layers, " + std::to_string(hp.L) +
                                     " in total");
        int l = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            Dev d;
            d.id = ids[i];
            d.l0 = l;
            d.l1 = l + split[i];
            l = d.l1;
            devs.push_back(std::move(d));
        }
    }

    void alloc_device(Dev& d, bool last) {
        HIPCHECK(hipSetDevice(d.id));
        HIPCHECK(hipStreamCreateWithFlags(&d.s, hipStreamNonBlocking));
        HIPCHECK(hipStreamCreateWithFlags(&d.cs, hipStreamNonBlocking));
        HIPCHECK(hipEventCreateWithFlags(&d.done, hipEventDisableTiming));
        BLASCHECK(hipblasCreate(&d.blas));
        BLASCHECK(hipblasSetStream(d.blas, d.s));
        const int H = hp.H, R = hp.qkv_rows(), NA = hp.NH * hp.HD, E = hp.E, F = hp.F;
        for (int l = d.l0; l < d.l1; ++l) {
            DevLayer L;
            const std::string p = "blk." + std::to_string(l) + ".";
            {
                std::vector<int8_t> q((size_t) R * H);
                std::vector<uint16_t> dd((size_t) R * (H / 32));
                q8_planes(p + "attn_q.weight", H, NA, q, dd, 0);
                q8_planes(p + "attn_k.weight", H, hp.NKV * hp.HD, q, dd, NA);
                q8_planes(p + "attn_v.weight", H, hp.NKV * hp.HD, q, dd, NA + hp.NKV * hp.HD);
                L.qkv_q = dupload(q.data(), q.size());
                L.qkv_d = dupload(dd.data(), dd.size());
                std::vector<float> b;
                for (const char* n : {"attn_q.bias", "attn_k.bias", "attn_v.bias"}) {
                    auto v = f32_tensor(p + n);
                    b.insert(b.end(), v.begin(), v.end());
                }
                L.qkv_b = dupload(b.data(), b.size());
            }
            {
                std::vector<int8_t> q((size_t) H * NA);
                std::vector<uint16_t> dd((size_t) H * (NA / 32));
                q8_planes(p + "attn_output.weight", NA, H, q, dd, 0);
                L.o_q = dupload(q.data(), q.size());
                L.o_d = dupload(dd.data(), dd.size());
                auto b = f32_tensor(p + "attn_output.bias");
                L.o_b = dupload(b.data(), b.size());
            }
            auto up = [&](const std::string& n) { auto v = f32_tensor(p + n); return dupload(v.data(), v.size()); };
            L.attn_norm = up("attn_norm.weight");
            L.post_norm = up("post_attention_norm.weight");
            L.sinks = up("attn_sinks.weight");
            L.router_w = up("ffn_gate_inp.weight");
            L.router_b = up("ffn_gate_inp.bias");
            HostLayer& h = hl[l];
            h.gate_b = f32_tensor(p + "ffn_gate_exps.bias");
            h.up_b = f32_tensor(p + "ffn_up_exps.bias");
            h.down_b = f32_tensor(p + "ffn_down_exps.bias");
            L.gate_b = dupload(h.gate_b.data(), h.gate_b.size());
            L.up_b = dupload(h.up_b.data(), h.up_b.size());
            L.down_b = dupload(h.down_b.data(), h.down_b.size());
            L.window = hp.is_swa(l) ? hp.swa : 0;
            L.cap = L.window ? std::min(o.ctx, hp.swa + std::max(o.swa_slack, o.chunk)) : o.ctx;
            const size_t kvn = (size_t) L.cap * hp.NKV * hp.HD;
            L.kv_stride = kvn;
            L.kc = dalloc<uint16_t>(kvn * o.parallel);
            L.vc = dalloc<uint16_t>(kvn * o.parallel);
            L.slot_of = dalloc<int32_t>(E);
            d.layers.push_back(L);
        }
        const int MC = kMaxCols;
        d.params = dalloc<StepParams>(1);
        d.x = dalloc<float>((size_t) MC * H);
        d.xq = dalloc<int8_t>((size_t) MC * NA);
        d.xd = dalloc<float>((size_t) MC * NA / 32);
        d.qkv = dalloc<float>((size_t) MC * R);
        d.attn_part = dalloc<float>(attn_part_floats(hp.NH, o.ctx, MC));
        d.attn_out = dalloc<float>((size_t) MC * NA);
        d.aoq = dalloc<int8_t>((size_t) MC * NA);
        d.aod = dalloc<float>((size_t) MC * NA / 32);
        d.h = dalloc<float>((size_t) MC * H);
        d.hq = dalloc<int8_t>((size_t) MC * H);
        d.hd = dalloc<float>((size_t) MC * H / 32);
        d.rlog = dalloc<float>((size_t) MC * E);
        d.topk = dalloc<int32_t>((size_t) MC * kTopK);
        d.topw = dalloc<float>((size_t) MC * kTopK);
        d.act = dalloc<float>((size_t) MC * kTopK * F);
        HIPCHECK(hipMemset(d.act, 0, sizeof(float) * MC * kTopK * F));
        d.actq = dalloc<int8_t>((size_t) MC * kTopK * F);
        d.actd = dalloc<float>((size_t) MC * kTopK * F / 32);
        d.y = dalloc<float>((size_t) MC * H);
        if (last) {
            auto nw = f32_tensor("output_norm.weight");
            d.out_norm = dupload(nw.data(), nw.size());
            std::vector<int8_t> q((size_t) hp.V * H);
            std::vector<uint16_t> dd((size_t) hp.V * (H / 32));
            q8_planes("output.weight", H, hp.V, q, dd, 0);
            d.head_q = dupload(q.data(), q.size());
            d.head_d = dupload(dd.data(), dd.size());
            d.logits = dalloc<float>((size_t) MC * hp.V);
        }
        // prompt scratch
        const size_t C = o.chunk;
        d.px = dalloc<float>(C * H);
        d.xn16 = dalloc<uint16_t>(C * H);
        d.pqkv = dalloc<float>(C * R);
        d.ao16 = dalloc<uint16_t>(C * NA);
        d.py = dalloc<float>(C * H);
        d.ph32 = dalloc<float>(C * H);
        d.ph16 = dalloc<uint16_t>(C * H);
        d.plog = dalloc<float>(C * E);
        d.ptopk = dalloc<int32_t>(C * kTopK);
        d.ptopw = dalloc<float>(C * kTopK);
        d.rows = dalloc<int32_t>(C * kTopK);
        d.inv = dalloc<int32_t>(C * kTopK);
        d.ys = dalloc<float>(C * kTopK * H);
        d.pq = dalloc<int8_t>(C * H);
        d.aoq_p = dalloc<int8_t>(C * NA);
        d.aod_p = dalloc<float>(C * NA / 32);
        d.pqd = dalloc<float>(C * H / 32);
        d.xsq = dalloc<int8_t>(C * kTopK * H);
        d.xsd = dalloc<float>(C * kTopK * H / 32);
        d.pact = dalloc<float>(C * kTopK * F);
        d.paq = dalloc<int8_t>(C * kTopK * F);
        d.pad = dalloc<float>(C * kTopK * F / 32);
        d.tiles = dalloc<int32_t>(4 * (C * kTopK / 32 + E));
        d.wptr = (const uint8_t**) dalloc<void*>(E);
        HIPCHECK(hipEventCreateWithFlags(&d.staged, hipEventDisableTiming));
        d.w16 = dalloc<uint16_t>((size_t) R * H);
        d.w2 = dalloc<uint16_t>(std::max((size_t) H * F, (size_t) H * NA));
        for (int r = 0; r < 16; ++r) {
            d.ring.push_back(dalloc<uint8_t>(blob_bytes));
            hipEvent_t a, b;
            HIPCHECK(hipEventCreateWithFlags(&a, hipEventDisableTiming));
            HIPCHECK(hipEventCreateWithFlags(&b, hipEventDisableTiming));
            d.ring_copied.push_back(a);
            d.ring_used.push_back(b);
        }
    }

    bool read_profile(std::vector<double>& heat) const {
        if (o.profile_in.empty()) return false;
        FILE* f = std::fopen(o.profile_in.c_str(), "rb");
        if (!f) return false;
        char magic[4];
        int32_t L = 0, E = 0;
        bool ok = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "OSSP", 4) && std::fread(&L, 4, 1, f) == 1 &&
                  std::fread(&E, 4, 1, f) == 1 && L == hp.L && E == hp.E;
        if (ok) {
            heat.resize((size_t) L * E);
            ok = std::fread(heat.data(), sizeof(double), heat.size(), f) == heat.size();
        }
        std::fclose(f);
        if (!ok) std::fprintf(stderr, "strata-oss: profile %s does not fit this model, ignored\n", o.profile_in.c_str());
        return ok;
    }

    void place_and_load() {
        // slots per GPU from the VRAM left, then which experts go there
        int64_t total_slots = 0;
        for (auto& d : devs) {
            HIPCHECK(hipSetDevice(d.id));
            size_t fr = 0, tot = 0;
            HIPCHECK(hipMemGetInfo(&fr, &tot));
            const double avail = (double) fr - o.reserve_mib * 1048576.0;
            const int64_t layer_experts = (int64_t) (d.l1 - d.l0) * hp.E;
            d.n_slots = std::max<int64_t>(0, std::min<int64_t>(layer_experts, (int64_t) (avail / (double) blob_bytes)));
            total_slots += d.n_slots;
        }
        if (o.max_slots >= 0 && total_slots > o.max_slots) {
            const double f = (double) o.max_slots / (double) total_slots;
            total_slots = 0;
            for (auto& d : devs) { d.n_slots = (int64_t) (d.n_slots * f); total_slots += d.n_slots; }
        }
        std::vector<double> prof;
        const bool have_prof = read_profile(prof);
        for (auto& h : hl) {
            h.slot.assign(hp.E, -1);
            h.blob.assign(hp.E, nullptr);
            h.heat.assign(hp.E, 0.0);
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            for (int l = d.l0; l < d.l1; ++l) hl[l].dev = (int) di;
            std::vector<std::pair<int, int>> pick;
            if (have_prof) {
                std::vector<std::pair<int, int>> all;
                for (int l = d.l0; l < d.l1; ++l)
                    for (int e = 0; e < hp.E; ++e) all.push_back({l, e});
                std::stable_sort(all.begin(), all.end(), [&](auto a, auto b) {
                    return prof[(size_t) a.first * hp.E + a.second] > prof[(size_t) b.first * hp.E + b.second];
                });
                all.resize((size_t) d.n_slots);
                pick = all;
                for (int l = d.l0; l < d.l1; ++l)
                    for (int e = 0; e < hp.E; ++e) hl[l].heat[e] = prof[(size_t) l * hp.E + e];
            } else {
                const int nl = d.l1 - d.l0;
                for (int l = d.l0; l < d.l1; ++l) {
                    const int64_t per = d.n_slots / nl + ((l - d.l0) < d.n_slots % nl ? 1 : 0);
                    for (int e = 0; e < per; ++e) pick.push_back({l, e});
                }
            }
            HIPCHECK(hipSetDevice(d.id));
            if (d.n_slots > 0) HIPCHECK(hipMalloc(&d.arena, (size_t) d.n_slots * blob_bytes));
            d.slot_owner = pick;
            for (size_t s = 0; s < pick.size(); ++s) hl[pick[s].first].slot[pick[s].second] = (int32_t) s;
        }
        // the RAM arena: every expert without a slot, page-locked in 1 GiB pieces
        int64_t n_host = 0;
        for (auto& h : hl)
            for (int e = 0; e < hp.E; ++e) n_host += h.slot[e] < 0;
        const int64_t per_chunk = std::max<int64_t>(1, (1ll << 30) / (int64_t) blob_bytes);
        std::vector<uint8_t*> blobs;
        for (int64_t left = n_host; left > 0;) {
            const int64_t k = std::min(left, per_chunk);
            void* p = nullptr;
            HIPCHECK(hipSetDevice(devs[0].id));
            if (hipHostMalloc(&p, (size_t) k * blob_bytes, hipHostMallocPortable) != hipSuccess) {
                std::fprintf(stderr, "strata-oss: page-locking the RAM arena failed, using pageable memory\n");
                p = std::malloc((size_t) k * blob_bytes);
                if (!p) throw std::runtime_error("strata-oss: out of RAM for the experts");
            }
            host_chunks.push_back((uint8_t*) p);
            for (int64_t i = 0; i < k; ++i) blobs.push_back((uint8_t*) p + (size_t) i * blob_bytes);
            left -= k;
        }
        std::fprintf(stderr,
                     "strata-oss: experts: %lld in VRAM (", (long long) total_slots);
        for (size_t i = 0; i < devs.size(); ++i)
            std::fprintf(stderr, "%sGPU %d: %lld, layers %d-%d", i ? "; " : "", devs[i].id, (long long) devs[i].n_slots,
                         devs[i].l0, devs[i].l1 - 1);
        std::fprintf(stderr, "), %lld in RAM (%.1f GiB), %.1f MiB each%s\n", (long long) n_host,
                     n_host * (double) blob_bytes / 1073741824.0, blob_bytes / 1048576.0,
                     have_prof ? ", placed by the profile" : "");
        // stream the experts: repack on worker threads, then upload to the slot or keep in RAM
        size_t next_host = 0;
        const int nthr = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        std::vector<uint8_t*> stage(nthr);
        for (auto& p : stage) HIPCHECK(hipHostMalloc((void**) &p, blob_bytes, hipHostMallocPortable));
        const auto t0 = Clock::now();
        int fd = ::open(gguf->shard(0).path().c_str(), O_RDONLY);
        for (int l = 0; l < hp.L; ++l) {
            HostLayer& h = hl[l];
            Dev& d = devs[h.dev];
            HIPCHECK(hipSetDevice(d.id));
            for (int e0 = 0; e0 < hp.E; e0 += nthr) {
                const int n = std::min(nthr, hp.E - e0);
                std::vector<uint8_t*> dst(n);
                for (int i = 0; i < n; ++i) {
                    const int e = e0 + i;
                    if (h.slot[e] < 0) {
                        h.blob[e] = blobs[next_host++];
                        dst[i] = h.blob[e];
                    } else {
                        dst[i] = stage[i];
                    }
                }
                std::vector<std::thread> th;
                for (int i = 0; i < n; ++i) th.emplace_back([this, l, e = e0 + i, p = dst[i]] { make_blob(l, e, p); });
                for (auto& t : th) t.join();
                for (int i = 0; i < n; ++i) {
                    const int e = e0 + i;
                    if (h.slot[e] >= 0)
                        HIPCHECK(hipMemcpyAsync(d.arena + (size_t) h.slot[e] * blob_bytes, stage[i], blob_bytes,
                                                hipMemcpyHostToDevice, d.s));
                }
                HIPCHECK(hipStreamSynchronize(d.s));
            }
            std::vector<int32_t> so(h.slot);
            HIPCHECK(hipMemcpy(d.layers[l - d.l0].slot_of, so.data(), sizeof(int32_t) * hp.E, hipMemcpyHostToDevice));
            // drop this layer's expert pages (our mapping and the OS cache): the RAM is the arena's
            for (const char* nm : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
                const std::string tn = "blk." + std::to_string(l) + "." + nm + ".weight";
                const auto& t = tensor(tn);
                const uintptr_t lo = (uintptr_t) tdata(tn), hi = lo + strata::tensor_payload_bytes(t);
                const uintptr_t pg = 4096, a0 = (lo + pg - 1) & ~(pg - 1), a1 = hi & ~(pg - 1);
                if (a1 > a0) madvise((void*) a0, a1 - a0, MADV_DONTNEED);
            }
            if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            if (o.verbose || l % 6 == 5 || l + 1 == hp.L)
                std::fprintf(stderr, "strata-oss: loaded layer %d/%d (%.0f s)\n", l + 1, hp.L, secs(t0, Clock::now()));
        }
        if (fd >= 0) ::close(fd);
        for (auto p : stage) HIPCHECK(hipHostFree(p));
        HIPCHECK(hipHostMalloc((void**) &swap_h, blob_bytes, hipHostMallocPortable));
    }

    void alloc_host() {
        HIPCHECK(hipSetDevice(devs[0].id));
        const unsigned coh = hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable;
        HIPCHECK(hipHostMalloc((void**) &mb, sizeof(LayerMailbox) * hp.L, coh));
        std::memset((void*) mb, 0, sizeof(LayerMailbox) * hp.L);
        if (devs.size() > 1) {
            HIPCHECK(hipHostMalloc((void**) &ho, sizeof(Handoff) * (devs.size() - 1), coh));
            std::memset((void*) ho, 0, sizeof(Handoff) * (devs.size() - 1));
        }
        HIPCHECK(hipHostMalloc((void**) &params_h, sizeof(StepParams), hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &emb_h, sizeof(float) * kMaxCols * hp.H, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &logits_h, sizeof(float) * kMaxCols * hp.V, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &prompt_h, sizeof(float) * o.chunk * hp.H, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &ptopk_h, sizeof(int32_t) * o.chunk * kTopK, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &rows_h, sizeof(int32_t) * o.chunk * kTopK, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &inv_h, sizeof(int32_t) * o.chunk * kTopK, hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &tiles_h, sizeof(int32_t) * 4 * (o.chunk * kTopK / 32 + hp.E), hipHostMallocPortable));
        HIPCHECK(hipHostMalloc((void**) &wptr_h, sizeof(void*) * hp.E, hipHostMallocPortable));
    }


    // ---------------------------------------------------------------- EAGLE3 draft
    void load_eagle() {
        Eagle& e = eg;
        e.g.reset(new strata::GgufModel(strata::GgufModel::open(o.draft_model)));
        const strata::GgufFile& m = e.g->meta();
        const strata::MetaValue* arch = m.get("general.architecture");
        if (!arch || arch->s != "eagle3") throw std::runtime_error("strata-oss: " + o.draft_model + " is not an EAGLE3 GGUF");
        auto num = [&](const char* k, double def) {
            const strata::MetaValue* v = m.get(std::string("eagle3.") + k);
            return v ? v->num() : def;
        };
        const strata::MetaValue* tls = m.get("eagle3.target_layers");
        if (!tls || tls->items.size() != 3) throw std::runtime_error("strata-oss: EAGLE3 needs 3 target layers");
        for (int k = 0; k < 3; ++k) e.tl[k] = (int) tls->items[k].num();
        Dev& d = devs.back();
        for (int k = 0; k < 3; ++k)
            if (e.tl[k] < d.l0) {
                std::fprintf(stderr, "strata-oss: EAGLE3 reads layer %d, which is not on the last GPU (layers %d-%d): "
                             "no drafts from it\n", e.tl[k], d.l0, d.l1 - 1);
                e.g.reset();
                return;
            }
        const int H = hp.H;
        if ((int) num("embedding_length", 0) != H || (int) num("target_hidden_size", 0) != H ||
            (int) num("attention.head_count", 0) != hp.NH || (int) num("attention.head_count_kv", 0) != hp.NKV ||
            (int) num("attention.key_length", 0) != hp.HD)
            throw std::runtime_error("strata-oss: the EAGLE3 model's shape does not match the target");
        e.FF = (int) num("feed_forward_length", 0);
        e.eps = (float) num("attention.layer_norm_rms_epsilon", 1e-5);
        e.IN = 2 * H;
        e.R = (hp.NH + 2 * hp.NKV) * hp.HD;
        const float base = (float) num("rope.freq_base", 10000), factor = (float) num("rope.scaling.factor", 1);
        const int n_orig = (int) num("rope.scaling.original_context_length", 4096);
        const float bf = (float) num("rope.scaling.yarn_beta_fast", 32), bs = (float) num("rope.scaling.yarn_beta_slow", 1);
        e.rope.freq_base = base;
        e.rope.freq_scale = 1.f / factor;
        e.rope.ext_factor = factor > 1.f ? 1.f : 0.f;
        e.rope.attn_factor = 1.f;
        e.rope.n_dims = hp.HD;
        e.rope.neox = 0;  // llama.cpp: LLAMA_ROPE_TYPE_NORM for eagle3
        auto corr_dim = [&](float n_rot) {
            return hp.HD * std::log((float) n_orig / (n_rot * 2 * (float) M_PI)) / (2 * std::log(base));
        };
        e.rope.corr0 = std::max(0.f, std::floor(corr_dim(bf)));
        e.rope.corr1 = std::min((float) (hp.HD - 1), std::ceil(corr_dim(bs)));
        HIPCHECK(hipSetDevice(d.id));
        auto up8 = [&](const std::string& n, int n_in, int n_out, int8_t*& q, uint16_t*& dd, std::vector<int8_t>* hq = nullptr,
                       std::vector<uint16_t>* hd = nullptr) {
            std::vector<int8_t> qv((size_t) n_in * n_out);
            std::vector<uint16_t> dv((size_t) n_out * (n_in / 32));
            q8_planes(n, n_in, n_out, qv, dv, 0, e.g.get());
            if (hq) { *hq = std::move(qv); *hd = std::move(dv); return; }
            q = dupload(qv.data(), qv.size());
            dd = dupload(dv.data(), dv.size());
        };
        auto upf = [&](const std::string& n) {
            size_t shard = 0;
            const strata::TensorInfo* t = e.g->find(n, &shard);
            if (!t || t->type != kF32) throw std::runtime_error("strata-oss: EAGLE3 " + n + " missing or not F32");
            return dupload((const float*) e.g->shard(shard).tensor_data(*t), (size_t) t->elements());
        };
        {   // q, k, v as one matrix over the 2H input
            std::vector<int8_t> q((size_t) e.R * e.IN);
            std::vector<uint16_t> dd((size_t) e.R * (e.IN / 32));
            const int NA = hp.NH * hp.HD, NK = hp.NKV * hp.HD;
            q8_planes("blk.0.attn_q.weight", e.IN, NA, q, dd, 0, e.g.get());
            q8_planes("blk.0.attn_k.weight", e.IN, NK, q, dd, NA, e.g.get());
            q8_planes("blk.0.attn_v.weight", e.IN, NK, q, dd, NA + NK, e.g.get());
            e.qkv_q = dupload(q.data(), q.size());
            e.qkv_d = dupload(dd.data(), dd.size());
        }
        up8("blk.0.attn_output.weight", hp.NH * hp.HD, H, e.o_q, e.o_d);
        {   // gate and up as one matrix
            std::vector<int8_t> q((size_t) 2 * e.FF * H);
            std::vector<uint16_t> dd((size_t) 2 * e.FF * (H / 32));
            q8_planes("blk.0.ffn_gate.weight", H, e.FF, q, dd, 0, e.g.get());
            q8_planes("blk.0.ffn_up.weight", H, e.FF, q, dd, e.FF, e.g.get());
            e.gu_q = dupload(q.data(), q.size());
            e.gu_d = dupload(dd.data(), dd.size());
        }
        up8("blk.0.ffn_down.weight", e.FF, H, e.down_q, e.down_d);
        up8("fc.weight", 3 * H, H, e.fc_q, e.fc_d);
        e.attn_norm = upf("blk.0.attn_norm.weight");
        e.attn_norm2 = upf("blk.0.attn_norm_2.weight");
        e.ffn_norm = upf("blk.0.ffn_norm.weight");
        e.out_norm = upf("output_norm.weight");
        e.enc_norm = upf("enc.output_norm.weight");
        {   // the head, cut down to the draft vocabulary
            std::vector<int32_t> ids;
            if (!o.draft_vocab.empty()) {
                FILE* f = std::fopen(o.draft_vocab.c_str(), "r");
                if (!f) throw std::runtime_error("strata-oss: cannot read " + o.draft_vocab);
                int x;
                while (std::fscanf(f, "%d", &x) == 1)
                    if (x >= 0 && x < hp.V) ids.push_back(x);
                std::fclose(f);
            }
            if (ids.empty())
                for (int i = 0; i < hp.V; ++i) ids.push_back(i);
            std::vector<int8_t> full;
            std::vector<uint16_t> fulld;
            up8("output.weight", H, hp.V, e.head_q, e.head_d, &full, &fulld);
            e.NS = (int) ids.size();
            std::vector<int8_t> q((size_t) e.NS * H);
            std::vector<uint16_t> dd((size_t) e.NS * (H / 32));
            for (int r = 0; r < e.NS; ++r) {
                std::memcpy(&q[(size_t) r * H], &full[(size_t) ids[r] * H], H);
                std::memcpy(&dd[(size_t) r * (H / 32)], &fulld[(size_t) ids[r] * (H / 32)], 2 * (H / 32));
            }
            e.head_q = dupload(q.data(), q.size());
            e.head_d = dupload(dd.data(), dd.size());
            e.map_d = dupload(ids.data(), ids.size());
        }
        e.win = o.draft_window > 0 && o.draft_window < o.ctx ? o.draft_window : 0;
        if (const char* v = std::getenv("OSS_EAGLE_CB")) e.cb = std::max(16, std::atoi(v));
        // a window keeps a ring: the window plus the most rows one call writes
        e.cap = e.win ? std::min(o.ctx, e.win + std::max(e.cb, o.chunk) + 16) : o.ctx;
        e.kv_stride = (size_t) e.cap * hp.NKV * hp.HD;
        e.kc = dalloc<uint16_t>(e.kv_stride * o.parallel);
        e.vc = dalloc<uint16_t>(e.kv_stride * o.parallel);
        const int MC = kMaxCols;
        e.feat = dalloc<float>((size_t) MC * 3 * H);
        e.pfeat16 = dalloc<uint16_t>((size_t) o.chunk * 3 * H);
        e.G = o.chunk + 64;
        e.gring = dalloc<float>((size_t) e.G * H * o.parallel);
        e.gtmp = dalloc<float>((size_t) MC * H);
        e.params = dalloc<StepParams>(1);
        e.xe = dalloc<float>((size_t) (e.cb + MC) * H);
        e.gin = dalloc<float>((size_t) MC * H);
        e.cat = dalloc<float>((size_t) MC * e.IN);
        e.qkv = dalloc<float>((size_t) MC * e.R);
        e.ao = dalloc<float>((size_t) MC * hp.NH * hp.HD);
        e.res = dalloc<float>((size_t) MC * H);
        e.gu = dalloc<float>((size_t) MC * 2 * e.FF);
        e.act = dalloc<float>((size_t) MC * e.FF);
        e.out = dalloc<float>((size_t) MC * H);
        e.logits = dalloc<float>((size_t) e.NS);
        e.catq = dalloc<int8_t>((size_t) MC * e.IN);
        e.catd = dalloc<float>((size_t) MC * e.IN / 32);
        e.aoq = dalloc<int8_t>((size_t) MC * hp.NH * hp.HD);
        e.aod = dalloc<float>((size_t) MC * hp.NH * hp.HD / 32);
        e.hq = dalloc<int8_t>((size_t) MC * H);
        e.hd = dalloc<float>((size_t) MC * H / 32);
        e.actq = dalloc<int8_t>((size_t) MC * e.FF);
        e.actd = dalloc<float>((size_t) MC * e.FF / 32);
        e.gq = dalloc<int8_t>((size_t) MC * 3 * H);
        e.gqd = dalloc<float>((size_t) MC * 3 * H / 32);
        e.tok_d = dalloc<int32_t>(2);
        {
            if (const char* v = std::getenv("OSS_EAGLE_CB")) e.cb = std::max(16, std::atoi(v));
            const size_t B = e.cb;
            e.b_g = dalloc<float>(B * H);
            e.b_cat = dalloc<float>(B * e.IN);
            e.b_catq = dalloc<int8_t>(B * e.IN);
            e.b_catd = dalloc<float>(B * e.IN / 32);
            e.b_qkv = dalloc<float>(B * e.R);
            e.b_ao16 = dalloc<uint16_t>(B * hp.NH * hp.HD);
            e.b_aoq = dalloc<int8_t>(B * hp.NH * hp.HD);
            e.b_aod = dalloc<float>(B * hp.NH * hp.HD / 32);
            e.b_res = dalloc<float>(B * H);
            e.b_hq = dalloc<int8_t>(B * H);
            e.b_hd = dalloc<float>(B * H / 32);
            e.b_gu = dalloc<float>(B * 2 * e.FF);
            e.b_act = dalloc<float>(B * e.FF);
            e.b_actq = dalloc<int8_t>(B * e.FF);
            e.b_actd = dalloc<float>(B * e.FF / 32);
            e.b_out = dalloc<float>(B * H);
        }
        if (const char* v = std::getenv("OSS_DRAFT_PMIN")) e.p_min = (float) std::atof(v);
        e.on = true;
        std::fprintf(stderr, "strata-oss: EAGLE3 draft %s on GPU %d (target layers %d, %d, %d; draft vocabulary %d)\n",
                     o.draft_model.c_str(), d.id, e.tl[0], e.tl[1], e.tl[2], e.NS);
    }

    float* gslot(int slot, int pos) { return eg.gring + ((size_t) slot * eg.G + (size_t) (pos % eg.G)) * hp.H; }
    uint16_t* ekc() { return eg.kc + (size_t) cur * eg.kv_stride; }
    uint16_t* evc() { return eg.vc + (size_t) cur * eg.kv_stride; }

    // g(p) = fc(norm(features(p))) for n rows of `feat` (3H floats each) -> the rings: row j is slot slots[j] (or
    // the selected one), position pos[j] (or pos0 + j)
    void eagle_g(const float* feat, int n, int pos0, const int* slots = nullptr, const int* pos = nullptr) {
        Eagle& e = eg;
        Dev& d = devs.back();
        const int H = hp.H;
        for (int r = 0; r < n; r += kMaxCols) {
            const int nc = std::min(kMaxCols, n - r);
            rmsnorm_quant(feat + (size_t) r * 3 * H, e.enc_norm, nullptr, e.gq, e.gqd, 3 * H, nc, e.eps, d.s);
            mv_q8_0(e.fc_q, e.fc_d, e.gq, e.gqd, nullptr, e.gtmp, 3 * H, H, nc, false, d.s);
            for (int j = 0; j < nc; ++j)
                HIPCHECK(hipMemcpyAsync(gslot(slots ? slots[r + j] : cur, pos ? pos[r + j] : pos0 + r + j),
                                        e.gtmp + (size_t) j * H, sizeof(float) * H, hipMemcpyDeviceToDevice, d.s));
        }
    }

    // the same from the prompt's f16 features, 8 rows at a time through the f32 decode buffer
    void eagle_g16(const uint16_t* feat16, int n, int pos0) {
        Eagle& e = eg;
        Dev& d = devs.back();
        const int H = hp.H;
        for (int r = 0; r < n; r += kMaxCols) {
            const int nc = std::min(kMaxCols, n - r);
            f16_to_f32(feat16 + (size_t) r * 3 * H, e.feat, (size_t) nc * 3 * H, d.s);
            eagle_g(e.feat, nc, pos0 + r);
        }
    }

    // the draft decoder over nc rows: embeddings xe (device, nc x H), g inputs (device pointers), positions pos0..
    // Returns the argmax (target id) of the last row when want, else -1.  e.out holds the rows' prenorm states.
    int eagle_rows(const float* xe, const float* const* gsrc, int pos0, int nc, bool want) {
        Eagle& e = eg;
        Dev& d = devs.back();
        const int H = hp.H, NA = hp.NH * hp.HD;
        hipStream_t s = d.s;
        StepParams sp;
        sp.ncols = nc;
        for (int c = 0; c < nc; ++c) sp.pos[c] = pos0 + c;
        HIPCHECK(hipMemcpyAsync(e.params, &sp, sizeof(sp), hipMemcpyHostToDevice, s));
        for (int c = 0; c < nc; ++c)
            HIPCHECK(hipMemcpyAsync(e.gin + (size_t) c * H, gsrc[c], sizeof(float) * H, hipMemcpyDeviceToDevice, s));
        rmsnorm_strided(xe, H, e.attn_norm, e.cat, e.IN, H, nc, e.eps, s);
        rmsnorm_strided(e.gin, H, e.attn_norm2, e.cat + H, e.IN, H, nc, e.eps, s);
        quant_q8(e.cat, e.catq, e.catd, e.IN, nc, s);
        mv_q8_0(e.qkv_q, e.qkv_d, e.catq, e.catd, nullptr, e.qkv, e.IN, e.R, nc, false, s);
        rope_store(e.qkv, ekc(), evc(), e.params, hp.NH, hp.NKV, hp.HD, e.cap, e.rope, nc, s);
        attn_decode(e.qkv, ekc(), evc(), nullptr, e.params, d.attn_part, e.ao, hp.NH, hp.NKV, hp.HD, e.cap, e.win, o.ctx,
                    nc, s);
        quant_q8(e.ao, e.aoq, e.aod, NA, nc, s);
        HIPCHECK(hipMemcpyAsync(e.res, e.gin, sizeof(float) * nc * H, hipMemcpyDeviceToDevice, s));
        mv_q8_0(e.o_q, e.o_d, e.aoq, e.aod, nullptr, e.res, NA, H, nc, true, s);  // ffn_inp = attn + g
        rmsnorm_quant(e.res, e.ffn_norm, nullptr, e.hq, e.hd, H, nc, e.eps, s);
        mv_q8_0(e.gu_q, e.gu_d, e.hq, e.hd, nullptr, e.gu, H, 2 * e.FF, nc, false, s);
        silu_mul(e.gu, e.act, e.FF, nc, s);
        quant_q8(e.act, e.actq, e.actd, e.FF, nc, s);
        HIPCHECK(hipMemcpyAsync(e.out, e.res, sizeof(float) * nc * H, hipMemcpyDeviceToDevice, s));
        mv_q8_0(e.down_q, e.down_d, e.actq, e.actd, nullptr, e.out, e.FF, H, nc, true, s);  // prenorm = ffn + ffn_inp
        if (!want) return -1;
        rmsnorm_quant(e.out + (size_t) (nc - 1) * H, e.out_norm, nullptr, e.hq, e.hd, H, 1, e.eps, s);
        mv_q8_0(e.head_q, e.head_d, e.hq, e.hd, nullptr, e.logits, H, e.NS, 1, false, s);
        argmax_map(e.logits, e.NS, e.map_d, e.tok_d, s);
        int32_t t[2] = {-1, 0};
        HIPCHECK(hipMemcpyAsync(t, e.tok_d, sizeof(t), hipMemcpyDeviceToHost, s));
        HIPCHECK(hipStreamSynchronize(s));
        std::memcpy(&e.last_p, &t[1], 4);
        return t[0];
    }


    // the draft decoder over n (<= kCB) consecutive real rows at positions pos0.. (catch-up for a prompt): the prompt
    // kernels (int8 WMMA projections, tiled attention); xe = their embeddings, g = the ring rows of those positions
    bool eagle_rows_batch(const float* xe, int pos0, int n) {
        Eagle& e = eg;
        Dev& d = devs.back();
        const int H = hp.H, NA = hp.NH * hp.HD;
        hipStream_t s = d.s;
        // g rows (the ring may wrap once)
        const int a = pos0 % e.G, n1 = std::min(n, e.G - a);
        HIPCHECK(hipMemcpyAsync(e.b_g, gslot(cur, pos0), sizeof(float) * n1 * H, hipMemcpyDeviceToDevice, s));
        if (n1 < n)
            HIPCHECK(hipMemcpyAsync(e.b_g + (size_t) n1 * H, gslot(cur, pos0 + n1), sizeof(float) * (n - n1) * H,
                                    hipMemcpyDeviceToDevice, s));
        rmsnorm_strided(xe, H, e.attn_norm, e.b_cat, e.IN, H, n, e.eps, s);
        rmsnorm_strided(e.b_g, H, e.attn_norm2, e.b_cat + H, e.IN, H, n, e.eps, s);
        quant_q8(e.b_cat, e.b_catq, e.b_catd, e.IN, n, s);
        if (!dense_q8_wmma(e.qkv_q, e.qkv_d, e.b_catq, e.b_catd, nullptr, e.b_qkv, e.IN, e.R, n, false, s)) return false;
        rope_store_prompt(e.b_qkv, ekc(), evc(), pos0, hp.NH, hp.NKV, hp.HD, e.cap, e.rope, n, s);
        attn_prompt(e.b_qkv, ekc(), evc(), nullptr, pos0, e.b_ao16, hp.NH, hp.NKV, hp.HD, e.cap, e.win, n, s);
        quant_q8_f16(e.b_ao16, e.b_aoq, e.b_aod, NA, n, s);
        HIPCHECK(hipMemcpyAsync(e.b_res, e.b_g, sizeof(float) * n * H, hipMemcpyDeviceToDevice, s));
        dense_q8_wmma(e.o_q, e.o_d, e.b_aoq, e.b_aod, nullptr, e.b_res, NA, H, n, true, s);
        rmsnorm_quant(e.b_res, e.ffn_norm, nullptr, e.b_hq, e.b_hd, H, n, e.eps, s);
        dense_q8_wmma(e.gu_q, e.gu_d, e.b_hq, e.b_hd, nullptr, e.b_gu, H, 2 * e.FF, n, false, s);
        silu_mul(e.b_gu, e.b_act, e.FF, n, s);
        quant_q8(e.b_act, e.b_actq, e.b_actd, e.FF, n, s);
        HIPCHECK(hipMemcpyAsync(e.b_out, e.b_res, sizeof(float) * n * H, hipMemcpyDeviceToDevice, s));
        dense_q8_wmma(e.down_q, e.down_d, e.b_actq, e.b_actd, nullptr, e.b_out, e.FF, H, n, true, s);
        return true;
    }

    // rows (hist[p + 1], g(p)) for p in [n_valid, npast - 2]
    void eagle_catchup() {
        Eagle& e = eg;
        if (!e.on) return;
        const int H = hp.H;
        const int p0 = S().n_valid, p1 = S().npast - 1;  // exclusive end
        if (p1 <= p0) return;
        Dev& d = devs.back();
        HIPCHECK(hipSetDevice(d.id));
        std::vector<float> emb((size_t) (p1 - p0) * H);
        embed(S().hist.data() + p0 + 1, p1 - p0, emb.data());
        for (int r = 0; r < p1 - p0; r += e.cb) {
            const int n = std::min(e.cb, p1 - p0 - r);
            HIPCHECK(hipMemcpyAsync(e.xe, emb.data() + (size_t) r * H, sizeof(float) * n * H, hipMemcpyHostToDevice, d.s));
            int c = 0;
            // long runs through the prompt kernels (their rounding differs a little from the decode ones: the
            // drafts change, the target's tokens do not)
            for (; n - c > kMaxCols; c += e.cb) {
                const int nb = std::min(e.cb, n - c);
                if (!eagle_rows_batch(e.xe + (size_t) c * H, p0 + r + c, nb)) break;
            }
            for (; c < n; c += kMaxCols) {
                const int nc = std::min(kMaxCols, n - c);
                const float* gs[kMaxCols];
                for (int j = 0; j < nc; ++j) gs[j] = gslot(cur, p0 + r + c + j);
                eagle_rows(e.xe + (size_t) c * H, gs, p0 + r + c, nc, false);
            }
            HIPCHECK(hipStreamSynchronize(d.s));
        }
        S().n_valid = p1;
    }

    int draft(int32_t pending, int k, std::vector<int32_t>& out) {
        out.clear();
        Eagle& e = eg;
        const int npast = S().npast;
        if (!e.on || k <= 0 || npast < 1) return 0;
        const auto t0 = Clock::now();
        const int H = hp.H;
        Dev& d = devs.back();
        HIPCHECK(hipSetDevice(d.id));
        eagle_catchup();
        // the first row is a real one: (pending, g(npast - 1)); the next ones feed the draft's own prenorm state
        int32_t tok = pending;
        const float* g = gslot(cur, npast - 1);
        for (int i = 0; i < k; ++i) {
            const int pos = npast - 1 + i;
            if (pos + 1 >= o.ctx) break;
            std::vector<float> emb(H);
            embed(&tok, 1, emb.data());
            HIPCHECK(hipMemcpyAsync(e.xe, emb.data(), sizeof(float) * H, hipMemcpyHostToDevice, d.s));
            const float* gs[1] = {g};
            tok = eagle_rows(e.xe, gs, pos, 1, true);
            ++e.n_draft_steps;
            if (i == 0) S().n_valid = npast;
            if (e.last_p < e.p_min) break;  // not confident: checking it would likely be wasted
            out.push_back(tok);
            // the next row's g is this row's prenorm state: copy it aside (e.out is rewritten by the next row)
            HIPCHECK(hipMemcpyAsync(e.gtmp, e.out, sizeof(float) * H, hipMemcpyDeviceToDevice, d.s));
            g = e.gtmp;
        }
        e.t_draft += secs(t0, Clock::now());
        return (int) out.size();
    }

    // ---------------------------------------------------------------- decode step
    void capture(Dev& d, int di, int nc) {
        HIPCHECK(hipSetDevice(d.id));
        const int H = hp.H, R = hp.qkv_rows(), NA = hp.NH * hp.HD, E = hp.E, F = hp.F;
        const bool first = di == 0, last = di + 1 == (int) devs.size();
        hipStream_t s = d.s;
        HIPCHECK(hipStreamBeginCapture(s, hipStreamCaptureModeRelaxed));
        HIPCHECK(hipMemcpyAsync(d.params, params_h, sizeof(StepParams), hipMemcpyHostToDevice, s));
        if (first) HIPCHECK(hipMemcpyAsync(d.x, emb_h, sizeof(float) * nc * H, hipMemcpyHostToDevice, s));
        else handoff_recv(&ho[di - 1], d.params, d.x, H, nc, s);
        auto take = [&](int l) {  // EAGLE3: the residual entering layer l (l == L: after the last layer)
            if (!last || !eg.on) return;
            for (int k = 0; k < 3; ++k)
                if (eg.tl[k] == l || (l == hp.L && eg.tl[k] >= hp.L))
                    HIPCHECK(hipMemcpy2DAsync(eg.feat + (size_t) k * H, sizeof(float) * 3 * H, d.x, sizeof(float) * H,
                                              sizeof(float) * H, nc, hipMemcpyDeviceToDevice, s));
        };
        for (int l = d.l0; l < d.l1; ++l) {
            DevLayer& L = d.layers[l - d.l0];
            take(l);
            rmsnorm_quant(d.x, L.attn_norm, nullptr, d.xq, d.xd, H, nc, hp.eps, s);
            mv_q8_0(L.qkv_q, L.qkv_d, d.xq, d.xd, L.qkv_b, d.qkv, H, R, nc, false, s);
            rope_store(d.qkv, L.kc, L.vc, d.params, hp.NH, hp.NKV, hp.HD, L.cap, hp.rope, nc, s, L.kv_stride);
            attn_decode(d.qkv, L.kc, L.vc, L.sinks, d.params, d.attn_part, d.attn_out, hp.NH, hp.NKV, hp.HD, L.cap,
                        L.window, o.ctx, nc, s, L.kv_stride);
            quant_q8(d.attn_out, d.aoq, d.aod, NA, nc, s);
            mv_q8_0(L.o_q, L.o_d, d.aoq, d.aod, L.o_b, d.x, NA, H, nc, true, s);
            rmsnorm_quant(d.x, L.post_norm, d.h, d.hq, d.hd, H, nc, hp.eps, s);
            mv_f32(L.router_w, d.h, L.router_b, d.rlog, H, E, nc, s);
            router_topk(d.rlog, E, nc, d.topk, d.topw, s);
            publish(&mb[l], d.params, d.topk, d.topw, d.hq, d.hd, H, nc, s);
            if (d.n_slots > 0) {
                moe_gate_up(d.arena, blob_bytes, L.slot_of, d.topk, d.hq, d.hd, L.gate_b, L.up_b, d.act, H, F, nc, s);
                quant_q8(d.act, d.actq, d.actd, F, nc * kTopK, s);
                moe_down(d.arena, blob_bytes, L.slot_of, d.topk, d.topw, d.actq, d.actd, L.down_b, d.y, H, F, nc, s);
            } else {
                HIPCHECK(hipMemsetAsync(d.y, 0, sizeof(float) * nc * H, s));
            }
            wait_ack(&mb[l], d.params, s);
            moe_residual(d.x, d.y, &mb[l], H, nc, s);
        }
        take(hp.L);
        if (!last) {
            handoff_send(&ho[di], d.params, d.x, H, nc, s);
        } else {
            rmsnorm_quant(d.x, d.out_norm, nullptr, d.xq, d.xd, H, nc, hp.eps, s);
            mv_q8_0(d.head_q, d.head_d, d.xq, d.xd, nullptr, d.logits, H, hp.V, nc, false, s);
            HIPCHECK(hipMemcpyAsync(logits_h, d.logits, sizeof(float) * nc * hp.V, hipMemcpyDeviceToHost, s));
        }
        hipGraph_t g;
        HIPCHECK(hipStreamEndCapture(s, &g));
        HIPCHECK(hipGraphInstantiate(&d.graph[nc], g, nullptr, nullptr, 0));
        HIPCHECK(hipGraphDestroy(g));
    }

    // one decode step: column c is token toks[c] of conversation slots[c] (consecutive positions when a slot appears
    // more than once: a verify window)
    const float* step_cols(const int32_t* toks, const int* slots, int nc) {
        const int H = hp.H;
        int pos[kMaxCols];
        {
            std::vector<int> used(seqs.size(), 0);
            for (int c = 0; c < nc; ++c) {
                Seq& q = seqs[(size_t) slots[c]];
                pos[c] = q.npast + used[(size_t) slots[c]]++;
                if (pos[c] + 1 > o.ctx) throw std::runtime_error("strata-oss: the context is full");
            }
        }
        for (size_t di = 0; di < devs.size(); ++di)
            if (!devs[di].graph[nc]) capture(devs[di], (int) di, nc);
        ++seq;
        params_h->ncols = nc;
        params_h->seq = seq;
        for (int c = 0; c < nc; ++c) { params_h->pos[c] = pos[c]; params_h->slot[c] = slots[c]; }
        embed(toks, nc, emb_h);
        for (auto& d : devs) {
            HIPCHECK(hipSetDevice(d.id));
            HIPCHECK(hipGraphLaunch(d.graph[nc], d.s));
            HIPCHECK(hipEventRecord(d.done, d.s));
        }
        std::vector<cpu::MissJob> jobs;
        for (int l = 0; l < hp.L; ++l) {
            LayerMailbox& m = mb[l];
            const auto tw = Clock::now();
            uint64_t spins = 0;
            while (load_acquire(&m.routed) != seq) {
                cpu_relax();
                if ((++spins & 0xfffff) == 0 && secs(tw, Clock::now()) > 30)
                    throw std::runtime_error("strata-oss: the GPU did not answer for layer " + std::to_string(l));
            }
            HostLayer& h = hl[l];
            jobs.clear();
            for (int c = 0; c < nc; ++c) {
                m.has_cpu[c] = 0;
                for (int k = 0; k < kTopK; ++k) {
                    const int e = m.topk[c][k];
                    h.heat[e] += 1.0;
                    ++st.routed;
                    if (h.slot[e] >= 0) { ++st.gpu_hits; continue; }
                    cpu::MissJob j;
                    j.blob = h.blob[e];
                    j.gate_b = h.gate_b.data() + (size_t) e * hp.F;
                    j.up_b = h.up_b.data() + (size_t) e * hp.F;
                    j.down_b = h.down_b.data() + (size_t) e * H;
                    j.col = c;
                    j.w = m.topw[c][k];
                    jobs.push_back(j);
                    m.has_cpu[c] = 1;
                }
            }
            if (!jobs.empty())
                cpu::compute_misses(*pool, jobs, &m.hq[0][0], H, &m.hd[0][0], H / 32, &m.y_cpu[0][0], H, scratch);
            std::atomic_thread_fence(std::memory_order_release);
            store_release(&m.acked, seq);
        }
        for (auto& d : devs) {
            HIPCHECK(hipSetDevice(d.id));
            while (hipEventQuery(d.done) == hipErrorNotReady) cpu_relax();
        }
        if (eg.on) {
            HIPCHECK(hipSetDevice(devs.back().id));
            eagle_g(eg.feat, nc, 0, slots, pos);
        }
        for (int c = 0; c < nc; ++c) {
            Seq& q = seqs[(size_t) slots[c]];
            q.npast = std::max(q.npast, pos[c] + 1);
        }
        if (o.adapt_every > 0 && ++steps_since_adapt >= o.adapt_every) {
            steps_since_adapt = 0;
            adapt();
        }
        return logits_h;
    }

    const float* step(const int32_t* toks, int nc) {
        int slots[kMaxCols];
        for (int c = 0; c < nc; ++c) slots[c] = cur;
        return step_cols(toks, slots, nc);
    }

    // ---------------------------------------------------------------- adaptive tier
    void swap(Dev& d, int64_t slot, int l_in, int e_in) {
        const auto [l_out, e_out] = d.slot_owner[(size_t) slot];
        HostLayer& hin = hl[l_in];
        HostLayer& hout = hl[l_out];
        uint8_t* buf = hin.blob[e_in];
        uint8_t* dst = d.arena + (size_t) slot * blob_bytes;
        HIPCHECK(hipMemcpyAsync(swap_h, dst, blob_bytes, hipMemcpyDeviceToHost, d.s));
        HIPCHECK(hipMemcpyAsync(dst, buf, blob_bytes, hipMemcpyHostToDevice, d.s));
        HIPCHECK(hipStreamSynchronize(d.s));
        std::memcpy(buf, swap_h, blob_bytes);
        hout.blob[e_out] = buf;
        hout.slot[e_out] = -1;
        hin.blob[e_in] = nullptr;
        hin.slot[e_in] = (int32_t) slot;
        d.slot_owner[(size_t) slot] = {l_in, e_in};
        ++st.swaps;
    }

    void adapt() {
        int budget = o.adapt_swaps;
        for (auto& d : devs) {
            if (d.n_slots == 0 || budget <= 0) continue;
            HIPCHECK(hipSetDevice(d.id));
            std::vector<std::pair<double, std::pair<int, int>>> cold, hot;
            for (int64_t s = 0; s < d.n_slots; ++s) {
                const auto [l, e] = d.slot_owner[(size_t) s];
                cold.push_back({hl[l].heat[e], {(int) s, 0}});
            }
            for (int l = d.l0; l < d.l1; ++l)
                for (int e = 0; e < hp.E; ++e)
                    if (hl[l].slot[e] < 0) hot.push_back({hl[l].heat[e], {l, e}});
            const int want = std::min<int>(budget, (int) std::min(cold.size(), hot.size()));
            std::partial_sort(cold.begin(), cold.begin() + want, cold.end(),
                              [](auto& a, auto& b) { return a.first < b.first; });
            std::partial_sort(hot.begin(), hot.begin() + want, hot.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            std::vector<int> touched;
            for (int i = 0; i < want; ++i) {
                if (hot[i].first < cold[i].first * 1.3 + 2.0) break;
                const int64_t slot = cold[i].second.first;
                touched.push_back(d.slot_owner[(size_t) slot].first);
                touched.push_back(hot[i].second.first);
                swap(d, slot, hot[i].second.first, hot[i].second.second);
                --budget;
            }
            std::sort(touched.begin(), touched.end());
            touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
            for (int l : touched)
                HIPCHECK(hipMemcpy(d.layers[l - d.l0].slot_of, hl[l].slot.data(), sizeof(int32_t) * hp.E,
                                   hipMemcpyHostToDevice));
        }
        for (auto& h : hl)
            for (auto& v : h.heat) v *= 0.985;
    }

    // ---------------------------------------------------------------- prompt path
    void gemm16(Dev& d, const uint16_t* W, int n_out, int n_in, const uint16_t* X, int ntok, float* Y) {
        const float one = 1.f, zero = 0.f;
        BLASCHECK(hipblasGemmEx(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n_out, ntok, n_in, &one, W, HIP_R_16F, n_in, X,
                                HIP_R_16F, n_in, &zero, Y, HIP_R_32F, n_out, HIPBLAS_COMPUTE_32F,
                                HIPBLAS_GEMM_DEFAULT));
    }

    void prompt_layer(Dev& d, int l, int ntok, int pos0) {
        const int H = hp.H, R = hp.qkv_rows(), NA = hp.NH * hp.HD, E = hp.E, F = hp.F;
        DevLayer L = d.layers[l - d.l0];  // a copy: its KV pointers move to the selected conversation's region
        L.kc += (size_t) cur * L.kv_stride;
        L.vc += (size_t) cur * L.kv_stride;
        HostLayer& h = hl[l];
        hipStream_t s = d.s;
        static const bool prof = std::getenv("OSS_PROF") != nullptr;
        const auto tp0 = Clock::now();
        static const bool dense_wmma = [] {
            const char* v = std::getenv("OSS_DENSE_WMMA");
            return v ? std::atoi(v) != 0 : true;
        }();
        bool qkv_done = false;
        if (dense_wmma) {
            rmsnorm_quant(d.px, L.attn_norm, nullptr, d.pq, d.pqd, H, ntok, hp.eps, s);
            qkv_done = dense_q8_wmma(L.qkv_q, L.qkv_d, d.pq, d.pqd, L.qkv_b, d.pqkv, H, R, ntok, false, s);
        }
        if (!qkv_done) {
            rmsnorm_f16(d.px, L.attn_norm, nullptr, d.xn16, H, ntok, hp.eps, s);
            dequant_q8_0_f16(L.qkv_q, L.qkv_d, d.w16, H, R, s);
            gemm16(d, d.w16, R, H, d.xn16, ntok, d.pqkv);
            add_bias(d.pqkv, L.qkv_b, R, ntok, s);
        }
        // OSS_PROF=1: where the attention block's time goes (synchronizes after each piece)
        Clock::time_point last = tp0;
        auto mark = [&](double& acc) {
            if (!prof) return;
            HIPCHECK(hipStreamSynchronize(s));
            const auto now = Clock::now();
            acc += 1e3 * secs(last, now);
            last = now;
        };
        static double t_qkv = 0, t_attn = 0, t_o = 0, t_rt = 0;
        mark(t_qkv);
        rope_store_prompt(d.pqkv, L.kc, L.vc, pos0, hp.NH, hp.NKV, hp.HD, L.cap, hp.rope, ntok, s);
        attn_prompt(d.pqkv, L.kc, L.vc, L.sinks, pos0, d.ao16, hp.NH, hp.NKV, hp.HD, L.cap, L.window, ntok, s);
        mark(t_attn);
        bool o_done = false;
        if (dense_wmma) {
            quant_q8_f16(d.ao16, d.aoq_p, d.aod_p, NA, ntok, s);
            o_done = dense_q8_wmma(L.o_q, L.o_d, d.aoq_p, d.aod_p, L.o_b, d.px, NA, H, ntok, true, s);  // px += W a + b
        }
        if (!o_done) {
            dequant_q8_0_f16(L.o_q, L.o_d, d.w2, NA, H, s);
            gemm16(d, d.w2, H, NA, d.ao16, ntok, d.py);
            add_bias_residual(d.px, d.py, L.o_b, H, ntok, s);
        }
        mark(t_o);
        rmsnorm_f16(d.px, L.post_norm, d.ph32, nullptr, H, ntok, hp.eps, s);
        {
            const float one = 1.f, zero = 0.f;
            BLASCHECK(hipblasSgemm(d.blas, HIPBLAS_OP_T, HIPBLAS_OP_N, E, ntok, H, &one, L.router_w, H, d.ph32, H,
                                   &zero, d.plog, E));
        }
        add_bias(d.plog, L.router_b, E, ntok, s);
        router_topk(d.plog, E, ntok, d.ptopk, d.ptopw, s);
        HIPCHECK(hipMemcpyAsync(ptopk_h, d.ptopk, sizeof(int32_t) * ntok * kTopK, hipMemcpyDeviceToHost, s));
        HIPCHECK(hipStreamSynchronize(s));
        mark(t_rt);
        if (prof && l == hp.L - 1)
            std::fprintf(stderr, "prof: totals qkv %.1f ms, attention %.1f, o-proj %.1f, router %.1f\n", t_qkv, t_attn, t_o, t_rt);
        const auto tp1 = Clock::now();
        // group the assignments by expert: the VRAM experts first, then the streamed ones in ring batches
        const int na = ntok * kTopK;
        std::vector<int> cnt(E, 0), off(E, 0);
        for (int a = 0; a < na; ++a) {
            const int e = ptopk_h[a];
            ++cnt[e];
            h.heat[e] += 0.25;
        }
        std::vector<int> order, miss;
        for (int e = 0; e < E; ++e)
            if (cnt[e] > 0) {
                st.prompt_routed += cnt[e];
                if (h.slot[e] >= 0) { st.prompt_hits += cnt[e]; order.push_back(e); }
                else miss.push_back(e);
            }
        const int n_res = (int) order.size();
        order.insert(order.end(), miss.begin(), miss.end());
        {
            int o = 0;
            for (int e : order) { off[e] = o; o += cnt[e]; }
        }
        std::vector<int> fill(off);
        for (int a = 0; a < na; ++a) {
            const int e = ptopk_h[a];
            rows_h[fill[e]] = a / kTopK;
            inv_h[a] = fill[e]++;
        }
        // tiles per group: group 0 = the VRAM experts, group g >= 1 = misses [(g-1)*B, g*B)
        const int RN = (int) d.ring.size(), B = RN / 2;
        const int n_groups = 1 + ((int) miss.size() + B - 1) / B;
        std::vector<int> gtile(n_groups + 1, 0), grow(n_groups + 1, 0);
        int nt = 0;
        for (int g = 0; g < n_groups; ++g) {
            gtile[g] = nt;
            const int i0 = g == 0 ? 0 : n_res + (g - 1) * B;
            const int i1 = g == 0 ? n_res : std::min((int) order.size(), n_res + g * B);
            grow[g] = i0 < (int) order.size() ? off[order[i0]] : na;
            for (int i = i0; i < i1; ++i) {
                const int e = order[i];
                for (int r = 0; r < cnt[e]; r += 32) {
                    int32_t* t = tiles_h + 4 * nt++;
                    t[0] = e; t[1] = off[e] + r; t[2] = std::min(32, cnt[e] - r); t[3] = 0;
                }
            }
        }
        gtile[n_groups] = nt;
        grow[n_groups] = na;
        for (int e = 0; e < E; ++e) wptr_h[e] = h.slot[e] >= 0 ? d.arena + (size_t) h.slot[e] * blob_bytes : nullptr;
        for (size_t i = 0; i < miss.size(); ++i) wptr_h[miss[i]] = d.ring[i % RN];
        HIPCHECK(hipMemcpyAsync(d.rows, rows_h, sizeof(int32_t) * na, hipMemcpyHostToDevice, s));
        HIPCHECK(hipMemcpyAsync(d.inv, inv_h, sizeof(int32_t) * na, hipMemcpyHostToDevice, s));
        HIPCHECK(hipMemcpyAsync(d.tiles, tiles_h, sizeof(int32_t) * 4 * nt, hipMemcpyHostToDevice, s));
        HIPCHECK(hipMemcpyAsync(d.wptr, wptr_h, sizeof(void*) * E, hipMemcpyHostToDevice, s));
        // the host buffers are rewritten by the next layer: wait for these copies before returning (cheap)
        HIPCHECK(hipEventRecord(d.staged, s));
        quant_q8(d.ph32, d.pq, d.pqd, H, ntok, s);
        gather_rows_q8(d.pq, d.pqd, d.rows, d.xsq, d.xsd, H, na, s);
        auto issue = [&](int i) {
            const int r = i % RN;
            HIPCHECK(hipStreamWaitEvent(d.cs, d.ring_used[r], 0));
            HIPCHECK(hipMemcpyAsync(d.ring[r], h.blob[miss[i]], blob_bytes, hipMemcpyHostToDevice, d.cs));
            HIPCHECK(hipEventRecord(d.ring_copied[r], d.cs));
        };
        for (int r = 0; r < RN; ++r) HIPCHECK(hipEventRecord(d.ring_used[r], s));
        int issued = 0;
        for (; issued < std::min<int>(RN, (int) miss.size()); ++issued) issue(issued);
        for (int g = 0; g < n_groups; ++g) {
            const int i0 = g == 0 ? 0 : (g - 1) * B, i1 = g == 0 ? 0 : std::min((int) miss.size(), g * B);
            for (int i = i0; i < i1; ++i) HIPCHECK(hipStreamWaitEvent(s, d.ring_copied[i % RN], 0));
            const int t0 = gtile[g], tn = gtile[g + 1] - gtile[g];
            mmq_gate_up(d.wptr, d.tiles, t0, tn, d.xsq, d.xsd, L.gate_b, L.up_b, d.pact, H, F, s);
            const int ra = grow[g], rb = grow[g + 1];
            if (rb > ra) quant_q8(d.pact + (size_t) ra * F, d.paq + (size_t) ra * F, d.pad + (size_t) ra * (F / 32), F,
                                  rb - ra, s);
            mmq_down(d.wptr, d.tiles, t0, tn, d.paq, d.pad, d.ys, H, F, s);
            for (int i = i0; i < i1; ++i) {
                HIPCHECK(hipEventRecord(d.ring_used[i % RN], s));
                if (issued < (int) miss.size()) issue(issued++);
            }
        }
        HIPCHECK(hipEventSynchronize(d.staged));
        moe_combine(d.px, d.ys, d.inv, d.ptopk, d.ptopw, L.down_b, H, ntok, s);
        if (prof) {
            HIPCHECK(hipStreamSynchronize(s));
            const auto tp2 = Clock::now();
            std::fprintf(stderr, "prof: layer %2d ntok %d attn+router %.1f ms, experts %.1f ms (%zu streamed, %d routed)\n", l,
                         ntok, 1e3 * secs(tp0, tp1), 1e3 * secs(tp1, tp2), miss.size(), (int) order.size());
        }
    }

    void prompt_chunk(const int32_t* toks, int ntok, bool want_all) {
        const int H = hp.H;
        int& npast = S().npast;
        if (npast + ntok > o.ctx) throw std::runtime_error("strata-oss: the context is full");
        embed(toks, ntok, prompt_h);
        HIPCHECK(hipSetDevice(devs[0].id));
        HIPCHECK(hipMemcpyAsync(devs[0].px, prompt_h, sizeof(float) * ntok * H, hipMemcpyHostToDevice, devs[0].s));
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& d = devs[di];
            HIPCHECK(hipSetDevice(d.id));
            if (di > 0) {
                HIPCHECK(hipMemcpyAsync(d.px, prompt_h, sizeof(float) * ntok * H, hipMemcpyHostToDevice, d.s));
            }
            const bool lastd = di + 1 == devs.size();
            auto take = [&](int l) {
                if (!lastd || !eg.on) return;
                for (int k = 0; k < 3; ++k)
                    if (eg.tl[k] == l || (l == hp.L && eg.tl[k] >= hp.L))
                        f32_to_f16_2d(d.px, H, eg.pfeat16 + (size_t) k * H, 3 * H, H, ntok, d.s);
            };
            for (int l = d.l0; l < d.l1; ++l) {
                take(l);
                prompt_layer(d, l, ntok, npast);
            }
            take(hp.L);
            if (di + 1 < devs.size()) {
                HIPCHECK(hipMemcpyAsync(prompt_h, d.px, sizeof(float) * ntok * H, hipMemcpyDeviceToHost, d.s));
                HIPCHECK(hipStreamSynchronize(d.s));
            }
        }
        // the head: the last token, or every token 8 at a time
        Dev& d = devs.back();
        const int first = want_all ? 0 : ntok - 1;
        for (int t = first; t < ntok; t += kMaxCols) {
            const int nc = std::min(kMaxCols, ntok - t);
            rmsnorm_quant(d.px + (size_t) t * H, d.out_norm, nullptr, d.xq, d.xd, H, nc, hp.eps, d.s);
            mv_q8_0(d.head_q, d.head_d, d.xq, d.xd, nullptr, d.logits, H, hp.V, nc, false, d.s);
            HIPCHECK(hipMemcpyAsync(logits_h, d.logits, sizeof(float) * nc * hp.V, hipMemcpyDeviceToHost, d.s));
            HIPCHECK(hipStreamSynchronize(d.s));
            if (want_all) all_logits.insert(all_logits.end(), logits_h, logits_h + (size_t) nc * hp.V);
        }
        if (eg.on) eagle_g16(eg.pfeat16, ntok, npast);
        npast += ntok;
    }

    const float* eval(const std::vector<int32_t>& toks, bool want_all) {
        all_logits.clear();
        const float* last = nullptr;
        size_t i = 0;
        const auto t0 = Clock::now();
        static const bool force_decode = std::getenv("OSS_FORCE_DECODE") != nullptr;
        const bool prompt = toks.size() > (size_t) kMaxCols && !force_decode;
        while (i < toks.size()) {
            static const int max_cols = std::getenv("OSS_MAX_COLS") ? std::atoi(std::getenv("OSS_MAX_COLS")) : kMaxCols;
            const int n = (int) std::min(toks.size() - i, (size_t) (prompt ? o.chunk : max_cols));
            if (n > kMaxCols) {
                prompt_chunk(toks.data() + i, n, want_all);
                S().hist.insert(S().hist.end(), toks.begin() + i, toks.begin() + i + n);
                eagle_catchup();  // before the next chunk reuses the g ring
                last = logits_h;
            } else {
                last = step(toks.data() + i, n);
                S().hist.insert(S().hist.end(), toks.begin() + i, toks.begin() + i + n);
                if (want_all) all_logits.insert(all_logits.end(), last, last + (size_t) n * hp.V);
                last += (size_t) (n - 1) * hp.V;
            }
            i += n;
        }
        const double dt = secs(t0, Clock::now());
        if (prompt) { st.t_prompt += dt; st.n_prompt += toks.size(); }
        else { st.t_decode += dt; st.n_decode += toks.size(); }
        if (want_all) return all_logits.data();
        return last;
    }

    int reuse_prefix(const std::vector<int32_t>& toks) {
        Seq& q = S();
        size_t n = 0;
        while (n < toks.size() && n < q.hist.size() && toks[n] == q.hist[n]) ++n;
        // keep at least one token to evaluate
        if (n == toks.size() && n > 0) --n;
        // a sliding-window layer's ring must still hold the window before n
        for (auto& d : devs)
            for (auto& L : d.layers)
                if (L.window && n > 0 && (int64_t) n - L.window + L.cap < (int64_t) q.npast) n = 0;
        q.hist.resize(n);
        q.npast = (int) n;
        q.n_valid = std::min(q.n_valid, std::max(0, (int) n - 1));
        return (int) n;
    }

    // multi-conversation step: one token per slot; also keeps each slot's EAGLE3 rows from falling behind its ring
    const float* step_multi(const std::vector<int>& slots, const std::vector<int32_t>& toks) {
        const int nc = (int) slots.size();
        if (nc < 1 || nc > kMaxCols) throw std::runtime_error("strata-oss: 1-8 conversations a step");
        const auto t0 = Clock::now();
        const float* lg = step_cols(toks.data(), slots.data(), nc);
        for (int c = 0; c < nc; ++c) seqs[(size_t) slots[c]].hist.push_back(toks[c]);
        st.t_decode += secs(t0, Clock::now());
        st.n_decode += nc;
        if (eg.on) {
            const int keep = cur;
            for (int c = 0; c < nc; ++c) {
                Seq& q = seqs[(size_t) slots[c]];
                if (q.npast - q.n_valid > eg.G / 2) {
                    cur = slots[c];
                    eagle_catchup();
                }
            }
            cur = keep;
        }
        return lg;
    }

    void save_profile() const {
        if (o.profile_out.empty()) return;
        const std::string tmp = o.profile_out + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return;
        const int32_t L = hp.L, E = hp.E;
        std::fwrite("OSSP", 1, 4, f);
        std::fwrite(&L, 4, 1, f);
        std::fwrite(&E, 4, 1, f);
        for (auto& h : hl) std::fwrite(h.heat.data(), sizeof(double), h.heat.size(), f);
        std::fclose(f);
        std::rename(tmp.c_str(), o.profile_out.c_str());
    }
};

Engine::Engine(const Options& o) : p_(new Impl) {
    Impl& m = *p_;
    m.o = o;
    m.gguf.reset(new strata::GgufModel(strata::GgufModel::open(o.model)));
    m.read_hparams();
    if (m.o.ctx > m.hp.ctx_train) m.o.ctx = m.hp.ctx_train;
    std::fprintf(stderr,
                 "strata-oss: %s: %d layers, %d experts (top %d), %d heads / %d kv, window %d, vocab %d, ctx %d\n",
                 o.model.c_str(), m.hp.L, m.hp.E, m.hp.K, m.hp.NH, m.hp.NKV, m.hp.swa, m.hp.V, m.o.ctx);
    m.hl.resize(m.hp.L);
    m.o.parallel = std::max(1, std::min(m.o.parallel, kMaxCols));
    m.seqs.resize((size_t) m.o.parallel);
    m.choose_devices();
    for (size_t i = 0; i < m.devs.size(); ++i) m.alloc_device(m.devs[i], i + 1 == m.devs.size());
    m.alloc_host();
    if (!o.draft_model.empty()) m.load_eagle();
    m.place_and_load();
    int nt = o.threads;
    if (nt <= 0) nt = std::max(1u, std::thread::hardware_concurrency() / 2);
    m.pool.reset(new cpu::Pool(nt));
    m.scratch.init(m.hp.H, m.hp.F, kMaxCols * kTopK);
    std::fprintf(stderr, "strata-oss: ready (%d CPU workers)\n", nt);
}

Engine::~Engine() {
    if (p_) p_->save_profile();
}

int Engine::n_vocab() const { return p_->hp.V; }
int Engine::n_ctx() const { return p_->o.ctx; }
int Engine::bos() const { return p_->hp.bos; }
int Engine::eos() const { return p_->hp.eos; }
const std::vector<std::string>& Engine::vocab() const { return p_->vocab; }
int Engine::reuse_prefix(const std::vector<int32_t>& t) { return p_->reuse_prefix(t); }
const float* Engine::eval(const std::vector<int32_t>& t, bool all) { return p_->eval(t, all); }
int Engine::n_past() const { return p_->seqs[(size_t) p_->cur].npast; }
void Engine::truncate(int n) {
    Seq& q = p_->S();
    if (n < q.npast) {
        q.npast = n;
        q.hist.resize(n);
        q.n_valid = std::min(q.n_valid, std::max(0, n - 1));
    }
}
int Engine::n_seqs() const { return (int) p_->seqs.size(); }
void Engine::select(int s) {
    if (s < 0 || s >= (int) p_->seqs.size()) throw std::runtime_error("strata-oss: no conversation slot " + std::to_string(s));
    p_->cur = s;
}
int Engine::selected() const { return p_->cur; }
const float* Engine::step_multi(const std::vector<int>& slots, const std::vector<int32_t>& toks) {
    return p_->step_multi(slots, toks);
}
bool Engine::has_draft() const { return p_->eg.on; }
int Engine::draft(int32_t pending, int k, std::vector<int32_t>& out) { return p_->draft(pending, k, out); }
const std::vector<int32_t>& Engine::history() const { return p_->seqs[(size_t) p_->cur].hist; }
Stats& Engine::stats() { return p_->st; }
int64_t Engine::expert_slots() const {
    int64_t n = 0;
    for (auto& d : p_->devs) n += d.n_slots;
    return n;
}
double Engine::expert_mib() const { return p_->blob_bytes / 1048576.0; }
void Engine::save_profile() { p_->save_profile(); }
void Engine::set_busy(bool b) { p_->pool->set_hot(b); }

void Engine::report(FILE* f) const {
    const Stats& s = p_->st;
    std::fprintf(f,
                 "strata-oss: prompt %llu tokens in %.2f s (%.1f tok/s), output %llu tokens in %.2f s (%.1f tok/s); "
                 "experts in VRAM: decode %.1f%%, prompt %.1f%%; %llu swaps\n",
                 (unsigned long long) s.n_prompt, s.t_prompt, s.t_prompt > 0 ? s.n_prompt / s.t_prompt : 0.0,
                 (unsigned long long) s.n_decode, s.t_decode, s.t_decode > 0 ? s.n_decode / s.t_decode : 0.0,
                 s.routed ? 100.0 * s.gpu_hits / s.routed : 0.0,
                 s.prompt_routed ? 100.0 * s.prompt_hits / s.prompt_routed : 0.0, (unsigned long long) s.swaps);
    if (p_->eg.on && p_->eg.n_draft_steps)
        std::fprintf(f, "strata-oss: EAGLE3 %llu draft tokens in %.2f s (%.2f ms each, incl. catch-up rows)\n",
                     (unsigned long long) p_->eg.n_draft_steps, p_->eg.t_draft,
                     1e3 * p_->eg.t_draft / p_->eg.n_draft_steps);
}

} // namespace oss
