// oss/oss_kernels.hip - see oss_kernels.hpp.
//
// The int8 dot products follow llama.cpp's ggml-cuda (MIT): the activations are rounded to int8 with one float
// scale per 32 values, the 4-bit MXFP4 codes become int8 through a byte permute (get_int_from_table_16, its HIP
// branch), and the sums run on the dot4 instructions.  So a weight means exactly what it means in llama.cpp, and
// the GPU and the CPU (oss_cpu.cpp) round the same way.
#include "oss_kernels.hpp"

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace oss {

namespace {

constexpr int WS = 32;  // RDNA wave32 (the build targets gfx11 / gfx12)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = WS / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, WS);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = WS / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WS));
    return v;
}

__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const int8_t* x = (const int8_t*) &a;
    const int8_t* y = (const int8_t*) &b;
    return c + x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
#endif
}

// ggml-cuda's get_int_from_table_16 (HIP branch) with the MXFP4 table (e2m1 values doubled):
// {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}
__device__ __forceinline__ int2 mxfp4_lut(uint32_t q4) {
    constexpr uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t q_even = q4;
    const uint32_t q_odd = q4 >> 4;
    const uint32_t v_even_low = __builtin_amdgcn_perm(t1, t0, q_even & 0x07070707u);
    const uint32_t v_odd_low = __builtin_amdgcn_perm(t1, t0, q_odd & 0x07070707u);
    const uint32_t v_even_high = __builtin_amdgcn_perm(t3, t2, q_even & 0x07070707u);
    const uint32_t v_odd_high = __builtin_amdgcn_perm(t3, t2, q_odd & 0x07070707u);
    const uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1);
    const uint32_t mask_odd = 0x03020100u | ((q_odd & 0x08080808u) >> 1);
    return make_int2((int) __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
                     (int) __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd));
}

// E8M0 scale, halved (the table above holds doubled values), as ggml_cuda_e8m0_to_fp32(e) * 0.5f
__device__ __forceinline__ float e8m0_half(uint8_t e) {
    const uint32_t bits = e ? ((uint32_t) e << 23) : 0x00400000u;
    return __uint_as_float(bits) * 0.5f;
}

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }

// one MXFP4 block (16 code bytes + its e) against 32 int8 activations
__device__ __forceinline__ float mxfp4_block_dot(const uint4 qv, uint8_t e, const int* xi, float xd) {
    int sumi = 0;
    const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const int2 v = mxfp4_lut(q[l]);
        sumi = dp4a(v.x, xi[l], sumi);
        sumi = dp4a(v.y, xi[l + 4], sumi);
    }
    return e8m0_half(e) * xd * (float) sumi;
}

// ------------------------------------------------------------------------------------------------ norms / quant
// One block per column.  y = x / rms(x) * w (float, optional), then int8 rounding per 32 values.
__global__ void k_rmsnorm_quant(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y,
                                int8_t* __restrict__ q, float* __restrict__ d, int n, float eps) {
    const int c = blockIdx.x;
    x += (size_t) c * n;
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const int lane = threadIdx.x % WS, wid = threadIdx.x / WS, nw = blockDim.x / WS;
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float v = lane < nw ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float) n + eps);
    const int nb = n / 32;
    for (int b = wid; b < nb; b += nw) {
        const int i = b * 32 + lane;
        const float v = x[i] * inv * w[i];
        if (y) y[(size_t) c * n + i] = v;
        if (q) {
            const float amax = warp_max(fabsf(v));
            const float dd = amax / 127.f;
            const float id = dd ? 1.f / dd : 0.f;
            q[(size_t) c * n + i] = (int8_t) roundf(v * id);
            if (lane == 0) d[(size_t) c * nb + b] = dd;
        }
    }
}

__global__ void k_quant_q8(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ d, int n,
                           int nvec) {
    const int lane = threadIdx.x % WS;
    const int nb = n / 32;
    const long gb = (long) blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;  // global block index
    if (gb >= (long) nb * nvec) return;
    const float v = x[gb * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float dd = amax / 127.f;
    const float id = dd ? 1.f / dd : 0.f;
    q[gb * 32 + lane] = (int8_t) roundf(v * id);
    if (lane == 0) d[gb] = dd;
}

// ------------------------------------------------------------------------------------------------ Q8_0 matvec
// A warp per row; lanes stride the row's 32-value blocks.  NC columns share each weight load.
template <int NC>
__global__ void k_mv_q8_0(const int8_t* __restrict__ wq, const uint16_t* __restrict__ wd,
                          const int8_t* __restrict__ xq, const float* __restrict__ xd, const float* __restrict__ bias,
                          float* __restrict__ y, int n_in, int n_out, bool accumulate) {
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (row >= n_out) return;
    const int lane = threadIdx.x % WS;
    const int nb = n_in / 32;
    const int8_t* wr = wq + (size_t) row * n_in;
    const uint16_t* dr = wd + (size_t) row * nb;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    for (int b = lane; b < nb; b += WS) {
        const int4 w0 = *(const int4*) (wr + b * 32);
        const int4 w1 = *(const int4*) (wr + b * 32 + 16);
        const float wdf = h2f(dr[b]);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int8_t* xc = xq + (size_t) c * n_in + b * 32;
            const int4 x0 = *(const int4*) xc;
            const int4 x1 = *(const int4*) (xc + 16);
            int s = 0;
            s = dp4a(w0.x, x0.x, s); s = dp4a(w0.y, x0.y, s); s = dp4a(w0.z, x0.z, s); s = dp4a(w0.w, x0.w, s);
            s = dp4a(w1.x, x1.x, s); s = dp4a(w1.y, x1.y, s); s = dp4a(w1.z, x1.z, s); s = dp4a(w1.w, x1.w, s);
            acc[c] += wdf * xd[(size_t) c * nb + b] * (float) s;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) {
            float o = v + (bias ? bias[row] : 0.f);
            float* yp = y + (size_t) c * n_out + row;
            *yp = accumulate ? *yp + o : o;
        }
    }
}

template <int NC>
__global__ void k_mv_f32(const float* __restrict__ w, const float* __restrict__ x, const float* __restrict__ bias,
                         float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (row >= n_out) return;
    const int lane = threadIdx.x % WS;
    const float* wr = w + (size_t) row * n_in;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;
    for (int i = lane * 4; i < n_in; i += WS * 4) {
        const float4 wv = *(const float4*) (wr + i);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float4 xv = *(const float4*) (x + (size_t) c * n_in + i);
            acc[c] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = warp_sum(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v + (bias ? bias[row] : 0.f);
    }
}

// ------------------------------------------------------------------------------------------------ RoPE (YaRN, NeoX)
// ggml's rope_yarn / rope_yarn_ramp, the NeoX pairing (x[i], x[i + n_dims/2])
__device__ __forceinline__ void yarn_cs(float pos, int i0, const RopeParams& rp, float& cs, float& sn) {
    const float theta_scale = powf(rp.freq_base, -2.0f / (float) rp.n_dims);
    const float theta_extrap = pos * powf(theta_scale, (float) (i0 / 2));
    const float theta_interp = rp.freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = rp.attn_factor;
    if (rp.ext_factor != 0.0f) {
        const float y = ((float) (i0 / 2) - rp.corr0) / fmaxf(0.001f, rp.corr1 - rp.corr0);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float ramp_mix = ramp * rp.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / rp.freq_scale);
    }
    cs = cosf(theta) * mscale;
    sn = sinf(theta) * mscale;
}

// grid (nh + nkv, ncols), block hd/2 threads: one rotation pair each; k and v are stored as f16
__global__ void k_rope_store(float* __restrict__ qkv, uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                             const StepParams* __restrict__ p, int pos0, int nh, int nkv, int hd, int cap,
                             RopeParams rp) {
    const int head = blockIdx.x, c = blockIdx.y, i = threadIdx.x;
    const int pos = p ? p->pos[c] : pos0 + c;
    const int stride = (nh + 2 * nkv) * hd;
    float* row = qkv + (size_t) c * stride;
    float* v = row + head * hd;  // q heads, then k heads
    const int half = hd / 2;
    float cs, sn;
    yarn_cs((float) pos, 2 * i, rp, cs, sn);
    const float x0 = v[i], x1 = v[i + half];
    const float r0 = x0 * cs - x1 * sn, r1 = x0 * sn + x1 * cs;
    v[i] = r0;
    v[i + half] = r1;
    if (head >= nh) {
        const int kh = head - nh;
        const size_t slot = (size_t) (pos % cap);
        uint16_t* kd = kc + (slot * nkv + kh) * hd;
        kd[i] = f2h(r0);
        kd[i + half] = f2h(r1);
        const float* vs = row + (nh + nkv) * hd + kh * hd;
        uint16_t* vd = vc + (slot * nkv + kh) * hd;
        vd[i] = f2h(vs[i]);
        vd[i + half] = f2h(vs[i + half]);
    }
}

// ------------------------------------------------------------------------------------------------ decode attention
constexpr int kSplit = 128;   // keys per split block
constexpr int kHD = 64;       // head dim (checked at load)

// grid (nkv, max_splits, ncols), block (nh/nkv) waves; a wave per query head of the group.  The split's K and V are
// staged once in LDS (rows padded by 2 halves so a lane per key does not hit one bank); a lane scores its own keys,
// then the probabilities go through LDS for the V sum (a lane per 2 dims).
__global__ void __launch_bounds__(256) k_attn_split(const float* __restrict__ qkv, const uint16_t* __restrict__ kc,
                                                    const uint16_t* __restrict__ vc, const StepParams* __restrict__ p,
                                                    float* __restrict__ part, int nh, int nkv, int cap, int window,
                                                    int max_splits) {
    constexpr int KP = kHD + 2;  // padded row, halves
    __shared__ uint16_t Ks[kSplit * KP];
    __shared__ uint16_t Vs[kSplit * kHD];
    __shared__ float P[8][kSplit];
    const int kvh = blockIdx.x, sp = blockIdx.y, c = blockIdx.z;
    const int g = nh / nkv;
    const int w = threadIdx.x / WS, lane = threadIdx.x % WS;
    const int h = kvh * g + w;
    const int pos = p->pos[c];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int t0 = lo + sp * kSplit;
    if (t0 > pos) return;
    const int n = min(pos + 1, t0 + kSplit) - t0;
    // stage: 8 threads x 16 bytes per 128-byte row
    for (int idx = threadIdx.x; idx < kSplit * 8; idx += blockDim.x) {
        const int j = idx / 8, q8 = (idx % 8) * 8;
        if (j < n) {
            const size_t slot = (size_t) ((t0 + j) % cap);
            const uint4 kv = *(const uint4*) (kc + (slot * nkv + kvh) * kHD + q8);
            const uint4 vv = *(const uint4*) (vc + (slot * nkv + kvh) * kHD + q8);
            uint32_t* kd = (uint32_t*) &Ks[j * KP + q8];  // 4-byte aligned (KP even)
            kd[0] = kv.x; kd[1] = kv.y; kd[2] = kv.z; kd[3] = kv.w;
            *(uint4*) &Vs[j * kHD + q8] = vv;
        }
    }
    __syncthreads();
    const float scale = rsqrtf((float) kHD);
    const float* q = qkv + (size_t) c * (nh + 2 * nkv) * kHD + h * kHD;
    float qr[kHD];
#pragma unroll
    for (int d = 0; d < kHD; ++d) qr[d] = q[d] * scale;
    constexpr int PER = kSplit / WS;
    float sc[PER];
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int j = lane + i * WS;
        float v = -INFINITY;
        if (j < n) {
            v = 0.f;
            const uint32_t* kr = (const uint32_t*) &Ks[j * KP];
#pragma unroll
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const uint32_t kk = kr[d2];
                v += qr[2 * d2] * h2f((uint16_t) (kk & 0xffff)) + qr[2 * d2 + 1] * h2f((uint16_t) (kk >> 16));
            }
        }
        sc[i] = v;
        m = fmaxf(m, v);
    }
    m = warp_max(m);
    float l = 0.f;
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const float pe = sc[i] == -INFINITY ? 0.f : __expf(sc[i] - m);
        l += pe;
        P[w][lane + i * WS] = pe;
    }
    l = warp_sum(l);
    __syncthreads();
    float a0 = 0.f, a1 = 0.f;
    for (int j = 0; j < n; ++j) {
        const float pe = P[w][j];
        const uint32_t vv = *(const uint32_t*) &Vs[j * kHD + 2 * lane];
        a0 += pe * h2f((uint16_t) (vv & 0xffff));
        a1 += pe * h2f((uint16_t) (vv >> 16));
    }
    float* o = part + (((size_t) c * nh + h) * max_splits + sp) * (kHD + 2);
    if (lane == 0) { o[0] = m; o[1] = l; }
    o[2 + 2 * lane] = a0;
    o[3 + 2 * lane] = a1;
}

// grid (nh, ncols), block 64: merges the splits and the head's sink logit
__global__ void k_attn_combine(const float* __restrict__ part, const float* __restrict__ sinks,
                               const StepParams* __restrict__ p, float* __restrict__ out, int nh, int window,
                               int max_splits) {
    const int h = blockIdx.x, c = blockIdx.y, d = threadIdx.x;
    const int pos = p->pos[c];
    const int lo = window > 0 ? max(0, pos - window + 1) : 0;
    const int ns = (pos - lo + 1 + kSplit - 1) / kSplit;
    const float* base = part + ((size_t) c * nh + h) * max_splits * (kHD + 2);
    const float sink = sinks[h];
    float mt = sink;
    for (int s = 0; s < ns; ++s) mt = fmaxf(mt, base[s * (kHD + 2)]);
    float l = __expf(sink - mt), a = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float* o = base + s * (kHD + 2);
        const float f = __expf(o[0] - mt);
        l += o[1] * f;
        a += o[2 + d] * f;
    }
    out[(size_t) c * nh * kHD + h * kHD + d] = a / l;
}

// ------------------------------------------------------------------------------------------------ router
// one wave per column: top-K of the logits (ties: lower index), softmax over the K picks
__global__ void k_router_topk(const float* __restrict__ logits, int n_expert, int32_t* __restrict__ topk,
                              float* __restrict__ topw) {
    const int c = blockIdx.x;
    const int lane = threadIdx.x;
    const float* lg = logits + (size_t) c * n_expert;
    // each lane keeps its own candidates in registers: n_expert <= 8 * WS
    float v[8];
    int taken = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int i = lane + j * WS;
        v[j] = i < n_expert ? lg[i] : -INFINITY;
    }
    int pick[kTopK];
    float pv[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        float best = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int i = lane + j * WS;
            if (!(taken >> j & 1) && i < n_expert && (v[j] > best || (v[j] == best && i < bi))) { best = v[j]; bi = i; }
        }
#pragma unroll
        for (int o = WS / 2; o > 0; o >>= 1) {
            const float ob = __shfl_xor(best, o, WS);
            const int oi = __shfl_xor(bi, o, WS);
            if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; }
        }
        pick[k] = bi;
        pv[k] = best;
        if (bi % WS == lane) taken |= 1 << (bi / WS);
    }
    if (lane == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < kTopK; ++k) mx = fmaxf(mx, pv[k]);
        float e[kTopK], s = 0.f;
        for (int k = 0; k < kTopK; ++k) { e[k] = expf(pv[k] - mx); s += e[k]; }
        for (int k = 0; k < kTopK; ++k) {
            topk[c * kTopK + k] = pick[k];
            topw[c * kTopK + k] = e[k] / s;
        }
    }
}

// ------------------------------------------------------------------------------------------------ mailbox
__device__ __forceinline__ void store_release_sys(volatile uint32_t* f, uint32_t v) {
    __hip_atomic_store((uint32_t*) f, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint32_t load_acquire_sys(const volatile uint32_t* f) {
    return __hip_atomic_load((uint32_t*) f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void k_publish(LayerMailbox* mb, const StepParams* __restrict__ p, const int32_t* __restrict__ topk,
                          const float* __restrict__ topw, const int8_t* __restrict__ hq, const float* __restrict__ hd,
                          int n) {
    const int nc = p->ncols;
    const int nb = n / 32;
    for (int i = threadIdx.x; i < nc * kTopK; i += blockDim.x) {
        (&mb->topk[0][0])[i] = topk[i];
        (&mb->topw[0][0])[i] = topw[i];
    }
    for (int c = 0; c < nc; ++c) {
        const int4* src = (const int4*) (hq + (size_t) c * n);
        int4* dst = (int4*) mb->hq[c];
        for (int i = threadIdx.x; i < n / 16; i += blockDim.x) dst[i] = src[i];
        for (int i = threadIdx.x; i < nb; i += blockDim.x) mb->hd[c][i] = hd[(size_t) c * nb + i];
    }
    if (threadIdx.x == 0) mb->ncols = nc;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) store_release_sys(&mb->routed, p->seq);
}

__global__ void k_wait_ack(const LayerMailbox* mb, const StepParams* __restrict__ p) {
    const uint32_t seq = p->seq;
    while (load_acquire_sys(&mb->acked) != seq) __builtin_amdgcn_s_sleep(2);
}

__global__ void k_moe_residual(float* __restrict__ x, const float* __restrict__ y, const LayerMailbox* mb, int n) {
    const int c = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[(size_t) c * n + i] + y[(size_t) c * n + i];
    if (mb->has_cpu[c]) v += mb->y_cpu[c][i];
    x[(size_t) c * n + i] = v;
}

__global__ void k_handoff_send(Handoff* h, const StepParams* __restrict__ p, const float* __restrict__ x, int n) {
    const int nc = p->ncols;
    for (int c = 0; c < nc; ++c)
        for (int i = threadIdx.x; i < n; i += blockDim.x) h->x[c][i] = x[(size_t) c * n + i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) store_release_sys(&h->seq, p->seq);
}

__global__ void k_handoff_recv(const Handoff* h, const StepParams* __restrict__ p, float* __restrict__ x, int n) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        while (load_acquire_sys(&h->seq) != p->seq) __builtin_amdgcn_s_sleep(2);
        ok = 1;
    }
    __syncthreads();
    const int nc = p->ncols;
    for (int c = 0; c < nc; ++c)
        for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) c * n + i] = h->x[c][i];
}

// ------------------------------------------------------------------------------------------------ MoE (decode)
__device__ __forceinline__ float swiglu_oai(float g, float u) {
    constexpr float alpha = 1.702f, limit = 7.0f;
    const float x = fminf(g, limit);
    const float y = fminf(fmaxf(u, -limit), limit);
    return x / (1.f + __expf(alpha * (-x))) * (y + 1.f);
}

// row dot of an MXFP4 matrix (planar) against int8 activations; the warp's lanes stride the blocks
__device__ __forceinline__ float mxfp4_row_dot(const uint8_t* __restrict__ mat, int rows, int nb, int row,
                                               const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                               int lane) {
    const uint4* qs = (const uint4*) mat + (size_t) row * nb;
    const uint8_t* es = mat + (size_t) rows * nb * 16 + (size_t) row * nb;
    float acc = 0.f;
    for (int b = lane; b < nb; b += WS) {
        const uint4 qv = qs[b];
        const int4 x0 = *(const int4*) (xq + b * 32);
        const int4 x1 = *(const int4*) (xq + b * 32 + 16);
        const int xi[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        acc += mxfp4_block_dot(qv, es[b], xi, xd[b]);
    }
    return warp_sum(acc);
}

// grid (n_ff / rows_per_block, ncols * K); act[(c*K + k)][r] = swiglu(gate . h + bg, up . h + bu) for VRAM experts
__global__ void k_moe_gate_up(const uint8_t* __restrict__ arena, size_t blob_bytes, const int32_t* __restrict__ slot_of,
                              const int32_t* __restrict__ topk, const int8_t* __restrict__ hq,
                              const float* __restrict__ hd, const float* __restrict__ gate_b,
                              const float* __restrict__ up_b, float* __restrict__ act, int n_embd, int n_ff) {
    const int a = blockIdx.y;  // assignment c*K + k
    const int c = a / kTopK;
    const int e = topk[a];
    const int slot = slot_of[e];
    if (slot < 0) return;
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (row >= n_ff) return;
    const int lane = threadIdx.x % WS;
    const int nb = n_embd / 32;
    const uint8_t* blob = arena + (size_t) slot * blob_bytes;
    const size_t mat_bytes = (size_t) n_ff * nb * 17;
    const int8_t* xq = hq + (size_t) c * n_embd;
    const float* xd = hd + (size_t) c * nb;
    const float g = mxfp4_row_dot(blob, n_ff, nb, row, xq, xd, lane);
    const float u = mxfp4_row_dot(blob + mat_bytes, n_ff, nb, row, xq, xd, lane);
    if (lane == 0)
        act[(size_t) a * n_ff + row] =
            swiglu_oai(g + gate_b[(size_t) e * n_ff + row], u + up_b[(size_t) e * n_ff + row]);
}

// grid (n_embd / rows_per_block, ncols): y[c][r] = sum over the column's VRAM experts of w * (down . act + bd)
__global__ void k_moe_down(const uint8_t* __restrict__ arena, size_t blob_bytes, const int32_t* __restrict__ slot_of,
                           const int32_t* __restrict__ topk, const float* __restrict__ topw,
                           const int8_t* __restrict__ aq, const float* __restrict__ ad,
                           const float* __restrict__ down_b, float* __restrict__ y, int n_embd, int n_ff) {
    const int c = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / WS) + threadIdx.x / WS;
    if (row >= n_embd) return;
    const int lane = threadIdx.x % WS;
    const int nbh = n_embd / 32, nbf = n_ff / 32;
    const size_t gu_bytes = 2 * (size_t) n_ff * nbh * 17;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int a = c * kTopK + k;
        const int e = topk[a];
        const int slot = slot_of[e];
        if (slot < 0) continue;
        const uint8_t* down = arena + (size_t) slot * blob_bytes + gu_bytes;
        const float v = mxfp4_row_dot(down, n_embd, nbf, row, aq + (size_t) a * n_ff, ad + (size_t) a * nbf, lane);
        acc += topw[a] * (v + down_b[(size_t) e * n_embd + row]);
    }
    if (lane == 0) y[(size_t) c * n_embd + row] = acc;
}

// ------------------------------------------------------------------------------------------------ prompt path
__global__ void k_dequant_q8_0(const int8_t* __restrict__ wq, const uint16_t* __restrict__ wd,
                               uint16_t* __restrict__ out, int n_in, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t row = i / n_in;
    const int col = (int) (i % n_in);
    out[i] = f2h((float) wq[i] * h2f(wd[row * (n_in / 32) + col / 32]));
}

// one thread per 32-value block: writes 32 halves
__global__ void k_dequant_mxfp4(const uint8_t* __restrict__ m, uint16_t* __restrict__ out, int n_in, int rows) {
    const int nb = n_in / 32;
    const size_t gb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= (size_t) rows * nb) return;
    const uint8_t* qs = m + gb * 16;
    const float d = e8m0_half(m[(size_t) rows * nb * 16 + gb]);
    constexpr float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    uint16_t* o = out + gb * 32;
    const uint4 qv = *(const uint4*) qs;
    const uint8_t* b = (const uint8_t*) &qv;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        o[j] = f2h(kv[b[j] & 0xf] * d);
        o[j + 16] = f2h(kv[b[j] >> 4] * d);
    }
}

__global__ void k_f32_to_f16(const float* __restrict__ x, uint16_t* __restrict__ y, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h(x[i]);
}

__global__ void k_rmsnorm_f16(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y32,
                              uint16_t* __restrict__ y16, int n, float eps) {
    const int c = blockIdx.x;
    x += (size_t) c * n;
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const int lane = threadIdx.x % WS, wid = threadIdx.x / WS, nw = blockDim.x / WS;
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    if (wid == 0) {
        float v = lane < nw ? red[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red[0] / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] * inv * w[i];
        if (y32) y32[(size_t) c * n + i] = v;
        if (y16) y16[(size_t) c * n + i] = f2h(v);
    }
}

__global__ void k_add_bias(float* __restrict__ y, const float* __restrict__ b, int n, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i % n];
}

__global__ void k_add_bias_residual(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ b,
                                    int n, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) x[i] += y[i] + b[i % n];
}

// Tiled attention for a prompt chunk: a thread per query, 64 queries a block, 64-key tiles in LDS.
// grid (ceil(ntok / 64), nh)
constexpr int kQT = 64, kKT = 64;
__global__ void __launch_bounds__(kQT) k_attn_prompt(const float* __restrict__ qkv, const uint16_t* __restrict__ kc,
                                                     const uint16_t* __restrict__ vc, const float* __restrict__ sinks,
                                                     int pos0, uint16_t* __restrict__ out, int nh, int nkv, int cap,
                                                     int window, int ntok) {
    __shared__ float Ks[kKT][kHD + 1];
    __shared__ float Vs[kKT][kHD];
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    const int qi = blockIdx.x * kQT + threadIdx.x;
    const bool active = qi < ntok;
    const int qpos = pos0 + (active ? qi : ntok - 1);
    const int lo = window > 0 ? max(0, qpos - window + 1) : 0;
    // the block's key range
    const int first_q = blockIdx.x * kQT;
    const int last_q = min(ntok, first_q + kQT) - 1;
    const int blo = window > 0 ? max(0, pos0 + first_q - window + 1) : 0;
    const int bhi = pos0 + last_q;
    const float scale = rsqrtf((float) kHD);
    float q[kHD], acc[kHD];
    const float* qp = qkv + (size_t) (active ? qi : 0) * (nh + 2 * nkv) * kHD + h * kHD;
#pragma unroll
    for (int d = 0; d < kHD; ++d) { q[d] = qp[d] * scale; acc[d] = 0.f; }
    float m = sinks[h], l = 1.f;  // the sink: exp(sink - m) in the denominator, no value
    for (int t0 = blo; t0 <= bhi; t0 += kKT) {
        __syncthreads();
        // load the tile: 64 keys x 64 dims of K and V
        for (int idx = threadIdx.x; idx < kKT * (kHD / 8); idx += kQT) {
            const int j = idx / (kHD / 8), d8 = (idx % (kHD / 8)) * 8;
            const int t = t0 + j;
            if (t <= bhi) {
                const size_t slot = (size_t) (t % cap);
                const uint4 kv = *(const uint4*) (kc + (slot * nkv + kvh) * kHD + d8);
                const uint4 vv = *(const uint4*) (vc + (slot * nkv + kvh) * kHD + d8);
                const uint16_t* kb = (const uint16_t*) &kv;
                const uint16_t* vb = (const uint16_t*) &vv;
#pragma unroll
                for (int u = 0; u < 8; ++u) { Ks[j][d8 + u] = h2f(kb[u]); Vs[j][d8 + u] = h2f(vb[u]); }
            }
        }
        __syncthreads();
        if (!active) continue;
        const int jn = min(kKT, bhi - t0 + 1);
        float s[kKT];
        float tmax = -INFINITY;
#pragma unroll 4
        for (int j = 0; j < kKT; ++j) {
            const int t = t0 + j;
            float v = -INFINITY;
            if (j < jn && t >= lo && t <= qpos) {
                v = 0.f;
#pragma unroll
                for (int d = 0; d < kHD; ++d) v += q[d] * Ks[j][d];
            }
            s[j] = v;
            tmax = fmaxf(tmax, v);
        }
        if (tmax == -INFINITY) continue;
        const float mn = fmaxf(m, tmax);
        const float sc = __expf(m - mn);
        l *= sc;
#pragma unroll
        for (int d = 0; d < kHD; ++d) acc[d] *= sc;
#pragma unroll 2
        for (int j = 0; j < kKT; ++j) {
            if (s[j] == -INFINITY) continue;
            const float pe = __expf(s[j] - mn);
            l += pe;
#pragma unroll
            for (int d = 0; d < kHD; ++d) acc[d] += pe * Vs[j][d];
        }
        m = mn;
    }
    if (!active) return;
    uint16_t* o = out + (size_t) qi * nh * kHD + h * kHD;
    const float il = 1.f / l;
#pragma unroll
    for (int d = 0; d < kHD; ++d) o[d] = f2h(acc[d] * il);
}

__global__ void k_gather_rows_f16(const uint16_t* __restrict__ src, const int32_t* __restrict__ rows,
                                  uint16_t* __restrict__ dst, int n) {
    const int r = blockIdx.x;
    const uint4* s = (const uint4*) (src + (size_t) rows[r] * n);
    uint4* d = (uint4*) (dst + (size_t) r * n);
    for (int i = threadIdx.x; i < n / 8; i += blockDim.x) d[i] = s[i];
}

__global__ void k_swiglu_bias_f16(const float* __restrict__ gu, const float* __restrict__ gb,
                                  const float* __restrict__ ub, uint16_t* __restrict__ out, int n_ff, size_t total) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const size_t r = i / n_ff;
    const int j = (int) (i % n_ff);
    const float g = gu[r * 2 * n_ff + j] + gb[j];
    const float u = gu[r * 2 * n_ff + n_ff + j] + ub[j];
    out[i] = f2h(swiglu_oai(g, u));
}

__global__ void k_moe_combine(float* __restrict__ x, const float* __restrict__ ys, const int32_t* __restrict__ inv,
                              const int32_t* __restrict__ topk, const float* __restrict__ topw,
                              const float* __restrict__ down_b, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        const int a = t * kTopK + k;
        acc += topw[a] * (ys[(size_t) inv[a] * n + i] + down_b[(size_t) topk[a] * n + i]);
    }
    x[(size_t) t * n + i] += acc;
}

// ------------------------------------------------------------------------------------------------ prompt MoE (grouped int8)
// The prompt's experts in one launch: MXFP4 weights x int8 activations, a 32-token tile of one expert per block.
// The assignment rows are sorted by expert (xq/xd row i = the i-th assignment); `tiles` = (expert, first row, rows).
// gate_up: 64 output rows of gate AND up per block, writes act = swiglu(g + bg, u + bu) as float.
constexpr int kTN = 32;  // tokens per tile
__global__ void __launch_bounds__(256) k_mmq_gate_up(const uint8_t* const* __restrict__ wptr,
                                                      const int4* __restrict__ tiles, int tile0,
                                                      const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                      const float* __restrict__ gate_b, const float* __restrict__ up_b,
                                                      float* __restrict__ act, int n_embd, int n_ff) {
    constexpr int TM = 64, KB = 2;
    __shared__ int Wg[TM][KB * 8 + 1], Wu[TM][KB * 8 + 1];
    __shared__ float Wgd[TM][KB], Wud[TM][KB];
    __shared__ int X[kTN][KB * 8 + 1];
    __shared__ float Xd[kTN][KB];
    const int4 tl = tiles[tile0 + blockIdx.y];
    const int e = tl.x, r0 = tl.y, nt = tl.z;
    const int row0 = blockIdx.x * TM;
    const int nb = n_embd / 32;
    const uint8_t* gate = wptr[e];
    const size_t mat = (size_t) n_ff * nb * 17;
    const uint8_t* up = gate + mat;
    const int t = threadIdx.x;
    const int tr = t % 32, tt = (t / 32) * 4;  // rows tr, tr + 32; tokens tt..tt+3
    float ag[2][4] = {}, au[2][4] = {};
    for (int b0 = 0; b0 < nb; b0 += KB) {
        {   // weights: 2 matrices x 64 rows x KB blocks = 256 (thread) loads
            const int m = t / (TM * KB), rem = t % (TM * KB), r = rem / KB, kb = rem % KB;
            const int row = row0 + r;
            const uint8_t* base = m ? up : gate;
            int* dst = m ? &Wu[r][kb * 8] : &Wg[r][kb * 8];
            float* dd = m ? &Wud[r][kb] : &Wgd[r][kb];
            if (row < n_ff) {
                const uint4 qv = ((const uint4*) base)[(size_t) row * nb + b0 + kb];
                const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int2 v = mxfp4_lut(q[l]);
                    dst[l] = v.x;
                    dst[l + 4] = v.y;
                }
                *dd = e8m0_half(base[(size_t) n_ff * nb * 16 + (size_t) row * nb + b0 + kb]);
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) dst[l] = 0;
                *dd = 0.f;
            }
        }
        if (t < kTN * KB * 2) {  // activations: 32 tokens x KB blocks x 2 halves of 16 bytes
            const int tok = t / (KB * 2), rem = t % (KB * 2), kb = rem / 2, h = rem % 2;
            int4 v = make_int4(0, 0, 0, 0);
            if (tok < nt) v = *(const int4*) (xq + (size_t) (r0 + tok) * n_embd + (b0 + kb) * 32 + h * 16);
            X[tok][kb * 8 + h * 4 + 0] = v.x; X[tok][kb * 8 + h * 4 + 1] = v.y;
            X[tok][kb * 8 + h * 4 + 2] = v.z; X[tok][kb * 8 + h * 4 + 3] = v.w;
            if (h == 0) Xd[tok][kb] = tok < nt ? xd[(size_t) (r0 + tok) * nb + b0 + kb] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = tr + 32 * i;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int sg = 0, su = 0;
#pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        const int xv = X[tt + j][kb * 8 + l];
                        sg = dp4a(Wg[r][kb * 8 + l], xv, sg);
                        su = dp4a(Wu[r][kb * 8 + l], xv, su);
                    }
                    const float xs = Xd[tt + j][kb];
                    ag[i][j] += Wgd[r][kb] * xs * (float) sg;
                    au[i][j] += Wud[r][kb] * xs * (float) su;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = row0 + tr + 32 * i;
        if (row >= n_ff) continue;
        const float bg = gate_b[(size_t) e * n_ff + row], bu = up_b[(size_t) e * n_ff + row];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (tt + j < nt) act[(size_t) (r0 + tt + j) * n_ff + row] = swiglu_oai(ag[i][j] + bg, au[i][j] + bu);
    }
}

// down: 128 output rows per block; y[row i][r] = down_e[r] . act_i (no bias: moe_combine adds it)
__global__ void __launch_bounds__(256) k_mmq_down(const uint8_t* const* __restrict__ wptr,
                                                   const int4* __restrict__ tiles, int tile0,
                                                   const int8_t* __restrict__ aq, const float* __restrict__ ad,
                                                   float* __restrict__ y, int n_embd, int n_ff) {
    constexpr int TM = 128, KB = 2;
    __shared__ int W[TM][KB * 8 + 1];
    __shared__ float Wd[TM][KB];
    __shared__ int X[kTN][KB * 8 + 1];
    __shared__ float Xd[kTN][KB];
    const int4 tl = tiles[tile0 + blockIdx.y];
    const int e = tl.x, r0 = tl.y, nt = tl.z;
    const int row0 = blockIdx.x * TM;
    const int nbh = n_embd / 32, nb = n_ff / 32;
    const uint8_t* down = wptr[e] + 2 * (size_t) n_ff * nbh * 17;
    const int t = threadIdx.x;
    const int tr = t % 32, tt = (t / 32) * 4;  // rows tr + 32 i (i < 4); tokens tt..tt+3
    float acc[4][4] = {};
    for (int b0 = 0; b0 < nb; b0 += KB) {
        {
            const int r = t / KB, kb = t % KB;
            const int row = row0 + r;
            int* dst = &W[r][kb * 8];
            if (row < n_embd) {
                const uint4 qv = ((const uint4*) down)[(size_t) row * nb + b0 + kb];
                const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int2 v = mxfp4_lut(q[l]);
                    dst[l] = v.x;
                    dst[l + 4] = v.y;
                }
                Wd[r][kb] = e8m0_half(down[(size_t) n_embd * nb * 16 + (size_t) row * nb + b0 + kb]);
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) dst[l] = 0;
                Wd[r][kb] = 0.f;
            }
        }
        if (t < kTN * KB * 2) {
            const int tok = t / (KB * 2), rem = t % (KB * 2), kb = rem / 2, h = rem % 2;
            int4 v = make_int4(0, 0, 0, 0);
            if (tok < nt) v = *(const int4*) (aq + (size_t) (r0 + tok) * n_ff + (b0 + kb) * 32 + h * 16);
            X[tok][kb * 8 + h * 4 + 0] = v.x; X[tok][kb * 8 + h * 4 + 1] = v.y;
            X[tok][kb * 8 + h * 4 + 2] = v.z; X[tok][kb * 8 + h * 4 + 3] = v.w;
            if (h == 0) Xd[tok][kb] = tok < nt ? ad[(size_t) (r0 + tok) * nb + b0 + kb] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = tr + 32 * i;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int sm = 0;
#pragma unroll
                    for (int l = 0; l < 8; ++l) sm = dp4a(W[r][kb * 8 + l], X[tt + j][kb * 8 + l], sm);
                    acc[i][j] += Wd[r][kb] * Xd[tt + j][kb] * (float) sm;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + tr + 32 * i;
        if (row >= n_embd) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (tt + j < nt) y[(size_t) (r0 + tt + j) * n_embd + row] = acc[i][j];
    }
}

__global__ void k_gather_rows_q8(const int8_t* __restrict__ q, const float* __restrict__ d,
                                 const int32_t* __restrict__ rows, int8_t* __restrict__ oq, float* __restrict__ od,
                                 int n) {
    const int r = blockIdx.x;
    const int src = rows[r];
    const int4* s = (const int4*) (q + (size_t) src * n);
    int4* o = (int4*) (oq + (size_t) r * n);
    for (int i = threadIdx.x; i < n / 16; i += blockDim.x) o[i] = s[i];
    for (int i = threadIdx.x; i < n / 32; i += blockDim.x) od[(size_t) r * (n / 32) + i] = d[(size_t) src * (n / 32) + i];
}

// WMMA prompt attention (gfx12): a block = 64 queries (4 waves x 16) of one head; 64-key tiles of K and V^T in LDS.
// Fragment layout (checked on gfx1201): A (16 x 16, M x K): lane l -> row l % 16, k = 8 (l / 16) + i;
// B (K x N): lane l -> column l % 16, k = 8 (l / 16) + i; C: lane l -> column l % 16, rows 8 (l / 16) + i.
// So a lane's 8 accumulator values share 8 query rows with the 15 other lanes of its half-wave: the row max / sum
// reduce over those 16 lanes, and the output's per-row rescale is the lane's own.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define OSS_WMMA 1
#endif
typedef _Float16 h8v __attribute__((ext_vector_type(8)));
typedef float f8v __attribute__((ext_vector_type(8)));
constexpr int kWQ = 64, kWK = 64, kWP = 8;  // queries a block, keys a tile, LDS row padding (halves)
__global__ void __launch_bounds__(128) k_attn_prompt_wmma(const float* __restrict__ qkv, const uint16_t* __restrict__ kc,
                                                          const uint16_t* __restrict__ vc,
                                                          const float* __restrict__ sinks, int pos0,
                                                          uint16_t* __restrict__ out, int nh, int nkv, int cap,
                                                          int window, int ntok) {
#if defined(OSS_WMMA)
    __shared__ _Float16 Ks[kWK][kHD + kWP];   // [key][dim]
    __shared__ _Float16 Vt[kHD][kWK + kWP];   // [dim][key]
    __shared__ _Float16 Ps[4][16][kWK + kWP]; // per wave: [query][key]
    const int h = blockIdx.y, kvh = h / (nh / nkv);
    const int wv = threadIdx.x / WS, lane = threadIdx.x % WS;
    const int lr = lane % 16, lh = lane / 16;
    const int q0 = blockIdx.x * kWQ + wv * 16;  // this wave's first query
    const int stride = (nh + 2 * nkv) * kHD;
    const float scale = rsqrtf((float) kHD);
    // Q fragments: 4 k-steps of 16 dims
    h8v qa[kHD / 16];
    {
        const int qi = min(q0 + lr, ntok - 1);
        const float* qp = qkv + (size_t) qi * stride + h * kHD;
#pragma unroll
        for (int ks = 0; ks < kHD / 16; ++ks)
#pragma unroll
            for (int i = 0; i < 8; ++i) qa[ks][i] = (_Float16) (qp[ks * 16 + 8 * lh + i] * scale);
    }
    // the 8 query rows this lane's accumulators hold
    int qpos[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qpos[i] = pos0 + min(q0 + 8 * lh + i, ntok - 1);
    f8v o[kHD / 16];
#pragma unroll
    for (int dt = 0; dt < kHD / 16; ++dt) o[dt] = f8v{};
    float m[8], l[8];
    const float sink = sinks[h];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = sink; l[i] = 1.f; }
    const int first_q = blockIdx.x * kWQ, last_q = min(ntok, first_q + kWQ) - 1;
    const int blo = window > 0 ? max(0, pos0 + first_q - window + 1) : 0;
    const int bhi = pos0 + last_q;
    const int wlo = window > 0 ? max(0, pos0 + min(q0, ntok - 1) - window + 1) : 0;
    const int whi = pos0 + min(q0 + 15, ntok - 1);
    for (int t0 = blo; t0 <= bhi; t0 += kWK) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < kWK * (kHD / 8); idx += blockDim.x) {
            const int j = idx / (kHD / 8), d8 = (idx % (kHD / 8)) * 8;
            const int t = t0 + j;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (t <= bhi) {
                const size_t slot = (size_t) (t % cap);
                kv = *(const uint4*) (kc + (slot * nkv + kvh) * kHD + d8);
                vv = *(const uint4*) (vc + (slot * nkv + kvh) * kHD + d8);
            }
            *(uint4*) &Ks[j][d8] = kv;
            const _Float16* vb = (const _Float16*) &vv;
#pragma unroll
            for (int u = 0; u < 8; ++u) Vt[d8 + u][j] = vb[u];
        }
        __syncthreads();
        if (t0 > whi || t0 + kWK - 1 < wlo) continue;  // nothing in this tile for this wave's queries
        // S = Q K^T: 4 tiles of 16 keys
        f8v sc[kWK / 16];
#pragma unroll
        for (int kt = 0; kt < kWK / 16; ++kt) {
            f8v acc = {};
#pragma unroll
            for (int ks = 0; ks < kHD / 16; ++ks) {
                const h8v kb = *(const h8v*) &Ks[kt * 16 + lr][ks * 16 + 8 * lh];
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[ks], kb, acc);
            }
            sc[kt] = acc;
        }
        // mask, row max over the tile (16 lanes of the half-wave), rescale
        float tmax[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) tmax[i] = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < kWK / 16; ++kt) {
            const int t = t0 + kt * 16 + lr;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int lo = window > 0 ? qpos[i] - window + 1 : 0;
                const bool ok = t <= qpos[i] && t >= lo && t <= bhi;
                sc[kt][i] = ok ? sc[kt][i] : -INFINITY;
                tmax[i] = fmaxf(tmax[i], sc[kt][i]);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) tmax[i] = fmaxf(tmax[i], __shfl_xor(tmax[i], off, WS));
        }
        float alpha[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float mn = fmaxf(m[i], tmax[i]);
            alpha[i] = __expf(m[i] - mn);
            m[i] = mn;
        }
        float rs[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) rs[i] = 0.f;
#pragma unroll
        for (int kt = 0; kt < kWK / 16; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float pe = sc[kt][i] == -INFINITY ? 0.f : __expf(sc[kt][i] - m[i]);
                rs[i] += pe;
                Ps[wv][8 * lh + i][kt * 16 + lr] = (_Float16) pe;
            }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) rs[i] += __shfl_xor(rs[i], off, WS);
            l[i] = l[i] * alpha[i] + rs[i];
        }
#pragma unroll
        for (int dt = 0; dt < kHD / 16; ++dt)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[dt][i] *= alpha[i];
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        // O += P V: P (16 x 64 keys) as A, V^T rows as B
#pragma unroll
        for (int kt = 0; kt < kWK / 16; ++kt) {
            const h8v pa = *(const h8v*) &Ps[wv][lr][kt * 16 + 8 * lh];
#pragma unroll
            for (int dt = 0; dt < kHD / 16; ++dt) {
                const h8v vb = *(const h8v*) &Vt[dt * 16 + lr][kt * 16 + 8 * lh];
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb, o[dt]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int qi = q0 + 8 * lh + i;
        if (qi >= ntok) continue;
        const float il = 1.f / l[i];
#pragma unroll
        for (int dt = 0; dt < kHD / 16; ++dt) out[(size_t) qi * nh * kHD + h * kHD + dt * 16 + lr] = f2h(o[dt][i] * il);
    }
#endif
}

// int8 WMMA versions of the prompt MoE kernels (gfx12): the same tiles (32 tokens of one expert), 4 waves a block.
// MATS = 2: gate and up (16 rows a wave each, act = swiglu written); MATS = 1: down (32 rows a wave, y written).
// Per 32-value block: two 16x16x16 int8 WMMAs into an int tile, then the block's two scales (weight row, token).
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
template <int MATS, int RSUB>
__global__ void __launch_bounds__(128) k_mmq_wmma(const uint8_t* const* __restrict__ wptr,
                                                  const int4* __restrict__ tiles, int tile0,
                                                  const int8_t* __restrict__ xq, const float* __restrict__ xd,
                                                  const float* __restrict__ b0p, const float* __restrict__ b1p,
                                                  float* __restrict__ outp, int n_in, int n_out, size_t mat_off,
                                                  size_t mat_stride) {
#if defined(OSS_WMMA)
    constexpr int KB = 4, WR = 16 * RSUB, TM = 4 * WR, RP = KB * 32 + 8;
    __shared__ int8_t W[MATS][TM][RP];
    __shared__ float Wd[MATS][TM][KB];
    __shared__ int8_t X[kTN][RP];
    __shared__ float Xd[kTN][KB];
    const int4 tl = tiles[tile0 + blockIdx.y];
    const int e = tl.x, r0 = tl.y, nt = tl.z;
    const int row0 = blockIdx.x * TM;
    const int nb = n_in / 32;
    const uint8_t* mat0 = wptr[e] + mat_off;
    const int wv = threadIdx.x / WS, lane = threadIdx.x % WS, lr = lane % 16, lh = lane / 16;
    float acc[MATS][RSUB][2][8];
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
        for (int rs = 0; rs < RSUB; ++rs)
#pragma unroll
            for (int ns = 0; ns < 2; ++ns)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[m][rs][ns][i] = 0.f;
    for (int b0 = 0; b0 < nb; b0 += KB) {
        for (int idx = threadIdx.x; idx < MATS * TM * KB; idx += blockDim.x) {
            const int m = idx / (TM * KB), rem = idx % (TM * KB), r = rem / KB, kb = rem % KB;
            const int row = row0 + r, b = b0 + kb;
            int* dst = (int*) &W[m][r][kb * 32];
            if (row < n_out && b < nb) {
                const uint8_t* base = mat0 + m * mat_stride;
                const uint4 qv = ((const uint4*) base)[(size_t) row * nb + b];
                const uint32_t q[4] = {qv.x, qv.y, qv.z, qv.w};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int2 v = mxfp4_lut(q[l]);
                    dst[l] = v.x;
                    dst[l + 4] = v.y;
                }
                Wd[m][r][kb] = e8m0_half(base[(size_t) n_out * nb * 16 + (size_t) row * nb + b]);
            } else {
#pragma unroll
                for (int l = 0; l < 8; ++l) dst[l] = 0;
                Wd[m][r][kb] = 0.f;
            }
        }
        for (int idx = threadIdx.x; idx < kTN * KB * 2; idx += blockDim.x) {
            const int tok = idx / (KB * 2), rem = idx % (KB * 2), kb = rem / 2, hh = rem % 2;
            const int b = b0 + kb;
            int4 v = make_int4(0, 0, 0, 0);
            if (tok < nt && b < nb) v = *(const int4*) (xq + (size_t) (r0 + tok) * n_in + b * 32 + hh * 16);
            *(int4*) &X[tok][kb * 32 + hh * 16] = v;
            if (hh == 0) Xd[tok][kb] = (tok < nt && b < nb) ? xd[(size_t) (r0 + tok) * nb + b] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            v2i bx[2][2];
#pragma unroll
            for (int ns = 0; ns < 2; ++ns) {
                bx[ns][0] = *(const v2i*) &X[ns * 16 + lr][kb * 32 + 8 * lh];
                bx[ns][1] = *(const v2i*) &X[ns * 16 + lr][kb * 32 + 16 + 8 * lh];
            }
#pragma unroll
            for (int m = 0; m < MATS; ++m)
#pragma unroll
                for (int rs = 0; rs < RSUB; ++rs) {
                    const int rb = wv * WR + rs * 16;
                    const v2i a0 = *(const v2i*) &W[m][rb + lr][kb * 32 + 8 * lh];
                    const v2i a1 = *(const v2i*) &W[m][rb + lr][kb * 32 + 16 + 8 * lh];
#pragma unroll
                    for (int ns = 0; ns < 2; ++ns) {
                        v8i ci = {};
                        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, bx[ns][0], ci, false);
                        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, bx[ns][1], ci, false);
                        const float xs = Xd[ns * 16 + lr][kb];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[m][rs][ns][i] += Wd[m][rb + 8 * lh + i][kb] * xs * (float) ci[i];
                    }
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rs = 0; rs < RSUB; ++rs)
#pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            const int tok = ns * 16 + lr;
            if (tok >= nt) continue;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + wv * WR + rs * 16 + 8 * lh + i;
                if (row >= n_out) continue;
                if (MATS == 2) {
                    const float g = acc[0][rs][ns][i] + b0p[(size_t) e * n_out + row];
                    const float u = acc[MATS - 1][rs][ns][i] + b1p[(size_t) e * n_out + row];
                    outp[(size_t) (r0 + tok) * n_out + row] = swiglu_oai(g, u);
                } else {
                    outp[(size_t) (r0 + tok) * n_out + row] = acc[0][rs][ns][i];
                }
            }
        }
#endif
}

bool use_mmq_wmma() {
    static const bool v = [] {
        int dev = 0;
        hipDeviceProp_t pr;
        (void) hipGetDevice(&dev);
        (void) hipGetDeviceProperties(&pr, dev);
        const char* env = std::getenv("OSS_MMQ_WMMA");
        return (env ? std::atoi(env) != 0 : true) && std::strncmp(pr.gcnArchName, "gfx120", 6) == 0;
    }();
    return v;
}

unsigned cdiv(size_t a, size_t b) { return (unsigned) ((a + b - 1) / b); }

} // namespace

// ================================================================================================ launchers
void rmsnorm_quant(const float* x, const float* w, float* y, int8_t* q, float* d, int n, int ncols, float eps,
                   hipStream_t s) {
    hipLaunchKernelGGL(k_rmsnorm_quant, dim3(ncols), dim3(256), 0, s, x, w, y, q, d, n, eps);
}

void rmsnorm(const float* x, const float* w, float* y, int n, int ncols, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_rmsnorm_quant, dim3(ncols), dim3(256), 0, s, x, w, y, (int8_t*) nullptr, (float*) nullptr,
                       n, eps);
}

void quant_q8(const float* x, int8_t* q, float* d, int n, int nvec, hipStream_t s) {
    const size_t blocks = (size_t) (n / 32) * nvec;
    hipLaunchKernelGGL(k_quant_q8, dim3(cdiv(blocks, 8)), dim3(256), 0, s, x, q, d, n, nvec);
}

void mv_q8_0(const int8_t* wq, const uint16_t* wd, const int8_t* xq, const float* xd, const float* bias, float* y,
             int n_in, int n_out, int ncols, bool accumulate, hipStream_t s) {
    const dim3 g(cdiv(n_out, 8)), b(256);
    switch (ncols) {
#define C(N) case N: hipLaunchKernelGGL(k_mv_q8_0<N>, g, b, 0, s, wq, wd, xq, xd, bias, y, n_in, n_out, accumulate); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: std::fprintf(stderr, "mv_q8_0: ncols %d\n", ncols); std::abort();
    }
}

void mv_f32(const float* w, const float* x, const float* bias, float* y, int n_in, int n_out, int ncols, hipStream_t s) {
    const dim3 g(cdiv(n_out, 8)), b(256);
    switch (ncols) {
#define C(N) case N: hipLaunchKernelGGL(k_mv_f32<N>, g, b, 0, s, w, x, bias, y, n_in, n_out); break;
        C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8)
#undef C
        default: std::fprintf(stderr, "mv_f32: ncols %d\n", ncols); std::abort();
    }
}

void rope_store(float* qkv, uint16_t* kc, uint16_t* vc, const StepParams* p, int nh, int nkv, int hd, int cap,
                RopeParams rp, int ncols, hipStream_t s) {
    hipLaunchKernelGGL(k_rope_store, dim3(nh + nkv, ncols), dim3(hd / 2), 0, s, qkv, kc, vc, p, 0, nh, nkv, hd, cap, rp);
}

void rope_store_prompt(float* qkv, uint16_t* kc, uint16_t* vc, int pos0, int nh, int nkv, int hd, int cap,
                       RopeParams rp, int ntok, hipStream_t s) {
    hipLaunchKernelGGL(k_rope_store, dim3(nh + nkv, ntok), dim3(hd / 2), 0, s, qkv, kc, vc, (const StepParams*) nullptr,
                       pos0, nh, nkv, hd, cap, rp);
}

size_t attn_part_floats(int nh, int max_ctx, int ncols) {
    const int ms = (max_ctx + kSplit - 1) / kSplit;
    return (size_t) ncols * nh * ms * (kHD + 2);
}

void attn_decode(const float* qkv, const uint16_t* kc, const uint16_t* vc, const float* sinks, const StepParams* p,
                 float* part, float* out, int nh, int nkv, int hd, int cap, int window, int max_ctx, int ncols,
                 hipStream_t s) {
    (void) hd;
    const int span = window > 0 ? window : max_ctx;
    const int ms_layer = (span + kSplit - 1) / kSplit;
    const int ms = (max_ctx + kSplit - 1) / kSplit;  // the part buffer's stride
    hipLaunchKernelGGL(k_attn_split, dim3(nkv, ms_layer, ncols), dim3((nh / nkv) * WS), 0, s, qkv, kc, vc, p, part, nh,
                       nkv, cap, window, ms);
    hipLaunchKernelGGL(k_attn_combine, dim3(nh, ncols), dim3(kHD), 0, s, part, sinks, p, out, nh, window, ms);
}

void router_topk(const float* logits, int n_expert, int ncols, int32_t* topk, float* topw, hipStream_t s) {
    hipLaunchKernelGGL(k_router_topk, dim3(ncols), dim3(WS), 0, s, logits, n_expert, topk, topw);
}

void publish(LayerMailbox* mb, const StepParams* p, const int32_t* topk, const float* topw, const int8_t* hq,
             const float* hd, int n, int ncols, hipStream_t s) {
    (void) ncols;
    hipLaunchKernelGGL(k_publish, dim3(1), dim3(256), 0, s, mb, p, topk, topw, hq, hd, n);
}

void moe_gate_up(const uint8_t* arena, size_t blob_bytes, const int32_t* slot_of, const int32_t* topk,
                 const int8_t* hq, const float* hd, const float* gate_b, const float* up_b, float* act, int n_embd,
                 int n_ff, int ncols, hipStream_t s) {
    hipLaunchKernelGGL(k_moe_gate_up, dim3(cdiv(n_ff, 8), ncols * kTopK), dim3(256), 0, s, arena, blob_bytes, slot_of,
                       topk, hq, hd, gate_b, up_b, act, n_embd, n_ff);
}

void moe_down(const uint8_t* arena, size_t blob_bytes, const int32_t* slot_of, const int32_t* topk,
              const float* topw, const int8_t* aq, const float* ad, const float* down_b, float* y, int n_embd,
              int n_ff, int ncols, hipStream_t s) {
    hipLaunchKernelGGL(k_moe_down, dim3(cdiv(n_embd, 8), ncols), dim3(256), 0, s, arena, blob_bytes, slot_of, topk,
                       topw, aq, ad, down_b, y, n_embd, n_ff);
}

void wait_ack(LayerMailbox* mb, const StepParams* p, hipStream_t s) {
    hipLaunchKernelGGL(k_wait_ack, dim3(1), dim3(1), 0, s, mb, p);
}

void moe_residual(float* x, const float* y_gpu, const LayerMailbox* mb, int n, int ncols, hipStream_t s) {
    hipLaunchKernelGGL(k_moe_residual, dim3(cdiv(n, 256), ncols), dim3(256), 0, s, x, y_gpu, mb, n);
}

void handoff_send(Handoff* h, const StepParams* p, const float* x, int n, int ncols, hipStream_t s) {
    (void) ncols;
    hipLaunchKernelGGL(k_handoff_send, dim3(1), dim3(256), 0, s, h, p, x, n);
}

void handoff_recv(const Handoff* h, const StepParams* p, float* x, int n, int ncols, hipStream_t s) {
    (void) ncols;
    hipLaunchKernelGGL(k_handoff_recv, dim3(1), dim3(256), 0, s, h, p, x, n);
}

void dequant_q8_0_f16(const int8_t* wq, const uint16_t* wd, uint16_t* out, int n_in, int n_out, hipStream_t s) {
    const size_t total = (size_t) n_in * n_out;
    hipLaunchKernelGGL(k_dequant_q8_0, dim3(cdiv(total, 256)), dim3(256), 0, s, wq, wd, out, n_in, total);
}

void dequant_mxfp4_f16(const uint8_t* m, uint16_t* out, int n_in, int rows, hipStream_t s) {
    const size_t blocks = (size_t) rows * (n_in / 32);
    hipLaunchKernelGGL(k_dequant_mxfp4, dim3(cdiv(blocks, 256)), dim3(256), 0, s, m, out, n_in, rows);
}

void f32_to_f16(const float* x, uint16_t* y, size_t n, hipStream_t s) {
    hipLaunchKernelGGL(k_f32_to_f16, dim3(cdiv(n, 256)), dim3(256), 0, s, x, y, n);
}

void rmsnorm_f16(const float* x, const float* w, float* y32, uint16_t* y16, int n, int ntok, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_rmsnorm_f16, dim3(ntok), dim3(256), 0, s, x, w, y32, y16, n, eps);
}

void add_bias(float* y, const float* bias, int n, int ntok, hipStream_t s) {
    const size_t total = (size_t) n * ntok;
    hipLaunchKernelGGL(k_add_bias, dim3(cdiv(total, 256)), dim3(256), 0, s, y, bias, n, total);
}

void add_bias_residual(float* x, const float* y, const float* bias, int n, int ntok, hipStream_t s) {
    const size_t total = (size_t) n * ntok;
    hipLaunchKernelGGL(k_add_bias_residual, dim3(cdiv(total, 256)), dim3(256), 0, s, x, y, bias, n, total);
}

void attn_prompt(const float* qkv, const uint16_t* kc, const uint16_t* vc, const float* sinks, int pos0,
                 uint16_t* out16, int nh, int nkv, int hd, int cap, int window, int ntok, hipStream_t s) {
    (void) hd;
    static const bool wmma = [] {
        int dev = 0;
        hipDeviceProp_t pr;
        (void) hipGetDevice(&dev);
        (void) hipGetDeviceProperties(&pr, dev);
        const char* env = std::getenv("OSS_ATTN_WMMA");
        return (env ? std::atoi(env) != 0 : true) && std::strncmp(pr.gcnArchName, "gfx120", 6) == 0;
    }();
    if (wmma)
        hipLaunchKernelGGL(k_attn_prompt_wmma, dim3(cdiv(ntok, kWQ), nh), dim3(128), 0, s, qkv, kc, vc, sinks, pos0,
                           out16, nh, nkv, cap, window, ntok);
    else
        hipLaunchKernelGGL(k_attn_prompt, dim3(cdiv(ntok, kQT), nh), dim3(kQT), 0, s, qkv, kc, vc, sinks, pos0, out16,
                           nh, nkv, cap, window, ntok);
}

void gather_rows_f16(const uint16_t* src, const int32_t* rows, uint16_t* dst, int n, int count, hipStream_t s) {
    if (count <= 0) return;
    hipLaunchKernelGGL(k_gather_rows_f16, dim3(count), dim3(128), 0, s, src, rows, dst, n);
}

void swiglu_bias_f16(const float* gu, const float* gate_b, const float* up_b, uint16_t* out, int n_ff, int count,
                     hipStream_t s) {
    const size_t total = (size_t) n_ff * count;
    hipLaunchKernelGGL(k_swiglu_bias_f16, dim3(cdiv(total, 256)), dim3(256), 0, s, gu, gate_b, up_b, out, n_ff, total);
}

void moe_combine(float* x, const float* ys, const int32_t* inv, const int32_t* topk, const float* topw,
                 const float* down_b, int n, int ntok, hipStream_t s) {
    hipLaunchKernelGGL(k_moe_combine, dim3(cdiv(n, 256), ntok), dim3(256), 0, s, x, ys, inv, topk, topw, down_b, n);
}

void mmq_gate_up(const uint8_t* const* wptr, const int32_t* tiles, int tile0, int ntiles, const int8_t* xq,
                 const float* xd, const float* gate_b, const float* up_b, float* act, int n_embd, int n_ff,
                 hipStream_t s) {
    if (ntiles <= 0) return;
    if (use_mmq_wmma()) {
        const size_t mat = (size_t) n_ff * (n_embd / 32) * 17;
        hipLaunchKernelGGL((k_mmq_wmma<2, 1>), dim3(cdiv(n_ff, 64), ntiles), dim3(128), 0, s, wptr, (const int4*) tiles,
                           tile0, xq, xd, gate_b, up_b, act, n_embd, n_ff, (size_t) 0, mat);
        return;
    }
    hipLaunchKernelGGL(k_mmq_gate_up, dim3(cdiv(n_ff, 64), ntiles), dim3(256), 0, s, wptr, (const int4*) tiles, tile0,
                       xq, xd, gate_b, up_b, act, n_embd, n_ff);
}

void mmq_down(const uint8_t* const* wptr, const int32_t* tiles, int tile0, int ntiles, const int8_t* aq,
              const float* ad, float* y, int n_embd, int n_ff, hipStream_t s) {
    if (ntiles <= 0) return;
    if (use_mmq_wmma()) {
        const size_t gu = 2 * (size_t) n_ff * (n_embd / 32) * 17;
        hipLaunchKernelGGL((k_mmq_wmma<1, 2>), dim3(cdiv(n_embd, 128), ntiles), dim3(128), 0, s, wptr,
                           (const int4*) tiles, tile0, aq, ad, (const float*) nullptr, (const float*) nullptr, y, n_ff,
                           n_embd, gu, (size_t) 0);
        return;
    }
    hipLaunchKernelGGL(k_mmq_down, dim3(cdiv(n_embd, 128), ntiles), dim3(256), 0, s, wptr, (const int4*) tiles, tile0,
                       aq, ad, y, n_embd, n_ff);
}

void gather_rows_q8(const int8_t* q, const float* d, const int32_t* rows, int8_t* oq, float* od, int n, int count,
                    hipStream_t s) {
    if (count <= 0) return;
    hipLaunchKernelGGL(k_gather_rows_q8, dim3(count), dim3(128), 0, s, q, d, rows, oq, od, n);
}

} // namespace oss
