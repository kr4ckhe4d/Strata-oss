// oss/oss_kernels.hpp - the GPU half of the GPT-OSS engine (HIP).
//
// Strata's mechanism, applied to gpt-oss: every dense weight lives on the GPU that owns its layer, the most-used
// experts fill the rest of that GPU's VRAM ("slots"), and the experts a token routes to that are not in a slot are
// computed by the CPU from RAM at the same time as the GPU computes the cached ones.  The GPU decides which experts
// it computes itself (it reads the routing and the slot table on the device), so the host never has to launch work
// in the middle of a layer: it only answers "here is the CPU's share" through a flag in page-locked memory.
//
// Formats (all fixed at load time, see oss_model.hpp):
//   * activations for the int8 dot products: int8 q[n] + float d[n / 32] per column (one scale per 32 values, the
//     Q8_1 rounding of ggml without the block sum)
//   * Q8_0 weights repacked to two planes: int8 qs[rows][n_in] then half d[rows][n_in / 32]
//   * MXFP4 weights repacked to two planes: uint8 qs[rows][n_in / 32][16] then uint8 e[rows][n_in / 32]
//     (ggml's block_mxfp4 is the same 17 bytes; the planes make every 16-byte load aligned)
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

namespace oss {

constexpr int kMaxCols = 8;     // decode / verify columns per step
constexpr int kTopK = 4;        // experts per token (checked against the GGUF at load)

/// What the host writes before each decode step (copied to the device at the start of the step).
struct StepParams {
    int ncols = 1;
    uint32_t seq = 0;            // step counter, the flags below compare against it
    int pos[kMaxCols] = {};
};

/// One layer's mailbox between a GPU and the host, in page-locked (coherent) host memory.
/// The GPU writes the routing and the quantized FFN input, then `routed = seq`; the host computes the experts that
/// are not in VRAM, writes `y_cpu`, `has_cpu`, then `acked = seq`; the GPU waits for it before the residual add.
struct LayerMailbox {
    alignas(64) volatile uint32_t routed;
    alignas(64) volatile uint32_t acked;
    alignas(64) int32_t ncols;
    int32_t has_cpu[kMaxCols];
    int32_t topk[kMaxCols][kTopK];
    float topw[kMaxCols][kTopK];
    float hd[kMaxCols][2880 / 32];
    alignas(64) int8_t hq[kMaxCols][2880];
    alignas(64) float y_cpu[kMaxCols][2880];
};

/// Hand-over of the residual between two GPUs (layer split), also in coherent host memory.
struct Handoff {
    alignas(64) volatile uint32_t seq;
    alignas(64) float x[kMaxCols][2880];
};

struct RopeParams {
    float freq_base, freq_scale, ext_factor, attn_factor, corr0, corr1;
    int n_dims;
    int neox = 1;  // 1: pairs (i, i + n_dims / 2) as gpt-oss; 0: pairs (2i, 2i + 1) as llama (the EAGLE3 draft)
};

// ---- elementwise / norms
void rmsnorm_quant(const float* x, const float* w, float* y, int8_t* q, float* d, int n, int ncols, float eps,
                   hipStream_t s);
void quant_q8(const float* x, int8_t* q, float* d, int n, int nvec, hipStream_t s);

// ---- Q8_0 matrix x int8 activations (decode, 1..8 columns). y[c][row] = W[row] . x[c] + bias[row] (+ y if accumulate)
void mv_q8_0(const int8_t* wq, const uint16_t* wd, const int8_t* xq, const float* xd, const float* bias, float* y,
             int n_in, int n_out, int ncols, bool accumulate, hipStream_t s);

// ---- f32 matrix (the router): logits[c][e] = W[e] . x[c] + b[e]
void mv_f32(const float* w, const float* x, const float* bias, float* y, int n_in, int n_out, int ncols, hipStream_t s);

// ---- decode attention block pieces
// qkv[c] = [q (nh*hd) | k (nkv*hd) | v (nkv*hd)]; ropes q and k in place and stores k, v (f16) at slot pos % cap
void rope_store(float* qkv, uint16_t* kc, uint16_t* vc, const StepParams* p, int nh, int nkv, int hd, int cap,
                RopeParams rp, int ncols, hipStream_t s);
// attention over the cache for the decode columns; window = 0 for full attention.  part = scratch from
// attn_part_floats(), out[c] = nh*hd floats
size_t attn_part_floats(int nh, int max_ctx, int ncols);
void attn_decode(const float* qkv, const uint16_t* kc, const uint16_t* vc, const float* sinks, const StepParams* p,
                 float* part, float* out, int nh, int nkv, int hd, int cap, int window, int max_ctx, int ncols,
                 hipStream_t s);

// ---- MoE (decode).  topk/topw: device copies [ncols][K]; slot_of: [n_expert] for this layer (-1 = not in VRAM)
void router_topk(const float* logits, int n_expert, int ncols, int32_t* topk, float* topw, hipStream_t s);
// writes topk/topw/hq/hd into the mailbox, then routed = seq (system-scope release)
void publish(LayerMailbox* mb, const StepParams* p, const int32_t* topk, const float* topw, const int8_t* hq,
             const float* hd, int n, int ncols, hipStream_t s);
void moe_gate_up(const uint8_t* arena, size_t blob_bytes, const int32_t* slot_of, const int32_t* topk,
                 const int8_t* hq, const float* hd, const float* gate_b, const float* up_b, float* act, int n_embd,
                 int n_ff, int ncols, hipStream_t s);
void moe_down(const uint8_t* arena, size_t blob_bytes, const int32_t* slot_of, const int32_t* topk,
              const float* topw, const int8_t* aq, const float* ad, const float* down_b, float* y, int n_embd,
              int n_ff, int ncols, hipStream_t s);
// waits for acked == seq, then x[c] += y_gpu[c] + (has_cpu ? y_cpu[c] : 0)
void wait_ack(LayerMailbox* mb, const StepParams* p, hipStream_t s);
void moe_residual(float* x, const float* y_gpu, const LayerMailbox* mb, int n, int ncols, hipStream_t s);

// ---- layer-split hand-over
void handoff_send(Handoff* h, const StepParams* p, const float* x, int n, int ncols, hipStream_t s);
void handoff_recv(const Handoff* h, const StepParams* p, float* x, int n, int ncols, hipStream_t s);

// ---- head
void rmsnorm(const float* x, const float* w, float* y, int n, int ncols, float eps, hipStream_t s);

// ---- prompt path (many tokens): GEMMs through hipBLAS on f16 copies, attention tiled
void dequant_q8_0_f16(const int8_t* wq, const uint16_t* wd, uint16_t* out, int n_in, int n_out, hipStream_t s);
void dequant_mxfp4_f16(const uint8_t* m, uint16_t* out, int n_in, int rows, hipStream_t s);
void f32_to_f16(const float* x, uint16_t* y, size_t n, hipStream_t s);
void rmsnorm_f16(const float* x, const float* w, float* y32, uint16_t* y16, int n, int ntok, float eps, hipStream_t s);
void add_bias(float* y, const float* bias, int n, int ntok, hipStream_t s);
void add_bias_residual(float* x, const float* y, const float* bias, int n, int ntok, hipStream_t s);
void rope_store_prompt(float* qkv, uint16_t* kc, uint16_t* vc, int pos0, int nh, int nkv, int hd, int cap,
                       RopeParams rp, int ntok, hipStream_t s);
void attn_prompt(const float* qkv, const uint16_t* kc, const uint16_t* vc, const float* sinks, int pos0,
                 uint16_t* out16, int nh, int nkv, int hd, int cap, int window, int ntok, hipStream_t s);
void gather_rows_f16(const uint16_t* src, const int32_t* rows, uint16_t* dst, int n, int count, hipStream_t s);
void swiglu_bias_f16(const float* gu, const float* gate_b, const float* up_b, uint16_t* out, int n_ff, int count,
                     hipStream_t s);
// x[t] += sum_k w[t][k] * (ys[inv[t][k]] + down_b[e]) over the token's experts
void moe_combine(float* x, const float* ys, const int32_t* inv, const int32_t* topk, const float* topw,
                 const float* down_b, int n, int ntok, hipStream_t s);

// grouped prompt MoE: tiles = int4 (expert, first sorted row, rows <= 32, 0); wptr[e] = the expert's blob on the device
void mmq_gate_up(const uint8_t* const* wptr, const int32_t* tiles, int tile0, int ntiles, const int8_t* xq,
                 const float* xd, const float* gate_b, const float* up_b, float* act, int n_embd, int n_ff,
                 hipStream_t s);
void mmq_down(const uint8_t* const* wptr, const int32_t* tiles, int tile0, int ntiles, const int8_t* aq,
              const float* ad, float* y, int n_embd, int n_ff, hipStream_t s);
void gather_rows_q8(const int8_t* q, const float* d, const int32_t* rows, int8_t* oq, float* od, int n, int count,
                    hipStream_t s);

// ---- the EAGLE3 draft (decode-style, 1..8 rows)
// y[c][0..n) at row stride ldy = x[c] (row stride ldx) / rms * w
void rmsnorm_strided(const float* x, int ldx, const float* w, float* y, int ldy, int n, int ncols, float eps,
                     hipStream_t s);
// act[c][i] = silu(gu[c][i]) * gu[c][n + i]
void silu_mul(const float* gu, float* act, int n, int ncols, hipStream_t s);
// out[0] = map[argmax(logits[0..n))], out[1] = its softmax probability (float bits)
void argmax_map(const float* logits, int n, const int32_t* map, int32_t* out, hipStream_t s);

} // namespace oss
