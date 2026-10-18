# strata-oss: findings log

GPT-OSS on Strata's mechanism. What was measured, on what, and why each bottleneck was there. Newest at the end.

**Test PC (all numbers):** Ryzen 7 9800X3D (8 cores, AVX-512), 30 GB DDR5, RX 9070 XT 16 GB (gfx1201) + Radeon AI PRO
R9700 32 GB (gfx1201), Linux (CachyOS), ROCm 7.2. Models: ggml-org `gpt-oss-120b-MXFP4.gguf` (63.4 GB) and
`gpt-oss-20b-mxfp4.gguf`. llama.cpp for reference: the local build at `~/llama.cpp` (5e03bdd87, HIP).

## 1. Why the Qwen engine could not just load gpt-oss

- The Qwen engine (`src/`) is written around Qwen3.8-Flash-Next: DeltaNet (GDN) layers, QSA attention, the n-gram (PLE)
  table, gated residuals, MTP. `gguf_reader.hpp` refuses other architectures. None of that exists in gpt-oss.
- gpt-oss is simpler: 36 layers (20b: 24), all attention; even layers have a 128-token sliding window, odd layers full
  attention; attention sinks (one learned logit per head in the softmax denominator); YaRN RoPE (factor 32, NeoX
  pairing); biases on q/k/v/o, router and experts; 128 experts (20b: 32), top-4, softmax over the 4 picks; SwiGLU with
  clamp (alpha 1.702, limit 7, `(up + 1)`); experts MXFP4, everything dense Q8_0, router F32.
- So the port is a new engine (`oss/`) that reuses the mechanism and some code (GGUF reader, tokenizer, ggml's int8 dot
  scheme), not the Qwen forward pass. The Qwen build is untouched (separate CMake project).

## 2. Memory plan (120b)

- One expert (gate + up + down, MXFP4) = 12.6 MiB; 4,608 experts = 58 GiB. Dense weights ~1 GiB, head 0.6 GiB.
- 48 GB of VRAM holds 3,331-3,365 experts (72-73%); the other ~1,250 (15.3-15.7 GiB) sit in page-locked RAM.
  Engine RSS 20 GB. Load time ~15 s (NVMe, file pages dropped per layer so the page cache does not fight the arena).
- Layer split by free VRAM: GPU 0 (16 GB) layers 0-11, GPU 1 (32 GB) layers 12-35 + head.

## 3. Correctness (gpt-oss-20b, 493-token harmony prompt, teacher-forced vs llama.cpp)

- Tokenizer: identical ids to `llama-tokenize` (o200k pattern = llama.cpp's GPT4O pre-tokenizer).
- Top-1 agreement 90.9%, mean KL 0.038 vs llama.cpp batched. Looks low, but **this model is very rounding-sensitive**:
  llama.cpp's own batched vs one-token-at-a-time logits agree only 92.7% (KL 0.026). MXFP4 + top-4 routing turns
  tiny differences into different expert picks.
- Better test: perplexity of the actual next tokens. llama.cpp 291.7 / 303.7 (batched / one at a time), strata-oss
  291.3 (prompt path), 288.0 (decode path), 296.0 (grouped int8 prompt kernels). Same band: no math bug.
  (gpt-oss's raw-text perplexity is high in general; it is a chat model.)
- `llama_logits ... 0` (n_gpu_layers 0) still ran on the GPU (op offload), so it is not a CPU noise floor.

## 4. Decode (output) speed

| run | output tok/s | experts hit in VRAM |
| --- | ---: | ---: |
| 20b, all experts in VRAM (R9700) | 136 | 100% |
| 20b, half in RAM, no adaptation | 66 | 49.5% |
| 20b, half in RAM, adaptive swaps | 86 | 72.8% (120 swaps) |
| 120b, 2 GPUs, no profile (first start) | 60.4 | 77.0% |
| 120b, started from the saved profile | 78-82 | - |
| llama.cpp 120b, best fit (`-ncmoe 10 -ts 17/19 -fa on -lm none`) | 50.5 | (whole layers on CPU) |

- **Why Strata's way wins on output:** llama.cpp puts *whole layers'* experts on the CPU (`--n-cpu-moe`), so every
  token's top-4 in those 10 layers runs on the CPU. Strata keeps the *most-routed* experts of every layer in VRAM
  (routing is skewed), so ~77%+ of routings hit VRAM and the CPU only computes the rest.
- The profile matters: a first start fills VRAM with an arbitrary per-layer set and only the adaptive swaps fix it
  (60 tok/s); starting from the profile the last run saved gives 78-82.
- GPU<->CPU hand-off per layer: the GPU writes the routing + int8 FFN input into page-locked host memory and sets a
  flag; the host computes the missed experts (AVX-512 VNNI MXFP4 x int8 dot, 8 workers) and sets an ack flag the GPU
  spins on. Each GPU's whole step is one HIP graph, so the host never launches work mid-layer.
- 20b GPU and CPU paths give the same tokens for ~100 tokens, then an equivalent answer (they round sums in a
  different order, as Strata documents for the Qwen models).

## 5. Prompt speed

| change | 120b, 1,005-token prompt | experts ms / layer |
| --- | ---: | ---: |
| first version: per expert dequant to f16 + hipBLAS GEMM | 415 tok/s | 55-70 |
| grouped int8 kernels (one launch per group of experts) | **1,337 tok/s** | 9-16 |
| llama.cpp, 512-token prompt (`llama-bench pp512`) | 791 tok/s | - |

- **The bottleneck in the first version was per-expert overhead, not PCIe.** Only 13-32 experts per layer were missing
  from VRAM and streamed; but each of the ~100 routed experts did 3 dequant kernels (50 MB of f16 writes) and 2 small
  hipBLAS GEMMs (m ~ 40 tokens), ~0.6 ms each.
- Fix: `k_mmq_gate_up` / `k_mmq_down` read MXFP4 directly (decoded to int8 in LDS with ggml's byte-permute table) and
  multiply by int8 activations (one scale per 32), 32-token x 64/128-row tiles, all experts of a group in one launch.
  The VRAM experts run first while the missing ones stream over PCIe into a 16-slot ring (batches of 8).
- Now attention + router (~8.5 ms / layer at 1K tokens) is about half of the prompt time: next target.
- Prompt attention kernel (`k_attn_prompt`) uses 207 VGPRs and spills 272 bytes/lane; fine at 1K, will matter at 32K.

## 6. Baseline gotchas

- `llama-bench` with mmap on this 30 GB PC hung for 28 min: the CPU-side experts kept getting evicted and re-read from
  the SSD (~200 MB/s steady reads). `-lm none` (no mmap; the flag was `-mmp` in older builds) fixed it.
- `--n-cpu-moe` takes the *first* N layers; with `-ts` those must land on GPU 0's share, or GPU 1 runs out of memory.

## 7. Where the rest of the prompt time goes (120b, 1,005 tokens, 750 ms total)

`OSS_PROF=1` totals over the 36 layers (synchronizing after each piece): experts ~420 ms, q/k/v projection 133 ms,
output projection 87 ms, attention 72 ms, router 9 ms.

- **The dense projections are bound by hipBLAS on gfx1201, not the card.** Standalone benchmark, R9700, the q/k/v
  shape (5120 x N x 2880): `hipblasGemmEx` f16 in, f32 or f16 out, 10-15 TFLOPS at N = 1005..4096 (one f16-compute
  case 26 TFLOPS at N = 4096). Padding N to 1024 changes nothing; weights stored transposed (opA = N) are ~25% faster
  than opA = T. RDNA4's f16 / int8 WMMA peak is several times higher, so rocBLAS's gfx1201 kernels are untuned for
  these shapes.
- Next lever when prompt speed matters again: own WMMA kernels (`__builtin_amdgcn_wmma_*_gfx12`) for the dense
  projections and for the expert tiles (int8 WMMA instead of dp4a), and a WMMA attention for long prompts.

## 8. The server (oss/server.py) on 120b

Standard library only; `strata-oss --serve` underneath (JSON lines). Checked with curl:
- plain chat (`reasoning_content` from the analysis channel, `content` from final), streaming SSE, `/v1/completions`;
- a client disconnect cancels the answer (the engine stops generating);
- tool calls: the model's `to=functions.get_weather` message comes back as `tool_calls`, `finish_reason: tool_calls`;
  sending the tool result back gives the final answer (137 of 188 prompt tokens reused from the cache);
- prefix reuse across turns (`usage.prompt_tokens_details.cached_tokens`).

| prompt | prompt tok/s | output tok/s | experts hit in VRAM |
| --- | ---: | ---: | ---: |
| ~80 tokens (chat turns) | 235-265 | 72-81 | 89-92% |
| 12,864 tokens (5 chunks of 2,048 + rest) | 913 | 67-70 | 95-97% |

- Long prompts slow down because `k_attn_prompt` (a thread per query, 207 VGPRs, spills) grows with the context; the
  full-attention layers dominate there. Output at 13K context is 67 tok/s vs ~80 at short context: the decode
  attention reduces each key across a wave (5 shuffles a key). Both are the next kernels to rewrite.
- Short prompts are dominated by fixed per-layer costs (the router sync, dequant of the dense weights to f16 for
  hipBLAS): 80 tokens take 0.3-0.4 s.

## 9. Decode attention rewrite (long context)

Before: a wave per query head walked the keys one at a time, each key's 64-dim dot reduced across the wave with 5
shuffles, with an online-softmax rescale per key. At 32K context that cost ~0.4 ms per full-attention layer
(output 80 -> 52 tok/s from short to 32K).
After: the split's K and V (128 keys) are staged once in LDS per kv head (8 query heads share them; rows padded by
2 halves so a lane-per-key read does not hit one bank), a lane scores its own keys, one max/sum per split, the
probabilities go through LDS for the V sum.

| 120b context | output before | output after |
| --- | ---: | ---: |
| 16K | 65.4 | 81.5 |
| 32K | 51.8 | 70.8 |

Also closer to llama.cpp's one-token path on 20b: KL 0.054 -> 0.029, top-1 90.5% -> 91.7%; the per-key online
rescale was the noisier formulation (the KL 5.6 outlier at position 160 is gone).
Prompt speed at long context is unchanged (16K 857, 32K 570 tok/s): that is `k_attn_prompt`, next.

## 10. WMMA prompt attention (gfx12)

- gfx12 WMMA fragment layout, verified with a 16x16 test against a CPU matmul (exact): A (M x K): lane l holds row
  l % 16, k = 8 (l / 16) + i; B (K x N): column l % 16, k = 8 (l / 16) + i; C: column l % 16, rows 8 (l / 16) + i.
  (gfx11's layout is different, it duplicates halves; this kernel is gfx120x only, `OSS_ATTN_WMMA=0` = the old one.)
- The C layout makes flash attention easy: a lane's 8 accumulators are 8 query rows shared with the 15 other lanes of
  its half-wave, so row max / sum are 16-lane reductions and the output rescale is lane-local. P goes through LDS
  (C layout -> A layout); V is stored transposed in LDS so each B fragment is one 16-byte read.
- The old kernel (a thread per query, 207 VGPRs, scratch spills) vs WMMA on 20b: KL 0.012 between them, perplexity
  on a 2,000-token document 23.2 vs 23.5, top-1 45.3% both (f16 rounding of Q and P).

| 120b prompt | before | WMMA |
| --- | ---: | ---: |
| 16K tokens | 857 tok/s | 1,549 tok/s |
| 32K tokens | 570 tok/s | 1,412 tok/s |
