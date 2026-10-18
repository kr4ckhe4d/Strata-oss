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

## 11. int8 WMMA for the prompt's experts, and the PCIe wall

- int8 WMMA (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`) has the same fragment layout as the f16 one (checked
  exact). The expert kernels now do two 16x16x16 int8 WMMAs per 32-value block into an int tile, then apply the
  block's two scales (weight row's E8M0, token's int8 scale) in float. Same tiles as the dp4a kernels
  (`OSS_MMQ_WMMA=0` keeps those). 20b, 2,000 tokens, half the experts in RAM: KL 0.008 vs dp4a, perplexity 23.43 vs
  23.48.
- 120b 16K prompt: experts 5.85 s -> 4.49 s, prompt 1,544 -> 1,782 tok/s. Less than the WMMA speed-up because
  **PCIe is now the wall**: both GPUs sit on PCIe 5.0 x8 CPU links (x8/x8 split; idle reads 2.5 GT/s, power saving),
  measured 28.7 GB/s pinned host -> GPU each. A 16K prompt in 2,048-token chunks streams 7,660 expert copies
  (each missing expert once per chunk per layer) = 96 GB = ~3.4 s of pure transfer.
- So the lever is the chunk: each missing expert crosses PCIe once per chunk, whatever the chunk's size.
- **Gotcha (mine): `/tmp` is a 16 GB tmpfs (RAM) on this PC, and swap is zram (compressed RAM).** 13 logits dumps
  (9.4 GB) in the scratchpad pushed the engine into swap: a 4K-chunk run crawled for 10 min with RAM at 0 free.
  Reference dumps now live on the SSD (`/mnt/fast/models/oss-test/`). With 30 GB of RAM and a 16 GB expert arena,
  anything else large in RAM costs real speed.

## 12. Prompt chunk size (120b, after the WMMA kernels)

| chunk | VRAM slots | 16K prompt | 32K prompt | output 16K / 32K ctx |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 3,231 | 1,813 | 1,633 | 81.9 / 71.2 |
| 4,096 | 3,171 | 1,981 | 1,768 | 81.1 / 70.7 |
| 8,192 | 3,050 | 2,104 | 1,871 | 78.8 / 69.3 |

- Bigger chunks help less than the PCIe arithmetic predicted (halving the streamed bytes saved ~0.8 s of ~3.4 s at
  16K), so the copies were already partly hidden behind the VRAM experts' compute; the rest of the prompt time is
  now spread over dense projections (hipBLAS, ~3 s at 16K), attention and experts.
- A bigger chunk's scratch takes VRAM from the expert cache (60 slots per doubling at 4K, 120 at 8K), which costs
  output speed: 8K -4%. Default is now 4,096 (+9% prompts, -1% output); `--chunk 8192` for long-document work.

## 13. Speculative decoding: verify cost, prompt lookup, and why -ffast-math had to go

**What a verify window costs** (120b, 1K context, `--bench-verify`, mean of 20 steps):

| tokens checked in one step | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ms a step | 9.28 | 11.96 | 14.32 | 16.64 | 21.26 | 26.64 |
| relative to 1 | 1.00 | 1.29 | 1.54 | 1.79 | 2.29 | 2.87 |

Cheap enough: the dense weights and the attention are read once for all columns; what grows is the experts (each
extra token brings ~4 more per layer, partly shared, partly CPU misses). So a drafter pays when it keeps > ~0.8 of
3 drafts.

**Prompt lookup** (`oss_spec.hpp`): drafts the continuation of the latest earlier match of the last 2-4 tokens; the
number of drafts is the k with the most expected tokens per unit of the measured step cost, from the running
acceptance rate (0 = no draft, a 2-token probe every 16 steps). Sampling stays exact (a draft is kept with the
target's probability, else redrawn without it); greedy keeps a draft only when it is the argmax.

| 120b, greedy, profile placement, adaptation off | no drafts | prompt lookup | tokens a step |
| --- | ---: | ---: | ---: |
| explain-in-3-paragraphs prompt | 103.6 tok/s | 103.4 tok/s | 1.01 (the cost model declines) |
| rename a variable in a 40-line function | 81.9 tok/s | **101.3 tok/s (+24%)** | 1.89 |

**-ffast-math broke "the drafts never change the answer"**: with it, greedy with drafts gave different tokens on the
first prompt. Checked on 20b: logits of 493 positions computed 1 column a step vs 8 columns a step were bitwise equal
at 0 positions (pos 0 already differed by 0.149) with `-ffast-math`, and at all 493 without it, at the same speed
(164 vs 166 tok/s at 1 column, 462 vs 466 at 8). Reassociation lets the compiler sum differently in the 1..8-column
template instances. The kernels now build without it, and greedy output with drafts is token-identical to without
(both prompts above).

**The output-speed jump to ~104 tok/s is the profile, not the drafts**: VRAM now covers 97% of routings vs 77% on the
first start (the profile saved by earlier runs). Adaptation itself costs nothing measurable: 105.4 / 104.8 / 105.1
tok/s with swaps off / every 16 / every 64 steps.

## 14. EAGLE3 for gpt-oss-120b: what acceptance to expect (measured with llama.cpp)

`ggml-org/gpt-oss-120b-GGUF` ships `eagle3-gpt-oss-120b-Q8_0.gguf` (849 MB). llama-server (`--spec-type draft-eagle3`,
`--spec-draft-n-max 4`, greedy, low reasoning effort), same PC:

| prompt | drafts kept | tokens a step | llama.cpp output (its no-draft 50.5) |
| --- | ---: | ---: | ---: |
| explain-in-3-paragraphs (prose) | 286 / 849 = 34% | 2.34 | 34.9 tok/s |
| Python date-parsing function (code) | 246 / 416 = 59% | 3.37 | 49.7 tok/s |

- Acceptance is the draft model's, not the engine's, so it carries over. With strata-oss's step costs (5 tokens a step
  ~2.05x one, plus the 1-layer draft run 4 times), projected: prose ~1.0-1.1x, code ~1.4-1.5x. In llama.cpp it is a
  loss on this PC because its verify step is expensive (its CPU-side layers run every drafted token's experts).
- Cost of the port: EAGLE3 is one decoder layer (input: token embedding + fc(concat of 3 target layers' hidden
  states)), its own KV cache, a reduced-vocabulary head with a d2t map; the target must save 3 layers' hidden states
  for each verified token.
- Gotcha: `pkill -f <pattern>` from a shell whose own command line contains the pattern kills that shell (twice
  here). Kill by PID.

## 15. EAGLE3 in strata-oss

The draft (`--draft-model eagle3-gpt-oss-120b-Q8_0.gguf`, server `--draft-model`) runs on the last GPU after each
verify step: the decode graph copies the residual entering target layers 24 and 30 and after layer 35 into a feature
buffer (one strided device copy each, inside the graph), g(p) = fc(norm(features)) goes into a ring, and the draft's
KV rows (token p + 1, g(p)) are caught up for the accepted tokens before drafting. Its RoPE is llama-style (adjacent
pairs; base 500000, YaRN factor 32), its attention has no sinks, its MLP is a dense SwiGLU (16,384 wide).

- **Reduced draft vocabulary** (`oss/draft_vocab.txt`, `oss/tools/draft_vocab.py`): the full head is 201,088 x 2,880
  Q8_0 = 615 MB, which a draft token would read every time. 49,152 ids (the most frequent of a 4.0M-token docs + code
  corpus, the specials, then low ids) cover 96.4% of held-out gpt-oss answers (ids < 32K alone: 81-87%; o200k's id
  order is a poor frequency proxy). Head 141 MB. A draft token measures 0.97-1.07 ms (~0.09 of a target step).
- **The draft model costs VRAM**: ~0.9 GB on GPU 1 = 75 fewer expert slots (3,290 -> 3,215). Greedy output with
  EAGLE3 is token-identical to the same placement without drafts (checked on two prompts); against the run without
  the model loaded it differs, because the GPU/CPU split of the experts changed.
- **How many to draft.** First version: the cost model with one acceptance rate for every depth -> ~1.5 drafts a
  step, 1.65-1.89 tokens a step, prose 120 / code 89 tok/s. A per-depth conditional rate changed little: the model
  never tried deep drafts, so it never learned they were kept (an exploration problem). What worked: let the draft
  stop itself when its own top probability falls below p_min (as llama.cpp's `p_min`), up to 6 drafts:

| 120b, greedy, low effort, adaptation off | no drafts (no model loaded) | p_min 0.4 | **p_min 0.6 (default)** | p_min 0.8 |
| --- | ---: | ---: | ---: | ---: |
| prose (3 paragraphs) | 103.7 | 123.0 | **123.4 (+19%)** | 117.9 |
| code (Python function) | 81.7 | 88.7 | **95.3 (+17%)** | 97.4 |
| tokens a step prose / code | 1.00 | 2.12 / 2.44 | 1.79 / 2.18 | 1.50 / 1.93 |

- Why more tokens a step is not always faster on code: the code prompt's single step already costs 12.2 ms vs 9.6
  ms for prose (it routes to more experts that are on the CPU), and every extra checked token brings more CPU misses,
  so the verify cost table measured on prose under-states code's. `OSS_DRAFT_PMIN` overrides p_min.
- Prompt lookup still goes first when it finds a match (free drafts); the code-edit prompt stays ~98-101 tok/s.
- Server, EAGLE3 on: a code answer at temperature 0: 98.6 tok/s, at temperature 1.0: 107.3 tok/s (sampling stays
  exact: a draft is kept with the target's probability).
