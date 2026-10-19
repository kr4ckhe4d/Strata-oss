# Porting another MoE model to Strata's mechanism

The procedure that took gpt-oss-120b from "Strata cannot load it" to a served, measured engine (`oss/`), written so
the next port (a person or an agent) knows what to read, what to change, in what order, and how to prove each step
right. What was measured along the way is in [FINDINGS.md](FINDINGS.md); the design is explained in
[paper/strata-oss.md](paper/strata-oss.md).

Line numbers below are as of the commit that added this file; symbols are given so `grep -n` finds them after edits.

---

## 0. First decide which engine to start from

| the new model has... | start from |
| --- | --- |
| plain attention layers (full and/or sliding window), MoE FFN, no recurrent layers | **strata-oss** (`oss/`), this engine |
| Gated DeltaNet / linear-attention layers, shared experts, top-10 routing, an MTP head (the Qwen "Next" family: Qwen3-Next, Qwen3-Coder-Next, Qwen3.5/3.6, Qwen3.8-Flash-Next) | **Strata's own engine** (`src/`, `include/`): it already has DeltaNet, top-10, shared experts and MTP. Port the new *features* of strata-oss into it (section 9), not the other way round |
| something else (MLA attention, dense layers mixed in, ...) | the closer of the two; expect new kernels either way |

How to tell: `general.architecture` and the tensor names in the GGUF (step 1), and llama.cpp's graph for that
architecture (`~/llama.cpp/src/models/<arch>.cpp`). If the tensor list has `ssm_*`, `*_conv*`, `*delta*` or
`time_mix*` tensors, the model has recurrent layers.

---

## 1. Establish the model's facts before writing code

Read three things; the engine must match all of them exactly.

1. **The GGUF metadata and every distinct tensor** (name, type, shape):
   ```bash
   PYTHONPATH=~/llama.cpp/gguf-py python3 - MODEL.gguf <<'EOF'
   import gguf, re, sys
   r = gguf.GGUFReader(sys.argv[1])
   for k, f in r.fields.items():
       if not k.startswith("tokenizer.ggml."): print(k, f.contents() if k != "tokenizer.chat_template" else "...")
   seen = set()
   for t in r.tensors:
       key = re.sub(r"blk\.\d+\.", "blk.N.", t.name)
       if key not in seen: seen.add(key); print(t.name, t.tensor_type.name, list(t.shape))
   EOF
   ```
   (A partly downloaded GGUF already has its header: parse it before the download ends, as was done here.)
2. **llama.cpp's graph** for the architecture (`src/models/<arch>.cpp`) and its hparam loading
   (`src/llama-model.cpp`, `src/llama-context.cpp` for derived values such as YaRN's attention factor). This is the
   spec: write down every operation of a layer in order.
3. **The chat template** (`tokenizer.chat_template` in the GGUF) and the tokenizer's pre-tokenizer
   (`tokenizer.ggml.pre`, then that case in `src/llama-vocab.cpp`).

The checklist to fill in (gpt-oss's answers in brackets; each one was a decision in this port):

| fact | gpt-oss | where it lives in strata-oss |
| --- | --- | --- |
| layers, hidden size, heads / KV heads, head dim | 36, 2,880, 64 / 8, 64 | `read_hparams` (`oss_engine.cpp:301`), the shape check (`:317`) |
| which layers attend over a window, and its size | even layers, 128 | `Hp::is_swa` (`:106`), `DevLayer::window/cap` |
| mask rule (`q - k < window`?) | yes | `k_attn_split` (`oss_kernels.hip:265`), `k_attn_prompt_wmma` (`:923`) |
| attention extras: sinks, logit soft-cap, QK norm, output gate | sinks | `k_attn_combine` (`:345`), the prompt kernels' `m`/`l` init |
| attention scale | 1/sqrt(64) | the kernels' `scale` |
| RoPE: type (NeoX vs adjacent pairs), base, scaling (YaRN factor, original ctx, betas), partial rotary | NeoX, 150000, YaRN 32 / 4096 / 32 / 1, full | `yarn_cs` (`:210`), `RopeParams` (`oss_kernels.hpp:56`), `read_hparams` |
| derived factors llama.cpp applies (e.g. YaRN's attention factor) | cancels to 1; the kernel's mscale stays | `hp.rope.attn_factor` (`oss_engine.cpp:330`) |
| biases (q/k/v/o, router, experts) | all | `alloc_device` (`:438`), the expert kernels' bias args |
| norms: type, eps, where (pre-attn, post-attn, pre-FFN, final) | RMSNorm 1e-5; attn_norm, post_attention_norm (= pre-FFN), output_norm | `capture` (`:1078`), `prompt_layer` (`:1281`) |
| routing: logits -> top-k -> softmax over k? or softmax -> top-k -> renormalize? sigmoid? bias for selection only? | top-4 of logits, softmax over the 4 | `k_router_topk` (`oss_kernels.hip:368`) |
| shared expert(s) | none | (none: would be a dense FFN beside the routed ones) |
| expert activation | SwiGLU-OAI: clamp 7, alpha 1.702, `(up + 1)` | `swiglu_oai` (`oss_kernels.hip:479`, `oss_cpu.cpp:156`) |
| expert / dense weight formats | MXFP4 / Q8_0, router F32 | `make_blob` (`oss_engine.cpp:354`), `q8_planes` (`:270`), loaders below |
| embedding / head tied? formats | separate, Q8_0 | `embed` (`:373`), the head in `alloc_device` |
| vocab, special tokens, stop tokens | 201,088; `<|return|>` 200002, `<|call|>` 200012 | `oss_tokenizer.py`, `server.py:119`, `oss_main.cpp:323` |
| pre-tokenizer regex | llama.cpp's GPT4O | `GPT4O_PATTERN` (`oss_tokenizer.py:18`) |
| chat format, reasoning / tool-call syntax | harmony (analysis / final channels, `to=functions.x`) | `harmony_prompt` (`oss_tokenizer.py:49`), `class Harmony` (`server.py:197`) |
| a draft model or MTP head | none built in; EAGLE3 published separately | `load_eagle` (`oss_engine.cpp:735`) |

---

## 2. The map: what in strata-oss is model-specific

### Model math and shapes (must change for a new model)

| what | where |
| --- | --- |
| architecture name check, hparams, the hard shape check (H 2880, head dim 64, top-4, 8 q heads per KV head, ≤ 256 experts) | `read_hparams`, `oss_engine.cpp:301-350` |
| `kTopK = 4` (columns x K arrays everywhere) | `oss_kernels.hpp:25` |
| `LayerMailbox` / `Handoff` sized for 2,880 floats and 4 experts | `oss_kernels.hpp:38-55` (`hd[kMaxCols][2880/32]`, `hq`, `y_cpu`, `x`) |
| head dim 64 baked into the attention kernels (`kHD`, per-lane 2 dims, WMMA 4 k-steps) | `oss_kernels.hip:260` and the attention kernels |
| GQA group of 8 (one wave per query head of a group, `P[8][..]`) | `k_attn_split` (`oss_kernels.hip:265`) |
| sinks (decode combine, both prompt kernels) | `k_attn_combine` (`:345`), `k_attn_prompt` (`:625`), `k_attn_prompt_wmma` (`:923`) |
| sliding-window pattern and ring capacity | `Hp::is_swa` (`oss_engine.cpp:106`), `alloc_device` (`L.window`, `L.cap`) |
| RoPE (YaRN, NeoX / adjacent pairs) | `yarn_cs` (`oss_kernels.hip:210`), `k_rope_store` (`:228`, `rp.neox` at `:238`) |
| the layer's order of operations, decode | `capture` (`oss_engine.cpp:1078`): one HIP graph per GPU per column count |
| the layer's order of operations, prompt | `prompt_layer` (`oss_engine.cpp:1281`) |
| routing | `k_router_topk` (`oss_kernels.hip:368`) |
| expert activation | `swiglu_oai` (`oss_kernels.hip:479`, `oss_cpu.cpp:156`) |
| tensor names (`attn_q`, `attn_sinks`, `ffn_gate_inp`, `post_attention_norm`, ...) | `alloc_device` (`oss_engine.cpp:438-490`), `make_blob` (`:354`) |

### Weight formats (change if the quantization differs)

| format | decode | prompt | CPU |
| --- | --- | --- | --- |
| MXFP4 experts (planar blob: codes then E8M0 scales; `blob_bytes`, `oss_engine.cpp:350`) | `mxfp4_lut` / `mxfp4_block_dot` (`oss_kernels.hip:49`), `k_moe_gate_up` (`:504`), `k_moe_down` (`:529`) | `k_mmq_wmma` loader (`:1075`), `k_dequant_mxfp4` (`:564`) | `mxfp4_row_dot` (`oss_cpu.cpp:56`), `rows2_cols` (`:111`), `kValues` (`:18`) |
| Q8_0 dense (planes: int8 + f16 scale per 32) | `k_mv_q8_0` (`oss_kernels.hip:145`) | `dense_q8_wmma` (`:1515`), the `Q8` branch of `k_mmq_wmma` | - |

A new expert format (e.g. Q4_K, IQ3) needs: a load-time repack into a blob both processors read (`make_blob`), the
decode dot (a lane per 32-block, int8 activations), the prompt tile loader (decode a block into int8 LDS rows +
a float scale per row and block), and the CPU dot with the **same rounding order** (section 4.4). Strata's
`native_mmvq.cu` and `src/kernels/cpu/iq_*.cpp` have i-quant / k-quant versions to start from.

### Chat format and tokens (change for a new model family)

| what | where |
| --- | --- |
| pre-tokenizer pattern (reuses `tools/strata_tokenizer.py`'s BPE) | `oss_tokenizer.py:18` |
| prompt formatting (system / developer / user / assistant / tool, reasoning effort, tools as text) | `harmony_prompt`, `oss_tokenizer.py:49` |
| output parsing (reasoning vs answer vs tool calls) | `class Harmony`, `server.py:197`; token ids `server.py:119` |
| stop tokens | `STOP_IDS` (`server.py:120`), the CLI's stop list (`oss_main.cpp:323`) |

### Draft model (change or drop)

EAGLE3 is specific to its target: the tapped layers (`eagle3.target_layers`, `oss_engine.cpp:745`), llama-style RoPE
(`e.rope.neox = 0`, `:773`), its own head and the draft vocabulary (`oss/draft_vocab.txt`, rebuilt with
`oss/tools/draft_vocab.py` for the new tokenizer). A model with an MTP head would replace `load_eagle` /
`eagle_rows` / `draft` (`oss_engine.cpp:735-1075`) with its MTP layer; the `Generator` (`oss_spec.hpp:120`) only
needs `has_draft()` / `draft()`.

### Model-independent (keep as it is)

The expert tiering (`place_and_load` `:582`, slots, the RAM arena, the profile, `adapt` `:1236`), the GPU/CPU
hand-off (`k_publish` `oss_kernels.hip:422`, `wait_ack`, `step_cols` `oss_engine.cpp:1134`), the CPU pool and the
grouped misses (`compute_misses`, `oss_cpu.cpp:250`), the prompt ring and grouping (`prompt_layer`'s expert half),
the KV regions per conversation and `step_multi` (`:1526`), the scheduler (`oss_main.cpp:349`), the server and the
dashboard (`server.py`: `Stats` `:39`, `_metrics` `:302`), the sampler, prompt lookup and the draft-length cost
model (`oss_spec.hpp`).

---

## 3. The order that worked, with the gate for each step

Do not move to the next step before its gate passes. Commit after each step (this repo: as the user, no
co-author line) and append what was measured to `FINDINGS.md`.

| # | step | gate (how to know it is right) |
| --- | --- | --- |
| 1 | facts (section 1); download the model and a **small sibling** of the same architecture (gpt-oss-20b here) | the checklist is filled in from the GGUF and llama.cpp's graph, not from memory |
| 2 | tokenizer + chat format | `tokenize(x)` equals `llama-tokenize` on mixed text (accents, CJK, code, numbers, whitespace runs); round trip decodes to the input |
| 3 | reference logits with llama.cpp (`oss/tools/llama_logits.cpp`, batched **and** one token at a time) | the two references' agreement measured: that is the noise floor (section 4.1) |
| 4 | loader + decode path (1 column), small sibling, all experts in VRAM | perplexity of the actual next tokens in llama.cpp's band; a greedy answer reads right |
| 5 | CPU path: `--max-slots` = half the experts | same text for a long prefix, then an equivalent answer; perplexity unchanged |
| 6 | prompt path (`--dump-logits` without `OSS_FORCE_DECODE`) | perplexity in the band; KL vs the decode path small |
| 7 | the big model across the GPUs | coherent answers; first numbers; `oss120.prof`-style profile saved |
| 8 | multi-column decode (1-8) | **bitwise**: logits computed 1 column a step = 8 columns a step (section 4.3) |
| 9 | performance work, one change at a time (section 5) | each change: perplexity / KL against the previous kernel, then speed |
| 10 | speculation (verify cost first: `--bench-verify`) | greedy with drafts token-identical to without (same placement) |
| 11 | server, batching (`--bench-parallel`) | N conversations together token-identical to each alone |

---

## 4. The validation recipe

### 4.1 The noise floor is not zero: use perplexity, not top-1 agreement

MoE + low-bit weights turn rounding into different expert picks. gpt-oss-20b: llama.cpp's batched vs one-at-a-time
logits agree on only 92.7% of positions (KL 0.026). So "my engine agrees with llama.cpp on 91%" is *not* a bug
signal. The test that catches real bugs is teacher-forced perplexity of the actual next tokens: it must fall in the
band of llama.cpp's own paths (FINDINGS.md section 3). Use a few hundred tokens of a harmony/chat prompt and a
~2,000-token document. Keep the dumps **on the SSD** (`/tmp` is RAM here, section 7).

### 4.2 A new kernel vs the old one

Every rewrite kept the old kernel behind an environment switch (`OSS_ATTN_WMMA=0`, `OSS_MMQ_WMMA=0`,
`OSS_DENSE_WMMA=0`, `OSS_GROUP_MISSES=0`) and was checked against it: KL between them, perplexity of both. Keep
that habit: it makes every speed claim an A/B and leaves a fallback for other GPUs.

### 4.3 Bitwise checks

- **1 column vs 8 columns a step** (`OSS_FORCE_DECODE=1 OSS_MAX_COLS=1` vs `=8`, `--adapt-every 0`): must match
  bitwise at every position, with experts on the CPU too (`--max-slots`). It failed with `-ffast-math`
  (reassociation differs per template instance); the kernels build without it.
- **Batched vs alone** (`--bench-parallel`, adaptation off): every conversation token-identical. Read each prompt
  the same way in both passes (the prompt path rounds differently from the decode path).
- **Greedy with drafts vs without**: identical with the same expert placement (loading a draft model changes the
  placement, so compare against the draft loaded with `--no-spec`).

### 4.4 GPU and CPU must round the same way

Both use int8 activations with one float scale per 32 values, and the CPU gets the activation **already quantized
by the GPU** (through the mailbox). A CPU kernel that sums in a different order is fine for quality but breaks the
bitwise checks when experts move between tiers; the grouped CPU kernel keeps the single-column order per column.

### 4.5 Fixed-layout hardware instructions: test them first

The gfx12 WMMA fragment layouts (f16 and int8) were verified with a 16x16 test against a CPU matmul before any
kernel used them (FINDINGS.md section 10). Do the same for any new instruction or architecture (gfx11's layout is
different).

---

## 5. Measuring and tuning

- `OSS_PROF=1`: the prompt path's time per layer and per piece (attention+router, experts, streamed count) and the
  totals of q/k/v, attention, o-proj, router. Always find the real bottleneck first: here it was per-expert
  overhead, then untuned hipBLAS (10-15 TFLOPS on gfx1201), then PCIe (28.7 GB/s per card), never what was guessed.
- `--bench-verify`: the cost of checking 1..8 tokens in one step: the input to any drafter's cost model
  (`cost[]`, `oss_spec.hpp:125`; the draft's own cost `draft_cost`, `:128`).
- `--bench-parallel`: batched throughput and exactness.
- Speeds depend on the **profile**: a first start (no profile) ran 60 tok/s, a matured one ~104. Compare runs with
  the same profile (`--profile-in`, not `--profile`, so the run does not rewrite it) and `--adapt-every 0`.
- Change a default only with a measurement that shows it faster, and write the numbers with what they ran on.

---

## 6. The engine's numerics conventions (keep them)

- Activations for every int8 dot: int8 + one float scale per 32 values (`quant_q8`, `rmsnorm_quant`), rounding
  half away from zero, `d = amax / 127`.
- Weight blobs repacked at load to aligned planes; both processors read the same bytes (an expert moves between
  RAM and VRAM as a plain copy).
- KV cache f16; attention math f32 (WMMA: f16 inputs, f32 accumulators).
- No `-ffast-math` in the kernels.

---

## 7. Pitfalls this port hit (check them first)

- `/tmp` is a 16 GB tmpfs and swap is zram on this PC: large dumps there eat RAM and push the engine into swap.
- llama.cpp on a small-RAM PC: use `-lm none` (no mmap) for benchmarks; `--n-cpu-moe` takes the *first* N layers.
- A GGUF's "target layer = n_layer" (EAGLE3) means after the last layer, before the output norm.
- A ring buffer (sliding window, draft KV) has a capacity smaller than the context: never use the capacity as the
  context bound (a real bug here).
- LDS rows used with 16-byte stores must be 16-byte aligned (pad rows to a multiple of 16).
- zsh: arrays start at 1, `$X` does not word-split (use arrays or `${=X}`), `pkill -f pattern` kills a shell whose
  own command line contains the pattern; `nohup`'d background jobs ignore SIGINT (the server handles SIGTERM).
- A comparison that reads the prompt through different paths is not a fair bitwise test.

---

## 8. Notes for a Qwen3-Next-family port (e.g. Qwen3-Coder-Next 80B-A3B)

Confirm against the GGUF first; from the published architecture of that family the differences from gpt-oss are:

| | gpt-oss | Qwen3-Next family (check) | consequence |
| --- | --- | --- | --- |
| mixers | attention only (full / window) | Gated DeltaNet layers + gated full-attention layers (3:1) | DeltaNet keeps a **recurrent state per conversation**, not a KV cache: per-slot state, and rollback (truncate after rejected drafts, prefix reuse) needs state snapshots. Strata's engine already does this |
| attention | sinks, window | output gate, possibly QK norm, partial RoPE | new attention epilogue; RoPE on part of the head dim |
| routing | top-4, softmax over 4 | top-10 of 512 experts + a shared expert | `kTopK`, mailbox sizes, a dense shared-expert FFN on the GPU every layer; 10 experts a token means more CPU misses at the same hit rate |
| expert size | 12.6 MiB (2880 x 2880 x 3, MXFP4) | small experts (hidden 2048, expert FFN 512): ~1-2 MiB at 4 bits | many more, smaller slots: per-expert overheads matter more, the prompt path's tiles want more tokens per expert |
| draft | EAGLE3 (external) | MTP head in the model | Strata's MTP path |
| chat | harmony | ChatML-style (`<|im_start|>`), Qwen tool-call format | Strata's `serve/frontend.py` already speaks it |

That is why section 0 points to Strata's engine for this family: its DeltaNet, top-10, shared-expert and MTP paths
exist and are validated. What would be worth carrying over from strata-oss: multi-conversation batched decode
(section 20 of FINDINGS.md), the grouped CPU misses (section 18), the int8 WMMA prompt kernels on gfx12
(sections 10, 11, 16), and the dashboard's metrics if missing.

---

## 9. Bringing strata-oss features into another engine

| feature | the pieces |
| --- | --- |
| grouped CPU misses | sort the window's jobs by expert; a 2-row x N-column dot that decodes a block once (`rows2_cols`, `oss_cpu.cpp:111`) and keeps per-column summation order |
| batched conversations | a slot id per column in the step parameters; KV / state regions per slot; per-slot history and draft state; a scheduler that reads one prompt at a time and batches decode (`step_cols`, `step_multi`, `oss_main.cpp:349`) |
| WMMA prompt kernels | verify the fragment layout first; the tile loader is the only format-specific part (`k_mmq_wmma`'s `Q8` / MXFP4 branches) |
| dashboard | Strata's `serve/web` as it is, plus `health`, `settings`, `metrics` in Strata's shape (`server.py:302`) |
