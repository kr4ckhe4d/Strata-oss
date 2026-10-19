# strata-oss: gpt-oss-120b on a normal PC, the Strata way

A separate engine for OpenAI's gpt-oss (120b and 20b) that uses Strata's mechanism: the dense weights and the
most-used experts live in VRAM (shared across the GPUs by layer), every other expert sits in page-locked RAM and is
computed by the CPU at the same time as the GPU computes the cached ones, a prompt's missing experts stream over
PCIe, and the cache follows the routing (saved as a profile for the next start). The Qwen engine in `src/` is not
touched; this is its own CMake project.

Measured numbers, what each one ran on, and why each bottleneck was there: [FINDINGS.md](FINDINGS.md). The design,
explained: [paper/strata-oss.md](paper/strata-oss.md). Porting another model: [PORTING.md](PORTING.md). In short, on
a Ryzen 7 9800X3D with 30 GB of RAM and an RX 9070 XT 16 GB + Radeon AI PRO R9700 32 GB: output 91-101 tok/s
(116-131 with the EAGLE3 draft), prompts 1,800-2,700 tok/s; llama.cpp's best hybrid split on the same PC: 50.5 and
791.

## Build (Linux, ROCm, gfx1200 / gfx1201)

```bash
cmake -S oss -B build-oss -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build build-oss
python3 -m venv .venv && .venv/bin/pip install regex numpy
```

HIP only. The WMMA kernels (prompt attention, prompt experts and projections) need gfx1200 / gfx1201; on other AMD
cards the dp4a / hipBLAS paths run instead (not measured). The CPU kernels build with `-march=native`
(`-DSTRATA_OSS_NATIVE=OFF`: AVX2).

## Models

```bash
hf download ggml-org/gpt-oss-120b-GGUF gpt-oss-120b-MXFP4.gguf --local-dir /path/to/models
hf download ggml-org/gpt-oss-120b-GGUF eagle3-gpt-oss-120b-Q8_0.gguf --local-dir /path/to/models   # optional draft
```

## Serve (OpenAI-compatible)

```bash
.venv/bin/python oss/server.py --model /path/to/models/gpt-oss-120b-MXFP4.gguf \
  --draft-model /path/to/models/eagle3-gpt-oss-120b-Q8_0.gguf --port 8096 -- --ctx 65536 --profile oss120.prof
```

`http://127.0.0.1:8096/v1`: `/chat/completions` (streaming, `reasoning_content`, `tool_calls`, `reasoning_effort`),
`/completions`, `/models`. Up to `--parallel` answers at once (batched into one decode step); a new request reuses
the KV cache of the free conversation slot that shares the longest prefix with it. Never listen beyond 127.0.0.1
without `--api-key`. Engine flags go after `--`:

| flag | default | |
| --- | --- | --- |
| `--ctx N` | 32768 | context per conversation (the model's maximum is 131072) |
| `--parallel N` | 1 | conversations at once (1-8); each has its own KV cache: ~68 expert slots per 16K of context |
| `--draft-window N` | 4096 | the EAGLE3 draft's attention window (0: the whole context) |
| `--chunk N` | 4096 | prompt chunk; 8192 is faster for long documents but takes expert slots |
| `--prompt-cpu N` | 8 | a prompt chunk's RAM expert used by at most N of its tokens runs on the CPU instead of crossing PCIe (0: off) |
| `--profile FILE` | none | read the expert profile at start, write the learned one at exit; first start is slower |
| `--gpus 0,1` / `--split 12,24` | all / by free VRAM | which GPUs, and how many layers each |
| `--no-spec` | drafts on | no speculative drafts (prompt lookup, EAGLE3) |
| `--adapt-every N` | 16 | decode steps between expert swap rounds (0: off) |

Environment switches (all measured in FINDINGS.md): `OSS_ATTN_WMMA=0`, `OSS_MMQ_WMMA=0`, `OSS_DENSE_WMMA=0` (the
older kernels), `OSS_GROUP_MISSES=0` (one CPU job per column), `OSS_DRAFT_PMIN` (EAGLE3's confidence cut, 0.6),
`OSS_EAGLE_CB` (catch-up rows a piece, 256), `OSS_PROF=1` (prompt-path timings).

## Command line

```bash
build-oss/strata-oss --model gpt-oss-120b-MXFP4.gguf --tokens ids.txt --n-predict 256   # greedy, prints ids
build-oss/strata-oss --model gpt-oss-120b-MXFP4.gguf --tokens ids.txt --bench-verify    # the cost of a verify step
build-oss/strata-oss --model gpt-oss-120b-MXFP4.gguf --parallel 4 --tokens a.txt,b.txt,c.txt,d.txt --bench-parallel
build-oss/strata-oss --model gpt-oss-20b-mxfp4.gguf --tokens ids.txt --dump-logits out.bin
```

`oss/tools/llama_logits.cpp` dumps llama.cpp's logits for the same ids (the reference the checks used).

## Files

`oss_engine.cpp` (loading, placement, the decode step with the GPU/CPU hand-off, the prompt path, adaptation, EAGLE3),
`oss_kernels.hip` (GPU kernels), `oss_cpu.cpp` (the CPU expert pool, MXFP4 x int8), `oss_spec.hpp` (sampling,
prompt lookup, the draft-length cost model), `oss_main.cpp` (CLI, `--serve` JSON lines), `server.py` (HTTP),
`oss_tokenizer.py` (o200k + harmony), `draft_vocab.txt` / `tools/draft_vocab.py` (EAGLE3's reduced vocabulary).
