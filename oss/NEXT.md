# Next session: Qwen3-Coder-Next on this PC (plan)

Written 2026-10-19 at the end of the gpt-oss session. Read first: [PORTING.md](PORTING.md) (§0, §8, §9),
[FINDINGS.md](FINDINGS.md) (what was measured and the pitfalls), the repo root `AGENTS.md`.

## Where things stand

- **Repos.** `~/code/Strata` is the user's working Qwen install: do not change its code. `~/code/Strata-oss`
  (branch `gpt-oss`, pushed to `git@github.com:kr4ckhe4d/Strata-oss.git` as `main`, remote name `fork`) holds the
  gpt-oss engine in `oss/`. Commit as the user, no "Co-Authored-By" line:
  `git -c user.name="Nipuna H Herath" -c user.email="hash.crackhead@gmail.com" commit ...`; push with
  `git push fork gpt-oss:main`.
- **Servers: all stopped** (the user asked). The user's Qwen Strata server (port 8095) and llama-server (8090) are
  restarted with the commands in section "Restore" below. Ask before stopping or starting any of them; they need both
  GPUs and most of the RAM, so they cannot run beside a test.
- **Machine.** Ryzen 7 9800X3D (8 cores, AVX-512), 30 GB RAM (`/tmp` is RAM, swap is zram: keep large files on
  `/mnt/fast`), RX 9070 XT 16 GB + Radeon AI PRO R9700 32 GB (gfx1201, PCIe 5.0 x8 each, 28.7 GB/s), ROCm 7.2.
  llama.cpp at `~/llama.cpp` (HIP build, `llama-bench`, `llama-server`); benchmarks need `-lm none`.
- **gpt-oss state:** models in `/mnt/fast/models/gpt-oss-120b/` (+ `eagle3-*.gguf`, `oss120.prof` = the learned
  expert profile). Serve command in `oss/README.md` (add `--host 0.0.0.0 --api-key "$(tr -d '\n' <
  ~/code/Strata/.strata-api-key)"` for the user's Mac; dashboard at http://192.168.4.94:8096/).

## Goal

Find out whether Strata's mechanism would run **Qwen3-Coder-Next** (80B total, ~3B active, Qwen3-Next
architecture) faster than llama.cpp on this PC, and if so, run it.

## Plan

1. **Identify the model (no GPU needed).** Find the GGUFs (e.g. unsloth / Qwen on Hugging Face) and their sizes per
   quant. Download the header of the likely candidates (or one small quant) to `/mnt/fast/models/qwen3-coder-next/`.
   Dump `general.architecture`, all `qwen3next.*` (or whatever) keys and every distinct tensor (script: PORTING.md
   §1). Fill PORTING.md §1's checklist for this model from the GGUF and from llama.cpp's graph
   (`~/llama.cpp/src/models/` for that arch).
2. **Compare with what Strata's engine runs (no GPU).** Strata knows `qwen4exp` (Qwen3.8-Flash-Next) and `qwen35moe`
   (Qwen3.6, `src/qwen35`): `include/strata/artifact/model_family.hpp`, `include/strata/core/geometry.hpp`,
   `gguf_reader.hpp` (`read_geometry`, `geometry_unsupported`), `docs/MODELS.md`. List exactly what differs (layer
   pattern, DeltaNet shape, attention gate / QK norm / partial RoPE, shared expert, top-k, expert sizes, quant types,
   MTP tensors, chat template). Decide: extend Strata's Qwen engine (expected, PORTING.md §0 / §8) vs strata-oss.
   Write the findings into a new section of `FINDINGS.md` or a new `qwen-next/FINDINGS.md`.
3. **Set the bar with llama.cpp (needs the GPUs: ask the user first).** Pick the quant the user would run.
   Measure `llama-bench` (`-ngl 99 -fa on -lm none -p 512 -n 128`, both GPUs, `-ts` by VRAM; if it does not fit,
   the best `--n-cpu-moe` / `-ts` split as found for gpt-oss: FINDINGS.md §6), plus a 16K prompt. Also try
   llama.cpp's MTP drafting if the GGUF has the MTP layer (`--spec-type draft-mtp`).
4. **Decide with the user.** If the quant fits in 48 GB VRAM and llama.cpp is already fast, Strata's advantage is
   small (expert tiering only pays when the model does not fit; expected ±20%). If it does not fit (5-8 bit), the
   tiering should win as it did for gpt-oss (2x output). Present the numbers and the port's size before starting.
5. **If porting:** follow PORTING.md §3's step order and gates in a fresh clone or branch of Strata (never in
   `~/code/Strata`): small sibling model first if one exists, tokenizer check vs `llama-tokenize`, reference logits
   (`oss/tools/llama_logits.cpp`), perplexity band (not top-1 agreement: PORTING.md §4.1), bitwise checks, then
   speed. Bring over strata-oss's features where they help (PORTING.md §9): batched conversations, grouped CPU
   misses, int8 WMMA prompt kernels, the dashboard metrics.
6. **Throughout:** log every measurement and the reason for each bottleneck (the user wants this), commit each step,
   push, keep `oss/README.md`-style run instructions current.

## Also open from the gpt-oss work (optional)

- Small-prompt speed-up for agent turns ("prompt CPU share"): a short new chunk still streams nearly every expert
  missing from VRAM over PCIe (~0.3-0.5 s); let the CPU compute the missing experts of small chunks instead. Seen on
  the user's dashboard: prefill 458-649 tok/s for ~200-300 new tokens with 12K reused.
- Reading a new prompt pauses the other conversations; batched steps use no drafts.

## Restore the user's servers (only when asked)

```bash
cd /home/nipuna/code/Strata && nohup .venv/bin/python /home/nipuna/code/Strata/serve/server.py --engine strata \
  --config /home/nipuna/code/Strata/strata-iq3_xxs.json --port 8095 >> /home/nipuna/code/Strata/strata-server.out 2>&1 &
cd /home/nipuna/llama.cpp && nohup /home/nipuna/llama.cpp/build/bin/llama-server --models-preset \
  /home/nipuna/llama.cpp/models-preset.ini --models-max 1 --sleep-idle-seconds 900 --host 0.0.0.0 --port 8090 \
  >> /tmp/llama-server.log 2>&1 &
```
