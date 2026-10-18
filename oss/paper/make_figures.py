"""Figures for oss/paper/strata-oss.md.  python oss/paper/make_figures.py  (writes fig*.svg next to this file)

Plain SVG, no dependencies.  White background so the figures read the same in light and dark viewers.
"""
from __future__ import annotations

import pathlib

OUT = pathlib.Path(__file__).resolve().parent
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
MONO = "JetBrains Mono, Consolas, Menlo, monospace"

# palette
INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
GPU, GPU_D = "#dbeafe", "#1d4ed8"      # blue
CPU, CPU_D = "#dcfce7", "#15803d"      # green
RAM, RAM_D = "#fef3c7", "#b45309"      # amber
HOST, HOST_D = "#f3e8ff", "#7e22ce"    # violet
SSD, SSD_D = "#f1f5f9", "#475569"      # slate
HOT, HOT_D = "#fee2e2", "#b91c1c"      # red


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.title = w, h, title
        self.parts: list[str] = []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke, r=8, sw=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=13, color=INK, anchor="start", weight="normal", font=FONT, italic=False):
        st = ' font-style="italic"' if italic else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" '
                 f'text-anchor="{anchor}" font-weight="{weight}"{st}>{s}</text>')

    def lines(self, x, y, rows, size=12, color=INK, anchor="start", gap=None, weight="normal", font=FONT):
        gap = gap or size + 4
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size, color, anchor, weight, font)

    def arrow(self, x1, y1, x2, y2, color=LINE, sw=1.8, dash=None, both=False, label=None, lx=None, ly=None,
              lsize=11, lcolor=MUTED, lanchor="middle"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        ms = ' marker-start="url(#a0)"' if both else ""
        mid = {"#8c959f": "a", }.get(color, "a")
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d} '
                 f'marker-end="url(#{mid})"{ms}/>')
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6, label,
                      lsize, lcolor, lanchor)

    def path(self, d, color=LINE, sw=1.8, dash=None, end=True):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        me = ' marker-end="url(#a)"' if end else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da}{me}/>')

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{self.title}"><title>{self.title}</title>'
                '<defs>'
                '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
                '<marker id="a0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
                '</defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>')
        (OUT / name).write_text(head + "".join(self.parts) + "</svg>\n")


# ---------------------------------------------------------------------------------------------------- fig 1
def fig1():
    s = Svg(1000, 560, "Figure 1: where gpt-oss-120b lives on the test PC")
    s.text(500, 30, "Figure 1 · where gpt-oss-120b (63.4 GB) lives on a 48 GB-VRAM, 30 GB-RAM PC", 16, INK,
           "middle", "bold")
    # GPU 0
    s.rect(30, 60, 300, 300, GPU, GPU_D)
    s.text(180, 85, "GPU 0 · RX 9070 XT · 16 GB", 14, GPU_D, "middle", "bold")
    s.rect(50, 100, 260, 60, "#ffffff", GPU_D, 6, 1)
    s.lines(62, 122, ["layers 0-11: attention (Q8_0), norms, sinks,", "router (F32), expert biases  · ~0.4 GB"], 12)
    s.rect(50, 170, 260, 40, "#ffffff", GPU_D, 6, 1)
    s.lines(62, 195, ["KV cache (f16), sliding-window rings"], 12)
    s.rect(50, 220, 260, 120, HOT, HOT_D, 6, 1.2)
    s.text(180, 245, "expert cache: 1,079 slots", 13, HOT_D, "middle", "bold")
    s.lines(180, 268, ["the most-routed experts of layers 0-11", "12.6 MiB each (gate + up + down, MXFP4)",
                       "picked by the saved profile,", "swapped as the routing drifts"], 12, INK, "middle")
    # GPU 1
    s.rect(360, 60, 300, 300, GPU, GPU_D)
    s.text(510, 85, "GPU 1 · AI PRO R9700 · 32 GB", 14, GPU_D, "middle", "bold")
    s.rect(380, 100, 260, 60, "#ffffff", GPU_D, 6, 1)
    s.lines(392, 122, ["layers 12-35 + output head (Q8_0, 0.6 GB)", "(+ EAGLE3 draft, 0.9 GB, optional)"], 12)
    s.rect(380, 170, 260, 40, "#ffffff", GPU_D, 6, 1)
    s.lines(392, 195, ["KV cache, rings, prompt scratch"], 12)
    s.rect(380, 220, 260, 120, HOT, HOT_D, 6, 1.2)
    s.text(510, 245, "expert cache: 2,252 slots", 13, HOT_D, "middle", "bold")
    s.lines(510, 268, ["the most-routed experts of layers 12-35", "VRAM experts are computed here",
                       "(device-side routing: the GPU reads", "top-4 + the slot table itself)"], 12, INK, "middle")
    # RAM / CPU
    s.rect(690, 60, 280, 300, RAM, RAM_D)
    s.text(830, 85, "RAM · 30 GB DDR5", 14, RAM_D, "middle", "bold")
    s.rect(710, 100, 240, 90, "#ffffff", RAM_D, 6, 1)
    s.lines(830, 122, ["the other 1,277 experts", "15.7 GB, page-locked, planar MXFP4",
                       "(the RAM holds exactly what", "the GPUs do not)"], 12, INK, "middle")
    s.rect(710, 200, 240, 70, CPU, CPU_D, 6, 1.2)
    s.lines(830, 222, ["CPU pool · 8 Zen 5 cores", "computes the missed experts in place", "AVX-512 VNNI MXFP4 x int8"],
            12, INK, "middle")
    s.rect(710, 280, 240, 60, HOST, HOST_D, 6, 1.2)
    s.lines(830, 302, ["mailboxes (coherent host memory)", "routing in, CPU results out, flags"], 12, INK, "middle")
    # SSD
    s.rect(30, 400, 210, 70, SSD, SSD_D)
    s.lines(135, 425, ["NVMe · the GGUF", "read once (~15 s)"], 12, INK, "middle")
    s.arrow(135, 400, 135, 362, label="load", lx=145, ly=385, lanchor="start")
    # links
    s.arrow(332, 290, 358, 290, both=True)
    s.text(372, 380, "↔ residual hand-off between the GPUs", 11, MUTED)
    s.text(372, 394, "via host memory (11.5 KB a token)", 11, MUTED)
    s.path("M 280 360 C 280 500, 800 500, 800 362", LINE, 1.6, "5 4")
    s.path("M 560 360 C 560 450, 840 450, 840 362", LINE, 1.6, "5 4")
    s.text(600, 520, "PCIe 5.0 x8 each, 28.7 GB/s measured: hand-off flags (decode), missing experts streamed "
           "(prompt), swaps", 12, MUTED, "middle")
    s.text(600, 540, "decode: the CPU computes the misses · prompt: the GPU computes all, the misses streamed in", 12,
           MUTED, "middle")
    s.save("fig1-tiers.svg")


# ---------------------------------------------------------------------------------------------------- fig 2
def fig2():
    s = Svg(1000, 520, "Figure 2: one layer of a decode step, GPU and CPU in parallel")
    s.text(500, 30, "Figure 2 · one layer of a decode step: the GPU and the CPU each compute their share of the experts",
           15, INK, "middle", "bold")
    # lanes
    s.rect(20, 50, 960, 170, GPU, GPU_D, 10)
    s.text(36, 72, "GPU (one HIP graph per step; nothing launched from the host mid-step)", 13, GPU_D, "start", "bold")
    s.rect(20, 330, 960, 140, CPU, CPU_D, 10)
    s.text(36, 352, "host + CPU pool", 13, CPU_D, "start", "bold")
    s.rect(300, 245, 400, 60, HOST, HOST_D, 8)
    s.text(500, 268, "LayerMailbox (page-locked, coherent)", 12, HOST_D, "middle", "bold")
    s.text(500, 288, "routed=seq · top-4 ids + weights · int8 FFN input · y_cpu · has_cpu · acked=seq", 11, INK,
           "middle")
    # gpu boxes
    xs = [40, 150, 260, 370, 480]
    labels = [["RMSNorm", "→ int8"], ["QKV GEMV", "Q8_0 +bias"], ["RoPE (YaRN)", "KV append"],
              ["attention", "sinks · window"], ["O-proj", "+ residual"]]
    for x, lb in zip(xs, labels):
        s.rect(x, 90, 100, 50, "#ffffff", GPU_D, 6, 1)
        s.lines(x + 50, 110, lb, 11, INK, "middle")
    for i in range(4):
        s.arrow(xs[i] + 100, 115, xs[i + 1], 115)
    s.rect(590, 90, 110, 50, "#ffffff", GPU_D, 6, 1)
    s.lines(645, 110, ["norm · router", "top-4 · softmax"], 11, INK, "middle")
    s.arrow(580, 115, 590, 115)
    s.rect(720, 90, 120, 50, HOT, HOT_D, 6, 1)
    s.lines(780, 110, ["VRAM experts", "gate/up → down"], 11, INK, "middle")
    s.arrow(700, 115, 720, 115)
    s.rect(860, 90, 105, 50, "#ffffff", GPU_D, 6, 1)
    s.lines(912, 110, ["wait ack", "x += y_gpu + y_cpu"], 11, INK, "middle")
    s.arrow(840, 115, 860, 115)
    s.text(40, 175, "the expert kernels read top-4 and the slot table on the device;", 11, MUTED)
    s.text(40, 192, "a slot of -1 is skipped (that expert is the CPU's)", 11, MUTED)
    s.text(40, 209, "publish: one kernel writes routing + int8 input to host memory, fence, routed = seq", 11, MUTED)
    # arrows to/from mailbox
    s.arrow(645, 140, 600, 245, label="publish", lx=632, ly=190, lanchor="start")
    s.arrow(870, 245, 900, 140, label="ack", lx=905, ly=205, lanchor="start")
    s.path("M 700 275 C 760 275, 820 260, 875 248", LINE, 1.6)
    # cpu boxes
    s.rect(60, 380, 180, 70, "#ffffff", CPU_D, 6, 1)
    s.lines(150, 402, ["spin on routed == seq", "(host thread)"], 11, INK, "middle")
    s.rect(280, 380, 230, 70, "#ffffff", CPU_D, 6, 1)
    s.lines(395, 400, ["misses: experts with slot -1", "phase 1: gate/up rows (32-row units)",
                       "phase 2: down rows · 8 workers"], 11, INK, "middle")
    s.rect(550, 380, 190, 70, "#ffffff", CPU_D, 6, 1)
    s.lines(645, 402, ["sum per column (job order)", "write y_cpu, has_cpu"], 11, INK, "middle")
    s.rect(780, 380, 180, 70, "#ffffff", CPU_D, 6, 1)
    s.lines(870, 402, ["acked = seq", "(release store)"], 11, INK, "middle")
    s.arrow(240, 415, 280, 415)
    s.arrow(510, 415, 550, 415)
    s.arrow(740, 415, 780, 415)
    s.arrow(330, 305, 150, 380)
    s.arrow(870, 380, 700, 305)
    s.text(500, 500, "a hit costs the GPU ~13 MB of reads at ~600 GB/s; a miss costs the CPU ~13 MB at DRAM speed. "
           "With 97% hits the CPU's part is short and often hidden.", 11, MUTED, "middle")
    s.save("fig2-decode.svg")


# ---------------------------------------------------------------------------------------------------- fig 3
def fig3():
    s = Svg(1000, 470, "Figure 3: the prompt path for one layer of a chunk")
    s.text(500, 30, "Figure 3 · the prompt path (one layer, a chunk of up to 4,096 tokens)", 15, INK, "middle", "bold")
    row = [("RMSNorm", "→ int8"), ("QKV", "int8 WMMA"), ("RoPE", "KV write"), ("attention", "f16 WMMA, tiled"),
           ("O-proj", "int8 WMMA +res"), ("router", "top-4 on GPU")]
    x = 30
    for i, (a, b) in enumerate(row):
        s.rect(x, 60, 135, 52, GPU, GPU_D, 6, 1.2)
        s.lines(x + 67, 82, [a, b], 12, INK, "middle")
        if i:
            s.arrow(x - 22, 86, x, 86)
        x += 157
    s.arrow(900, 112, 900, 150)
    s.rect(560, 150, 410, 70, HOST, HOST_D, 8)
    s.lines(765, 172, ["host: copy top-4 back, group the assignments by expert;",
                       "order = [experts in VRAM] then [missing ones, batches of 8];",
                       "tiles of 32 tokens of one expert; pointer table per expert"], 11, INK, "middle")
    # streams
    s.rect(30, 250, 940, 90, GPU, GPU_D, 8)
    s.text(46, 270, "compute stream", 12, GPU_D, "start", "bold")
    blocks = [("VRAM experts: gate/up + down", 330, HOT), ("batch 1", 130, RAM), ("batch 2", 130, RAM),
              ("batch 3", 130, RAM), ("combine", 90, "#ffffff")]
    x = 46
    for lb, w, c in blocks:
        s.rect(x, 285, w, 40, c, GPU_D, 5, 1)
        s.text(x + w / 2, 310, lb, 12, INK, "middle")
        x += w + 12
    s.rect(30, 360, 940, 70, SSD, SSD_D, 8)
    s.text(46, 380, "copy stream (PCIe, 28.7 GB/s)", 12, SSD_D, "start", "bold")
    x = 46
    for i in range(3):
        s.rect(x + i * 150, 390, 140, 30, RAM, RAM_D, 5, 1)
        s.text(x + i * 150 + 70, 410, f"8 experts → ring {i % 2}", 11, INK, "middle")
    s.arrow(186, 390, 400, 325, dash="4 3", label="event", lx=245, ly=372)
    s.arrow(336, 390, 540, 325, dash="4 3")
    s.text(700, 410, "16-slot ring (2 x 8); a slot is refilled when the batch that read it has finished", 11, MUTED,
           "middle")
    s.text(500, 455, "each missing expert crosses PCIe once per chunk and layer, whatever the chunk's size: "
           "the reason bigger chunks help (and cost VRAM)", 11, MUTED, "middle")
    s.save("fig3-prompt.svg")


# ---------------------------------------------------------------------------------------------------- fig 4
def fig4():
    s = Svg(1000, 520, "Figure 4: a speculative step with the EAGLE3 draft")
    s.text(500, 30, "Figure 4 · a speculative step: draft k tokens, verify them in one target step", 15, INK,
           "middle", "bold")
    # target
    s.rect(30, 60, 560, 200, GPU, GPU_D, 10)
    s.text(46, 82, "target: gpt-oss-120b, one decode step over the window [t, d1 .. dk] (k ≤ 6)", 12, GPU_D, "start",
           "bold")
    for i, l in enumerate(["layers 0-23", "24-29", "30-35", "head"]):
        w = [200, 90, 90, 90][i]
        x = [50, 270, 380, 490][i]
        s.rect(x, 110, w, 50, "#ffffff", GPU_D, 6, 1)
        s.text(x + w / 2, 140, l, 12, INK, "middle")
    s.arrow(250, 135, 270, 135)
    s.arrow(360, 135, 380, 135)
    s.arrow(470, 135, 490, 135)
    for x, lb in ((262, "x→24"), (372, "x→30"), (482, "x after 35")):
        s.arrow(x, 160, x, 205, color=HOT_D)
        s.text(x, 222, lb, 11, HOT_D, "middle")
    s.text(300, 250, "taps: strided device copies inside the graph", 11, MUTED, "start")
    # accept
    s.rect(610, 60, 360, 200, HOST, HOST_D, 10)
    s.text(626, 82, "accept (host)", 12, HOST_D, "start", "bold")
    s.lines(626, 108, ["greedy: keep d_i while it is the target's argmax",
                       "sampling: keep d_i with probability p(d_i), else",
                       "   draw from p without d_i (exact sample of p)",
                       "add the target's own next token",
                       "drop the rejected positions' KV (truncate)",
                       "",
                       "same placement ⇒ same tokens as no drafts",
                       "(kernels built without -ffast-math)"], 12, INK)
    s.arrow(590, 160, 610, 160)
    # draft
    s.rect(30, 290, 940, 200, RAM, RAM_D, 10)
    s.text(46, 312, "drafts: prompt lookup first (free), else EAGLE3 (one decoder layer on GPU 1)", 12, RAM_D,
           "start", "bold")
    boxes = [(50, "features (3 x 2880)", "→ RMSNorm → fc", "g(p), into a ring"),
             (250, "row p: emb(token p+1)", "‖ norm(g(p))", "→ q, k, v (5760 in)"),
             (450, "attention (no sinks)", "own KV, llama RoPE", "base 500000, YaRN"),
             (650, "SwiGLU 16384 · res", "→ prenorm state", "(g for the next row)"),
             (830, "head: 49,152 ids", "argmax + its p", "stop if p < 0.6")]
    for x, a, b, c in boxes:
        w = 120 if x == 830 else 180
        s.rect(x, 330, w, 80, "#ffffff", RAM_D, 6, 1)
        s.lines(x + w / 2, 352, [a, b, c], 11, INK, "middle")
    for x in (230, 430, 630, 810):
        s.arrow(x, 370, x + 20, 370)
    s.path("M 740 410 C 740 470, 340 470, 340 412", RAM_D, 1.4, "4 3")
    s.text(540, 462, "drafting: each next row feeds the previous row's prenorm state instead of g", 11, MUTED,
           "middle")
    s.text(500, 485, "after a verify step the accepted positions' rows are recomputed with the target's real g "
           "(catch-up; prompts in 512-row pieces)", 11, MUTED, "middle")
    s.arrow(262, 230, 120, 330, color=HOT_D, dash="4 3")
    s.arrow(890, 330, 560, 262, color=RAM_D, dash="4 3", label="drafts d1..dk", lx=800, ly=284, lanchor="start")
    s.save("fig4-spec.svg")


# ---------------------------------------------------------------------------------------------------- fig 5
def fig5():
    s = Svg(1000, 400, "Figure 5: the MXFP4 expert blob and its two dot-product paths")
    s.text(500, 30, "Figure 5 · one expert's blob (12.6 MiB) and how both processors read it", 15, INK, "middle",
           "bold")
    s.rect(30, 60, 940, 70, SSD, SSD_D, 8)
    segs = [("gate qs", 254, GPU), ("gate e", 52, RAM), ("up qs", 254, GPU), ("up e", 52, RAM), ("down qs", 254, GPU),
            ("down e", 52, RAM)]
    x = 40
    for lb, w, c in segs:
        s.rect(x, 75, w - 4, 40, c, LINE, 4, 1)
        s.text(x + w / 2 - 2, 100, lb, 11, INK, "middle")
        x += w + 1
    s.text(500, 150, "planar: 16 code bytes per 32 weights (aligned uint4 loads), then one E8M0 scale byte per 32 — "
           "the same 17 bytes as ggml's block_mxfp4", 11, MUTED, "middle")
    # gpu path
    s.rect(30, 175, 460, 195, GPU, GPU_D, 10)
    s.text(46, 197, "GPU", 13, GPU_D, "start", "bold")
    s.lines(46, 222, ["4-bit codes → int8 with two byte-permutes (ggml's",
                      "   get_int_from_table_16, values {0,±1,±2,±3,±4,±6,±8,±12})",
                      "activations: int8, one float scale per 32",
                      "decode (1-8 tokens): sudot4 per lane, a wave per row",
                      "prompt: int8 WMMA 16x16x16, 32 tokens x 64/128 rows",
                      "   per tile, two WMMAs per 32-block, then",
                      "   acc += e8m0/2 · scale_tok · int_sum"], 12, INK)
    s.rect(510, 175, 460, 195, CPU, CPU_D, 10)
    s.text(526, 197, "CPU", 13, CPU_D, "start", "bold")
    s.lines(526, 222, ["4-bit codes → int8 with pshufb on the same table",
                       "same int8 activations (copied from the GPU's",
                       "   mailbox: identical rounding of the input)",
                       "|w| x sign(x) through vpdpbusd (AVX-512 VNNI),",
                       "   two rows at a time sharing the activation",
                       "per-block float scale, then swiglu, int8 again,",
                       "   down rows; workers claim 32-row units"], 12, INK)
    s.save("fig5-mxfp4.svg")


# ---------------------------------------------------------------------------------------------------- fig 6
def fig6():
    s = Svg(1000, 470, "Figure 6: the measured effect of each step")
    s.text(500, 30, "Figure 6 · each step, measured (gpt-oss-120b, same PC)", 15, INK, "middle", "bold")
    # left: prompt at 16K
    data_p = [("first version (f16 dequant + hipBLAS)", 0), ("grouped int8 expert kernels", 857),
              ("WMMA attention", 1549), ("int8 WMMA experts + 4K chunks", 1981), ("int8 WMMA dense", 2705)]
    data_o = [("first start, no profile", 60.4), ("saved profile", 103.6), ("+ prompt lookup (prose)", 103.4),
              ("+ EAGLE3, one rate", 120.1), ("+ EAGLE3, p_min 0.6", 123.4)]

    def bars(x0, title, data, vmax, unit, color, stroke):
        s.text(x0, 70, title, 13, INK, "start", "bold")
        for i, (lb, v) in enumerate(data):
            y = 90 + i * 66
            s.text(x0, y + 14, lb, 11, MUTED)
            w = 380 * v / vmax
            if v > 0:
                s.rect(x0, y + 22, w, 24, color, stroke, 3, 1)
                s.text(x0 + w + 6, y + 39, f"{v:,.1f} {unit}" if v < 200 else f"{v:,.0f} {unit}", 11, INK)
            else:
                s.text(x0, y + 39, "not measured at 16K (415 tok/s at 1K)", 11, MUTED, italic=True)

    bars(40, "prompt, 16,000 tokens (tok/s)", data_p, 2800, "", GPU, GPU_D)
    bars(540, "output, prose answer (tok/s)", data_o, 130, "", HOT, HOT_D)
    s.text(500, 440, "llama.cpp on the same PC, best hybrid split: 791 tok/s prompt (512 tokens), 50.5 tok/s output",
           12, MUTED, "middle")
    s.text(500, 458, "16K prompt before the WMMA attention: 857 tok/s with the grouped kernels; 32K output: 52 → 71 "
           "tok/s from the decode attention rewrite", 11, MUTED, "middle")
    s.save("fig6-steps.svg")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    print("written to", OUT)
