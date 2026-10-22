#!/usr/bin/env python3
"""Block diagram of the S4D accelerator on ZCU104 -> images/architecture.pdf and .png

Usage: python arch_diagram.py   (needs matplotlib)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "images"
INK, MUTED, LINE = "#1f1f1e", "#6b6a66", "#3d3d3a"
PS_FILL, PL_FILL = "#f0efec", "#ffffff"
BLOCK, ACCENT, ACCENT_TXT = "#e8eef8", "#256abf", "#ffffff"
WEIGHT = "#fbe9d0"

fig, ax = plt.subplots(figsize=(13.0, 8.6))
ax.set_xlim(0, 130); ax.set_ylim(0, 86); ax.axis("off")


def box(x, y, w, h, title, lines=(), fill=BLOCK, edge=LINE, title_color=INK, size=10, lw=1.2, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=size,
            fontweight="bold", color=title_color)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.6 - 2.6 * i, line, ha="center", va="top", fontsize=8,
                color=title_color if fill == ACCENT else MUTED)


def arrow(x0, y0, x1, y1, label="", label_dx=0.0, label_dy=1.0, color=LINE, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2 + label_dx, (y0 + y1) / 2 + label_dy, label, ha=ha, va="bottom",
                fontsize=7.5, color=MUTED)


# ---------------------------------------------------------------- PS
ax.add_patch(FancyBboxPatch((2, 59), 126, 25, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor=PS_FILL, edgecolor=MUTED, linewidth=1.0))
ax.text(4, 82.3, "Processing System (Zynq UltraScale+ PS)", fontsize=10, fontweight="bold", color=INK, va="top")
box(6, 62, 30, 15, "ARM Cortex-A53", ["Linux (PYNQ image)", "test program via /dev/mem", "pl_clk0 set by pl0_clk_set"],
    fill=PL_FILL)
box(92, 62, 32, 15, "DDR4", ["input buffer   0x7000_0000", "logits buffer  0x7000_8000", "(DMA buffers)"],
    fill=PL_FILL)
box(42, 67, 18, 6, "M_AXI_HPM0_FPD", fill=PL_FILL, size=8)
box(68, 67, 18, 6, "S_AXI_HPC0_FPD", fill=PL_FILL, size=8)
arrow(36, 70, 42, 70)
arrow(86, 70, 92, 70)

# ---------------------------------------------------------------- PL
ax.add_patch(FancyBboxPatch((2, 1), 126, 55, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor=PL_FILL, edgecolor=MUTED, linewidth=1.0))
ax.text(4, 54.3, "Programmable Logic", fontsize=10, fontweight="bold", color=INK, va="top")
ax.text(126, 2.5, "single clock pl_clk0: 213.33 MHz in the block design, 214.28 MHz in the board runs",
        fontsize=8, color=MUTED, ha="right", va="bottom")

box(42, 40.5, 18, 8, "axi_smc", ["SmartConnect"], size=9)
box(68, 40.5, 18, 8, "axi_smc_1", ["SmartConnect, 2 SI"], size=9)
box(42, 29, 44, 8, "axi_dma_0", ["simple mode, registers @ 0xA000_0000"], size=9)
arrow(51, 67, 51, 48.5, "AXI-Lite (DMA registers)", label_dx=1.0, label_dy=-1, ha="left")
arrow(51, 40.5, 51, 37)
arrow(77, 37, 77, 40.5, "AXI4 (MM2S + S2MM)", label_dx=1.0, label_dy=-1.5, ha="left")
arrow(77, 48.5, 77, 67, "DDR access", label_dx=1.0, label_dy=-1, ha="left")

# datapath
box(6, 14, 22, 11, "Encoder", ["NIN inputs x N_BIT_IN", "-> 128 channels, Q5.11"], size=9)
box(34, 13, 48, 13, "S4D core  (s4d_core_axis)", ["6 layers, H = 128, L = 1024",
                                                  "norm | biquad bank 32x4x32 | GELU",
                                                  "mixing 256 MAC (URAM) + GLU | residual"],
    fill=ACCENT, edge=ACCENT, title_color=ACCENT_TXT, size=9)
box(88, 14, 24, 11, "Decoder", ["affine norm, mean pool", "INT8 classifier -> K logits"], size=9)
box(46, 2.5, 24, 8, "Weight ROM", ["(s4d_weightrom_axis), 6 x 14977 beats"], fill=WEIGHT, size=9)

arrow(46, 29, 17, 25, "MM2S, 64 bit: input sequence", label_dx=-2, label_dy=0.8)
arrow(28, 19.5, 34, 19.5, "16 bit", label_dy=0.4)
arrow(82, 19.5, 88, 19.5, "16 bit", label_dy=0.4)
arrow(100, 25, 84, 29, "S2MM, 32 bit: K logits", label_dx=6, label_dy=0.8)
WCOL = "#b56a10"
arrow(54, 10.5, 54, 13, color=WCOL)
ax.text(53, 11.2, "weights, 64 bit", ha="right", va="center", fontsize=7.5, color=WCOL)
ax.annotate("", xy=(62, 10.5), xytext=(62, 13),
            arrowprops=dict(arrowstyle="-|>", color=WCOL, lw=1.0, linestyle="--"))
ax.text(63, 11.2, "wload_start / wload_layer", ha="left", va="center", fontsize=7.5, color=WCOL)

fig.tight_layout()
fig.savefig(OUT / "architecture.pdf")
fig.savefig(OUT / "architecture.png", dpi=150)
