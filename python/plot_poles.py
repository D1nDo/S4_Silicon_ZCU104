#!/usr/bin/env python3
"""
plot_poles.py -- poles of the QUANTIZED biquads of the four models, read from the
weight streams in mem/mem_<DS>/ with the bit-exact model (python/s4d_golden.py).

Each biquad has denominator 1 + a1 z^-1 + a2 z^-2, with a1, a2 24-bit integers
with 22 fractional bits; the poles are the roots of z^2 + a1 z + a2. The figure
shows |z| and the distance 1 - |z| from the unit circle, with the quantization
limit: the largest a2 below 1 is 1 - 2^-22, so the largest radius of a complex
pair is sqrt(1 - 2^-22) and the smallest distance is about 2^-23.

Outputs: images/quantized_poles.{pdf,png} and, for each model,
images/poles_<DS>.txt (one row per biquad).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import s4d_golden

OUT = Path(__file__).resolve().parent.parent / "images"
F_A = 22
MODELS = [("ECG", "PTB-XL"), ("RadioML", "RadioML"), ("SC", "Speech C."), ("sCIFAR", "sCIFAR")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def model_poles(dataset):
    model = s4d_golden.S4DGolden(dataset)
    rows = []
    for m, layer in enumerate(model.layers):
        a1, a2 = layer["a1"], layer["a2"]
        for h in range(model.H):
            for n in range(model.MODES):
                r = np.roots([1.0, a1[h, n] / 2 ** F_A, a2[h, n] / 2 ** F_A])
                r = sorted(np.append(r, 0.0)[:2], key=lambda z: -abs(z))
                rows.append([m, h, n, a1[h, n], a2[h, n], r[0].real, r[0].imag, abs(r[0]), r[1].real, r[1].imag, abs(r[1])])
    return model, np.array(rows)


r_max = np.sqrt(1 - 2.0 ** -F_A)
d_min = 1 - r_max
data = []
for (ds, label), col in zip(MODELS, COLORS):
    model, rows = model_poles(ds)
    mags = np.concatenate([rows[:, 7], rows[:, 10]])
    data.append((f"{label} ({model.H}, {2 * model.MODES})", mags, col))
    np.savetxt(OUT / f"poles_{ds}.txt", rows, comments="",
               header="layer channel mode a1_int a2_int z1_re z1_im z1_abs z2_re z2_im z2_abs",
               fmt=["%d"] * 5 + ["%.10f"] * 6)
    print(f"{label}: {len(rows)} biquads, max |z| = {mags.max():.8f}, min 1-|z| = {(1 - mags).min():.3e}, "
          f"poles with |z| >= 1: {int((mags >= 1).sum())}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))
b1 = np.linspace(0.95, 1.0, 101)
b2 = np.logspace(np.floor(np.log10(d_min)) - 0.5, 0, 81)
for name, mags, col in data:
    ax1.hist(np.clip(mags, b1[0], 1), bins=b1, histtype="step", lw=2, color=col)
    ax2.hist(1 - mags, bins=b2, histtype="step", lw=2, color=col)
for ax in (ax1, ax2):
    ax.set_yscale("log"); ax.set_ylabel("poles")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e0", lw=0.6)
ax1.axvline(r_max, color="#555", ls="--", lw=1)
ax1.set_xlabel(r"$|z|$ (poles with $|z|<0.95$ in the first bin)")
ax2.set_xscale("log"); ax2.axvline(d_min, color="#555", ls="--", lw=1)
ax2.set_xlabel(r"$1-|z|$ (distance from the unit circle)")
ax2.text(d_min, ax2.get_ylim()[1], r" $1-\sqrt{1-2^{-22}}\approx2^{-23}$", ha="left", va="top", fontsize=8, color="#555")
fig.legend([Line2D([], [], color=c, lw=2) for _, _, c in data], [d[0] for d in data],
           frameon=False, fontsize=8, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"quantized_poles.{ext}", dpi=200)
