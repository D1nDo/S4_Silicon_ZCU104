"""
train_and_quantize_bn.py -- training from scratch + BatchNorm quantization
for the S4D accelerator. Standalone: it includes gen_s4d_mem, s4d_export, s4d_bn.
"""

from pathlib import Path
import sys, os, json, math, time, shutil, types

FILL      = 0.75
N_CAL     = 128
N_EVAL_FX = 10000      # was 32: with 32 images the standard error on acc_fx is ~+-9 points
CHUNK_FX  = 64       # blocks of the fixed check; see the comment where it is used
SEED      = 1337
MAX_EPOCHS   = 100
TRAIN_SUBSET = None
BATCH = 64
PATIENCE = 10

REPO = WORK = Path(__file__).resolve().parent
DATA_ROOT = REPO / "data"

MEM_OUT  = WORK / "mem_trained_bn"
CKPT_DIR = WORK / "ckpt"
LOG_DIR  = WORK / "logs"
for d in (MEM_OUT, CKPT_DIR, LOG_DIR, DATA_ROOT):
    d.mkdir(parents=True, exist_ok=True)

BEST_CKPT = CKPT_DIR / "s4d_scifar_configC_bn_best.pt"
LAST_CKPT = CKPT_DIR / "last_bn.pt"
RUN_LOG   = LOG_DIR / "train_log_bn.jsonl"


# ============================================================================
# gen_s4d_mem / s4d_export / s4d_bn modules, embedded as separate modules
# ============================================================================

def _load_module(name, source):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    exec(compile(source, name, "exec"), mod.__dict__)
    return mod


_GEN_S4D_MEM_SRC = r'''
#!/usr/bin/env python3
"""
gen_s4d_mem.py -- coefficients and golden model of an S4D layer.

It does the work that in S4D_matematica_completa.tex belongs to the training and
not to the hardware: complex exponentials, ZOH, reduction to a real biquad. The
FPGA receives only four constant coefficients and two registers per pole pair.

  1. lambda = -exp(log_A_real) + i*A_imag,  Delta = exp(log_dt)
  2. ZOH:    lbar = exp(Delta*lambda),  Bbar = (lbar-1)/lambda
  3. biquad: a1 = -2Re(lbar), a2 = |lbar|^2,
             b0 = 2Re(C*Bbar), b1 = -2Re(lbar * conj(C*Bbar))
  4. JURY CLAMP on the quantized coefficients  <-- see below
  5. .mem export + bit-accurate model + comparison with the float

------------------------------------------------------------------------------
THE JURY CLAMP
------------------------------------------------------------------------------
Quantizing a1 and a2 independently does not preserve the position of the pole.
The critical case is mode n=0 of S4D-Lin, where A_imag = 0: the pole is real and
DOUBLE and the distance from the unit circle is

    1 + a1 + a2 = (1 - lbar)^2

i.e. a SECOND-order infinitesimal. For tau = 1024 it is 9.5e-7: below the
quantum of any format with fewer than 21 fractional bits. Rounding therefore
puts a root exactly on z=1 and the filter becomes an integrator.

The remedy costs nothing in hardware: the coefficients are pulled inside the
Jury triangle  (a2 < 1,  |a1| < 1 + a2)  leaving one quantum of margin.
Checked on a realistic S4D-Lin grid: 3/4096 unstable biquads at 24 bits
(75/4096 at 16 bits), zero after the clamp, with a perturbation on a1 of one
quantum. The time constant stays correct within 0.4% even at 16 bits.
"""

import argparse, json, math, os
import numpy as np

# ---------------------------------------------------------------- config
# Folded architecture: NUNITS=8 biquad units of NCH=16 channels (was 64x2) and
# NMAC=64 MACs in the mixing (were TH=256, one per row of W_out). Each MAC serves
# NPH = TH/NMAC = 4 rows, in NPH phases of H cycles.
H, MODES, NUNITS, NCH = 128, 32, 8, 16
TH = 2 * H
NMAC = 64
assert TH % NMAC == 0, "TH must be a multiple of NMAC"
NPH = TH // NMAC
assert H == NUNITS * NCH, "H must be NUNITS*NCH"

W_ACT, F_ACT = 16, 11
W_PAR, F_GAMMA = 16, 13
W_U, F_U = 16, 11
W_A, F_A = 24, 22
W_B, F_B = 16, 15
W_S, F_Y = 32, 19
W_D, F_D = 16, 14
W_BACC = 40
W_SSM, F_SSM = 16, 11
W_W, F_W = 12, 11
W_MACC = 32
F_SIGI, F_SIGO = 11, 14

# SH_B is PER CHANNEL, not global: see the comment in s4d_biquad_unit.v
SH_B0 = F_B + F_U - F_Y
SH_D0 = F_D + F_U - F_Y   # nominal; the real value is calibrated in main()
SH_A = F_A
SH_SSM = F_Y - F_SSM
FACC = F_W + F_SSM
RQ_G = FACC - F_SIGI
RQ_A = FACC - F_ACT

EPS_REAL = 1e-5
EPS_NUM = max(1, int(round(EPS_REAL * 2.0 ** (2 * F_ACT) * H ** 2)))


# ---------------------------------------------------------------- fixed point
def shr_round(x, s):
    if s <= 0:
        return x << (-s)
    return (x + (1 << (s - 1))) >> s


def sat(x, w):
    lo, hi = -(1 << (w - 1)), (1 << (w - 1)) - 1
    return max(lo, min(hi, x))


def satw(v, s, w=W_S):
    return sat(shr_round(v, s), w)


def calib(x, w, hard=None):
    """Fractional bits that fill w bits without saturating. Setting them by eye
    is the easiest way to lose 20 dB without noticing: with F_D = 14 (range
    +-2) the values of D ~ N(0,1) with tails at 3.5 were clipped, and the S4D
    branch dropped to 18 dB of SNR."""
    m = float(np.abs(x).max()) if hard is None else hard
    if m <= 0:
        return w - 1
    return int(math.floor(math.log2(((1 << (w - 1)) - 1) / m)))


def q(arr, f, w):
    v = np.round(np.asarray(arr, dtype=np.float64) * 2.0 ** f)
    return np.clip(v, -(1 << (w - 1)), (1 << (w - 1)) - 1).astype(np.int64)


def hx(v, w):
    """Formats v on w bits in hexadecimal, rejecting the values that do not fit.

    The previous version masked with `& ((1 << w) - 1)` and returned a
    perfectly valid string even when the value was too large: a field in
    overflow disappeared without a trace, and the RTL downstream read wrong
    numbers without raising any flag. That is exactly how the bsh field
    (4 bits, calibrated values up to 22) stayed broken and invisible.

    Now the error is loud. Both unsigned integers in [0, 2^w) and signed ones
    in [-2^(w-1), 2^(w-1)) are accepted, the latter converted to two's
    complement: it is the case of all the quantized tensors, which go through
    q() and are already saturated to the right format.
    """
    v = int(v)
    lo, hi = -(1 << (w - 1)), (1 << w) - 1
    if not (lo <= v <= hi):
        raise ValueError(
            f"hx: {v} does not fit in {w} bits (allowed [{lo}, {hi}]). "
            f"Widen the field or fix the calibration: do NOT mask."
        )
    return format(v & ((1 << w) - 1), 'x').zfill((w + 3) // 4)


def write(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  {os.path.basename(path):18s} {len(lines):>6d} rows")


# ---------------------------------------------------------------- biquad
def to_biquad(log_A_real, A_imag, log_dt, C):
    lam = -np.exp(log_A_real) + 1j * A_imag
    dt = np.exp(log_dt)[:, None]
    lbar = np.exp(dt * lam)
    Bbar = (lbar - 1) / lam
    r = C * Bbar
    return (-2 * lbar.real, np.abs(lbar) ** 2,
            2 * r.real, -2 * (lbar * np.conj(r)).real, lbar)


def jury_clamp(a1i, a2i, f):
    """Brings the quantized coefficients back inside the stability triangle."""
    one = 1 << f
    a2i = np.minimum(a2i, one - 1)
    lim = one + a2i - 1
    return np.clip(a1i, -lim, lim), a2i


# ---------------------------------------------------------------- PWL
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_pwl(fn, f_out, xh=3, nseg=256, w=16):
    """{slope, base} LUT with linear interpolation on [-2^xh, 2^xh)."""
    xs = -(2.0 ** xh) + np.arange(nseg + 1) * (2.0 ** (xh + 1)) / nseg
    ys = q(fn(xs), f_out, w)
    base, slope = ys[:-1], ys[1:] - ys[:-1]
    return base, np.clip(slope, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def pwl_eval(x, base, slope, f_in, f_out, xh=3, nseg=256, log2n=8,
             mode_hi=1, y_lo=0, y_hi=0, w=16):
    fracw = f_in + xh + 1 - log2n
    offs = 1 << (xh + f_in)
    if x < -offs:
        return y_lo
    if x >= offs:
        return sat(x, w) if mode_hi else y_hi
    xs = x + offs
    i, fr = xs >> fracw, xs & ((1 << fracw) - 1)
    return sat(int(base[i]) + shr_round(int(slope[i]) * fr, fracw), w)


# ---------------------------------------------------------------- RTL model
class LayerModel:
    """Bit-accurate mirror of s4d_layer.v."""

    def __init__(self, p):
        self.p = p
        self.s1 = np.zeros((H, MODES), dtype=object)
        self.s2 = np.zeros((H, MODES), dtype=object)

    def layernorm(self, x):
        p = self.p
        s = int(np.sum(x))
        ssq = int(np.sum(np.asarray(x, dtype=object) ** 2))
        vnum = (ssq << 7) - s * s
        v = vnum + EPS_NUM
        msb = v.bit_length() - 1
        e2 = msb & ~1
        sh = (v >> (e2 - 6)) if e2 >= 6 else (v << (6 - e2))
        inv = p['isqrt_lut'][sh & 0xFF]
        shift = 15 + F_GAMMA - F_U + e2 // 2
        return [sat(shr_round(((int(x[h]) << 7) - s) * inv * int(p['gamma'][h]),
                              shift) + int(p['beta'][h]), W_U)
                for h in range(H)]

    def biquads(self, u):
        p = self.p
        out = []
        for h in range(H):
            uh = int(u[h])
            acc = satw(int(p['D'][h]) * uh, p['sh_d'])      # skip D*u
            for n in range(MODES):
                a1, a2 = int(p['a1'][h, n]), int(p['a2'][h, n])
                b0, b1 = int(p['b0'][h, n]), int(p['b1'][h, n])
                s1, s2 = self.s1[h, n], self.s2[h, n]
                shb = int(p['bsh'][h])
                y = satw(b0 * uh + (s1 << shb), shb)
                a1y, a2y = satw(a1 * y, SH_A), satw(a2 * y, SH_A)
                self.s1[h, n] = satw(satw(b1 * uh, shb) - a1y + s2, 0)
                self.s2[h, n] = -a2y
                acc += y
            out.append(sat(shr_round(acc, SH_SSM), W_SSM))
        return out

    def step(self, x):
        p = self.p
        u = self.layernorm(x)
        yssm = self.biquads(u)
        g = [pwl_eval(v, p['gelu_b'], p['gelu_s'], F_SSM, F_SSM, mode_hi=1)
             for v in yssm]

        acc = [int(p['bout'][c]) for c in range(TH)]
        for c in range(TH):
            for h in range(H):
                acc[c] += int(p['wout'][c, h]) * g[h]
            acc[c] = sat(acc[c], W_MACC)

        out = []
        for h in range(H):
            a = sat(shr_round(acc[h], RQ_A), 16)
            gg = sat(shr_round(acc[H + h], RQ_G), 16)
            sg = pwl_eval(gg, p['sig_b'], p['sig_s'], F_SIGI, F_SIGO,
                          mode_hi=0, y_lo=0, y_hi=1 << F_SIGO)
            out.append(sat(shr_round(a * sg, F_SIGO), W_ACT))
        return [sat(int(x[h]) + out[h], W_ACT) for h in range(H)]


# ---------------------------------------------------------------- float
class LayerFloat:
    """Same layer in floating point: reference to measure what the fixed point
    really costs. It uses the same NON-quantized biquad coefficients."""

    def __init__(self, gamma, beta, a1, a2, b0, b1, D, wout, bout):
        self.g, self.b = gamma, beta
        self.a1, self.a2, self.b0, self.b1 = a1, a2, b0, b1
        self.D, self.w, self.bo = D, wout, bout
        self.s1 = np.zeros((H, MODES))
        self.s2 = np.zeros((H, MODES))

    def step(self, x):
        mu, var = x.mean(), x.var()
        u = self.g * (x - mu) / np.sqrt(var + EPS_REAL) + self.b

        uu = u[:, None]
        y = self.b0 * uu + self.s1
        self.s1 = self.b1 * uu - self.a1 * y + self.s2
        self.s2 = -self.a2 * y
        yssm = y.sum(axis=1) + self.D * u

        gl = gelu(yssm)
        v = self.w @ gl + self.bo
        return x + v[:H] * sigmoid(v[H:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../mem')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # ---- S4D-Lin parameters ------------------------------------------------
    log_A_real = np.log(0.5) + rng.normal(0, 0.3, (H, MODES))
    A_imag = np.pi * np.arange(MODES)[None, :] * np.ones((H, 1))
    log_dt = rng.uniform(np.log(1e-3), np.log(1e-1), H)
    C = (rng.normal(0, 1, (H, MODES)) + 1j * rng.normal(0, 1, (H, MODES))) / np.sqrt(2)
    D = rng.normal(0, 1, H)
    gamma_pre = rng.normal(1.0, 0.15, H)
    gamma = gamma_pre

    a1f, a2f, b0f, b1f, lbar = to_biquad(log_A_real, A_imag, log_dt, C)

    print("coefficient ranges:")
    for nm, v in [('a1', a1f), ('a2', a2f), ('b0', b0f), ('b1', b1f)]:
        print(f"  {nm}: [{v.min():+.5f}, {v.max():+.5f}]  max|.| {np.abs(v).max():.5f}")

    # saturation check on all the tensors before quantizing
    print("\ncalibration of the exponents (max|.| -> fractional bits):")
    for nm, v, w, fx in [('a1', a1f, W_A, F_A), ('a2', a2f, W_A, F_A),
                         ('gamma', gamma_pre, W_PAR, F_GAMMA)]:
        c = calib(v, w)
        flag = '' if c >= fx else f'  <-- F={fx} SATURATES, {c} would be needed'
        print(f"  {nm:7s} max {np.abs(v).max():8.4f}  suggested F {c:3d}{flag}")

    a1i, a2i = q(a1f, F_A, W_A), q(a2f, F_A, W_A)
    bad = int(((np.abs(a1i) >= (1 << F_A) + a2i) | (a2i >= (1 << F_A))).sum())
    a1i, a2i = jury_clamp(a1i, a2i, F_A)
    good = int(((np.abs(a1i) >= (1 << F_A) + a2i) | (a2i >= (1 << F_A))).sum())
    print(f"\nJury clamp: {bad}/{a1i.size} unstable biquads before, {good} after")

    # per-channel exponent on b0,b1: fills the 16 bits on the maximum of that channel
    bmax = np.maximum(np.abs(b0f), np.abs(b1f)).max(axis=1)
    fb_ch = np.floor(np.log2(((1 << (W_B - 1)) - 1) / np.maximum(bmax, 1e-30))).astype(int)
    fb_ch = np.clip(fb_ch, F_B, F_B + 15)
    bsh = fb_ch + F_U - F_Y                       # shift applied in hardware
    print(f"per-channel b exponent: F_B in [{fb_ch.min()}, {fb_ch.max()}], "
          f"shift in [{bsh.min()}, {bsh.max()}]  (global would be {SH_B0})")
    print(f"  dynamic range of max|b| between channels: {bmax.max()/bmax.min():.0f}x")

    # The bsh field is W_BSH bits wide (5 after the widening; it was 4, and the
    # values above 15 were silently truncated).
    #
    # The real limit is not the field but the datapath: in s4d_biquad_unit the sum
    # is (b0*u + s1<<bsh) >> bsh on PW = W_A+W_S bits, and with |s1| < 2^(W_S-1)
    # W_S + bsh <= PW - 1 is needed, i.e. bsh <= W_A - 1 = 23. The clip of fb_ch at
    # F_B+15 keeps bsh <= 22, one bit below the limit.
    W_BSH   = 5
    BSH_MAX = W_A - 1
    assert bsh.max() < (1 << W_BSH), (
        f"bsh max {bsh.max()} does not fit in {W_BSH} bits: widen the field in "
        f"gen_s4d_mem.py, rtl/s4d_biquad_{{unit,bank}}.v, rtl/s4d_layer.v and in "
        f"the corresponding VHDL")
    assert bsh.max() <= BSH_MAX, (
        f"bsh max {bsh.max()} exceeds the datapath limit ({BSH_MAX}): "
        f"s1<<bsh exceeds the {W_A+W_S} bits of PW and silently overflows")
    assert bsh.min() >= SH_B0, (
        f"bsh min {bsh.min()} below SH_B0 = {SH_B0}: impossible by "
        f"construction, it indicates a truncation upstream")

    b0i = np.stack([q(b0f[h], fb_ch[h], W_B) for h in range(H)])
    b1i = np.stack([q(b1f[h], fb_ch[h], W_B) for h in range(H)])
    f_d = calib(D, W_D)
    print(f"  D       max {np.abs(D).max():8.4f}  F used        {f_d:3d}"
          f"  (was fixed at {F_D})")
    Di = q(D, f_d, W_D)

    beta = rng.normal(0.0, 0.10, H)
    wout = rng.normal(0, 0.08, (TH, H))
    bout = rng.normal(0, 0.02, TH)
    gi, bi = q(gamma, F_GAMMA, W_PAR), q(beta, F_ACT, W_PAR)
    wi = q(wout, F_W, W_W)
    boi = q(bout, FACC, W_MACC)

    # ---- LUT ---------------------------------------------------------------
    isq = [min(65535, int(round(32768 / math.sqrt((max(i, 64) + .5) / 64))))
           for i in range(256)]
    gb, gs = build_pwl(gelu, F_SSM)
    sb, ss = build_pwl(sigmoid, F_SIGO)

    print(f"\nwriting in {os.path.abspath(a.out)}")
    write(os.path.join(a.out, 'isqrt_lut.mem'), [hx(v, 16) for v in isq])
    write(os.path.join(a.out, 'ln_gamma.mem'), [hx(v, 16) for v in gi])
    write(os.path.join(a.out, 'ln_beta.mem'), [hx(v, 16) for v in bi])
    write(os.path.join(a.out, 'gelu_lut.mem'),
          [hx((int(s) & 0xFFFF) << 16 | (int(b) & 0xFFFF), 32) for b, s in zip(gb, gs)])
    write(os.path.join(a.out, 'sigmoid_lut.mem'),
          [hx((int(s) & 0xFFFF) << 16 | (int(b) & 0xFFFF), 32) for b, s in zip(sb, ss)])
    # D and the b shift travel together: {bsh[W_BSH-1:0], D[W_D-1:0]}
    # The bsh field went from 4 to 5 bits: the position does not change (it
    # always starts at bit W_D), so the files written with 4 bits stay readable.
    write(os.path.join(a.out, 's4d_D.mem'),
          [hx((int(bsh[h]) << W_D) | (int(Di[h]) & ((1 << W_D) - 1)), W_D + W_BSH)
           for h in range(H)])

    # coefficients: order (unit, slot, mode); unit j serves the channels
    # j*NCH .. j*NCH+NCH-1. Expanding idx = j*NCH*MODES + slot*MODES + n with
    # h = j*NCH + slot gives idx = h*MODES + n: the file does NOT depend on
    # NUNITS, and it is invariant with respect to the folding of the bank.
    lines = []
    for j in range(NUNITS):
        for slot in range(NCH):
            h = j * NCH + slot
            for n in range(MODES):
                word = ((int(b1i[h, n]) & 0xFFFF) << 64 |
                        (int(b0i[h, n]) & 0xFFFF) << 48 |
                        (int(a2i[h, n]) & 0xFFFFFF) << 24 |
                        (int(a1i[h, n]) & 0xFFFFFF))
                lines.append(hx(word, 80))
    write(os.path.join(a.out, 's4d_coef.mem'), lines)

    # W_out, format of the folded mixing: NPH*H rows of NMAC*W_W bits,
    #     row(r*H + h) [m*W_W +: W_W] = wout[r*NMAC + m][h]
    # with h in NATURAL order. The folded bank reorders its outputs in a
    # buffer and emits 0..127 in sequence, so here there is no permutation:
    # before, the rows were ordered even-then-odd because that was the emission
    # order of the 64x2 bank.
    # The previous H rows of TH*W_W = 3072 bits did not fit in any BRAM port
    # and ended up in distributed registers.
    rows = []
    for r in range(NPH):
        for h in range(H):
            word = 0
            for m in range(NMAC):
                word |= (int(wi[r * NMAC + m, h]) & ((1 << W_W) - 1)) << (m * W_W)
            rows.append(hx(word, NMAC * W_W))
    write(os.path.join(a.out, 's4d_wout.mem'), rows)
    write(os.path.join(a.out, 's4d_bout.mem'), [hx(v, W_MACC) for v in boi])

    # ---- bit-accurate model -----------------------------------------------
    p = dict(isqrt_lut=isq, gamma=gi, beta=bi, a1=a1i, a2=a2i, b0=b0i, b1=b1i,
             D=Di, sh_d=f_d + F_U - F_Y, bsh=bsh, wout=wi, bout=boi, gelu_b=gb, gelu_s=gs, sig_b=sb, sig_s=ss)
    m = LayerModel(p)

    x = q(rng.normal(0, 1.5, (a.steps, H)), F_ACT, W_ACT)
    outs, stim = [], []
    for k in range(a.steps):
        stim.extend(int(v) for v in x[k])
        outs.append(m.step(x[k]))

    write(os.path.join(a.out, 's4d_stim.mem'), [hx(v, W_ACT) for v in stim])
    write(os.path.join(a.out, 's4d_exp.mem'),
          [hx(v, W_ACT) for row in outs for v in row])

    o = np.array(outs, dtype=np.float64) / 2 ** F_ACT
    print(f"\nlayer output (fixed point): min {o.min():+.4f} max {o.max():+.4f} "
          f"rms {np.sqrt((o**2).mean()):.4f}")
    st = np.array([[int(v) for v in x[k]] for k in range(a.steps)]) / 2 ** F_ACT
    print(f"input:                      min {st.min():+.4f} max {st.max():+.4f} "
          f"rms {np.sqrt((st**2).mean()):.4f}")

    sat_out = int((np.abs(np.array(outs)) >= (1 << (W_ACT - 1)) - 1).sum())
    print(f"saturated samples at the output: {sat_out}/{a.steps*H}")

    # ---- comparison with the float reference --------------------------------
    # TWO references, to separate two errors that add up but have different
    # causes:
    #   (A) exact coefficients -> includes the pole shift due to the
    #       quantization of a1,a2. For high-Q resonant filters a pole shifted by
    #       one quantum progressively dephases the output: it is an unavoidable
    #       error, not a defect of the datapath.
    #   (B) same quantized coefficients -> isolates the datapath arithmetic
    #       (state, rounding, saturation, LUT). This is the number that tells
    #       whether the RTL is sized correctly.
    fl = LayerFloat(gamma, beta, a1f, a2f, b0f, b1f, D, wout, bout)
    ref = np.array([fl.step(x[k] / 2.0 ** F_ACT) for k in range(a.steps)])

    flq = LayerFloat(gi / 2.0**F_GAMMA, bi / 2.0**F_ACT,
                     a1i / 2.0**F_A, a2i / 2.0**F_A,
                     b0i / 2.0**fb_ch[:, None], b1i / 2.0**fb_ch[:, None],
                     Di / 2.0**f_d, wi / 2.0**F_W, boi / 2.0**FACC)
    refq = np.array([flq.step(x[k] / 2.0 ** F_ACT) for k in range(a.steps)])
    err = o - ref
    print("\n--- fixed point vs float ---")
    print(f"max absolute error   = {np.abs(err).max():.5f}")
    print(f"rms error            = {np.sqrt((err**2).mean()):.5f}")
    print(f"signal rms           = {np.sqrt((ref**2).mean()):.5f}")
    print(f"SNR                  = {20*np.log10(np.sqrt((ref**2).mean())/max(np.sqrt((err**2).mean()),1e-12)):.1f} dB")

    # contribution of the S4D branch alone (the output is dominated by the residual)
    br_f = ref - x[:a.steps] / 2.0 ** F_ACT
    br_q = o   - x[:a.steps] / 2.0 ** F_ACT
    be = br_q - br_f
    xr = x[:a.steps] / 2.0 ** F_ACT
    br_fq = refq - xr

    def snr(sig, err):
        return 20*np.log10(np.sqrt((sig**2).mean()) /
                           max(np.sqrt((err**2).mean()), 1e-12))

    print(f"\nS4D branch (the total output is dominated by the residual):")
    print(f"  rms of the branch                {np.sqrt((br_f**2).mean()):.5f}")
    print(f"  (A) vs exact coefficients        SNR {snr(br_f, br_q-br_f):5.1f} dB"
          f"   <- includes the quantization of the coefficients")
    print(f"  (B) vs same quantized coeffs     SNR {snr(br_fq, br_q-br_fq):5.1f} dB"
          f"   <- datapath arithmetic only")
    print(f"  share due to the coefficients    SNR {snr(br_f, br_fq-br_f):5.1f} dB")

    with open(os.path.join(a.out, 's4d_config.vh'), 'w') as f:
        f.write("// generated by gen_s4d_mem.py -- do not edit by hand\n")
        for k, v in [('TB_H', H), ('TB_MODES', MODES), ('TB_NUNITS', NUNITS),
                     ('TB_NCH', NCH), ('TB_NMAC', NMAC), ('TB_NPH', NPH),
                     ('TB_STEPS', a.steps), ('TB_EPS_NUM', EPS_NUM),
                     ('TB_F_A', F_A), ('TB_F_B', F_B), ('TB_F_Y', F_Y)]:
            f.write(f"localparam integer {k} = {v};\n")

    json.dump({'jury_before': bad, 'jury_after': good, 'EPS_NUM': EPS_NUM,
               'sat_out': sat_out},
              open(os.path.join(a.out, 's4d_report.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
'''

_S4D_EXPORT_SRC = r'''
#!/usr/bin/env python3
"""
s4d_export.py -- engine shared between the notebooks: quantization, packing of
the .mem files, bit-accurate model, range calibration.

Why it exists. The first version of the training notebook had everything
inline. As soon as a second notebook was needed (the range repair) there would
have been two copies, and they would have diverged at the first change --
exactly the problem that the README of the repo describes for ln_gamma.mem,
written by two different generators with different contents. Here there is a
single copy, imported by both.

The formats are NOT redefined here: they come from gen_s4d_mem.py, which is the
verified mirror of the Verilog. This module uses them, it does not copy them.

Dependencies: numpy and gen_s4d_mem (mandatory), torch (only for the model and
the range calibration; the export works without it).
"""

import math
import os
import sys

import numpy as np

import gen_s4d_mem as G

# Version of the interface. It is needed because the modules live on Drive and
# the notebook loads them only if they are MISSING: an old copy of this file
# next to a new s4d_bn.py does not fail at import time, it fails twenty minutes
# later, in the middle of the rescaling cell, with a TypeError on a keyword argument.
#   1: original LayerNorm version
#   2: optional parameters for the norm (with_norm, norm, norms/norm_final),
#      needed by s4d_bn.py
API = 2

H, MODES, NUNITS, NCH, TH = G.H, G.MODES, G.NUNITS, G.NCH, G.TH
# NMAC / NPH: folding of the mixing. NMAC MACs in parallel, each responsible
# for NPH = TH/NMAC rows of W_out, served in NPH phases of H cycles.
NMAC, NPH = G.NMAC, G.NPH

# field of the per-channel exponent of b, packed with D
W_BSH = 5
BSH_MAX = G.W_A - 1          # datapath limit: s1<<bsh must fit in PW bits

# encoder
W_WENC, W_BENC, W_OUT_ENC, F_OUT_ENC = 18, 32, 16, 11

# decoder
W_CIN, W_CLS, W_ACC_D, K_CLS = 8, 8, 32, 10

LOG2_H = 7

# format ceilings, in real units
CEIL_ACT = 2.0 ** (G.W_ACT - 1 - G.F_ACT)     # residual and "a" branch: Q5.11 -> 16
CEIL_SSM = 2.0 ** (G.W_SSM - 1 - G.F_SSM)     # biquad bank output


# ============================================================================
# Formats
# ============================================================================
def set_f_ssm(new_f_ssm):
    """Changes F_SSM and recomputes the derived values inside gen_s4d_mem.

    F_SSM is a plain generic of s4d_layer (default 11) and the constants that
    depend on it are derived INSIDE the RTL:
        s4d_biquad_bank : SH_O = F_Y - F_OUT
        s4d_mixing_glu  : FACC = F_W + F_IN, RQ_G = FACC - F_SIGI,
                          RQ_A = FACC - F_A, SH_O = F_A + F_SIGO - F_OUT
    So changing it is NOT a change to the VHDL code: it is a generic to pass.
    But the generated .mem files depend on the value (bout is quantized at
    FACC bits, the GELU LUT is built on F_SSM), so if the RTL is elaborated
    with the default and the .mem files were generated with another value
    the result is silently wrong. See required_generics().
    """
    G.F_SSM = int(new_f_ssm)
    G.SH_SSM = G.F_Y - G.F_SSM
    G.FACC = G.F_W + G.F_SSM
    G.RQ_G = G.FACC - G.F_SIGI
    G.RQ_A = G.FACC - G.F_ACT
    globals()["CEIL_SSM"] = 2.0 ** (G.W_SSM - 1 - G.F_SSM)
    return G.F_SSM


def formats_table():
    return "\n".join([
        f"activations / residual Q{16-G.F_ACT}.{G.F_ACT}  W_ACT={G.W_ACT}   ceiling +-{CEIL_ACT:g}",
        f"bank output (yssm)     Q{16-G.F_SSM}.{G.F_SSM}  W_SSM={G.W_SSM}   ceiling +-{CEIL_SSM:g}",
        f"a1,a2                  W_A={G.W_A} F_A={G.F_A}     ceiling +-{2**(G.W_A-1-G.F_A)}",
        f"b0,b1                  W_B={G.W_B} F_B={G.F_B}+    PER-CHANNEL exponent",
        f"W_out                  W_W={G.W_W} F_W={G.F_W}     ceiling +-{2**(G.W_W-1-G.F_W)}",
        f"gamma                  W_PAR={G.W_PAR} F_GAMMA={G.F_GAMMA}",
        f"mixing accumulator     W_MACC={G.W_MACC} FACC={G.FACC}  RQ_A={G.RQ_A} RQ_G={G.RQ_G}",
    ])


# ============================================================================
# Quantization of a layer
# ============================================================================
def quantize_layer(d, verbose=True, f_d=None):
    """d: dict with log_A_real, A_imag, log_dt, C, D, wout, bout, gamma, beta.

    f_d: forces the fractional bits of D instead of calibrating them on this layer.
    It is needed because in the RTL `F_D` is a GLOBAL generic -- `SH_D = F_D + F_U -
    F_Y` is a constant of s4d_biquad_unit, a single one for the whole design --
    while `G.calib` chooses it per layer and does not give the same value.
    Exporting D with a per-layer f_d and then passing a single F_D to the RTL
    means scaling the D branch by a power of two in all the layers that do not
    have that value. Measured: logit SNR from 33 dB to 5-7 dB, and the predicted
    class changes. The caller must pass min(f_d) over all the layers: the
    minimum because an f_d higher than the calibrated one clips D, which is much
    worse than one bit less.

    gamma/beta are OPTIONAL: if they are missing (BatchNorm variant, where the
    norm is a per-channel affine quantized by s4d_bn.quantize_affine) the rest
    of the layer is quantized anyway. Everything downstream of the norm --
    biquad, D, mixing, GLU -- does not depend on which normalization is used,
    and that is exactly why this function is not duplicated in the BN module.
    """
    a1f, a2f, b0f, b1f, lbar = G.to_biquad(d["log_A_real"], d["A_imag"],
                                           d["log_dt"], d["C"])
    rho = float(np.abs(lbar).max())

    # --- Jury clamp ---------------------------------------------------------
    # a1 and a2 quantized independently go out of the stability triangle.
    # The critical case is mode n=0 (double real pole): 1+a1+a2 = (1-lbar)^2,
    # a second-order infinitesimal that rounding cancels, and the
    # biquad becomes an integrator.
    a1i, a2i = G.q(a1f, G.F_A, G.W_A), G.q(a2f, G.F_A, G.W_A)
    bad = int(((np.abs(a1i) >= (1 << G.F_A) + a2i) | (a2i >= (1 << G.F_A))).sum())
    a1i, a2i = G.jury_clamp(a1i, a2i, G.F_A)
    good = int(((np.abs(a1i) >= (1 << G.F_A) + a2i) | (a2i >= (1 << G.F_A))).sum())

    # --- per-channel exponent of b -----------------------------------------
    bmax = np.maximum(np.abs(b0f), np.abs(b1f)).max(axis=1)
    fb_ch = np.floor(np.log2(((1 << (G.W_B - 1)) - 1) / np.maximum(bmax, 1e-30))).astype(int)
    fb_ch = np.clip(fb_ch, G.F_B, G.F_B + 15)
    bsh = fb_ch + G.F_U - G.F_Y
    assert bsh.max() < (1 << W_BSH), f"bsh {bsh.max()} does not fit in {W_BSH} bits"
    assert bsh.max() <= BSH_MAX, f"bsh {bsh.max()} > {BSH_MAX}: s1<<bsh exceeds the datapath"
    assert bsh.min() >= G.SH_B0, f"bsh min {bsh.min()} < SH_B0 {G.SH_B0}"

    b0i = np.stack([G.q(b0f[h], fb_ch[h], G.W_B) for h in range(H)])
    b1i = np.stack([G.q(b1f[h], fb_ch[h], G.W_B) for h in range(H)])

    # --- remaining tensors --------------------------------------------------
    f_d_cal = G.calib(d["D"], G.W_D)
    f_d = f_d_cal if f_d is None else int(f_d)
    qz = dict(
        a1=a1i, a2=a2i, b0=b0i, b1=b1i, bsh=bsh, fb_ch=fb_ch,
        D=G.q(d["D"], f_d, G.W_D), f_d=f_d, f_d_cal=f_d_cal,
        sh_d=f_d + G.F_U - G.F_Y,
        wout=G.q(d["wout"], G.F_W, G.W_W),
        bout=G.q(d["bout"], G.FACC, G.W_MACC),
        jury_before=bad, jury_after=good, rho=rho,
    )
    if "gamma" in d:
        qz["gamma"] = G.q(d["gamma"], G.F_GAMMA, G.W_PAR)
        qz["beta"] = G.q(d["beta"], G.F_ACT, G.W_PAR)

    # --- saturation: loud, never masked -------------------------------------
    warn = []
    checks = [("D", d["D"], G.W_D, f_d),
              ("wout", d["wout"], G.W_W, G.F_W),
              ("bout", d["bout"], G.W_MACC, G.FACC)]
    if "gamma" in d:
        checks = [("gamma", d["gamma"], G.W_PAR, G.F_GAMMA),
                  ("beta", d["beta"], G.W_PAR, G.F_ACT)] + checks
    for nm, arr, w, f in checks:
        lim = (1 << (w - 1)) - 1
        over = np.abs(np.asarray(arr))[np.abs(np.round(np.asarray(arr) * 2.0 ** f)) >= lim]
        if over.size:
            warn.append(f"{nm}: {over.size}/{np.asarray(arr).size} weights clipped by the "
                        f"Q{w-f}.{f} format (ceiling {lim/2**f:.4f}, "
                        f"max|.|={np.abs(arr).max():.4f})")
    qz["warn"] = warn

    if verbose:
        print(f"  max |lambda_bar| {rho:.6f}   Jury {bad} -> {good} outside the triangle")
        print(f"  per-channel F_B [{fb_ch.min()}, {fb_ch.max()}], bsh [{bsh.min()}, {bsh.max()}]"
              f"   dynamic range of max|b| between channels {bmax.max()/bmax.min():.0f}x")
        note = "" if f_d == f_d_cal else f"  (calibrated {f_d_cal}, forced to the global value)"
        print(f"  F_D {f_d}  (max|D| {np.abs(d['D']).max():.4f}, "
              f"fills {np.abs(d['D']).max()*2**f_d:.0f}/{(1<<(G.W_D-1))-1}){note}")
        for w_ in warn:
            print("  WARNING  " + w_)
    return qz


# ============================================================================
# Writing the .mem files
# ============================================================================
def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return f"  {os.path.basename(str(path)):24s} {len(lines):>6d} rows"


def write_layer_mem(m, qz, outdir, verbose=True, with_norm=True):
    """with_norm=False skips ln_gamma/ln_beta: the BatchNorm variant writes
    l<m>_bn_w / l<m>_bn_b in their place (see s4d_bn.write_affine_mem), which are
    NOT the same thing and must not end up in the same folder."""
    pre, out = f"l{m}_", str(outdir)
    j = lambda n: os.path.join(out, pre + n)
    msgs = []
    if with_norm:
        msgs += [
            _write(j("ln_gamma.mem"), [G.hx(v, G.W_PAR) for v in qz["gamma"]]),
            _write(j("ln_beta.mem"), [G.hx(v, G.W_PAR) for v in qz["beta"]]),
        ]
    msgs += [
        # {bsh[W_BSH-1:0], D[W_D-1:0]}: D and its shift travel together
        _write(j("s4d_D.mem"),
               [G.hx((int(qz["bsh"][h]) << G.W_D) | (int(qz["D"][h]) & ((1 << G.W_D) - 1)),
                     G.W_D + W_BSH) for h in range(H)]),
    ]
    # coefficients: order (unit j, slot, mode n), the same with which
    # s4d_biquad_unit addresses its own memory (cnt = slot*MODES + n)
    lines = []
    for u in range(NUNITS):
        for slot in range(NCH):
            h = u * NCH + slot
            for n in range(MODES):
                lines.append(G.hx(
                    (int(qz["b1"][h, n]) & 0xFFFF) << 64 |
                    (int(qz["b0"][h, n]) & 0xFFFF) << 48 |
                    (int(qz["a2"][h, n]) & 0xFFFFFF) << 24 |
                    (int(qz["a1"][h, n]) & 0xFFFFFF), 80))
    msgs.append(_write(j("s4d_coef.mem"), lines))

    # W_out: NPH*H rows of NMAC*W_W bits, row (r*H + h) is phase r of
    # input channel h. See write_wout_mem().
    msgs.append(write_wout_mem(j("s4d_wout.mem"), qz["wout"]))
    msgs.append(_write(j("s4d_bout.mem"), [G.hx(v, G.W_MACC) for v in qz["bout"]]))
    if verbose:
        print("\n".join(msgs))


def wout_row_perm():
    """Permutation of the COLUMNS of W_out, today the identity.

    The ordering convention between export and RTL lived here, and it stays here
    now that there is nothing left to permute: it is the place where the
    convention is documented, not a leftover to delete.

    How it was. The spatial biquad bank (NUNITS=64 units of NCH=2 channels)
    emitted the channels in the order in which they were ready -- unit j served
    channels 2j and 2j+1, so first all the even ones 0,2,...,126 and then all
    the odd ones 1,3,...,127. The export wrote the rows of s4d_wout.mem in that
    order, so that the mixing could consume them in sequence without
    addressing.

    How it is now. The folded bank (NUNITS=8 units of NCH=16) reorders the
    results in a buffer and emits the channels in natural order 0..127. The
    reordering happens in HARDWARE: it is no longer needed here, and the
    permutation is the identity.

    If the buffer were removed from the RTL this function must go back to
    containing the permutation corresponding to the NUNITS of that time, i.e.
    the order in which the bank emits the channels:

        [h for slot in range(NCH) for h in range(slot, H, NCH)]

    which for NCH=2 is exactly even-then-odd. Caution: with the current format
    the permutation would have to be applied to the index h INSIDE the row
    (r*H + h), not to the order of the rows as the old version did.
    """
    return list(range(H))


def write_wout_mem(path, wout):
    """Writes s4d_wout.mem in the format of the folded mixing and reads it back.

    Format. NPH*H = 512 rows of NMAC*W_W = 768 bits. Row (r*H + h)
    contains the weights of the NMAC output channels served by phase r, all
    related to the same input channel h:

        row(r*H + h) [m*W_W +: W_W] = W_out[r*NMAC + m][h]

    with r = 0..NPH-1, h = 0..H-1 in natural order, m = 0..NMAC-1.

    Why not the H rows of TH*W_W = 3072 bits of before: 3072 bits do not fit
    in any BRAM port, and the synthesizer built the memory out of distributed
    registers -- tens of thousands of LUTs for data that fits in a few BRAMs.
    768 bits are NMAC parallel reads from a reasonable port, and it is exactly
    what the mixing consumes in one cycle: NMAC MACs, each responsible for
    NPH rows of W_out, in NPH phases of H cycles.

    The read-back is not a luxury. A wrong permutation here produces neither
    overflow nor instability: it produces a network that computes a different
    function with perfectly normal statistics, and it is not caught unless
    the logits are compared -- i.e. much later, if ever.
    """
    wout = np.asarray(wout, dtype=np.int64)
    if wout.shape != (TH, H):
        raise ValueError(f"wout has shape {wout.shape}, {(TH, H)} expected")
    perm = wout_row_perm()
    rows = []
    for r in range(NPH):
        for h in perm:
            word = 0
            for m in range(NMAC):
                word |= (int(wout[r * NMAC + m, h]) & ((1 << G.W_W) - 1)) << (m * G.W_W)
            rows.append(G.hx(word, NMAC * G.W_W))
    msg = _write(path, rows)
    check_wout_mem(path, wout)
    return msg


def read_wout_mem(path):
    """Rebuilds W_out (TH, H) from s4d_wout.mem with the bit indices of the RTL."""
    words = [int(t, 16) for t in open(str(path)).read().split()]
    if len(words) != NPH * H:
        raise ValueError(f"{path}: {len(words)} rows, {NPH*H} expected")
    mask, half = (1 << G.W_W) - 1, 1 << (G.W_W - 1)
    inv = np.empty(H, dtype=np.int64)          # position -> channel
    inv[:] = wout_row_perm()
    out = np.zeros((TH, H), dtype=np.int64)
    for idx, word in enumerate(words):
        if word >> (NMAC * G.W_W):
            raise ValueError(f"{path}: row {idx} exceeds {NMAC*G.W_W} bits")
        r, pos = divmod(idx, H)
        h = int(inv[pos])
        for m in range(NMAC):
            v = (word >> (m * G.W_W)) & mask
            out[r * NMAC + m, h] = v - (1 << G.W_W) if v >= half else v
    return out


def check_wout_mem(path, wout):
    """Reads s4d_wout.mem back and compares it with the starting quantized matrix."""
    got = read_wout_mem(path)
    want = np.asarray(wout, dtype=np.int64)
    if got.shape != want.shape or not np.array_equal(got, want):
        bad = np.argwhere(got != want)
        c, h = (int(bad[0][0]), int(bad[0][1])) if bad.size else (-1, -1)
        raise ValueError(
            f"{os.path.basename(str(path))}: the read-back does not match W_out. "
            f"{len(bad)}/{want.size} different weights, the first at (c={c}, h={h}): "
            f"written {got[c, h]} instead of {want[c, h]}. "
            f"Wrong packing or permutation: it is NOT a quantization error, "
            f"and downstream it is visible only in the logits.")
    return True


def quantize_encoder(enc_w, enc_b, mu, sigma, verbose=True):
    """Folds the normalization into the weights: the FPGA receives the raw pixel.

        u = (p/255 - mu)/sigma,  x = w u + b
        =>  x = (w/(255 sigma)) p + (b - w mu/sigma)

    The product w'*p is exact (p is an integer): the only rounding is the final one.
    """
    w_fold = np.asarray(enc_w) / (255.0 * sigma)
    b_fold = np.asarray(enc_w) * (-mu / sigma) + np.asarray(enc_b)
    f_enc = min(G.calib(w_fold, W_WENC), G.calib(b_fold, W_BENC))
    sh_out = f_enc - F_OUT_ENC
    if sh_out < 0:
        raise ValueError("negative SH_OUT: the weights do not fill the residual format")
    wq, bq = G.q(w_fold, f_enc, W_WENC), G.q(b_fold, f_enc, W_BENC)
    if verbose:
        print(f"  max|w'| {np.abs(w_fold).max():.3e}  max|b'| {np.abs(b_fold).max():.4f}")
        print(f"  F_WENC {f_enc}   SH_OUT {sh_out}")
    return dict(w=wq, b=bq, f_enc=f_enc, sh_out=sh_out)


def write_encoder_mem(enc, outdir, verbose=True):
    msgs = [_write(os.path.join(str(outdir), "enc_w.mem"), [G.hx(v, W_WENC) for v in enc["w"]]),
            _write(os.path.join(str(outdir), "enc_b.mem"), [G.hx(v, W_BENC) for v in enc["b"]])]
    if verbose:
        print("\n".join(msgs))


def write_decoder_mem(nf_gamma, nf_beta, cls_w, cls_b, f_cin, outdir,
                      nlayer=6, verbose=True, with_norm=True):
    """Final LN (index nlayer for mem_name) + INT8 classifier.

    with_norm=False writes only the classifier: in the BatchNorm variant the
    final norm is an affine and it is written by s4d_bn.write_decoder_affine_mem.
    In that case nf_gamma/nf_beta can be None.
    """
    f_w = int(math.floor(math.log2(((1 << (W_CLS - 1)) - 1) / float(np.abs(cls_w).max()))))
    rq_shift = G.F_ACT - f_cin
    if rq_shift < 0:
        raise ValueError("negative RQ_SHIFT: xbar does not fit in the INT8 format")
    gq = G.q(nf_gamma, G.F_GAMMA, G.W_PAR) if with_norm else None
    bq = G.q(nf_beta, G.F_ACT, G.W_PAR) if with_norm else None
    wq = G.q(cls_w, f_w, W_CLS)
    bcq = G.q(cls_b, f_w + f_cin, W_ACC_D)
    o = str(outdir)
    rows = []
    for h in range(H):
        word = 0
        for c in range(K_CLS):
            word |= (int(wq[c, h]) & 0xFF) << (c * W_CLS)
        rows.append(G.hx(word, K_CLS * W_CLS))
    msgs = []
    if with_norm:
        msgs += [_write(os.path.join(o, f"l{nlayer}_ln_gamma.mem"), [G.hx(v, G.W_PAR) for v in gq]),
                 _write(os.path.join(o, f"l{nlayer}_ln_beta.mem"), [G.hx(v, G.W_PAR) for v in bq])]
    msgs += [_write(os.path.join(o, "cls_w.mem"), rows),
             _write(os.path.join(o, "cls_b.mem"), [G.hx(v, W_ACC_D) for v in bcq])]
    if verbose:
        print("\n".join(msgs))
    return dict(gamma=gq, beta=bq, w=wq, b=bcq, f_w=f_w, f_cin=f_cin, rq_shift=rq_shift)


def write_luts(outdir, verbose=True):
    """isqrt, gelu, sigmoid: they do not depend on the weights, but the GELU depends on F_SSM."""
    isq = [min(65535, int(round(32768 / math.sqrt((max(i, 64) + .5) / 64)))) for i in range(256)]
    gb, gs = G.build_pwl(G.gelu, G.F_SSM)
    sb, ss = G.build_pwl(G.sigmoid, G.F_SIGO)
    o = str(outdir)
    pack = lambda b, s: [G.hx((int(y) & 0xFFFF) << 16 | (int(x) & 0xFFFF), 32) for x, y in zip(b, s)]
    msgs = [_write(os.path.join(o, "isqrt_lut.mem"), [G.hx(v, 16) for v in isq]),
            _write(os.path.join(o, "gelu_lut.mem"), pack(gb, gs)),
            _write(os.path.join(o, "sigmoid_lut.mem"), pack(sb, ss))]
    if verbose:
        print("\n".join(msgs))
    return dict(isqrt=np.asarray(isq, np.int64),
                gelu_b=np.asarray(gb, np.int64), gelu_s=np.asarray(gs, np.int64),
                sig_b=np.asarray(sb, np.int64), sig_s=np.asarray(ss, np.int64))


# ============================================================================
# Bit-accurate model, vectorized (mirror of s4d_layer)
# ============================================================================
def shr_r(x, s):
    """Arithmetic shift with rounding. s scalar or array, also negative."""
    s = np.asarray(s)
    if s.ndim == 0:
        s = int(s)
        return x << (-s) if s <= 0 else (x + (1 << (s - 1))) >> s
    sp = np.maximum(s, 1)
    pos = (x + (np.int64(1) << (sp - 1))) >> sp
    neg = x << np.maximum(-s, 0)
    return np.where(s > 0, pos, neg)


def satn(x, w):
    return np.clip(x, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def satw(v, s, w=None):
    return satn(shr_r(v, s), G.W_S if w is None else w)


def isqrt_n(v, lut):
    """Replica of s4d_isqrt. v > 0 and < 2^53, so frexp gives the exact MSB."""
    _, e = np.frexp(v.astype(np.float64))
    msb = e.astype(np.int64) - 1
    e2 = msb & ~np.int64(1)
    sh = np.where(e2 >= 6, np.right_shift(v, np.maximum(e2 - 6, 0)),
                  np.left_shift(v, np.maximum(6 - e2, 0)))
    return lut[sh & 0xFF], e2 // 2


def pwl_n(x, base, slope, f_in, xh=3, log2n=8, mode_hi=True, y_lo=0, y_hi=0, w=16):
    fracw = f_in + xh + 1 - log2n
    offs = 1 << (xh + f_in)
    xs = np.clip(x, -offs, offs - 1) + offs
    i, fr = xs >> fracw, xs & ((1 << fracw) - 1)
    y = satn(base[i] + shr_r(slope[i] * fr, fracw), w)
    y = np.where(x < -offs, y_lo, y)
    return np.where(x >= offs, satn(x, w) if mode_hi else y_hi, y)


def layernorm_n(x, gamma, beta, f_out, lut, eps_num=None):
    eps_num = G.EPS_NUM if eps_num is None else eps_num
    s = x.sum(1, keepdims=True)
    ssq = (x * x).sum(1, keepdims=True)
    vnum = (ssq << LOG2_H) - s * s
    inv, half_e = isqrt_n(vnum + eps_num, lut)
    sh = (15 + G.F_GAMMA - f_out) + half_e
    p = ((x << LOG2_H) - s) * inv * gamma[None, :]
    return satn(shr_r(p, sh) + beta[None, :], G.W_ACT)


class LayerFX:
    """Bit-accurate mirror of s4d_layer, with the batch as first dimension.

    It also counts WHERE it saturates: the residual and the "a" branch are the
    two sites where a trained model goes out of Q5.11, and without counting them
    separately it is not clear which correction is needed.
    """

    def __init__(self, qz, luts, B, norm=None):
        """norm: callable (B,H) int64 -> (B,H) int64 in F_U format that replaces
        the LayerNorm. It is needed by the BatchNorm variant, where the norm is a
        per-channel affine: all the rest of the datapath is identical and must
        not be duplicated."""
        self.q, self.lut = qz, luts
        self.norm = norm
        self.s1 = np.zeros((B, H, MODES), dtype=np.int64)
        self.s2 = np.zeros((B, H, MODES), dtype=np.int64)
        self.sat_x = self.sat_a = self.sat_y = 0
        self.acc_bound = 0

    def step(self, x):
        q, lut = self.q, self.lut
        u = (self.norm(x) if self.norm is not None
             else layernorm_n(x, q["gamma"], q["beta"], G.F_U, lut["isqrt"]))

        uh = u[:, :, None]
        shb = q["bsh"][None, :, None]
        y = satw(q["b0"][None] * uh + np.left_shift(self.s1, shb), shb)
        a1y = satw(q["a1"][None] * y, G.SH_A)
        a2y = satw(q["a2"][None] * y, G.SH_A)
        self.s1 = satw(satw(q["b1"][None] * uh, shb) - a1y + self.s2, 0)
        self.s2 = -a2y
        acc = satw(q["D"][None, :] * u, q["sh_d"]) + y.sum(2)
        raw = shr_r(acc, G.SH_SSM)
        self.sat_y += int((np.abs(raw) > (1 << (G.W_SSM - 1)) - 1).sum())
        yssm = satn(raw, G.W_SSM)

        g = pwl_n(yssm, lut["gelu_b"], lut["gelu_s"], G.F_SSM, mode_hi=True)

        z = g @ q["wout"].T + q["bout"][None, :]
        # if the bound on the partial sums stays below 2^31 no intermediate
        # saturation of the W_MACC-bit accumulator is possible
        self.acc_bound = max(self.acc_bound, int(
            (np.abs(g) @ np.abs(q["wout"]).T + np.abs(q["bout"])[None, :]).max()))
        z = satn(z, G.W_MACC)

        raw_a = shr_r(z[:, :H], G.RQ_A)
        self.sat_a += int((np.abs(raw_a) > (1 << 15) - 1).sum())
        a = satn(raw_a, 16)
        gg = satn(shr_r(z[:, H:], G.RQ_G), 16)
        sg = pwl_n(gg, lut["sig_b"], lut["sig_s"], G.F_SIGI,
                   mode_hi=False, y_lo=0, y_hi=1 << G.F_SIGO)
        out = satn(shr_r(a * sg, G.F_SIGO), G.W_ACT)
        xn = x + out
        self.sat_x += int((np.abs(xn) > (1 << (G.W_ACT - 1)) - 1).sum())
        return satn(xn, G.W_ACT)


def encoder_n(pix, enc):
    acc = enc["w"][None, :] * pix[:, None] + enc["b"][None, :]
    return satn(shr_r(acc, enc["sh_out"]), W_OUT_ENC)


def run_chain_fx(pixels, QZ, enc, dec_gamma, dec_beta, luts, log2_l=None,
                 norms=None, norm_final=None):
    """pixels: (B, L) integers 0..255. Returns (xbar, layers).

    norms / norm_final: lists of callables that replace the LayerNorm of the six
    layers and the one of the decoder. With the defaults (None) the behaviour is
    the previous one, bit for bit: the LayerNorm notebooks do not change.
    """
    B, L = pixels.shape
    log2_l = int(math.log2(L)) if log2_l is None else log2_l
    layers = [LayerFX(q, luts, B, norm=(norms[i] if norms is not None else None))
              for i, q in enumerate(QZ)]
    nf = (norm_final if norm_final is not None else
          (lambda v: layernorm_n(v, dec_gamma, dec_beta, G.F_ACT, luts["isqrt"])))
    pool = np.zeros((B, H), dtype=np.int64)
    for k in range(L):
        x = encoder_n(pixels[:, k], enc)
        for lay in layers:
            x = lay.step(x)
        pool += nf(x)
    return satn(shr_r(pool, log2_l), G.W_ACT), layers


def classify_fx(xbar, dec):
    xq = satn(shr_r(xbar, dec["rq_shift"]), W_CIN)
    return satn(xq @ dec["w"].T + dec["b"][None, :], W_ACC_D)


# ============================================================================
# Rescaling of the value branch
# ============================================================================
def rescale_value_branch(W, s):
    """Scales the residual by s WITHOUT changing the computed function.

    LayerNorm is scale invariant, so scaling x does not change its normalized
    input. The output of each layer must therefore be scaled too, and this is
    done by scaling ONLY the "a" branch of output_linear (rows 0:H) and leaving
    the gate one untouched: the sigmoid is not homogeneous, scaling the gate
    would change the function. With

        enc_w, enc_b        *= s
        wout[0:H], bout[0:H] *= s     for each layer

    one gets x_m -> s*x_m for every m, and since norm_f is scale invariant too
    the logits stay identical. The only difference is the eps=1e-5 of
    LayerNorm, which is not invariant: with residuals of amplitude >> 1e-2 the
    effect is below one thousandth (measured: 2.6e-3 on logits of amplitude 26 with s=1/2).

    s should be a power of two: this way the rescaling is a plain shift of the
    exponent and adds no rounding error to the weights.
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in W.items() if k != "layers"}
    out["enc_w"] = W["enc_w"] * s
    out["enc_b"] = W["enc_b"] * s
    out["layers"] = []
    for d in W["layers"]:
        e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        e["wout"] = d["wout"].copy()
        e["bout"] = d["bout"].copy()
        e["wout"][:H] *= s
        e["bout"][:H] *= s
        out["layers"].append(e)
    return out


def choose_scale(max_act, ceiling=None, fill=0.75):
    """Largest power of two that brings max_act below fill*ceiling.

    A power of two and not an arbitrary factor: this way the rescaling only
    moves the exponent and adds no error to the weights. The price is that it
    rounds down, so the final margin is between fill*ceiling/2 and fill*ceiling.
    """
    ceiling = CEIL_ACT if ceiling is None else ceiling
    if max_act <= 0:
        return 1.0
    return 2.0 ** min(0, int(math.floor(math.log2(fill * ceiling / max_act))))


def needed_f_ssm(max_yssm, fill=0.75):
    """Largest (thus most precise) F_SSM that keeps max_yssm below the ceiling."""
    lim = (1 << (G.W_SSM - 1)) - 1
    # never above 11: it is the RTL default, raising it would not help and
    # would make the .mem files incompatible with a default elaboration
    return min(11, int(math.floor(math.log2(fill * lim / max(max_yssm, 1e-12)))))


# ============================================================================
# PyTorch model (identical to CIFAR/Train_CIFAR_s4.py) and calibration
# ============================================================================
def torch_model(d_model=128, n_layers=6, d_state=64, dropout=0.2, n_classes=10):
    """Builds the classifier. It imports torch only when called."""
    import torch
    import torch.nn as nn

    class S4DKernel(nn.Module):
        def __init__(self, dm, N=64, dt_min=1e-3, dt_max=1e-1):
            super().__init__()
            Hh, n = dm, N // 2
            log_dt = torch.rand(Hh) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            C = torch.randn(Hh, n, dtype=torch.cfloat)
            self.C = nn.Parameter(torch.view_as_real(C))
            self.log_dt = nn.Parameter(log_dt)
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(Hh, n)))
            self.A_imag = nn.Parameter(math.pi * torch.arange(n).repeat(Hh, 1).float())

        def forward(self, L):
            dt = torch.exp(self.log_dt)
            C = torch.view_as_complex(self.C)
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag
            dtA = A * dt.unsqueeze(-1)
            K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
            Ce = C * (torch.exp(dtA) - 1.0) / A
            return 2 * torch.einsum('hn,hnl->hl', Ce, torch.exp(K)).real

    class S4D(nn.Module):
        def __init__(self, dm, d_state=64, dropout=0.0):
            super().__init__()
            self.h, self.n = dm, d_state
            self.D = nn.Parameter(torch.randn(self.h))
            self.kernel = S4DKernel(self.h, N=self.n)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2 * self.h, kernel_size=1), nn.GLU(dim=-2))

        def forward(self, u):
            L = u.size(-1)
            k = self.kernel(L)
            y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(k, n=2 * L),
                                n=2 * L)[..., :L]
            y = y + u * self.D.unsqueeze(-1)
            y = self.dropout(self.activation(y))
            return self.output_linear(y)

    class S4DsCIFARClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.layers = nn.ModuleList([S4D(d_model, d_state) for _ in range(n_layers)])
            self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.norm_f = nn.LayerNorm(d_model)
            self.head_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x):
            x = self.input_proj(x)
            for layer, norm, drop in zip(self.layers, self.norms, self.dropouts):
                res = x
                x = norm(x).transpose(1, 2)
                x = layer(x).transpose(1, 2)
                x = res + drop(x)
            return self.classifier(self.head_dropout(self.norm_f(x).mean(dim=1)))

    return S4DsCIFARClassifier()


def weights_to_numpy(model, n_layers=6):
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    W = dict(enc_w=sd["input_proj.weight"].reshape(-1), enc_b=sd["input_proj.bias"].reshape(-1),
             nf_gamma=sd["norm_f.weight"], nf_beta=sd["norm_f.bias"],
             cls_w=sd["classifier.weight"], cls_b=sd["classifier.bias"], layers=[])
    for m in range(n_layers):
        p = f"layers.{m}."
        c = sd[p + "kernel.C"]
        W["layers"].append(dict(
            log_A_real=sd[p + "kernel.log_A_real"], A_imag=sd[p + "kernel.A_imag"],
            log_dt=sd[p + "kernel.log_dt"], C=c[..., 0] + 1j * c[..., 1],
            D=sd[p + "D"], wout=sd[p + "output_linear.0.weight"][:, :, 0],
            bout=sd[p + "output_linear.0.bias"],
            gamma=sd[f"norms.{m}.weight"], beta=sd[f"norms.{m}.bias"]))
    return W


def numpy_to_torch_(model, W, with_norm=True):
    """Rewrites the weights of the torch model from the numpy dictionary.

    with_norm=False skips gamma/beta: in the BatchNorm variant the norms have
    four tensors instead of two and they are rewritten by s4d_bn.numpy_to_torch_.
    All the rest -- kernel, D, mixing, encoder, classifier -- is identical and
    goes through here.
    """
    import torch
    sd = model.state_dict()
    sd["input_proj.weight"].copy_(torch.tensor(W["enc_w"].reshape(-1, 1)))
    sd["input_proj.bias"].copy_(torch.tensor(W["enc_b"]))
    if with_norm:
        sd["norm_f.weight"].copy_(torch.tensor(W["nf_gamma"]))
        sd["norm_f.bias"].copy_(torch.tensor(W["nf_beta"]))
    sd["classifier.weight"].copy_(torch.tensor(W["cls_w"]))
    sd["classifier.bias"].copy_(torch.tensor(W["cls_b"]))
    for m, d in enumerate(W["layers"]):
        p = f"layers.{m}."
        sd[p + "kernel.log_A_real"].copy_(torch.tensor(d["log_A_real"]))
        sd[p + "kernel.A_imag"].copy_(torch.tensor(d["A_imag"]))
        sd[p + "kernel.log_dt"].copy_(torch.tensor(d["log_dt"]))
        sd[p + "kernel.C"].copy_(torch.tensor(np.stack([d["C"].real, d["C"].imag], -1)))
        sd[p + "D"].copy_(torch.tensor(d["D"]))
        sd[p + "output_linear.0.weight"].copy_(torch.tensor(d["wout"][:, :, None]))
        sd[p + "output_linear.0.bias"].copy_(torch.tensor(d["bout"]))
        if with_norm:
            sd[f"norms.{m}.weight"].copy_(torch.tensor(d["gamma"]))
            sd[f"norms.{m}.bias"].copy_(torch.tensor(d["beta"]))
    return model


def calibrate_ranges(model, loader, device, n_images=128, n_layers=6, quantile=None):
    """Measures the activation ranges on the REAL model, with hooks.

    Reimplementing the chain in numpy to measure the ranges would have meant
    measuring a reimplementation: the hooks observe the modules that will be
    exported, so there is no intermediate model to trust.

    Observed sites, which are exactly the three that can go out of the format:
      x     input of norms[m] and of norm_f -> the residual, Q5.11
      yssm  input of the GELU               -> output of the biquad bank, Q5.11
      a     first half of output_linear[0]  -> value branch before the GLU
    """
    import torch
    Hh = model.norm_f.normalized_shape[0]
    acc = dict(x=[0.0] * (n_layers + 1), yssm=[0.0] * n_layers, a=[0.0] * n_layers,
               over_x=[0] * (n_layers + 1), over_y=[0] * n_layers, over_a=[0] * n_layers,
               n_x=0, n_y=0, n_a=0)
    hs = []

    def pre_x(i):
        def f(_, inp):
            v = inp[0].detach()
            acc["x"][i] = max(acc["x"][i], v.abs().max().item())
            acc["over_x"][i] += int((v.abs() > CEIL_ACT).sum().item())
        return f

    def pre_y(i):
        def f(_, inp):
            v = inp[0].detach()
            acc["yssm"][i] = max(acc["yssm"][i], v.abs().max().item())
            acc["over_y"][i] += int((v.abs() > CEIL_SSM).sum().item())
        return f

    def post_a(i):
        def f(_, __, out):
            v = out.detach()[:, :Hh]
            acc["a"][i] = max(acc["a"][i], v.abs().max().item())
            acc["over_a"][i] += int((v.abs() > CEIL_ACT).sum().item())
        return f

    for m in range(n_layers):
        hs.append(model.norms[m].register_forward_pre_hook(pre_x(m)))
        hs.append(model.layers[m].activation.register_forward_pre_hook(pre_y(m)))
        hs.append(model.layers[m].output_linear[0].register_forward_hook(post_a(m)))
    hs.append(model.norm_f.register_forward_pre_hook(pre_x(n_layers)))

    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_images:
                    break
                x = x[: n_images - seen].to(device)
                model(x)
                b, L = x.shape[0], x.shape[1]
                acc["n_x"] += b * L * Hh
                acc["n_y"] += b * L * Hh
                acc["n_a"] += b * L * Hh
                seen += b
    finally:
        for h in hs:
            h.remove()
    acc["n_images"] = seen
    acc["max_x"] = max(acc["x"])
    acc["max_a"] = max(acc["a"])
    acc["max_yssm"] = max(acc["yssm"])
    return acc


def print_ranges(r, n_layers=6):
    print(f"ranges measured on {r['n_images']} images   "
          f"(residual/a ceiling +-{CEIL_ACT:g}, yssm ceiling +-{CEIL_SSM:g})")
    print(f"{'':9s}{'max|yssm|':>11s}{'over':>9s}{'max|a|':>10s}{'over':>9s}"
          f"{'max|x|':>10s}{'over':>9s}")
    for m in range(n_layers):
        print(f"layer {m:<3d}{r['yssm'][m]:11.2f}{100*r['over_y'][m]/max(r['n_y'],1):8.2f}%"
              f"{r['a'][m]:10.2f}{100*r['over_a'][m]/max(r['n_a'],1):8.2f}%"
              f"{r['x'][m+1]:10.2f}{100*r['over_x'][m+1]/max(r['n_x'],1):8.2f}%")


def required_generics(enc, dec, f_ssm=None, nlayer=6, mem_dir="mem_trained/"):
    """The generics that MUST be passed to s4d_top: the defaults are not enough.

    Each of these has a default that was right for the random weights of the
    testbench and is not right for trained weights. Passing a wrong one raises
    no flag: the datapath works in one format and the .mem files are in another.

    F_SSM deserves a note. Until recently s4d_top did NOT expose it: it wired
    `W_SSM => W_ACT, F_SSM => F_ACT` to s4d_layer, i.e. it took for granted
    that the output of the biquad bank was in the same format as the residual.
    That is true only by coincidence, and the coincidence breaks as soon as the
    trained weights push yssm out of range. Now it is a generic with default
    16/11, which reproduces the previous behaviour.
    """
    f_ssm = G.F_SSM if f_ssm is None else f_ssm
    rows = [
        ("MEM_DIR", f'"{mem_dir}"', None),
        ("PER_LAYER_MEM", "true", "the six layers have different weights"),
        ("WITH_LOADER", "true", "false in the final synthesis"),
        ("NUNITS", NUNITS, "folded bank: %d units (was 64)" % NUNITS),
        ("NCH", NCH, "%d channels per unit (was 2)" % NCH),
        ("NMAC", NMAC, "folded mixing: %d MACs, NPH = TH/NMAC = %d phases"
                       % (NMAC, NPH)),
        ("EPS_NUM", G.EPS_NUM, "default 16384"),
        ("F_WENC", enc["f_enc"], "default 23; SH_OUT = F_WENC-F_ACT = %d" % enc["sh_out"]),
        ("F_SSM", f_ssm, "default 11"),
        ("RQ_SHIFT", dec["rq_shift"], "default 7"),
    ]
    out = ["generic map ("]
    for k, v, c in rows:
        line = f"    {k:<14s}=> {v},"
        out.append(line if c is None else f"{line:<44s}-- {c}")
    out.append(")")
    out += ["", f"-- decoder: F_CIN {dec['f_cin']}, F_W {dec['f_w']}",
            "-- the defaults above are those of the random-weight testbench:",
            "-- with these .mem files they are all wrong, and nobody notices."]
    return "\n".join(out)
'''

_S4D_BN_SRC = r'''
#!/usr/bin/env python3
"""
s4d_bn.py -- BatchNorm variant of the S4D model: training, folding,
quantization and export of the .mem files.

Why a separate module exists
--------------------------------
Changing the normalization touches ONE point of the chain only. Biquad, D,
mixing, GLU, encoder, classifier, fixed-point formats: all identical. The
engine stays `s4d_export.py`; here there is only what the BatchNorm really
changes, plus the three or four non-obvious consequences.

What changes with respect to LayerNorm
--------------------------------
1. **At inference the BatchNorm is not a normalization, it is an affine.**
   With frozen statistics (`model.eval()`)

       u = gamma * (x - mu) / sqrt(var + eps) + beta
         = w * x + b,      w = gamma / sqrt(var + eps),  b = beta - w * mu

   i.e. one multiplication and one addition per channel, with constants known
   at export time. In hardware the whole `s4d_layernorm` disappears: no sum and
   sum-of-squares accumulators, no `s4d_isqrt`, no 1/sqrt LUT, no latency of
   two passes over the vector. One DSP and one adder per channel remain.
   `EPS_NUM` also disappears from the generics: there is no runtime variance
   to regularize any more.

   Be careful not to read the analogy too literally though: `w` takes the
   place of `gamma` and fits in 16 bits, but `b` is NOT `beta` and does not fit
   in Q5.11. See the comment on `W_BNB` below: it is the error that brought the
   affine of layer 0 down to 28 dB.

2. **The BatchNorm is NOT scale invariant.** The rescaling of the value
   branch -- the trick that makes the activations fit in Q5.11 -- was free with
   LayerNorm because `LN(s*x) = LN(x)`. Here it is not, but the compensation is
   just as free and moreover *exact*: if the residual becomes `s*x` it is
   enough to export `w/s` instead of `w`, and `b` stays as it is. With
   LayerNorm the before/after comparison had a residual of 2.6e-3 due to the
   eps, i.e. 1e-4 relative; here the only difference is the reordering of the
   float32 operations (`(x-mu)/sqrt(var+eps)*gamma+beta` against `w*x+b`,
   propagated through six FFTs on 2048 points): a few ulp, ~1e-7 relative.
   Exact in exact arithmetic, not bit-identical in float32 -- and the checks
   must be written in ulp, not relative, or they fail on the noise.

3. **The model is exact only in eval().** In training the BatchNorm uses the
   batch statistics; the weights we export are those of the running averages.
   Every float-versus-fixed comparison of this module assumes `model.eval()`.
   With batch 64 x 1024 steps there are 65536 samples per channel per batch,
   so the running averages are stable -- but the difference exists and must
   not be confused with a quantization error.

4. **The .mem files are different files.** `l<m>_bn_w.mem` / `l<m>_bn_b.mem`
   instead of `l<m>_ln_gamma.mem` / `l<m>_ln_beta.mem`. Different names on
   purpose: if we reused the old names, an RTL still with `s4d_layernorm`
   would read plausible numbers, would normalize at runtime anyway and would
   give a wrong result without raising any flag. They go to a separate folder
   (`mem_trained_bn/`) for the same reason.

Dependencies: numpy, gen_s4d_mem, s4d_export (mandatory); torch only for the
model and for the folding of the parameters.
"""

import math
import os

import numpy as np

import gen_s4d_mem as G
import s4d_export as E

# The check must be done HERE, at import time, not where it is needed. On Colab
# the modules live on Drive and the notebook loads them only if they are
# missing: updating s4d_bn.py without updating s4d_export.py is the normal case,
# not the pathological one. Without this check the incompatibility shows up in
# the middle of the rescaling cell -- i.e. after the training -- with a
# TypeError on a keyword.
API_MIN = 2
if getattr(E, "API", 1) < API_MIN:
    raise ImportError(
        f"s4d_export.py is at API version {getattr(E, 'API', 1)}, >= {API_MIN} is needed.\n"
        f"  file: {getattr(E, '__file__', '?')}\n"
        "The copy on Drive is old: upload it again together with s4d_bn.py.\n"
        "In the notebook REFRESH_MODULES = True in the first cell is enough, or:\n"
        "    from google.colab import files; import importlib\n"
        "    for n, d in files.upload().items(): (SCRIPTS / n).write_bytes(d)\n"
        "    importlib.reload(E); importlib.reload(B)"
    )

H = G.H

# ----------------------------------------------------------------------------
# Format of the affine
# ----------------------------------------------------------------------------
# w fits in 16 bits as gamma of the LayerNorm: same silicon, no extra bit.
W_BNW = G.W_PAR

# b does NOT. The first version kept it at 16 bits in the output format (Q5.11,
# ceiling +-16) and added it AFTER the shift, by analogy with beta of the
# LayerNorm. It is wrong, and asymmetrically wrong: b is not beta.
#
#     b = beta - w*mu = beta - gamma * mu/sigma
#
# and mu/sigma is not bounded by anything. The worst case is layer 0, whose
# input is the output of the encoder: a single input channel, so
# x_h = w_h*p + b_h has mean b_h and deviation |w_h|, and where |w_h| is small
# the ratio mu/sigma becomes huge. Measured on a trained model:
# max|b| = 36.6 against a ceiling of 16, with the affine of layer 0 down to 28 dB.
#
# The point is that u = w*x + b STAYS in range -- a large b is cancelled by w*mu
# inside w*x -- so the datapath is sound and only the constant does not fit.
# The correction is therefore one of format, not of model: b travels with
# f_bnw + F_ACT fractional bits in 32 bits and is added BEFORE the shift, in the
# same accumulator as the product. It is exactly what is already done with
# bout in the mixing (quantized at FACC and added before the requantization).
# It costs a 32-bit ROM instead of a 16-bit one -- 4 kbit per norm -- and in
# exchange the bias is even more precise than before, because it does not go
# through the rounding at the output.
W_BNB = 32

# width of the accumulator w*x + b before the shift. |w|,|x| < 2^15 give a
# product below 2^30; added to b one guard bit is needed.
W_BNACC = 34

# ceiling on the realignment shift. A channel with a tiny w (gamma ~ 0, or a
# huge variance) is a dead channel: bringing F_BNW to 30 would not revive it
# and would require a wider barrel shifter than needed.
SH_MAX = 24

BN_EPS = 1e-5


# ============================================================================
# PyTorch model
# ============================================================================
def batchnorm_seq(d_model):
    """BatchNorm1d on channel D of a (B, L, D) tensor.

    The statistics are over the batch AND the time: 64 x 1024 samples per channel.
    `normalized_shape` is not needed by torch, it is needed by us: it is the
    attribute with which `s4d_export.calibrate_ranges` finds H without knowing
    which norm it is looking at, and it allows reusing the calibration instead
    of copying it.
    """
    import torch.nn as nn

    class BatchNormSeq(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.bn = nn.BatchNorm1d(d, eps=BN_EPS)
            self.normalized_shape = (d,)

        def forward(self, x):
            return self.bn(x.transpose(1, 2)).transpose(1, 2)

    return BatchNormSeq(d_model)


def torch_model(d_model=128, n_layers=6, d_state=64, dropout=0.2, n_classes=10):
    """The same classifier as s4d_export.torch_model, with the norms replaced.

    It is not a copy of the model: it is *that* model with `norms` and `norm_f`
    reassigned. The S4D core -- kernel, ZOH, GLU, the order of the operations --
    stays literally the object that the quantization knows how to export. If I
    rewrote it here, the first change to one of the two would make it diverge
    from the other without anything noticing.
    """
    import torch.nn as nn

    m = E.torch_model(d_model, n_layers, d_state, dropout, n_classes)
    m.norms = nn.ModuleList([batchnorm_seq(d_model) for _ in range(n_layers)])
    m.norm_f = batchnorm_seq(d_model)
    return m


# ============================================================================
# Folding BatchNorm -> affine
# ============================================================================
def fold(gamma, beta, mean, var, eps=BN_EPS):
    """(gamma, beta, running_mean, running_var) -> (w, b) with u = w*x + b."""
    w = np.asarray(gamma, dtype=np.float64) / np.sqrt(np.asarray(var, dtype=np.float64) + eps)
    b = np.asarray(beta, dtype=np.float64) - w * np.asarray(mean, dtype=np.float64)
    return w, b


def weights_to_numpy(model, n_layers=6):
    """As s4d_export.weights_to_numpy, but the norms arrive already folded.

    The keys of the norms are `bn_w`/`bn_b` and `nf_w`/`nf_b`, not `gamma`/`beta`:
    a different name because it is a different thing, and because this way
    `E.quantize_layer` (which treats gamma/beta as optional) does not take them
    by mistake and does not quantize them in the LayerNorm format.
    """
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    nf_w, nf_b = fold(sd["norm_f.bn.weight"], sd["norm_f.bn.bias"],
                      sd["norm_f.bn.running_mean"], sd["norm_f.bn.running_var"])
    W = dict(enc_w=sd["input_proj.weight"].reshape(-1),
             enc_b=sd["input_proj.bias"].reshape(-1),
             nf_w=nf_w, nf_b=nf_b,
             cls_w=sd["classifier.weight"], cls_b=sd["classifier.bias"], layers=[])
    for m in range(n_layers):
        p = f"layers.{m}."
        c = sd[p + "kernel.C"]
        bw, bb = fold(sd[f"norms.{m}.bn.weight"], sd[f"norms.{m}.bn.bias"],
                      sd[f"norms.{m}.bn.running_mean"], sd[f"norms.{m}.bn.running_var"])
        W["layers"].append(dict(
            log_A_real=sd[p + "kernel.log_A_real"], A_imag=sd[p + "kernel.A_imag"],
            log_dt=sd[p + "kernel.log_dt"], C=c[..., 0] + 1j * c[..., 1],
            D=sd[p + "D"], wout=sd[p + "output_linear.0.weight"][:, :, 0],
            bout=sd[p + "output_linear.0.bias"],
            bn_w=bw, bn_b=bb))
    return W


def numpy_to_torch_(model, W):
    """Rewrites the torch weights from the numpy dictionary. Valid ONLY in eval().

    The folded affine has no unique inverse in (gamma, beta, mu, var): we choose
    the simplest one, mu=0 and var=1-eps, so sqrt(var+eps)=1 and the BatchNorm
    module in eval computes exactly w*x + b. In train() it would start using the
    batch statistics again and this identity would break: the rebuilt model is
    used to check the rescaling, not to retrain.
    """
    import torch
    with torch.no_grad():
        E.numpy_to_torch_(model, W, with_norm=False)   # everything that is not a norm
        sd = model.state_dict()
        sd["norm_f.bn.weight"].copy_(torch.tensor(W["nf_w"]))
        sd["norm_f.bn.bias"].copy_(torch.tensor(W["nf_b"]))
        sd["norm_f.bn.running_mean"].zero_()
        sd["norm_f.bn.running_var"].fill_(1.0 - BN_EPS)
        for m, d in enumerate(W["layers"]):
            sd[f"norms.{m}.bn.weight"].copy_(torch.tensor(d["bn_w"]))
            sd[f"norms.{m}.bn.bias"].copy_(torch.tensor(d["bn_b"]))
            sd[f"norms.{m}.bn.running_mean"].zero_()
            sd[f"norms.{m}.bn.running_var"].fill_(1.0 - BN_EPS)
    return model


# ============================================================================
# Rescaling of the value branch -- exact, here
# ============================================================================
def rescale_value_branch(W, s):
    """Scales the residual by s without changing the computed function.

    With LayerNorm it was enough to scale the encoder and the "a" branch of the
    mixing, because the norm absorbed the scale by itself. The BatchNorm does
    not: if its input becomes s*x, the output w*(s*x) + b is no longer the
    previous one. The compensation is one line -- export w/s -- and unlike the
    LayerNorm case it is exact: there is no eps that breaks the homogeneity. In
    float32 the reordering of the operations remains anyway, a few ulp on the
    logits: whoever checks must use a threshold in ulp, not relative, or fails
    on the noise.

    s should stay a power of two, so that w/s only moves the exponent and adds
    no rounding error to the weights.
    """
    out = E.rescale_value_branch(W, s)
    out["nf_w"] = np.asarray(W["nf_w"], dtype=np.float64) / s
    for e, d in zip(out["layers"], W["layers"]):
        e["bn_w"] = np.asarray(d["bn_w"], dtype=np.float64) / s
        e["bn_b"] = np.asarray(d["bn_b"], dtype=np.float64).copy()
    return out


# ============================================================================
# Quantization of the affine
# ============================================================================
def quantize_affine(w, b, f_out=None, name="bn", verbose=True):
    """w, b float (H,) -> integers + the realignment shift.

    The datapath is:

        u = sat( round( (w_q * x_q + b_q) / 2^SH ) ),   SH = F_BNW + F_ACT - F_OUT

    with `b` quantized at F_BNW + F_ACT fractional bits, i.e. in the format of
    the accumulator, and added BEFORE the shift. See the comment on W_BNB at the
    top of the file: adding it afterwards, at 16 bits, clips it at +-16 and b
    does not respect that ceiling -- not because the model goes out of the
    datapath, but because b is large and opposite to w*mu, and the two cancel
    only AFTER being added.

    F_BNW is chosen to fill the 16 bits of w, and lowered if b does not fit in
    the 32 bits of the accumulator. Lowering it costs precision on w, but it is
    the only one of the two that can be paid for: a clipped b is not a rounding,
    it is a channel that computes another function.
    """
    f_out = G.F_U if f_out is None else f_out
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_b = float(np.abs(b).max())

    f_bnw = G.calib(w, W_BNW)
    lim_b = (1 << (W_BNB - 1)) - 1
    while f_bnw > 0 and max_b * 2.0 ** (f_bnw + G.F_ACT) > lim_b:
        f_bnw -= 1
    sh = f_bnw + G.F_ACT - f_out
    if sh < 0:
        raise ValueError(f"{name}: negative SH ({sh}). The norm weights are "
                         f"too large for {W_BNW} bits: max|w| = {np.abs(w).max():.3f}")
    if sh > SH_MAX:
        f_bnw -= sh - SH_MAX
        sh = SH_MAX
    f_b = f_bnw + G.F_ACT
    wq = G.q(w, f_bnw, W_BNW)
    bq = G.q(b, f_b, W_BNB)

    warn = []
    if max_b * 2.0 ** f_b > lim_b:
        n = int((np.abs(b) * 2.0 ** f_b > lim_b).sum())
        warn.append(f"{name}: {n}/{b.size} biases clipped even at {W_BNB} bits "
                    f"(max|b| = {max_b:.4f}): widen W_BNB, do not clip")
    # bound on the accumulator: if it overflows, the RTL adder must be widened
    acc = int(np.abs(wq).max()) * ((1 << (G.W_ACT - 1)) - 1) + int(np.abs(bq).max())
    if acc >= (1 << (W_BNACC - 1)):
        warn.append(f"{name}: accumulator w*x+b up to 2^{math.log2(acc):.1f}, "
                    f"does not fit in {W_BNACC} bits")
    # relative error of the quantum on the weakest channel: it is the sign of an
    # F_BNW dominated by a single out-of-scale channel
    nz = np.abs(w) > 0
    rel = (0.5 / 2.0 ** f_bnw) / np.abs(w[nz]).min() if nz.any() else 0.0

    qa = dict(w=wq, b=bq, f_bnw=f_bnw, f_b=f_b, f_out=f_out, sh=sh, warn=warn,
              max_w=float(np.abs(w).max()), max_b=max_b,
              acc_bound=acc, rel_worst=float(rel))
    if verbose:
        print(f"  {name}: F_BNW {f_bnw}  F_B {f_b}  SH {sh}   max|w| {np.abs(w).max():.4f}"
              f"  max|b| {max_b:.4f}   quantum/|w|min {rel:.1%}"
              f"   acc 2^{math.log2(max(acc,1)):.1f}/{W_BNACC-1}")
        for x in warn:
            print("  WARNING  " + x)
    return qa


def write_affine_mem(m, qa, outdir, verbose=True):
    """l<m>_bn_w.mem (16 bit), l<m>_bn_b.mem (32 bit).

    The shift is not in the .mem files: it is a generic. The two ROMs have
    different widths, and it is not a detail to deduce from the file -- a reader
    that assumed 16 bits also for b would read two words per channel and get
    everything wrong without noticing.
    """
    o = str(outdir)
    msgs = [E._write(os.path.join(o, f"l{m}_bn_w.mem"), [G.hx(v, W_BNW) for v in qa["w"]]),
            E._write(os.path.join(o, f"l{m}_bn_b.mem"), [G.hx(v, W_BNB) for v in qa["b"]])]
    if verbose:
        print("\n".join(msgs))


# ============================================================================
# Bit-accurate mirror of the affine
# ============================================================================
class AffineFX:
    """u = sat( shr_round(w*x + b, SH) ). It counts the saturations.

    It is the function passed to `E.LayerFX(..., norm=...)` and to
    `E.run_chain_fx(..., norms=..., norm_final=...)`: the rest of the datapath
    stays the one verified against the Verilog, and it is not copied here.

    `sat` counts the samples in which u goes out of Q5.11. A non-zero count here
    means something different from the saturations at the other sites: it is
    not the format of b (which now has 32 bits), it is the output of the norm
    that really goes out of the datapath.
    """

    def __init__(self, qa, w_out=None):
        self.q = qa
        self.w_out = G.W_U if w_out is None else w_out
        self.sat = 0
        self.acc_bound = 0

    def __call__(self, x):
        q = self.q
        acc = q["w"][None, :] * x + q["b"][None, :]
        self.acc_bound = max(self.acc_bound, int(np.abs(acc).max()))
        v = E.shr_r(acc, q["sh"])
        lim = (1 << (self.w_out - 1)) - 1
        self.sat += int((np.abs(v) > lim).sum())
        return E.satn(v, self.w_out)


def affine_float(x, w, b):
    """Float reference of the affine, for the self-check."""
    return np.asarray(x, dtype=np.float64) * np.asarray(w) + np.asarray(b)


def collect_norm_inputs(model, loader, device, n_images=4, n_layers=6,
                        stride=8, max_rows=4096):
    """Samples the REAL inputs of each norm. Returns n_layers+1 arrays (n, H).

    It is needed because there is no correct synthetic stimulus for this block.
    A uniform or Gaussian x over the global range is physically impossible:
    w_h = gamma_h/sigma_h is large exactly where sigma_h is small, and in those
    channels the real activation lives in mu_h +- a few sigma_h, not over the
    whole range. Sweeping the whole range drives those channels off scale by
    construction, and the self-check ends up measuring the absurdity of the
    stimulus instead of the quality of the quantization -- with two-digit
    saturation percentages that the real model never produces.

    stride subsamples the time: the 1024 steps of an image are strongly
    correlated, one every 8 covers the same distribution with an eighth of the RAM.
    """
    import torch
    store = [[] for _ in range(n_layers + 1)]
    hs = []

    def pre(i):
        def f(_, inp):
            v = inp[0].detach()                      # (B, L, H)
            v = v[:, ::stride, :].reshape(-1, v.shape[-1])
            store[i].append(v.double().cpu().numpy())
        return f

    for m in range(n_layers):
        hs.append(model.norms[m].register_forward_pre_hook(pre(m)))
    hs.append(model.norm_f.register_forward_pre_hook(pre(n_layers)))

    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for x, _ in loader:
                if seen >= n_images:
                    break
                x = x[: n_images - seen].to(device)
                model(x)
                seen += x.shape[0]
    finally:
        for h in hs:
            h.remove()
    return [np.concatenate(s)[:max_rows] for s in store]


def diagnose_affine(xs, w, b, qa, top=6):
    """Why an affine loses SNR: the channel, and which of the two limits binds it.

    There are two error terms on u = w*x + b, and they have opposite remedies:

      A) resolution of the INPUT. x arrives already quantized in Q5.11, with
         quantum q = 2^-F_ACT. The error w*q/sqrt(12) must be compared with
         the useful amplitude |w|*sigma_h, and the ratio simplifies:

             SNR_max(h) ~ sigma_h / q * sqrt(12)

         i.e. it depends ONLY on how many Q5.11 quanta the deviation of that
         channel spans. A channel with sigma below the quantum is already dead
         in the residual: no format of the affine can recover it, and widening
         F_BNW does not help at all.

      B) quantum of W. The error dw*|x| ~ 2^-(F_BNW+1) * |mu_h| compared with
         |w|*sigma_h. This one does depend on F_BNW -- and it grows with
         |mu_h|/sigma_h, i.e. it is the cancellation between w*x and b that
         eats the digits.

    If A dominates the problem is the rescaling or the format of the residual;
    if B dominates it is F_BNW, and the remedy is a per-channel exponent as the
    bsh of the biquad. Confusing them wastes days.
    """
    x = np.asarray(xs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sig = x.std(axis=0)
    mu = np.abs(x.mean(axis=0))
    q_x = 2.0 ** -G.F_ACT
    dw = 2.0 ** -(qa["f_bnw"] + 1)

    amp = np.abs(w) * np.maximum(sig, 1e-30)          # useful amplitude of u
    err_a = np.abs(w) * q_x / math.sqrt(12)
    err_b = dw * mu
    snr_a = 20 * np.log10(amp / np.maximum(err_a, 1e-300))
    snr_b = 20 * np.log10(amp / np.maximum(err_b, 1e-300))
    snr = np.minimum(snr_a, snr_b)

    order = np.argsort(snr)[:top]
    print(f"  worst channels (residual quantum q = {q_x:.2e}, "
          f"quantum of w = {dw:.2e})")
    print(f"  {'h':>4s}{'sigma/q':>10s}{'|mu|/sigma':>12s}{'|w|':>10s}"
          f"{'SNR_in':>9s}{'SNR_w':>8s}{'limit':>9s}")
    for h in order:
        lim = "input" if snr_a[h] <= snr_b[h] else "F_BNW"
        print(f"  {h:4d}{sig[h]/q_x:10.2f}{mu[h]/max(sig[h],1e-30):12.1f}"
              f"{abs(w[h]):10.2f}{snr_a[h]:8.1f}dB{snr_b[h]:7.1f}dB{lim:>9s}")
    dead = int((sig < q_x).sum())
    print(f"  channels with sigma below the residual quantum: {dead}/{sig.size}"
          + ("  <- dead in the format, not in the affine" if dead else ""))
    return dict(snr_in=snr_a, snr_w=snr_b, sigma=sig, mu=mu, dead=dead,
                limited_by=("ingresso" if snr_a[order[0]] <= snr_b[order[0]] else "F_BNW"))


# ============================================================================
# Decoder
# ============================================================================
def write_decoder_affine_mem(nf_w, nf_b, outdir, nlayer=6, verbose=True):
    """Final norm as an affine, index nlayer to stay consistent with mem_name."""
    qa = quantize_affine(nf_w, nf_b, f_out=G.F_ACT, name=f"l{nlayer}_bn", verbose=verbose)
    write_affine_mem(nlayer, qa, outdir, verbose=verbose)
    return qa


# ============================================================================
# Reading the .mem files back -- the chain rebuilt from the bits, not from the floats
# ============================================================================
def _hex_rows(path):
    return [int(t, 16) for t in open(str(path)).read().split()]


def _signed(words, lo, w):
    m, half = (1 << w) - 1, 1 << (w - 1)
    v = np.array([(x >> lo) & m for x in words], dtype=np.int64)
    return np.where(v >= half, v - (1 << w), v)


# The permutation of the W_out columns lives in a single place, E.wout_row_perm
# (today the identity: the channel reordering moved to hardware). Here there
# was a second hand-written copy, even-then-odd: two definitions of the same
# convention are exactly how the read-back stops checking the export and
# starts confirming its error.
WOUT_PERM = E.wout_row_perm()


def read_mem_dir(mem_dir, report, n_layers=6):
    """Rebuilds (QZ, QA, enc, dec, luts) by reading the .mem files as the RTL reads them.

    Why from the .mem files and not from the .npz: requantizing the float
    weights checks the math but NOT the packing. If `s4d_coef.mem` put a2 where
    the VHDL reads a1, or if `s4d_wout.mem` associated row (r*H + h) with the
    wrong channel, a requantization would not notice. Here the fields are
    extracted with the same bit indices as the VHDL.

    `report` is quant_report.json: it is needed for the parameters that are NOT
    in the .mem files -- F_SSM, f_d per layer, the shifts. The fact that f_d is
    per layer while the RTL takes it as a global generic is an open problem, see
    check_global_f_d().
    """
    mem = str(mem_dir).rstrip("/") + "/"
    E.set_f_ssm(int(report["f_ssm"]))

    QZ = []
    for m in range(n_layers):
        coef = _hex_rows(mem + f"l{m}_s4d_coef.mem")
        assert len(coef) == H * MODES_, f"l{m}_s4d_coef.mem: {len(coef)} rows"
        r = lambda v: v.reshape(H, MODES_)
        a1 = r(_signed(coef, 0, G.W_A))
        a2 = r(_signed(coef, G.W_A, G.W_A))
        b0 = r(_signed(coef, 2 * G.W_A, G.W_B))
        b1 = r(_signed(coef, 2 * G.W_A + G.W_B, G.W_B))

        dw = _hex_rows(mem + f"l{m}_s4d_D.mem")
        D = _signed(dw, 0, G.W_D)
        bsh = np.array([(x >> G.W_D) & ((1 << E.W_BSH) - 1) for x in dw], dtype=np.int64)

        # s4d_wout.mem: NPH*H rows of NMAC*W_W bits, row (r*H + h) carries the
        # weights of the NMAC output channels of phase r for the input channel
        # h. The parsing is in E.read_wout_mem, which is also what the export
        # uses to read itself back: a single definition of the bits.
        wout = E.read_wout_mem(mem + f"l{m}_s4d_wout.mem")
        bout = _signed(_hex_rows(mem + f"l{m}_s4d_bout.mem"), 0, G.W_MACC)

        f_d = int(report["layers"][m]["f_d"])
        QZ.append(dict(a1=a1, a2=a2, b0=b0, b1=b1, bsh=bsh, D=D, f_d=f_d,
                       sh_d=f_d + G.F_U - G.F_Y, wout=wout, bout=bout))

    QA = []
    for m in range(n_layers + 1):
        a = report["affine"][m]
        QA.append(dict(w=_signed(_hex_rows(mem + f"l{m}_bn_w.mem"), 0, W_BNW),
                       b=_signed(_hex_rows(mem + f"l{m}_bn_b.mem"), 0, W_BNB),
                       f_bnw=int(a["F_BNW"]), f_b=int(a["F_B"]),
                       sh=int(a["SH"]),
                       f_out=G.F_ACT if m == n_layers else G.F_U))

    enc = dict(w=_signed(_hex_rows(mem + "enc_w.mem"), 0, E.W_WENC),
               b=_signed(_hex_rows(mem + "enc_b.mem"), 0, E.W_BENC),
               f_enc=int(report["encoder"]["F_WENC"]),
               sh_out=int(report["encoder"]["SH_OUT"]))

    crows = _hex_rows(mem + "cls_w.mem")
    cw = np.zeros((E.K_CLS, H), dtype=np.int64)
    for h, word in enumerate(crows):
        col = np.array([(word >> (c * E.W_CLS)) & 0xFF for c in range(E.K_CLS)],
                       dtype=np.int64)
        cw[:, h] = np.where(col >= 128, col - 256, col)
    dec = dict(w=cw, b=_signed(_hex_rows(mem + "cls_b.mem"), 0, E.W_ACC_D),
               f_w=int(report["decoder"]["F_W"]), f_cin=int(report["decoder"]["F_CIN"]),
               rq_shift=int(report["decoder"]["RQ_SHIFT"]))

    def _lut(name):
        w = _hex_rows(mem + name)
        base = np.array([x & 0xFFFF for x in w], dtype=np.int64)
        slope = np.array([(x >> 16) & 0xFFFF for x in w], dtype=np.int64)
        f = lambda v: np.where(v >= 1 << 15, v - (1 << 16), v)
        return f(base), f(slope)

    gb, gs = _lut("gelu_lut.mem")
    sb, ss = _lut("sigmoid_lut.mem")
    luts = dict(gelu_b=gb, gelu_s=gs, sig_b=sb, sig_s=ss,
                isqrt=np.zeros(256, dtype=np.int64))   # not needed without LayerNorm
    return QZ, QA, enc, dec, luts


MODES_ = G.MODES


def check_jury(QZ, verbose=True):
    """Poles inside the unit circle, checked ON THE BITS READ.

    The clamp has already been applied at export time, but checking it here
    costs nothing and is the only way to know that it made it into the .mem
    files. A single biquad outside and the layer becomes an integrator.
    """
    one = 1 << G.F_A
    tot = 0
    for m, q in enumerate(QZ):
        bad = int(((np.abs(q["a1"]) >= one + q["a2"]) | (q["a2"] >= one)).sum())
        rho = float(np.sqrt(np.maximum(q["a2"], 0) / one).max())
        tot += bad
        if verbose:
            print(f"  layer {m}: outside the triangle {bad}/{q['a1'].size}"
                  f"   max|lambda_bar| {rho:.6f}")
    if verbose:
        print("  OK: all the poles inside the unit circle" if tot == 0
              else f"  WARNING: {tot} unstable biquads")
    return tot


def check_global_f_d(QZ, verbose=True):
    """F_D in the RTL is a GLOBAL generic: here the six layers must agree.

    `SH_D = F_D + F_U - F_Y` is a constant of s4d_biquad_unit, a single one for
    the whole design. If the export calibrated F_D per layer -- and `G.calib`
    does, giving different values -- then whatever value is passed to the RTL is
    wrong on some layer, and the D branch of those layers comes out scaled by a
    power of two. No flag says so: the .mem files are in one format and the
    datapath in another.

    It is not theory. Measured on this model, with D exported at per-layer f_d
    [13,13,13,14,14,13]: passing F_D=13 the logit SNR drops from 33 dB to
    7.0, passing F_D=14 to 4.9, and the predicted class changes on real images.

    The remedy is in the export, not in the RTL: quantize all the D with the
    same f_d, and choose it equal to the MINIMUM of the calibrated ones. The
    minimum because an f_d higher than the calibrated one clips D -- with f_d=14
    on this model 7 channels would have been clipped -- and a clipped channel is
    a different model, while one bit of resolution less on two layers costs 52 dB
    with respect to the reference, i.e. twenty dB below the quantization noise:
    invisible.
    """
    fds = [q["f_d"] for q in QZ]
    ok = len(set(fds)) == 1
    if verbose:
        print(f"  F_D per layer: {fds}")
        print(f"  OK: a single value ({fds[0]}), the global generic is consistent" if ok else
              "  WARNING: different values. These .mem files can NOT be used with\n"
              "  any global F_D. Export again with f_d = min(f_d) on all the layers\n"
              f"  (here it would be {min(fds)}): it is a parameter of quantize_layer.")
    return ok


# ============================================================================
# Generics
# ============================================================================
def required_generics(enc, dec, qas, f_ssm=None, nlayer=6, mem_dir="mem_trained_bn/",
                      f_d=None):
    """The generics for s4d_top in the BatchNorm variant.

    Two differences with respect to the LayerNorm version, and both are gains:

    * `EPS_NUM` **is no longer needed**. It regularized the variance computed at
      runtime; with the folded BatchNorm there is no runtime variance.
      The eps is already inside w = gamma/sqrt(var+eps), at export time.
    * `BN_SH_<m>` appear, one per norm (six layers plus the decoder). They are
      scalars: the realignment shift of the accumulator w*x + b. A wrong value
      is NOT silent only if it is passed; if the default is left the result is
      scaled by a power of two, which is exactly the kind of error that goes
      unnoticed in a classifier -- the argmax almost never changes, the SNR
      collapses.

    `NUNITS`, `NCH` and `NMAC` also appear, which describe the folding of the
    datapath (8x16 bank instead of 64x2, mixing with 64 MACs instead of 256).
    They change no numeric format, but `NMAC` determines the shape of
    `s4d_wout.mem` -- NPH*H rows of NMAC*W_W bits -- so a value different from
    the one with which the .mem files were generated makes it read wrong rows.

    And a trap: `BN_W_B` is **32**, not 16. The two ROMs of the norm have
    different widths because b is not beta and does not fit in Q5.11 (see the
    comment on W_BNB at the top of the file). A reader that assumed 16 bits also
    for b would read two words per channel, and get everything wrong silently.
    """
    f_ssm = G.F_SSM if f_ssm is None else f_ssm
    rows = [
        ("MEM_DIR", f'"{mem_dir}"', None),
        ("NORM_KIND", '"affine"', "folded BatchNorm: no s4d_layernorm"),
        ("PER_LAYER_MEM", "true", "the six layers have different weights"),
        ("WITH_LOADER", "true", "false in the final synthesis"),
        ("NUNITS", G.NUNITS, "folded bank: %d units (was 64)" % G.NUNITS),
        ("NCH", G.NCH, "%d channels per unit (was 2)" % G.NCH),
        ("NMAC", G.NMAC, "folded mixing: %d MACs, NPH = TH/NMAC = %d phases"
                         % (G.NMAC, G.NPH)),
        ("F_WENC", enc["f_enc"], "default 23; SH_OUT = F_WENC-F_ACT = %d" % enc["sh_out"]),
        ("F_SSM", f_ssm, "default 11"),
        ("RQ_SHIFT", dec["rq_shift"], "default 7"),
    ]
    if f_d is not None:
        rows.append(("F_D", int(f_d),
                     "default 14: CLIPS D on half of the layers"))
    rows.append(("BN_W_W", W_BNW, "ROM of w"))
    rows.append(("BN_W_B", W_BNB, "ROM of b: 32 bits, NOT 16 -- added before the shift"))
    for m, qa in enumerate(qas):
        tag = "decoder" if m == nlayer else f"layer {m}"
        rows.append((f"BN_SH_{m}", qa["sh"], f"{tag}: F_BNW {qa['f_bnw']}, F_B {qa['f_b']}"))
    out = ["generic map ("]
    for k, v, c in rows:
        line = f"    {k:<14s}=> {v},"
        out.append(line if c is None else f"{line:<44s}-- {c}")
    out.append(")")
    out += ["",
            f"-- decoder: F_CIN {dec['f_cin']}, F_W {dec['f_w']}",
            "-- EPS_NUM is no longer needed: no variance computed at runtime.",
            "-- s4d_layernorm must be replaced by a per-channel affine (one DSP and",
            "-- one adder): it is the only RTL change that the BatchNorm requires."]
    return "\n".join(out)


def print_affine_table(qas, nlayer=6):
    print(f"{'norm':>9s}{'F_BNW':>7s}{'F_B':>6s}{'SH':>5s}{'max|w|':>10s}{'max|b|':>10s}"
          f"{'quantum/|w|min':>15s}{'acc':>8s}")
    for m, qa in enumerate(qas):
        nm = "decoder" if m == nlayer else f"layer {m}"
        print(f"{nm:>9s}{qa['f_bnw']:7d}{qa['f_b']:6d}{qa['sh']:5d}{qa['max_w']:10.4f}"
              f"{qa['max_b']:10.4f}{qa['rel_worst']:14.2%}"
              f"   2^{math.log2(max(qa['acc_bound'], 1)):.1f}")
    print(f"  (max|b| above 16 is normal and harmless: b is {W_BNB} bits wide and is added "
          f"before the shift)")
'''

G = _load_module("gen_s4d_mem", _GEN_S4D_MEM_SRC)
E = _load_module("s4d_export", _S4D_EXPORT_SRC)
B = _load_module("s4d_bn", _S4D_BN_SRC)

# ============================================================================
# training + quantization
# ============================================================================

# ############################################################################
# PART 2 -- it goes at the END of the file, it replaces everything AFTER the lines
#
#     G = _load_module("gen_s4d_mem", _GEN_S4D_MEM_SRC)
#     E = _load_module("s4d_export", _S4D_EXPORT_SRC)
#     B = _load_module("s4d_bn", _S4D_BN_SRC)
#
# The three embedded modules (_GEN_S4D_MEM_SRC, _S4D_EXPORT_SRC, _S4D_BN_SRC) and
# the three lines above must NOT be touched: they contain the fixed-point
# formats, the Jury clamp and the packing of the .mem files, i.e. the contract
# with the RTL.
# ############################################################################

# ============================================================================
# training + quantization
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
print("device:", DEVICE)

N_LAYERS, SEQ_LEN = 6, 1024
model = B.torch_model().to(DEVICE)

MU, SIGMA = 0.4734, 0.2516
transform = T.Compose([T.Grayscale(1), T.ToTensor(), T.Normalize((MU,), (SIGMA,))])
train_full = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.CIFAR10(str(DATA_ROOT), train=False, download=True, transform=transform)

n_tr = int(0.9 * len(train_full))
train_set, val_set = random_split(train_full, [n_tr, len(train_full) - n_tr],
                                  generator=torch.Generator().manual_seed(SEED))
if TRAIN_SUBSET:
    train_set = Subset(train_set, range(min(TRAIN_SUBSET, len(train_set))))
    val_set   = Subset(val_set,   range(min(TRAIN_SUBSET // 8, len(val_set))))

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return imgs.view(imgs.size(0), -1, 1), y

mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH, shuffle=sh, drop_last=sh,
                               num_workers=2, collate_fn=collate)
train_loader, val_loader, test_loader = mk(train_set, True), mk(val_set, False), mk(test_set, False)

# Range calibration on the VALIDATION set, not on the test set.
# The ranges measured here decide the scale s and F_SSM: tuning them on the test
# set means choosing the quantization by looking at the same images on which the
# fixed-point accuracy is then measured.
cal_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate)
print("train", len(train_set), "val", len(val_set), "test", len(test_set))


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def evaluate(loader, limit=None):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if limit and i * BATCH >= limit:
                break
            pred = model(x.to(DEVICE)).argmax(-1).cpu()
            ok += (pred == y).sum().item(); tot += y.numel()
    return ok / max(tot, 1)


# ---------------------------------------------------------------- optimizer
# The S4D kernel must not be decayed.
#
#   A_imag      starts from pi*n, so up to ~97. A weight_decay of 0.01 applied
#               ~700 times per epoch squeezes it towards zero: the high-frequency
#               modes collapse and the 32 modes per channel serve less than what
#               they were put there for.
#   log_A_real  starts from log(0.5); the decay brings it towards 0, i.e. A_real -> 1,
#               i.e. poles pushed towards the unit circle. It is a term that
#               acts directly on rho = |lambda_bar|, hence on how many biquads
#               the Jury clamp must correct at export time: the position of the
#               poles is being regularized without meaning to.
#   log_dt      same story on the discretization steps.
#
# In the reference S4D repo these three have weight_decay 0 and a fixed
# learning rate, excluded from the scheduler.
#
# Also out of the decay:
#   D           per-channel gain, not a matrix. Decaying it lowers max|D| and
#               shifts the calibration of f_d without any justification.
#   BatchNorm   gamma becomes w = gamma/sqrt(var+eps) at export time; shrinking it
#               worsens the quantum/|w|min ratio of print_affine_table.
#   bias        standard practice.
#
# For the ablation: SSM_KEYS = () reproduces the old behaviour (in that
# case the first assert must be removed too).
SSM_KEYS = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
LR_SSM = 1e-3

ssm, no_decay, decay = [], [], []
for n_, p_ in model.named_parameters():
    if not p_.requires_grad:
        continue
    if any(k in n_ for k in SSM_KEYS):
        ssm.append(p_)
    elif n_.endswith(".bias") or ".bn." in n_ or n_.endswith(".D"):
        no_decay.append(p_)
    else:
        decay.append(p_)

# A filter that finds nothing is silent: without these asserts, a rename of the
# submodules restarts a training identical to the previous one.
assert len(ssm) == 3 * N_LAYERS, f"SSM filter: {len(ssm)} parameters, {3*N_LAYERS} expected"
assert no_decay, "empty no_decay filter: have the BatchNormSeq names changed?"
print(f"parameters: {len(decay)} with WD, {len(no_decay)} without, {len(ssm)} SSM")

opt = torch.optim.AdamW([
    {"params": decay,    "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
    {"params": ssm,      "weight_decay": 0.0, "lr": LR_SSM},
], lr=1e-3)
SSM_GROUP = 2

sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
crit = nn.CrossEntropyLoss()
best, patience, ep0 = 0.0, 0, 0

for ep in range(ep0, MAX_EPOCHS):
    model.train(); tot = 0.0; t0 = time.time()
    for x, y in train_loader:
        loss = crit(model(x.to(DEVICE)), y.to(DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    acc = evaluate(val_loader)

    # ReduceLROnPlateau scales ALL the groups: the SSM group must be brought back
    # to its fixed lr right after, otherwise the poles stop moving much earlier
    # than the rest of the network.
    sched.step(acc)
    opt.param_groups[SSM_GROUP]["lr"] = LR_SSM

    dt = time.time() - t0
    avg_loss = tot / max(len(train_loader), 1)
    line = dict(epoch=ep + 1, loss=avg_loss, val_acc=acc, sec=round(dt, 1),
                lr=opt.param_groups[0]["lr"], lr_ssm=opt.param_groups[SSM_GROUP]["lr"])
    print(f"epoch {ep+1}/{MAX_EPOCHS}  loss {avg_loss:.4f}  val_acc {acc:.4f}  ({dt:.0f}s)")
    if acc > best:
        best, patience = acc, 0
        save_atomic(model.state_dict(), BEST_CKPT)
    else:
        patience += 1
    save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(),
                     sched=sched.state_dict(), epoch=ep, best=best,
                     patience=patience), LAST_CKPT)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(line) + "\n")
    if patience >= PATIENCE:
        print("early stopping"); break

model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE, weights_only=True))
model.eval()
ACC_FP32 = evaluate(test_loader)
print(f"FP32 accuracy = {ACC_FP32:.4f}")

# --- quantization ---

W = B.weights_to_numpy(model)
rng_before = E.calibrate_ranges(model, cal_loader, DEVICE, n_images=N_CAL)
E.print_ranges(rng_before)

peak_act = max(rng_before["max_x"], rng_before["max_a"])
s = E.choose_scale(peak_act, fill=FILL)
f_ssm = E.needed_f_ssm(rng_before["max_yssm"], fill=FILL)
E.set_f_ssm(f_ssm)
print(f"s = {s}   F_SSM = {f_ssm}")

W_fix = B.rescale_value_branch(W, s)

model_fix = B.torch_model().to(DEVICE)
B.numpy_to_torch_(model_fix, W_fix)
model_fix = model_fix.to(DEVICE).eval()

xb, _ = next(iter(cal_loader))
with torch.no_grad():
    l0 = model(xb.to(DEVICE)).double().cpu().numpy()
    l1 = model_fix(xb.to(DEVICE)).double().cpu().numpy()
d = np.abs(l1 - l0).max()
ulp = float(np.spacing(np.float32(max(np.abs(l0).max(), 1.0))))
assert d < 256 * ulp, "the rescaling changed the function"

rng_after = E.calibrate_ranges(model_fix, cal_loader, DEVICE, n_images=N_CAL)

np.savez(CKPT_DIR / "weights_bn.npz",
         **{k: v for k, v in W_fix.items() if k != "layers"},
         **{f"l{m}_{k}": v for m, dd in enumerate(W_fix["layers"]) for k, v in dd.items()})

luts = E.write_luts(MEM_OUT, verbose=False)
os.remove(MEM_OUT / "isqrt_lut.mem")

f_d_cal = [G.calib(dd["D"], G.W_D) for dd in W_fix["layers"]]
F_D = min(f_d_cal)
print(f"F_D = {F_D}")

QZ, QA = [], []
for m, dd in enumerate(W_fix["layers"]):
    q = E.quantize_layer(dd, f_d=F_D)
    E.write_layer_mem(m, q, MEM_OUT, with_norm=False)
    qa = B.quantize_affine(dd["bn_w"], dd["bn_b"], name=f"l{m}_bn")
    B.write_affine_mem(m, qa, MEM_OUT)
    QZ.append(q); QA.append(qa)

enc = E.quantize_encoder(W_fix["enc_w"], W_fix["enc_b"], MU, SIGMA)
E.write_encoder_mem(enc, MEM_OUT)

nf_qa = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], MEM_OUT, nlayer=N_LAYERS)

B.print_affine_table(QA + [nf_qa], nlayer=N_LAYERS)

# --- fixed-point check on the test set ---

def raw_pixels(ds, idx):
    out = []
    for i in idx:
        x, y = ds[i]
        pix = np.clip(np.round((x.numpy().reshape(-1) * SIGMA + MU) * 255.0), 0, 255)
        out.append((pix.astype(np.int64), y))
    return np.stack([o[0] for o in out]), np.array([o[1] for o in out])


idx = list(range(min(N_EVAL_FX, len(test_set))))
PIX, LBL = raw_pixels(test_set, idx)

norms_fx = [B.AffineFX(qa) for qa in QA]
nf_fx = B.AffineFX(nf_qa, w_out=G.W_ACT)

# The bit-accurate chain runs in blocks: LayerFX allocates s1 and s2 of shape
# (B, H, MODES) int64 for each of the six layers, ~6 MB per image, and with
# N_EVAL_FX = 512 in one go the RAM is not enough. The sequences are
# independent of each other, so splitting the batch does not change a bit.
#
# norms_fx / nf_fx are shared between the blocks and their counters accumulate
# by themselves; the LayerFX instead are recreated by run_chain_fx at each call,
# so sat_x / sat_a / sat_y must be summed by hand. Losing them is not a detail:
# they are the only proof that the trained model stays inside Q5.11.
sat_layers = [dict(sat_x=0, sat_a=0, sat_y=0, acc_bound=0) for _ in range(N_LAYERS)]
_parts = []
for _i in range(0, PIX.shape[0], CHUNK_FX):
    _xb, _lay = E.run_chain_fx(PIX[_i:_i + CHUNK_FX], QZ, enc, None, None, luts,
                               norms=norms_fx, norm_final=nf_fx)
    _parts.append(_xb)
    for _a, _l in zip(sat_layers, _lay):
        _a["sat_x"] += _l.sat_x
        _a["sat_a"] += _l.sat_a
        _a["sat_y"] += _l.sat_y
        _a["acc_bound"] = max(_a["acc_bound"], _l.acc_bound)
    print(f"  fixed {min(_i + CHUNK_FX, PIX.shape[0])}/{PIX.shape[0]}", end="\r")
print()
xbar = np.concatenate(_parts)

xbar_max = float(np.abs(xbar).max()) / 2.0 ** G.F_ACT
f_cin = int(math.floor(math.log2(127 / max(xbar_max, 1e-12))))
print(f"F_CIN = {f_cin}, RQ_SHIFT = {G.F_ACT - f_cin}")

dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin,
                          MEM_OUT, nlayer=N_LAYERS, with_norm=False)

logits_fx = E.classify_fx(xbar, dec)

# the float reference in blocks too: 512 images of 1024 steps in one go
# do not fit in GPU memory
_lf = []
with torch.no_grad():
    for _i in range(0, PIX.shape[0], CHUNK_FX):
        xin = torch.tensor((PIX[_i:_i + CHUNK_FX] / 255.0 - MU) / SIGMA,
                           dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        _lf.append(model(xin).double().cpu().numpy())
logits_f = np.concatenate(_lf)

scaled = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
err = scaled - logits_f
snr = 20 * math.log10(np.sqrt((logits_f ** 2).mean()) / max(np.sqrt((err ** 2).mean()), 1e-12))
acc_fx = float((logits_fx.argmax(1) == LBL).mean())
acc_f = float((logits_f.argmax(1) == LBL).mean())
agree = float((logits_fx.argmax(1) == logits_f.argmax(1)).mean())

# Confidence interval: it is needed not to read as a degradation of the
# quantization what is sampling noise. With 32 images it was
# +-9 percentage points, i.e. the number said nothing.
se95 = 1.96 * math.sqrt(max(acc_fx * (1 - acc_fx), 1e-12) / len(idx))
print(f"logit SNR {snr:.1f} dB   float acc {acc_f:.4f}   "
      f"fixed acc {acc_fx:.4f} +-{se95:.4f} (95%, n={len(idx)})   agreement {agree:.4f}")

tot_sat = (sum(a["sat_x"] + a["sat_a"] + a["sat_y"] for a in sat_layers)
           + sum(n.sat for n in norms_fx) + nf_fx.sat)
print(f"total saturations: {tot_sat}")

# --- generics, testbench stimuli, bundle ---

QA_ALL = QA + [nf_qa]
gen_str = B.required_generics(enc, dec, QA_ALL, f_ssm=f_ssm, nlayer=N_LAYERS,
                          mem_dir=str(MEM_OUT) + "/", f_d=F_D)
(MEM_OUT / "top_generics.txt").write_text(gen_str + "\n")

NTB = min(2, len(idx))
E._write(str(MEM_OUT / "top_stim.mem"), [G.hx(int(v), 8) for v in PIX[:NTB].reshape(-1)])
E._write(str(MEM_OUT / "top_exp.mem"),
         [G.hx(int(v), E.W_ACC_D) for v in logits_fx[:NTB].reshape(-1)])

with open(MEM_OUT / "top_config.vh", "w") as f:
    for k, v in [("TB_L", PIX.shape[1]), ("TB_LOG2_L", int(math.log2(PIX.shape[1]))),
                 ("TB_H", E.H), ("TB_K", E.K_CLS), ("TB_NLAYER", N_LAYERS),
                 ("TB_IMGS", NTB), ("TB_NORM_AFFINE", 1),
                 ("TB_F_SSM", G.F_SSM), ("TB_F_OUT", G.F_ACT),
                 ("TB_RQ_SHIFT", dec["rq_shift"]), ("TB_F_D", F_D),
                 ("ENC_F_WENC", enc["f_enc"]), ("ENC_SH_OUT", enc["sh_out"]),
                 ("CLS_F_CIN", dec["f_cin"]), ("CLS_F_W", dec["f_w"])]:
        f.write(f"localparam integer {k} = {v};\n")
    f.write(f"localparam integer BN_W_W = {B.W_BNW};\n")
    f.write(f"localparam integer BN_W_B = {B.W_BNB};\n")
    f.write(f"localparam integer BN_W_ACC = {B.W_BNACC};\n")
    for m, qa in enumerate(QA_ALL):
        sh_val = qa["sh"]
        f.write(f"localparam integer BN_SH_{m} = {sh_val};\n")
    for m, qa in enumerate(QA_ALL):
        fbnw_val = qa["f_bnw"]
        f.write(f"localparam integer BN_F_W_{m} = {fbnw_val};\n")
    for m, qa in enumerate(QA_ALL):
        fb_val = qa["f_b"]
        f.write(f"localparam integer BN_F_B_{m} = {fb_val};\n")

report = dict(
    norm="batchnorm", seed=SEED, acc_fp32_test=ACC_FP32,
    scale=s, f_ssm=f_ssm, f_d=F_D, f_d_calibrati=f_d_cal,
    fill=FILL, n_cal=N_CAL, cal_split="val", n_eval_fx=len(idx),
    optimizer=dict(wd=0.01, lr=1e-3, lr_ssm=LR_SSM, ssm_keys=list(SSM_KEYS),
                   n_decay=len(decay), n_no_decay=len(no_decay), n_ssm=len(ssm)),
    peaks_before=dict(x=rng_before["max_x"], a=rng_before["max_a"], yssm=rng_before["max_yssm"]),
    peaks_after=dict(x=rng_after["max_x"], a=rng_after["max_a"], yssm=rng_after["max_yssm"]),
    ceilings=dict(act=E.CEIL_ACT, ssm=E.CEIL_SSM),
    acc_fp32_subset=acc_f, acc_fixed_subset=acc_fx, acc_fixed_ci95=se95,
    agreement=agree, logit_snr_db=round(snr, 2),
    encoder=dict(F_WENC=enc["f_enc"], SH_OUT=enc["sh_out"]),
    decoder=dict(F_CIN=dec["f_cin"], F_W=dec["f_w"], RQ_SHIFT=dec["rq_shift"]),
    affine=[dict(idx=m, F_BNW=q["f_bnw"], F_B=q["f_b"], SH=q["sh"],
                 W_W=B.W_BNW, W_B=B.W_BNB, max_w=q["max_w"], max_b=q["max_b"],
                 rel_worst=q["rel_worst"], acc_bound_log2=round(math.log2(max(q["acc_bound"], 1)), 1),
                 sat=(norms_fx[m].sat if m < N_LAYERS else nf_fx.sat),
                 warn=q["warn"]) for m, q in enumerate(QA_ALL)],
    layers=[dict(jury_before=q["jury_before"], jury_after=q["jury_after"], rho=q["rho"],
                 f_d=q["f_d"], bsh_min=int(q["bsh"].min()), bsh_max=int(q["bsh"].max()),
                 sat_x=sat_layers[m]["sat_x"], sat_a=sat_layers[m]["sat_a"],
                 sat_y=sat_layers[m]["sat_y"],
                 acc_bound_log2=round(math.log2(max(sat_layers[m]["acc_bound"], 1)), 1),
                 warn=q["warn"]) for m, q in enumerate(QZ)],
)
(MEM_OUT / "quant_report.json").write_text(json.dumps(report, indent=2))

stamp = time.strftime("%Y%m%d_%H%M")
bundle = WORK / "bundle" / f"s4d_weights_bn_{stamp}"
bundle.mkdir(parents=True, exist_ok=True)
for f in sorted(MEM_OUT.glob("*")):
    shutil.copy2(f, bundle / f.name)
for f in [BEST_CKPT, CKPT_DIR / "weights_bn.npz", RUN_LOG]:
    if Path(f).is_file():
        shutil.copy2(f, bundle / Path(f).name)
readme_txt = (
    f"S4D weights quantized for the accelerator -- BatchNorm variant\n"
    f"generated {time.strftime('%Y-%m-%d %H:%M')}   seed {SEED}\n"
    f"FP32 accuracy {ACC_FP32:.4f} | fixed on {len(idx)} img {acc_fx:.4f} +-{se95:.4f} (95%)\n"
    f"logit SNR {snr:.1f} dB   total saturations {tot_sat}\n"
    f"rescaling of the value branch: s = {s} (compensated on w of the BN)\n"
    f"ranges calibrated on validation ({N_CAL} img)\n"
    f"weight decay excluded from: {list(SSM_KEYS)}, D, bias, BatchNorm\n\n"
    f"The norm is a per-channel AFFINE (l<m>_bn_w/l<m>_bn_b), not a LayerNorm:\n"
    f"these .mem files are not compatible with s4d_layernorm.\n\n{gen_str}\n"
)
(bundle / "README.txt").write_text(readme_txt)
zp = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
print("archive:", zp, f"({os.path.getsize(zp)/1e3:.0f} kB)")
