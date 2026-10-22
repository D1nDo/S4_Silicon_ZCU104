#!/usr/bin/env python3
"""
s4d_golden.py -- bit-exact fixed-point model of the S4D accelerator on ZCU104.

The model reads ONLY the .mem files that are loaded into the FPGA (the same
files referenced by the block designs in rebuild_S4D_ZCU104.tcl) plus the
per-dataset generics set in the block designs, and reproduces the integer
logits produced by the board, bit for bit.

    encoder (s4d_encoder_axis)  ->  6 x S4D layer (s4d_core_axis)
        ->  final affine norm + mean pool + INT8 classifier (s4d_decoder_axis)

Files read from mem_<DS>/ (DS = ECG, RadioML, SP, sCIFAR):
    w_stream_nmac256_<DS>.mem   per-layer weight stream (ROM of s4d_weightrom_axis),
                                6 layers x 14977 beats of 64 bit
    enc_w_<DS>.mem, enc_b_<DS>.mem          encoder weights / biases
    gelu_lut_<DS>.mem, sigmoid_lut_<DS>.mem PWL lookup tables
    l6_bn_w_<DS>.mem, l6_bn_b_<DS>.mem      final affine norm (decoder)
    cls_w_<DS>.mem, cls_b_<DS>.mem          INT8 classifier

Usage as a library:
    model = S4DGolden("ECG")                 # or RadioML, SC, sCIFAR
    logits = model.run(x)                    # x: (B, 1024, NIN) integer codes

Command line (checks the model against logits read back from the board):
    python s4d_golden.py ECG --samples ../board/ECG/samples_2158ECG.bin \
                             --board   ../board/ECG/logits_board_ZCU104.bin -n 8

Arithmetic conventions (they mirror the RTL):
    shr_r(x, s) = round-half-up arithmetic right shift (left shift if s <= 0)
    satn(x, w)  = saturation to a signed w-bit integer
"""

import argparse
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent      # repository root

# ============================================================================
# Fixed architecture and number formats (identical for the four datasets)
# ============================================================================
H, L, NLAYER, MODES = 128, 1024, 6, 32       # channels, sequence length, layers, modes
TH = 2 * H                                   # mixing outputs: value branch + gate branch
W_ACT, F_ACT = 16, 11                        # residual stream, Q5.11
W_U, F_U = 16, 11                            # biquad input (output of the layer norm)
W_A, F_A = 24, 22                            # a1, a2
W_B = 16                                     # b0, b1 (per-channel exponent bsh)
W_S, F_Y = 32, 19                            # biquad state / output
W_D = 16                                     # D (fractional bits = generic F_D)
W_BSH = 5                                    # per-channel exponent of b0, b1
W_SSM = 16                                   # output of the biquad bank (fractional bits = F_SSM)
W_W, F_W = 12, 11                            # mixing weights
W_MACC = 32                                  # mixing accumulator
F_SIGI, F_SIGO = 11, 14                      # sigmoid input / output fractional bits
W_WENC, W_BENC = 18, 32                      # encoder weight / bias width
W_BNW, W_BNB = 16, 32                        # affine norm weight / bias width
W_CIN, W_CLS, W_ACC = 8, 8, 32               # classifier input, weight, accumulator
PWL_XH, PWL_LOG2N = 3, 8                     # PWL range [-2^XH, 2^XH), 2^LOG2N segments

# weight stream (one layer = NBEAT beats of 64 bit, see s4d_wload_axis.v)
NCOEF = H * MODES                            # biquad coefficient entries (h, mode)
NWO_ROWS, NWO_BEATS = H, (TH * W_W) // 64    # mixing rows (one per input channel) x beats per row
C_BEG = 1                                    # beat 0 = bn_sh of the layer
D_BEG = C_BEG + 2 * NCOEF                    # 2 beats per coefficient entry (80 bit)
NW_BEG = D_BEG + H                           # {bsh, D}
NB_BEG = NW_BEG + H                          # affine norm weight
BO_BEG = NB_BEG + H                          # affine norm bias
WO_BEG = BO_BEG + TH                         # mixing bias, then mixing weights
NBEAT = WO_BEG + NWO_ROWS * NWO_BEATS        # 14977

# ============================================================================
# Per-dataset generics, exactly as set in the four block designs
# ============================================================================
DATASETS = {
    #           mem suffix  NIN  input bits signed  F_WENC  F_SSM  F_D   K   BN_SH  RQ_SHIFT
    "ECG":     dict(suffix="ECG",     nin=12, n_bit_in=16, signed=True,  f_wenc=31, f_ssm=9, f_d=13, k=5,  bn_sh=13, rq_shift=6),
    "RadioML": dict(suffix="RadioML", nin=2,  n_bit_in=16, signed=True,  f_wenc=34, f_ssm=9, f_d=13, k=24, bn_sh=8,  rq_shift=8),
    "SC":      dict(suffix="SP",      nin=12, n_bit_in=16, signed=True,  f_wenc=31, f_ssm=9, f_d=13, k=35, bn_sh=9,  rq_shift=7),
    "sCIFAR":  dict(suffix="sCIFAR",  nin=1,  n_bit_in=8,  signed=False, f_wenc=26, f_ssm=8, f_d=13, k=10, bn_sh=12, rq_shift=6),
}


# ============================================================================
# Integer helpers
# ============================================================================
def shr_r(x, s):
    """Arithmetic shift right with round-half-up; s scalar or array (s <= 0 = shift left)."""
    s = np.asarray(s)
    if s.ndim == 0:
        s = int(s)
        return x << (-s) if s <= 0 else (x + (1 << (s - 1))) >> s
    s_pos = np.maximum(s, 1)
    shifted_right = (x + (np.int64(1) << (s_pos - 1))) >> s_pos
    shifted_left = x << np.maximum(-s, 0)
    return np.where(s > 0, shifted_right, shifted_left)


def satn(x, w):
    """Saturate to a signed w-bit integer."""
    return np.clip(x, -(1 << (w - 1)), (1 << (w - 1)) - 1)


def to_signed(value, w):
    """Interpret the low w bits of an unsigned integer (or array) as two's complement."""
    value = np.asarray(value, dtype=object) & ((1 << w) - 1)
    return np.where(value >= (1 << (w - 1)), value - (1 << w), value).astype(np.int64)


def read_hex(path):
    """One hexadecimal word per line, returned as Python ints (no width limit)."""
    return [int(token, 16) for token in Path(path).read_text().split()]


def pwl(x, base, slope, f_in, high_is_identity, y_lo=0, y_hi=0, w=16):
    """Piecewise-linear function of s4d_pwl.v: 2^LOG2N segments on [-2^XH, 2^XH)."""
    frac_bits = f_in + PWL_XH + 1 - PWL_LOG2N
    offset = 1 << (PWL_XH + f_in)
    x_shifted = np.clip(x, -offset, offset - 1) + offset
    segment, fraction = x_shifted >> frac_bits, x_shifted & ((1 << frac_bits) - 1)
    y = satn(base[segment] + shr_r(slope[segment] * fraction, frac_bits), w)
    y = np.where(x < -offset, y_lo, y)
    return np.where(x >= offset, satn(x, w) if high_is_identity else y_hi, y)


# ============================================================================
# Reading the .mem files
# ============================================================================
def read_pwl_lut(path):
    """PWL LUT: word = {slope[15:0], base[15:0]}."""
    words = read_hex(path)
    return to_signed([wd for wd in words], 16), to_signed([wd >> 16 for wd in words], 16)


def decode_layer(beats):
    """Decode one layer of the nmac256 weight stream into integer tensors."""
    assert len(beats) == NBEAT
    layer = {"bn_sh": beats[0] & 0x3F}

    # biquad coefficients, entry e = h*MODES + mode, 80 bit split in two beats
    coef = [beats[C_BEG + 2 * e] | (beats[C_BEG + 2 * e + 1] << 64) for e in range(NCOEF)]
    layer["a1"] = to_signed([c for c in coef], 24).reshape(H, MODES)
    layer["a2"] = to_signed([c >> 24 for c in coef], 24).reshape(H, MODES)
    layer["b0"] = to_signed([c >> 48 for c in coef], 16).reshape(H, MODES)
    layer["b1"] = to_signed([c >> 64 for c in coef], 16).reshape(H, MODES)

    # {bsh[20:16], D[15:0]} per channel
    d_words = beats[D_BEG:D_BEG + H]
    layer["D"] = to_signed(d_words, W_D)
    layer["bsh"] = np.array([(wd >> W_D) & ((1 << W_BSH) - 1) for wd in d_words], dtype=np.int64)

    # affine norm of the layer and mixing bias
    layer["bn_w"] = to_signed(beats[NW_BEG:NW_BEG + H], W_BNW)
    layer["bn_b"] = to_signed(beats[NB_BEG:NB_BEG + H], W_BNB)
    layer["bout"] = to_signed(beats[BO_BEG:BO_BEG + TH], W_MACC)

    # mixing weights: row h (input channel) = 48 beats, output c at row[c*12 +: 12]
    wout = np.zeros((TH, H), dtype=np.int64)
    for h in range(H):
        row = 0
        for k in range(NWO_BEATS):
            row |= beats[WO_BEG + h * NWO_BEATS + k] << (64 * k)
        wout[:, h] = to_signed([row >> (c * W_W) for c in range(TH)], W_W)
    layer["wout"] = wout
    return layer


# ============================================================================
# Model
# ============================================================================
class S4DGolden:
    """Bit-exact model of encoder + 6 S4D layers + decoder for one dataset."""

    def __init__(self, dataset, mem_root=REPO_DIR):
        self.cfg = cfg = DATASETS[dataset]
        mem_dir = Path(mem_root) / f"mem_{cfg['suffix']}"
        sfx = cfg["suffix"]

        # derived shifts (same expressions as in the RTL)
        self.f_ssm = cfg["f_ssm"]
        self.sh_ssm = F_Y - self.f_ssm                   # biquad accumulator -> SSM output
        f_acc = F_W + self.f_ssm                         # mixing accumulator fractional bits
        self.rq_a = f_acc - F_ACT                        # value branch -> Q5.11
        self.rq_g = f_acc - F_SIGI                       # gate branch  -> sigmoid input
        self.sh_d = cfg["f_d"] + F_U - F_Y               # D * u -> biquad output format
        self.enc_shift = cfg["f_wenc"] - F_ACT           # encoder accumulator -> Q5.11

        # weight stream: 6 layers
        beats = read_hex(mem_dir / f"w_stream_nmac256_{sfx}.mem")
        assert len(beats) == NLAYER * NBEAT, f"weight stream has {len(beats)} beats"
        self.layers = [decode_layer(beats[m * NBEAT:(m + 1) * NBEAT]) for m in range(NLAYER)]

        # encoder: one row per output channel, NIN weights of 18 bit, input c0 in the LSBs
        nin = cfg["nin"]
        enc_rows = read_hex(mem_dir / f"enc_w_{sfx}.mem")
        self.enc_w = np.array([[int(to_signed(row >> (c * W_WENC), W_WENC)) for c in range(nin)]
                               for row in enc_rows], dtype=np.int64)           # (H, NIN)
        self.enc_b = to_signed(read_hex(mem_dir / f"enc_b_{sfx}.mem"), W_BENC)

        # PWL lookup tables
        self.gelu_base, self.gelu_slope = read_pwl_lut(mem_dir / f"gelu_lut_{sfx}.mem")
        self.sig_base, self.sig_slope = read_pwl_lut(mem_dir / f"sigmoid_lut_{sfx}.mem")

        # decoder: final affine norm and INT8 classifier
        self.nf_w = to_signed(read_hex(mem_dir / f"l6_bn_w_{sfx}.mem"), W_BNW)
        self.nf_b = to_signed(read_hex(mem_dir / f"l6_bn_b_{sfx}.mem"), W_BNB)
        k = cfg["k"]
        cls_rows = read_hex(mem_dir / f"cls_w_{sfx}.mem")                    # row h: K bytes
        self.cls_w = np.array([[int(to_signed(row >> (c * W_CLS), W_CLS)) for row in cls_rows]
                               for c in range(k)], dtype=np.int64)             # (K, H)
        self.cls_b = to_signed(read_hex(mem_dir / f"cls_b_{sfx}.mem"), W_ACC)

    # ------------------------------------------------------------------ blocks
    def encoder(self, x_step):
        """x_step: (B, NIN) input codes of one time step -> (B, H) in Q5.11."""
        acc = x_step @ self.enc_w.T + self.enc_b[None, :]
        return satn(shr_r(acc, self.enc_shift), W_ACT)

    @staticmethod
    def affine(x, w, b, shift, w_out):
        """Affine norm (folded BatchNorm): sat(shr_r(w*x + b, shift))."""
        return satn(shr_r(w[None, :] * x + b[None, :], shift), w_out)

    def layer_step(self, layer, state, x):
        """One time step of one S4D layer (s4d_layer.v). state = [s1, s2]."""
        s1, s2 = state
        u = self.affine(x, layer["bn_w"], layer["bn_b"], layer["bn_sh"], W_U)

        # biquad bank: H channels x MODES modes, transposed direct form II
        u3 = u[:, :, None]
        bsh = layer["bsh"][None, :, None]
        y = satn(shr_r(layer["b0"][None] * u3 + np.left_shift(s1, bsh), bsh), W_S)
        a1y = satn(shr_r(layer["a1"][None] * y, F_A), W_S)
        a2y = satn(shr_r(layer["a2"][None] * y, F_A), W_S)
        state[0] = satn(satn(shr_r(layer["b1"][None] * u3, bsh), W_S) - a1y + s2, W_S)
        state[1] = -a2y

        # D feed-through + sum of the modes -> SSM output, GELU
        acc = satn(shr_r(layer["D"][None, :] * u, self.sh_d), W_S) + y.sum(2)
        y_ssm = satn(shr_r(acc, self.sh_ssm), W_SSM)
        g = pwl(y_ssm, self.gelu_base, self.gelu_slope, self.f_ssm, high_is_identity=True)

        # mixing (TH outputs) + GLU: value * sigmoid(gate), residual add
        z = satn(g @ layer["wout"].T + layer["bout"][None, :], W_MACC)
        value = satn(shr_r(z[:, :H], self.rq_a), 16)
        gate = satn(shr_r(z[:, H:], self.rq_g), 16)
        sig = pwl(gate, self.sig_base, self.sig_slope, F_SIGI,
                  high_is_identity=False, y_lo=0, y_hi=1 << F_SIGO)
        out = satn(shr_r(value * sig, F_SIGO), W_ACT)
        return satn(x + out, W_ACT)

    # ------------------------------------------------------------------ model
    def run(self, x):
        """x: (B, L, NIN) integer input codes -> (B, K) integer logits (int32 range)."""
        x = np.asarray(x, dtype=np.int64)
        if x.ndim == 2:                                  # (B, L) for NIN = 1
            x = x[:, :, None]
        batch = x.shape[0]
        assert x.shape[1:] == (L, self.cfg["nin"]), f"input shape {x.shape}"
        states = [[np.zeros((batch, H, MODES), np.int64), np.zeros((batch, H, MODES), np.int64)]
                  for _ in range(NLAYER)]
        pool = np.zeros((batch, H), dtype=np.int64)
        for t in range(L):
            h = self.encoder(x[:, t])
            for layer, state in zip(self.layers, states):
                h = self.layer_step(layer, state, h)
            pool += self.affine(h, self.nf_w, self.nf_b, self.cfg["bn_sh"], W_ACT)
        x_bar = satn(shr_r(pool, int(np.log2(L))), W_ACT)                 # mean pool
        x_int8 = satn(shr_r(x_bar, self.cfg["rq_shift"]), W_CIN)
        return satn(x_int8 @ self.cls_w.T + self.cls_b[None, :], W_ACC)


# ============================================================================
# Sample files used on the board
# ============================================================================
def load_samples(dataset, path, first, count):
    """Read `count` records starting at `first` from a board samples .bin file."""
    cfg = DATASETS[dataset]
    dtype = ("<i2" if cfg["signed"] else "<u2") if cfg["n_bit_in"] == 16 else ("i1" if cfg["signed"] else "u1")
    record_values = L * cfg["nin"]
    record_bytes = record_values * np.dtype(dtype).itemsize
    raw = np.fromfile(path, dtype=dtype, count=count * record_values, offset=first * record_bytes)
    return raw.reshape(count, L, cfg["nin"]).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="Bit-exact S4D model: compare with board logits")
    parser.add_argument("dataset", choices=list(DATASETS))
    parser.add_argument("--samples", required=True, help="samples .bin used on the board")
    parser.add_argument("--board", help="logits_board_*.bin written by the board program")
    parser.add_argument("--first", type=int, default=0, help="first record")
    parser.add_argument("-n", "--count", type=int, default=8, help="number of records")
    args = parser.parse_args()

    model = S4DGolden(args.dataset)
    x = load_samples(args.dataset, args.samples, args.first, args.count)
    logits = model.run(x)
    print(f"{args.dataset}: logits of records {args.first}..{args.first + args.count - 1}")
    for i, row in enumerate(logits[:3]):
        print(f"  record {args.first + i}: {row.tolist()}")
    if args.board:
        k = DATASETS[args.dataset]["k"]
        board = np.fromfile(args.board, dtype="<i4").reshape(-1, k)[args.first:args.first + args.count]
        n_equal = int((board == logits).all(1).sum())
        print(f"  identical to the board: {n_equal}/{args.count} records")


if __name__ == "__main__":
    main()
