#!/usr/bin/env python3
"""Export_s4d_wstream.py -- repacks the per-layer .mem files already generated
(mem_trained_sc/l<m>_*.mem) into ONE stream at AXI_DW=64 bit, in the exact
order and format that s4d_wload_axis.v expects on s_axis_w.

It does not generate weights: it reads the ones ALREADY written by
Quantize_s4d_speech_command.py (WRITE_MEM=True) and repacks them. If those
files change, it must be run again.

Per-layer format (read from rtl/s4d_wload_axis.v, lines 63-77), NBEAT beats:

  pos                    content                       source
  ---------------------  ---------------------------  ------------------------
  [0]                    bn_sh (6 bit)                 quant_report.json
                                                        affine[m].SH  (m=0..5;
                                                        index 6 is the final
                                                        norm, OUTSIDE this
                                                        stream)
  [1 .. 1+2*NCOEF)        coef, 2 beats/entry:          l<m>_s4d_coef.mem
                          even beat = value[63:0]       (same order:
                          odd beat  = value[79:64]      unit, slot, mode)
                          in the low bits
  [.. +H)                 {bsh[20:16], D[15:0]}         l<m>_s4d_D.mem
                          (value ALREADY packed,        (already contains bsh+D,
                          21 bit, in the low bits)       see write_layer_mem)
  [.. +H)                 bn_w (16 bit)                 l<m>_bn_w.mem
  [.. +H)                 bn_b (32 bit)                 l<m>_bn_b.mem
  [.. +TH)                bout (32 bit)                 l<m>_s4d_bout.mem
  [.. +NWO*WOBEAT)        wout, 12 beats/row of 64 bit  l<m>_s4d_wout.mem
                          each, LSB-first:              (768-bit row
                          beat k = row[64k +: 64]       split into 12 pieces,
                          (see s4d_mixing_glu.v,         see s4d_mixing_glu.v
                          lines 216-224: mem[fi] =       line 223)
                          w_file[fi][wb*LD_W +: LD_W])

The final file concatenates the 6 layers in order (l0, l1, ..., l5): it is the
order in which the FSM of s4d_core_axis requests them (wl_start repeats for m=0..5).

Check: it reads back the generated stream, decodes it with the SAME logic as
s4d_wload_axis (in Python) and compares the result with the original values
read from the per-layer .mem files. It is not a luxury: a packing error here
produces a core that loads wrong weights without any error signal.
"""
import json
import sys
from pathlib import Path

MEM_DIR = Path("mem_trained_sc")
OUT_FILE = MEM_DIR / "w_stream.mem"
REPORT = MEM_DIR / "quant_report.json"

H, NUNITS, NCH, MODES = 128, 8, 16, 32
LOG2_NB = 9                       # log2(NCH*MODES) = log2(512)
NCOEF = NUNITS * (1 << LOG2_NB)   # 4096
NMAC, AXI_DW = 64, 64
W_W = 12
WOBEAT = (NMAC * W_W) // AXI_DW   # 12
TH = 2 * H                        # 256
NPH = 4
NWO = NPH * H                     # 512
W_D, W_BSH = 16, 5
NLAYER = 6

H_END = 1
C_END = H_END + 2 * NCOEF
D_END = C_END + H
NW_END = D_END + H
NB_END = NW_END + H
BO_END = NB_END + TH
WO_END = BO_END + NWO * WOBEAT
NBEAT = WO_END


def read_hex_rows(path):
    return [int(t, 16) for t in Path(path).read_text().split()]


def pack_layer(m, bn_sh):
    """Returns the list of NBEAT 64-bit integers for layer m."""
    beats = [bn_sh & 0x3F]

    coef = read_hex_rows(MEM_DIR / f"l{m}_s4d_coef.mem")
    assert len(coef) == NCOEF, f"l{m}_s4d_coef.mem: {len(coef)} rows, {NCOEF} expected"
    for v in coef:
        beats.append(v & ((1 << 64) - 1))
        beats.append((v >> 64) & 0xFFFF)

    D = read_hex_rows(MEM_DIR / f"l{m}_s4d_D.mem")
    assert len(D) == H, f"l{m}_s4d_D.mem: {len(D)} rows, {H} expected"
    beats.extend(v & ((1 << (W_D + W_BSH)) - 1) for v in D)

    bn_w = read_hex_rows(MEM_DIR / f"l{m}_bn_w.mem")
    assert len(bn_w) == H
    beats.extend(v & 0xFFFF for v in bn_w)

    bn_b = read_hex_rows(MEM_DIR / f"l{m}_bn_b.mem")
    assert len(bn_b) == H
    beats.extend(v & 0xFFFFFFFF for v in bn_b)

    bout = read_hex_rows(MEM_DIR / f"l{m}_s4d_bout.mem")
    assert len(bout) == TH, f"l{m}_s4d_bout.mem: {len(bout)} rows, {TH} expected"
    beats.extend(v & 0xFFFFFFFF for v in bout)

    wout = read_hex_rows(MEM_DIR / f"l{m}_s4d_wout.mem")
    assert len(wout) == NWO, f"l{m}_s4d_wout.mem: {len(wout)} rows, {NWO} expected"
    for row in wout:
        for k in range(WOBEAT):
            beats.append((row >> (64 * k)) & ((1 << 64) - 1))

    assert len(beats) == NBEAT, f"layer {m}: {len(beats)} beats, {NBEAT} expected"
    return beats


def decode_layer(beats):
    """Python mirror of s4d_wload_axis.v: rebuilds the original fields from
    the beats, for the check. Same addressing arithmetic as the RTL
    (lines 89-94, 140-177 of the source)."""
    assert len(beats) == NBEAT
    bn_sh = beats[0]

    coef = []
    for i in range(NCOEF):
        lo, hi = beats[H_END + 2 * i], beats[H_END + 2 * i + 1]
        coef.append((hi << 64) | lo)

    D = beats[C_END:D_END]
    bn_w = beats[D_END:NW_END]
    bn_b = beats[NW_END:NB_END]
    bout = beats[NB_END:BO_END]

    wo_beats = beats[BO_END:WO_END]
    wout = []
    for r in range(NWO):
        row = 0
        for k in range(WOBEAT):
            row |= wo_beats[r * WOBEAT + k] << (64 * k)
        wout.append(row)

    return dict(bn_sh=bn_sh, coef=coef, D=D, bn_w=bn_w, bn_b=bn_b, bout=bout, wout=wout)


def main():
    if not REPORT.is_file():
        sys.exit(f"{REPORT} not found: it is needed for bn_sh per layer "
                 f"(run Quantize_s4d_speech_command.py again with WRITE_MEM=True)")
    rep = json.loads(REPORT.read_text())
    sh_per_layer = [a["SH"] for a in rep["affine"][:NLAYER]]
    print(f"bn_sh per layer (from quant_report.json): {sh_per_layer}")

    all_beats = []
    for m in range(NLAYER):
        layer_beats = pack_layer(m, sh_per_layer[m])

        # --- check: decode and compare with the original .mem files -------
        dec = decode_layer(layer_beats)
        coef0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_coef.mem")
        D0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_D.mem")
        bn_w0 = read_hex_rows(MEM_DIR / f"l{m}_bn_w.mem")
        bn_b0 = read_hex_rows(MEM_DIR / f"l{m}_bn_b.mem")
        bout0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_bout.mem")
        wout0 = read_hex_rows(MEM_DIR / f"l{m}_s4d_wout.mem")
        assert dec["bn_sh"] == (sh_per_layer[m] & 0x3F)
        assert dec["coef"] == coef0, f"layer {m}: coef does not match"
        assert dec["D"] == [v & ((1 << 21) - 1) for v in D0], f"layer {m}: D does not match"
        assert dec["bn_w"] == [v & 0xFFFF for v in bn_w0], f"layer {m}: bn_w does not match"
        assert dec["bn_b"] == [v & 0xFFFFFFFF for v in bn_b0], f"layer {m}: bn_b does not match"
        assert dec["bout"] == [v & 0xFFFFFFFF for v in bout0], f"layer {m}: bout does not match"
        assert dec["wout"] == wout0, f"layer {m}: wout does not match"
        print(f"  layer {m}: {NBEAT} beats, check OK "
              f"(coef/D/bn_w/bn_b/bout/wout identical to the source .mem files)")

        all_beats.extend(layer_beats)

    total = NLAYER * NBEAT
    assert len(all_beats) == total
    OUT_FILE.write_text("\n".join(f"{b:016x}" for b in all_beats) + "\n")
    print(f"\nwritten {OUT_FILE}: {total} rows of 64 bit "
          f"({total * 8 / 1024:.1f} KiB), NBEAT={NBEAT} per layer x {NLAYER} layers")
    print(f"beat boundaries per layer: bn_sh=1 coef={2*NCOEF} D={H} bn_w={H} bn_b={H} "
          f"bout={TH} wout={NWO*WOBEAT}  (NBEAT={NBEAT})")


if __name__ == "__main__":
    main()
