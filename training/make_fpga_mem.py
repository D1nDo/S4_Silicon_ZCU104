#!/usr/bin/env python3
"""
make_fpga_mem.py -- turns the output of a quantization run into the 9 .mem
files used by the Vivado project (folders mem_<SUFFIX>/ of the repository).

Input: the mem/ folder written by a quantization script (e.g. Quantize_s4d_ecg.py
or the sCIFAR script), containing the per-layer files l<m>_*.mem, the encoder,
decoder and LUT files, and quant_report.json (it provides the per-layer BN shift).

Steps:
  1. per-layer files -> weight stream for the Zynq-7020 core (4-phase mixing
     layout, 512 rows x 12 beats per layer), with Export_s4d_wstream.py, which
     also decodes the stream back and checks it against the source files;
  2. 4-phase stream -> ZCU104 stream (single-row mixing layout, 128 rows x 48
     beats per layer). It is a pure permutation of the mixing section:
         new (row h, beat ph*12 + k)  <-  old (row ph*H + h, beat k)
     everything else in the layer (bn_sh, biquad coefficients, D, BN, mixing
     bias) is copied unchanged;
  3. copy of the 8 ROM files of encoder, decoder and LUTs with the suffix used
     by the block design (e.g. enc_w_ECG.mem).

Output, in --out:
    w_stream_nmac256_<SUFFIX>.mem   weight ROM of s4d_weightrom_axis
    enc_w_<SUFFIX>.mem, enc_b_<SUFFIX>.mem
    gelu_lut_<SUFFIX>.mem, sigmoid_lut_<SUFFIX>.mem
    l6_bn_w_<SUFFIX>.mem, l6_bn_b_<SUFFIX>.mem
    cls_w_<SUFFIX>.mem, cls_b_<SUFFIX>.mem

Usage (from the training/ folder):
    python make_fpga_mem.py --mem-dir experiments/<run>/quant_<stamp>_hw/mem --suffix ECG --out ../mem_ECG
"""

import argparse
import shutil
from pathlib import Path

import Export_s4d_wstream as XS

H = 128
N_LAYERS = 6
N_PHASES = 4                     # Zynq-7020 mixing: 4 phases of 64 MACs
BEATS_PER_ROW_Z7020 = 12         # 768-bit mixing row = 12 beats of 64 bit
ROM_FILES = ["enc_w", "enc_b", "gelu_lut", "sigmoid_lut", "l6_bn_w", "l6_bn_b", "cls_w", "cls_b"]


def to_nmac256(beats_z7020):
    """4-phase (Zynq-7020) weight stream -> single-row (ZCU104) weight stream."""
    assert len(beats_z7020) == N_LAYERS * XS.NBEAT
    out = []
    for layer in range(N_LAYERS):
        layer_beats = beats_z7020[layer * XS.NBEAT:(layer + 1) * XS.NBEAT]
        head = layer_beats[:XS.BO_END]                    # bn_sh, coef, D, bn_w, bn_b, bout: unchanged
        mixing_old = layer_beats[XS.BO_END:XS.WO_END]     # 512 rows x 12 beats
        mixing_new = []
        for h in range(H):                                # new row = input channel h
            for phase in range(N_PHASES):                 # the 4 phases become 4 blocks of 12 beats
                old_row = phase * H + h
                mixing_new.extend(mixing_old[old_row * BEATS_PER_ROW_Z7020:
                                             (old_row + 1) * BEATS_PER_ROW_Z7020])
        out.extend(head + mixing_new)
    return out


def main():
    parser = argparse.ArgumentParser(description="quantization output -> .mem files of the Vivado project")
    parser.add_argument("--mem-dir", required=True, help="mem/ folder of a quantization run")
    parser.add_argument("--suffix", required=True, help="suffix of the block design files (ECG, RadioML, SP, sCIFAR)")
    parser.add_argument("--out", required=True, help="output folder, e.g. ../mem_ECG")
    args = parser.parse_args()

    mem_dir, out = Path(args.mem_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. per-layer files -> 4-phase stream (written next to the source files and checked)
    XS.MEM_DIR = mem_dir
    XS.OUT_FILE = mem_dir / "w_stream.mem"
    XS.REPORT = mem_dir / "quant_report.json"
    XS.main()
    beats = [int(token, 16) for token in XS.OUT_FILE.read_text().split()]

    # 2. 4-phase stream -> ZCU104 stream
    stream_path = out / f"w_stream_nmac256_{args.suffix}.mem"
    stream_path.write_text("\n".join(f"{b:016x}" for b in to_nmac256(beats)) + "\n")
    print(f"written {stream_path} ({N_LAYERS} x {XS.NBEAT} beats)")

    # 3. ROM files of encoder, decoder and LUTs
    for name in ROM_FILES:
        shutil.copy2(mem_dir / f"{name}.mem", out / f"{name}_{args.suffix}.mem")
        print(f"written {out / f'{name}_{args.suffix}.mem'}")


if __name__ == "__main__":
    main()
