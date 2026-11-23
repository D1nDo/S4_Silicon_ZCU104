#!/usr/bin/env python3
"""
make_fpga_mem.py -- turns the output of a quantization run into the 9 .mem
files used by the Vivado project (folders mem_<SUFFIX>/ of the repository).

Input: the mem/ folder written by a quantization script (e.g. Quantize_s4d_ecg.py
or the sCIFAR script), containing the per-layer files l<m>_*.mem, the encoder,
decoder and LUT files, and quant_report.json (it provides the per-layer BN shift).

Steps:
  1. per-layer files -> ZCU104 weight stream (single-row mixing layout, NMAC = 2H),
     with make_wstream.py; H and MODES = N/2 are read from the files, so the same
     script serves every (H, N). With H = 128, N = 64 the stream is identical,
     bit for bit, to the one built through the 4-phase Zynq-7020 stream;
  2. copy of the 8 ROM files of encoder, decoder and LUTs with the suffix used
     by the block design (e.g. enc_w_ECG.mem).

Output, in --out:
    w_stream_nmac256_<SUFFIX>.mem   weight ROM of s4d_weightrom_axis
    enc_w_<SUFFIX>.mem, enc_b_<SUFFIX>.mem
    gelu_lut_<SUFFIX>.mem, sigmoid_lut_<SUFFIX>.mem
    l6_bn_w_<SUFFIX>.mem, l6_bn_b_<SUFFIX>.mem
    cls_w_<SUFFIX>.mem, cls_b_<SUFFIX>.mem

Usage (from the training/ folder):
    python make_fpga_mem.py --mem-dir experiments/<run>/quant_<stamp>_hw/mem --suffix ECG --out ../mem/mem_ECG
"""

import argparse
import shutil
from pathlib import Path

import make_wstream

ROM_FILES = ["enc_w", "enc_b", "gelu_lut", "sigmoid_lut", "l6_bn_w", "l6_bn_b", "cls_w", "cls_b"]


def main():
    parser = argparse.ArgumentParser(description="quantization output -> .mem files of the Vivado project")
    parser.add_argument("--mem-dir", required=True, help="mem/ folder of a quantization run")
    parser.add_argument("--suffix", required=True, help="suffix of the block design files (ECG, RadioML, SP, sCIFAR)")
    parser.add_argument("--out", required=True, help="output folder, e.g. ../mem/mem_ECG")
    args = parser.parse_args()

    mem_dir, out = Path(args.mem_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. per-layer files -> ZCU104 weight stream
    beats, info = make_wstream.build(mem_dir)
    stream_path = out / f"w_stream_nmac256_{args.suffix}.mem"
    stream_path.write_text("\n".join(f"{b:016x}" for b in beats) + "\n")
    print(f"written {stream_path}: H = {info['H']}, MODES = {info['MODES']}, NBEAT = {info['NBEAT']} "
          f"(set NBEAT, H, MODES in the block design)")

    # 2. ROM files of encoder, decoder and LUTs
    for name in ROM_FILES:
        shutil.copy2(mem_dir / f"{name}.mem", out / f"{name}_{args.suffix}.mem")
        print(f"written {out / f'{name}_{args.suffix}.mem'}")


if __name__ == "__main__":
    main()
