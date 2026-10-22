#!/usr/bin/env python3
"""
make_board_files.py -- regenerate the binary files used by the board test
programs (board/<dataset>/*.c) from the original public datasets.

For each dataset it writes, in board/<task>/ (next to the C test program) or in --out:
    samples_*.bin          input codes exactly as streamed to the encoder by the DMA
    labels_*.bin           ground-truth labels, one byte per example
    snr_RadioML_9984.bin   (RadioML only) SNR in dB of each frame, int8
    logits_golden_*.bin    logits of the bit-exact model (s4d_golden.py) for
                           the examples that the board program compares bit by bit

The preprocessing is the one used for training and quantization; the files
produced are byte-identical to the ones used for the measurements reported in
the README. The golden logits are computed with the bit-exact model
python/s4d_golden.py.

Datasets (paths given with --data):
    ECG      PTB-XL 1.0.3 folder (ptbxl_database.csv, scp_statements.csv, records100/)
    RadioML  RadioML 2018.01A file GOLD_XYZ_OSC.0001_1024.hdf5
    SC       Speech Commands v0.02 folder (testing_list.txt and the word folders)
    sCIFAR   CIFAR-10 python folder cifar-10-batches-py (test_batch)

Usage (from the board/ folder):
    python make_board_files.py ECG     --data /path/to/ptb-xl
    python make_board_files.py RadioML --data /path/to/GOLD_XYZ_OSC.0001_1024.hdf5
    python make_board_files.py SC      --data /path/to/speech_commands_v0.02
    python make_board_files.py sCIFAR  --data /path/to/cifar-10-batches-py
Options:
    --out DIR       output folder (default: board/<task>/)
    --workers N     processes used for the golden logits (default: all cores)
    --no-golden     write only samples and labels
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import sys

import numpy as np

HERE = Path(__file__).resolve().parent                       # board/
sys.path.insert(0, str(HERE.parent / "python"))              # bit-exact model
import s4d_golden  # noqa: E402

TASK_FOLDER = {"ECG": "ECG", "RadioML": "RadioML", "SC": "SpeechCommands", "sCIFAR": "sCIFAR"}
L = 1024                     # sequence length of the core
F_IN = 11                    # input codes are Q5.11 (except sCIFAR: raw uint8 pixels)


def quantize_q511(values):
    """Real values -> int16 Q5.11 codes with saturation."""
    return np.clip(np.round(values * (1 << F_IN)), -32768, 32767).astype(np.int16)


# ============================================================================
# ECG -- PTB-XL superdiagnostic, fold 10, records with at least one superclass
# ============================================================================
ECG_CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]      # label bit k = class k
ECG_SCALE = 1.8083178552618622   # 1/mV; measured on the training folds (99.9th percentile of |x - median| -> 4.0)


def make_ecg(data):
    import ast
    import pandas as pd
    data = Path(data)
    db = pd.read_csv(data / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db.scp_codes.apply(ast.literal_eval)
    statements = pd.read_csv(data / "scp_statements.csv", index_col=0)
    statements = statements[statements.diagnostic == 1]
    superclasses = db.scp_codes.apply(
        lambda codes: sorted({statements.loc[c].diagnostic_class for c in codes if c in statements.index}))
    test = db[(db.strat_fold == 10) & (superclasses.apply(len) >= 1)]          # 2158 records

    samples = np.zeros((len(test), L, 12), dtype=np.int16)
    labels = np.zeros(len(test), dtype=np.uint8)
    for i, (ecg_id, row) in enumerate(test.iterrows()):
        mv = np.fromfile(data / f"{row.filename_lr}.dat", dtype="<i2").reshape(1000, 12) / 1000.0
        samples[i, :1000] = quantize_q511((mv - np.median(mv, axis=0)) * ECG_SCALE)   # 100 Hz, 10 s + 24 zero steps
        labels[i] = sum(1 << k for k, c in enumerate(ECG_CLASSES) if c in superclasses[ecg_id])
    return dict(samples=("samples_2158ECG.bin", samples), labels=("labels_2158ECG.bin", labels),
                golden=("logits_golden_PTB_XL_1000.bin", list(range(1000)), "plain"))


# ============================================================================
# RadioML 2018.01A -- 9984 test frames, 16 per (modulation, SNR) cell
# ============================================================================
RADIOML_SCALE = 1.3024024307265825   # fixed input scale, measured on the training split


def make_radioml(data):
    import h5py
    frames = np.loadtxt(HERE / "radioml_test_frames.txt", dtype=np.int64)    # indices in the HDF5 file
    with h5py.File(data, "r") as h5:
        x = np.stack([np.asarray(h5["X"][i], dtype=np.float32) for i in frames])  # (N, 1024, 2), I then Q
        labels = np.array([int(np.argmax(h5["Y"][i])) for i in frames], dtype=np.uint8)
        snr = np.array([int(h5["Z"][i][0]) for i in frames], dtype=np.int8)
    samples = quantize_q511(x * RADIOML_SCALE)
    return dict(samples=("samples_RadioML_9984.bin", samples), labels=("labels_RadioML_9984.bin", labels),
                extra=[("snr_RadioML_9984.bin", snr)],
                golden=("logits_golden_RadioML_9984.bin", list(range(len(frames))), "plain"))


# ============================================================================
# Speech Commands v0.02 -- 35 words, official test split (11005 recordings)
# ============================================================================
SC_INPUT_SCALE = 8.0       # real value = int16 / 32768 * 8 -> code = int16 / 2
SC_GOLDEN = [0, 165, 372, 557, 751, 971, 1377, 1785, 2230, 2402]   # first recording of words 0..9


def antialias_fir():
    """Kaiser FIR designed at 48 kHz: flat to 5 kHz, 80 dB from 6 kHz (243 taps)."""
    from scipy import signal
    fs_poly, f_pass, f_stop = 48000, 5000.0, 6000.0
    num_taps, beta = signal.kaiserord(80.0, (f_stop - f_pass) / (fs_poly / 2))
    return signal.firwin(num_taps | 1, 0.5 * (f_pass + f_stop), window=("kaiser", beta), fs=fs_poly)


def make_sc(data):
    import soundfile
    from scipy import signal
    data = Path(data)
    words = sorted(p.name for p in data.iterdir() if p.is_dir() and not p.name.startswith("_"))
    files = sorted(line.strip() for line in (data / "testing_list.txt").read_text().splitlines() if line.strip())
    wav16 = np.zeros((len(files), 16000), dtype=np.int16)        # 1 s at 16 kHz, zero padded at the end
    labels = np.zeros(len(files), dtype=np.uint8)
    for i, name in enumerate(files):
        audio, rate = soundfile.read(data / name, dtype="float32")
        assert rate == 16000
        n = min(len(audio), 16000)
        wav16[i, :n] = np.clip(audio[:n] * 32767.0, -32768, 32767).astype(np.int16)
        labels[i] = words.index(name.split("/")[0])
    # 16 kHz -> 12 kHz (polyphase 3/4) with the anti-aliasing FIR, rounded to int16
    wav12 = signal.resample_poly(wav16.astype(np.float32), 3, 4, axis=-1, window=antialias_fir())[:, :12000]
    wav12 = np.clip(np.round(wav12), -32768, 32767).astype(np.int16)
    codes = np.clip(np.round(wav12.astype(np.float32) * (SC_INPUT_SCALE * (1 << F_IN) / 32768.0)),
                    -32768, 32767)
    samples = np.zeros((len(files), L, 12), dtype=np.int16)      # 1000 steps of 12 samples + 24 zero steps
    samples[:, :1000] = codes.reshape(len(files), 1000, 12)
    return dict(samples=("samples_SC_11005.bin", samples), labels=("labels_SC_11005.bin", labels),
                golden=("logits_golden_SC_10.bin", SC_GOLDEN, "indexed"))


# ============================================================================
# sCIFAR -- CIFAR-10 test set, 32x32 grayscale, one raw uint8 pixel per step
# ============================================================================
def make_scifar(data):
    import pickle
    from PIL import Image
    with open(Path(data) / "test_batch", "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    rgb = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    samples = np.stack([np.asarray(Image.fromarray(img).convert("L")) for img in rgb]).reshape(-1, L)
    labels = np.array(batch[b"labels"], dtype=np.uint8)
    return dict(samples=("samples_sCIFAR_10000.bin", samples.astype(np.uint8)),
                labels=("labels_sCIFAR_10000.bin", labels),
                golden=("logits_golden_sCIFAR_1000.bin", list(range(1000)), "plain"))


# ============================================================================
# Golden logits (bit-exact model), in parallel
# ============================================================================
_WORK = {}


def _golden_block(block):
    model = s4d_golden.S4DGolden(_WORK["dataset"])
    return model.run(_WORK["x"][block])


def golden_logits(dataset, x, workers, block_size=16):
    _WORK.update(dataset=dataset, x=x)
    blocks = [np.arange(i, min(i + block_size, len(x))) for i in range(0, len(x), block_size)]
    with mp.get_context("fork").Pool(workers) as pool:
        out = []
        for n_done, res in enumerate(pool.imap(_golden_block, blocks), 1):
            out.append(res)
            print(f"  golden {min(n_done * block_size, len(x))}/{len(x)}", end="\r", flush=True)
    print()
    return np.concatenate(out).astype("<i4")


MAKERS = {"ECG": make_ecg, "RadioML": make_radioml, "SC": make_sc, "sCIFAR": make_scifar}


def main():
    parser = argparse.ArgumentParser(description="Regenerate the board test files")
    parser.add_argument("dataset", choices=list(MAKERS))
    parser.add_argument("--data", required=True, help="original dataset (see the docstring)")
    parser.add_argument("--out", help="output folder (default: board/<task>/)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--no-golden", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else HERE / TASK_FOLDER[args.dataset]
    out.mkdir(parents=True, exist_ok=True)
    files = MAKERS[args.dataset](args.data)

    to_write = [files["samples"], files["labels"]] + files.get("extra", [])
    for name, array in to_write:
        array.astype(array.dtype.newbyteorder("<")).tofile(out / name)
        print(f"  {name}: {array.shape[0]} examples, {(out / name).stat().st_size} bytes")

    if not args.no_golden:
        name, indices, layout = files["golden"]
        x = files["samples"][1][indices].astype(np.int64)
        logits = golden_logits(args.dataset, x, args.workers)
        if layout == "indexed":                  # rows [index in the test set, K logits]
            logits = np.concatenate([np.array(indices, dtype="<i4")[:, None], logits], axis=1)
        logits.astype("<i4").tofile(out / name)
        print(f"  {name}: {len(indices)} examples, {(out / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()
