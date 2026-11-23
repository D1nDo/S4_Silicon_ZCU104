#!/usr/bin/env python3
"""
Train_S4D_RadioML.py -- S4D training on RadioML 2018.01A (automatic
modulation classification, 24 classes), for the S4D core with L = 1024.

It runs without arguments, from the root of the repository:

    conda run -n mamba_env python Train_S4D_RadioML.py

All the knobs are constants in the CONFIGURATION block, as in
Train_s4d_ecg.py: the experiment is documented by the file, not by the
command line. A copy of the file is saved in the experiment folder.

DATASET
RadioML 2018.01A (DeepSig), file GOLD_XYZ_OSC.0001_1024.hdf5 (~21 GB):
  X (2555904, 1024, 2) float32   I/Q frames of 1024 samples
  Y (2555904, 24)      one-hot   modulation
  Z (2555904, 1)       int       SNR in dB, from -20 to +30 in steps of 2 (26 values)
4096 frames for each (modulation, SNR) pair. It is the "hard" version:
24 modulations (also 128QAM/256QAM, 64/128APSK, AM-SSB) and a simulated
over-the-air channel with frequency/phase offset, multipath and noise.

ADAPTATION TO THE HARDWARE
  - L = 1024 = frame length: no padding, the mean pool divides by
    1024 exactly as the decoder (>>> LOG2_L).
  - 2 inputs per time step (I, Q): generic encoder with NIN = 2,
    N_BIT_IN = 16, IN_SIGNED = 1, samples in Q5.11 (F_ACT = 11).
    In PyTorch it is a Linear(2, H), as the Linear(12, H) of the ECG.
  - K = 24 classes in the decoder.
  - H = 128, 6 layers, 32 modes: the model is the BN one of s4d_bn, exportable.

PREPROCESSING (the SAME must be used for the board stimuli)
  1. normalization (NORM_MODE):
       "fixed": ONE SCALE for the whole dataset, measured on the training set
                (FIXED_PCT percentile of |I|,|Q| brought to FIXED_TARGET).
                The relative amplitude between frames is kept: it is useful
                information (e.g. SNR), but the data on the board must have
                the same gain.
       "rms"  : each frame divided by its own RMS value and multiplied
                by RMS_TARGET: it is an automatic gain control (AGC),
                what a real receiver does before the classifier.
  2. Q5.11: round(x * 2048), int16 saturation.
The model receives q / 2048, i.e. exactly the fixed-point value.

SPLIT
For each (modulation, SNR) pair the 4096 frames are shuffled with SEED and
split 80/10/10 into training/validation/test: every SNR and every class weigh
the same in all three sets. The accuracy-SNR curves are therefore comparable
with the literature that uses random splits (there is no official split).

METRICS
Global accuracy (mean over all the SNRs, dominated by the low SNRs where no
classifier goes beyond chance), accuracy per SNR, accuracy at SNR >= 0 dB and
at SNR >= 10 dB, confusion matrix at SNR >= 10 dB.
"""

import ast
import copy
import datetime
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# CONFIGURATION -- edited here, not from the command line
# ============================================================================

MONO = "Train_and_quantize_s4d_sCIFAR.py"   # from which the gen_s4d_mem/s4d_export/s4d_bn modules are extracted
H5_PATH = Path("data/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5")   # original DeepSig file
CACHE_ROOT = Path("data")                   # the int16 cache goes to data/radioml_cache_<norm>/
EXP_ROOT = Path("experiments")

# --- dataset ----------------------------------------------------------------
# CORRECT order of the classes in the indices of Y. The classes.txt distributed
# with the dataset has a wrong order (known: see cyclostationary.blog,
# 2020-09-24); this is the order used by the community and checked on the data.
# It is needed only for printing: the training uses the indices.
CLASSES = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
           "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
           "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
           "AM-DSB-SC", "FM", "GMSK", "OQPSK"]
N_FRAMES_EXPECTED = 2555904                 # integrity check of the file
L = 1024                                    # samples per frame = L of the core
NIN = 2                                     # inputs per time step: I and Q
F_IN = 11                                   # Q5.11, F_ACT of the encoder
HW_COMPATIBLE = True                        # L = 1024, H = 128, MODES = 32
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)           # training / validation / test per cell

# --- input normalization ----------------------------------------------------
NORM_MODE = "fixed"     # "fixed" | "rms"  (see the docstring)
FIXED_TARGET = 4.0      # "fixed": the FIXED_PCT percentile of |x| goes to 4.0
FIXED_PCT = 99.9        #          (Q5.11 saturates at +-16: 4x margin)
FIXED_CAL_FRAMES = 20000    # training frames used to measure the scale
RMS_TARGET = 1.0        # "rms": RMS value of the frame after the AGC
                        #        (peaks ~4-5x the rms for the QAMs: far from 16)
H5_CHUNK = 16384        # frames read from the HDF5 at a time while building the cache (~128 MB)

# --- model: the defaults keep the compatibility with the RTL (H=128, MODES=32)
D_MODEL = 128
N_LAYERS = 6
D_STATE = 64            # = 2 * MODES
DROPOUT = 0.1

# --- optimization -----------------------------------------------------------
LR = 1e-2               # groups with and without weight decay
LR_SSM = 1e-3           # S4D kernel (log_A_real, A_imag, log_dt), WD = 0
WD = 0.05
BATCH = 128             # 2M training frames: larger batch than the ECG
EPOCHS = 30
# An "epoch" here is FRAMES_PER_EPOCH frames drawn at random from the training
# set (the whole 2.04M frames would be a very long epoch and a validation
# measurement only once in a while). None = full epoch.
FRAMES_PER_EPOCH = 400_000
WARMUP_EPOCHS = 1
PATIENCE = 6
WORKERS = 8
SEED = 1337
VAL_FRAMES_PER_EPOCH = 52_000   # stratified subset of the validation set for
                                # the selection at each epoch (full val: 255k)
# Data augmentation: random rotation of the constellation by k*90 degrees
# (I,Q) -> (-Q,I). The modulations of the dataset are invariant to these
# rotations (phase offset already random). Off in the base run: one variable
# is changed at a time.
AUG_ROT90 = False
ALLOW_TF32 = True       # TF32 only in the matmuls (Linear); the FFTs of the S4D kernel stay fp32

# --- diagnostics ------------------------------------------------------------
PRINT_SUMMARY = True
SUMMARY_ONLY = False    # True: print the summary and exit without training
SUBSET = None           # int: training frames for a quick test; None = all


# ============================================================================
# project modules
# ============================================================================
def load_modules(mono_path=MONO):
    """gen_s4d_mem / s4d_export / s4d_bn, from the separate files or from the monolith.

    Same extraction as Train_s4d_ecg.py: the strings of the three modules are the
    contract with the RTL and must be executed verbatim.
    """
    try:
        import gen_s4d_mem as G, s4d_export as E, s4d_bn as B
        return G, E, B
    except ImportError:
        pass
    if not Path(mono_path).is_file():
        sys.exit(f"modules not importable and {mono_path} not found: "
                 f"fix MONO at the top of the file")
    import types
    targets = {"_GEN_S4D_MEM_SRC": "gen_s4d_mem",
               "_S4D_EXPORT_SRC": "s4d_export",
               "_S4D_BN_SRC": "s4d_bn"}
    src = {}
    for node in ast.walk(ast.parse(Path(mono_path).read_text())):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in targets:
                    src[targets[t.id]] = node.value.value
    missing = set(targets.values()) - set(src)
    if missing:
        sys.exit(f"{mono_path}: strings not found for {sorted(missing)}")
    mods = {}
    for name in ("gen_s4d_mem", "s4d_export", "s4d_bn"):     # order: dependencies
        m = types.ModuleType(name)
        sys.modules[name] = m
        exec(compile(src[name], name, "exec"), m.__dict__)
        mods[name] = m
    return mods["gen_s4d_mem"], mods["s4d_export"], mods["s4d_bn"]


def radioml_model(B, n_classes):
    """The BN model of s4d_bn with the NIN = 2 input encoder (I, Q).

    Only input_proj is replaced: S4D core, norm, GLU and classifier stay the
    object that the quantization knows how to export. The generic encoder
    computes y[ch] = sum_c W[ch][c] * x[c] + b[ch], i.e. a Linear(2, H).
    """
    model = B.torch_model(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                          dropout=DROPOUT, n_classes=n_classes)
    model.input_proj = nn.Linear(NIN, D_MODEL)
    return model


# ============================================================================
# data: int16 Q5.11 cache built once from the HDF5
# ============================================================================
def cache_dir_for(mode=NORM_MODE):
    """One cache per normalization: the runs are compared on the same frames."""
    return CACHE_ROOT / f"radioml_cache_{mode}"


def stratified_split(labels, snrs, seed=SEED, fractions=SPLIT_FRACTIONS):
    """Training/validation/test indices, 80/10/10 inside EACH
    (modulation, SNR) pair. The returned indices are sorted (sequential
    read from the cache)."""
    rng = np.random.default_rng(seed)
    cell_id = labels.astype(np.int64) * 1000 + (snrs.astype(np.int64) + 100)   # unique key per cell
    split_idx = {"train": [], "val": [], "test": []}
    for cell in np.unique(cell_id):
        members = np.flatnonzero(cell_id == cell)          # frames of this cell
        rng.shuffle(members)
        n_train = int(round(fractions[0] * len(members)))
        n_val = int(round(fractions[1] * len(members)))
        split_idx["train"].append(members[:n_train])
        split_idx["val"].append(members[n_train:n_train + n_val])
        split_idx["test"].append(members[n_train + n_val:])
    return {name: np.sort(np.concatenate(parts)) for name, parts in split_idx.items()}


def calibrate_fixed_scale(h5_x, train_idx, n_frames=FIXED_CAL_FRAMES,
                          pct=FIXED_PCT, target=FIXED_TARGET):
    """ONE SCALE measured ONLY on the training set: the pct percentile of
    |I|,|Q| is brought to `target`. Contiguous blocks scattered in the file are
    read (random access to single frames of the HDF5 is very slow)."""
    rng = np.random.default_rng(SEED)
    train_mask_lookup = np.zeros(h5_x.shape[0], dtype=bool)
    train_mask_lookup[train_idx] = True
    block_len = 1024                                         # frames per block read
    n_blocks_in_file = max(1, h5_x.shape[0] // block_len)   # whole blocks available
    n_blocks = min(max(1, n_frames // block_len), n_blocks_in_file)
    block_starts = np.sort(rng.choice(n_blocks_in_file, size=n_blocks,
                                      replace=False)) * block_len
    magnitudes = []
    for start in block_starts:
        block = h5_x[start:start + block_len]                # (block_len, 1024, 2) float32
        keep = train_mask_lookup[start:start + block_len]    # only training frames
        magnitudes.append(np.abs(block[keep]).ravel())
    magnitudes = np.concatenate(magnitudes)
    p_value = float(np.percentile(magnitudes, pct))
    rms_value = float(np.sqrt((magnitudes.astype(np.float64) ** 2).mean()))
    scale = target / p_value
    print(f"  fixed scale: p{pct} of |I|,|Q| = {p_value:.4f}, rms = {rms_value:.4f} "
          f"-> SCALE {scale:.4f}  (Q5.11 full scale = +-{16 / scale:.3f} file units)")
    return scale


def normalize(frames, mode=NORM_MODE, scale=None):
    """(N, 1024, 2) float32 of the file -> real values before the Q5.11."""
    if mode == "fixed":
        return frames * scale
    if mode == "rms":
        # RMS value per frame over I and Q together (mean complex power / 2)
        frame_rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=(1, 2), keepdims=True))
        return frames * (RMS_TARGET / np.maximum(frame_rms, 1e-12))
    raise ValueError(f"unknown NORM_MODE: {mode}")


def quantize_q511(values):
    """Real values -> int16 Q5.11 codes with saturation, plus the number of
    saturated samples. It is the input format of the encoder."""
    rounded = np.round(values * (1 << F_IN))
    n_saturated = int((np.abs(rounded) > 32767).sum())
    return np.clip(rounded, -32768, 32767).astype(np.int16), n_saturated


def build_cache(mode=NORM_MODE):
    """Converts the HDF5 (21 GB float32) into an int16 Q5.11 cache (10.5 GB),
    then read through a memory map. It creates, in data/radioml_cache_<mode>/:
        x.npy        (N, 1024, 2) int16   Q5.11 codes (what the encoder receives)
        y.npy        (N,)         int8    modulation index
        snr.npy      (N,)         int8    SNR in dB
        {train,val,test}_idx.npy          indices of the stratified split
        meta.json                         scale, saturations, checks
    If the cache already exists the HDF5 is not read again."""
    cache = cache_dir_for(mode)
    meta_path = cache / "meta.json"
    if meta_path.is_file():
        return cache, json.loads(meta_path.read_text())

    import h5py
    if not H5_PATH.is_file():
        sys.exit(f"{H5_PATH} not found: download RadioML 2018.01A "
                 f"(GOLD_XYZ_OSC.0001_1024.hdf5) and put it there")
    cache.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_PATH, "r") as h5:
        h5_x, h5_y, h5_z = h5["X"], h5["Y"], h5["Z"]
        n_frames = h5_x.shape[0]
        print(f"  HDF5: X {h5_x.shape} {h5_x.dtype}, Y {h5_y.shape}, Z {h5_z.shape}")
        assert h5_x.shape[1:] == (L, NIN), f"unexpected shape of X: {h5_x.shape}"
        if n_frames != N_FRAMES_EXPECTED:
            print(f"  WARNING: {n_frames} frames instead of {N_FRAMES_EXPECTED}")

        # labels and SNR, read in blocks (one-hot int64 Y takes ~490 MB)
        labels = np.empty(n_frames, dtype=np.int8)
        snrs = np.empty(n_frames, dtype=np.int8)
        for start in range(0, n_frames, 262144):
            stop = min(start + 262144, n_frames)
            labels[start:stop] = np.asarray(h5_y[start:stop]).argmax(axis=1)
            snrs[start:stop] = np.asarray(h5_z[start:stop]).reshape(-1)
        snr_values, snr_counts = np.unique(snrs, return_counts=True)
        print(f"  SNR: {snr_values.min()}..{snr_values.max()} dB, {len(snr_values)} values; "
              f"frames per class: {np.bincount(labels, minlength=len(CLASSES)).min()}"
              f"..{np.bincount(labels, minlength=len(CLASSES)).max()}")

        split = stratified_split(labels, snrs)
        meta = dict(norm_mode=mode, n_frames=int(n_frames),
                    snr_values=snr_values.tolist(), split_sizes={k: int(len(v)) for k, v in split.items()})
        if mode == "fixed":
            meta["scale"] = calibrate_fixed_scale(h5_x, split["train"])

        # conversion in sequential blocks: contiguous HDF5 read, write to a memmap
        x_out = np.lib.format.open_memmap(cache / "x.npy", mode="w+",
                                          dtype=np.int16, shape=(n_frames, L, NIN))
        total_saturated = 0
        t_start = time.time()
        for start in range(0, n_frames, H5_CHUNK):
            stop = min(start + H5_CHUNK, n_frames)
            frames = np.asarray(h5_x[start:stop], dtype=np.float32)
            codes, n_sat = quantize_q511(normalize(frames, mode, meta.get("scale")))
            x_out[start:stop] = codes
            total_saturated += n_sat
            if (start // H5_CHUNK) % 20 == 0:
                done = stop / n_frames
                print(f"    cache {100 * done:5.1f}%  ({time.time() - t_start:.0f}s)", flush=True)
        x_out.flush()
        del x_out

    np.save(cache / "y.npy", labels)
    np.save(cache / "snr.npy", snrs)
    for name, idx in split.items():
        np.save(cache / f"{name}_idx.npy", idx.astype(np.int64))
    meta["saturated_samples"] = total_saturated
    meta["saturated_frac"] = total_saturated / (n_frames * L * NIN)
    print(f"  samples saturated in Q5.11: {total_saturated} "
          f"({100 * meta['saturated_frac']:.5f}%)")
    meta_path.write_text(json.dumps(meta, indent=2))    # written last: cache complete
    return cache, meta


class RadioMLDataset(torch.utils.data.Dataset):
    """(1024, 2) float = Q5.11 code / 2048: the value the encoder sees.
    Returns (x, modulation index, SNR in dB).

    The memmap is opened lazily in the process that reads (DataLoader
    worker): passing an already open memmap to the workers would copy it."""

    def __init__(self, cache, split_name, rot90_augment=False):
        self.x_path = cache / "x.npy"
        self.frame_idx = np.load(cache / f"{split_name}_idx.npy")
        self.labels = np.load(cache / "y.npy")[self.frame_idx].astype(np.int64)
        self.snrs = np.load(cache / "snr.npy")[self.frame_idx].astype(np.int64)
        self.rot90_augment = rot90_augment
        self.x_codes = None                                   # memmap, opened at the first access

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, i):
        if self.x_codes is None:
            self.x_codes = np.load(self.x_path, mmap_mode="r")
        x = np.asarray(self.x_codes[self.frame_idx[i]], dtype=np.float32) / (1 << F_IN)
        if self.rot90_augment:
            # rotation by k*90 degrees: (I,Q) -> (-Q,I) repeated k times; exact
            # also in Q5.11 (only swaps and sign changes; -(-32768) saturates to
            # 32767/2048 in the corner case, irrelevant)
            for _ in range(np.random.randint(4)):
                x = np.stack((-x[:, 1], x[:, 0]), axis=1)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.labels[i]), int(self.snrs[i])


def stratified_subset(dataset, n_target, seed=SEED):
    """Fixed subset with the same number of frames per (class, SNR)."""
    if n_target is None or n_target >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed + 1)
    cell_id = dataset.labels * 1000 + (dataset.snrs + 100)
    cells = np.unique(cell_id)
    per_cell = max(1, n_target // len(cells))
    chosen = [rng.choice(np.flatnonzero(cell_id == c), size=per_cell, replace=False)
              for c in cells]
    return torch.utils.data.Subset(dataset, np.sort(np.concatenate(chosen)).tolist())


def build_loaders():
    cache, meta = build_cache()
    train_set = RadioMLDataset(cache, "train", rot90_augment=AUG_ROT90)
    val_set = RadioMLDataset(cache, "val")
    test_set = RadioMLDataset(cache, "test")
    if SUBSET:                                               # quick test
        train_set = stratified_subset(train_set, SUBSET)
        val_set = stratified_subset(val_set, max(SUBSET // 8, 26 * 24))
        test_set = stratified_subset(test_set, max(SUBSET // 8, 26 * 24))
    val_select_set = stratified_subset(val_set, VAL_FRAMES_PER_EPOCH)

    n_train = len(train_set)
    frames_per_epoch = min(FRAMES_PER_EPOCH or n_train, n_train)
    # random sampling WITHOUT repetition inside the epoch, epoch shorter than the dataset
    train_sampler = torch.utils.data.RandomSampler(train_set, replacement=False,
                                                   num_samples=frames_per_epoch)

    def make_loader(dataset, sampler=None):
        return torch.utils.data.DataLoader(
            dataset, batch_size=BATCH, sampler=sampler, shuffle=False,
            drop_last=sampler is not None, num_workers=WORKERS, pin_memory=True,
            persistent_workers=WORKERS > 0)

    loaders = dict(train=make_loader(train_set, train_sampler),
                   val_select=make_loader(val_select_set),
                   val=make_loader(val_set), test=make_loader(test_set))
    print(f"  frames: training {n_train} ({frames_per_epoch} per epoch), "
          f"validation {len(val_set)} (selection on {len(val_select_set)}), "
          f"test {len(test_set)}")
    return loaders, meta


# ============================================================================
# scheduler
# ============================================================================
def cosine_with_warmup(opt, warmup_steps, total_steps, min_ratio=0.0):
    def lr_factor(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)


# ============================================================================
# metrics
# ============================================================================
def evaluate(model, loader, device):
    """Global accuracy, per SNR, at SNR >= 0 / >= 10 dB, and confusion matrix
    at SNR >= 10 dB. The prediction is the argmax of the logits, as on the
    board."""
    model.eval()
    n_classes = len(CLASSES)
    predictions, labels, snrs = [], [], []
    with torch.no_grad():
        for x, y, snr in loader:
            predictions.append(model(x.to(device, non_blocking=True)).argmax(-1).cpu().numpy())
            labels.append(y.numpy()); snrs.append(snr.numpy())
    predictions, labels, snrs = map(np.concatenate, (predictions, labels, snrs))
    correct = predictions == labels
    acc_per_snr = {int(s): float(correct[snrs == s].mean()) for s in np.unique(snrs)}
    high_snr = snrs >= 10
    confusion_high = np.zeros((n_classes, n_classes), dtype=np.int64)   # rows true, columns predicted
    np.add.at(confusion_high, (labels[high_snr], predictions[high_snr]), 1)
    recall_high = np.diag(confusion_high) / np.maximum(confusion_high.sum(1), 1)
    return dict(acc=float(correct.mean()),
                acc_snr_ge0=float(correct[snrs >= 0].mean()) if (snrs >= 0).any() else float("nan"),
                acc_snr_ge10=float(correct[high_snr].mean()) if high_snr.any() else float("nan"),
                acc_per_snr=acc_per_snr,
                recall_snr_ge10={c: float(r) for c, r in zip(CLASSES, recall_high)},
                cm_snr_ge10=confusion_high.tolist(), n_frames=int(len(labels)))


def print_summary(model):
    print("\n===== ARCHITECTURE =====")
    print(model)
    print("\n===== PARAMETERS =====")
    total = 0
    for name, p in model.named_parameters():
        total += p.numel()
        print(f"{name:52s} {str(tuple(p.shape)):20s} {p.numel():>9,}")
    print(f"\nTOTAL PARAMETERS: {total:,}")
    print("=" * 60 + "\n")
    return total


def param_groups(model):
    """The S4D kernel out of the weight decay and with its own LR, as for ECG and
    Speech Commands."""
    ssm_keys = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
    ssm_params, no_decay_params, decay_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ssm_keys):
            ssm_params.append(p)
        elif name.endswith(".bias") or ".bn." in name or name.endswith(".D"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    assert len(ssm_params) == 3 * N_LAYERS, f"SSM filter: {len(ssm_params)}, {3*N_LAYERS} expected"
    assert no_decay_params, "empty no_decay filter: BatchNormSeq names changed?"
    return decay_params, no_decay_params, ssm_params


# ============================================================================
# training
# ============================================================================
def train(model, loaders, dev, exp_dir):
    decay_params, no_decay_params, ssm_params = param_groups(model)
    print(f"parameters: {len(decay_params)} with WD, {len(no_decay_params)} without, "
          f"{len(ssm_params)} SSM")

    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WD},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": ssm_params, "weight_decay": 0.0, "lr": LR_SSM},
    ], lr=LR)
    steps_per_epoch = max(len(loaders["train"]), 1)
    sched = cosine_with_warmup(opt, WARMUP_EPOCHS * steps_per_epoch, EPOCHS * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()
    log_path = exp_dir / "train_log.jsonl"

    best_acc, best_state, epochs_without_gain = -1.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(EPOCHS):
        model.train(); loss_sum = 0.0; t_epoch = time.time(); nan_hit = False
        for x, y, _snr in loaders["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = criterion(model(x), y)
            # always float32: in half precision the S4D kernel goes to nan (see Speech Commands)
            if not torch.isfinite(loss):
                nan_hit = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
        if nan_hit:
            print(f"non-finite loss at epoch {epoch+1}: stopped (lower LR). "
                  f"The best model stays in {exp_dir/'best.pt'}")
            break

        metrics = evaluate(model, loaders["val_select"], dev)
        seconds = time.time() - t_epoch
        avg_loss = loss_sum / steps_per_epoch
        print(f"epoch {epoch+1}/{EPOCHS}  loss {avg_loss:.4f}  val acc {metrics['acc']:.4f}  "
              f"SNR>=0 {metrics['acc_snr_ge0']:.4f}  SNR>=10 {metrics['acc_snr_ge10']:.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({seconds:.0f}s)", flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(epoch=epoch + 1, loss=avg_loss, val_acc=metrics["acc"],
                                    val_acc_snr_ge0=metrics["acc_snr_ge0"],
                                    val_acc_snr_ge10=metrics["acc_snr_ge10"],
                                    lr=opt.param_groups[0]["lr"], sec=round(seconds, 1))) + "\n")

        if metrics["acc"] > best_acc:                        # selection on the global accuracy
            best_acc, best_state, epochs_without_gain = metrics["acc"], copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, exp_dir / "best.pt")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"early stopping (best val acc {best_acc:.4f})")
                break
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=epoch, best=best_acc,
                        patience=epochs_without_gain), exp_dir / "last.pt")

    model.load_state_dict(best_state)
    return model, best_acc


def print_snr_curve(name, metrics):
    print(f"{name}: acc {metrics['acc']:.4f}  SNR>=0 {metrics['acc_snr_ge0']:.4f}  "
          f"SNR>=10 {metrics['acc_snr_ge10']:.4f}  ({metrics['n_frames']} frames)")
    print("  acc per SNR: " + "  ".join(f"{s:+d}:{a:.3f}" for s, a in metrics["acc_per_snr"].items()))


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = load_modules()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    print("device:", dev)

    if D_MODEL != G.H:
        print(f"WARNING: D_MODEL={D_MODEL} but the RTL has H={G.H}: not exportable.")
    if D_STATE // 2 != G.MODES:
        print(f"WARNING: D_STATE/2={D_STATE//2} but the RTL has MODES={G.MODES}.")

    loaders, meta = build_loaders()
    print(f"normalization: {NORM_MODE}" +
          (f"  SCALE {meta['scale']:.4f}" if "scale" in meta else f"  RMS_TARGET {RMS_TARGET}"))

    model = radioml_model(B, len(CLASSES)).to(dev)
    n_params = print_summary(model) if PRINT_SUMMARY else None
    if SUMMARY_ONLY:
        print("SUMMARY_ONLY active: no training.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_name = f"exp_{stamp}_radioml_l{L}_{NORM_MODE}" + ("_rot90" if AUG_ROT90 else "")
    exp_dir = EXP_ROOT / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)              # never overwrite a run
    shutil.copy2(__file__, exp_dir / Path(__file__).name)
    print("output:", exp_dir)

    model, best_val = train(model, loaders, dev, exp_dir)
    val = evaluate(model, loaders["val"], dev)               # full validation
    test = evaluate(model, loaders["test"], dev)
    print()
    print_snr_curve("validation", val)
    print_snr_curve("test", test)
    print("recall per class at SNR>=10 (test):",
          "  ".join(f"{c} {r:.3f}" for c, r in test["recall_snr_ge10"].items()))

    (exp_dir / "report.json").write_text(json.dumps(dict(
        task="radioml2018_amc24", classes=CLASSES, input=f"I/Q {NORM_MODE} Q5.11",
        hw_compatible=HW_COMPATIBLE, norm_mode=NORM_MODE, scale=meta.get("scale"),
        fixed_target=FIXED_TARGET, fixed_pct=FIXED_PCT, rms_target=RMS_TARGET,
        saturated_frac=meta.get("saturated_frac"), seq_len=L, n_in=NIN, f_in=F_IN,
        split_sizes=meta.get("split_sizes"), val=val, test=test, best_val_acc=best_val,
        n_params=n_params, seed=SEED, d_model=D_MODEL, n_layers=N_LAYERS,
        d_state=D_STATE, dropout=DROPOUT, lr=LR, lr_ssm=LR_SSM, wd=WD, batch=BATCH,
        epochs=EPOCHS, frames_per_epoch=FRAMES_PER_EPOCH, warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE, aug_rot90=AUG_ROT90, allow_tf32=ALLOW_TF32,
        subset=SUBSET), indent=2))
    print("report:", exp_dir / "report.json")

    print("\n--- notes for the export to the accelerator ---")
    print("1. generic encoder: NIN = 2, N_BIT_IN = 16, IN_SIGNED = 1 (I and Q in Q5.11);")
    print("   the quantizer must export W (128x2) and b, as for the ECG (128x12).")
    print("2. decoder: K = 24.")
    print(f"3. Board stimuli: same normalization ({NORM_MODE}"
          + (f", SCALE {meta['scale']:.4f}" if "scale" in meta else "") + ") and quantize_q511().")


if __name__ == "__main__":
    main()
