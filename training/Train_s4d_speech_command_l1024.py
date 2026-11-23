#!/usr/bin/env python3
"""
Train_s4d_speech_command_l1024.py -- S4D on Speech Commands with L = 1024.

It runs without arguments, from the experiment folder or from any other:

    conda run -n mamba_env --no-capture-output python Train_s4d_speech_command_l1024.py

IDEA
The August run (Train_s4d_speech_command_8bit.py) decimated 16 kHz by 4 WITHOUT
a filter and reached L = 4096: it does not fit the core, which has L = 1024. Here
the sequence is 1024 steps long and each step carries 12 consecutive samples:

    int16 16 kHz
      -> polyphase anti-aliasing FIR 3/4 (pass 5 kHz, stop 6 kHz, 80 dB)
      -> 12 kHz, 12000 samples
      -> fixed-scale Q5.11 code (INPUT_SCALE)
      -> 1000 frames of 12 samples + 24 zero frames = 1024 steps x 12 inputs
      -> encoder Linear(12 -> 128)   [same scheme as the 12-lead ECG encoder]

The S4D core (H = 128, 6 layers, 32 modes) does NOT change: it is literally
B.torch_model, in which only input_proj is replaced as Train_s4d_ecg.py does.

WHAT STAYS THE SAME AS THE AUGUST RUN (to be able to compare)
optimizer, lr, kernel lr, weight decay, batch, epochs, dropout, seed,
official splits, 35 classes.

WHAT CHANGES
1. frequency 4 kHz -> 12 kHz with anti-aliasing filter
2. L 4096 -> 1024, input 1 -> 12 values per step
3. input uint8 -> fixed-scale int16 Q5.11
4. augmentation on the training set: random time shift and gain
   (AUG_* below; set to 0 to turn them off)

OUTPUTS (in the folder of this script)
    ckpt/sc12k_best.pt, ckpt/sc12k_last.pt
    train_log.jsonl
    train_report.json
The 12 kHz cache goes to data/sc_cache_12k/ (a NEW folder; data/sc_cache is
not modified) and is built only once.

QUICK TEST
    SC12K_SMOKE=1 python Train_s4d_speech_command_l1024.py
uses 2048 examples for one epoch and writes to smoke/, without touching the real outputs.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal

# ============================================================================
# CONFIGURATION -- edited here, not from the command line
# ============================================================================
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR                                 # training/ (all the scripts are in the same folder)
MONO = PROJECT_ROOT / "Train_and_quantize_s4d_sCIFAR.py"      # from which gen_s4d_mem/s4d_export/s4d_bn are extracted
CACHE_16K_DIR = PROJECT_ROOT / "data" / "sc_cache"            # int16 16 kHz, READ ONLY
CACHE_12K_DIR = PROJECT_ROOT / "data" / "sc_cache_12k"        # filtered int16 12 kHz, created here

# --- input chain -------------------------------------------------------------
FS_IN = 16000
RESAMPLE_UP = 3               # 16 kHz * 3 / 4 = 12 kHz
RESAMPLE_DOWN = 4
FS_OUT = FS_IN * RESAMPLE_UP // RESAMPLE_DOWN                 # 12000
FS_POLY = FS_IN * RESAMPLE_UP                                 # 48000: internal frequency of the polyphase
AA_PASS_HZ = 5000.0           # the filter is flat up to here
AA_STOP_HZ = 6000.0           # new Nyquist: from here up 80 dB of attenuation
AA_ATTEN_DB = 80.0

FRAME_LEN = 12                # samples per step = encoder inputs
N_FRAMES_AUDIO = FS_OUT // FRAME_LEN                          # 1000 audio steps
SEQ_LEN = 1024                # L of the core: the last 24 steps are zeros

F_IN = 11                     # fractional bits of the input: Q5.11 as W_ACT/F_ACT
W_IN = 16                     # width of the signed input
# Fixed scale: real value = (int16 / 32768) * INPUT_SCALE.
# With 8 the Q5.11 code is int16/2 (a shift), the rms of the dataset is ~0.68 and
# the peak measured after the filter (1.37) reaches 11, below the ceiling of 16.
INPUT_SCALE = 8.0

# --- model: FIXED, it is the contract with the RTL ---------------------------
D_MODEL = 128                 # H
N_LAYERS = 6                  # NLAYER
D_STATE = 64                  # 2 * MODES -> 32 modes
DROPOUT = 0.1

# --- optimization: identical to the August run --------------------------------
LR = 1e-2
LR_SSM = 1e-3                 # S4D kernel group, weight decay 0
WD = 0.05
BATCH = 32
EPOCHS = 40
WARMUP_EPOCHS = 2
PATIENCE = 10
WORKERS = 8
SEED = 1337

# --- augmentation (training only) ---------------------------------------------
AUG_SHIFT_MS = 100.0          # random time shift +-100 ms; 0 = off
AUG_GAIN_DB = 6.0             # random gain +-6 dB; 0 = off

# --- quick test ----------------------------------------------------------------
SMOKE = os.environ.get("SC12K_SMOKE", "0") == "1"
SUBSET = 2048 if SMOKE else None          # number of training examples (None = all)
if SMOKE:
    EPOCHS, WARMUP_EPOCHS = 1, 0
OUT_DIR = EXPERIMENT_DIR / "smoke" if SMOKE else EXPERIMENT_DIR

# The functions of the August run are reused instead of copied: loading of the
# RTL contract modules, scheduler, parameter groups, evaluation.
sys.path.insert(0, str(PROJECT_ROOT))
from Train_s4d_speech_command_8bit import (   # noqa: E402
    load_modules, cosine_with_warmup, param_groups, evaluate)


# ============================================================================
# anti-aliasing filter and 12 kHz cache
# ============================================================================
def design_antialias_fir():
    """Kaiser-window low-pass FIR, designed at 48 kHz.

    Unity DC gain: the RESAMPLE_UP factor that compensates the zero insertion
    is already applied by scipy.signal.resample_poly.
    It is the same design used by sc_data_report.py (same three parameters).
    """
    transition_norm = (AA_STOP_HZ - AA_PASS_HZ) / (FS_POLY / 2)   # transition normalized to Nyquist
    num_taps, kaiser_beta = signal.kaiserord(AA_ATTEN_DB, transition_norm)
    num_taps |= 1                                                 # odd: linear phase, integer delay
    cutoff_hz = 0.5 * (AA_PASS_HZ + AA_STOP_HZ)                   # -6 dB at mid transition
    return signal.firwin(num_taps, cutoff_hz, window=("kaiser", kaiser_beta), fs=FS_POLY)


def build_cache_12k():
    """16 kHz int16 -> filtered 12 kHz int16, only once.

    The int16 is kept and not the Q5.11 code: the scale and the augmentations
    are applied on the fly, so changing INPUT_SCALE does not require rebuilding
    the cache.
    """
    CACHE_12K_DIR.mkdir(parents=True, exist_ok=True)
    aa_taps = design_antialias_fir()
    meta = dict(fs_in=FS_IN, fs_out=FS_OUT, up=RESAMPLE_UP, down=RESAMPLE_DOWN,
                aa_pass_hz=AA_PASS_HZ, aa_stop_hz=AA_STOP_HZ, aa_atten_db=AA_ATTEN_DB,
                aa_num_taps=int(len(aa_taps)))
    meta_path = CACHE_12K_DIR / "meta.json"
    if meta_path.is_file():
        # a cache made with another filter must NOT be reused silently
        assert json.loads(meta_path.read_text()) == meta, \
            f"{meta_path} describes a different filter: delete {CACHE_12K_DIR} or change folder"

    chunk_size = 4096                                             # recordings per block: ~250 MB of float
    for split in ("training", "validation", "testing"):
        out_path = CACHE_12K_DIR / f"sc_{split}_x12k.npy"
        if out_path.is_file():
            continue
        src = np.load(CACHE_16K_DIR / f"sc_{split}_x.npy", mmap_mode="r")   # (N, 16000) int16
        n_utts = src.shape[0]
        tmp_path = out_path.with_suffix(".tmp.npy")               # written to a temporary file
        dst = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.int16, shape=(n_utts, FS_OUT))
        n_clipped = 0
        t_start = time.time()
        for first in range(0, n_utts, chunk_size):
            block = np.asarray(src[first:first + chunk_size], dtype=np.float32)
            resampled = signal.resample_poly(block, RESAMPLE_UP, RESAMPLE_DOWN, axis=-1,
                                             window=aa_taps)[:, :FS_OUT]
            rounded = np.round(resampled)
            # the filter can exceed the full scale on files already clipped (overshoot)
            n_clipped += int((np.abs(rounded) > 32767).sum())
            dst[first:first + chunk_size] = np.clip(rounded, -32768, 32767).astype(np.int16)
        dst.flush(); del dst
        tmp_path.rename(out_path)                                 # renamed only when complete
        print(f"  12 kHz cache {split}: {n_utts} recordings, "
              f"{n_clipped} samples clipped to int16 ({time.time() - t_start:.0f}s)")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ============================================================================
# data
# ============================================================================
class SC12kFrameDataset(torch.utils.data.Dataset):
    """One recording -> (1024, 12) tensor of Q5.11 values and the label.

    ORDER OF THE OPERATIONS (it is the hardware chain):
        [augmentation] -> fixed scale -> Q5.11 quantization with saturation
        -> frames of 12 -> zero padding up to 1024 steps
    The model sees the ALREADY quantized value (code / 2^F_IN), i.e. exactly
    what the encoder will receive on its port.
    """

    def __init__(self, x_path, y_path, augment):
        self.waves = np.load(x_path, mmap_mode="r")               # (N, 12000) int16 at 12 kHz
        self.labels = np.load(y_path)
        self.augment = augment
        self.max_shift = int(round(AUG_SHIFT_MS * 1e-3 * FS_OUT))  # 100 ms = 1200 samples
        self.code_scale = INPUT_SCALE * (1 << F_IN) / 32768.0     # int16 -> Q5.11 code (0.5 with scale 8)
        self.code_max = (1 << (W_IN - 1)) - 1                     # +32767
        self.code_min = -(1 << (W_IN - 1))                        # -32768

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        wave = np.asarray(self.waves[index], dtype=np.float32)    # (12000,)
        if self.augment:
            if self.max_shift > 0:
                # Shift NOT a multiple of 12: it also changes the phase of the frame,
                # which is what the stride-12 encoder is not invariant to.
                shift = np.random.randint(-self.max_shift, self.max_shift + 1)
                shifted = np.zeros_like(wave)
                if shift > 0:
                    shifted[shift:] = wave[:-shift]               # delay: zeros at the start
                elif shift < 0:
                    shifted[:shift] = wave[-shift:]               # advance: zeros at the end
                else:
                    shifted = wave
                wave = shifted
            if AUG_GAIN_DB > 0:
                gain_db = np.random.uniform(-AUG_GAIN_DB, AUG_GAIN_DB)
                wave = wave * np.float32(10.0 ** (gain_db / 20.0))   # fixed scale -> robustness to the level is needed
        codes = np.clip(np.round(wave * self.code_scale), self.code_min, self.code_max)   # saturated Q5.11
        values = (codes / float(1 << F_IN)).astype(np.float32)    # real value seen by the encoder
        frames = np.zeros((SEQ_LEN, FRAME_LEN), dtype=np.float32) # the last 24 steps stay exactly zero
        frames[:N_FRAMES_AUDIO] = values.reshape(N_FRAMES_AUDIO, FRAME_LEN)
        return torch.from_numpy(frames), int(self.labels[index])


def seed_worker(worker_id):
    """Different seed for each worker: without it, all draw the same augmentations."""
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def build_loaders():
    def make_loader(split, is_train):
        dataset = SC12kFrameDataset(CACHE_12K_DIR / f"sc_{split}_x12k.npy",
                                    CACHE_16K_DIR / f"sc_{split}_y.npy",   # the labels are those of the original cache
                                    augment=is_train)
        if SUBSET:
            n_keep = min(SUBSET if is_train else max(SUBSET // 4, 1), len(dataset))
            keep = np.random.default_rng(SEED).choice(len(dataset), n_keep, replace=False)
            dataset = torch.utils.data.Subset(dataset, keep.tolist())
        return torch.utils.data.DataLoader(
            dataset, batch_size=BATCH, shuffle=is_train, drop_last=is_train,
            num_workers=WORKERS, pin_memory=True, worker_init_fn=seed_worker,
            persistent_workers=True)

    return make_loader("training", True), make_loader("validation", False), make_loader("testing", False)


# ============================================================================
# model
# ============================================================================
def build_model(B, n_classes):
    """The BatchNorm model of s4d_bn with the FRAME_LEN-input encoder.

    ONLY input_proj is replaced (as build_model of Train_s4d_ecg.py):
    S4D kernel, ZOH, norm, GLU and classifier stay the object that the
    quantization knows how to export.
    """
    model = B.torch_model(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                          dropout=DROPOUT, n_classes=n_classes)
    model.input_proj = nn.Linear(FRAME_LEN, D_MODEL)              # 12 samples -> 128 channels
    return model


# ============================================================================
# training
# ============================================================================
def train(model, train_loader, val_loader, device, ckpt_dir, log_path, after_step=None):
    """after_step: optional function called after each optimizer step
    (used by the fine-tuning to project the weights back into the hardware formats)."""
    decay_params, no_decay_params, ssm_params = param_groups(model)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WD},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": ssm_params, "weight_decay": 0.0, "lr": LR_SSM},   # poles and dt: lower lr, no decay
    ], lr=LR)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = cosine_with_warmup(optimizer, WARMUP_EPOCHS * steps_per_epoch, EPOCHS * steps_per_epoch)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, epochs_without_gain = 0.0, copy.deepcopy(model.state_dict()), 0
    for epoch in range(EPOCHS):
        model.train()
        loss_sum, n_correct, n_seen = 0.0, 0, 0
        t_epoch = time.time()
        for frames, target in train_loader:
            frames = frames.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(frames)                                # float32: in half precision the S4D kernel gives nan
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                # it is the forward that overflows, not a gradient: no point going on
                print(f"non-finite loss at epoch {epoch + 1}: stopped, lower LR. "
                      f"The best model stays in {ckpt_dir / 'sc12k_best.pt'}")
                model.load_state_dict(best_state)
                return model, best_val_acc
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if after_step is not None:
                after_step(model)                               # e.g. projection of the weights into the fixed formats
            scheduler.step()
            loss_sum += loss.item()
            n_correct += (logits.argmax(-1) == target).sum().item()
            n_seen += target.numel()

        val_acc = evaluate(model, val_loader, device)
        epoch_sec = time.time() - t_epoch
        train_loss = loss_sum / steps_per_epoch
        train_acc = n_correct / max(n_seen, 1)                    # with augmentation and dropout active
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {train_loss:.4f}  train {train_acc:.4f}  "
              f"val {val_acc:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  ({epoch_sec:.0f}s)", flush=True)
        with open(log_path, "a") as log_file:
            log_file.write(json.dumps(dict(epoch=epoch + 1, loss=train_loss, train_acc=train_acc,
                                           val_acc=val_acc, lr=optimizer.param_groups[0]["lr"],
                                           sec=round(epoch_sec, 1))) + "\n")

        if val_acc > best_val_acc:
            best_val_acc, best_state, epochs_without_gain = val_acc, copy.deepcopy(model.state_dict()), 0
            torch.save(best_state, ckpt_dir / "sc12k_best.pt")
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"early stopping (best val {best_val_acc:.4f})")
                break
        torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(),
                        sched=scheduler.state_dict(), epoch=epoch, best=best_val_acc,
                        patience=epochs_without_gain), ckpt_dir / "sc12k_last.pt")

    model.load_state_dict(best_state)
    return model, best_val_acc


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = load_modules(str(MONO))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("device:", device, "  SMOKE" if SMOKE else "")

    # the model must stay the RTL one: better to stop than to find out at export time
    assert D_MODEL == G.H, f"D_MODEL={D_MODEL} but the RTL has H={G.H}"
    assert D_STATE // 2 == G.MODES, f"D_STATE/2={D_STATE // 2} but the RTL has MODES={G.MODES}"
    assert N_FRAMES_AUDIO * FRAME_LEN == FS_OUT and N_FRAMES_AUDIO <= SEQ_LEN

    ckpt_dir = OUT_DIR / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "train_log.jsonl"
    report_path = OUT_DIR / "train_report.json"
    if not SMOKE:
        # never overwrite a result: a second run goes to another folder
        assert not log_path.exists() and not report_path.exists(), \
            f"{OUT_DIR} already contains a run: move it or use a new experiment folder"
    elif log_path.exists():
        log_path.unlink()                                         # the quick test can be redone

    print("12 kHz cache:")
    cache_meta = build_cache_12k()
    class_names = json.loads((CACHE_16K_DIR / "sc_labels.json").read_text())
    train_loader, val_loader, test_loader = build_loaders()
    print(f"train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  "
          f"test {len(test_loader.dataset)}  classes {len(class_names)}")
    print(f"input: {SEQ_LEN} steps x {FRAME_LEN} samples, Q{W_IN - F_IN}.{F_IN}, scale {INPUT_SCALE}")

    model = build_model(B, len(class_names)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    model, best_val_acc = train(model, train_loader, val_loader, device, ckpt_dir, log_path)
    test_acc = evaluate(model, test_loader, device)               # level 1: floating point
    print(f"\nFP32 test accuracy = {test_acc:.4f}   (best val {best_val_acc:.4f})")

    report_path.write_text(json.dumps(dict(
        task="speechcommands35_raw_frame12", level="floating point (FP32), NOT quantized",
        acc_test=test_acc, best_val=best_val_acc, n_classes=len(class_names), n_params=n_params,
        seq_len=SEQ_LEN, frame_len=FRAME_LEN, n_frames_audio=N_FRAMES_AUDIO,
        sample_rate_eff=FS_OUT, antialias=cache_meta,
        input_format=f"int{W_IN} Q{W_IN - F_IN}.{F_IN}", input_scale=INPUT_SCALE,
        aug_shift_ms=AUG_SHIFT_MS, aug_gain_db=AUG_GAIN_DB,
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE, modes=D_STATE // 2, dropout=DROPOUT,
        lr=LR, lr_ssm=LR_SSM, wd=WD, batch=BATCH, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE, seed=SEED, subset=SUBSET), indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
