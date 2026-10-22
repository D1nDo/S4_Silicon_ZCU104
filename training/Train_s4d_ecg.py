#!/usr/bin/env python3
"""
Train_s4d_ecg.py -- S4D training on PTB-XL (12-lead ECG),
5 diagnostic superclasses, for the S4D core with L = 1024.

It runs without arguments, from the root of the repository:

    conda run -n mamba_env python Train_s4d_ecg.py

All the knobs are constants in the CONFIGURATION block: as for
Train_s4d_speech_command_8bit.py, the experiment is documented by the file,
not by a command line.

WHY A NEW TRAINING
The previous ECG model (paper_S4D/ECG/mem/) was trained on L = 2500 steps
(500 Hz, stride 2, POOL_DIVISOR = 2500), but the core on the board has L = 1024
and LOG2_L = 10: Export_ECG_stimulus.py sends it only the first 1024 steps, i.e.
41% of the signal seen in training, and the mean pool divides by 1024 instead
of 2500. Here the training is done directly on the sequence the hardware sees.

DATA: 100 Hz, WHOLE 10 s
records100/ of PTB-XL: 1000 samples x 12 leads, already resampled by
PhysioNet. 1000 steps fit in L = 1024 with 24 trailing zeros, so the network
sees the whole record instead of 4.1 s. The padding comes AFTER the
normalization, so the tail is exactly zero; the mean pool also sums those 24
steps and divides by 1024, exactly as mean_pool.v (>>> LOG2_L).

Preprocessing, identical to that of Export_ECG_stimulus.py except for the
frequency (and for the board it must be the SAME as preprocess() here):
  1. adu int16 / 1000 -> mV
  2. z-score per record and per lead on the 1000 samples
  3. Q5.11: round(z * 2048), int16 saturation  (input format of
     encoder_ecg.v: signed, F_ACT = 11)
  4. zero padding up to 1024
The model receives q / 2048, i.e. exactly the fixed-point value: no MU/SIGMA
to fold into the encoder, the normalization is already in the data.

LABELS
Diagnostic superclass (CD, HYP, MI, NORM, STTC, alphabetical order as in
Export_ECG_stimulus.py), from all the SCP codes with diagnostic == 1 at any
likelihood, keeping only the records with ONE superclass. With this rule
fold 10 matches record by record and label by label
data_ptbxl/fold10_single.csv (checked: 1650 records). Split recommended
by the authors: folds 1-8 training, 9 validation, 10 test.

UNBALANCED CLASSES
NORM is ~55% of the test set, HYP ~3%. Besides the accuracy, balanced accuracy
and macro-F1 are recorded: a model that always says NORM gets 55% accuracy.
CLASS_WEIGHTING = "inv_sqrt" weights the cross-entropy with 1/sqrt(frequency), a
compromise: pure "inv" sacrifices global accuracy for the rare classes.
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

MONO = "Train_and_quantize_s4d_sCIFAR.py"   # from which the three modules are extracted
DATA = Path("data_ptbxl")                   # ptbxl_database.csv, records100/
CACHE_DIR = Path("data/ptbxl_cache")        # preprocessed int16 .npy
EXP_ROOT = Path("experiments")

# --- input normalization ----------------------------------------------------
# "zscore": (x - mean) / std per record and per lead (run A). It removes the
#           absolute amplitude and the amplitude ratios between leads.
# "fixed" : (x - median per lead) * SCALE, with ONE SCALE for the whole
#           dataset: amplitudes stay comparable between patients and between
#           leads (voltage criteria for HYP, Q waves for MI).
NORM_MODE = "fixed"
FIXED_TARGET = 4.0      # SCALE brings the FIXED_PCT percentile of |x| to this
FIXED_PCT = 99.9        # value (Q5.11 saturates at +-16: 4x margin)
FIXED_CAL_N = 2000      # training records used to measure SCALE

# --- data -------------------------------------------------------------------
CLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]
NLEAD = 12
# FS = 100: records100, 1000 samples + 24 zeros = L of the core (LOG2_L = 10).
# FS = 500: records500, 5000 samples, whole record at full resolution. It is an
#           UPPER BOUND in float: with L = 5000 the model does NOT run on the
#           current core (L = 1024 fixed) and must not be exported.
FS = 100
RAW_L = 10 * FS
L = 1024 if FS == 100 else RAW_L
REC_COL = "filename_lr" if FS == 100 else "filename_hr"
HW_COMPATIBLE = (L == 1024)
F_IN = 11           # Q5.11, F_ACT of the encoder
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLDS = (9,)
TEST_FOLDS = (10,)

# --- model: the defaults keep the compatibility with the RTL (H=128, MODES=32)
D_MODEL = 128
N_LAYERS = 6
D_STATE = 64
DROPOUT = 0.1

# --- optimization -----------------------------------------------------------
LR = 1e-2
LR_SSM = 1e-3       # S4D kernel group, weight decay 0
WD = 0.05
BATCH = 32
EPOCHS = 40
WARMUP_EPOCHS = 2
PATIENCE = 10
WORKERS = 4
SEED = 1337
# LABEL_MODE = "single": only records with ONE superclass, softmax + cross-entropy
#              (runs A, B). It optimizes the single-label accuracy.
# LABEL_MODE = "multi" : all the records with >= 1 superclass, multi-hot labels,
#              sigmoid per class + BCE: the setup of the PTB-XL benchmark, which
#              optimizes the macro-AUC. Nothing changes on the board: 5 logits, argmax.
LABEL_MODE = "multi"
CLASS_WEIGHTING = None          # only LABEL_MODE="single": None | "inv_sqrt" | "inv"
SELECT_ON = "macro_auc"         # validation metric for the best model:
                                # "macro_auc" | "acc" | "bal_acc"

# --- diagnostics ------------------------------------------------------------
PRINT_SUMMARY = True
SUMMARY_ONLY = False    # True: print the summary and exit without training
SUBSET = None           # int for a quick test, None for the whole dataset


# ============================================================================
# project modules
# ============================================================================
def load_modules(mono_path=MONO):
    """gen_s4d_mem / s4d_export / s4d_bn, from the separate files or from the monolith.

    Same extraction as Train_s4d_speech_command_8bit.py: the strings of the three
    modules are the contract with the RTL and must be executed verbatim.
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


def ecg_model(B, n_classes):
    """The BN model of s4d_bn with the NLEAD-input encoder.

    Only input_proj is replaced: S4D core, norm, GLU and classifier stay the
    object that the quantization knows how to export. encoder_ecg.v computes
    y[ch] = sum_c W[ch][c] * x[c] + b[ch], i.e. exactly a Linear(12, H).
    """
    m = B.torch_model(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                      dropout=DROPOUT, n_classes=n_classes)
    m.input_proj = nn.Linear(NLEAD, D_MODEL)
    return m


# ============================================================================
# data
# ============================================================================
def read_mv(path):
    """One records{FS} record -> (RAW_L, NLEAD) float64 in mV."""
    x = np.fromfile(path, dtype="<i2")
    assert x.size == RAW_L * NLEAD, f"{path}: {x.size} samples"
    return x.reshape(RAW_L, NLEAD).astype(np.float64) / 1000.0


def normalize(x, mode=NORM_MODE, scale=None):
    """(RAW_L, NLEAD) mV -> real values before the Q5.11 quantization."""
    if mode == "zscore":
        sd = x.std(0)
        return (x - x.mean(0)) / np.where(sd > 0, sd, 1.0)
    if mode == "fixed":
        # the median removes the baseline without being pulled by the QRS as the mean is
        return (x - np.median(x, axis=0)) * scale
    raise ValueError(f"unknown NORM_MODE: {mode}")


def preprocess(path, mode=NORM_MODE, scale=None):
    """One records{FS} record -> ((L, NLEAD) int16 Q5.11 padded to L,
    number of saturated samples).

    It is also the function to use to generate the board stimuli: the integer
    code returned is the one the encoder receives.
    """
    z = normalize(read_mv(path), mode, scale)
    r = np.round(z * (1 << F_IN))
    n_sat = int((np.abs(r) > 32767).sum())
    q = np.clip(r, -32768, 32767).astype(np.int16)
    out = np.zeros((L, NLEAD), dtype=np.int16)
    out[:RAW_L] = q
    return out, n_sat


def calibrate_scale(files, data=DATA, n=FIXED_CAL_N, pct=FIXED_PCT,
                    target=FIXED_TARGET):
    """One SCALE, measured ONLY on the training set: the pct percentile of
    |x - median| (over all the leads) goes to `target`."""
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    a = np.concatenate([np.abs(normalize(read_mv(data / f"{files[i]}.dat"),
                                         "fixed", 1.0)).ravel() for i in pick])
    p = float(np.percentile(a, pct))
    print(f"  fixed scale: p{pct} of |x - median| = {p:.3f} mV "
          f"-> SCALE {target / p:.4f} /mV  (Q5.11 full scale = "
          f"+-{16 * p / target:.1f} mV)")
    return target / p


def superclass_table(data=DATA, label_mode=LABEL_MODE):
    """ecg_id -> strat_fold, file, label (superclass if single, otherwise
    None) and one 0/1 column per class. "single" keeps the records with ONE
    superclass, "multi" those with at least one. Rule that reproduces
    fold10_single.csv (see the docstring)."""
    import pandas as pd

    db = pd.read_csv(data / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db.scp_codes.apply(ast.literal_eval)
    ag = pd.read_csv(data / "scp_statements.csv", index_col=0)
    ag = ag[ag.diagnostic == 1]
    sup = db.scp_codes.apply(
        lambda d: sorted({ag.loc[k].diagnostic_class for k in d if k in ag.index}))
    n = sup.apply(len)
    db = db[(n == 1) if label_mode == "single" else (n >= 1)].copy()
    db["label"] = sup[db.index].apply(lambda s: s[0] if len(s) == 1 else None)
    for c in CLASSES:
        db[c] = sup[db.index].apply(lambda s, c=c: int(c in s))

    ref_path = data / "fold10_single.csv"
    if ref_path.is_file():
        ref = pd.read_csv(ref_path, index_col="ecg_id")
        f10 = db[(db.strat_fold == 10) & db.label.notna()]
        assert set(f10.index) == set(ref.index) and \
            (f10.loc[ref.index, "label"] == ref.label).all(), \
            "fold 10 does not match fold10_single.csv: label rule changed?"
    return db[["strat_fold", REC_COL, "label"] + CLASSES].rename(columns={REC_COL: "file"})


def build_cache(data=DATA, mode=NORM_MODE):
    """Preprocesses the records once: {split}_x.npy (N, L, 12) int16.

    One cache per normalization (the z-score of run A stays in CACHE_DIR):
    the two runs are compared on the same records. meta.json keeps SCALE and
    the saturated samples, which are needed later by the quantizer and the
    board stimuli.
    """
    if FS == 100:       # names of run A / run B, unchanged
        cache_dir = CACHE_DIR if mode == "zscore" else \
            CACHE_DIR.with_name(f"{CACHE_DIR.name}_{mode}")
    else:
        cache_dir = CACHE_DIR.with_name(f"{CACHE_DIR.name}_fs{FS}_{mode}")
    if LABEL_MODE == "multi":   # different set of records: separate cache
        cache_dir = cache_dir.with_name(cache_dir.name + "_multi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else \
        dict(norm_mode=mode, sat={})
    tab = None
    for split, folds in (("train", TRAIN_FOLDS), ("val", VAL_FOLDS), ("test", TEST_FOLDS)):
        fx, fy, fi, fm = (cache_dir / f"{split}_{s}.npy" for s in ("x", "y", "id", "ym"))
        if fx.is_file() and fy.is_file() and fi.is_file() and \
                (LABEL_MODE == "single" or fm.is_file()):
            continue
        if tab is None:
            tab = superclass_table(data)
            missing = [f for f in tab.file
                       if not (data / f"{f}.dat").is_file()]
            if missing:
                sys.exit(f"{len(missing)} records{FS} records missing "
                         f"(e.g. {data / missing[0]}.dat): download PTB-XL 1.0.3")
            if mode == "fixed" and "scale" not in meta:
                tr_files = tab[tab.strat_fold.isin(TRAIN_FOLDS)].file.to_numpy()
                meta["scale"] = calibrate_scale(tr_files, data)
        sel = tab[tab.strat_fold.isin(folds)]
        print(f"  preprocessing {split}: {len(sel)} records ...")
        out = [preprocess(data / f"{f}.dat", mode, meta.get("scale"))
               for f in sel.file]
        X = np.stack([o[0] for o in out])
        n_sat = sum(o[1] for o in out)
        meta["sat"][split] = dict(samples=n_sat,
                                  frac=n_sat / (len(sel) * RAW_L * NLEAD))
        print(f"    samples saturated in Q5.11: {n_sat} "
              f"({100 * meta['sat'][split]['frac']:.4f}%)")
        # y: index of the superclass if single, -1 if the record has more than one
        Y = np.array([CLASSES.index(c) if c is not None else -1 for c in sel.label],
                     dtype=np.int64)
        np.save(fx, X); np.save(fy, Y); np.save(fi, sel.index.to_numpy())
        np.save(fm, sel[CLASSES].to_numpy(dtype=np.int8))
        meta_path.write_text(json.dumps(meta, indent=2))
    return {s: (cache_dir / f"{s}_x.npy", cache_dir / f"{s}_y.npy",
                cache_dir / f"{s}_ym.npy") for s in ("train", "val", "test")}, meta


class ECGDataset(torch.utils.data.Dataset):
    """(L, 12) float = Q5.11 code / 2048: the value the encoder sees.
    Returns (x, multi-hot labels, single superclass or -1)."""

    def __init__(self, x_path, y_path, ym_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path)
        self.YM = np.load(ym_path).astype(np.float32) if Path(ym_path).is_file() \
            else np.eye(len(CLASSES), dtype=np.float32)[self.Y]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        x = np.asarray(self.X[i], dtype=np.float32) / (1 << F_IN)
        return torch.from_numpy(x), torch.from_numpy(self.YM[i]), int(self.Y[i])


def build_loaders():
    paths, meta = build_cache()

    def mk(split, shuffle):
        ds = ECGDataset(*paths[split])
        if SUBSET:
            n = min(SUBSET if split == "train" else max(SUBSET // 8, 1), len(ds))
            ds = torch.utils.data.Subset(ds, range(n))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
            num_workers=WORKERS, pin_memory=True)

    loaders = mk("train", True), mk("val", False), mk("test", False)
    counts = {}
    for s in paths:
        ys = np.load(paths[s][1])
        c = np.bincount(ys[ys >= 0], minlength=len(CLASSES))
        pos = loaders[("train", "val", "test").index(s)].dataset
        pos = (pos.dataset if isinstance(pos, torch.utils.data.Subset) else pos).YM.sum(0)
        counts[s] = c
        print(f"  {s:5s} {len(ys):6d} records, {int(c.sum())} single-label.  "
              "positive: " + "  ".join(f"{k} {int(v)}" for k, v in zip(CLASSES, pos)))
    return loaders, counts, meta


def class_weights(counts, mode=CLASS_WEIGHTING):
    if mode is None:
        return None
    f = counts / counts.sum()
    w = 1.0 / np.sqrt(f) if mode == "inv_sqrt" else 1.0 / f
    w = w / (w * f).sum()           # mean weight 1 on the training distribution
    return torch.tensor(w, dtype=torch.float32)


# ============================================================================
# scheduler
# ============================================================================
def cosine_with_warmup(opt, warmup_steps, total_steps, min_ratio=0.0):
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ============================================================================
# metrics
# ============================================================================
def auc(y, s):
    """Binary AUC with the Mann-Whitney statistic (ties: mean rank)."""
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, r) / cnt)[inv]
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def evaluate(model, loader, device):
    """Macro-AUC over all the records (multi-hot labels, logits as scores),
    and accuracy / balanced accuracy / macro-F1 / confusion matrix with the
    argmax on the single-label records, i.e. what the board does."""
    model.eval()
    k = len(CLASSES)
    Z, YM, Y = [], [], []
    with torch.no_grad():
        for x, ym, y in loader:
            Z.append(model(x.to(device)).float().cpu().numpy())
            YM.append(ym.numpy()); Y.append(y.numpy())
    Z, YM, Y = np.concatenate(Z), np.concatenate(YM), np.concatenate(Y)
    aucs = [auc(YM[:, c], Z[:, c]) for c in range(k)]
    one = Y >= 0
    cm = np.zeros((k, k), dtype=np.int64)          # rows true, columns predicted
    np.add.at(cm, (Y[one], Z[one].argmax(-1)), 1)
    tp = np.diag(cm).astype(np.float64)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return dict(acc=float(tp.sum() / max(cm.sum(), 1)),
                bal_acc=float(rec.mean()), macro_f1=float(f1.mean()),
                recall={c: float(r) for c, r in zip(CLASSES, rec)},
                cm=cm.tolist(), macro_auc=float(np.nanmean(aucs)),
                auc={c: a for c, a in zip(CLASSES, aucs)},
                n_records=int(len(Y)), n_single=int(one.sum()))


def print_summary(model):
    print("\n===== ARCHITECTURE =====")
    print(model)
    print("\n===== PARAMETERS =====")
    tot = 0
    for name, p in model.named_parameters():
        tot += p.numel()
        print(f"{name:52s} {str(tuple(p.shape)):20s} {p.numel():>9,}")
    print(f"\nTOTAL PARAMETERS: {tot:,}")
    print("=" * 60 + "\n")
    return tot


def param_groups(model):
    """The S4D kernel out of the weight decay, as for Speech Commands."""
    ssm_keys = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
    ssm, nod, dec = [], [], []
    for n_, p_ in model.named_parameters():
        if not p_.requires_grad:
            continue
        if any(k in n_ for k in ssm_keys):
            ssm.append(p_)
        elif n_.endswith(".bias") or ".bn." in n_ or n_.endswith(".D"):
            nod.append(p_)
        else:
            dec.append(p_)
    assert len(ssm) == 3 * N_LAYERS, f"SSM filter: {len(ssm)}, {3*N_LAYERS} expected"
    assert nod, "empty no_decay filter: BatchNormSeq names changed?"
    return dec, nod, ssm


# ============================================================================
# training
# ============================================================================
def train(model, loaders, dev, exp_dir, weights):
    tr, va, te = loaders
    dec, nod, ssm = param_groups(model)
    print(f"parameters: {len(dec)} with WD, {len(nod)} without, {len(ssm)} SSM")

    opt = torch.optim.AdamW([
        {"params": dec, "weight_decay": WD},
        {"params": nod, "weight_decay": 0.0},
        {"params": ssm, "weight_decay": 0.0, "lr": LR_SSM},
    ], lr=LR)
    spe = max(len(tr), 1)
    sched = cosine_with_warmup(opt, WARMUP_EPOCHS * spe, EPOCHS * spe)
    if LABEL_MODE == "multi":
        bce = nn.BCEWithLogitsLoss()
        crit = lambda z, ym, y: bce(z, ym)
    else:
        ce = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(dev))
        crit = lambda z, ym, y: ce(z, y)
    log_path = exp_dir / "train_log.jsonl"

    best, best_sd, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; t0 = time.time(); nan_hit = False
        for x, ym, y in tr:
            x, ym, y = (t.to(dev, non_blocking=True) for t in (x, ym, y))
            loss = crit(model(x), ym, y)
            # always float32: in half precision the S4D kernel goes to nan (see the
            # Speech Commands script). A nan here comes from the forward, not the gradient.
            if not torch.isfinite(loss):
                nan_hit = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if nan_hit:
            print(f"non-finite loss at epoch {ep+1}: stopped (lower LR). "
                  f"The best model stays in {exp_dir/'best.pt'}")
            break

        m = evaluate(model, va, dev)
        dt = time.time() - t0
        avg = tot / spe
        print(f"epoch {ep+1}/{EPOCHS}  loss {avg:.4f}  val AUC {m['macro_auc']:.4f}  "
              f"acc {m['acc']:.4f}  "
              f"bal {m['bal_acc']:.4f}  F1 {m['macro_f1']:.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.0f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(epoch=ep + 1, loss=avg,
                                    val_macro_auc=m["macro_auc"], val_acc=m["acc"],
                                    val_bal_acc=m["bal_acc"],
                                    val_macro_f1=m["macro_f1"],
                                    lr=opt.param_groups[0]["lr"],
                                    sec=round(dt, 1))) + "\n")

        score = m[SELECT_ON]
        if score > best:
            best, best_sd, bad = score, copy.deepcopy(model.state_dict()), 0
            torch.save(best_sd, exp_dir / "best.pt")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping (best val {SELECT_ON} {best:.4f})")
                break
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, best=best,
                        patience=bad), exp_dir / "last.pt")

    model.load_state_dict(best_sd)
    return model, best


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = load_modules()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("device:", dev)

    if D_MODEL != G.H:
        print(f"WARNING: D_MODEL={D_MODEL} but the RTL has H={G.H}: not exportable.")
    if D_STATE // 2 != G.MODES:
        print(f"WARNING: D_STATE/2={D_STATE//2} but the RTL has MODES={G.MODES}.")
    if not HW_COMPATIBLE:
        print(f"WARNING: FS={FS} Hz, L={L}: float reference experiment. "
              f"The core has L=1024 fixed: this model is NOT exportable.")

    loaders, counts, meta = build_loaders()
    print(f"normalization: {NORM_MODE}" +
          (f"  SCALE {meta['scale']:.4f} /mV" if "scale" in meta else ""))
    weights = class_weights(counts["train"])
    if weights is not None:
        print("loss weights:", "  ".join(f"{c} {w:.2f}"
                                         for c, w in zip(CLASSES, weights.tolist())))

    model = ecg_model(B, len(CLASSES)).to(dev)
    n_par = print_summary(model) if PRINT_SUMMARY else None
    if SUMMARY_ONLY:
        print("SUMMARY_ONLY active: no training.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    exp_dir = EXP_ROOT / f"exp_{stamp}_ecg{FS}_l{L}_{NORM_MODE}_{LABEL_MODE}"
    exp_dir.mkdir(parents=True, exist_ok=False)     # never overwrite a run
    shutil.copy2(__file__, exp_dir / Path(__file__).name)
    print("output:", exp_dir)

    model, best_val = train(model, loaders, dev, exp_dir, weights)
    val = evaluate(model, loaders[1], dev)
    test = evaluate(model, loaders[2], dev)
    print(f"\ntest: macro-AUC {test['macro_auc']:.4f} ({test['n_records']} records)   "
          + "  ".join(f"{c} {a:.3f}" for c, a in test["auc"].items()))
    print(f"single-label test ({test['n_single']} records, argmax as on the board): "
          f"acc {test['acc']:.4f}  bal {test['bal_acc']:.4f}  "
          f"F1 {test['macro_f1']:.4f}   (best val {SELECT_ON} {best_val:.4f})")
    print("recall per class:", "  ".join(f"{c} {r:.3f}" for c, r in test["recall"].items()))

    (exp_dir / "report.json").write_text(json.dumps(dict(
        task=f"ptbxl_superdiag_{LABEL_MODE}", label_mode=LABEL_MODE, classes=CLASSES,
        input=f"records{FS} {NORM_MODE} Q5.11", fs=FS, hw_compatible=HW_COMPATIBLE, norm_mode=NORM_MODE,
        scale=meta.get("scale"), fixed_target=FIXED_TARGET, fixed_pct=FIXED_PCT,
        sat=meta.get("sat"),
        raw_len=RAW_L, seq_len=L, n_lead=NLEAD, f_in=F_IN,
        folds=dict(train=TRAIN_FOLDS, val=VAL_FOLDS, test=TEST_FOLDS),
        counts={s: c.tolist() for s, c in counts.items()},
        val=val, test=test, select_on=SELECT_ON, best_val=best_val,
        n_params=n_par, seed=SEED, d_model=D_MODEL, n_layers=N_LAYERS,
        d_state=D_STATE, dropout=DROPOUT, lr=LR, lr_ssm=LR_SSM, wd=WD,
        batch=BATCH, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE, class_weighting=CLASS_WEIGHTING,
        subset=SUBSET), indent=2))
    print("report:", exp_dir / "report.json")

    if not HW_COMPATIBLE:
        print(f"\nFloat reference at {FS} Hz, L={L}: no export to the accelerator.")
        return
    print("\n--- notes for the export to the accelerator ---")
    print("1. K_CLS = 5; 12-input encoder (encoder_ecg.v, signed Q5.11):")
    print("   the quantizer must export W (128x12) and b, not the 1-channel w/b.")
    print(f"2. No MU/SIGMA: the normalization ({NORM_MODE}) is already in the stimulus"
          + (f", SCALE {meta['scale']:.4f} /mV." if "scale" in meta else "."))
    print("3. Board stimuli: generate them with preprocess() of this file, same")
    print("   NORM_MODE and same SCALE (records100, 1000 samples + 24 zeros),")
    print("   NOT with the 500 Hz version of Export_ECG_stimulus.py.")


if __name__ == "__main__":
    main()
