#!/usr/bin/env python3
"""
Quantize_s4d_ecg.py -- rescaling, quantization, .mem export and bit-accurate
check of the ECG (PTB-XL) model trained by Train_s4d_ecg.py.

It runs without arguments, from the root of the repository:

    conda run -n mamba_env python Quantize_s4d_ecg.py

It is the same chain as Train_and_quantize_s4d_sCIFAR.py (BatchNorm variant)
and Quantize_s4d_speech_command.py, and it uses the SAME modules gen_s4d_mem /
s4d_export / s4d_bn extracted from the monolithic script, without modifying them:

  1. range calibration with hooks on the real model (E.calibrate_ranges)
  2. global scale s, a power of two, on the value branch, compensated exactly
     on w of the folded BN (B.rescale_value_branch) + check in ulp
  3. S4D layers: ZOH biquad, Jury clamp, per-channel exponent of b, D, W_out,
     bout (E.quantize_layer); BN folded into an affine (B.quantize_affine)
  4. GELU/sigmoid LUTs (E.write_luts)
  5. bit-accurate check with E.LayerFX + B.AffineFX, i.e. the mirror verified
     against the RTL

DIFFERENCES FROM sCIFAR, imposed by the ECG hardware and NOT by the model
The S4D_PTB_XL_UltraTiny bitstream is already synthesized. From the block design
(s4d_PTB_XL.bd) and the RTL defaults the generics are FIXED:

    F_SSM   = 9    (s4d_core_axis_0, set in the bd)
    F_D     = 13   (default of s4d_core_axis)
    F_WENC  = 31   (default of s4d_encoder_ecg_axis) -> encoder weights with 20
                   fractional bits, bias with 31, SH_OUT = 20
    RQ_SHIFT= 6    (default of s4d_decoder_axis)     -> F_CIN = 5
    BN_SH   = 13   (s4d_decoder_axis_0, set in the bd) -> final norm F_BNW 13
    K       = 5, L = 1024, LOG2_L = 10

sCIFAR CHOOSES them from the calibration and then prints the generics to pass;
here, with HW_LOCK = True, the bitstream ones are IMPOSED and their cost is
measured (saturations, SNR, accuracy). The BN shift of the six layers is not a
generic: it travels in the weight stream (ld_bn_sh), so it stays calibrated per
layer as in sCIFAR. With HW_LOCK = False the free quantization of sCIFAR is
obtained, as a reference: those .mem files do NOT run on the current bitstream.

The ECG encoder differs from the sCIFAR one (12 signed Q5.11 inputs instead of
one uint8 pixel) and has its own quantizer here, written on
encoder_ecg.v: y = sat16(round((sum_c W[ch][c]*x[c] + b[ch]) >> SH_OUT)),
enc_w.mem = 128 rows of 12x18 bit with c0 in the low bits, enc_b.mem at 32 bit.

NOTE ON THE BOARD
enc_w/enc_b, l6_bn_*, cls_w/cls_b and the LUTs are ROMs initialized by $readmemh:
they change only by synthesizing again (same generics, no RTL change). The
weights of the six layers instead go through the stream (Export_s4d_wstream.py,
with MEM_DIR pointing to the folder produced here).
"""

import json
import math
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import Train_s4d_ecg as T

# ============================================================================
# CONFIGURATION
# ============================================================================
EXP = Path("experiments/exp_20260923_1249_ecg100_l1024_fixed_multi")  # run E: set it to the folder created by Train_s4d_ecg.py
CKPT = EXP / "best.pt"

HW_LOCK = True          # True: generics of the ECG bitstream; False: free (sCIFAR)
HW = dict(F_SSM=9, F_D=13, F_WENC=31, RQ_SHIFT=6, BN_SH_FINAL=13, K=5, L=1024)

FILL = 0.75             # fraction of the ceiling after rescaling, as sCIFAR
N_CAL = None            # validation records for the calibration (None = all)
N_EVAL_FX = None        # test records for the bit-accurate check (None = all)
CHUNK_FX = 32           # records per process in the bit-accurate check
WORKERS_FX = 32         # parallel processes (the bit-accurate chain runs on CPU)
BATCH = 32
SEED = 1337
WRITE_MEM = True        # False: measurement only, the .mem files go to a temporary folder


# ============================================================================
# model <-> numpy (12-input encoder)
# ============================================================================
def weights_to_numpy_ecg(B, model):
    """B.weights_to_numpy flattens enc_w (made for a single input):
    here the (H, 12) shape is restored."""
    W = B.weights_to_numpy(model, T.N_LAYERS)
    sd = model.state_dict()
    W["enc_w"] = sd["input_proj.weight"].detach().cpu().double().numpy()
    W["enc_b"] = sd["input_proj.bias"].detach().cpu().double().numpy()
    return W


def numpy_to_torch_ecg(B, model, W):
    """B.numpy_to_torch_ writes enc_w as a column (H, 1): it is overwritten."""
    W1 = dict(W)
    W1["enc_w"] = np.zeros(W["enc_w"].shape[0])
    B.numpy_to_torch_(model, W1)
    with torch.no_grad():
        model.input_proj.weight.copy_(torch.tensor(W["enc_w"]))
        model.input_proj.bias.copy_(torch.tensor(W["enc_b"]))
    return model


# ============================================================================
# ECG encoder: quantization, .mem, bit-accurate mirror
# ============================================================================
W_WENC, W_BENC, W_ENCACC, NIN = 18, 32, 40, 12


def quantize_encoder_ecg(G, enc_w, enc_b, f_wenc=None):
    """acc = sum_c wq[ch][c]*xq[c] + bq[ch], with xq in Q5.11: F_WENC = f_w + 11.

    f_wenc None: the largest that saturates neither w (18 bit) nor b (32 bit),
    as quantize_encoder of sCIFAR. Otherwise imposed (generic of the bitstream).
    """
    w = np.asarray(enc_w, dtype=np.float64)
    b = np.asarray(enc_b, dtype=np.float64)
    f_free = min(G.calib(w, W_WENC) + G.F_ACT, G.calib(b, W_BENC))
    f = f_free if f_wenc is None else int(f_wenc)
    f_w = f - G.F_ACT
    sh_out = f - G.F_ACT
    lim_w, lim_b = (1 << (W_WENC - 1)) - 1, (1 << (W_BENC - 1)) - 1
    sat_w = int((np.abs(np.round(w * 2.0 ** f_w)) > lim_w).sum())
    sat_b = int((np.abs(np.round(b * 2.0 ** f)) > lim_b).sum())
    return dict(w=G.q(w, f_w, W_WENC), b=G.q(b, f, W_BENC), f_enc=f, f_w=f_w,
                sh_out=sh_out, f_free=f_free, sat_w=sat_w, sat_b=sat_b,
                max_w=float(np.abs(w).max()), max_b=float(np.abs(b).max()))


def write_encoder_mem_ecg(G, enc, outdir):
    rows = []
    for h in range(enc["w"].shape[0]):
        word = 0
        for c in range(NIN):                          # C0_AT_LSB = 1
            word |= (int(enc["w"][h, c]) & ((1 << W_WENC) - 1)) << (c * W_WENC)
        rows.append(G.hx(word, NIN * W_WENC))
    (outdir / "enc_w.mem").write_text("\n".join(rows) + "\n")
    (outdir / "enc_b.mem").write_text("\n".join(G.hx(v, W_BENC) for v in enc["b"]) + "\n")


def encoder_ecg_n(E, G, xq, enc):
    """xq (B, 12) int64 Q5.11 -> (B, H) Q5.11, as encoder_ecg.v (S2..S6)."""
    acc = xq @ enc["w"].T + enc["b"][None, :]
    lim = 1 << (W_ENCACC - 1)
    assert np.abs(acc).max() < lim, "encoder accumulator beyond W_ENCACC"
    return E.satn(E.shr_r(acc, enc["sh_out"]), G.W_ACT)


# ============================================================================
# quantization of the final norm with an imposed SH
# ============================================================================
def quantize_affine_fixed_sh(G, B, w, b, sh, f_out, name):
    """B.quantize_affine with F_BNW = sh + f_out - F_ACT imposed (BN_SH of the bd)."""
    f_bnw = sh + f_out - G.F_ACT
    f_b = f_bnw + G.F_ACT
    w = np.asarray(w, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    lim_w, lim_b = (1 << (B.W_BNW - 1)) - 1, (1 << (B.W_BNB - 1)) - 1
    warn = []
    nw = int((np.abs(np.round(w * 2.0 ** f_bnw)) > lim_w).sum())
    nb = int((np.abs(np.round(b * 2.0 ** f_b)) > lim_b).sum())
    if nw:
        warn.append(f"{name}: {nw}/{w.size} w clipped with F_BNW {f_bnw} "
                    f"(max|w| {np.abs(w).max():.3f}, ceiling {lim_w / 2 ** f_bnw:.3f})")
    if nb:
        warn.append(f"{name}: {nb}/{b.size} b clipped with F_B {f_b}")
    wq, bq = G.q(w, f_bnw, B.W_BNW), G.q(b, f_b, B.W_BNB)
    nz = np.abs(w) > 0
    rel = (0.5 / 2.0 ** f_bnw) / np.abs(w[nz]).min() if nz.any() else 0.0
    acc = int(np.abs(wq).max()) * ((1 << (G.W_ACT - 1)) - 1) + int(np.abs(bq).max())
    qa = dict(w=wq, b=bq, f_bnw=f_bnw, f_b=f_b, f_out=f_out, sh=sh, warn=warn,
              max_w=float(np.abs(w).max()), max_b=float(np.abs(b).max()),
              acc_bound=acc, rel_worst=float(rel),
              f_bnw_free=B.quantize_affine(w, b, f_out=f_out, name=name,
                                           verbose=False)["f_bnw"])
    print(f"  {name}: F_BNW {f_bnw} (imposed; free {qa['f_bnw_free']})  SH {sh}  "
          f"max|w| {qa['max_w']:.4f}  max|b| {qa['max_b']:.4f}")
    for x in warn:
        print("  WARNING  " + x)
    return qa


# ============================================================================
# bit-accurate check, parallel
# ============================================================================
_CTX = {}


def _chain_worker(args):
    """Whole chain on a block of records: encoder -> 6 layers -> final norm
    -> mean pool. Runs in a child process (fork): reads the weights from _CTX."""
    i0, i1 = args
    E, G, B = _CTX["E"], _CTX["G"], _CTX["B"]
    X = np.asarray(_CTX["X"][i0:i1], dtype=np.int64)          # (b, L, 12)
    b, L, _ = X.shape
    norms = [B.AffineFX(qa) for qa in _CTX["QA"]]            # W_OUT = W_U
    nf = B.AffineFX(_CTX["NF"], w_out=G.W_ACT)
    layers = [E.LayerFX(q, _CTX["luts"], b, norm=n) for q, n in zip(_CTX["QZ"], norms)]
    pool = np.zeros((b, G.H), dtype=np.int64)
    enc_sat = 0
    lim = (1 << (G.W_ACT - 1)) - 1
    for k in range(L):
        acc = X[:, k] @ _CTX["enc"]["w"].T + _CTX["enc"]["b"][None, :]
        raw = E.shr_r(acc, _CTX["enc"]["sh_out"])
        enc_sat += int((np.abs(raw) > lim).sum())
        x = E.satn(raw, G.W_ACT)
        for lay in layers:
            x = lay.step(x)
        pool += nf(x)
    xbar = E.satn(E.shr_r(pool, int(math.log2(L))), G.W_ACT)
    sat = dict(enc=enc_sat,
               layers=[dict(sat_x=l.sat_x, sat_a=l.sat_a, sat_y=l.sat_y,
                            acc_bound=l.acc_bound, norm=n.sat)
                       for l, n in zip(layers, norms)],
               final=nf.sat)
    return i0, xbar, sat


def run_chain_parallel(idx_blocks):
    with mp.get_context("fork").Pool(WORKERS_FX) as pool:
        out, done = [], 0
        for r in pool.imap_unordered(_chain_worker, idx_blocks):
            out.append(r)
            done += r[1].shape[0]
            print(f"  bit-accurate {done}/{idx_blocks[-1][1]}", end="\r", flush=True)
    print()
    out.sort(key=lambda r: r[0])
    xbar = np.concatenate([r[1] for r in out])
    sat = dict(enc=sum(r[2]["enc"] for r in out), final=sum(r[2]["final"] for r in out),
               layers=[])
    for m in range(T.N_LAYERS):
        ls = [r[2]["layers"][m] for r in out]
        sat["layers"].append(dict(
            sat_x=sum(l["sat_x"] for l in ls), sat_a=sum(l["sat_a"] for l in ls),
            sat_y=sum(l["sat_y"] for l in ls), norm=sum(l["norm"] for l in ls),
            acc_bound=max(l["acc_bound"] for l in ls)))
    return xbar, sat


# ============================================================================
# metrics
# ============================================================================
def metrics(logits, ym, y):
    aucs = [T.auc(ym[:, c], logits[:, c]) for c in range(len(T.CLASSES))]
    one = y >= 0
    p = logits[one].argmax(1)
    k = len(T.CLASSES)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y[one], p), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return dict(macro_auc=float(np.nanmean(aucs)),
                auc={c: a for c, a in zip(T.CLASSES, aucs)},
                acc=float(np.diag(cm).sum() / max(cm.sum(), 1)),
                bal_acc=float(rec.mean()), cm=cm.tolist(),
                n_records=int(len(y)), n_single=int(one.sum()))


def auc_table(title, ym, m):
    n = len(ym)
    lines = [f"=== {title} ===",
             f"{'class':6s} {'positive':>9s} {'negative':>9s} {'AUC':>8s}"]
    for k, c in enumerate(T.CLASSES):
        pz = int(ym[:, k].sum())
        lines.append(f"{c:6s} {pz:9d} {n - pz:9d} {m['auc'][c]:8.4f}")
    lines.append(f"{'macro':6s} {'':9s} {'':9s} {m['macro_auc']:8.4f}")
    return "\n".join(lines)


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = T.load_modules()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    assert T.FS == 100 and T.L == HW["L"], "Train_s4d_ecg.py must be at FS=100, L=1024"
    E.K_CLS = HW["K"]

    rep_train = json.loads((EXP / "report.json").read_text())
    cache = T.CACHE_DIR.with_name(f"{T.CACHE_DIR.name}_{rep_train['norm_mode']}")
    if rep_train.get("label_mode") == "multi":
        cache = cache.with_name(cache.name + "_multi")
    print(f"checkpoint {CKPT}\ncache {cache}   HW_LOCK {HW_LOCK}")

    stamp = time.strftime("%Y%m%d_%H%M")
    tag = "hw" if HW_LOCK else "free"
    out_dir = EXP / f"quant_{stamp}_{tag}"
    mem_dir = out_dir / "mem" if WRITE_MEM else Path("/tmp") / f"ecg_quant_dry_{stamp}"
    mem_dir.mkdir(parents=True, exist_ok=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = T.ecg_model(B, len(T.CLASSES)).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=True))
    model.eval()

    # --- calibration on the validation set (fold 9) ------------------------
    class CalDS(torch.utils.data.Dataset):          # calibrate_ranges wants (x, y)
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            x, _, y = self.ds[i]
            return x, y
    val = T.ECGDataset(cache / "val_x.npy", cache / "val_y.npy", cache / "val_ym.npy")
    n_cal = len(val) if N_CAL is None else min(N_CAL, len(val))
    cal = torch.utils.data.DataLoader(CalDS(val), batch_size=BATCH, shuffle=False)
    W = weights_to_numpy_ecg(B, model)
    rng_before = E.calibrate_ranges(model, cal, dev, n_images=n_cal, n_layers=T.N_LAYERS)
    print(f"\ncalibration on {n_cal} validation records. Ranges before rescaling:")
    E.print_ranges(rng_before, T.N_LAYERS)

    s = E.choose_scale(max(rng_before["max_x"], rng_before["max_a"]), fill=FILL)
    W_fix = B.rescale_value_branch(W, s)

    # --- the rescaling must not change the function -------------------------
    model_fix = numpy_to_torch_ecg(B, T.ecg_model(B, len(T.CLASSES)), W_fix).to(dev).eval()
    xb, _ = next(iter(cal))
    with torch.no_grad():
        l0 = model(xb.to(dev)).double().cpu().numpy()
        l1 = model_fix(xb.to(dev)).double().cpu().numpy()
    d = float(np.abs(l1 - l0).max())
    ulp = float(np.spacing(np.float32(max(np.abs(l0).max(), 1.0))))
    print(f"\ns = {s}   rescaling check: max deviation {d:.3e} ({d/ulp:.0f} ulp)")
    assert d < 256 * ulp, "the rescaling changed the function"

    rng_after = E.calibrate_ranges(model_fix, cal, dev, n_images=n_cal, n_layers=T.N_LAYERS)
    print("\nranges after rescaling:")
    E.print_ranges(rng_after, T.N_LAYERS)

    # --- F_SSM and F_D ------------------------------------------------------
    f_ssm_free = E.needed_f_ssm(rng_before["max_yssm"], fill=FILL)
    f_ssm = HW["F_SSM"] if HW_LOCK else f_ssm_free
    E.set_f_ssm(f_ssm)
    f_d_cal = [G.calib(dd["D"], G.W_D) for dd in W_fix["layers"]]
    F_D = HW["F_D"] if HW_LOCK else min(f_d_cal)
    print(f"\nF_SSM {f_ssm} (calibrated {f_ssm_free}; yssm ceiling +-{E.CEIL_SSM:g}, "
          f"measured max|yssm| {rng_before['max_yssm']:.2f})")
    print(f"F_D   {F_D} (calibrated per layer {f_d_cal})")

    luts = E.write_luts(mem_dir, verbose=False)
    (mem_dir / "isqrt_lut.mem").unlink()            # not needed without LayerNorm

    # --- layers ------------------------------------------------------------
    QZ, QA = [], []
    for m, dd in enumerate(W_fix["layers"]):
        print(f"\nlayer {m}")
        q = E.quantize_layer(dd, verbose=True, f_d=F_D)
        E.write_layer_mem(m, q, mem_dir, verbose=False, with_norm=False)
        qa = B.quantize_affine(dd["bn_w"], dd["bn_b"], name=f"l{m}_bn", verbose=True)
        B.write_affine_mem(m, qa, mem_dir, verbose=False)
        QZ.append(q); QA.append(qa)

    # --- encoder -----------------------------------------------------------
    enc = quantize_encoder_ecg(G, W_fix["enc_w"], W_fix["enc_b"],
                               HW["F_WENC"] if HW_LOCK else None)
    print(f"\nencoder: F_WENC {enc['f_enc']} (free {enc['f_free']})  SH_OUT {enc['sh_out']}  "
          f"max|w| {enc['max_w']:.4f}  max|b| {enc['max_b']:.4f}  "
          f"clipped w {enc['sat_w']}/{enc['w'].size}  b {enc['sat_b']}/{enc['b'].size}")
    write_encoder_mem_ecg(G, enc, mem_dir)

    # --- final norm ---------------------------------------------------------
    print()
    if HW_LOCK:
        nf_qa = quantize_affine_fixed_sh(G, B, W_fix["nf_w"], W_fix["nf_b"],
                                         HW["BN_SH_FINAL"], G.F_ACT, f"l{T.N_LAYERS}_bn")
        B.write_affine_mem(T.N_LAYERS, nf_qa, mem_dir, verbose=False)
    else:
        nf_qa = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], mem_dir,
                                           nlayer=T.N_LAYERS, verbose=True)

    # --- bit-accurate check on the test set ---------------------------------
    Xte = np.load(cache / "test_x.npy", mmap_mode="r")
    Yte = np.load(cache / "test_y.npy")
    YMte = np.load(cache / "test_ym.npy").astype(np.int64)
    n = len(Yte) if N_EVAL_FX is None else min(N_EVAL_FX, len(Yte))
    blocks = [(i, min(i + CHUNK_FX, n)) for i in range(0, n, CHUNK_FX)]
    print(f"\nbit-accurate check on {n} test records, {WORKERS_FX} processes ...")
    _CTX.update(E=E, G=G, B=B, X=Xte, QZ=QZ, QA=QA, NF=nf_qa, enc=enc, luts=luts)
    t0 = time.time()
    xbar, sat = run_chain_parallel(blocks)
    print(f"  {time.time() - t0:.0f} s")

    # --- classifier ---------------------------------------------------------
    xbar_max = float(np.abs(xbar).max()) / 2.0 ** G.F_ACT
    f_cin_free = int(math.floor(math.log2(127 / max(xbar_max, 1e-12))))
    f_cin = G.F_ACT - HW["RQ_SHIFT"] if HW_LOCK else f_cin_free
    dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin,
                              mem_dir, nlayer=T.N_LAYERS, verbose=False, with_norm=False)
    xq = E.satn(E.shr_r(xbar, dec["rq_shift"]), E.W_CIN)
    sat_cin = int((np.abs(E.shr_r(xbar, dec["rq_shift"])) > 127).sum())
    logits_fx = E.classify_fx(xbar, dec)
    print(f"classifier: F_CIN {f_cin} (free {f_cin_free}), RQ_SHIFT {dec['rq_shift']}, "
          f"F_W {dec['f_w']}, max|xbar| {xbar_max:.3f}, clipped INT8 inputs {sat_cin}")

    # --- float reference on the same records --------------------------------
    lf = []
    with torch.no_grad():
        for i in range(0, n, 64):
            x = torch.from_numpy(np.asarray(Xte[i:min(i + 64, n)], dtype=np.float32)
                                 / 2 ** T.F_IN).to(dev)
            lf.append(model(x).double().cpu().numpy())
    logits_f = np.concatenate(lf)

    scaled = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
    err = scaled - logits_f
    snr = 20 * math.log10(np.sqrt((logits_f ** 2).mean()) / max(np.sqrt((err ** 2).mean()), 1e-12))
    mf = metrics(logits_f, YMte[:n], Yte[:n])
    mq = metrics(logits_fx.astype(np.float64), YMte[:n], Yte[:n])
    agree = float((logits_fx.argmax(1) == logits_f.argmax(1)).mean())

    tot_sat = (sat["enc"] + sat["final"] + sat_cin
               + sum(l["sat_x"] + l["sat_a"] + l["sat_y"] + l["norm"] for l in sat["layers"]))
    print(f"\nlogit SNR {snr:.1f} dB   argmax agreement {agree:.4f}   total saturations {tot_sat}")
    print(f"  {'layer':>6s}{'sat_x':>10s}{'sat_a':>10s}{'sat_y':>10s}{'norm':>10s}")
    for m, l in enumerate(sat["layers"]):
        print(f"  {m:>6d}{l['sat_x']:>10d}{l['sat_a']:>10d}{l['sat_y']:>10d}{l['norm']:>10d}")
    print(f"  encoder {sat['enc']}   final norm {sat['final']}   INT8 input {sat_cin}")

    tf = auc_table(f"PTB-XL superdiagnostic, fold 10 -- FLOAT ({n} records)", YMte[:n], mf)
    tq = auc_table(f"PTB-XL superdiagnostic, fold 10 -- FIXED bit-accurate "
                   f"({'bitstream generics' if HW_LOCK else 'free generics'}, {n} records)",
                   YMte[:n], mq)
    summary = (f"{tf}\n\n{tq}\n\n"
               f"single-label accuracy ({mf['n_single']} records, argmax):  "
               f"float {mf['acc']:.4f}   fixed {mq['acc']:.4f}   "
               f"(drop {100 * (mf['acc'] - mq['acc']):+.2f} points)\n"
               f"balanced accuracy: float {mf['bal_acc']:.4f}   fixed {mq['bal_acc']:.4f}\n"
               f"logit SNR {snr:.1f} dB   argmax agreement {agree:.4f}   saturations {tot_sat}")
    print("\n" + summary)

    # --- report -------------------------------------------------------------
    rep = dict(
        checkpoint=str(CKPT), hw_lock=HW_LOCK, hw_generics=HW, n_cal=n_cal, n_eval_fx=n,
        scale=s, fill=FILL, f_ssm=f_ssm, f_ssm_free=f_ssm_free, f_d=F_D, f_d_calibrati=f_d_cal,
        peaks_before={k: rng_before[f"max_{k}"] for k in ("x", "a", "yssm")},
        peaks_after={k: rng_after[f"max_{k}"] for k in ("x", "a", "yssm")},
        per_layer_x_before=rng_before["x"], per_layer_x_after=rng_after["x"],
        encoder=dict(F_WENC=enc["f_enc"], F_WENC_free=enc["f_free"], SH_OUT=enc["sh_out"],
                     sat_w=enc["sat_w"], sat_b=enc["sat_b"], max_w=enc["max_w"], max_b=enc["max_b"]),
        decoder=dict(F_CIN=dec["f_cin"], F_CIN_free=f_cin_free, F_W=dec["f_w"],
                     RQ_SHIFT=dec["rq_shift"], xbar_max=xbar_max, sat_cin=sat_cin),
        # `affine[m].SH` is the field that Export_s4d_wstream.py reads for ld_bn_sh
        affine=[dict(idx=m, F_BNW=q["f_bnw"], F_B=q["f_b"], SH=q["sh"], max_w=q["max_w"],
                     max_b=q["max_b"], rel_worst=q["rel_worst"], warn=q["warn"])
                for m, q in enumerate(QA + [nf_qa])],
        layers=[dict(jury_before=q["jury_before"], jury_after=q["jury_after"], rho=q["rho"],
                     f_d=q["f_d"], f_d_cal=q["f_d_cal"], bsh_min=int(q["bsh"].min()),
                     bsh_max=int(q["bsh"].max()), warn=q["warn"], **sat["layers"][m])
                for m, q in enumerate(QZ)],
        sat_total=tot_sat, sat_enc=sat["enc"], sat_final=sat["final"],
        logit_snr_db=round(snr, 2), agreement=agree, float=mf, fixed=mq,
    )
    (out_dir / "quant_report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "summary.txt").write_text(summary + "\n")
    if WRITE_MEM:
        shutil.copy2(out_dir / "quant_report.json", mem_dir / "quant_report.json")
        shutil.copy2(Path(__file__), out_dir / Path(__file__).name)
    else:
        shutil.rmtree(mem_dir, ignore_errors=True)
    print(f"\noutput: {out_dir}" + (f"   .mem: {mem_dir}" if WRITE_MEM else ""))


if __name__ == "__main__":
    main()
