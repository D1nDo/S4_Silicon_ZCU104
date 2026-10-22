#!/usr/bin/env python3
"""
Train_s4d_speech_command.py -- S4D training on Speech Commands, raw
waveform.

It runs without arguments:

    python Train_s4d_speech_command.py

All the knobs are constants in the CONFIGURATION block below: to change an
experiment the file is edited, not the command line. This way the shared
code is also the documentation of how a result was obtained -- a command
line never ends up in a repository, and six months later nobody remembers
with which batch size the good run was trained.

8-BIT INPUT QUANTIZATION
With QUANT_UINT8 = True the audio sample is reduced to 256 levels BEFORE
training, exactly as the sCIFAR pixel. It answers a precise question: how
much accuracy it costs to feed the audio through the unsigned 8-bit port that
the encoder of the accelerator already has, instead of widening it to 16 bits
with sign.

The point where the quantization happens is NOT arbitrary. The hardware chain is

    integer sample  ->  encoder that folds (p/255 - MU)/SIGMA into the weights

so the quantization comes BEFORE the normalization, and MU/SIGMA must be
measured on the ALREADY quantized signal. Measuring them on the float and
then quantizing would give a normalization that does not match what the
model sees, and the error would be found only by comparing the logits with
the RTL.

The 8-bit range is calibrated on the PERCENTILES of the training set, not on
the full scale: speech has rare peaks, and mapping +-32768 would waste almost
all the 256 levels on silence. The samples outside the range are clipped, on
purpose -- better to saturate half a percent of peaks than to lose resolution
on everything else.

WHY SPEECH COMMANDS
Sequences of thousands of samples against the 1024 of sCIFAR: it is the regime
where long memory really matters. And above all it is a SIGNAL, not an unrolled
image -- for an accelerator that implements IIR filters the argument "I process
a signal with a bank of biquads" is much more natural than "I classify an image
read in zig-zag".

THE DATA DOWNLOAD BY THEMSELVES
torchaudio.datasets.SPEECHCOMMANDS(download=True). ~2.3 GB, 105k files, 35
classes. The official splits (validation_list.txt, testing_list.txt) are in
the archive and are respected: a random split would make the numbers not
comparable with the literature, and since the same speaker appears in several
files it would also leak information between train and test.

POWER-OF-TWO LENGTHS
The S4D kernel computes the convolution with an FFT on 2L points. The sequence
is padded with zeros up to the next power of two (4000 -> 4096), not truncated
to the previous one: padding adds silence where the signal has already ended,
truncating would throw away the tail of the word.

WHY AMP IS OFF
In half precision the S4D kernel produces nan after a few epochs:
exp(l*dt*A) with l up to L underflows in float16, and (exp(dtA)-1)/A divides
denormalized quantities. The GradScaler does not catch the problem because it
looks at the gradients, while here it is the forward that overflows. Measured:
val 0.8443 at epoch 5, nan at epoch 7. In float32 it does not happen.
"""

import ast
import copy
import json
import math
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
ROOT = "data/speechcommands"                # where torchaudio downloads the archive
CACHE_DIR = "data/sc_cache"                 # decoded int16 .npy

# --- data -------------------------------------------------------------------
DECIMATE = 4        # 16000/4 = 4000 samples -> 4096 with the padding
POW2 = True         # pad up to the next power of two

# --- input quantization -----------------------------------------------------
# True  = 256 levels, as the uint8 pixel of sCIFAR: the RTL encoder is fine
#         as it is, no hardware change
# False = float from the int16 sample: a SIGNED 16-bit encoder is needed
QUANT_UINT8 = True
Q8_PCT = (0.5, 99.5)    # training percentiles onto which the 256 levels are mapped
Q8_CAL_N = 3000         # examples used to measure the percentiles

# --- model: the defaults keep the compatibility with the RTL (H=128, MODES=32)
D_MODEL = 128
N_LAYERS = 6
D_STATE = 64
DROPOUT = 0.1

# --- optimization -----------------------------------------------------------
LR = 1e-2           # high: the cosine brings it down by itself
LR_SSM = 1e-3       # S4D kernel group, weight decay 0
WD = 0.05
BATCH = 32
EPOCHS = 40
WARMUP_EPOCHS = 2
PATIENCE = 10
WORKERS = 4
AMP = False         # see the note in the docstring: in half precision the kernel gives nan
SEED = 1337

# --- diagnostics ------------------------------------------------------------
PRINT_SUMMARY = True    # architecture and parameter count at startup
SUMMARY_ONLY = False    # True: print the summary and exit without training
SUBSET = None           # int for a quick test, None for the whole dataset

# --- dataset constants ------------------------------------------------------
SR = 16000          # sampling frequency of Speech Commands
L_FULL = 16000      # one second; shorter files are padded with zeros


# ============================================================================
# project modules
# ============================================================================
def load_modules(mono_path=MONO):
    """gen_s4d_mem / s4d_export / s4d_bn, from the separate files or from the monolith.

    The extraction with ast copies the strings verbatim: those three modules
    contain the fixed-point formats, the Jury clamp and the packing of the .mem
    files, i.e. the contract with the RTL.
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


# ============================================================================
# data
# ============================================================================
def build_cache(root=ROOT, cache_dir=CACHE_DIR):
    """Decodes the WAVs once into int16 .npy + labels.

    int16 is kept and NOT uint8: the 8-bit quantization depends on DECIMATE
    and on the percentiles, and redoing it from the cache takes seconds while
    rebuilding the cache takes ten minutes. This way the same file serves both
    values of QUANT_UINT8 and the two experiments are comparable on the same data.
    """
    import torchaudio

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)     # torchaudio does not create it by itself
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):
        def __init__(self, subset):
            super().__init__(str(root), download=True)

            def load_list(fn):
                p = Path(self._path) / fn
                return set((Path(self._path) / line.strip()).as_posix()
                           for line in p.read_text().splitlines() if line.strip())

            if subset == "validation":
                keep = load_list("validation_list.txt")
                self._walker = [w for w in self._walker if Path(w).as_posix() in keep]
            elif subset == "testing":
                keep = load_list("testing_list.txt")
                self._walker = [w for w in self._walker if Path(w).as_posix() in keep]
            else:
                excl = load_list("validation_list.txt") | load_list("testing_list.txt")
                self._walker = [w for w in self._walker if Path(w).as_posix() not in excl]

    lab_file = cache_dir / "sc_labels.json"
    labels = json.loads(lab_file.read_text()) if lab_file.is_file() else None

    for split in ("training", "validation", "testing"):
        fx = cache_dir / f"sc_{split}_x.npy"
        fy = cache_dir / f"sc_{split}_y.npy"
        if fx.is_file() and fy.is_file():
            continue
        print(f"  decoding {split} ...")
        ds = SubsetSC(split)
        if labels is None:
            labels = sorted({Path(w).parent.name for w in ds._walker})
            lab_file.write_text(json.dumps(labels))
        lab2i = {l: i for i, l in enumerate(labels)}

        n = len(ds)
        X = np.zeros((n, L_FULL), dtype=np.int16)
        Y = np.zeros(n, dtype=np.int64)
        t0 = time.time()
        for i in range(n):
            wav, sr, label, *_ = ds[i]
            assert sr == SR, f"unexpected frequency {sr}"
            w = wav.squeeze(0).numpy()
            # files shorter than one second exist: they are padded with zeros AT
            # THE END, not in the middle, so the onset of the word stays at t=0
            k = min(len(w), L_FULL)
            X[i, :k] = np.clip(w[:k] * 32767.0, -32768, 32767).astype(np.int16)
            Y[i] = lab2i[label]
            if (i + 1) % 20000 == 0:
                print(f"    {i+1}/{n}  ({time.time()-t0:.0f}s)")
        np.save(fx, X); np.save(fy, Y)
        print(f"    {split}: {n} examples")
    return labels


def to_uint8(w_int16, q_lo, q_hi):
    """int16 -> integers 0..255, linear mapping onto [q_lo, q_hi].

    It is exactly what the accelerator will receive on its input port: the
    function returns the integer CODE, not a rescaled float, because it is the
    code that must then be compared with what the RTL reads.

    The clip outside the range is on purpose: with the 0.5/99.5 percentiles half
    a percent of the samples saturates, in exchange for two good bits of
    resolution on everything else. Mapping +-32768 would put almost all the
    signal into a handful of levels around zero.
    """
    span = max(float(q_hi) - float(q_lo), 1.0)
    p = np.round((w_int16 - q_lo) / span * 255.0)
    return np.clip(p, 0, 255)


class SCDataset(torch.utils.data.Dataset):
    """Reads the cache with mmap. Decimates, quantizes, normalizes and pads on the fly.

    ORDER OF THE OPERATIONS, which is not negotiable:
        decimation -> 8-bit quantization -> normalization -> padding
    It is the hardware chain: the sample arrives already quantized, and
    the encoder folds (p/255 - MU)/SIGMA into its own weights. Normalizing
    before quantizing would give numbers different from what the datapath sees.

    The padding comes AFTER the normalization, so the tail is exactly zero and
    not (0-MU)/SIGMA, which would be a spurious constant at the end of every sequence.

    The decimation is a plain subsampling (one every q), without anti-aliasing
    filter. It is also what the S4 reference does, and it is acceptable
    because most of the speech energy is below 4 kHz -- but it must be stated
    in the paper, because formally it introduces aliasing on the fricatives.
    """

    def __init__(self, x_path, y_path, mu, sigma, q_lo=None, q_hi=None,
                 decimate=DECIMATE, pow2=POW2, quant=QUANT_UINT8):
        self.X = np.load(x_path, mmap_mode="r")
        self.Y = np.load(y_path)
        self.mu, self.sigma, self.q = mu, sigma, decimate
        self.quant, self.q_lo, self.q_hi = quant, q_lo, q_hi
        L = L_FULL // decimate
        # Rounding UP: silence is added at the end instead of cutting
        # the end of the word.
        self.L = (1 << int(math.ceil(math.log2(L)))) if pow2 else L

    def _prep(self, i):
        """Decimation + quantization + normalization, without padding.

        Factored out because calibrate_stats() uses it too: measuring MU and SIGMA
        with code different from the one that generates the samples is the
        classic way to make them diverge at the first change.
        """
        d = np.asarray(self.X[i][:: self.q], dtype=np.float32)
        if self.quant:
            p = to_uint8(d, self.q_lo, self.q_hi)      # 0..255, as a pixel
            w = p / 255.0
        else:
            w = d / 32768.0
        return (w - self.mu) / self.sigma

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        w = self._prep(i)
        if len(w) < self.L:
            w = np.pad(w, (0, self.L - len(w)))
        return torch.from_numpy(w[: self.L].copy()).unsqueeze(-1), int(self.Y[i])


def calibrate_stats(x_path, decimate=DECIMATE, quant=QUANT_UINT8,
                    n=Q8_CAL_N, pct=Q8_PCT):
    """Percentiles for the 8 bits and then MU/SIGMA, measured on the TRAINING set.

    The order matters: first the percentiles on the decimated signal, then the
    quantization, then MU and SIGMA on the quantized result. MU and SIGMA
    measured on the float and applied to a quantized signal do not describe
    what the model sees.

    Everything on the training set: calibrating on validation or test would let
    information from there leak in through the two normalization scalars.
    """
    X = np.load(x_path, mmap_mode="r")
    n = min(n, X.shape[0])
    d = np.asarray(X[:n], dtype=np.float32)[:, ::decimate]
    if quant:
        q_lo, q_hi = [float(v) for v in np.percentile(d, list(pct))]
        w = to_uint8(d, q_lo, q_hi) / 255.0
        sat = float((np.abs(d - np.clip(d, q_lo, q_hi)) > 0).mean())
        print(f"  linear 8 bits on [{q_lo:+.0f}, {q_hi:+.0f}] "
              f"(percentiles {pct[0]}/{pct[1]} of the int16), "
              f"clipped samples {100*sat:.2f}%")
        print(f"  quantization step {(q_hi-q_lo)/255:.1f} int16 LSB "
              f"-> theoretical SNR ~{6.02*8:.0f} dB on the chosen range")
    else:
        q_lo = q_hi = None
        w = d / 32768.0
    return float(w.mean()), float(w.std()), q_lo, q_hi


def build_loaders():
    labels = build_cache()
    cd = Path(CACHE_DIR)

    print("calibration on the training set:")
    MU, SIGMA, Q_LO, Q_HI = calibrate_stats(cd / "sc_training_x.npy")
    print(f"  MU {MU:+.5f}  SIGMA {SIGMA:.5f}"
          + ("   (p/255 scale, as sCIFAR)" if QUANT_UINT8
             else "   (p/32768 scale, signed int16)"))

    L = None

    def mk(split, shuffle):
        nonlocal L
        ds = SCDataset(cd / f"sc_{split}_x.npy", cd / f"sc_{split}_y.npy",
                       MU, SIGMA, Q_LO, Q_HI)
        L = ds.L
        if SUBSET:
            n = min(SUBSET if split == "training" else max(SUBSET // 8, 1), len(ds))
            ds = torch.utils.data.Subset(ds, range(n))
        return torch.utils.data.DataLoader(
            ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
            num_workers=WORKERS, pin_memory=True)

    tr, va, te = mk("training", True), mk("validation", False), mk("testing", False)
    raw = L_FULL // DECIMATE
    print(f"train {len(tr.dataset)}  val {len(va.dataset)}  test {len(te.dataset)}   "
          f"classes {len(labels)}")
    print(f"sequence length {L}" + (f"  (from {raw}, padded with {L-raw} zeros "
          f"to get a power of two)" if L != raw else ""))
    print(f"input: {'uint8 0..255' if QUANT_UINT8 else 'signed int16'}")
    return (tr, va, te), MU, SIGMA, Q_LO, Q_HI, labels, L


# ============================================================================
# scheduler
# ============================================================================
def cosine_with_warmup(opt, warmup_steps, total_steps, min_ratio=0.0):
    """Linear warmup then cosine, per STEP.

    The warmup is needed because lr 1e-2 at the first step, with the second
    moment estimates of AdamW still empty, is the fastest way to make the
    model diverge.
    """
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ============================================================================
# training
# ============================================================================
def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in loader:
            ok += (model(x.to(device)).argmax(-1).cpu() == y).sum().item()
            tot += y.numel()
    return ok / max(tot, 1)


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
    """The S4D kernel out of the weight decay.

    On Speech Commands it matters even more than on sCIFAR: log_dt sets the
    discretization step, and with thousands of samples the slow poles are
    exactly what carries the information from one end of the sequence to the
    other -- decaying log_dt shortens them. log_A_real -> 0 instead pushes the
    poles towards the unit circle, i.e. it acts on rho and on how much the
    Jury clamp works at export time.
    """
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
    # A filter that finds nothing is silent: without these asserts, a rename
    # of the submodules restarts a training identical to the previous one.
    assert len(ssm) == 3 * N_LAYERS, f"SSM filter: {len(ssm)}, {3*N_LAYERS} expected"
    assert nod, "empty no_decay filter: BatchNormSeq names changed?"
    return dec, nod, ssm


def train(model, loaders, dev, ckpt_dir, log_path, ckpt_name):
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
    crit = nn.CrossEntropyLoss()
    use_amp = AMP and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_sd, bad = 0.0, copy.deepcopy(model.state_dict()), 0
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; t0 = time.time(); nan_hit = False
        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(x), y)
            # A nan here is not an exploding gradient, it is the forward that
            # overflows: the kernel computes exp(l*dt*A) over L steps. The
            # GradScaler does not catch it, and without this check the run goes
            # on for epochs printing nan and wasting hours.
            if not torch.isfinite(loss):
                nan_hit = True
                break
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            tot += loss.item()
        if nan_hit:
            print(f"non-finite loss at epoch {ep+1}: stopped. "
                  f"If AMP is active, turn it off; otherwise lower LR. "
                  f"The best model stays in {ckpt_dir/ckpt_name}")
            break

        acc = evaluate(model, va, dev)
        dt = time.time() - t0
        avg = tot / spe
        print(f"epoch {ep+1}/{EPOCHS}  loss {avg:.4f}  val {acc:.4f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}  ({dt:.0f}s)")
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(epoch=ep + 1, loss=avg, val_acc=acc,
                                    lr=opt.param_groups[0]["lr"],
                                    sec=round(dt, 1))) + "\n")

        if acc > best:
            best, best_sd, bad = acc, copy.deepcopy(model.state_dict()), 0
            torch.save(best_sd, ckpt_dir / ckpt_name)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping (best val {best:.4f})")
                break
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, best=best,
                        patience=bad), ckpt_dir / ckpt_name.replace("best", "last"))

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
        print(f"WARNING: D_MODEL={D_MODEL} but the RTL has H={G.H}: the model "
              f"trains but is NOT exportable to this accelerator.")
    if D_STATE // 2 != G.MODES:
        print(f"WARNING: D_STATE/2={D_STATE//2} but the RTL has MODES={G.MODES}.")

    ckpt = Path("ckpt"); ckpt.mkdir(exist_ok=True)
    logs = Path("logs"); logs.mkdir(exist_ok=True)
    # distinct names for the two experiments: overwriting the checkpoint of one
    # with the other would make the comparison impossible to redo
    tag = "u8" if QUANT_UINT8 else "i16"

    loaders, MU, SIGMA, Q_LO, Q_HI, labels, L = build_loaders()

    model = B.torch_model(d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE,
                          dropout=DROPOUT, n_classes=len(labels)).to(dev)

    n_par = print_summary(model) if PRINT_SUMMARY else None
    if SUMMARY_ONLY:
        print("SUMMARY_ONLY active: no training.")
        return

    model, best_val = train(model, loaders, dev, ckpt,
                            logs / f"speechcommands_{tag}.jsonl",
                            f"sc_{tag}_best.pt")
    acc_test = evaluate(model, loaders[2], dev)
    print(f"\ntest accuracy = {acc_test:.4f}   (best val {best_val:.4f})   "
          f"input {'uint8' if QUANT_UINT8 else 'int16'}")

    Path(f"speechcommands_report_{tag}.json").write_text(json.dumps(dict(
        task="speechcommands35_raw", input_quant="uint8" if QUANT_UINT8 else "int16",
        q8_lo=Q_LO, q8_hi=Q_HI, q8_pct=list(Q8_PCT),
        decimate=DECIMATE, seq_len=L, sample_rate_eff=SR // DECIMATE,
        acc_test=acc_test, best_val=best_val, n_classes=len(labels),
        n_params=n_par, mu=MU, sigma=SIGMA, seed=SEED,
        d_model=D_MODEL, n_layers=N_LAYERS, d_state=D_STATE, dropout=DROPOUT,
        lr=LR, lr_ssm=LR_SSM, wd=WD, batch=BATCH, epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS, amp=AMP), indent=2))

    print("\n--- notes for the export to the accelerator ---")
    print(f"1. K_CLS in s4d_export.py: {len(labels)} instead of 10")
    print(f"2. MU {MU:+.5f} / SIGMA {SIGMA:.5f}, not the CIFAR ones")
    if QUANT_UINT8:
        print("3. uint8 input 0..255: quantize_encoder is used AS IT IS, the")
        print(f"   factor 255 is correct. In hardware the input chain must")
        print(f"   map int16 -> uint8 on [{Q_LO:+.0f}, {Q_HI:+.0f}] before")
        print("   the encoder, which is a subtraction and a shift.")
    else:
        print("3. SIGNED int16 input. quantize_encoder folds")
        print("   (p/255 - MU)/SIGMA assuming an unsigned 8-bit integer:")
        print("   here the factor is 32768 and the sample is signed, so the input")
        print("   port of the encoder and the product w'*p change, which")
        print("   becomes signed x signed. It is an RTL change, not a generic.")
    print(f"4. Sequence of {L} steps against the 1024 of sCIFAR: LOG2_L goes from 10")
    print(f"   to {int(math.log2(L))} and the pooling accumulator sums {L} addends.")
    print(f"   At ~512 cycles per sample that is {512*L/1e6:.1f} M cycles per word,")
    print(f"   {512*L/1e8*1000:.0f} ms at 100 MHz.")


if __name__ == "__main__":
    main()
