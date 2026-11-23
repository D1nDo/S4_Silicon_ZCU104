"""Training float S4D per lo studio di dimensionamento (H, N).

Riusa dati, preprocessing e modello degli script del paper in training/:
  ptbxl  Train_s4d_ecg.py
  radio  Train_S4D_RadioML.py
  sc     Train_s4d_speech_command_l1024.py
  scifar Train_and_quantize_s4d_sCIFAR.py (modulo s4d_bn + caricamento CIFAR del paper)
Il modello e' B.torch_model(d_model=H, d_state=N) con l'encoder del task: unica
modifica rispetto al paper. Ricetta (Tab. I) letta dalle costanti degli script.
Split e sottoinsiemi di validazione restano quelli del paper (seed dati 1337);
il seed del run governa solo inizializzazione, ordine dei batch e augmentation.

uso: python train_sweep.py --task ptbxl --H 128 --N 64 --seed 0 --out RUNDIR
     [--max-epochs K]  (calibrazione: K epoche, niente test)
"""
import argparse, copy, importlib.util, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent          # training/ (paper scripts and data/)
SCRIPTS = dict(
    ptbxl=ROOT / "Train_s4d_ecg.py",
    radio=ROOT / "Train_S4D_RadioML.py",
    sc=ROOT / "Train_s4d_speech_command_l1024.py",
)
N_IN = dict(ptbxl=12, radio=2, sc=12, scifar=1)
TRAIN_EVAL_N = 5000
L = 1024


def import_script(task):
    spec = importlib.util.spec_from_file_location(f"paper_{task}", SCRIPTS[task])
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(m)
    return m


def macs_per_inference(task, H, N):
    return L * (N_IN[task] * H + 6 * (H * (2 * N + 1) + 2 * H * H))


# ---------------------------------------------------------------------------
# task: loader, modello, ricetta
# ---------------------------------------------------------------------------
def fixed_subset(ds, n, seed=0):
    if n is None or n >= len(ds):
        return ds
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), n, replace=False))
    return torch.utils.data.Subset(ds, idx.tolist())


def setup(task, H, N, workers_scale=1.0):
    DL = torch.utils.data.DataLoader
    if task == "scifar":
        import torchvision, torchvision.transforms as T
        from torch.utils.data import random_split
        _, _, B = import_script("ptbxl").load_modules(str(ROOT / "Train_and_quantize_s4d_sCIFAR.py"))
        MU, SIGMA = 0.4734, 0.2516                       # come nel paper
        tf = T.Compose([T.Grayscale(1), T.ToTensor(), T.Normalize((MU,), (SIGMA,))])
        full = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True, download=False, transform=tf)
        test = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=False, download=False, transform=tf)
        # stesso transform del paper, applicato una volta sola e messo in cache
        # (nessuna augmentation: il tensore per immagine e' deterministico)
        cache = ROOT / "data" / "scifar_gray_norm.pt"
        if not cache.is_file():
            cache.parent.mkdir(exist_ok=True)
            pack = lambda ds: (torch.stack([ds[i][0].view(-1, 1) for i in range(len(ds))]),
                               torch.tensor(ds.targets))
            torch.save(dict(train=pack(full), test=pack(test)), cache)
        c = torch.load(cache)
        full = torch.utils.data.TensorDataset(*c["train"])
        test = torch.utils.data.TensorDataset(*c["test"])
        n_tr = int(0.9 * len(full))
        tr, va = random_split(full, [n_tr, len(full) - n_tr], generator=torch.Generator().manual_seed(1337))
        mk = lambda ds, sh: DL(ds, batch_size=64, shuffle=sh, drop_last=sh, num_workers=0, pin_memory=True)
        loaders = dict(train=mk(tr, True), val=mk(va, False), test=mk(test, False),
                       train_eval=mk(fixed_subset(tr, TRAIN_EVAL_N), False))
        model = B.torch_model(d_model=H, n_layers=6, d_state=N, dropout=0.2, n_classes=10)
        rec = dict(kind="ce", lr=1e-3, lr_ssm=1e-3, wd=0.01, epochs=100, patience=10,
                   sched="plateau", warmup=0, tf32=False)
        return model, loaders, rec

    M = import_script(task)
    G, E, B = M.load_modules(str(ROOT / "Train_and_quantize_s4d_sCIFAR.py"))
    n_cls = len(M.CLASSES) if task != "sc" else len(json.load(open(ROOT / "data/sc_cache/sc_labels.json")))
    model = B.torch_model(d_model=H, n_layers=M.N_LAYERS, d_state=N, dropout=M.DROPOUT, n_classes=n_cls)
    model.input_proj = nn.Linear(N_IN[task], H)
    rec = dict(lr=M.LR, lr_ssm=M.LR_SSM, wd=M.WD, epochs=M.EPOCHS, patience=M.PATIENCE,
               sched="cosine", warmup=M.WARMUP_EPOCHS, tf32=bool(getattr(M, "ALLOW_TF32", False)))
    if task == "radio":   # decisione utente 2026-11-21: RadioML senza early stopping (30 epoche fisse)
        rec["patience"] = rec["epochs"]
    w = lambda k: max(1, int(k * workers_scale))

    if task == "ptbxl":
        paths, _ = M.build_cache()
        ds = {s: M.ECGDataset(*paths[s]) for s in ("train", "val", "test")}
        mk = lambda d, sh: DL(d, batch_size=M.BATCH, shuffle=sh, drop_last=sh, num_workers=w(4),
                              pin_memory=True, persistent_workers=True)
        loaders = dict(train=mk(ds["train"], True), val=mk(ds["val"], False), test=mk(ds["test"], False),
                       train_eval=mk(ds["train"], False))          # tutto il training
        rec.update(kind="bce", auc=M.auc)
    elif task == "radio":
        cache, _ = M.build_cache()
        trs = M.RadioMLDataset(cache, "train"); vas = M.RadioMLDataset(cache, "val")
        tes = M.RadioMLDataset(cache, "test")
        sel = M.stratified_subset(vas, M.VAL_FRAMES_PER_EPOCH)       # seed 1337+1 come nel paper
        sampler = torch.utils.data.RandomSampler(trs, replacement=False, num_samples=M.FRAMES_PER_EPOCH)
        mk = lambda d, s=None: DL(d, batch_size=M.BATCH, sampler=s, shuffle=False, drop_last=s is not None,
                                  num_workers=w(8), pin_memory=True, persistent_workers=True)
        loaders = dict(train=mk(trs, sampler), val=mk(sel), val_full=mk(vas), test=mk(tes),
                       train_eval=mk(fixed_subset(trs, TRAIN_EVAL_N)))
        rec.update(kind="ce3")
    elif task == "sc":
        def mk_ds(split, aug):
            return M.SC12kFrameDataset(M.CACHE_12K_DIR / f"sc_{split}_x12k.npy",
                                       M.CACHE_16K_DIR / f"sc_{split}_y.npy", augment=aug)
        mk = lambda d, sh: DL(d, batch_size=M.BATCH, shuffle=sh, drop_last=sh, num_workers=w(8),
                              pin_memory=True, worker_init_fn=M.seed_worker, persistent_workers=True)
        loaders = dict(train=mk(mk_ds("training", True), True), val=mk(mk_ds("validation", False), False),
                       test=mk(mk_ds("testing", False), False),
                       train_eval=mk(fixed_subset(mk_ds("training", False), TRAIN_EVAL_N), False))
        rec.update(kind="ce")
    return model, loaders, rec


def param_groups(model):
    ssm_keys = ("kernel.log_A_real", "kernel.A_imag", "kernel.log_dt")
    ssm, nod, dec = [], [], []
    for n_, p_ in model.named_parameters():
        if any(k in n_ for k in ssm_keys):
            ssm.append(p_)
        elif n_.endswith(".bias") or ".bn." in n_ or n_.endswith(".D"):
            nod.append(p_)
        else:
            dec.append(p_)
    assert len(ssm) == 18 and nod
    return dec, nod, ssm


def cosine_with_warmup(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        p = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def unpack(batch, kind):
    if kind == "bce":
        x, ym, _ = batch; return x, ym
    if kind == "ce3":
        x, y, _ = batch; return x, y
    return batch


def loss_fn(kind):
    return nn.BCEWithLogitsLoss() if kind == "bce" else nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader, rec, dev):
    """(metrica, loss media): macro-AUC per PTB-XL, accuracy altrove."""
    model.eval()
    crit = loss_fn(rec["kind"])
    Z, Y, ls, n = [], [], 0.0, 0
    for b in loader:
        x, y = unpack(b, rec["kind"])
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        z = model(x)
        ls += crit(z, y).item() * len(y); n += len(y)
        Z.append(z.float().cpu().numpy()); Y.append(y.cpu().numpy())
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    if rec["kind"] == "bce":
        metric = float(np.nanmean([rec["auc"](Y[:, c], Z[:, c]) for c in range(Y.shape[1])]))
    else:
        metric = float((Z.argmax(-1) == Y).mean())
    return metric, ls / max(n, 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ptbxl", "radio", "sc", "scifar"])
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--workers-scale", type=float, default=1.0)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").is_file():
        print("gia' completato:", out); return
    os.chdir(ROOT)
    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model, loaders, rec = setup(a.task, a.H, a.N, a.workers_scale)
    torch.backends.cuda.matmul.allow_tf32 = rec["tf32"]
    torch.backends.cudnn.allow_tf32 = rec["tf32"]
    model = model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    dec, nod, ssm = param_groups(model)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": rec["wd"]},
                             {"params": nod, "weight_decay": 0.0},
                             {"params": ssm, "weight_decay": 0.0, "lr": rec["lr_ssm"]}], lr=rec["lr"])
    spe = len(loaders["train"])
    epochs = rec["epochs"]
    if rec["sched"] == "cosine":
        sched = cosine_with_warmup(opt, rec["warmup"] * spe, epochs * spe)
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    crit = loss_fn(rec["kind"])
    run_epochs = min(epochs, a.max_epochs) if a.max_epochs else epochs

    log_path, last_path, best_path = out / "log.jsonl", out / "last.pt", out / "best.pt"
    ep0, best, bad, best_ep, diverged = 0, -1.0, 0, 0, False
    if last_path.is_file():                                     # ripresa
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best, bad, best_ep = ck["epoch"] + 1, ck["best"], ck["bad"], ck["best_ep"]
        torch.set_rng_state(ck["rng_cpu"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        np.random.set_state(ck["rng_np"])
        lines = [l for l in log_path.read_text().splitlines() if json.loads(l)["epoch"] <= ep0]
        log_path.write_text("".join(l + "\n" for l in lines))
        print(f"ripresa dall'epoca {ep0 + 1}")
    print(f"{a.task} H={a.H} N={a.N} seed={a.seed} params={n_params} steps/epoca={spe}", flush=True)

    t_start = time.time()
    for ep in range(ep0, run_epochs):
        model.train(); tot = 0.0; t0 = time.time()
        for b in loaders["train"]:
            x, y = unpack(b, rec["kind"])
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            loss = crit(model(x), y)
            if not torch.isfinite(loss):
                diverged = True; break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if rec["sched"] == "cosine":
                sched.step()
            tot += loss.item()
        if diverged:
            print(f"DIVERGENZA: loss non finita all'epoca {ep + 1}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(epoch=ep + 1, diverged=True)) + "\n")
            break
        t_train = time.time() - t0
        va_m, va_l = evaluate(model, loaders["val"], rec, dev)
        tr_m, tr_l = evaluate(model, loaders["train_eval"], rec, dev)
        lr_now = opt.param_groups[0]["lr"]
        if rec["sched"] == "plateau":
            sched.step(va_m); opt.param_groups[2]["lr"] = rec["lr_ssm"]   # SSM a lr fisso, come nel paper
        row = dict(epoch=ep + 1, train_loss=tot / spe, train_eval_metric=tr_m, train_eval_loss=tr_l,
                   val_metric=va_m, val_loss=va_l, lr=lr_now, sec_train=round(t_train, 1),
                   sec_total=round(time.time() - t0, 1))
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"ep {ep + 1}: loss {row['train_loss']:.4f} train {tr_m:.4f} val {va_m:.4f} "
              f"({row['sec_total']:.0f}s)", flush=True)
        if va_m > best:
            best, bad, best_ep = va_m, 0, ep + 1
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        epoch=ep, best=best, bad=bad, best_ep=best_ep, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state(), rng_np=np.random.get_state()),
                   str(last_path) + ".tmp")
        os.replace(str(last_path) + ".tmp", last_path)
        if bad >= rec["patience"]:
            print("early stopping", flush=True); break

    if a.max_epochs:                                            # calibrazione: niente test
        return
    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
    rows = [r for r in rows if not r.get("diverged")]
    done = dict(task=a.task, H=a.H, N=a.N, seed=a.seed, n_params=n_params,
                macs_per_inference=macs_per_inference(a.task, a.H, a.N),
                epochs_run=len(rows), diverged=diverged, best_epoch=best_ep, best_val_metric=best,
                wall_sec=round(time.time() - t_start, 1))
    if best_ep:
        br = rows[best_ep - 1]
        done.update(train_eval_metric_at_best=br["train_eval_metric"],
                    gen_gap=br["train_eval_metric"] - br["val_metric"],
                    val_loss_at_best=br["val_loss"], train_eval_loss_at_best=br["train_eval_loss"])
        model.load_state_dict(torch.load(best_path, map_location=dev, weights_only=True))
        done["test_metric"], done["test_loss"] = evaluate(model, loaders["test"], rec, dev)   # una sola volta
        if "val_full" in loaders:
            done["val_full_metric"], _ = evaluate(model, loaders["val_full"], rec, dev)
    (out / "done.json").write_text(json.dumps(done, indent=2))
    print("fine:", json.dumps({k: done.get(k) for k in ("best_val_metric", "test_metric", "best_epoch")}))


if __name__ == "__main__":
    main()
