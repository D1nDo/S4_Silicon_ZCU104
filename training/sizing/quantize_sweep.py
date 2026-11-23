#!/usr/bin/env python3
"""Quantizzazione dei modelli scelti nello studio di dimensionamento (H, N diversi da 128/64).

Riusa SENZA MODIFICARLI gli script di quantizzazione del paper in training/.
Cambiano solo due cose, entrambe dall'esterno:
  1. i moduli gen_s4d_mem / s4d_export / s4d_bn vengono estratti dal monolitico come fa
     load_modules() e la sola riga "H, MODES, NUNITS, NCH = 128, 32, 8, 16" viene sostituita
     con H, MODES = N/2, NCH = 16 e NUNITS = H/16. NMAC resta 64, con NPH = 2H/64 fasi;
  2. checkpoint, cartella d'uscita e D_MODEL / D_STATE puntano al run dello studio.
Generic liberi (HW_LOCK = False): i generic del bitstream del paper valgono solo per H = 128.
I .mem sono generati dalla stessa catena, ma NON sono validati contro un RTL con questo H.

uso: python quantize_sweep.py --task ptbxl --H 32 --N 16 [--seed 0]
"""
import argparse, ast, json, shutil, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # training/
STUDY = Path(__file__).resolve().parent                # training/sizing/: runs/ e quant/
MONO = ROOT / "Train_and_quantize_s4d_sCIFAR.py"
ORIG_LINE = "H, MODES, NUNITS, NCH = 128, 32, 8, 16"


def patched_loader(H, N):
    def load_modules(mono_path=MONO):
        targets = {"_GEN_S4D_MEM_SRC": "gen_s4d_mem", "_S4D_EXPORT_SRC": "s4d_export",
                   "_S4D_BN_SRC": "s4d_bn"}
        src = {}
        for node in ast.walk(ast.parse(Path(mono_path).read_text())):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in targets:
                        src[targets[t.id]] = node.value.value
        assert src["gen_s4d_mem"].count(ORIG_LINE) == 1, "riga delle dimensioni non trovata"
        nch = 16
        assert H % nch == 0 and (2 * H) % 64 == 0 and N % 2 == 0
        src["gen_s4d_mem"] = src["gen_s4d_mem"].replace(
            ORIG_LINE, f"H, MODES, NUNITS, NCH = {H}, {N // 2}, {H // nch}, {nch}")
        mods = {}
        for name in ("gen_s4d_mem", "s4d_export", "s4d_bn"):
            m = types.ModuleType(name)
            sys.modules[name] = m
            exec(compile(src[name], name, "exec"), m.__dict__)
            mods[name] = m
        G = mods["gen_s4d_mem"]
        assert (G.H, G.MODES) == (H, N // 2)
        return G, mods["s4d_export"], mods["s4d_bn"]
    return load_modules


def run_ptbxl(H, N, ckpt, out, mode):
    """mode 'hw': generic del bitstream ECG del paper (default dello script); 'free': calibrati."""
    import Quantize_s4d_ecg as Q
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text('{"norm_mode": "fixed", "label_mode": "multi"}')   # cache PTB-XL del training
    Q.T.load_modules = patched_loader(H, N)
    Q.T.D_MODEL, Q.T.D_STATE = H, N
    Q.EXP, Q.CKPT, Q.HW_LOCK = out, ckpt, (mode == "hw")
    Q.main()


def _skip_streams(mem_dir):
    (mem_dir / "STREAM_NON_GENERATO.txt").write_text(
        "Lo stream dei pesi (Export_s4d_wstream) e' impacchettato per H = 128 (4 fasi da 64 MAC):\n"
        "non generato per questa configurazione. I .mem per layer sono quelli della catena.\n")
    print("stream dei pesi: non generato (impacchettamento specifico di H = 128)")


def run_sc(H, N, ckpt, out, mode):
    """mode 'core': F_SSM 9 / F_D 13 imposti (default dello script); 'free': calibrati.
    Il modello dello studio non ha il fine-tuning con clamp +-16 delle norm del paper."""
    exp = ROOT
    sys.path.insert(0, str(exp))
    import Quantize_s4d_speech_command_l1024 as S
    S.T.load_modules = patched_loader(H, N)
    S.T.D_MODEL, S.T.D_STATE, S.H = H, N, H
    ft = out.parent / "finetune_clamp16" / "ckpt" / "sc12k_best.pt"
    if mode.endswith("_ft"):          # modello dopo il fine-tuning con clamp +-16 (come nel paper)
        assert ft.is_file(), f"{ft}: lanciare prima finetune_sc.py"
        ckpt = ft
    S.CKPT, S.CLAMP_NORMS, S.CORE_LOCK = ckpt, mode.endswith("_ft"), mode.startswith("core")
    S.EXPERIMENT_DIR = out
    S.export_weight_streams = _skip_streams
    out.mkdir(parents=True, exist_ok=True)
    S.main()


def run_radio(H, N, ckpt, out, mode):
    """Lo script del paper impone sempre F_SSM 9 / F_D 13 (core) con riscalatura per canale."""
    assert mode == "core", "lo script RadioML del paper ha solo la modalita' core"
    exp = ROOT
    sys.path.insert(0, str(exp))
    import Quantize_S4D_RadioML as R
    R.T.load_modules = patched_loader(H, N)
    R.T.D_MODEL, R.T.D_STATE, R.H = H, N, H
    R.CKPT, R.EXPERIMENT_DIR = ckpt, out
    R.export_weight_streams = _skip_streams
    out.mkdir(parents=True, exist_ok=True)
    R.main()


RUNNERS = dict(ptbxl=run_ptbxl, sc=run_sc, radio=run_radio)
DEFAULT_MODE = dict(ptbxl="hw", sc="core", radio="core")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default=None, help="hw/free (ptbxl), core/free (sc), core (radio)")
    a = ap.parse_args()
    mode = a.mode or DEFAULT_MODE[a.task]
    run = STUDY / "runs" / f"{a.task}_H{a.H}_N{a.N}_s{a.seed}"
    ckpt = run / "best.pt"
    assert (run / "done.json").is_file(), f"{run}: run non finito"
    out = STUDY / "quant" / f"{a.task}_H{a.H}_N{a.N}_s{a.seed}" / mode
    import os
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    RUNNERS[a.task](a.H, a.N, ckpt, out, mode)


if __name__ == "__main__":
    main()
