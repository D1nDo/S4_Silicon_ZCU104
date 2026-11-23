#!/usr/bin/env python3
"""Fine-tuning con clamp +-16 sulle norm (come il paper per Speech Commands) del modello scelto.

Riusa Finetune_clamp_s4d_speech_command_l1024.py del paper senza modificarlo: cambiano dall'esterno
solo D_MODEL / D_STATE, il checkpoint di partenza (best.pt del run dello studio) e la cartella d'uscita.
Ricetta del fine-tuning invariata (10 epoche, lr 1e-3, kernel 1e-4, warm-up 1).
uso: python finetune_sc.py --H 64 --N 32
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_sweep import patched_loader, ROOT, STUDY

ap = argparse.ArgumentParser(); ap.add_argument("--H", type=int, required=True); ap.add_argument("--N", type=int, required=True)
a = ap.parse_args()
exp = ROOT
os.chdir(ROOT); sys.path.insert(0, str(exp)); sys.path.insert(0, str(ROOT))
import Finetune_clamp_s4d_speech_command_l1024 as F
F.T.load_modules = patched_loader(a.H, a.N)
F.T.D_MODEL, F.T.D_STATE = a.H, a.N
F.START_CKPT = STUDY / "runs" / f"sc_H{a.H}_N{a.N}_s0" / "best.pt"
F.OUT_DIR = STUDY / "quant" / f"sc_H{a.H}_N{a.N}_s0" / "finetune_clamp16"
F.main()
