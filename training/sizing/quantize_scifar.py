#!/usr/bin/env python3
"""Quantizzazione sCIFAR del modello scelto (H = 128, N = 64, seed 0 dello studio).

Lo script del paper (Train_and_quantize_s4d_sCIFAR.py) e' un file piatto che addestra e poi quantizza.
Qui se ne scrive una COPIA con tre sostituzioni testuali, senza toccare l'originale:
  - MAX_EPOCHS = 0          -> niente training, il ciclo non parte;
  - WORK = cartella dello studio  -> mem/, ckpt/ e log non toccano training/;
  - BEST_CKPT = best.pt del run dello studio.
La configurazione e' quella del paper (H = 128, N = 64), quindi i moduli restano identici.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # training/
STUDY = Path(__file__).resolve().parent                # training/sizing/
run = STUDY / "runs" / "scifar_H128_N64_s0"
out = STUDY / "quant" / "scifar_H128_N64_s0" / "paper"
out.mkdir(parents=True, exist_ok=True)
src = (ROOT / "Train_and_quantize_s4d_sCIFAR.py").read_text()
subs = [("MAX_EPOCHS   = 100", "MAX_EPOCHS   = 0"),
        ("REPO = WORK = Path(__file__).resolve().parent",
         f'REPO = Path("{ROOT}"); WORK = Path("{out}")'),
        ('BEST_CKPT = CKPT_DIR / "s4d_scifar_configC_bn_best.pt"', f'BEST_CKPT = Path("{run / "best.pt"}")')]
subs.append(('assert d < 256 * ulp, "la riscalatura ha cambiato la funzione"',
             'print(f"controllo riscalatura: scarto {d:.3e} = {d / ulp:.0f} ulp (soglia 256)")\n'
             'assert d < 256 * ulp, "la riscalatura ha cambiato la funzione"'))
# riferimento float in fp32 pieno, come RadioML (torch abilita TF32 nelle convoluzioni cuDNN per default)
subs.append(('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"',
             'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
             'torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False'))
for a, b in subs:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
dst = out / "Train_and_quantize_s4d_sCIFAR_studio.py"
dst.write_text(src)
sys.exit(subprocess.call([sys.executable, str(dst)], cwd=str(ROOT)))
