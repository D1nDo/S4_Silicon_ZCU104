#!/usr/bin/env python3
"""
Quantize_S4D_RadioML.py -- quantizzazione, export dei .mem e dello stream dei
pesi, verifica bit-accurate del modello RadioML 2018.01A (24 modulazioni,
L = 1024, ingresso I/Q) addestrato da Train_S4D_RadioML.py.

Si lancia senza argomenti, dalla radice del progetto o da qui:

    conda run -n mamba_env --no-capture-output python Quantize_S4D_RadioML.py

E' la catena di Quantize_s4d_speech_command_l1024.py (a sua volta quella di
Quantize_s4d_ecg.py): encoder generico, norm, layer e verifica bit-accurate
parallela vengono importati da Quantize_s4d_ecg, non ricopiati, cosi' lo
specchio dell'RTL resta uno solo. Differenze:

  1. encoder a NIN = 2 ingressi (I, Q) in Q5.11: Quantize_s4d_ecg.NIN = 2.
  2. RISCALATURA PER CANALE dell'uscita delle norm (u), nuova.
  3. verifica bit-accurate su un sottoinsieme stratificato del test
     (stesso numero di frame per ogni coppia modulazione/SNR): il test
     completo (255k frame) richiederebbe ~20 h di CPU.

RISCALATURA PER CANALE (punto 2)
  In hardware u = BN(x) e' in Q5.11 (W_U 16, F_U 11): satura a +-16. Questo
  modello ha picchi di |u| fino a ~800 negli ultimi layer; il solo taglio a
  +-16 costa 3.4 punti a SNR >= 10 dB (misurato in float).
  Il nucleo S4D e' depthwise e lineare in u fino a yssm:
      yssm_h = SSM_h(u_h) + D_h * u_h,     SSM_h lineare in C_h (quindi in b0, b1)
  quindi per ogni canale h e ogni intero k_h la trasformazione
      bn_w_h, bn_b_h  *= 2^-k_h      (u_h piu' piccolo)
      C_h, D_h        *= 2^+k_h      (b0, b1 e D compensano)
  lascia yssm IDENTICO in float. In hardware b ha un esponente per canale
  (fb_ch -> bsh, nello stream), quindi alzare b costa solo bit di esponente.
  Vincoli per canale:
      b : fb_ch - k_h >= F_B           (|b| < 1 dopo la riscalatura)
      D : |D_h| * 2^k_h < 2^(W_D-1-F_D)   (F_D = 13 globale: |D| < 4)
  k_h e' il minimo che porta MARGIN_U * picco(u_h) sotto +-16, limitato dai
  due vincoli. Se |D_h| >= 4 gia' in partenza, k_h diventa NEGATIVO (u_h piu'
  grande, D_h rientra nel formato): D tagliato sarebbe un errore su ogni
  campione, u tagliato solo sui picchi.
  La norm finale NON viene riscalata: la compensazione andrebbe nel
  classificatore INT8 e il suo ingresso ha solo 8 bit. Resta col clamp.

GENERIC (target ZCU104, Desktop/S4D_RTL/new)
  nucleo:   F_SSM 9, F_D 13 imposti (CLAUDE.md)
  encoder:  NIN 2, N_BIT_IN 16, IN_SIGNED 1, F_WENC calibrato
  decoder:  K 24, BN_SH e RQ_SHIFT calibrati
  BN_SH dei layer: nello stream (beat 0 di ogni layer)
  -> scritti in top_generics.txt

USCITE, in una cartella nuova quant_<data>_core/ accanto a questo script:
    mem/                                  .mem per layer + encoder + decoder + LUT
    mem/w_stream_RadioML.mem/.bin          stream Zynq-7020 (mixing 512 x 12 beat)
    mem/w_stream_RadioML_nmac256.mem/.bin  stream ZCU104   (mixing 128 x 48 beat)
    quant_report.json, summary.txt, top_generics.txt, channel_shift.json

Variabili d'ambiente per le prove:
    RML_RESCALE=0     niente riscalatura per canale (riferimento: solo clamp)
    RML_N_EVAL=624    frame di test per la verifica (default 9984 = 16 per cella)
"""

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR                                 # training/ (all the scripts are in the same folder)
sys.path.insert(0, str(PROJECT_ROOT))           # Quantize_s4d_ecg, Export_s4d_wstream
sys.path.insert(0, str(EXPERIMENT_DIR))         # la copia dello script di training usata dal run
os.chdir(PROJECT_ROOT)                          # i moduli importati usano percorsi relativi alla radice

import Train_S4D_RadioML as T                   # noqa: E402  dati e modello, come nel run
import Quantize_s4d_ecg as Q                    # noqa: E402  encoder generico + verifica bit-accurate
import Export_s4d_wstream as XS                 # noqa: E402  impacchettamento dello stream

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
CKPT = EXPERIMENT_DIR / "best.pt"
RESCALE_CHANNELS = os.environ.get("RML_RESCALE", "1") == "1"
CORE_GENERICS = dict(F_SSM=9, F_D=13)           # valori protetti di CLAUDE.md (ZCU104)
FILL = 0.75             # frazione del tetto occupata dopo la riscalatura globale (come SC/ECG)
MARGIN_U = 2.0          # picco di u sulla calibrazione x2: il test puo' andare oltre
N_CAL = 10_000          # frame di validazione per la calibrazione (stratificati)
N_EVAL_FX = int(os.environ.get("RML_N_EVAL", "9984"))   # 9984 = 24 classi x 26 SNR x 16
CHUNK_FX = 32           # frame per blocco nella verifica bit-accurate
WORKERS_FX = 56         # processi paralleli (come SC: limite di memoria, 32 core fisici)
BATCH = 256
SEED = 1337

N_LAYERS = T.N_LAYERS
H = T.D_MODEL
N_PHASES = 4                                    # mixing Zynq-7020: 4 fasi da 64 MAC
STREAM_BEATS_Z7020 = 12                         # beat per riga di mixing nello stream Zynq-7020
U_CEIL = 2.0 ** (16 - 1 - 11)                   # W_U 16, F_U 11 -> +-16
CLASSES = T.CLASSES


# ============================================================================
# dati
# ============================================================================
class XYOnly(torch.utils.data.Dataset):
    """calibrate_ranges legge coppie (x, y): il dataset RadioML restituisce
    anche l'SNR, che qui si scarta."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y, _snr = self.base[i]
        return x, y


def stratified_positions(dataset, n_target, seed=SEED):
    """Posizioni (nel dataset) di un sottoinsieme con lo stesso numero di
    frame per ogni (classe, SNR): stessa regola di T.stratified_subset."""
    subset = T.stratified_subset(dataset, n_target, seed=seed)
    return np.asarray(subset.indices, dtype=np.int64)


class UClampHook:
    """Taglia l'uscita di una norm a +-16 come la saturazione di u in hardware.
    Serve al riferimento float 'con i vincoli', non al modello libero."""

    def __call__(self, module, inputs, output):
        return output.clamp(-U_CEIL, U_CEIL)


class UPeakHook:
    """Registra il picco di |u| per canale all'uscita di una norm."""

    def __init__(self):
        self.peak = torch.zeros(H)

    def __call__(self, module, inputs, output):
        self.peak = torch.maximum(self.peak, output.detach().abs().amax(dim=(0, 1)).cpu())


def add_u_clamps(model):
    """Clamp a +-16 sulle 7 norm (6 layer + finale): il modello float vede gli
    stessi tetti dell'hardware su u."""
    clamp_hook = UClampHook()
    for norm in list(model.norms) + [model.norm_f]:
        norm.register_forward_hook(clamp_hook)
    return model


# ============================================================================
# riscalatura per canale
# ============================================================================
def measure_u_peaks(model, loader, device):
    """Picco di |u| per canale e per layer (6 norm dei layer) sulla calibrazione."""
    peak_hooks = [UPeakHook() for _ in range(N_LAYERS)]
    handles = [norm.register_forward_hook(hook) for norm, hook in zip(model.norms, peak_hooks)]
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(device))
    for handle in handles:
        handle.remove()
    return [hook.peak.numpy().astype(np.float64) for hook in peak_hooks]


def choose_channel_shifts(G, W, u_peaks):
    """k_h per ogni layer e canale, con le stesse formule di quantize_layer.

    Restituisce (lista di array k per layer, diagnostica per layer).
    """
    d_ceiling = 2.0 ** (G.W_D - 1 - CORE_GENERICS["F_D"])     # |D| < 4 con F_D 13
    shifts, diagnostics = [], []
    for layer_idx, layer_w in enumerate(W["layers"]):
        # margine di b: stesso calcolo di fb_ch in s4d_export.quantize_layer
        _, _, b0f, b1f, _ = G.to_biquad(layer_w["log_A_real"], layer_w["A_imag"],
                                        layer_w["log_dt"], layer_w["C"])
        bmax = np.maximum(np.abs(b0f), np.abs(b1f)).max(axis=1)
        fb_ch = np.floor(np.log2(((1 << (G.W_B - 1)) - 1) / np.maximum(bmax, 1e-30))).astype(int)
        headroom_b = np.clip(fb_ch, G.F_B, G.F_B + 15) - G.F_B         # >= 0 sempre
        # margine di D: k massimo con |D| * 2^k < tetto (negativo se |D| >= 4 gia' oggi)
        abs_d = np.abs(np.asarray(layer_w["D"], dtype=np.float64))
        headroom_d = np.floor(np.log2(d_ceiling / np.maximum(abs_d, 1e-30)) - 1e-9).astype(int)
        # k minimo che porta MARGIN_U * picco sotto il tetto di u (puo' essere <= 0)
        peak = u_peaks[layer_idx]
        k_for_u = np.ceil(np.log2(np.maximum(MARGIN_U * peak / U_CEIL, 1e-30))).astype(int)
        k_max = np.minimum(headroom_b, headroom_d)                    # tetto imposto da b e D
        # non si riscala chi non ne ha bisogno (k >= 0), salvo D gia' fuori formato
        k_used = np.minimum(np.maximum(k_for_u, 0), k_max)
        residual_peak = peak / 2.0 ** k_used
        shifts.append(k_used)
        diagnostics.append(dict(
            layer=layer_idx,
            channels_over_ceiling=int((peak > U_CEIL).sum()),
            k_min=int(k_used.min()), k_max=int(k_used.max()),
            k_histogram={int(k): int((k_used == k).sum()) for k in np.unique(k_used)},
            limited_by_b=int((k_for_u > headroom_b).sum()),
            limited_by_d=int((k_for_u > headroom_d).sum()),
            d_over_ceiling_before=int((abs_d >= d_ceiling).sum()),
            channels_still_over_ceiling=int((residual_peak > U_CEIL).sum()),
            peak_before=float(peak.max()), residual_peak=float(residual_peak.max())))
    return shifts, diagnostics


def apply_channel_shifts(W, shifts):
    """u_h -> u_h * 2^-k, C_h e D_h -> * 2^k: la funzione float non cambia.
    Potenze di due: si sposta solo l'esponente, nessun errore aggiunto."""
    out = {key: value for key, value in W.items() if key != "layers"}
    out["layers"] = []
    for layer_w, k_layer in zip(W["layers"], shifts):
        factor = 2.0 ** k_layer.astype(np.float64)                    # (H,)
        new_w = dict(layer_w)
        new_w["bn_w"] = np.asarray(layer_w["bn_w"], dtype=np.float64) / factor
        new_w["bn_b"] = np.asarray(layer_w["bn_b"], dtype=np.float64) / factor
        new_w["C"] = np.asarray(layer_w["C"]) * factor[:, None]       # C (H, N) complesso
        new_w["D"] = np.asarray(layer_w["D"], dtype=np.float64) * factor
        out["layers"].append(new_w)
    return out


# ============================================================================
# stream dei pesi: Zynq-7020 (4 fasi) e ZCU104 (riga unica)
# ============================================================================
def export_weight_streams(mem_dir):
    """Stessa funzione di Quantize_s4d_speech_command_l1024.py, nomi RadioML.

    ZCU104: permutazione pura della sola sezione di mixing (CLAUDE.md 1.0):
        nuovo (riga h, beat ph*12 + k)  <-  vecchio (riga ph*H + h, beat k)
    """
    XS.MEM_DIR = mem_dir                                   # l'esportatore legge variabili di modulo
    XS.OUT_FILE = mem_dir / "w_stream_RadioML.mem"
    XS.REPORT = mem_dir / "quant_report.json"
    XS.main()
    beats_z7020 = [int(tok, 16) for tok in XS.OUT_FILE.read_text().split()]
    assert len(beats_z7020) == N_LAYERS * XS.NBEAT

    beats_zcu104 = []
    for layer in range(N_LAYERS):
        layer_beats = beats_z7020[layer * XS.NBEAT:(layer + 1) * XS.NBEAT]
        head = layer_beats[:XS.BO_END]                     # bn_sh, coef, D, bn_w, bn_b, bout: identici
        mixing_old = layer_beats[XS.BO_END:XS.WO_END]      # 512 righe x 12 beat
        mixing_new = []
        for h in range(H):                                 # nuova riga = canale di ingresso h
            for phase in range(N_PHASES):                  # le 4 fasi diventano 4 blocchi da 12 beat
                old_row = phase * H + h
                mixing_new.extend(mixing_old[old_row * STREAM_BEATS_Z7020:
                                             (old_row + 1) * STREAM_BEATS_Z7020])
        assert len(mixing_new) == len(mixing_old)
        # controllo sui pesi veri: uscita c = phase*64 + m, 12 bit a [c*12 +: 12] della riga nuova
        for h in (0, 1, 63, 127):
            new_row_value = 0
            for beat_idx in range(N_PHASES * STREAM_BEATS_Z7020):
                new_row_value |= mixing_new[h * 48 + beat_idx] << (64 * beat_idx)
            for phase in range(N_PHASES):
                old_row_value = 0
                for beat_idx in range(STREAM_BEATS_Z7020):
                    old_row_value |= mixing_old[(phase * H + h) * 12 + beat_idx] << (64 * beat_idx)
                for mac in (0, 17, 63):
                    out_ch = phase * 64 + mac
                    assert (new_row_value >> (out_ch * 12)) & 0xFFF == (old_row_value >> (mac * 12)) & 0xFFF
        beats_zcu104.extend(head + mixing_new)
    assert len(beats_zcu104) == len(beats_z7020)

    zcu_mem = mem_dir / "w_stream_RadioML_nmac256.mem"
    zcu_mem.write_text("\n".join(f"{b:016x}" for b in beats_zcu104) + "\n")
    # .bin: 64 bit per beat, little endian (stesso formato di w_stream_ECG.bin)
    (mem_dir / "w_stream_RadioML.bin").write_bytes(np.array(beats_z7020, dtype=np.uint64).astype("<u8").tobytes())
    (mem_dir / "w_stream_RadioML_nmac256.bin").write_bytes(np.array(beats_zcu104, dtype=np.uint64).astype("<u8").tobytes())
    print(f"stream: {len(beats_z7020)} beat ({N_LAYERS} x {XS.NBEAT}); "
          f"ZCU104 = permutazione della sezione di mixing, verificata")


# ============================================================================
# metriche
# ============================================================================
def accuracy_table(predictions, labels, snrs):
    """Accuratezza globale, per SNR, a SNR >= 0 / >= 10 e recall per classe a SNR >= 10."""
    correct = predictions == labels
    high_snr = snrs >= 10
    return dict(
        acc=float(correct.mean()),
        acc_snr_ge0=float(correct[snrs >= 0].mean()),
        acc_snr_ge10=float(correct[high_snr].mean()),
        acc_per_snr={int(s): float(correct[snrs == s].mean()) for s in np.unique(snrs)},
        recall_snr_ge10={name: float(correct[high_snr & (labels == class_idx)].mean())
                         for class_idx, name in enumerate(CLASSES)})


def float_logits(model, codes, device):
    """Logit float sugli STESSI codici Q5.11 che riceve la catena bit-accurate."""
    chunks = []
    with torch.no_grad():
        for first in range(0, len(codes), BATCH):
            x = torch.from_numpy(codes[first:first + BATCH].astype(np.float32) / 2 ** T.F_IN).to(device)
            chunks.append(model(x).double().cpu().numpy())
    return np.concatenate(chunks)


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = T.load_modules()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False          # riferimento float in fp32 pieno

    n_classes = len(CLASSES)
    E.K_CLS = n_classes                                    # 24: cls_w.mem ha parole da 24 x 8 bit
    Q.NIN = T.NIN                                          # encoder generico a 2 ingressi (I, Q)
    assert T.F_IN == G.F_ACT                               # ingresso Q5.11

    tag = "core" + ("" if RESCALE_CHANNELS else "_norescale")
    out_dir = EXPERIMENT_DIR / f"quant_{time.strftime('%Y%m%d_%H%M')}_{tag}"
    if N_EVAL_FX != 9984:
        out_dir = out_dir.with_name(out_dir.name + f"_n{N_EVAL_FX}")   # prove parziali riconoscibili
    mem_dir = out_dir / "mem"
    mem_dir.mkdir(parents=True, exist_ok=False)            # mai sovrascrivere una quantizzazione esistente
    print(f"checkpoint {CKPT}\nriscalatura per canale {RESCALE_CHANNELS}   output {out_dir}")

    # --- modello float libero (il riferimento vero: quello addestrato) -------
    model_free = T.radioml_model(B, n_classes).to(device)
    model_free.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model_free.eval()

    # --- dati: cache int16 Q5.11 del run --------------------------------------
    cache_dir = T.cache_dir_for()
    input_scale = json.loads((cache_dir / "meta.json").read_text())["scale"]   # scala fissa del run
    val_set = T.RadioMLDataset(cache_dir, "val")
    test_set = T.RadioMLDataset(cache_dir, "test")
    cal_positions = stratified_positions(val_set, N_CAL)
    cal_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(XYOnly(val_set), cal_positions.tolist()),
        batch_size=BATCH, shuffle=False, num_workers=8)

    # --- riscalatura globale del ramo del valore (come SC/ECG) ---------------
    W = Q.weights_to_numpy_ecg(B, model_free)              # encoder in forma (H, 2)
    rng_before = E.calibrate_ranges(model_free, cal_loader, device, n_images=len(cal_positions),
                                    n_layers=N_LAYERS)
    print(f"\ncalibrazione su {len(cal_positions)} frame di validazione. Range prima della riscalatura:")
    E.print_ranges(rng_before, N_LAYERS)
    value_scale = E.choose_scale(max(rng_before["max_x"], rng_before["max_a"]), fill=FILL)
    W_fix = B.rescale_value_branch(W, value_scale)

    # --- riscalatura per canale di u ------------------------------------------
    # i picchi di u non dipendono dalla riscalatura globale (u = BN(x) e' invariante)
    u_peaks = measure_u_peaks(model_free, cal_loader, device)
    if RESCALE_CHANNELS:
        channel_shifts, shift_diag = choose_channel_shifts(G, W_fix, u_peaks)
    else:
        channel_shifts = [np.zeros(H, dtype=int) for _ in range(N_LAYERS)]
        shift_diag = [dict(layer=m, channels_over_ceiling=int((u_peaks[m] > U_CEIL).sum()),
                           peak_before=float(u_peaks[m].max())) for m in range(N_LAYERS)]
    W_fix = apply_channel_shifts(W_fix, channel_shifts)
    print("\nriscalatura per canale di u (k = esponente: u / 2^k, C e D * 2^k):")
    for diag in shift_diag:
        print("  " + "  ".join(f"{key} {value}" for key, value in diag.items()))
    (out_dir / "channel_shift.json").write_text(json.dumps(dict(
        margin_u=MARGIN_U, shifts=[k.tolist() for k in channel_shifts], diagnostics=shift_diag), indent=2))

    # --- le due trasformazioni non devono cambiare la funzione (senza clamp) --
    model_fix = Q.numpy_to_torch_ecg(B, T.radioml_model(B, n_classes), W_fix).to(device).eval()
    probe_x, _ = next(iter(cal_loader))
    with torch.no_grad():
        logits_ref = model_free(probe_x.to(device)).double().cpu().numpy()
        logits_resc = model_fix(probe_x.to(device)).double().cpu().numpy()
    max_dev = float(np.abs(logits_resc - logits_ref).max())
    max_logit = float(np.abs(logits_ref).max())
    print(f"\ns = {value_scale}   controllo riscalature: scarto max {max_dev:.3e} "
          f"(logit max {max_logit:.1f})")
    # soglia relativa come in SC: rumore float32 di 6 layer con le operazioni riordinate
    assert max_dev < 1e-4 * max_logit, "la riscalatura ha cambiato la funzione"
    assert (logits_ref.argmax(1) == logits_resc.argmax(1)).all(), "la riscalatura ha cambiato una decisione"

    # da qui il modello riscalato porta i clamp di u: e' il float "con i vincoli"
    add_u_clamps(model_fix)
    rng_after = E.calibrate_ranges(model_fix, cal_loader, device, n_images=len(cal_positions),
                                   n_layers=N_LAYERS)
    print("\nrange dopo le riscalature (con clamp di u):")
    E.print_ranges(rng_after, N_LAYERS)

    # --- F_SSM e F_D: imposti (ZCU104) ----------------------------------------
    f_ssm_free = E.needed_f_ssm(rng_after["max_yssm"], fill=FILL)
    f_ssm = CORE_GENERICS["F_SSM"]
    E.set_f_ssm(f_ssm)
    f_d_per_layer = [G.calib(layer_w["D"], G.W_D) for layer_w in W_fix["layers"]]
    f_d = CORE_GENERICS["F_D"]
    print(f"\nF_SSM {f_ssm} (calibrato {f_ssm_free}; max|yssm| {rng_after['max_yssm']:.2f})")
    print(f"F_D   {f_d} (calibrati per layer {f_d_per_layer}: sotto 13 = D tosato)")

    luts = E.write_luts(mem_dir, verbose=False)
    (mem_dir / "isqrt_lut.mem").unlink()                   # serve solo alla LayerNorm

    # --- i sei layer: biquad ZOH + Jury, D, mixing, BN ripiegata ------------
    layers_q, affines_q = [], []
    for layer_idx, layer_w in enumerate(W_fix["layers"]):
        print(f"\nlayer {layer_idx}")
        layer_q = E.quantize_layer(layer_w, verbose=True, f_d=f_d)
        E.write_layer_mem(layer_idx, layer_q, mem_dir, verbose=False, with_norm=False)
        affine_q = B.quantize_affine(layer_w["bn_w"], layer_w["bn_b"], name=f"l{layer_idx}_bn", verbose=True)
        B.write_affine_mem(layer_idx, affine_q, mem_dir, verbose=False)
        layers_q.append(layer_q); affines_q.append(affine_q)

    # --- encoder a 2 ingressi (F_WENC calibrato: generic dell'encoder) ------
    enc = Q.quantize_encoder_ecg(G, W_fix["enc_w"], W_fix["enc_b"], None)
    print(f"\nencoder: F_WENC {enc['f_enc']}  SH_OUT {enc['sh_out']}  max|w| {enc['max_w']:.4f}  "
          f"max|b| {enc['max_b']:.4f}  tosati w {enc['sat_w']}  b {enc['sat_b']}")
    Q.write_encoder_mem_ecg(G, enc, mem_dir)

    # --- norm finale (BN_SH: generic del decoder) ----------------------------
    print()
    final_affine_q = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], mem_dir,
                                                nlayer=N_LAYERS, verbose=True)

    # --- verifica bit-accurate sul TEST (sottoinsieme stratificato) ----------
    eval_positions = stratified_positions(test_set, N_EVAL_FX)
    eval_frames = test_set.frame_idx[eval_positions]       # indici nel file completo
    eval_labels = test_set.labels[eval_positions]
    eval_snrs = test_set.snrs[eval_positions]
    all_codes = np.load(cache_dir / "x.npy", mmap_mode="r")
    test_codes = np.asarray(all_codes[eval_frames])        # (n, 1024, 2) int16, I in c0
    n_eval = len(eval_positions)
    blocks = [(first, min(first + CHUNK_FX, n_eval)) for first in range(0, n_eval, CHUNK_FX)]
    print(f"\nverifica bit-accurate su {n_eval} frame di test "
          f"({n_eval // (n_classes * 26)} per classe e SNR), {WORKERS_FX} processi ...")
    Q.WORKERS_FX = WORKERS_FX
    Q._CTX.update(E=E, G=G, B=B, X=test_codes, QZ=layers_q, QA=affines_q, NF=final_affine_q,
                  enc=enc, luts=luts)
    t_start = time.time()
    pooled, sat = Q.run_chain_parallel(blocks)             # uscita del mean pool, Q5.11
    fx_seconds = time.time() - t_start
    print(f"  {fx_seconds:.0f} s")

    # --- classificatore INT8 ----------------------------------------------------
    pooled_max = float(np.abs(pooled).max()) / 2.0 ** G.F_ACT
    f_cin = int(math.floor(math.log2(127 / max(pooled_max, 1e-12))))   # RQ_SHIFT e' del decoder
    dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin, mem_dir,
                              nlayer=N_LAYERS, verbose=False, with_norm=False)
    sat_cin = int((np.abs(E.shr_r(pooled, dec["rq_shift"])) > 127).sum())
    logits_fx = E.classify_fx(pooled, dec)
    print(f"classificatore: F_CIN {f_cin}, RQ_SHIFT {dec['rq_shift']}, F_W {dec['f_w']}, "
          f"max|xbar| {pooled_max:.3f}, ingressi INT8 tosati {sat_cin}")

    # --- riferimenti float sugli STESSI frame e codici -------------------------
    logits_free = float_logits(model_free, test_codes, device)     # modello addestrato, senza vincoli
    logits_clamped = float_logits(model_fix, test_codes, device)   # riscalato + clamp di u

    logits_fx_real = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
    logit_err = logits_fx_real - logits_free
    snr_db = 20 * math.log10(np.sqrt((logits_free ** 2).mean()) / max(np.sqrt((logit_err ** 2).mean()), 1e-12))
    pred_free, pred_clamped, pred_fx = (logits.argmax(1) for logits in (logits_free, logits_clamped, logits_fx))
    metrics = {name: accuracy_table(pred, eval_labels, eval_snrs)
               for name, pred in (("float_free", pred_free), ("float_clamped", pred_clamped),
                                  ("fixed", pred_fx))}
    agreement = float((pred_free == pred_fx).mean())
    acc_fx = metrics["fixed"]["acc"]
    ci95 = 1.96 * math.sqrt(acc_fx * (1 - acc_fx) / n_eval)        # intervallo binomiale, calcolato

    sat_total = (sat["enc"] + sat["final"] + sat_cin
                 + sum(l["sat_x"] + l["sat_a"] + l["sat_y"] + l["norm"] for l in sat["layers"]))
    print(f"\nSNR logit {snr_db:.1f} dB   argmax concordi col float libero {agreement:.4f}   "
          f"saturazioni totali {sat_total}")
    print(f"  {'layer':>6s}{'sat_x':>10s}{'sat_a':>10s}{'sat_y':>10s}{'norm':>10s}")
    for layer_idx, l in enumerate(sat["layers"]):
        print(f"  {layer_idx:>6d}{l['sat_x']:>10d}{l['sat_a']:>10d}{l['sat_y']:>10d}{l['norm']:>10d}")
    print(f"  encoder {sat['enc']}   norm finale {sat['final']}   ingresso INT8 {sat_cin}")

    lines = [f"RadioML 2018.01A, 24 classi, L=1024 x 2 (I,Q), test {n_eval} frame stratificati",
             f"generic del nucleo IMPOSTI: F_SSM {f_ssm}  F_D {f_d}   riscalatura per canale {RESCALE_CHANNELS}",
             f"{'':28s}{'globale':>9s}{'SNR>=0':>9s}{'SNR>=10':>9s}"]
    for name, label in (("float_free", "float libero"), ("float_clamped", "float riscalato + clamp u"),
                        ("fixed", "fixed (bit-accurate)")):
        m = metrics[name]
        lines.append(f"{label:28s}{m['acc']:9.4f}{m['acc_snr_ge0']:9.4f}{m['acc_snr_ge10']:9.4f}")
    lines += [f"perdita fixed vs float libero: {100 * (acc_fx - metrics['float_free']['acc']):+.2f} punti "
              f"(globale), {100 * (metrics['fixed']['acc_snr_ge10'] - metrics['float_free']['acc_snr_ge10']):+.2f} "
              f"(SNR>=10);  IC 95% fixed +-{ci95:.4f}",
              f"argmax concordi {agreement:.4f}   SNR logit {snr_db:.1f} dB   saturazioni totali {sat_total}",
              "acc per SNR (float libero / fixed): " + "  ".join(
                  f"{s:+d}:{metrics['float_free']['acc_per_snr'][s]:.3f}/{metrics['fixed']['acc_per_snr'][s]:.3f}"
                  for s in sorted(metrics["fixed"]["acc_per_snr"]))]
    summary = "\n".join(lines)
    print("\n" + summary)

    # --- report ---------------------------------------------------------------
    report = dict(
        checkpoint=str(CKPT), core_generics=CORE_GENERICS, rescale_channels=RESCALE_CHANNELS,
        margin_u=MARGIN_U, n_cal=len(cal_positions), n_eval_fx=n_eval, fx_seconds=round(fx_seconds, 1),
        input=dict(seq_len=T.L, n_in=T.NIN, f_in=T.F_IN, norm_mode=T.NORM_MODE, n_classes=n_classes),
        value_scale=value_scale, fill=FILL, f_ssm=f_ssm, f_ssm_free=f_ssm_free, f_d=f_d,
        f_d_calibrati=f_d_per_layer, channel_shift=shift_diag,
        peaks_before={k: rng_before[f"max_{k}"] for k in ("x", "a", "yssm")},
        peaks_after={k: rng_after[f"max_{k}"] for k in ("x", "a", "yssm")},
        encoder=dict(F_WENC=enc["f_enc"], SH_OUT=enc["sh_out"], sat_w=enc["sat_w"], sat_b=enc["sat_b"],
                     max_w=enc["max_w"], max_b=enc["max_b"]),
        decoder=dict(F_CIN=dec["f_cin"], F_W=dec["f_w"], RQ_SHIFT=dec["rq_shift"],
                     xbar_max=pooled_max, sat_cin=sat_cin),
        # `affine[m].SH` e' il campo che Export_s4d_wstream.py legge per il beat 0 di ogni layer
        affine=[dict(idx=m, F_BNW=q["f_bnw"], F_B=q["f_b"], SH=q["sh"], max_w=q["max_w"],
                     max_b=q["max_b"], rel_worst=q["rel_worst"], warn=q["warn"])
                for m, q in enumerate(affines_q + [final_affine_q])],
        layers=[dict(jury_before=q["jury_before"], jury_after=q["jury_after"], rho=q["rho"],
                     f_d=q["f_d"], f_d_cal=q["f_d_cal"], bsh_min=int(q["bsh"].min()),
                     bsh_max=int(q["bsh"].max()), warn=q["warn"], **sat["layers"][m])
                for m, q in enumerate(layers_q)],
        sat_total=sat_total, sat_enc=sat["enc"], sat_final=sat["final"],
        metrics=metrics, acc_fixed_ci95=ci95, agreement=agreement, logit_snr_db=round(snr_db, 2),
    )
    (out_dir / "quant_report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(out_dir / "quant_report.json", mem_dir / "quant_report.json")
    (out_dir / "summary.txt").write_text(summary + "\n")
    np.savez(out_dir / "eval_logits.npz", eval_frames=eval_frames, labels=eval_labels, snrs=eval_snrs,
             logits_fixed=logits_fx.astype(np.int64), logits_float=logits_free)

    generics = (f"-- RadioML 2018.01A, L=1024 x 2 ingressi (I,Q), {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"-- nucleo s4d_core_axis (ZCU104)\n"
                f"F_SSM    = {f_ssm}\n"
                f"F_D      = {f_d}\n"
                f"-- BN_SH dei layer 0..5 (nello stream, beat 0 di ogni layer): "
                f"{[q['sh'] for q in affines_q]}\n"
                f"-- encoder s4d_encoder_axis: NIN = 2, N_BIT_IN = 16, IN_SIGNED = 1, C0_AT_LSB = 1 (c0 = I)\n"
                f"F_WENC   = {enc['f_enc']}   (OUT_SHIFT = {enc['sh_out']})\n"
                f"-- decoder\n"
                f"BN_SH    = {final_affine_q['sh']}   (norm finale l6: F_BNW {final_affine_q['f_bnw']}, F_B {final_affine_q['f_b']})\n"
                f"RQ_SHIFT = {dec['rq_shift']}   (F_CIN {dec['f_cin']}, F_W {dec['f_w']})\n"
                f"K        = {n_classes}   (cls_w.mem: 128 righe da {n_classes * 8} bit, cls_b.mem: {n_classes} righe)\n"
                f"-- ingresso: codice Q5.11 = sat16(round(campione HDF5 * SCALE * 2048)), "
                f"SCALE = {input_scale!r} ({cache_dir}/meta.json)\n")
    (out_dir / "top_generics.txt").write_text(generics)
    print("\n" + generics)

    export_weight_streams(mem_dir)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)       # lo script che ha prodotto i file
    print(f"\noutput: {out_dir}\n.mem e stream: {mem_dir}")
    return out_dir


if __name__ == "__main__":
    main()
