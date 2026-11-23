#!/usr/bin/env python3
"""
Quantize_s4d_speech_command_l1024.py -- quantizzazione, export dei .mem e dello
stream dei pesi, verifica bit-accurate del modello Speech Commands L = 1024
addestrato da Train_s4d_speech_command_l1024.py (12 kHz, 12 campioni per passo).

Si lancia senza argomenti:

    conda run -n mamba_env --no-capture-output python Quantize_s4d_speech_command_l1024.py

E' la STESSA catena di Quantize_s4d_ecg.py (che ha gia' l'encoder a 12 ingressi
Q5.11): le funzioni dell'encoder, della norm e della verifica bit-accurate
parallela vengono importate da li', non ricopiate, cosi' lo specchio dell'RTL
resta uno solo. I moduli gen_s4d_mem / s4d_export / s4d_bn sono quelli estratti
dal monolitico, non modificati.

GENERIC
  nucleo (s4d_core_axis):  F_SSM e F_D
      CORE_LOCK = True  -> imposti ai valori di CLAUDE.md (F_SSM 9, F_D 13)
      CORE_LOCK = False -> calibrati sui dati (riferimento: quanto costa il vincolo)
  encoder / decoder:       F_WENC, BN_SH della norm finale, RQ_SHIFT, K
      sempre calibrati: encoder e decoder di questo progetto vengono sistemati a
      parte, qui si STAMPANO i valori da usare (top_generics.txt).
  BN_SH dei sei layer:     non e' un generic, viaggia nello stream (beat 0).

USCITE, in una cartella nuova quant_<data>_<core|free>/ dentro l'esperimento:
    mem/                      .mem per layer + encoder + decoder + LUT
    mem/w_stream_SC.mem/.bin          stream Zynq-7020 (mixing 512 righe x 12 beat)
    mem/w_stream_SC_nmac256.mem/.bin  stream ZCU104   (mixing 128 righe x 48 beat)
    quant_report.json, summary.txt, top_generics.txt

Variabili d'ambiente per le prove:
    SC12K_CORE_LOCK=0   quantizzazione libera
    SC12K_N_EVAL=256    verifica bit-accurate su un sottoinsieme (default: tutto il test)
    SC12K_EVAL_IDX=0,165,372   verifica su registrazioni di test scelte (per gli stimoli)
    SC12K_F_CIN=4       impone F_CIN del classificatore invece di calibrarlo sul sottoinsieme
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
sys.path.insert(0, str(EXPERIMENT_DIR))         # lo script di training di questo esperimento
os.chdir(PROJECT_ROOT)                          # i moduli importati usano percorsi relativi alla radice

import Train_s4d_speech_command_l1024 as T      # noqa: E402  catena dati e modello
import Quantize_s4d_ecg as Q                    # noqa: E402  encoder 12 ingressi + verifica bit-accurate
import Export_s4d_wstream as XS                 # noqa: E402  impacchettamento dello stream
import Finetune_clamp_s4d_speech_command_l1024 as FT   # noqa: E402  clamp a +-16 sulle norm (stesso hook del fine-tuning)

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
# Modello da quantizzare: di default quello uscito dal fine-tuning con il clamp.
# SC12K_CKPT=ckpt/sc12k_best.pt (relativo all'esperimento) per il modello float originale.
CKPT = EXPERIMENT_DIR / os.environ.get("SC12K_CKPT", "finetune_clamp16/ckpt/sc12k_best.pt")
# Il modello del fine-tuning e' stato addestrato CON il taglio a +-16 all'uscita
# delle norm (quello che fa l'hardware): il riferimento float e la calibrazione
# devono usarlo anche loro, altrimenti si confronta il fixed con una funzione diversa.
CLAMP_NORMS = os.environ.get("SC12K_CLAMP", "1" if "finetune_clamp" in str(CKPT) else "0") == "1"

CORE_LOCK = os.environ.get("SC12K_CORE_LOCK", "1") == "1"
CORE_GENERICS = dict(F_SSM=9, F_D=13)           # valori protetti di CLAUDE.md

FILL = 0.75             # frazione del tetto occupata dopo la riscalatura (come sCIFAR/ECG)
N_CAL = None            # registrazioni di validation per la calibrazione (None = tutte)
_n_eval_env = os.environ.get("SC12K_N_EVAL")
N_EVAL_FX = int(_n_eval_env) if _n_eval_env else None   # registrazioni di test (None = tutte)
_eval_idx_env = os.environ.get("SC12K_EVAL_IDX")
EVAL_IDX = [int(tok) for tok in _eval_idx_env.split(",")] if _eval_idx_env else None   # indici espliciti nel test
_f_cin_env = os.environ.get("SC12K_F_CIN")
F_CIN_FORCED = int(_f_cin_env) if _f_cin_env else None   # F_CIN imposto: su pochi record la calibrazione darebbe un altro shift
CHUNK_FX = 32           # registrazioni per blocco nella verifica bit-accurate
WORKERS_FX = 56         # processi paralleli: la catena bit-accurate gira su CPU
BATCH = 64
SEED = 1337

N_LAYERS = T.N_LAYERS
H = T.D_MODEL
STREAM_ROWS_Z7020 = 4 * H     # mixing Zynq-7020: 4 fasi x 128 righe
STREAM_BEATS_Z7020 = 12       # 64 MAC x 12 bit / 64 bit
N_PHASES = 4


# ============================================================================
# dati: codici interi Q5.11 esattamente come li riceve l'encoder
# ============================================================================
def waves_to_codes(waves_int16):
    """(N, 12000) int16 a 12 kHz -> (N, 1024, 12) int16 di codici Q5.11.

    Stessa aritmetica di SC12kFrameDataset.__getitem__ senza augmentation:
    scala fissa, arrotondamento, saturazione, frame da 12, 24 passi di zeri.
    """
    code_scale = T.INPUT_SCALE * (1 << T.F_IN) / 32768.0
    codes = np.clip(np.round(waves_int16.astype(np.float32) * code_scale), -32768, 32767)
    frames = np.zeros((len(codes), T.SEQ_LEN, T.FRAME_LEN), dtype=np.int16)
    frames[:, :T.N_FRAMES_AUDIO] = codes.reshape(len(codes), T.N_FRAMES_AUDIO, T.FRAME_LEN)
    return frames


def add_norm_clamps(model):
    """Aggiunge il clamp a +-16 all'uscita delle 7 norm, se CLAMP_NORMS e' attivo.

    Va chiamata PRIMA di E.calibrate_ranges: gli hook girano nell'ordine in cui
    sono registrati, quindi quelli della calibrazione vedono l'uscita gia' tagliata.
    """
    if CLAMP_NORMS:
        clamp_hook = FT.NormClamp()
        for norm in list(model.norms) + [model.norm_f]:
            norm.register_forward_hook(clamp_hook)
    return model


# ============================================================================
# stream dei pesi: Zynq-7020 (4 fasi) e ZCU104 (riga unica)
# ============================================================================
def export_weight_streams(mem_dir):
    """Scrive lo stream nei due formati e ne verifica la corrispondenza.

    Zynq-7020: e' Export_s4d_wstream.py com'e', puntato a mem_dir (rilegge lo
    stream e lo confronta con i .mem sorgente).
    ZCU104: permutazione pura della sola sezione di mixing, da CLAUDE.md 1.0:
        nuovo (riga h, beat ph*12 + k)  <-  vecchio (riga ph*H + h, beat k)
    """
    XS.MEM_DIR = mem_dir                                   # l'esportatore legge variabili di modulo
    XS.OUT_FILE = mem_dir / "w_stream_SC.mem"
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

    zcu_mem = mem_dir / "w_stream_SC_nmac256.mem"
    zcu_mem.write_text("\n".join(f"{b:016x}" for b in beats_zcu104) + "\n")
    # .bin: 64 bit per beat, little endian (stesso formato di w_stream_ECG.bin)
    (mem_dir / "w_stream_SC.bin").write_bytes(np.array(beats_z7020, dtype=np.uint64).astype("<u8").tobytes())
    (mem_dir / "w_stream_SC_nmac256.bin").write_bytes(np.array(beats_zcu104, dtype=np.uint64).astype("<u8").tobytes())
    print(f"stream: {len(beats_z7020)} beat ({N_LAYERS} x {XS.NBEAT}); "
          f"ZCU104 = permutazione della sezione di mixing, verificata")


# ============================================================================
# main
# ============================================================================
def main():
    G, E, B = T.load_modules(str(T.MONO))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)

    class_names = json.loads((T.CACHE_16K_DIR / "sc_labels.json").read_text())
    n_classes = len(class_names)
    E.K_CLS = n_classes                                    # 35: cls_w.mem ha parole da 35 x 8 bit
    assert Q.NIN == T.FRAME_LEN and T.F_IN == G.F_ACT      # encoder ECG: 12 ingressi Q5.11

    tag = "core" if CORE_LOCK else "free"
    out_dir = EXPERIMENT_DIR / f"quant_{time.strftime('%Y%m%d_%H%M')}_{tag}"
    if EVAL_IDX:
        out_dir = out_dir.with_name(out_dir.name + f"_stim{len(EVAL_IDX)}")   # run fatta per generare gli stimoli
    elif N_EVAL_FX:
        out_dir = out_dir.with_name(out_dir.name + f"_n{N_EVAL_FX}")   # le prove parziali non si confondono con la run vera
    mem_dir = out_dir / "mem"
    mem_dir.mkdir(parents=True, exist_ok=False)            # mai sovrascrivere una quantizzazione esistente
    print(f"checkpoint {CKPT}\nCORE_LOCK {CORE_LOCK}   CLAMP_NORMS {CLAMP_NORMS}   output {out_dir}")

    model = T.build_model(B, n_classes).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    add_norm_clamps(model)                                 # riferimento float nelle condizioni dell'hardware
    model.eval()

    # --- calibrazione dei range sul VALIDATION (mai sul test) ---------------
    val_set = T.SC12kFrameDataset(T.CACHE_12K_DIR / "sc_validation_x12k.npy",
                                  T.CACHE_16K_DIR / "sc_validation_y.npy", augment=False)
    n_cal = len(val_set) if N_CAL is None else min(N_CAL, len(val_set))
    cal_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=4)
    W = Q.weights_to_numpy_ecg(B, model)                   # encoder in forma (H, 12)
    rng_before = E.calibrate_ranges(model, cal_loader, device, n_images=n_cal, n_layers=N_LAYERS)
    print(f"\ncalibrazione su {n_cal} registrazioni di validation. Range prima della riscalatura:")
    E.print_ranges(rng_before, N_LAYERS)

    # scala globale s (potenza di due) sul ramo del valore, compensata sulla BN ripiegata
    scale = E.choose_scale(max(rng_before["max_x"], rng_before["max_a"]), fill=FILL)
    W_fix = B.rescale_value_branch(W, scale)

    # la riscalatura deve lasciare la funzione identica: controllo sui logit
    # u = BN(x) non cambia con la riscalatura (w/s compensa s*x): il clamp resta a +-16
    model_fix = add_norm_clamps(Q.numpy_to_torch_ecg(B, T.build_model(B, n_classes), W_fix)).to(device).eval()
    probe_x, _ = next(iter(cal_loader))
    with torch.no_grad():
        logits_ref = model(probe_x.to(device)).double().cpu().numpy()
        logits_resc = model_fix(probe_x.to(device)).double().cpu().numpy()
    max_dev = float(np.abs(logits_resc - logits_ref).max())
    ulp = float(np.spacing(np.float32(max(np.abs(logits_ref).max(), 1.0))))
    print(f"\ns = {scale}   controllo riscalatura: scarto max {max_dev:.3e} ({max_dev / ulp:.0f} ulp)")
    # Soglia RELATIVA (1e-4 del logit piu' grande) piu' argmax invariato. La soglia
    # fissa di 256 ulp del quantizzatore ECG e' al limite del rumore float32 di 6
    # layer con le operazioni riordinate: qui si misurano 255-260 ulp (5e-4 su
    # logit di ampiezza ~30), cioe' rumore di arrotondamento e non una funzione diversa.
    max_logit = float(np.abs(logits_ref).max())
    assert max_dev < 1e-4 * max_logit, "la riscalatura ha cambiato la funzione"
    assert (logits_ref.argmax(1) == logits_resc.argmax(1)).all(), "la riscalatura ha cambiato una decisione"

    rng_after = E.calibrate_ranges(model_fix, cal_loader, device, n_images=n_cal, n_layers=N_LAYERS)
    print("\nrange dopo la riscalatura:")
    E.print_ranges(rng_after, N_LAYERS)

    # --- F_SSM e F_D: i due generic del nucleo ------------------------------
    f_ssm_free = E.needed_f_ssm(rng_before["max_yssm"], fill=FILL)
    f_ssm = CORE_GENERICS["F_SSM"] if CORE_LOCK else f_ssm_free
    E.set_f_ssm(f_ssm)
    f_d_per_layer = [G.calib(layer_w["D"], G.W_D) for layer_w in W_fix["layers"]]
    f_d = CORE_GENERICS["F_D"] if CORE_LOCK else min(f_d_per_layer)
    print(f"\nF_SSM {f_ssm} (calibrato {f_ssm_free}; tetto yssm +-{E.CEIL_SSM:g}, "
          f"max|yssm| misurato {rng_before['max_yssm']:.2f})")
    print(f"F_D   {f_d} (calibrati per layer {f_d_per_layer})")

    luts = E.write_luts(mem_dir, verbose=False)
    (mem_dir / "isqrt_lut.mem").unlink()                   # serve solo alla LayerNorm, qui c'e' la BN ripiegata

    # --- i sei layer: biquad ZOH + Jury, D, mixing, BN ripiegata ------------
    layers_q, affines_q = [], []
    for layer_idx, layer_w in enumerate(W_fix["layers"]):
        print(f"\nlayer {layer_idx}")
        layer_q = E.quantize_layer(layer_w, verbose=True, f_d=f_d)
        E.write_layer_mem(layer_idx, layer_q, mem_dir, verbose=False, with_norm=False)
        affine_q = B.quantize_affine(layer_w["bn_w"], layer_w["bn_b"], name=f"l{layer_idx}_bn", verbose=True)
        B.write_affine_mem(layer_idx, affine_q, mem_dir, verbose=False)
        layers_q.append(layer_q); affines_q.append(affine_q)

    # --- encoder (libero: F_WENC e' un generic dell'encoder, da impostare) --
    enc = Q.quantize_encoder_ecg(G, W_fix["enc_w"], W_fix["enc_b"], None)
    print(f"\nencoder: F_WENC {enc['f_enc']}  SH_OUT {enc['sh_out']}  max|w| {enc['max_w']:.4f}  "
          f"max|b| {enc['max_b']:.4f}  tosati w {enc['sat_w']}  b {enc['sat_b']}")
    Q.write_encoder_mem_ecg(G, enc, mem_dir)

    # --- norm finale (libera: BN_SH e' un generic del decoder) --------------
    print()
    final_affine_q = B.write_decoder_affine_mem(W_fix["nf_w"], W_fix["nf_b"], mem_dir,
                                                nlayer=N_LAYERS, verbose=True)

    # --- verifica bit-accurate sul TEST --------------------------------------
    test_waves = np.load(T.CACHE_12K_DIR / "sc_testing_x12k.npy", mmap_mode="r")
    test_labels = np.load(T.CACHE_16K_DIR / "sc_testing_y.npy")
    n_eval = len(test_labels) if N_EVAL_FX is None else min(N_EVAL_FX, len(test_labels))
    if EVAL_IDX:
        eval_idx = np.array(EVAL_IDX, dtype=np.int64)      # ordine scelto da chi chiama (ordine dello stimolo)
        n_eval = len(eval_idx)
    elif N_EVAL_FX:
        # sottoinsieme casuale e non i primi n: la cache e' ordinata per classe
        eval_idx = np.sort(np.random.default_rng(SEED).choice(len(test_labels), n_eval, replace=False))
    else:
        eval_idx = np.arange(n_eval)
    test_codes = waves_to_codes(np.asarray(test_waves[eval_idx]))      # (n, 1024, 12) int16
    eval_labels = test_labels[eval_idx]
    blocks = [(first, min(first + CHUNK_FX, n_eval)) for first in range(0, n_eval, CHUNK_FX)]
    print(f"\nverifica bit-accurate su {n_eval} registrazioni di test, {WORKERS_FX} processi ...")
    Q.WORKERS_FX = WORKERS_FX
    Q._CTX.update(E=E, G=G, B=B, X=test_codes, QZ=layers_q, QA=affines_q, NF=final_affine_q,
                  enc=enc, luts=luts)
    t_start = time.time()
    pooled, sat = Q.run_chain_parallel(blocks)             # uscita del mean pool, Q5.11
    fx_seconds = time.time() - t_start
    print(f"  {fx_seconds:.0f} s")

    # --- classificatore INT8 --------------------------------------------------
    pooled_max = float(np.abs(pooled).max()) / 2.0 ** G.F_ACT
    f_cin = int(math.floor(math.log2(127 / max(pooled_max, 1e-12))))   # libero: RQ_SHIFT e' del decoder
    if F_CIN_FORCED is not None:
        f_cin = F_CIN_FORCED                               # stesso classificatore della quantizzazione completa
    dec = E.write_decoder_mem(None, None, W_fix["cls_w"], W_fix["cls_b"], f_cin, mem_dir,
                              nlayer=N_LAYERS, verbose=False, with_norm=False)
    sat_cin = int((np.abs(E.shr_r(pooled, dec["rq_shift"])) > 127).sum())
    logits_fx = E.classify_fx(pooled, dec)
    print(f"classificatore: F_CIN {f_cin}, RQ_SHIFT {dec['rq_shift']}, F_W {dec['f_w']}, "
          f"max|xbar| {pooled_max:.3f}, ingressi INT8 tosati {sat_cin}")

    # --- riferimento float sulle STESSE registrazioni e sugli stessi codici --
    float_chunks = []
    with torch.no_grad():
        for first in range(0, n_eval, 128):
            x = torch.from_numpy(test_codes[first:first + 128].astype(np.float32) / 2 ** T.F_IN).to(device)
            float_chunks.append(model(x).double().cpu().numpy())
    logits_float = np.concatenate(float_chunks)

    logits_fx_real = logits_fx / 2.0 ** (dec["f_w"] + dec["f_cin"])
    logit_err = logits_fx_real - logits_float
    snr_db = 20 * math.log10(np.sqrt((logits_float ** 2).mean()) / max(np.sqrt((logit_err ** 2).mean()), 1e-12))
    pred_float, pred_fx = logits_float.argmax(1), logits_fx.argmax(1)
    acc_float = float((pred_float == eval_labels).mean())
    acc_fx = float((pred_fx == eval_labels).mean())
    agreement = float((pred_float == pred_fx).mean())
    ci95 = 1.96 * math.sqrt(acc_fx * (1 - acc_fx) / n_eval)           # intervallo binomiale, calcolato
    # accuratezza per classe: dove si perde
    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        members = eval_labels == class_idx
        if members.any():
            per_class[class_name] = dict(n=int(members.sum()),
                                         float=float((pred_float[members] == class_idx).mean()),
                                         fixed=float((pred_fx[members] == class_idx).mean()))

    sat_total = (sat["enc"] + sat["final"] + sat_cin
                 + sum(l["sat_x"] + l["sat_a"] + l["sat_y"] + l["norm"] for l in sat["layers"]))
    print(f"\nSNR logit {snr_db:.1f} dB   argmax concordi {agreement:.4f}   saturazioni totali {sat_total}")
    print(f"  {'layer':>6s}{'sat_x':>10s}{'sat_a':>10s}{'sat_y':>10s}{'norm':>10s}")
    for layer_idx, l in enumerate(sat["layers"]):
        print(f"  {layer_idx:>6d}{l['sat_x']:>10d}{l['sat_a']:>10d}{l['sat_y']:>10d}{l['norm']:>10d}")
    print(f"  encoder {sat['enc']}   norm finale {sat['final']}   ingresso INT8 {sat_cin}")

    summary = (f"Speech Commands 35 classi, L=1024 x 12, test {n_eval} registrazioni\n"
               f"generic del nucleo: {'IMPOSTI' if CORE_LOCK else 'liberi'}  F_SSM {f_ssm}  F_D {f_d}\n"
               f"modello {CKPT.parent.parent.name}/{CKPT.name}   clamp +-16 sulle norm nel riferimento float: {CLAMP_NORMS}\n"
               f"accuratezza FP32          {acc_float:.4f}\n"
               f"accuratezza fixed (bit-accurate) {acc_fx:.4f}  +-{ci95:.4f} (95%)\n"
               f"perdita                   {100 * (acc_float - acc_fx):+.2f} punti\n"
               f"argmax concordi           {agreement:.4f}\n"
               f"SNR logit                 {snr_db:.1f} dB\n"
               f"saturazioni totali        {sat_total}")
    print("\n" + summary)

    # --- report ---------------------------------------------------------------
    report = dict(
        checkpoint=str(CKPT), core_lock=CORE_LOCK, core_generics=CORE_GENERICS, clamp_norms=CLAMP_NORMS,
        n_cal=n_cal, n_eval_fx=n_eval, fx_seconds=round(fx_seconds, 1),
        input=dict(fs=T.FS_OUT, frame_len=T.FRAME_LEN, seq_len=T.SEQ_LEN, f_in=T.F_IN,
                   input_scale=T.INPUT_SCALE, n_classes=n_classes),
        scale=scale, fill=FILL, f_ssm=f_ssm, f_ssm_free=f_ssm_free, f_d=f_d, f_d_calibrati=f_d_per_layer,
        peaks_before={k: rng_before[f"max_{k}"] for k in ("x", "a", "yssm")},
        peaks_after={k: rng_after[f"max_{k}"] for k in ("x", "a", "yssm")},
        per_layer_x_before=rng_before["x"], per_layer_x_after=rng_after["x"],
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
        acc_float=acc_float, acc_fixed=acc_fx, acc_fixed_ci95=ci95, agreement=agreement,
        logit_snr_db=round(snr_db, 2), per_class=per_class,
    )
    (out_dir / "quant_report.json").write_text(json.dumps(report, indent=2))
    shutil.copy2(out_dir / "quant_report.json", mem_dir / "quant_report.json")
    (out_dir / "summary.txt").write_text(summary + "\n")
    # logit interi (quelli che escono dal decoder), riferimento float e codici di ingresso:
    # servono a chi genera gli stimoli per la simulazione
    np.savez(out_dir / "eval_logits.npz", eval_idx=eval_idx, labels=eval_labels,
             logits_fixed=logits_fx.astype(np.int64), logits_float=logits_float,
             input_codes=test_codes if n_eval <= 64 else np.zeros(0, dtype=np.int16))

    generics = (f"-- Speech Commands L=1024 x 12 ingressi, {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"-- nucleo s4d_core_axis\n"
                f"F_SSM    = {f_ssm}{'' if CORE_LOCK else '   (LIBERO: diverso dal valore di CLAUDE.md se != 9)'}\n"
                f"F_D      = {f_d}\n"
                f"L        = {T.SEQ_LEN}, LOG2_L = {int(math.log2(T.SEQ_LEN))}\n"
                f"-- BN_SH dei layer 0..5 (nello stream, beat 0 di ogni layer): "
                f"{[q['sh'] for q in affines_q]}\n"
                f"-- encoder a 12 ingressi Q5.11 (enc_w.mem: 128 righe da 12 x 18 bit, c0 nei bit bassi)\n"
                f"F_WENC   = {enc['f_enc']}   (SH_OUT = {enc['sh_out']})\n"
                f"-- decoder\n"
                f"BN_SH    = {final_affine_q['sh']}   (norm finale l6: F_BNW {final_affine_q['f_bnw']}, F_B {final_affine_q['f_b']})\n"
                f"RQ_SHIFT = {dec['rq_shift']}   (F_CIN {dec['f_cin']}, F_W {dec['f_w']})\n"
                f"K        = {n_classes}   (cls_w.mem: 128 righe da {n_classes * 8} bit, cls_b.mem: {n_classes} righe)\n"
                f"-- ingresso: campione int16 a 12 kHz -> codice = round(int16 * {T.INPUT_SCALE * (1 << T.F_IN) / 32768.0:g}), saturato a 16 bit\n")
    (out_dir / "top_generics.txt").write_text(generics)
    print("\n" + generics)

    export_weight_streams(mem_dir)
    shutil.copy2(Path(__file__), out_dir / Path(__file__).name)       # lo script che ha prodotto i file
    print(f"\noutput: {out_dir}\n.mem e stream: {mem_dir}")
    return out_dir


if __name__ == "__main__":
    main()
