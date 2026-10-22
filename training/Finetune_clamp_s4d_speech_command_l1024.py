#!/usr/bin/env python3
"""
Finetune_clamp_s4d_speech_command_l1024.py -- "saturation-aware" fine-tuning
of the Speech Commands L = 1024 model.

WHY
In hardware the output of each folded BatchNorm (the signal u that enters the
biquad bank) and the output of the final norm are 16-bit Q5.11: ceiling +-16.
The model trained in float exceeds that ceiling in 0.1-1.8% of the samples and
the clipping alone costs ~3 points (measured in float: 95.65% -> 92.48%).
Here the clipping is put INSIDE the forward, so the network learns to work
within the format. No hardware parameter changes.

WHAT IT DOES
  1. starts from ckpt/sc12k_best.pt
  2. adds a clamp at +-U_CEIL to each norm (6 layers + final)
  3. after each step it projects back into the fixed formats the weights that
     went out of them:
       D                      -> Q3.13  (F_D = 13, 16 bit)
       gate branch of mixing  -> Q1.11  (W_W = 12, F_W = 11)
     (the "value" branch of the mixing is not touched: it is rescaled at export)
  4. 10 epochs, lr 1e-3 (kernel 1e-4), same augmentation as the training

OUTPUTS: finetune_clamp16/ckpt/sc12k_best.pt, sc12k_last.pt, train_log.jsonl,
train_report.json. No quantization here: the accuracies are in FLOAT,
with and without the clamp.

    conda run -n mamba_env --no-capture-output python Finetune_clamp_s4d_speech_command_l1024.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))
import Train_s4d_speech_command_l1024 as T      # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================
START_CKPT = EXPERIMENT_DIR / "ckpt" / "sc12k_best.pt"     # starting float model
OUT_DIR = EXPERIMENT_DIR / "finetune_clamp16"

FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-3            # a tenth of the training: it adjusts, it does not retrain
FINETUNE_LR_SSM = 1e-4        # poles and dt: even more careful, the stability must not move
FINETUNE_WARMUP_EPOCHS = 1

W_ACT, F_ACT = 16, 11         # format of u and of the final norm output (W_U/F_U, W_ACT/F_ACT)
U_CEIL = ((1 << (W_ACT - 1)) - 1) / (1 << F_ACT)           # 15.99951: largest representable value
W_D, F_D = 16, 13
D_CEIL = ((1 << (W_D - 1)) - 1) / (1 << F_D)               # 3.99988
W_W, F_W = 12, 11
MIX_CEIL = ((1 << (W_W - 1)) - 1) / (1 << F_W)             # 0.99951


class NormClamp:
    """Hook that clips the norm outputs at +-U_CEIL and counts how much it clips."""

    def __init__(self):
        self.enabled = True
        self.n_clipped = 0
        self.n_total = 0

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return None                                    # output unchanged
        self.n_clipped += int((output.abs() > U_CEIL).sum().item())
        self.n_total += output.numel()
        return output.clamp(-U_CEIL, U_CEIL)               # replaces the norm output

    def reset(self):
        self.n_clipped = self.n_total = 0


def project_weights(model):
    """Brings D and the gate branch of the mixing back into their fixed formats."""
    hidden = T.D_MODEL
    with torch.no_grad():
        for layer in model.layers:
            layer.D.clamp_(-D_CEIL, D_CEIL)                              # D: Q3.13
            mixing_weight = layer.output_linear[0].weight                # (2H, H, 1): rows 0:H value, H:2H gate
            mixing_weight[hidden:].clamp_(-MIX_CEIL, MIX_CEIL)           # only the gate: Q1.11


def main():
    assert not OUT_DIR.exists(), f"{OUT_DIR} already exists: move it or change OUT_DIR"
    ckpt_dir = OUT_DIR / "ckpt"
    ckpt_dir.mkdir(parents=True)

    G, E, B = T.load_modules(str(T.MONO))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(T.SEED); np.random.seed(T.SEED)
    class_names = json.loads((T.CACHE_16K_DIR / "sc_labels.json").read_text())

    T.build_cache_12k()                                     # already present: it only checks that the filter matches
    train_loader, val_loader, test_loader = T.build_loaders()
    model = T.build_model(B, len(class_names)).to(device)
    model.load_state_dict(torch.load(START_CKPT, map_location=device, weights_only=True))

    clamp = NormClamp()
    for norm in list(model.norms) + [model.norm_f]:         # 6 layer norms + final norm
        norm.register_forward_hook(clamp)

    # starting point, on the test set: without clamp, with clamp, with clamp + projected weights
    clamp.enabled = False
    start_noclamp = T.evaluate(model, test_loader, device)
    clamp.enabled = True; clamp.reset()
    start_clamp = T.evaluate(model, test_loader, device)
    start_clip_frac = clamp.n_clipped / max(clamp.n_total, 1)
    project_weights(model)
    start_clamp_proj = T.evaluate(model, test_loader, device)
    print(f"start (test): free float {start_noclamp:.4f}   with clamp {start_clamp:.4f} "
          f"(clipped {100 * start_clip_frac:.3f}%)   with clamp and projected weights {start_clamp_proj:.4f}", flush=True)

    # the training loop is the one of the main run: only its knobs change
    T.EPOCHS, T.WARMUP_EPOCHS = FINETUNE_EPOCHS, FINETUNE_WARMUP_EPOCHS
    T.LR, T.LR_SSM = FINETUNE_LR, FINETUNE_LR_SSM
    model, best_val = T.train(model, train_loader, val_loader, device, ckpt_dir,
                              OUT_DIR / "train_log.jsonl", after_step=project_weights)

    # result: the best model (chosen on the validation WITH clamp) on the test set
    clamp.enabled = True; clamp.reset()
    final_clamp = T.evaluate(model, test_loader, device)
    final_clip_frac = clamp.n_clipped / max(clamp.n_total, 1)
    clamp.enabled = False
    final_noclamp = T.evaluate(model, test_loader, device)
    max_d = max(float(layer.D.abs().max()) for layer in model.layers)
    max_gate = max(float(layer.output_linear[0].weight[T.D_MODEL:].abs().max()) for layer in model.layers)
    print(f"\ntest after fine-tuning: with clamp {final_clamp:.4f} (clipped {100 * final_clip_frac:.3f}%)   "
          f"without clamp {final_noclamp:.4f}   best val with clamp {best_val:.4f}")

    (OUT_DIR / "train_report.json").write_text(json.dumps(dict(
        level="floating point (FP32) with clamp at +-16 on the norms; weights NOT quantized",
        start_ckpt=str(START_CKPT), u_ceil=U_CEIL, d_ceil=D_CEIL, mix_gate_ceil=MIX_CEIL,
        epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, lr_ssm=FINETUNE_LR_SSM, warmup_epochs=FINETUNE_WARMUP_EPOCHS,
        test_start_float=start_noclamp, test_start_clamp=start_clamp, test_start_clamp_projected=start_clamp_proj,
        start_clip_fraction=start_clip_frac,
        best_val_clamp=best_val, test_final_clamp=final_clamp, test_final_noclamp=final_noclamp,
        final_clip_fraction=final_clip_frac, max_abs_D=max_d, max_abs_mix_gate=max_gate), indent=2))
    print(f"report: {OUT_DIR / 'train_report.json'}")


if __name__ == "__main__":
    main()
