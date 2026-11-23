# Training and quantization

The scripts used to train the four models and to produce the `.mem` files of the Vivado project. They are the scripts that produced the weights in `mem/mem_ECG/`, `mem/mem_RadioML/`, `mem/mem_SP/` and `mem/mem_sCIFAR/`. Comments of the scripts added for the sizing study (`sizing/`, `Quantize_s4d_speech_command_l1024.py`, `Quantize_S4D_RadioML.py`, `make_wstream.py`) are in Italian.

Run every script from this folder: the paths to data, caches and outputs are relative to it.

## Files

| File | Role |
|---|---|
| `Train_and_quantize_s4d_sCIFAR.py` | sCIFAR: training **and** quantization in one script. It also contains the three shared modules (as source strings) used by all the other scripts: `gen_s4d_mem` (fixed-point formats, ZOH/biquad conversion, Jury clamp), `s4d_export` (layer quantization, `.mem` packing, bit-accurate model) and `s4d_bn` (BatchNorm folding into a per-channel affine) |
| `Train_s4d_ecg.py` | ECG (PTB-XL) training |
| `Train_S4D_RadioML.py` | RadioML 2018.01A training |
| `Train_s4d_speech_command_8bit.py` | Speech Commands: decoding of the WAV files into the 16 kHz cache (`build_cache`) and helper functions used by the 12 kHz script |
| `Train_s4d_speech_command_l1024.py` | Speech Commands training (12 kHz, 12 samples per step, L = 1024) |
| `Finetune_clamp_s4d_speech_command_l1024.py` | Speech Commands saturation-aware fine-tuning (clamp at +-16 on the norm outputs) |
| `Quantize_s4d_ecg.py` | **Example quantization script** (ECG): rescaling, quantization, `.mem` export and bit-accurate check |
| `Quantize_s4d_speech_command_l1024.py` | Speech Commands quantization (core generics F_SSM 9 / F_D 13 imposed) |
| `Quantize_S4D_RadioML.py` | RadioML quantization (per-channel power-of-two rescaling of u) |
| `Export_s4d_wstream.py` | per-layer `.mem` files -> Zynq-7020 weight stream (H = 128 only) |
| `make_wstream.py` | per-layer `.mem` files -> ZCU104 weight stream for any (H, N) |
| `make_fpga_mem.py` | quantization output -> the 9 `.mem` files of the Vivado project, any (H, N) |
| `sizing/train_sweep.py` | training of one (H, N) configuration with the recipe of the paper (H, N as arguments) |
| `sizing/quantize_sweep.py` | quantization of a sized model (PTB-XL, Speech Commands, RadioML) with the scripts above, H and N set from outside |
| `sizing/finetune_sc.py` | clamp fine-tuning of a sized Speech Commands model |
| `sizing/quantize_scifar.py` | sCIFAR quantization of a trained checkpoint without re-training |

## Data layout

```
training/
  data_ptbxl/                        PTB-XL 1.0.3 (ptbxl_database.csv, scp_statements.csv, records100/)
  data/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5
  data/speechcommands/               downloaded by torchaudio
  data/cifar-10-batches-py/          downloaded by torchvision
```

The scripts build their caches under `data/` the first time they run.

## Workflow

1. **Train** the float model.

   ```
   python Train_s4d_ecg.py
   python Train_S4D_RadioML.py
   python -c "import Train_s4d_speech_command_8bit as T; T.build_cache()"   # 16 kHz cache of Speech Commands
   python Train_s4d_speech_command_l1024.py
   python Finetune_clamp_s4d_speech_command_l1024.py
   python Train_and_quantize_s4d_sCIFAR.py           # sCIFAR: training and quantization
   ```

   Outputs:
   - ECG and RadioML: a new folder `experiments/exp_<date>_<task>/` with `best.pt`, `report.json` and the training log.
   - Speech Commands: `ckpt/sc12k_best.pt`, then `finetune_clamp16/ckpt/sc12k_best.pt`.
   - sCIFAR: `ckpt/`, and the quantized files in `mem_trained_bn/`.

2. **Quantize.** `Quantize_s4d_ecg.py` is the example.
   1. Set `EXP` at the top of the file to the folder created by `Train_s4d_ecg.py`.
   2. Run:
      ```
      python Quantize_s4d_ecg.py
      ```
   3. The script writes `experiments/<run>/quant_<date>_hw/`, containing:
      - `mem/` with the per-layer `.mem` files and `quant_report.json`;
      - `summary.txt` with float vs fixed-point macro AUC.

   With `HW_LOCK = True` the generics of the block design are imposed (`F_SSM`, `F_D`, `F_WENC`, `RQ_SHIFT`, final `BN_SH`); with `False` they are calibrated freely.

   For another task, start from this script. The main differences are:
   - the number of encoder inputs (`NIN`);
   - the number of classes (`K`);
   - the input preprocessing.

   The RadioML weights also used a per-channel power-of-two rescaling of the norm output u (exact in float: u is divided by 2^k and C, D are multiplied by 2^k), because its activations exceeded the ±16 range of Q5.11.

3. **Export to the Vivado project.**

   ```
   python make_fpga_mem.py --mem-dir experiments/<run>/quant_<date>_hw/mem --suffix ECG --out ../mem/mem_ECG
   python make_fpga_mem.py --mem-dir mem_trained_bn --suffix sCIFAR --out ../mem/mem_sCIFAR
   ```

   The script:
   - builds the weight stream and decodes it back to check it;
   - packs the mixing weights for the ZCU104 core (NMAC = 2H, one row per input channel) and prints H, MODES and NBEAT for the block design;
   - copies the encoder, decoder and LUT files with the suffix of the block design.

   Run on the outputs of the original quantization runs, it reproduces the four `mem_*` folders of this repository byte for byte.

4. **Update the generics** of the block design if they changed. The values to use are printed by the quantization script, which also writes them to `top_generics.txt` and `quant_report.json`: `F_SSM`, `F_WENC`, `K`, decoder `BN_SH` and `RQ_SHIFT`, and `NIN`/`N_BIT_IN`/`IN_SIGNED` of the encoder. The per-layer BN shifts travel in the weight stream and need no change.

## Model sizing (`sizing/`)

The four models were selected with a sweep over H ∈ {32, 64, 128} and N ∈ {16, 32, 64, 128} (plus H = 256, N = 64 for PTB-XL; RadioML only N = 64, H ∈ {32, 64, 128}), seed 0, with the training recipe of the paper (RadioML: 30 epochs without early stopping). For each task the selected model is the smallest one whose validation metric is within δ of the best (δ = 0.002 macro AUC for PTB-XL, 0.5 points of accuracy for the other tasks); sCIFAR keeps H = 128, N = 64.

```
python sizing/train_sweep.py --task ptbxl --H 32 --N 16 --seed 0 --out sizing/runs/ptbxl_H32_N16_s0
python sizing/quantize_sweep.py --task ptbxl --H 32 --N 16 --mode free
python sizing/finetune_sc.py --H 128 --N 16                 # Speech Commands, before quantization
python sizing/quantize_sweep.py --task sc --H 128 --N 16 --mode core_ft
python sizing/quantize_sweep.py --task radio --H 64 --N 64
```

The fixed-point formats of the RTL are the same for every size (Q5.11 activations, 24-bit a1/a2, 16-bit b0/b1 with a per-channel exponent, 12-bit mixing weights, INT8 classifier). The block design takes H, MODES, NUNITS, NCH, NMAC and NBEAT (see the main README).

## Float reference results

Float test metric of the deployed models (from the training and quantization reports). The fixed-point values are the ones measured on the board, listed in the main README.

| Task | Model (H, N) | Float model |
|---|---|---|
| ECG | (32, 16) | macro AUC 0.9228 |
| RadioML | (64, 64) | accuracy over all SNRs 0.6113 (9984 test frames) |
| Speech Commands | (128, 16) | top-1 accuracy 0.9506 (after the clamp fine-tuning) |
| sCIFAR | (128, 64) | top-1 accuracy 0.7637 |
