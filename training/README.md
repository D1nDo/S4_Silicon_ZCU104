# Training and quantization

The scripts used to train the four models and to produce the `.mem` files of the Vivado project. They are the scripts that produced the weights in `mem_ECG/`, `mem_RadioML/`, `mem_SP/` and `mem_sCIFAR/`; only comments and messages have been translated.

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
| `Export_s4d_wstream.py` | per-layer `.mem` files -> weight stream (used by `make_fpga_mem.py`) |
| `make_fpga_mem.py` | quantization output -> the 9 `.mem` files of the Vivado project |

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
   python make_fpga_mem.py --mem-dir experiments/<run>/quant_<date>_hw/mem --suffix ECG --out ../mem_ECG
   python make_fpga_mem.py --mem-dir mem_trained_bn --suffix sCIFAR --out ../mem_sCIFAR
   ```

   The script:
   - builds the weight stream and decodes it back to check it;
   - reorders the mixing weights for the ZCU104 core (256 MACs, one row per input channel);
   - copies the encoder, decoder and LUT files with the suffix of the block design.

   Run on the outputs of the original quantization runs, it reproduces the four `mem_*` folders of this repository byte for byte.

4. **Update the generics** of the block design if they changed. The values to use are printed by the quantization script, which also writes them to `top_generics.txt` and `quant_report.json`: `F_SSM`, `F_WENC`, `K`, decoder `BN_SH` and `RQ_SHIFT`, and `NIN`/`N_BIT_IN`/`IN_SIGNED` of the encoder. The per-layer BN shifts travel in the weight stream and need no change.

## Fixed hardware contract

The model must keep:
- H = 128 channels, 6 layers, 32 modes (`D_STATE = 64`), L = 1024;
- the fixed-point formats of the RTL (Q5.11 activations, 24-bit a1/a2, 16-bit b0/b1 with a per-channel exponent, 12-bit mixing weights, INT8 classifier).

The training scripts check H and MODES at startup. The quantization reports every saturation and checks that the rescaling does not change the float function.

## Float reference results

These are the float values from the original training and quantization reports. The fixed-point values are the ones measured on the board, listed in the main README.

| Task | Float model |
|---|---|
| ECG | macro AUC 0.9255 |
| RadioML | accuracy over all SNRs 0.6054 (9984 test frames) |
| Speech Commands | top-1 accuracy 0.9538 (after the clamp fine-tuning) |
| sCIFAR | top-1 accuracy 0.7637 |
