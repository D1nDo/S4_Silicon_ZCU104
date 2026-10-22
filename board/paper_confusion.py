#!/usr/bin/env python3
"""
paper_confusion.py -- confusion matrices of the paper, built from the logits
READ BACK FROM THE ZCU104 BOARD (logits_board_*.bin written by the board programs):
    RadioML 2018.01A at SNR = +10 dB and +20 dB,
    Speech Commands v0.02 (35 words) on the official test split,
    sCIFAR on the 10000 CIFAR-10 test images.
The ECG task is multi-label and is reported with the macro AUC only.

Inputs : board/<task>/logits_board_*.bin (measured on the board, in the repository)
         board/<task>/labels_*.bin, snr_RadioML_9984.bin (generated with
         `python make_board_files.py <task> --data ... --no-golden`)
         board/<task>/logits_golden_*.bin, optional: if present, the board
         logits are checked against them
Outputs: images/*_confusion*.pdf (for the paper) and .png (for the README)
Usage  : python paper_confusion.py      (needs numpy and matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BOARD = Path(__file__).resolve().parent                    # board/<dataset>/ data folders
OUT_DIR = BOARD.parent / "images"                          # PDFs for the paper
RADIOML_CLASSES = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
                   "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
                   "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
                   "AM-DSB-SC", "FM", "GMSK", "OQPSK"]
SC_CLASSES = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
              "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
              "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
              "tree", "two", "up", "visual", "wow", "yes", "zero"]
SCIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# single-hue sequential ramp (blue): white = 0, dark blue = 1
SEQ_RAMP = LinearSegmentedColormap.from_list("seq_blue",
    ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK_PRIMARY, INK_MUTED = "#1f1f1e", "#6b6a66"


def predicted_class(logits):
    return logits.argmax(1)        # on ties the lowest index wins, as in the board program


def plot_confusion(true_class, predicted, class_names, title, stem, fig_size, font_size):
    n_classes = len(class_names)
    counts = np.zeros((n_classes, n_classes), int)          # rows = true class, columns = predicted
    np.add.at(counts, (true_class, predicted), 1)
    row_share = counts / counts.sum(1, keepdims=True)
    accuracy = np.trace(counts) / counts.sum()

    fig, ax = plt.subplots(figsize=fig_size)
    image = ax.imshow(row_share, cmap=SEQ_RAMP, vmin=0, vmax=1)
    for row in range(n_classes):
        for col in range(n_classes):
            if counts[row, col]:                            # numbers only in non-empty cells
                ax.text(col, row, counts[row, col], ha="center", va="center", fontsize=font_size,
                        color="#ffffff" if row_share[row, col] > 0.55 else INK_PRIMARY)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=90, fontsize=font_size + 1, color=INK_PRIMARY)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names, fontsize=font_size + 1, color=INK_PRIMARY)
    ax.set_xlabel("Predicted class", color=INK_PRIMARY); ax.set_ylabel("True class", color=INK_PRIMARY)
    ax.set_title(f"{title}\naccuracy {accuracy:.4f} ({np.trace(counts)}/{counts.sum()})", fontsize=10, color=INK_PRIMARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label("Share of true class (row-normalised)", color=INK_MUTED, fontsize=8)
    colorbar.outline.set_visible(False); colorbar.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"{stem}: accuracy {accuracy:.4f} ({np.trace(counts)}/{counts.sum()})")
    return counts


def check_golden(board_logits, golden_path, n_classes):
    """If the golden file has been generated, the board logits must match it."""
    if golden_path.is_file():
        golden = np.fromfile(golden_path, "<i4").reshape(-1, n_classes)
        assert (board_logits[:len(golden)] == golden).all(), f"board logits differ from {golden_path.name}"
        print(f"  board logits identical to {golden_path.name} ({len(golden)} examples)")


def print_top_confusions(counts, class_names, n_top):
    """Most frequent off-diagonal entries (true -> predicted)."""
    off_diag = counts.copy(); np.fill_diagonal(off_diag, 0)
    for flat in np.argsort(off_diag, axis=None)[::-1][:n_top]:
        r, c = divmod(flat, len(class_names))
        print(f"  {class_names[r]:>10s} -> {class_names[c]:<10s} {off_diag[r, c]:3d} / {counts[r].sum()}")


OUT_DIR.mkdir(exist_ok=True)

# --- RadioML: board logits, labels and SNRs of the 9984 frames ---------------
radioml_logits = np.fromfile(BOARD / "RadioML/logits_board_ZCU104_RadioML.bin", "<i4").reshape(-1, 24)
radioml_true = np.fromfile(BOARD / "RadioML/labels_RadioML_9984.bin", np.uint8)
radioml_snr = np.fromfile(BOARD / "RadioML/snr_RadioML_9984.bin", np.int8)
check_golden(radioml_logits, BOARD / "RadioML/logits_golden_RadioML_9984.bin", 24)
radioml_pred = predicted_class(radioml_logits)
for snr_value in (10, 20):
    in_snr = radioml_snr == snr_value
    plot_confusion(radioml_true[in_snr], radioml_pred[in_snr], RADIOML_CLASSES,
                   f"RadioML 2018.01A, SNR = +{snr_value} dB — ZCU104 board, {in_snr.sum() // 24} frames per class",
                   f"radioml_confusion_snr+{snr_value}dB", (9.2, 8.2), 6.5)

# --- Speech Commands: board logits on the official test split -----------------
sc_logits = np.fromfile(BOARD / "SpeechCommands/logits_board_ZCU104_SC.bin", "<i4").reshape(-1, 35)
sc_true = np.fromfile(BOARD / "SpeechCommands/labels_SC_11005.bin", np.uint8)
sc_counts = plot_confusion(sc_true, predicted_class(sc_logits), SC_CLASSES,
                           "Speech Commands v0.02 (35 classes), official test split — ZCU104 board",
                           "sc_confusion", (12.5, 11.2), 5)
print_top_confusions(sc_counts, SC_CLASSES, 12)

# --- sCIFAR: board logits on the 10000 test images ----------------------------
scifar_logits = np.fromfile(BOARD / "sCIFAR/logits_board_ZCU104_sCIFAR.bin", "<i4").reshape(-1, 10)
scifar_true = np.fromfile(BOARD / "sCIFAR/labels_sCIFAR_10000.bin", np.uint8)
check_golden(scifar_logits, BOARD / "sCIFAR/logits_golden_sCIFAR_1000.bin", 10)
scifar_counts = plot_confusion(scifar_true, predicted_class(scifar_logits), SCIFAR_CLASSES,
                               "sCIFAR (CIFAR-10, grayscale, 10 000 test images) — ZCU104 board",
                               "scifar_confusion", (7.4, 6.6), 8)
print_top_confusions(scifar_counts, SCIFAR_CLASSES, 8)
