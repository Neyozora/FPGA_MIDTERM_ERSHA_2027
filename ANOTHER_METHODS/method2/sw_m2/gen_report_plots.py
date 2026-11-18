"""Generate verified Method 2 figures in `reports_m2/` for the Indonesian Typst report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sw_m2.config import CLASS_NAMES, PROCESSED_M2_DIR, REPORTS_M2_DIR
from sw_m2.gui_app_m2 import int8_to_rgb_u8, load_m2_params
from sw_m2.train_qat import run_int8_inference


def main():
    REPORTS_M2_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(PROCESSED_M2_DIR / "cifar10_int8.npz")
    params = load_m2_params()

    # 1. 10-Class CIFAR-10 ROM Sample Grid
    x_hw_q, y_hw = d["x_hw_q"], d["y_hw"]
    hw_scores = run_int8_inference(x_hw_q, params)
    hw_preds = hw_scores.argmax(axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), dpi=150)
    for c, ax in enumerate(axes.flat):
        hits = np.where((y_hw == c) & (hw_preds == c))[0]
        idx = int(hits[0]) if len(hits) > 0 else c * 10
        ax.imshow(int8_to_rgb_u8(x_hw_q[idx]), interpolation="nearest")
        ax.set_title(f"{c}: {CLASS_NAMES[c]}\n(score={int(hw_scores[idx, c])})", fontsize=9)
        ax.axis("off")
    fig.suptitle("10 Citra Uji CIFAR-10 (32×32×3 RGB) pada On-Chip ROM DE10-Standard", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_M2_DIR / "cifar10_samples_m2.png", bbox_inches="tight")
    plt.close(fig)

    # 2. 10,000-Image Pure INT8 Confusion Matrix
    x_test_q, y_test = d["x_test_q"], d["y_test"]
    preds = []
    for i in range(0, len(y_test), 1000):
        preds.append(run_int8_inference(x_test_q[i : i + 1000], params).argmax(axis=1))
    preds = np.concatenate(preds)

    cm = np.zeros((10, 10), dtype=np.int32)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
    acc = 100.0 * np.trace(cm) / cm.sum()

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Prediksi Model INT8 FPGA", fontsize=9.5)
    ax.set_ylabel("Label Sebenarnya (Ground Truth)", fontsize=9.5)
    ax.set_title(f"Confusion Matrix Pure INT8 CIFAR-10 (10.000 Uji, Akurasi = {acc:.2f}%)", fontsize=10.5, fontweight="bold")
    for r in range(10):
        for c in range(10):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() * 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_M2_DIR / "int8_confusion_matrix_m2.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Method 1 vs Method 2 Performance Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=150)
    methods = ["Method 1\n(1-MAC, 2-Layer)", "Method 2\n(36-MAC, 4-Layer)"]
    cycles = [302978, 202538]
    fps = [165.0, 246.9]
    b1 = ax1.bar(methods, cycles, color=["#64748b", "#0284c7"], width=0.5)
    ax1.set_title("Siklus Clock per Citra (Lebih Kecil = Lebih Cepat)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Clock Cycles @ 50 MHz")
    ax1.bar_label(b1, fmt="%d", padding=3, fontsize=8.5)
    ax1.set_ylim(0, 355000)

    b2 = ax2.bar(methods, fps, color=["#64748b", "#10b981"], width=0.5)
    ax2.set_title("Throughput Hardware (FPS @ 50 MHz)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Frames Per Second (FPS)")
    ax2.bar_label(b2, fmt="%.1f FPS", padding=3, fontsize=8.5)
    ax2.set_ylim(0, 290)
    fig.tight_layout()
    fig.savefig(REPORTS_M2_DIR / "method1_vs_method2_chart.png", bbox_inches="tight")
    plt.close(fig)

    print("Generated Method 2 report plots in reports_m2/")


if __name__ == "__main__":
    main()
