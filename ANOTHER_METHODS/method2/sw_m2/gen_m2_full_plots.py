"""Generate 10x10 100-image HW vector grid, weight distribution histograms, and exact split accuracies for Method 2."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sw_m2.config import CHECKPOINTS_M2_DIR, CLASS_NAMES, PROCESSED_M2_DIR, REPORTS_M2_DIR
from sw_m2.gui_app_m2 import int8_to_rgb_u8, load_m2_params
from sw_m2.model import CifarCNNQAT
from sw_m2.train_qat import run_int8_inference


def eval_int8_split(x_q, y, params):
    preds = []
    for i in range(0, len(y), 1000):
        preds.append(run_int8_inference(x_q[i : i + 1000], params).argmax(axis=1))
    preds = np.concatenate(preds)
    return 100.0 * float((preds == y).mean())


def main():
    REPORTS_M2_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(PROCESSED_M2_DIR / "cifar10_int8.npz")
    params = load_m2_params()

    # 1. Exact Pure INT8 Accuracies across Train, Val, Test, HW-100
    acc_train = eval_int8_split(d["x_train_q"], d["y_train"], params)
    acc_val = eval_int8_split(d["x_val_q"], d["y_val"], params)
    acc_test = eval_int8_split(d["x_test_q"], d["y_test"], params)
    acc_hw = eval_int8_split(d["x_hw_q"], d["y_hw"], params)

    split_metrics = {
        "train_int8": round(acc_train, 2),
        "val_int8": round(acc_val, 2),
        "test_int8": round(acc_test, 2),
        "hw100_int8": round(acc_hw, 2),
    }
    (REPORTS_M2_DIR / "split_int8_metrics.json").write_text(json.dumps(split_metrics, indent=2))
    print("Split INT8 accuracies:", split_metrics)

    # 2. 10x10 Grid of all 100 CIFAR-10 Hardware Test Vectors
    x_hw_q = d["x_hw_q"]
    fig, axes = plt.subplots(10, 10, figsize=(8.5, 8.8), dpi=150)
    for r in range(10):
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(int8_to_rgb_u8(x_hw_q[r * 10 + c]), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.set_ylabel(CLASS_NAMES[r], rotation=0, ha="right", va="center", fontsize=8, labelpad=6)
    fig.suptitle("CIFAR-10 Test Set: 10 Citra Pertama per Kelas (100 Vektor Uji Hardware, INT8 MSB-Flip)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    fig.savefig(REPORTS_M2_DIR / "hw_vectors_100_grid_m2.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Weight Distribution Histograms (FP32 BN-Folded vs Dequantized INT8)
    qat_model = CifarCNNQAT()
    qat_model.load_state_dict(torch.load(CHECKPOINTS_M2_DIR / "cifar_qat_m2.pt", map_location="cpu"))

    layers = [
        ("conv1", qat_model.conv1.conv.weight.detach().numpy()),
        ("conv2", qat_model.conv2.conv.weight.detach().numpy()),
        ("conv3", qat_model.conv3.conv.weight.detach().numpy()),
        ("conv4", qat_model.conv4.conv.weight.detach().numpy()),
        ("fc", qat_model.fc.fc.weight.detach().numpy()),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.8), dpi=150)
    for ax, (name, w_fp) in zip(axes, layers):
        w_q = params[name]["w_q"].astype(np.float32)
        if name == "fc":
            w_max = np.maximum(np.abs(w_fp).max(axis=1, keepdims=True), 1e-8)
        else:
            w_max = np.maximum(np.abs(w_fp).max(axis=(1, 2, 3), keepdims=True), 1e-8)
        w_deq = w_q * (w_max / 127.0)
        ax.hist(w_fp.ravel(), bins=35, density=True, alpha=0.55, label="FP32 (Folded)", color="#3b82f6")
        ax.hist(w_deq.ravel(), bins=35, density=True, alpha=0.55, label="Dequant INT8", color="#f97316")
        ax.set_title(f"{name.upper()} weights", fontsize=9, fontweight="bold")
        ax.set_xlabel("Weight value", fontsize=8)
        ax.tick_params(labelsize=7)
        if name == "conv1":
            ax.legend(fontsize=7)
    fig.suptitle("Distribusi Bobot FP32 (BN-Folded) vs. Dequantized Per-Channel INT8 (Method 2)", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REPORTS_M2_DIR / "weight_distributions_m2.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated hw_vectors_100_grid_m2.png and weight_distributions_m2.png")


if __name__ == "__main__":
    main()
