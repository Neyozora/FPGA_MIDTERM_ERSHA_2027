"""
Milestone 4 pipeline:
    1. Load trained FP32 model (`checkpoints/fp32_best.pt`)
    2. Run Post-Training Quantization (PTQ) with percentile calibration sweep
    3. Run Quantization-Aware Training (QAT) fine-tuning and extract integer parameters
    4. Evaluate FP32, PTQ INT8, and QAT INT8 on train, validation, test (10k),
       and the 100 hardware test vectors
    5. Save the best INT8 parameter bundle to `checkpoints/int8_quantized.npz`
       and write `reports/quantization/quantization_report.md` + figures

Usage:
    python -m sw.quant.evaluate_quant
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from sw.config import (CHECKPOINTS_DIR, CLASS_NAMES, NUM_CLASSES, QUANT_REPORT_DIR, SEED)  # noqa: E402
from sw.data.dataset import TEST_VECTORS_FILE, load_processed  # noqa: E402
from sw.data.preprocess import int8_to_float  # noqa: E402
from sw.model.cnn import FashionCNN  # noqa: E402
from sw.quant.quantize import (QATWrapper, calibrate_and_quantize, forward_int8_torch)  # noqa: E402
from sw.train.metrics import (confusion_matrix_np, per_class_metrics, plot_confusion_matrix)  # noqa: E402
from sw.train.train import set_seed  # noqa: E402

QUANT_NPZ_PATH = CHECKPOINTS_DIR / "int8_quantized.npz"


@torch.no_grad()
def eval_int8_split(x_int8: np.ndarray, y_u8: np.ndarray, qparams: dict,
                    device: torch.device, batch_size: int = 2048) -> dict:
    n = x_int8.shape[0]
    x_t = torch.from_numpy(x_int8)
    preds = []
    for i in range(0, n, batch_size):
        out = forward_int8_torch(x_t[i:i + batch_size], qparams, device=device)
        preds.append(out["pred"].cpu().numpy())
    y_pred = np.concatenate(preds).astype(np.int64)
    y_true = y_u8.astype(np.int64)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix_np(y_true, y_pred)
    pc = per_class_metrics(cm)
    return {
        "num_samples": int(n),
        "accuracy": acc,
        "correct": int((y_pred == y_true).sum()),
        "confusion_matrix": cm.tolist(),
        "per_class": pc,
        "y_pred": y_pred,
    }


@torch.no_grad()
def eval_fp32_preds(model: FashionCNN, x_int8: np.ndarray, device: torch.device,
                    batch_size: int = 2048) -> np.ndarray:
    model.eval()
    x_f32 = torch.from_numpy(int8_to_float(x_int8)[:, None, :, :]).to(device)
    preds = []
    for i in range(0, x_f32.size(0), batch_size):
        preds.append(model(x_f32[i:i + batch_size]).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


def run_qat_finetuning(base_model: FashionCNN, init_qparams: dict, x_tr_i8: np.ndarray,
                       y_tr_u8: np.ndarray, x_va_i8: np.ndarray, y_va_u8: np.ndarray,
                       device: torch.device, epochs: int = 20, batch_size: int = 128,
                       lr: float = 1e-3, seed: int = SEED) -> Tuple[FashionCNN, dict]:
    set_seed(seed)
    qat = QATWrapper(base_model, init_qparams).to(device)
    x_tr = torch.from_numpy(int8_to_float(x_tr_i8)[:, None, :, :]).to(device)
    y_tr = torch.from_numpy(y_tr_u8.astype(np.int64)).to(device)
    x_calib = x_tr[:8192]

    optimizer = torch.optim.AdamW(qat.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator(device=device).manual_seed(seed)

    best_val_acc = -1.0
    best_qparams = None
    best_state = None

    for ep in range(1, epochs + 1):
        qat.train()
        perm = torch.randperm(x_tr.size(0), generator=gen, device=device)
        for i in range(0, x_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = qat(x_tr[idx])
            loss = criterion(logits, y_tr[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate TRUE integer accuracy on validation set at each QAT epoch
        for pct, wpct in ((99.95, 100.0), (99.9, 99.0), (99.8, 99.0)):
            cand_qparams = calibrate_and_quantize(qat.model, x_calib, percentile=pct, weight_clip_pct=wpct)
            va_res = eval_int8_split(x_va_i8, y_va_u8, cand_qparams, device)
            if va_res["accuracy"] > best_val_acc:
                best_val_acc = va_res["accuracy"]
                best_qparams = cand_qparams
                best_state = {k: v.detach().cpu().clone() for k, v in qat.model.state_dict().items()}

    best_model = FashionCNN(check_shapes=True).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()
    return best_model, best_qparams


def plot_weight_distributions(fp32_model: FashionCNN, qparams: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    layers = [
        ("conv1", fp32_model.conv1.weight.detach().cpu().numpy().ravel(),
         qparams["conv1_weight"].ravel(), float(qparams["s_w1"])),
        ("conv2", fp32_model.conv2.weight.detach().cpu().numpy().ravel(),
         qparams["conv2_weight"].ravel(), float(qparams["s_w2"])),
        ("fc", fp32_model.fc.weight.detach().cpu().numpy().ravel(),
         qparams["fc_weight"].ravel(), float(qparams["s_wfc"])),
    ]
    for ax, (name, w_f, w_q, s_w) in zip(axes, layers):
        ax.hist(w_f, bins=35, alpha=0.55, label="FP32 W", density=True)
        ax.hist(w_q.astype(np.float64) * s_w, bins=35, alpha=0.55, label="Dequant INT8 W", density=True)
        ax.set_title(f"{name} weights (s_w={s_w:.4f})")
        ax.set_xlabel("Weight value")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_quantization_pipeline(qat_epochs: int = 20, device_str: str = "auto") -> dict:
    QUANT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device_str == "auto" else device_str))

    ckpt = torch.load(CHECKPOINTS_DIR / "fp32_best.pt", map_location="cpu", weights_only=False)
    fp32_model = FashionCNN(check_shapes=True)
    fp32_model.load_state_dict(ckpt["model_state_dict"])
    fp32_model.to(device).eval()

    d = load_processed()
    x_calib = torch.from_numpy(int8_to_float(d["x_train"][:8192])[:, None, :, :]).to(device)

    # ---- 1. Post-Training Quantization (PTQ) sweep over percentiles ----------
    best_ptq_val = -1.0
    best_ptq_qparams = None
    for pct in (99.5, 99.8, 99.9, 99.95, 100.0):
        for wpct in (99.0, 99.5, 100.0):
            qp = calibrate_and_quantize(fp32_model, x_calib, percentile=pct, weight_clip_pct=wpct)
            va = eval_int8_split(d["x_val"], d["y_val"], qp, device)
            if va["accuracy"] > best_ptq_val:
                best_ptq_val = va["accuracy"]
                best_ptq_qparams = qp
                print(f"[PTQ calib act={pct:6.2f}% w={wpct:5.1f}%] val INT8 acc = {va['accuracy']*100:.2f}% *best*")

    ptq_test = eval_int8_split(d["x_test"], d["y_test"], best_ptq_qparams, device)
    print(f"[PTQ best] val={best_ptq_val*100:.2f}%  test={ptq_test['accuracy']*100:.2f}%")

    # ---- 2. Quantization-Aware Training (QAT) fine-tuning --------------------
    t0 = time.perf_counter()
    qat_model, qat_qparams = run_qat_finetuning(
        fp32_model, best_ptq_qparams, d["x_train"], d["y_train"],
        d["x_val"], d["y_val"], device, epochs=qat_epochs,
    )
    qat_sec = time.perf_counter() - t0
    qat_val = eval_int8_split(d["x_val"], d["y_val"], qat_qparams, device)
    qat_test = eval_int8_split(d["x_test"], d["y_test"], qat_qparams, device)
    print(f"[QAT {qat_epochs}ep ({qat_sec:.1f}s)] val={qat_val['accuracy']*100:.2f}%  "
          f"test={qat_test['accuracy']*100:.2f}%")

    # ---- 3. Select winner by validation accuracy -----------------------------
    if qat_val["accuracy"] >= best_ptq_val:
        selected_method = "QAT"
        selected_qparams = qat_qparams
        selected_float_model = qat_model
    else:
        selected_method = "PTQ"
        selected_qparams = best_ptq_qparams
        selected_float_model = fp32_model

    print(f"Selected INT8 model: {selected_method}")
    np.savez_compressed(QUANT_NPZ_PATH, **selected_qparams,
                        method=np.array(selected_method))
    torch.save({
        "method": selected_method,
        "model_state_dict": {k: v.cpu() for k, v in selected_float_model.state_dict().items()},
    }, CHECKPOINTS_DIR / "int8_source_float.pt")

    # ---- 4. Full evaluation across all splits + agreement vs FP32 ------------
    fp32_metrics = json.loads(( Path(CHECKPOINTS_DIR).parents[0] / "reports/training/eval_metrics.json" ).read_text())
    splits_int8 = {}
    for s in ("train", "val", "test"):
        res = eval_int8_split(d[f"x_{s}"], d[f"y_{s}"], selected_qparams, device)
        fp32_pred = eval_fp32_preds(fp32_model, d[f"x_{s}"], device)
        agree = float((res["y_pred"] == fp32_pred).mean())
        res_clean = {k: v for k, v in res.items() if k != "y_pred"}
        res_clean["agreement_with_fp32"] = agree
        res_clean["fp32_accuracy"] = fp32_metrics["splits"][s]["accuracy"]
        res_clean["accuracy_delta_vs_fp32"] = res["accuracy"] - res_clean["fp32_accuracy"]
        splits_int8[s] = res_clean
        print(f"[INT8 {s:5s}] acc={res['accuracy']*100:.2f}% "
              f"(FP32 {res_clean['fp32_accuracy']*100:.2f}%, "
              f"delta {res_clean['accuracy_delta_vs_fp32']*100:+.2f}%, "
              f"agree {agree*100:.2f}%)")

    # Hardware 100 test vectors
    with np.load(TEST_VECTORS_FILE) as tv:
        tv_i8 = tv["images_int8"]
        tv_y = tv["labels"].astype(np.int64)
    tv_res = eval_int8_split(tv_i8, tv_y, selected_qparams, device)
    tv_fp32_pred = eval_fp32_preds(fp32_model, tv_i8, device)
    tv_agree = float((tv_res["y_pred"] == tv_fp32_pred).mean())

    # Range check of intermediate accumulators on the 10k test set
    out_test = forward_int8_torch(torch.from_numpy(d["x_test"]), selected_qparams, device=device)
    acc_ranges = {
        "conv1_acc": [int(out_test["conv1_acc"].min()), int(out_test["conv1_acc"].max())],
        "conv2_acc": [int(out_test["conv2_acc"].min()), int(out_test["conv2_acc"].max())],
        "gap_sum": [int(out_test["gap_sum"].min()), int(out_test["gap_sum"].max())],
        "fc_acc": [int(out_test["fc_acc"].min()), int(out_test["fc_acc"].max())],
    }

    # Plots
    test_cm = np.array(splits_int8["test"]["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(test_cm, QUANT_REPORT_DIR / "int8_confusion_matrix.png",
                          title=f"INT8 ({selected_method}) Test Confusion Matrix (Acc: {splits_int8['test']['accuracy']*100:.2f}%)")
    plot_weight_distributions(selected_float_model, selected_qparams,
                              QUANT_REPORT_DIR / "weight_distributions.png")

    # Assemble JSON summary
    qp_summary = {
        "input": {
            "scale": float(selected_qparams["s_in"]),
            "zero_point": 0,
            "q_range": [0, 127],
            "formula": "q_in = pixel >> 1 (with 2-px zero border)",
        },
        "conv1": {
            "weight_scale": float(selected_qparams["s_w1"]),
            "bias_scale": float(selected_qparams["s_b1"]),
            "output_scale": float(selected_qparams["s_a1"]),
            "zero_point": 0,
            "mult_M": int(selected_qparams["conv1_mult"]),
            "shift_n": int(selected_qparams["conv1_shift"]),
            "effective_ratio": int(selected_qparams["conv1_mult"]) / (1 << int(selected_qparams["conv1_shift"])),
            "ideal_ratio": float(selected_qparams["s_in"] * selected_qparams["s_w1"] / selected_qparams["s_a1"]),
            "rounding": "round-half-up: (acc * M + 2^(n-1)) >>> n",
            "saturation": "ReLU + INT8 clamp to [0, 127]",
        },
        "conv2": {
            "weight_scale": float(selected_qparams["s_w2"]),
            "bias_scale": float(selected_qparams["s_b2"]),
            "output_scale": float(selected_qparams["s_a2"]),
            "zero_point": 0,
            "mult_M": int(selected_qparams["conv2_mult"]),
            "shift_n": int(selected_qparams["conv2_shift"]),
            "effective_ratio": int(selected_qparams["conv2_mult"]) / (1 << int(selected_qparams["conv2_shift"])),
            "ideal_ratio": float(selected_qparams["s_a1"] * selected_qparams["s_w2"] / selected_qparams["s_a2"]),
            "rounding": "round-half-up: (acc * M + 2^(n-1)) >>> n",
            "saturation": "ReLU + INT8 clamp to [0, 127]",
        },
        "gap": {
            "input_scale": float(selected_qparams["s_a2"]),
            "output_scale": float(selected_qparams["s_gap"]),
            "zero_point": 0,
            "mult_M": int(selected_qparams["gap_mult"]),
            "shift_n": int(selected_qparams["gap_shift"]),
            "rounding": "exact round(sum / 169): (sum * 776 + 2^16) >> 17",
            "saturation": "INT8 [0, 127]",
        },
        "fc": {
            "weight_scale": float(selected_qparams["s_wfc"]),
            "bias_scale": float(selected_qparams["s_bfc"]),
            "output_scale": float(selected_qparams["s_fc"]),
            "zero_point": 0,
            "mult_M": int(selected_qparams["fc_mult"]),
            "shift_n": int(selected_qparams["fc_shift"]),
            "effective_ratio": int(selected_qparams["fc_mult"]) / (1 << int(selected_qparams["fc_shift"])),
            "ideal_ratio": float(selected_qparams["s_a2"] * selected_qparams["s_wfc"] / selected_qparams["s_fc"]),
            "rounding": "round-half-up: (acc * M + 2^(n-1)) >>> n",
            "saturation": "signed INT8 clamp to [-128, 127]",
        },
    }

    report_json = {
        "selected_method": selected_method,
        "ptq_summary": {
            "val_accuracy": best_ptq_val,
            "test_accuracy": ptq_test["accuracy"],
            "calibration_percentile": float(best_ptq_qparams["percentile"]),
        },
        "qat_summary": {
            "epochs": qat_epochs,
            "val_accuracy": qat_val["accuracy"],
            "test_accuracy": qat_test["accuracy"],
        },
        "quantization_parameters": qp_summary,
        "accumulator_observed_ranges_test10k": acc_ranges,
        "splits": splits_int8,
        "hardware_test_vectors": {
            "count": 100,
            "fp32_accuracy": fp32_metrics["hardware_test_vectors"]["accuracy"],
            "int8_accuracy": tv_res["accuracy"],
            "agreement_with_fp32": tv_agree,
            "y_true": tv_y.tolist(),
            "y_pred_int8": tv_res["y_pred"].tolist(),
        },
    }
    (QUANT_REPORT_DIR / "quant_metrics.json").write_text(json.dumps(report_json, indent=2))

    # Write Markdown Report
    L = [
        "# INT8 Quantization and Evaluation Report (Milestone 4)", "",
        "Generated by `python -m sw.quant.evaluate_quant`. Do not edit by hand.",
        "Full mathematical specification: [`docs/quantization_spec.md`](../../docs/quantization_spec.md).", "",
        "## 1. FP32 vs INT8 (PTQ & QAT) Accuracy Comparison", "",
        "| Split | FP32 Baseline | PTQ INT8 | QAT INT8 | Selected INT8 (`" + selected_method + "`) | Delta vs FP32 | Classification Agreement vs FP32 |",
        "|---|---|---|---|---|---|---|",
        f"| **Train (54,000)** | {splits_int8['train']['fp32_accuracy']*100:.2f}% | – | – | **{splits_int8['train']['accuracy']*100:.2f}%** | {splits_int8['train']['accuracy_delta_vs_fp32']*100:+.2f}% | {splits_int8['train']['agreement_with_fp32']*100:.2f}% |",
        f"| **Validation (6,000)** | {splits_int8['val']['fp32_accuracy']*100:.2f}% | {best_ptq_val*100:.2f}% | {qat_val['accuracy']*100:.2f}% | **{splits_int8['val']['accuracy']*100:.2f}%** | {splits_int8['val']['accuracy_delta_vs_fp32']*100:+.2f}% | {splits_int8['val']['agreement_with_fp32']*100:.2f}% |",
        f"| **Test (10,000)** | {splits_int8['test']['fp32_accuracy']*100:.2f}% | {ptq_test['accuracy']*100:.2f}% | {qat_test['accuracy']*100:.2f}% | **{splits_int8['test']['accuracy']*100:.2f}%** | {splits_int8['test']['accuracy_delta_vs_fp32']*100:+.2f}% | {splits_int8['test']['agreement_with_fp32']*100:.2f}% |",
        f"| **HW Test Vectors (100)** | {fp32_metrics['hardware_test_vectors']['accuracy']*100:.1f}% | – | – | **{tv_res['accuracy']*100:.1f}%** | {(tv_res['accuracy'] - fp32_metrics['hardware_test_vectors']['accuracy'])*100:+.1f}% | {tv_agree*100:.1f}% |",
        "",
        "## 2. Layer-by-Layer Fixed-Point & Quantization Parameters", "",
        "| Layer | Weight Scale `s_w` | Bias Scale `s_b` | Output Scale `s_out` | Zero Point | Multiplier `M` | Shift `n` | Rounding | Saturation |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| `input` | – | – | {qp_summary['input']['scale']:.7f} (`2^-7`) | 0 | `1` | `1` (`pixel >> 1`) | Truncate LSB | `int8 [0, 127]` |",
        f"| `conv1` | {qp_summary['conv1']['weight_scale']:.6e} | {qp_summary['conv1']['bias_scale']:.6e} | {qp_summary['conv1']['output_scale']:.6e} | 0 | `{qp_summary['conv1']['mult_M']}` | `{qp_summary['conv1']['shift_n']}` | Round-half-up | ReLU `int8 [0, 127]` |",
        f"| `conv2` | {qp_summary['conv2']['weight_scale']:.6e} | {qp_summary['conv2']['bias_scale']:.6e} | {qp_summary['conv2']['output_scale']:.6e} | 0 | `{qp_summary['conv2']['mult_M']}` | `{qp_summary['conv2']['shift_n']}` | Round-half-up | ReLU `int8 [0, 127]` |",
        f"| `gap` | – | – | {qp_summary['gap']['output_scale']:.6e} | 0 | `{qp_summary['gap']['mult_M']}` | `{qp_summary['gap']['shift_n']}` | Exact `round(sum/169)` | `int8 [0, 127]` |",
        f"| `fc` | {qp_summary['fc']['weight_scale']:.6e} | {qp_summary['fc']['bias_scale']:.6e} | {qp_summary['fc']['output_scale']:.6e} | 0 | `{qp_summary['fc']['mult_M']}` | `{qp_summary['fc']['shift_n']}` | Round-half-up | `int8 [-128, 127]` |",
        "",
        "## 3. Observed INT32 Accumulator Ranges on 10,000 Test Images", "",
        "| Accumulator | Observed Min | Observed Max | Worst-Case Theoretical Bound | Fits in Signed INT32 (`±2.14e9`) |",
        "|---|---|---|---|---|",
        f"| `conv1_acc` | {acc_ranges['conv1_acc'][0]:,} | {acc_ranges['conv1_acc'][1]:,} | ±146,304 (+ bias) | Yes (< 18 bits) |",
        f"| `conv2_acc` | {acc_ranges['conv2_acc'][0]:,} | {acc_ranges['conv2_acc'][1]:,} | ±1,170,432 (+ bias) | Yes (< 21 bits) |",
        f"| `gap_sum` | {acc_ranges['gap_sum'][0]:,} | {acc_ranges['gap_sum'][1]:,} | [0, 21,463] | Yes (< 15 bits unsigned) |",
        f"| `fc_acc` | {acc_ranges['fc_acc'][0]:,} | {acc_ranges['fc_acc'][1]:,} | ±262,144 (+ bias) | Yes (< 19 bits) |",
        "",
        "## 4. Per-Class Test Accuracy: FP32 vs INT8", "",
        "| Class ID | Class Name | FP32 Test Acc (%) | INT8 Test Acc (%) | Delta (%) |",
        "|---|---|---|---|---|",
    ]
    pc_fp32 = fp32_metrics["splits"]["test"]["per_class"]["accuracy"]
    pc_int8 = splits_int8["test"]["per_class"]["accuracy"]
    for c in range(NUM_CLASSES):
        d_c = (pc_int8[c] - pc_fp32[c]) * 100
        L.append(f"| {c} | {CLASS_NAMES[c]} | {pc_fp32[c]*100:.2f}% | "
                 f"**{pc_int8[c]*100:.2f}%** | {d_c:+.2f}% |")

    L += [
        "", "## 5. Figures", "",
        "![Weight distributions](weight_distributions.png)", "",
        "![INT8 Confusion Matrix](int8_confusion_matrix.png)", "",
    ]
    (QUANT_REPORT_DIR / "quantization_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {QUANT_NPZ_PATH} and {QUANT_REPORT_DIR / 'quantization_report.md'}")
    return report_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qat-epochs", type=int, default=20)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args(argv)
    run_quantization_pipeline(qat_epochs=args.qat_epochs, device_str=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
