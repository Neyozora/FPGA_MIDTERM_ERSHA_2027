"""
Milestone 3: Validation and Test Evaluation script for the trained FP32 model.

Loads `checkpoints/fp32_best.pt`, evaluates on train, validation, and test splits,
computes the 10x10 confusion matrix and per-class accuracy, and writes:
    - reports/training/eval_metrics.json
    - reports/training/confusion_matrix.png
    - reports/training/training_report.md

Usage:
    python -m sw.train.evaluate                    # evaluate val + test + train
    python -m sw.train.evaluate --split val        # validation only
    python -m sw.train.evaluate --split test       # test only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sw.config import CHECKPOINTS_DIR, CLASS_NAMES, NUM_CLASSES, TRAINING_REPORT_DIR
from sw.data.dataset import TEST_VECTORS_FILE, load_processed
from sw.data.preprocess import int8_to_float
from sw.model.cnn import SHAPES, FashionCNN, expected_macs, expected_param_counts
from sw.train.metrics import (confusion_matrix_np, evaluate_model, make_tensor_dataset,
                              per_class_metrics, plot_confusion_matrix)


def evaluate_checkpoint(checkpoint_path: Path = CHECKPOINTS_DIR / "fp32_best.pt",
                        report_dir: Path = TRAINING_REPORT_DIR,
                        device_str: str = "auto") -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found - run `python -m sw.train.train` first")
    report_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device_str == "auto" else device_str))

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FashionCNN(check_shapes=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    d = load_processed()
    pin = device.type == "cuda"
    splits_out = {}
    for split_name in ("train", "val", "test"):
        ds = make_tensor_dataset(d[f"x_{split_name}"], d[f"y_{split_name}"])
        loader = DataLoader(ds, batch_size=512, shuffle=False, pin_memory=pin)
        loss, acc, y_true, y_pred = evaluate_model(model, loader, device)
        cm = confusion_matrix_np(y_true, y_pred)
        pc = per_class_metrics(cm)
        splits_out[split_name] = {
            "num_samples": int(y_true.size),
            "loss": loss,
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),
            "per_class": pc,
        }
        print(f"[FP32 {split_name:5s}] loss={loss:.4f}  accuracy={acc*100:.2f}% "
              f"({int(round(acc * y_true.size))}/{y_true.size})")

    # ---- also evaluate on the 100 hardware test vectors ----------------------
    with np.load(TEST_VECTORS_FILE) as tv:
        tv_x = torch.from_numpy(int8_to_float(tv["images_int8"])[:, None, :, :]).to(device)
        tv_y = tv["labels"].astype(np.int64)
        tv_idx = tv["test_idx"].astype(np.int64)
    with torch.no_grad():
        tv_preds = model(tv_x).argmax(dim=1).cpu().numpy()
    tv_acc = float((tv_preds == tv_y).mean())
    print(f"[FP32 hw_tv] 100 hardware test vectors accuracy={tv_acc*100:.1f}% ({int((tv_preds == tv_y).sum())}/100)")

    # ---- plot test confusion matrix -----------------------------------------
    test_cm = np.array(splits_out["test"]["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(test_cm, report_dir / "confusion_matrix.png",
                          title=f"FP32 Test Confusion Matrix (Acc: {splits_out['test']['accuracy']*100:.2f}%)")

    # ---- assemble JSON & Markdown reports -----------------------------------
    param_counts = expected_param_counts()
    n_params = sum(param_counts.values())
    macs = expected_macs()
    history_file = report_dir / "training_history.json"
    train_meta = json.loads(history_file.read_text()) if history_file.exists() else {}

    result = {
        "checkpoint": str(checkpoint_path.name),
        "best_epoch": ckpt.get("epoch"),
        "training_config": ckpt.get("config", train_meta.get("config", {})),
        "model": {
            "parameter_counts": dict(param_counts),
            "total_params": n_params,
            "fp32_size_bytes": n_params * 4,
            "int8_weights_int32_bias_bytes": 1384 + 34 * 4,
            "macs_per_image": dict(macs),
            "total_macs": sum(macs.values()),
            "tensor_shapes_chw": {k: list(v) for k, v in SHAPES.items()},
        },
        "splits": splits_out,
        "hardware_test_vectors": {
            "count": int(tv_y.size),
            "accuracy": tv_acc,
            "test_idx": tv_idx.tolist(),
            "y_true": tv_y.tolist(),
            "y_pred_fp32": tv_preds.tolist(),
        },
    }
    (report_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2))

    # ---- Markdown report ----------------------------------------------------
    cfg = result["training_config"]
    L = [
        "# FP32 Training and Evaluation Report (Milestone 3)", "",
        "Generated by `python -m sw.train.evaluate`. Do not edit by hand.", "",
        "## 1. Summary Metrics", "",
        "| Split | Samples | Cross-Entropy Loss | Accuracy (%) | Correct / Total |",
        "|---|---|---|---|---|",
    ]
    for s in ("train", "val", "test"):
        st = splits_out[s]
        corr = sum(st["per_class"]["correct"])
        L.append(f"| **{s}** | {st['num_samples']:,} | {st['loss']:.4f} | "
                 f"**{st['accuracy']*100:.2f}%** | {corr:,} / {st['num_samples']:,} |")
    L.append(f"| **hardware test vectors** | 100 | – | **{tv_acc*100:.1f}%** | "
             f"{int((tv_preds == tv_y).sum())} / 100 |")

    L += [
        "", "## 2. Training Configuration", "",
        "| Setting | Value |", "|---|---|",
        f"| Checkpoint | `{result['checkpoint']}` (epoch {result['best_epoch']}) |",
        f"| Epochs | {cfg.get('epochs', '–')} |",
        f"| Batch size | {cfg.get('batch_size', '–')} |",
        f"| Optimizer | AdamW (max_lr={cfg.get('lr', '–')}, weight_decay={cfg.get('weight_decay', '–')}) |",
        f"| LR Scheduler | OneCycleLR (15% warmup + cosine anneal) |",
        f"| Weight Init | Kaiming Normal |",
        f"| Seed | {cfg.get('seed', '–')} |",
        f"| Input tensor | `1 × 32 × 32` (`28×28` + 2-px zero border, `x = (pixel >> 1) / 128`) |",
        f"| Total parameters | {n_params:,} (1,384 weights + 34 biases) |",
        f"| Model size | {n_params * 4:,} B (FP32) / 1,520 B (INT8 weights + INT32 biases) |",
        f"| Total MACs / image | {sum(macs.values()):,} |",
        "", "## 3. Inference Tensor Dimensions (Single Image, `N = 1`)", "",
        "| Stage | Shape `(C, H, W)` | Elements |", "|---|---|---|",
    ]
    for k, shp in SHAPES.items():
        L.append(f"| `{k}` | `{tuple(shp)}` | {int(np.prod(shp)):,} |")

    L += [
        "", "## 4. Per-Class Test Accuracy (Official 10,000-Image Test Split)", "",
        "| Class ID | Class Name | Correct / Support | Test Accuracy (%) | Precision (%) |",
        "|---|---|---|---|---|",
    ]
    pc_test = splits_out["test"]["per_class"]
    for c in range(NUM_CLASSES):
        L.append(f"| {c} | {CLASS_NAMES[c]} | {pc_test['correct'][c]} / {pc_test['support'][c]} | "
                 f"{pc_test['accuracy'][c]*100:.2f}% | {pc_test['precision'][c]*100:.2f}% |")

    L += ["", "## 5. Test Confusion Matrix (Rows = True, Columns = Predicted)", ""]
    hdr = "| True \\ Pred | " + " | ".join(f"**{c}**" for c in range(NUM_CLASSES)) + " |"
    sep = "|---|" + "|".join(["---"] * NUM_CLASSES) + "|"
    L += [hdr, sep]
    for r in range(NUM_CLASSES):
        row_cells = " | ".join(f"**{test_cm[r, c]}**" if r == c else str(test_cm[r, c])
                               for c in range(NUM_CLASSES))
        L.append(f"| **{r} ({CLASS_NAMES[r]})** | {row_cells} |")

    L += [
        "", "## 6. Figures", "",
        "![Training curves](training_curves.png)", "",
        "![Confusion matrix](confusion_matrix.png)", "",
    ]
    (report_dir / "training_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {report_dir / 'training_report.md'} and {report_dir / 'eval_metrics.json'}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "fp32_best.pt")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args(argv)
    evaluate_checkpoint(checkpoint_path=args.checkpoint, device_str=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
