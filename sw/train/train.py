"""
Milestone 3: FP32 training script for FashionCNN (32x32 zero-padded input).

Features:
    - Uses exact hardware-grid inputs x = q_in / 128 (q_in = pixel >> 1)
    - Trains on GPU (CUDA) by default when available
    - CosineAnnealingLR schedule with AdamW/Adam
    - Saves `checkpoints/fp32_best.pt` (highest validation accuracy) and
      `checkpoints/fp32_last.pt`
    - Logs epoch-by-epoch train/val loss, accuracy, learning rate, and time to
      `reports/training/training_log.csv` and `reports/training/training_history.json`
    - Plots `reports/training/training_curves.png`

Usage:
    python -m sw.train.train --epochs 60
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sw.config import CHECKPOINTS_DIR, IMG_C, IMG_H, IMG_W, SEED, TRAINING_REPORT_DIR
from sw.data.dataset import load_processed
from sw.data.preprocess import int8_to_float
from sw.model.cnn import SHAPES, FashionCNN, expected_macs, expected_param_counts
from sw.train.metrics import confusion_matrix_np, plot_training_curves


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_gpu_tensors(x_int8: np.ndarray, y_u8: np.ndarray, device: torch.device):
    """Preload dataset directly onto `device` for sub-second epochs."""
    x = torch.from_numpy(int8_to_float(x_int8)[:, None, :, :]).to(device)
    y = torch.from_numpy(y_u8.astype(np.int64)).to(device)
    return x, y


def train_one_epoch(model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor,
                    batch_size: int, optimizer: torch.optim.Optimizer,
                    scheduler, gen: torch.Generator) -> tuple:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n = x_tr.size(0)
    perm = torch.randperm(n, generator=gen, device=x_tr.device)
    total_loss = 0.0
    total_correct = 0

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        xb = x_tr[idx]
        yb = y_tr[idx]
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        (loss / yb.size(0)).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())

    return total_loss / n, total_correct / n


@torch.no_grad()
def eval_tensors(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                 batch_size: int = 2048) -> tuple:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n = x.size(0)
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, batch_size):
        xb = x[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        total_loss += float(criterion(logits, yb).item())
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
    return total_loss / n, total_correct / n


def run_training(epochs: int = 120, batch_size: int = 128, lr: float = 1e-2,
                 weight_decay: float = 0.0, seed: int = SEED, device_str: str = "auto",
                 checkpoint_dir: Path = CHECKPOINTS_DIR,
                 report_dir: Path = TRAINING_REPORT_DIR,
                 max_train_samples: int = 0) -> dict:
    set_seed(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    d = load_processed()
    x_tr_np, y_tr_np = d["x_train"], d["y_train"]
    if max_train_samples > 0:
        x_tr_np, y_tr_np = x_tr_np[:max_train_samples], y_tr_np[:max_train_samples]

    x_tr, y_tr = _make_gpu_tensors(x_tr_np, y_tr_np, device)
    x_va, y_va = _make_gpu_tensors(d["x_val"], d["y_val"], device)

    gen = torch.Generator(device=device).manual_seed(seed)
    steps_per_epoch = (x_tr.size(0) + batch_size - 1) // batch_size

    model = FashionCNN(check_shapes=True).to(device)
    # Kaiming normal initialization for ReLU convs
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.15, anneal_strategy="cos",
    )

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    t_start = time.perf_counter()

    print(f"Training FashionCNN on {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}) "
          f"| epochs={epochs} batch={batch_size} max_lr={lr} wd={weight_decay} seed={seed}")

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, x_tr, y_tr, batch_size, optimizer, scheduler, gen)
        cur_lr = float(optimizer.param_groups[0]["lr"])
        va_loss, va_acc = eval_tensors(model, x_va, y_va)
        dt = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "lr": cur_lr,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "epoch_sec": dt,
        }
        history.append(row)

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_acc": va_acc,
                "val_loss": va_loss,
                "train_acc": tr_acc,
                "train_loss": tr_loss,
                "config": {
                    "epochs": epochs, "batch_size": batch_size, "lr": lr,
                    "weight_decay": weight_decay, "seed": seed,
                    "input_shape": [IMG_C, IMG_H, IMG_W],
                    "shapes": {k: list(v) for k, v in SHAPES.items()},
                },
            }, checkpoint_dir / "fp32_best.pt")

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs or is_best:
            mark = " *best*" if is_best else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | lr={cur_lr:.5f} | "
                  f"train {tr_loss:.4f} / {tr_acc*100:5.2f}% | "
                  f"val {va_loss:.4f} / {va_acc*100:5.2f}% | {dt:.2f}s{mark}")

    total_sec = time.perf_counter() - t_start
    torch.save({
        "epoch": epochs,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "val_acc": history[-1]["val_acc"],
        "val_loss": history[-1]["val_loss"],
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
    }, checkpoint_dir / "fp32_last.pt")

    # ---- write CSV & JSON logs ----------------------------------------------
    csv_path = report_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    summary = {
        "config": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "seed": seed,
            "optimizer": "AdamW", "scheduler": "OneCycleLR (15% warmup + cosine anneal)",
            "init": "Kaiming normal",
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "input_shape": [IMG_C, IMG_H, IMG_W],
        },
        "model": {
            "total_params": sum(expected_param_counts().values()),
            "fp32_bytes": sum(expected_param_counts().values()) * 4,
            "int8_int32_bytes": 1384 + 34 * 4,
            "total_macs": sum(expected_macs().values()),
            "shapes": {k: list(v) for k, v in SHAPES.items()},
        },
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_acc": history[-1]["train_acc"],
        "total_training_sec": total_sec,
        "history": history,
    }
    (report_dir / "training_history.json").write_text(json.dumps(summary, indent=2))
    plot_training_curves(history, report_dir / "training_curves.png")

    print(f"Done in {total_sec:.1f}s | Best val accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    print(f"Saved {checkpoint_dir / 'fp32_best.pt'} and {report_dir / 'training_log.csv'}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args(argv)
    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 weight_decay=args.weight_decay, seed=args.seed, device_str=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
