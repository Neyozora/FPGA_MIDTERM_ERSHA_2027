"""
Shared evaluation utilities for Milestone 3 (FP32) and Milestone 4 (INT8):
    - Tensor dataset creation from processed INT8 arrays
    - Loss & accuracy computation over a DataLoader
    - Confusion matrix (10x10) and per-class accuracy / precision / recall
    - Plotting utilities (training curves, confusion matrix heatmap)
"""
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sw.config import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from sw.data.preprocess import int8_to_float  # noqa: E402


def make_tensor_dataset(x_int8: np.ndarray, y_u8: np.ndarray) -> TensorDataset:
    """Convert [N, 32, 32] int8 and [N] uint8 into a float32 TensorDataset ([N,1,32,32], int64)."""
    x_f32 = int8_to_float(x_int8)[:, None, :, :]
    return TensorDataset(torch.from_numpy(x_f32), torch.from_numpy(y_u8.astype(np.int64)))


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray,
                        num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Compute [num_classes, num_classes] confusion matrix (rows = true, cols = predicted)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    assert y_true.min(initial=0) >= 0 and y_true.max(initial=0) < num_classes
    assert y_pred.min(initial=0) >= 0 and y_pred.max(initial=0) < num_classes
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class_metrics(cm: np.ndarray) -> Dict[str, list]:
    """From a confusion matrix [C, C], return per-class support, correct, accuracy, precision."""
    cm = np.asarray(cm, dtype=np.int64)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    correct = np.diag(cm)
    acc = np.where(support > 0, correct / np.maximum(support, 1), 0.0)
    prec = np.where(pred_count > 0, correct / np.maximum(pred_count, 1), 0.0)
    return {
        "support": support.tolist(),
        "correct": correct.tolist(),
        "accuracy": [float(v) for v in acc],
        "precision": [float(v) for v in prec],
    }


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate `model` on `loader`.
    Returns (mean_cross_entropy_loss, accuracy, y_true_np, y_pred_np).
    """
    was_training = model.training
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_true = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        total_loss += float(criterion(logits, yb).item())
        preds = logits.argmax(dim=1)
        total_correct += int((preds == yb).sum().item())
        total_samples += int(yb.numel())
        all_true.append(yb.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    if was_training:
        model.train()

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return total_loss / total_samples, total_correct / total_samples, y_true, y_pred


def plot_training_curves(history: list, out_path) -> None:
    """Plot epoch loss and accuracy curves for train and validation splits."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train loss", linewidth=1.8)
    ax1.plot(epochs, [h["val_loss"] for h in history], label="Val loss", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training & Validation Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [h["train_acc"] * 100 for h in history], label="Train acc", linewidth=1.8)
    ax2.plot(epochs, [h["val_acc"] * 100 for h in history], label="Val acc", linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out_path, title: str = "Test Confusion Matrix (FP32)") -> None:
    """Render a 10x10 confusion matrix with counts in every cell."""
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = int(cm[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
