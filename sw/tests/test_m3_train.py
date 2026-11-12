"""
Milestone 3 tests: training utilities, confusion matrix, per-class metrics,
checkpoint saving/loading, and trained FP32 model evaluation.

    python -m sw.tests.test_m3_train
    python -m pytest sw/tests/test_m3_train.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from sw.config import CHECKPOINTS_DIR, NUM_CLASSES, TRAINING_REPORT_DIR
from sw.data.dataset import PROCESSED_FILE
from sw.model.cnn import FashionCNN
from sw.tests._runner import need, run_all
from sw.train.metrics import confusion_matrix_np, make_tensor_dataset, per_class_metrics
from sw.train.train import run_training


def test_confusion_matrix_and_per_class_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)
    cm = confusion_matrix_np(y_true, y_pred, num_classes=3)
    expected = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 1],
    ])
    assert np.array_equal(cm, expected)
    pc = per_class_metrics(cm)
    assert pc["support"] == [2, 2, 2]
    assert pc["correct"] == [1, 2, 1]
    assert np.allclose(pc["accuracy"], [0.5, 1.0, 0.5])
    assert np.allclose(pc["precision"], [0.5, 2 / 3, 1.0])


def test_make_tensor_dataset_shapes_and_range():
    x_i8 = np.array([[[0, 64], [127, 32]]], dtype=np.int8)
    y_u8 = np.array([5], dtype=np.uint8)
    ds = make_tensor_dataset(x_i8, y_u8)
    x, y = ds[0]
    assert tuple(x.shape) == (1, 2, 2) and x.dtype == torch.float32
    assert int(y) == 5 and y.dtype == torch.int64
    assert np.allclose(x.numpy(), x_i8[None] / 128.0)


def test_short_training_smoke_saves_checkpoint():
    need(PROCESSED_FILE)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ckpt_dir = tmp_path / "ckpt"
        rep_dir = tmp_path / "rep"
        summary = run_training(epochs=1, batch_size=256, lr=3e-3, seed=42,
                               checkpoint_dir=ckpt_dir, report_dir=rep_dir,
                               max_train_samples=1024)
        assert (ckpt_dir / "fp32_best.pt").exists()
        assert (rep_dir / "training_log.csv").exists()
        assert (rep_dir / "training_curves.png").exists()
        assert 0.0 <= summary["best_val_acc"] <= 1.0


def test_trained_fp32_checkpoint_and_metrics():
    ckpt_path = CHECKPOINTS_DIR / "fp32_best.pt"
    eval_path = TRAINING_REPORT_DIR / "eval_metrics.json"
    need(ckpt_path)
    need(eval_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = FashionCNN()
    m.load_state_dict(ckpt["model_state_dict"])

    metrics = json.loads(eval_path.read_text())
    assert metrics["model"]["total_params"] == 1418
    assert metrics["model"]["total_macs"] == 259648

    for split_name, expected_n in (("train", 54000), ("val", 6000), ("test", 10000)):
        st = metrics["splits"][split_name]
        cm = np.array(st["confusion_matrix"], dtype=np.int64)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert int(cm.sum()) == expected_n
        assert np.isclose(np.diag(cm).sum() / expected_n, st["accuracy"], atol=1e-6)

    # Sanity check: trained FP32 model must be well above random (10%) and reach >= 80% on Fashion-MNIST
    assert metrics["splits"]["test"]["accuracy"] >= 0.80, metrics["splits"]["test"]["accuracy"]
    assert (TRAINING_REPORT_DIR / "confusion_matrix.png").exists()
    assert (TRAINING_REPORT_DIR / "training_report.md").exists()


if __name__ == "__main__":
    sys.exit(run_all(globals()))
