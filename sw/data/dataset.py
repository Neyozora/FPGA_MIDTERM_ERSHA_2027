"""
Loading of the raw Fashion-MNIST files, the stratified train/validation split,
and loading of the processed (INT8) dataset produced by prepare_dataset.py.
"""
import hashlib
from pathlib import Path

import numpy as np

from sw.config import (IMG_H, IMG_W, NUM_CLASSES, NUM_TEST_OFFICIAL, NUM_TRAIN_OFFICIAL,
                       PROCESSED_DIR, RAW_DIR)
from sw.data.idx import read_idx

RAW_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", NUM_TRAIN_OFFICIAL),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", NUM_TEST_OFFICIAL),
}

PROCESSED_FILE = PROCESSED_DIR / "fashion_mnist_int8.npz"
TEST_VECTORS_FILE = PROCESSED_DIR / "test_vectors.npz"


def load_raw(split: str, raw_dir: Path = RAW_DIR):
    """Return (images uint8 [N,28,28], labels uint8 [N]) for 'train' or 'test'."""
    img_name, lbl_name, n_expected = RAW_FILES[split]
    images = read_idx(raw_dir / img_name)
    labels = read_idx(raw_dir / lbl_name)
    assert images.dtype == np.uint8 and labels.dtype == np.uint8
    assert images.shape == (n_expected, IMG_H, IMG_W), images.shape
    assert labels.shape == (n_expected,), labels.shape
    assert labels.min() >= 0 and labels.max() < NUM_CLASSES
    return images, labels


def stratified_split(labels: np.ndarray, val_per_class: int, seed: int):
    """
    Pick `val_per_class` random indices of every class for validation.
    Returns (train_idx, val_idx), both sorted ascending, int64, disjoint,
    and together covering every index exactly once.
    """
    rng = np.random.default_rng(seed)
    val_parts = []
    for c in range(NUM_CLASSES):
        idx_c = np.flatnonzero(labels == c)
        if idx_c.size < val_per_class:
            raise ValueError(f"class {c} has only {idx_c.size} samples")
        val_parts.append(rng.choice(idx_c, size=val_per_class, replace=False))
    val_idx = np.sort(np.concatenate(val_parts)).astype(np.int64)
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[val_idx] = False
    train_idx = np.flatnonzero(mask).astype(np.int64)
    return train_idx, val_idx


def sha256_of_arrays(*arrays: np.ndarray) -> str:
    """Fingerprint of array contents (dtype + shape + bytes) for reproducibility logs."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def load_processed(path: Path = PROCESSED_FILE) -> dict:
    """
    Load the processed dataset. Keys:
        x_train, x_val, x_test : int8  [N, 28, 28], values in [0, 127]
        y_train, y_val, y_test : uint8 [N]
        train_idx, val_idx     : int64 indices into the official 60k train set
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m sw.data.prepare_dataset` first")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
