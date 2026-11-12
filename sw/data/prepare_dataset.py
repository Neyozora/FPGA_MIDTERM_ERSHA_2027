"""
Milestone 1 pipeline: download -> parse -> split -> convert to INT8 -> save.

Outputs
    data/splits/split_seed42.npz          train/val indices (small, tracked in git)
    data/processed/fashion_mnist_int8.npz  x_{train,val,test} int8, y_* uint8
    data/processed/test_vectors.npz        representative test images for hardware
    data/processed/manifest.json           shapes, counts, settings, SHA-256 fingerprints

Usage (from the repository root):
    python -m sw.data.prepare_dataset
"""
import argparse
import json
import sys

import numpy as np

from sw.config import (CLASS_NAMES, INPUT_FRAC_BITS, INPUT_SCALE, INPUT_SHIFT, INPUT_ZERO_POINT,
                       NUM_CLASSES, PROCESSED_DIR, RAW_DIR, SEED, SPLITS_DIR,
                       TEST_VECTORS_PER_CLASS, VAL_PER_CLASS)
from sw.data.dataset import (PROCESSED_FILE, TEST_VECTORS_FILE, load_raw, sha256_of_arrays,
                             stratified_split)
from sw.data.download import fetch_all
from sw.data.preprocess import pixels_to_int8


def select_test_vectors(labels: np.ndarray, per_class: int) -> np.ndarray:
    """First `per_class` test indices of each class, ordered by class then index."""
    out = []
    for c in range(NUM_CLASSES):
        idx_c = np.flatnonzero(labels == c)[:per_class]
        assert idx_c.size == per_class
        out.append(idx_c)
    return np.concatenate(out).astype(np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    args = ap.parse_args(argv)

    if not args.no_download:
        fetch_all(RAW_DIR)

    # ---- load raw --------------------------------------------------------
    x_train_full_u8, y_train_full = load_raw("train")
    x_test_u8, y_test = load_raw("test")
    print(f"raw train: {x_train_full_u8.shape} {x_train_full_u8.dtype}, labels {y_train_full.shape}")
    print(f"raw test : {x_test_u8.shape} {x_test_u8.dtype}, labels {y_test.shape}")

    # ---- split (reuse saved indices if present so the split never drifts) --
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_file = SPLITS_DIR / f"split_seed{SEED}.npz"
    if split_file.exists():
        with np.load(split_file) as z:
            train_idx, val_idx = z["train_idx"], z["val_idx"]
        print(f"reusing split from {split_file}")
    else:
        train_idx, val_idx = stratified_split(y_train_full, VAL_PER_CLASS, SEED)
        np.savez_compressed(split_file, train_idx=train_idx, val_idx=val_idx,
                            seed=np.int64(SEED), val_per_class=np.int64(VAL_PER_CLASS))
        print(f"created split {split_file}")

    # split sanity checks
    assert np.intersect1d(train_idx, val_idx).size == 0, "train/val overlap"
    assert train_idx.size + val_idx.size == y_train_full.size, "split does not cover all samples"
    assert np.array_equal(np.bincount(y_train_full[val_idx], minlength=NUM_CLASSES),
                          np.full(NUM_CLASSES, VAL_PER_CLASS)), "validation split not stratified"

    # ---- convert to INT8 -------------------------------------------------
    x_train = pixels_to_int8(x_train_full_u8[train_idx])
    x_val = pixels_to_int8(x_train_full_u8[val_idx])
    x_test = pixels_to_int8(x_test_u8)
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_FILE, x_train=x_train, y_train=y_train, x_val=x_val,
                        y_val=y_val, x_test=x_test, y_test=y_test,
                        train_idx=train_idx, val_idx=val_idx)

    # ---- representative test vectors for RTL simulation -------------------
    tv_idx = select_test_vectors(y_test, TEST_VECTORS_PER_CLASS)
    np.savez_compressed(TEST_VECTORS_FILE, test_idx=tv_idx, labels=y_test[tv_idx],
                        images_u8=x_test_u8[tv_idx], images_int8=x_test[tv_idx])

    # ---- manifest --------------------------------------------------------
    manifest = {
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "class_names": CLASS_NAMES,
        "input_conversion": {
            "formula": f"q_in = pixel >> {INPUT_SHIFT};  x = q_in * 2^-{INPUT_FRAC_BITS}",
            "input_scale": INPUT_SCALE,
            "zero_point": INPUT_ZERO_POINT,
            "q_range": [int(x_test.min()), int(x_test.max())],
            "dtype": "int8",
        },
        "split": {"seed": SEED, "val_per_class": VAL_PER_CLASS, "file": split_file.name},
        "counts": {
            "train": int(y_train.size), "val": int(y_val.size), "test": int(y_test.size),
            "train_per_class": np.bincount(y_train, minlength=NUM_CLASSES).tolist(),
            "val_per_class": np.bincount(y_val, minlength=NUM_CLASSES).tolist(),
            "test_per_class": np.bincount(y_test, minlength=NUM_CLASSES).tolist(),
        },
        "shapes": {k: list(v.shape) for k, v in
                   dict(x_train=x_train, x_val=x_val, x_test=x_test).items()},
        "test_vectors": {
            "per_class": TEST_VECTORS_PER_CLASS,
            "selection": "first N official-test-set images of each class, ordered by class",
            "test_idx": tv_idx.tolist(),
        },
        "sha256": {
            "split_indices": sha256_of_arrays(train_idx, val_idx),
            "x_train": sha256_of_arrays(x_train), "y_train": sha256_of_arrays(y_train),
            "x_val": sha256_of_arrays(x_val), "y_val": sha256_of_arrays(y_val),
            "x_test": sha256_of_arrays(x_test), "y_test": sha256_of_arrays(y_test),
        },
    }
    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}  (int8, range "
          f"[{x_train.min()}, {x_train.max()}])")
    print(f"wrote {PROCESSED_FILE}")
    print(f"wrote {TEST_VECTORS_FILE} ({tv_idx.size} images)")
    print(f"wrote {PROCESSED_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
