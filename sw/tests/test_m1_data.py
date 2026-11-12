"""
Milestone 1 tests. Runs with or without pytest:

    python -m sw.tests.test_m1_data
    python -m pytest sw/tests/test_m1_data.py      (if pytest is installed)

Tests that need the downloaded / processed dataset are skipped (and reported
as SKIPPED, never silently passed) if those files are missing.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from sw.config import (INPUT_SCALE, NUM_CLASSES, PROCESSED_DIR, RAW_DIR, SEED, SPLITS_DIR,
                       TEST_VECTORS_PER_CLASS, VAL_PER_CLASS)
from sw.data.dataset import (PROCESSED_FILE, TEST_VECTORS_FILE, load_processed, load_raw,
                             sha256_of_arrays, stratified_split)
from sw.data.download import FILES, md5sum
from sw.data.idx import read_idx, write_idx
from sw.data.preprocess import float_to_int8, int8_to_float, pixels_to_int8
from sw.tests._runner import need as _need
from sw.tests._runner import run_all


# ----------------------------------------------------------------- unit tests
def test_idx_roundtrip_uint8_gz():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, size=(5, 28, 28), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x-idx3-ubyte.gz"
        write_idx(p, a)
        b = read_idx(p)
    assert b.dtype == np.uint8 and b.shape == (5, 28, 28)
    assert np.array_equal(a, b)


def test_idx_rejects_truncated_payload():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad"
        write_idx(p, np.zeros((3, 4), np.uint8))
        p.write_bytes(p.read_bytes()[:-1])
        try:
            read_idx(p)
        except ValueError:
            return
    raise AssertionError("truncated IDX file was accepted")


def test_pixels_to_int8_exhaustive():
    p = np.arange(256, dtype=np.uint8)
    q = pixels_to_int8(p)
    assert q.dtype == np.int8
    assert q.min() == 0 and q.max() == 127
    assert np.array_equal(q.astype(np.int32), np.arange(256) // 2)  # floor(p/2)


def test_int8_float_roundtrip_exact():
    q = np.arange(0, 128, dtype=np.int8)
    x = int8_to_float(q)
    assert x.dtype == np.float32
    assert np.array_equal(x.astype(np.float64), np.arange(128) / 128.0)  # exact, no rounding
    assert INPUT_SCALE == 1 / 128
    assert np.array_equal(float_to_int8(x), q)


def test_pixels_to_int8_rejects_wrong_dtype():
    try:
        pixels_to_int8(np.array([1, 2, 3], dtype=np.int32))
    except TypeError:
        return
    raise AssertionError("non-uint8 input was accepted")


def test_stratified_split_synthetic():
    labels = np.repeat(np.arange(NUM_CLASSES, dtype=np.uint8), 50)
    tr, va = stratified_split(labels, 7, seed=1)
    assert np.intersect1d(tr, va).size == 0
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(labels.size))
    assert np.array_equal(np.bincount(labels[va], minlength=NUM_CLASSES), np.full(NUM_CLASSES, 7))
    tr2, va2 = stratified_split(labels, 7, seed=1)
    assert np.array_equal(va, va2), "split is not deterministic for a fixed seed"


# ---------------------------------------------------------- data-level tests
def test_raw_files_md5():
    for name, md5 in FILES.items():
        _need(RAW_DIR / name)
        assert md5sum(RAW_DIR / name) == md5, name


def test_raw_dataset_shapes():
    _need(RAW_DIR / "t10k-images-idx3-ubyte.gz")
    x, y = load_raw("train")
    assert x.shape == (60000, 28, 28) and y.shape == (60000,)
    assert np.array_equal(np.bincount(y), np.full(10, 6000))
    x, y = load_raw("test")
    assert x.shape == (10000, 28, 28) and y.shape == (10000,)
    assert np.array_equal(np.bincount(y), np.full(10, 1000))


def test_processed_dataset_consistent_with_raw():
    _need(PROCESSED_FILE)
    d = load_processed()
    x_tr_u8, y_tr = load_raw("train")
    x_te_u8, y_te = load_raw("test")
    assert d["x_train"].shape == (60000 - 10 * VAL_PER_CLASS, 28, 28)
    assert d["x_val"].shape == (10 * VAL_PER_CLASS, 28, 28)
    assert d["x_test"].shape == (10000, 28, 28)
    for k in ("x_train", "x_val", "x_test"):
        assert d[k].dtype == np.int8 and d[k].min() >= 0 and d[k].max() <= 127
    # recompute every processed image from the raw data
    assert np.array_equal(d["x_train"], pixels_to_int8(x_tr_u8[d["train_idx"]]))
    assert np.array_equal(d["x_val"], pixels_to_int8(x_tr_u8[d["val_idx"]]))
    assert np.array_equal(d["x_test"], pixels_to_int8(x_te_u8))
    assert np.array_equal(d["y_train"], y_tr[d["train_idx"]])
    assert np.array_equal(d["y_val"], y_tr[d["val_idx"]])
    assert np.array_equal(d["y_test"], y_te)
    assert np.intersect1d(d["train_idx"], d["val_idx"]).size == 0


def test_split_file_matches_manifest():
    split_file = SPLITS_DIR / f"split_seed{SEED}.npz"
    _need(split_file)
    _need(PROCESSED_DIR / "manifest.json")
    with np.load(split_file) as z:
        fp = sha256_of_arrays(z["train_idx"], z["val_idx"])
    manifest = json.loads((PROCESSED_DIR / "manifest.json").read_text())
    assert manifest["sha256"]["split_indices"] == fp


def test_test_vectors():
    _need(TEST_VECTORS_FILE)
    x_te_u8, y_te = load_raw("test")
    with np.load(TEST_VECTORS_FILE) as tv:
        idx, lbl, u8, i8 = tv["test_idx"], tv["labels"], tv["images_u8"], tv["images_int8"]
    assert idx.size == NUM_CLASSES * TEST_VECTORS_PER_CLASS
    assert np.array_equal(lbl, y_te[idx])
    assert np.array_equal(u8, x_te_u8[idx])
    assert np.array_equal(i8, pixels_to_int8(u8))
    assert np.array_equal(np.bincount(lbl, minlength=NUM_CLASSES),
                          np.full(NUM_CLASSES, TEST_VECTORS_PER_CLASS))


if __name__ == "__main__":
    sys.exit(run_all(globals()))
