"""
Dataset inspection: statistics, sample grids and a markdown report.

Outputs (reports/dataset/):
    dataset_report.md        human-readable summary
    dataset_stats.json       machine-readable statistics
    samples_grid.png         10 samples per class (raw uint8 pixels)
    samples_int8_grid.png    same samples after the INT8 conversion
    class_distribution.png   per-split class histogram
    class_mean_images.png    mean image of every class
    pixel_histogram.png      raw pixel value histogram

Usage (from the repository root, after prepare_dataset):
    python -m sw.data.inspect_dataset
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sw.config import CLASS_NAMES, DATASET_REPORT_DIR, NUM_CLASSES  # noqa: E402
from sw.data.dataset import TEST_VECTORS_FILE, load_processed, load_raw  # noqa: E402
from sw.data.preprocess import int8_to_float  # noqa: E402


def pixel_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std()), "frac_zero": float((x == 0).mean())}


def ascii_image(img: np.ndarray, levels: str = " .:-=+*#%@") -> str:
    """Render an [H,W] image with values in [0,127] or [0,255] as ASCII art."""
    vmax = 255 if img.max() > 127 else 127
    idx = (img.astype(np.int32) * (len(levels) - 1) + vmax // 2) // vmax
    return "\n".join("".join(levels[v] for v in row) for row in idx)


def main() -> int:
    DATASET_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    d = load_processed()
    x_test_u8, y_test_raw = load_raw("test")
    x_train_u8, _ = load_raw("train")
    assert np.array_equal(y_test_raw, d["y_test"])

    # ------------------------------------------------------------------ stats
    stats = {"splits": {}, "input_conversion": {}}
    for s in ("train", "val", "test"):
        x, y = d[f"x_{s}"], d[f"y_{s}"]
        stats["splits"][s] = {
            "shape": list(x.shape), "dtype": str(x.dtype),
            "label_dtype": str(y.dtype),
            "per_class": np.bincount(y, minlength=NUM_CLASSES).tolist(),
            "int8_stats": pixel_stats(x),
        }
    stats["raw_train_pixel_stats"] = pixel_stats(x_train_u8)
    stats["raw_test_pixel_stats"] = pixel_stats(x_test_u8)

    # Effect of the >>1 conversion relative to the textbook p/255 normalization.
    # NOTE: the model is TRAINED on q_in/128, so this is not a train/hardware
    # mismatch - it only documents the information discarded by dropping the LSB.
    p = np.arange(256, dtype=np.uint8)
    x_conv = int8_to_float((p >> 1).astype(np.int8)).astype(np.float64)
    err = np.abs(x_conv - p / 255.0)
    stats["input_conversion"] = {
        "distinct_levels_raw": 256, "distinct_levels_int8": int(np.unique(p >> 1).size),
        "max_abs_diff_vs_p_div_255": float(err.max()),
        "mean_abs_diff_vs_p_div_255_uniform": float(err.mean()),
    }
    with open(DATASET_REPORT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---------------------------------------------------------------- figures
    per = 10
    sel = np.concatenate([np.flatnonzero(y_test_raw == c)[:per] for c in range(NUM_CLASSES)])
    for name, imgs, vmax, title in (
        ("samples_grid.png", x_test_u8[sel], 255, "raw uint8 pixels [0,255]"),
        ("samples_int8_grid.png", d["x_test"][sel], 127, "INT8 hardware input q_in = p>>1 [0,127]"),
    ):
        fig, axes = plt.subplots(NUM_CLASSES, per, figsize=(per * 1.0, NUM_CLASSES * 1.1))
        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i % per == 0:
                ax.set_ylabel(CLASS_NAMES[i // per], rotation=0, ha="right", va="center", fontsize=8)
        fig.suptitle(f"Fashion-MNIST test set, first {per} per class - {title}")
        fig.tight_layout()
        fig.savefig(DATASET_REPORT_DIR / name, dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.28
    for k, s in enumerate(("train", "val", "test")):
        ax.bar(np.arange(NUM_CLASSES) + (k - 1) * w, stats["splits"][s]["per_class"], w, label=s)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("images"); ax.set_title("Class distribution per split"); ax.legend()
    fig.tight_layout(); fig.savefig(DATASET_REPORT_DIR / "class_distribution.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(15, 2))
    for c, ax in enumerate(axes):
        ax.imshow(d["x_train"][d["y_train"] == c].mean(axis=0), cmap="gray", vmin=0, vmax=127)
        ax.set_title(CLASS_NAMES[c], fontsize=8); ax.axis("off")
    fig.suptitle("Mean INT8 image per class (training split)")
    fig.tight_layout(); fig.savefig(DATASET_REPORT_DIR / "class_mean_images.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(x_train_u8.ravel(), bins=256, range=(0, 256), log=True)
    ax.set_xlabel("raw pixel value"); ax.set_ylabel("count (log)")
    ax.set_title("Raw pixel histogram (official 60k training set)")
    fig.tight_layout(); fig.savefig(DATASET_REPORT_DIR / "pixel_histogram.png", dpi=110); plt.close(fig)

    # ----------------------------------------------------------------- report
    with np.load(TEST_VECTORS_FILE) as tv:
        tv_idx, tv_lbl, tv_img = tv["test_idx"], tv["labels"], tv["images_int8"]

    lines = ["# Dataset Inspection Report (Milestone 1)", "",
             "Generated by `python -m sw.data.inspect_dataset`. Do not edit by hand.", "",
             "## Splits", "",
             "| Split | Shape | dtype | Images per class |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        st = stats["splits"][s]
        lines.append(f"| {s} | {tuple(st['shape'])} | {st['dtype']} | {st['per_class']} |")
    lines += ["", "## Pixel statistics", "",
              "| Data | min | max | mean | std | fraction of zeros |", "|---|---|---|---|---|---|"]
    for lab, st in (("raw train (uint8)", stats["raw_train_pixel_stats"]),
                    ("raw test (uint8)", stats["raw_test_pixel_stats"]),
                    ("train INT8 (q_in)", stats["splits"]["train"]["int8_stats"]),
                    ("test INT8 (q_in)", stats["splits"]["test"]["int8_stats"])):
        lines.append(f"| {lab} | {st['min']:.0f} | {st['max']:.0f} | {st['mean']:.3f} | "
                     f"{st['std']:.3f} | {st['frac_zero']:.4f} |")
    ic = stats["input_conversion"]
    lines += ["", "## Input conversion `q_in = p >> 1`, `x = q_in / 128`", "",
              f"- Distinct grey levels: {ic['distinct_levels_raw']} -> {ic['distinct_levels_int8']}",
              f"- Max |x - p/255| over all 256 pixel values: {ic['max_abs_diff_vs_p_div_255']:.6f}",
              f"- Mean |x - p/255| over all 256 pixel values: {ic['mean_abs_diff_vs_p_div_255_uniform']:.6f}",
              "- The FP32 model is trained directly on `x = q_in / 128`, so this difference is",
              "  *not* a mismatch between training and hardware.", "",
              "## Class names", "", "| Label | Class |", "|---|---|"]
    lines += [f"| {i} | {n} |" for i, n in enumerate(CLASS_NAMES)]
    lines += ["", "## Hardware test vectors", "",
              f"{tv_idx.size} images (first {tv_idx.size // NUM_CLASSES} of each class in the official "
              "test set). Official test indices of the first vector of each class:", "",
              "| Class | Label | Test index |", "|---|---|---|"]
    for c in range(NUM_CLASSES):
        k = c * (tv_idx.size // NUM_CLASSES)
        lines.append(f"| {CLASS_NAMES[c]} | {int(tv_lbl[k])} | {int(tv_idx[k])} |")
    lines += ["", f"### Test vector 0 (test index {int(tv_idx[0])}, label {int(tv_lbl[0])} = "
              f"{CLASS_NAMES[int(tv_lbl[0])]}) as ASCII art", "", "```",
              ascii_image(tv_img[0]), "```", "",
              "## Figures", "",
              "![samples](samples_grid.png)", "", "![samples int8](samples_int8_grid.png)", "",
              "![class distribution](class_distribution.png)", "",
              "![class means](class_mean_images.png)", "", "![pixel histogram](pixel_histogram.png)", ""]
    (DATASET_REPORT_DIR / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------------------------------------------------------------- console
    for s in ("train", "val", "test"):
        st = stats["splits"][s]
        print(f"{s:5s}: {tuple(st['shape'])} {st['dtype']}  per-class {st['per_class']}")
    print(f"input conversion max |x - p/255| = {ic['max_abs_diff_vs_p_div_255']:.6f}")
    print(f"report written to {DATASET_REPORT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
