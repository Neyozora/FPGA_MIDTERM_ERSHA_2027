"""
Milestone 5: Export utility that generates all required `.mem` and `.mif`
hardware memory initialization files from `checkpoints/int8_quantized.npz`
and `data/processed/test_vectors.npz`, then performs an exhaustive read-back
verification against the original quantized tensors.

Required output files in `mem/`:
    - conv1_weights.mem / .mif       (72 x 8-bit signed INT8)
    - conv1_bias.mem    / .mif       (8 x 32-bit signed INT32)
    - conv2_weights.mem / .mif       (1152 x 8-bit signed INT8)
    - conv2_bias.mem    / .mif       (16 x 32-bit signed INT32)
    - fc_weights.mem    / .mif       (160 x 8-bit signed INT8)
    - fc_bias.mem       / .mif       (10 x 32-bit signed INT32)
    - quantization_params.mem / .mif (8 x 32-bit signed INT32)
    - test_image.mem    / .mif       (1024 x 8-bit signed INT8, 32x32 image)
    - test_images_100.mem            (102,400 x 8-bit signed INT8, 100 test vectors)
    - test_labels_100.mem            (100 x 8-bit expected class labels 0..9)

Usage:
    python -m sw.export.export_mem
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from sw.config import CHECKPOINTS_DIR, CLASS_NAMES, EXPORT_REPORT_DIR, MEM_DIR
from sw.data.dataset import TEST_VECTORS_FILE
from sw.export.formatters import (pack_conv_weights_oihw, pack_fc_weights_oi, pack_image_hw,
                                  read_mem, read_mif, unpack_conv_weights_oihw,
                                  unpack_fc_weights_oi, write_mem, write_mif)

QUANT_NPZ_PATH = CHECKPOINTS_DIR / "int8_quantized.npz"

QUANT_PARAM_NAMES = [
    "conv1_mult",
    "conv1_shift",
    "conv2_mult",
    "conv2_shift",
    "gap_mult",
    "gap_shift",
    "fc_mult",
    "fc_shift",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_all(quant_npz_path: Path = QUANT_NPZ_PATH,
               test_vectors_path: Path = TEST_VECTORS_FILE,
               mem_dir: Path = MEM_DIR,
               report_dir: Path = EXPORT_REPORT_DIR,
               test_vector_index: int = 0) -> dict:
    if not quant_npz_path.exists():
        raise FileNotFoundError(f"{quant_npz_path} missing - run Milestone 4 first")
    if not test_vectors_path.exists():
        raise FileNotFoundError(f"{test_vectors_path} missing - run Milestone 1 first")

    mem_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with np.load(quant_npz_path) as qz:
        qparams = {k: qz[k] for k in qz.files}
    with np.load(test_vectors_path) as tv:
        tv_images_i8 = tv["images_int8"]    # [100, 32, 32] int8
        tv_labels = tv["labels"].astype(np.int8)
        tv_test_idx = tv["test_idx"].astype(np.int64)

    # ---- 1. Explicitly pack PyTorch tensors into hardware memory order ------
    c1_w_flat = pack_conv_weights_oihw(qparams["conv1_weight"])
    c1_b_flat = qparams["conv1_bias"].astype(np.int32).reshape(-1)
    c2_w_flat = pack_conv_weights_oihw(qparams["conv2_weight"])
    c2_b_flat = qparams["conv2_bias"].astype(np.int32).reshape(-1)
    fc_w_flat = pack_fc_weights_oi(qparams["fc_weight"])
    fc_b_flat = qparams["fc_bias"].astype(np.int32).reshape(-1)
    qparam_flat = np.array([int(qparams[k]) for k in QUANT_PARAM_NAMES], dtype=np.int32)

    single_img = tv_images_i8[test_vector_index]
    single_lbl = int(tv_labels[test_vector_index])
    single_official_idx = int(tv_test_idx[test_vector_index])
    img_flat = pack_image_hw(single_img)

    specs = [
        ("conv1_weights", c1_w_flat, 8, list(qparams["conv1_weight"].shape),
         "Conv1 3x3 weights, order (oc, ic, ky, kx), addr = oc*9 + ky*3 + kx"),
        ("conv1_bias", c1_b_flat, 32, list(qparams["conv1_bias"].shape),
         "Conv1 INT32 biases, addr = oc"),
        ("conv2_weights", c2_w_flat, 8, list(qparams["conv2_weight"].shape),
         "Conv2 3x3 weights, order (oc, ic, ky, kx), addr = oc*72 + ic*9 + ky*3 + kx"),
        ("conv2_bias", c2_b_flat, 32, list(qparams["conv2_bias"].shape),
         "Conv2 INT32 biases, addr = oc"),
        ("fc_weights", fc_w_flat, 8, list(qparams["fc_weight"].shape),
         "FC 16->10 weights, order (out_class, in_ch), addr = out_class*16 + in_ch"),
        ("fc_bias", fc_b_flat, 32, list(qparams["fc_bias"].shape),
         "FC INT32 biases, addr = out_class"),
        ("quantization_params", qparam_flat, 32, [8],
         "Requantization (M, shift) for conv1, conv2, gap, fc"),
        ("test_image", img_flat, 8, list(single_img.shape),
         f"Test vector #{test_vector_index} (official test idx {single_official_idx}, label {single_lbl}={CLASS_NAMES[single_lbl]}), row-major 32x32"),
    ]

    exported_files = {}
    for base, arr, width, orig_shape, desc in specs:
        mem_path = mem_dir / f"{base}.mem"
        mif_path = mem_dir / f"{base}.mif"
        write_mem(mem_path, arr, width)
        write_mif(mif_path, arr, width, comment=desc)

        # ---- Read back and verify every single value ------------------------
        rb_mem = read_mem(mem_path, width)
        rb_mif, mif_meta = read_mif(mif_path)
        assert np.array_equal(rb_mem, arr), f"{mem_path.name} read-back mismatch!"
        assert np.array_equal(rb_mif, arr), f"{mif_path.name} read-back mismatch!"
        assert mif_meta["depth"] == arr.size and mif_meta["width"] == width

        exported_files[base] = {
            "mem_file": mem_path.name,
            "mif_file": mif_path.name,
            "original_shape": orig_shape,
            "depth": int(arr.size),
            "width_bits": int(width),
            "bytes": int(arr.size * (width // 8)),
            "min_val": int(arr.min()),
            "max_val": int(arr.max()),
            "first_4_hex": [ln.strip() for ln in mem_path.read_text().splitlines()[:4]],
            "sha256_mem": _sha256_file(mem_path),
            "sha256_mif": _sha256_file(mif_path),
            "description": desc,
            "readback_verified": True,
        }
        print(f"[ok] {base:20s} depth={arr.size:5d} width={width:2d}b "
              f"range=[{int(arr.min()):7d}, {int(arr.max()):6d}] (.mem & .mif verified)")

    # ---- Also verify 4D/2D tensor unpacking matches original PyTorch tensors -
    assert np.array_equal(
        unpack_conv_weights_oihw(read_mem(mem_dir / "conv1_weights.mem", 8), (8, 1, 3, 3)),
        qparams["conv1_weight"],
    )
    assert np.array_equal(
        unpack_conv_weights_oihw(read_mem(mem_dir / "conv2_weights.mem", 8), (16, 8, 3, 3)),
        qparams["conv2_weight"],
    )
    assert np.array_equal(
        unpack_fc_weights_oi(read_mem(mem_dir / "fc_weights.mem", 8), (10, 16)),
        qparams["fc_weight"],
    )
    assert np.array_equal(
        read_mem(mem_dir / "test_image.mem", 8).reshape(32, 32),
        single_img,
    )

    # ---- Export full 100-image test suite for batch RTL verification ---------
    all_imgs_flat = pack_image_hw(tv_images_i8)
    write_mem(mem_dir / "test_images_100.mem", all_imgs_flat, 8)
    write_mem(mem_dir / "test_labels_100.mem", tv_labels, 8)
    assert np.array_equal(read_mem(mem_dir / "test_images_100.mem", 8), all_imgs_flat)
    assert np.array_equal(read_mem(mem_dir / "test_labels_100.mem", 8), tv_labels)

    manifest = {
        "source_checkpoint": str(quant_npz_path.name),
        "test_image_info": {
            "vector_index": test_vector_index,
            "official_test_index": single_official_idx,
            "expected_label": single_lbl,
            "expected_class_name": CLASS_NAMES[single_lbl],
        },
        "quantization_params_map": {
            idx: {"name": name, "value": int(qparam_flat[idx]), "hex": f"{int(qparam_flat[idx]) & 0xFFFFFFFF:08X}"}
            for idx, name in enumerate(QUANT_PARAM_NAMES)
        },
        "files": exported_files,
        "batch_testbench_files": {
            "test_images_100.mem": {"depth": int(all_imgs_flat.size), "width_bits": 8},
            "test_labels_100.mem": {"depth": int(tv_labels.size), "width_bits": 8},
        },
    }
    (mem_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    (report_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- Write Markdown Report ----------------------------------------------
    L = [
        "# Weight, Bias, and Test Image Export Report (Milestone 5)", "",
        "Generated by `python -m sw.export.export_mem`. Do not edit by hand.",
        "Full memory layout & address specification: [`docs/memory_map.md`](../../docs/memory_map.md).", "",
        "## 1. Exported `.mem` and `.mif` Summary", "",
        "| Base Name | `.mem` / `.mif` | Original Tensor Shape | Hardware Depth | Width (bits) | Size (Bytes) | Value Range | Read-Back Verified |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for base, info in exported_files.items():
        L.append(
            f"| `{base}` | `{info['mem_file']}` / `{info['mif_file']}` | "
            f"`{tuple(info['original_shape'])}` | {info['depth']:,} | {info['width_bits']} | "
            f"{info['bytes']:,} B | `[{info['min_val']}, {info['max_val']}]` | "
            f"{'PASS' if info['readback_verified'] else 'FAIL'} |"
        )

    L += [
        "", "## 2. `quantization_params.mem` / `.mif` Contents", "",
        "| Address | Parameter | Signed Decimal | 32-Bit Two's-Complement Hex |",
        "|---|---|---|---|",
    ]
    for idx, name in enumerate(QUANT_PARAM_NAMES):
        val = int(qparam_flat[idx])
        L.append(f"| `{idx}` | `{name}` | `{val}` | `{val & 0xFFFFFFFF:08X}` |")

    L += [
        "", "## 3. Default `test_image.mem` / `.mif` Metadata", "",
        f"- **Hardware test vector index:** `{test_vector_index}`",
        f"- **Official Fashion-MNIST `t10k` index:** `{single_official_idx}`",
        f"- **Expected ground-truth label:** `{single_lbl}` (`{CLASS_NAMES[single_lbl]}`)",
        f"- **Dimensions:** `32 × 32` (`1,024` bytes, 2-pixel zero-padded border, INT8 in `[0, 127]`)",
        f"- **Batch simulation files:** `mem/test_images_100.mem` (102,400 bytes) and `mem/test_labels_100.mem` (100 bytes)",
        "",
    ]
    (report_dir / "export_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {report_dir / 'export_report.md'} and verified all exported files.")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-vector-index", type=int, default=0)
    args = ap.parse_args(argv)
    export_all(test_vector_index=args.test_vector_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
