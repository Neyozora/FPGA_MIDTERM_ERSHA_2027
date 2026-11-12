"""
Milestone 5 tests: two's-complement hex conversion, explicit PyTorch <-> hardware
layout packing/unpacking, .mem and .mif writers/parsers, and full read-back
verification of every exported file in `mem/`.

    python -m sw.tests.test_m5_export
    python -m pytest sw/tests/test_m5_export.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from sw.config import CHECKPOINTS_DIR, MEM_DIR
from sw.data.dataset import TEST_VECTORS_FILE
from sw.export.export_mem import QUANT_PARAM_NAMES
from sw.export.formatters import (int_to_twos_hex, pack_conv_weights_oihw, pack_fc_weights_oi,
                                  pack_image_hw, read_mem, read_mif, twos_hex_to_int,
                                  unpack_conv_weights_oihw, unpack_fc_weights_oi, write_mem,
                                  write_mif)
from sw.tests._runner import need, run_all


def test_int8_twos_complement_exhaustive():
    for v in range(-128, 128):
        hx = int_to_twos_hex(v, 8)
        assert len(hx) == 2
        assert twos_hex_to_int(hx, 8) == v
    assert int_to_twos_hex(0, 8) == "00"
    assert int_to_twos_hex(127, 8) == "7F"
    assert int_to_twos_hex(-1, 8) == "FF"
    assert int_to_twos_hex(-127, 8) == "81"
    assert int_to_twos_hex(-128, 8) == "80"


def test_int32_twos_complement_edge_cases():
    cases = [
        (0, "00000000"),
        (1, "00000001"),
        (-1, "FFFFFFFF"),
        (-128, "FFFFFF80"),
        (23202, "00005AA2"),
        ((1 << 31) - 1, "7FFFFFFF"),
        (-(1 << 31), "80000000"),
    ]
    for val, expected_hex in cases:
        hx = int_to_twos_hex(val, 32)
        assert hx == expected_hex, (val, hx, expected_hex)
        assert twos_hex_to_int(hx, 32) == val


def test_conv_and_fc_layout_pack_unpack_address_formulas():
    rng = np.random.default_rng(42)
    w_c2 = rng.integers(-127, 128, size=(16, 8, 3, 3), dtype=np.int8)
    flat_c2 = pack_conv_weights_oihw(w_c2)
    assert flat_c2.shape == (1152,)
    # Spot check explicit address formula: addr = oc*72 + ic*9 + ky*3 + kx
    for oc in (0, 5, 15):
        for ic in (0, 3, 7):
            for ky in range(3):
                for kx in range(3):
                    addr = oc * 72 + ic * 9 + ky * 3 + kx
                    assert flat_c2[addr] == w_c2[oc, ic, ky, kx]
    assert np.array_equal(unpack_conv_weights_oihw(flat_c2, (16, 8, 3, 3)), w_c2)

    w_fc = rng.integers(-127, 128, size=(10, 16), dtype=np.int8)
    flat_fc = pack_fc_weights_oi(w_fc)
    assert flat_fc.shape == (160,)
    for o in range(10):
        for i in range(16):
            assert flat_fc[o * 16 + i] == w_fc[o, i]
    assert np.array_equal(unpack_fc_weights_oi(flat_fc, (10, 16)), w_fc)


def test_mem_and_mif_roundtrip_tempfiles():
    rng = np.random.default_rng(7)
    arr8 = rng.integers(-128, 128, size=(72,), dtype=np.int8)
    arr32 = rng.integers(-100_000, 100_000, size=(16,), dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        write_mem(p / "a8.mem", arr8, 8)
        write_mif(p / "a8.mif", arr8, 8, comment="test 8b")
        write_mem(p / "a32.mem", arr32, 32)
        write_mif(p / "a32.mif", arr32, 32, comment="test 32b")

        assert np.array_equal(read_mem(p / "a8.mem", 8), arr8)
        rb8, meta8 = read_mif(p / "a8.mif")
        assert np.array_equal(rb8, arr8) and meta8["depth"] == 72 and meta8["width"] == 8

        assert np.array_equal(read_mem(p / "a32.mem", 32), arr32)
        rb32, meta32 = read_mif(p / "a32.mif")
        assert np.array_equal(rb32, arr32) and meta32["depth"] == 16 and meta32["width"] == 32


def test_mif_rejects_missing_declarations():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.mif"
        bad.write_text("DEPTH = 4;\nWIDTH = 8;\nCONTENT\nBEGIN\n0 : 00;\nEND;\n", encoding="ascii")
        try:
            read_mif(bad)
        except ValueError:
            return
    raise AssertionError("MIF missing ADDRESS_RADIX/DATA_RADIX was accepted")


def test_exported_files_match_quantized_tensors_bit_for_bit():
    need(CHECKPOINTS_DIR / "int8_quantized.npz")
    need(TEST_VECTORS_FILE)
    required_bases = [
        ("conv1_weights", 72, 8),
        ("conv1_bias", 8, 32),
        ("conv2_weights", 1152, 8),
        ("conv2_bias", 16, 32),
        ("fc_weights", 160, 8),
        ("fc_bias", 10, 32),
        ("quantization_params", 8, 32),
        ("test_image", 1024, 8),
    ]
    for base, _, _ in required_bases:
        need(MEM_DIR / f"{base}.mem")
        need(MEM_DIR / f"{base}.mif")

    with np.load(CHECKPOINTS_DIR / "int8_quantized.npz") as qz:
        qparams = {k: qz[k] for k in qz.files}
    with np.load(TEST_VECTORS_FILE) as tv:
        img0 = tv["images_int8"][0]

    expected_arrays = {
        "conv1_weights": pack_conv_weights_oihw(qparams["conv1_weight"]),
        "conv1_bias": qparams["conv1_bias"].astype(np.int32),
        "conv2_weights": pack_conv_weights_oihw(qparams["conv2_weight"]),
        "conv2_bias": qparams["conv2_bias"].astype(np.int32),
        "fc_weights": pack_fc_weights_oi(qparams["fc_weight"]),
        "fc_bias": qparams["fc_bias"].astype(np.int32),
        "quantization_params": np.array([int(qparams[k]) for k in QUANT_PARAM_NAMES], dtype=np.int32),
        "test_image": pack_image_hw(img0),
    }

    for base, exp_depth, exp_width in required_bases:
        exp_arr = expected_arrays[base]
        mem_arr = read_mem(MEM_DIR / f"{base}.mem", exp_width)
        mif_arr, meta = read_mif(MEM_DIR / f"{base}.mif")
        assert mem_arr.shape == (exp_depth,), (base, mem_arr.shape)
        assert meta["depth"] == exp_depth and meta["width"] == exp_width
        assert np.array_equal(mem_arr, exp_arr), f"{base}.mem mismatch"
        assert np.array_equal(mif_arr, exp_arr), f"{base}.mif mismatch"


if __name__ == "__main__":
    sys.exit(run_all(globals()))
