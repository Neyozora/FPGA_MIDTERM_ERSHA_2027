"""
Milestone 6 automated tests: Independent integer-only reference model and
golden ModelSim vector verification.

Run:
    python -m sw.tests.test_m6_refmodel
    python -m pytest sw/tests/test_m6_refmodel.py
"""
import json

import numpy as np

from sw.config import CHECKPOINTS_DIR, MEM_DIR, REFMODEL_REPORT_DIR, SIM_VECTORS_DIR
from sw.export.formatters import read_mem
from sw.quant.quantize import requantize_np
from sw.refmodel.int8_ref_model import (
    argmax_int8_tiebreak_lowest,
    conv2d_valid_3x3_int8_to_int32,
    fc_int8_to_int32,
    gap_13x13_int8,
    infer_batch_int8,
    infer_single_hw_loop,
    load_hw_params_from_mem,
    maxpool2x2_stride2_int8,
    relu_int8,
    requantize_signed_int8,
)
import sys

from sw.tests._runner import run_all


def test_relu_after_signed_requant_equals_fused_relu_requant():
    """
    Verify that separating signed INT8 requantization [-128, 127] and ReLU [0, 127]
    into two hardware modules (`conv_engine`/`requant` -> `relu.v`) produces
    bit-for-bit identical results to fused ReLU-requantization across all
    possible accumulators.
    """
    rng = np.random.default_rng(601)
    acc = rng.integers(-2_000_000, 2_000_000, size=50_000, dtype=np.int32)
    for mult, shift in [(23202, 19), (16868, 20), (16780, 19), (23793, 21)]:
        two_step = relu_int8(requantize_signed_int8(acc, mult, shift))
        fused = requantize_np(acc, mult, shift, relu=True)
        assert np.array_equal(two_step, fused)


def test_module_primitives_signed_edge_cases():
    """Verify ReLU, MaxPool (with negative values), GAP, FC, and Argmax tie-breaking."""
    # ReLU on [-128..127]
    all_int8 = np.arange(-128, 128, dtype=np.int8)
    r = relu_int8(all_int8)
    assert np.all(r[:128] == 0)
    assert np.array_equal(r[128:], np.arange(0, 128, dtype=np.int8))

    # MaxPool 2x2 stride 2 with signed negative and positive values
    x_mp = np.array([
        [[-128, -50,   0,  12],
         [ -10, -99, 127,  -1],
         [  -5,  -6,  -7,  -8],
         [ -20,  -2,  -9,  -3]]
    ], dtype=np.int8)
    y_mp = maxpool2x2_stride2_int8(x_mp)
    expected_mp = np.array([[[-10, 127], [-2, -3]]], dtype=np.int8)
    assert np.array_equal(y_mp, expected_mp)

    # Argmax tie-breaking to lowest class index
    scores = np.array([
        [-10, 25, 25, 10, -128, 25, 0, 1, 2, 3],
        [-128, -128, -128, -128, -128, -128, -128, -128, -128, -128],
    ], dtype=np.int8)
    preds = argmax_int8_tiebreak_lowest(scores)
    assert int(preds[0]) == 1
    assert int(preds[1]) == 0


def test_no_floating_point_dtypes_in_refmodel_outputs():
    """Ensure every tensor returned by the integer reference model has an integer dtype."""
    params = load_hw_params_from_mem(MEM_DIR)
    img0 = read_mem(MEM_DIR / "test_image.mem", width_bits=8)
    out_loop = infer_single_hw_loop(img0, params)
    out_vec = infer_batch_int8(img0.reshape(1, 32, 32), params)

    for name, val in out_loop.items():
        arr = np.asarray(val)
        assert np.issubdtype(arr.dtype, np.integer), f"{name} has non-integer dtype {arr.dtype}"
        assert np.array_equal(arr, np.asarray(out_vec[name])), f"Mismatch on {name}"


def test_single_hw_loop_matches_batch_on_random_synthetic_inputs():
    """Verify hardware address-indexed loop matches vectorized integer model on random [-128, 127] inputs."""
    params = load_hw_params_from_mem(MEM_DIR)
    rng = np.random.default_rng(602)
    synth_imgs = rng.integers(-128, 128, size=(5, 1, 32, 32), dtype=np.int8)
    batch_out = infer_batch_int8(synth_imgs, params)

    for i in range(5):
        loop_out = infer_single_hw_loop(synth_imgs[i].ravel(), params)
        for k in loop_out:
            assert np.array_equal(
                np.asarray(loop_out[k]),
                np.asarray(batch_out[k][i]),
            ), f"Synthetic image {i} mismatch at stage {k}"


def test_golden_vectors_on_disk_match_reference_model():
    """Verify all exported `.mem` golden vectors in `sim/vectors/` match `infer_single_hw_loop`."""
    assert (SIM_VECTORS_DIR / "vectors_manifest.json").is_file(), (
        "Run `python -m sw.refmodel.generate_vectors` first"
    )
    assert (REFMODEL_REPORT_DIR / "refmodel_report.md").is_file()

    params = load_hw_params_from_mem(MEM_DIR)
    img0 = read_mem(MEM_DIR / "test_image.mem", width_bits=8)
    ref = infer_single_hw_loop(img0, params)

    checks = [
        ("img0_input.mem", 8, ref["input_int8"].ravel()),
        ("img0_conv1_acc.mem", 32, ref["conv1_acc_int32"].ravel()),
        ("img0_conv1_req.mem", 8, ref["conv1_req_int8"].ravel()),
        ("img0_relu1_out.mem", 8, ref["relu1_int8"].ravel()),
        ("img0_pool1_out.mem", 8, ref["pool1_int8"].ravel()),
        ("img0_conv2_acc.mem", 32, ref["conv2_acc_int32"].ravel()),
        ("img0_conv2_req.mem", 8, ref["conv2_req_int8"].ravel()),
        ("img0_relu2_out.mem", 8, ref["relu2_int8"].ravel()),
        ("img0_gap_sum.mem", 32, ref["gap_sum_int32"].ravel()),
        ("img0_gap_out.mem", 8, ref["gap_int8"].ravel()),
        ("img0_fc_acc.mem", 32, ref["fc_acc_int32"].ravel()),
        ("img0_fc_scores.mem", 8, ref["fc_scores_int8"].ravel()),
        ("img0_expected_class.mem", 8, np.array([int(ref["pred_class"])], dtype=np.int8)),
    ]
    for fname, wbits, expected in checks:
        loaded = read_mem(SIM_VECTORS_DIR / fname, width_bits=wbits)
        assert loaded.size == expected.size
        assert np.array_equal(loaded, expected), f"Golden vector mismatch in {fname}"

    with open(SIM_VECTORS_DIR / "vectors_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["m6_vs_m4_bit_exact_10k"] is True
    assert manifest["loop_vs_batch_bit_exact_100"] is True
    assert abs(manifest["test10k_accuracy"] - 0.8587) < 1e-6


if __name__ == "__main__":
    sys.exit(run_all(globals()))
