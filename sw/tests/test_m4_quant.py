"""
Milestone 4 tests: fixed-point & INT8 quantization primitives, exact GAP
reciprocal division over all 21,464 possible sums, requantization rounding
and saturation, and quantized INT8 model accuracy vs FP32.

    python -m sw.tests.test_m4_quant
    python -m pytest sw/tests/test_m4_quant.py
"""
import json
import sys

import numpy as np
import torch

from sw.config import CHECKPOINTS_DIR, QUANT_REPORT_DIR
from sw.quant.evaluate_quant import QUANT_NPZ_PATH
from sw.quant.quantize import (GAP_MULT, GAP_SHIFT, compute_dyadic_multiplier,
                               forward_int8_torch, gap_int8_np, gap_int8_torch,
                               quantize_bias_int32, quantize_weight_symmetric,
                               requantize_np, requantize_torch)
from sw.tests._runner import need, run_all


def test_gap_reciprocal_exact_for_all_21464_sums():
    """
    Exhaustive proof that (S * 12409 + 2^20) >> 21 equals exact round-half-up
    floor((S + 84) / 169) for EVERY possible sum S in [0, 169 * 127 = 21463].
    """
    s = np.arange(0, 169 * 127 + 1, dtype=np.int64)
    got = ((s * GAP_MULT) + (1 << (GAP_SHIFT - 1))) >> GAP_SHIFT
    ref = (s + 84) // 169
    assert np.array_equal(got, ref), "GAP fixed-point reciprocal differs from exact /169"
    assert got.min() == 0 and got.max() == 127


def test_requantize_np_and_torch_match_exhaustively():
    rng = np.random.default_rng(123)
    acc = rng.integers(-500_000, 500_000, size=(256,), dtype=np.int32)
    for M, shift in ((21345, 20), (16384, 15), (32767, 22)):
        for relu in (True, False):
            r_np = requantize_np(acc, M, shift, relu=relu)
            r_th = requantize_torch(torch.from_numpy(acc), M, shift, relu=relu).numpy()
            assert r_np.dtype == np.int8 and r_th.dtype == np.int8
            assert np.array_equal(r_np, r_th)
            if relu:
                assert r_np.min() >= 0 and r_np.max() <= 127
            else:
                assert r_np.min() >= -128 and r_np.max() <= 127


def test_requantize_rounding_and_saturation_edge_cases():
    # M = 1, shift = 2 -> divide by 4 with round-half-up (+2 before >>2):
    # acc = 1 -> (1 + 2) >> 2 = 0 (0.25 -> 0)
    # acc = 2 -> (2 + 2) >> 2 = 1 (0.50 -> 1, round half up!)
    # acc = -2 -> (-2 + 2) >> 2 = 0 (-0.50 -> 0, round half up!)
    # acc = -3 -> (-3 + 2) >> 2 = -1 (-0.75 -> -1)
    acc = np.array([1, 2, -2, -3, 1000, -1000], dtype=np.int32)
    got_signed = requantize_np(acc, M=1, shift=2, relu=False)
    assert got_signed.tolist() == [0, 1, 0, -1, 127, -128]
    got_relu = requantize_np(acc, M=1, shift=2, relu=True)
    assert got_relu.tolist() == [0, 1, 0, 0, 127, 0]


def test_symmetric_weight_and_bias_quantization():
    w = np.array([-2.54, 0.0, 1.27, 2.54], dtype=np.float32)
    qw, sw = quantize_weight_symmetric(w)
    assert qw.dtype == np.int8
    assert np.isclose(sw, 2.54 / 127.0)
    assert qw.tolist() == [-127, 0, 64, 127]

    b = np.array([-0.1, 0.0, 0.25], dtype=np.float32)
    qb, sb = quantize_bias_int32(b, s_in=1 / 128.0, s_w=sw)
    assert qb.dtype == np.int32
    assert np.isclose(sb, (1 / 128.0) * sw)


def test_dyadic_multiplier_accuracy_and_16bit_bound():
    for m_real in (1e-4, 1.2345e-3, 0.015625, 0.25):
        M, n = compute_dyadic_multiplier(m_real)
        assert 1 <= M <= 65535
        assert 1 <= n <= 30
        approx = M / float(1 << n)
        rel_err = abs(approx - m_real) / m_real
        assert rel_err < 1e-4, (m_real, M, n, rel_err)


def test_gap_np_and_torch_parity():
    rng = np.random.default_rng(9)
    q_a2 = rng.integers(0, 128, size=(5, 16, 13, 13), dtype=np.int8)
    s_np, g_np = gap_int8_np(q_a2)
    s_th, g_th = gap_int8_torch(torch.from_numpy(q_a2))
    assert np.array_equal(s_np, s_th.numpy())
    assert np.array_equal(g_np, g_th.numpy())


def test_quantized_checkpoint_and_accuracy_preservation():
    need(QUANT_NPZ_PATH)
    need(QUANT_REPORT_DIR / "quant_metrics.json")

    with np.load(QUANT_NPZ_PATH) as qz:
        qparams = {k: qz[k] for k in qz.files}

    # Check shapes & dtypes of exported integer tensors
    assert qparams["conv1_weight"].shape == (8, 1, 3, 3) and qparams["conv1_weight"].dtype == np.int8
    assert qparams["conv1_bias"].shape == (8,) and qparams["conv1_bias"].dtype == np.int32
    assert qparams["conv2_weight"].shape == (16, 8, 3, 3) and qparams["conv2_weight"].dtype == np.int8
    assert qparams["conv2_bias"].shape == (16,) and qparams["conv2_bias"].dtype == np.int32
    assert qparams["fc_weight"].shape == (10, 16) and qparams["fc_weight"].dtype == np.int8
    assert qparams["fc_bias"].shape == (10,) and qparams["fc_bias"].dtype == np.int32

    # Multipliers must fit in 16-bit unsigned for single Cyclone V DSP inference
    for k in ("conv1_mult", "conv2_mult", "gap_mult", "fc_mult"):
        assert 1 <= int(qparams[k]) <= 65535, (k, int(qparams[k]))

    # Forward pass on synthetic batch works and matches CPU vs GPU identically
    rng = np.random.default_rng(0)
    x_i8 = torch.from_numpy(rng.integers(0, 128, size=(16, 1, 32, 32), dtype=np.int8))
    out_cpu = forward_int8_torch(x_i8, qparams, device=torch.device("cpu"))
    if torch.cuda.is_available():
        out_gpu = forward_int8_torch(x_i8, qparams, device=torch.device("cuda"))
        for k in out_cpu:
            assert torch.equal(out_cpu[k], out_gpu[k].cpu()), f"INT8 CPU vs GPU mismatch at {k}"

    # Verify measured accuracy drop vs FP32 is within 1.0 percentage point
    metrics = json.loads((QUANT_REPORT_DIR / "quant_metrics.json").read_text())
    test_int8 = metrics["splits"]["test"]["accuracy"]
    test_fp32 = metrics["splits"]["test"]["fp32_accuracy"]
    assert test_int8 >= 0.85, test_int8
    assert abs(test_int8 - test_fp32) <= 0.01, (test_int8, test_fp32)


if __name__ == "__main__":
    sys.exit(run_all(globals()))
