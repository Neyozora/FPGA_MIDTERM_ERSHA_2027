"""
Milestone 2 tests: model structure and tensor shapes.

    python -m sw.tests.test_m2_model
    python -m pytest sw/tests/test_m2_model.py
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

from sw.data.dataset import PROCESSED_FILE, load_processed
from sw.data.preprocess import int8_to_float
from sw.model.cnn import (C1_H, C2_H, GAP_N, P1_H, SHAPES, FashionCNN, ShapeError,
                          expected_macs, expected_param_counts, out_size)
from sw.tests._runner import need, run_all, skip


def _model(seed=0):
    torch.manual_seed(seed)
    return FashionCNN().eval()


def test_output_size_formula():
    assert out_size(32, 3) == 30
    assert out_size(30, 2, 2) == 15
    assert out_size(15, 3) == 13
    assert (C1_H, P1_H, C2_H, GAP_N) == (30, 15, 13, 169)


def test_spec_shapes_table():
    assert list(SHAPES.items()) == [
        ("input", (1, 32, 32)), ("conv1", (8, 30, 30)), ("relu1", (8, 30, 30)),
        ("pool1", (8, 15, 15)), ("conv2", (16, 13, 13)), ("relu2", (16, 13, 13)),
        ("gap", (16,)), ("fc", (10,))]


def test_forward_shapes_various_batch_sizes():
    m = _model()
    for n in (1, 2, 7, 64):
        with torch.no_grad():
            inter = m.forward_intermediates(torch.rand(n, 1, 32, 32))
        for name, t in inter.items():
            assert tuple(t.shape) == (n,) + SHAPES[name], (name, t.shape)


def test_param_counts():
    m = _model()
    got = {n: p.numel() for n, p in m.named_parameters()}
    assert got == dict(expected_param_counts())
    assert sum(got.values()) == 1418
    assert tuple(m.conv1.weight.shape) == (8, 1, 3, 3)    # PyTorch OIHW layout
    assert tuple(m.conv2.weight.shape) == (16, 8, 3, 3)
    assert tuple(m.fc.weight.shape) == (10, 16)           # [out, in]


def test_macs():
    assert dict(expected_macs()) == {"conv1": 64800, "conv2": 194688, "fc": 160}


def test_wrong_input_shapes_raise():
    m = _model()
    for bad in (torch.rand(1, 1, 28, 28), torch.rand(1, 3, 32, 32), torch.rand(1, 32, 32)):
        try:
            m(bad)
        except ShapeError:
            continue
        except RuntimeError:  # a conv may reject the shape before our check does
            continue
        raise AssertionError(f"input {list(bad.shape)} was accepted")


def test_forward_equals_forward_intermediates():
    m = _model()
    x = torch.rand(5, 1, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(x), m.forward_intermediates(x)["fc"])


def test_gap_is_sum_over_169():
    m = _model()
    with torch.no_grad():
        t = m.forward_intermediates(torch.rand(3, 1, 32, 32))
        ref = t["relu2"].sum(dim=(2, 3)) / 169.0
    assert torch.allclose(t["gap"], ref, atol=1e-6)


def test_relu_maxpool_commute():
    """max(relu(a), relu(b)) == relu(max(a, b)): lets the RTL pool before or after ReLU."""
    x = torch.randn(4, 8, 30, 30)
    assert torch.equal(F.max_pool2d(F.relu(x), 2, 2), F.relu(F.max_pool2d(x, 2, 2)))


def test_backward_gradients_finite():
    m = FashionCNN().train()
    out = m(torch.rand(8, 1, 32, 32))
    F.cross_entropy(out, torch.randint(0, 10, (8,))).backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_real_dataset_batch():
    need(PROCESSED_FILE)
    d = load_processed()
    x = torch.from_numpy(int8_to_float(d["x_val"][:16])[:, None])
    assert tuple(x.shape) == (16, 1, 32, 32)
    assert x.min() >= 0 and x.max() <= 127 / 128
    with torch.no_grad():
        out = _model()(x)
    assert tuple(out.shape) == (16, 10) and torch.isfinite(out).all()


def test_predict_tie_breaks_to_lowest_index():
    # documents the convention the RTL argmax must follow
    scores = torch.tensor([[1.0, 3.0, 3.0, 0.0, 0, 0, 0, 0, 0, 0]])
    assert int(scores.argmax(dim=1)) == 1


def test_gpu_matches_cpu():
    if not torch.cuda.is_available():
        skip("CUDA not available")
    m = _model()
    x = torch.rand(32, 1, 32, 32)
    with torch.no_grad():
        cpu = m(x)
        m_gpu = FashionCNN().cuda().eval()
        m_gpu.load_state_dict(m.state_dict())
        gpu = m_gpu(x.cuda()).cpu()
    assert float((cpu - gpu).abs().max()) < 1e-4


def test_init_is_seed_deterministic():
    a = _model(seed=123).state_dict()
    b = _model(seed=123).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


if __name__ == "__main__":
    sys.exit(run_all(globals()))
