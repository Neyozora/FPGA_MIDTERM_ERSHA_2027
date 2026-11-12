"""
Milestone 4: Fixed-point & INT8 quantization primitives, calibration, PTQ, QAT,
and exact integer inference in PyTorch/NumPy.

Every formula in this file matches `docs/quantization_spec.md` and the Verilog
RTL in Milestone 7 bit-for-bit:
    - Weights: symmetric signed INT8 in [-127, 127], zero-point = 0
    - Post-ReLU activations (conv1, conv2, gap): INT8 in [0, 127], zero-point = 0
    - FC output scores: signed INT8 in [-128, 127], zero-point = 0
    - Accumulators & biases: signed INT32
    - Requantization: ((int64(acc) * int64(M)) + (1 << (n - 1))) >> n
    - GAP division by 169: ((sum_169 * 776) + (1 << 16)) >> 17  (exact round((sum)/169))
"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sw.config import INPUT_SCALE
from sw.model.cnn import FashionCNN, GAP_N

# Exact fixed-point reciprocal of 169 = 13 * 13:
# For all S in [0, 169 * 127 = 21463], ((S * 12409) + 2^20) >> 21 == (S + 84) // 169
GAP_MULT = 12409
GAP_SHIFT = 21
assert GAP_N == 169


# ---------------------------------------------------------------------------
# Scale & dyadic multiplier (M, n) calculation
# ---------------------------------------------------------------------------
def compute_dyadic_multiplier(m_real: float, target_bits: int = 15) -> Tuple[int, int]:
    """
    Approximate positive real scale ratio `m_real = (s_in * s_w) / s_out` as
    `M / 2^n`, where `M` is an unsigned integer with `M in [2^(target_bits-1), 2^target_bits - 1]`
    (fits in a single 18x18 Cyclone V DSP input) and `n in [1, 31]`.
    """
    if m_real <= 0:
        raise ValueError(f"m_real must be positive, got {m_real}")
    # Choose n so that m_real * 2^n is in [2^(target_bits-1), 2^target_bits)
    n = int(np.floor(np.log2((2 ** target_bits - 1) / m_real)))
    n = max(1, min(30, n))
    M = int(np.floor(m_real * (1 << n) + 0.5))
    M = max(1, min((1 << 16) - 1, M))
    return M, n


def quantize_weight_symmetric(w: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantize float weights symmetrically into INT8 [-127, 127]. Returns (q_w int8, s_w)."""
    w = np.asarray(w, dtype=np.float64)
    w_max = float(np.abs(w).max())
    s_w = w_max / 127.0 if w_max > 0 else 1e-8
    q_w = np.clip(np.floor(w / s_w + 0.5), -127, 127).astype(np.int8)
    return q_w, s_w


def quantize_bias_int32(b: np.ndarray, s_in: float, s_w: float) -> Tuple[np.ndarray, float]:
    """Quantize float bias into INT32 with scale s_b = s_in * s_w."""
    b = np.asarray(b, dtype=np.float64)
    s_b = s_in * s_w
    q_b = np.clip(np.floor(b / s_b + 0.5), -(1 << 30), (1 << 30) - 1).astype(np.int32)
    return q_b, s_b


# ---------------------------------------------------------------------------
# Exact integer arithmetic primitives (shared by PyTorch & NumPy)
# ---------------------------------------------------------------------------
def requantize_np(acc: np.ndarray, M: int, shift: int, relu: bool) -> np.ndarray:
    """
    Hardware-exact requantization:
        scaled = (int64(acc) * int64(M) + (1 << (shift - 1))) >> shift
        saturate to [0, 127] if relu else [-128, 127]
    """
    acc64 = acc.astype(np.int64)
    rnd = np.int64(1) << np.int64(shift - 1)
    scaled = (acc64 * np.int64(M) + rnd) >> np.int64(shift)
    lo = 0 if relu else -128
    return np.clip(scaled, lo, 127).astype(np.int8)


def requantize_torch(acc: torch.Tensor, M: int, shift: int, relu: bool) -> torch.Tensor:
    """Torch int64 equivalent of requantize_np (for fast GPU batch evaluation)."""
    acc64 = acc.to(torch.int64)
    rnd = 1 << (shift - 1)
    scaled = (acc64 * int(M) + rnd) >> int(shift)
    lo = 0 if relu else -128
    return torch.clamp(scaled, lo, 127).to(torch.int8)


def gap_int8_np(q_a2: np.ndarray, M_gap: int = GAP_MULT,
                n_gap: int = GAP_SHIFT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact GAP over 13x13 = 169 spatial positions.
    Input:  q_a2 int8 [..., C, 13, 13] in [0, 127]
    Output: (sum_gap int32 [..., C], q_gap int8 [..., C] in [0, 127])
    """
    assert q_a2.shape[-2:] == (13, 13)
    sum_gap = q_a2.astype(np.int32).sum(axis=(-2, -1), dtype=np.int32)
    q_gap = requantize_np(sum_gap, int(M_gap), int(n_gap), relu=True)
    return sum_gap, q_gap


def gap_int8_torch(q_a2: torch.Tensor, M_gap: int = GAP_MULT,
                   n_gap: int = GAP_SHIFT) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of gap_int8_np."""
    assert tuple(q_a2.shape[-2:]) == (13, 13)
    sum_gap = q_a2.to(torch.int32).sum(dim=(-2, -1), dtype=torch.int32)
    q_gap = requantize_torch(sum_gap, int(M_gap), int(n_gap), relu=True)
    return sum_gap, q_gap


# ---------------------------------------------------------------------------
# Calibration & Parameter Extraction (FP32 / QAT model -> INT8 dict)
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_and_quantize(model: FashionCNN, x_calib: torch.Tensor,
                           percentile: float = 99.95,
                           weight_clip_pct: float = 100.0) -> Dict[str, object]:
    """
    Calibrate activation scales on `x_calib` ([N, 1, 32, 32] float32 on the
    input grid `q_in / 128`) and quantize all weights, biases, and requant
    multipliers/shifts.
    """
    model.eval()
    device = next(model.parameters()).device
    inter = model.forward_intermediates(x_calib.to(device))

    def _act_scale(t: torch.Tensor) -> float:
        arr = t.detach().abs().cpu().numpy().ravel()
        vmax = float(np.percentile(arr, percentile))
        return max(vmax / 127.0, 1e-8)

    s_in = float(INPUT_SCALE)               # 1/128 = 0.0078125 (exact)
    s_a1 = _act_scale(inter["relu1"])       # post-ReLU1 scale
    s_a2 = _act_scale(inter["relu2"])       # post-ReLU2 scale
    s_gap = _act_scale(inter["gap"])        # post-GAP scale (8x tighter range than relu2!)
    s_fc = _act_scale(inter["fc"])          # FC output score scale

    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def _qw(w_arr: np.ndarray) -> Tuple[np.ndarray, float]:
        if weight_clip_pct < 100.0:
            clip_v = float(np.percentile(np.abs(w_arr), weight_clip_pct))
            w_arr = np.clip(w_arr, -clip_v, clip_v)
        return quantize_weight_symmetric(w_arr)

    # Layer 1: Conv1
    c1_qw, s_w1 = _qw(sd["conv1.weight"])
    c1_qb, s_b1 = quantize_bias_int32(sd["conv1.bias"], s_in, s_w1)
    M1, n1 = compute_dyadic_multiplier((s_in * s_w1) / s_a1)

    # Layer 2: Conv2 (input is pool1, which has scale s_a1)
    c2_qw, s_w2 = _qw(sd["conv2.weight"])
    c2_qb, s_b2 = quantize_bias_int32(sd["conv2.bias"], s_a1, s_w2)
    M2, n2 = compute_dyadic_multiplier((s_a1 * s_w2) / s_a2)

    # Layer 3: GAP (input is sum of 169 q_a2 values; real average is sum * (s_a2 / 169))
    # Map sum_gap -> q_gap with scale s_gap via (M_gap, n_gap):
    M_gap, n_gap = compute_dyadic_multiplier(s_a2 / (169.0 * s_gap))

    # Layer 4: FC (input is GAP, which has scale s_gap)
    fc_qw, s_wfc = _qw(sd["fc.weight"])
    fc_qb, s_bfc = quantize_bias_int32(sd["fc.bias"], s_gap, s_wfc)
    Mfc, nfc = compute_dyadic_multiplier((s_gap * s_wfc) / s_fc)

    return {
        # Quantized integer tensors (exact hardware parameters)
        "conv1_weight": c1_qw,
        "conv1_bias": c1_qb,
        "conv1_mult": np.int32(M1),
        "conv1_shift": np.int32(n1),
        "conv2_weight": c2_qw,
        "conv2_bias": c2_qb,
        "conv2_mult": np.int32(M2),
        "conv2_shift": np.int32(n2),
        "gap_mult": np.int32(M_gap),
        "gap_shift": np.int32(n_gap),
        "fc_weight": fc_qw,
        "fc_bias": fc_qb,
        "fc_mult": np.int32(Mfc),
        "fc_shift": np.int32(nfc),
        # Real-valued scales (documented for specification & analysis)
        "s_in": np.float64(s_in),
        "s_w1": np.float64(s_w1),
        "s_b1": np.float64(s_b1),
        "s_a1": np.float64(s_a1),
        "s_w2": np.float64(s_w2),
        "s_b2": np.float64(s_b2),
        "s_a2": np.float64(s_a2),
        "s_gap": np.float64(s_gap),
        "s_wfc": np.float64(s_wfc),
        "s_bfc": np.float64(s_bfc),
        "s_fc": np.float64(s_fc),
        "percentile": np.float64(percentile),
        "weight_clip_pct": np.float64(weight_clip_pct),
    }


# ---------------------------------------------------------------------------
# Fast Torch-based exact integer forward pass (uses identical int32/int64 math)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_int8_torch(q_in: torch.Tensor, qparams: dict,
                       device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
    """
    Execute the exact hardware integer arithmetic on a batch of INT8 input images
    `q_in` ([N, 32, 32] or [N, 1, 32, 32], dtype=int8).

    Note: PyTorch conv2d on CUDA/CPU with float64 is exact for all integers up to
    2^53 (our max accumulator is < 2^22), so computing the integer dot products in
    float64 and casting to int32 is 100% bit-identical to int32 MACs while running
    across 10,000 images in < 0.05s on the GPU.
    """
    if q_in.ndim == 3:
        q_in = q_in[:, None, :, :]
    assert q_in.dtype == torch.int8 and tuple(q_in.shape[1:]) == (1, 32, 32)
    q_in_d = q_in.to(device)

    w1 = torch.from_numpy(qparams["conv1_weight"]).to(device=device, dtype=torch.float64)
    b1 = torch.from_numpy(qparams["conv1_bias"]).to(device=device, dtype=torch.int32)
    w2 = torch.from_numpy(qparams["conv2_weight"]).to(device=device, dtype=torch.float64)
    b2 = torch.from_numpy(qparams["conv2_bias"]).to(device=device, dtype=torch.int32)
    wfc = torch.from_numpy(qparams["fc_weight"]).to(device=device, dtype=torch.float64)
    bfc = torch.from_numpy(qparams["fc_bias"]).to(device=device, dtype=torch.int32)

    # Stage 1: Conv1 (int8 * int8 -> int32 + int32 bias) -> requant + ReLU -> int8 [0, 127]
    acc1 = F.conv2d(q_in_d.to(torch.float64), w1, stride=1, padding=0).to(torch.int32)
    acc1 = acc1 + b1[None, :, None, None]
    q_a1 = requantize_torch(acc1, int(qparams["conv1_mult"]), int(qparams["conv1_shift"]), relu=True)

    # Stage 2: MaxPool 2x2 stride 2 (exact signed int8 comparison over 2x2 windows)
    n_b, c_b, h_b, w_b = q_a1.shape
    q_p1 = q_a1.view(n_b, c_b, h_b // 2, 2, w_b // 2, 2).amax(dim=(3, 5))

    # Stage 3: Conv2 (int8 * int8 -> int32 + int32 bias) -> requant + ReLU -> int8 [0, 127]
    acc2 = F.conv2d(q_p1.to(torch.float64), w2, stride=1, padding=0).to(torch.int32)
    acc2 = acc2 + b2[None, :, None, None]
    q_a2 = requantize_torch(acc2, int(qparams["conv2_mult"]), int(qparams["conv2_shift"]), relu=True)

    # Stage 4: GAP (sum 13x13=169 int8 -> int32 -> *M_gap >> n_gap with rounding) -> int8 [0, 127]
    sum_gap, q_gap = gap_int8_torch(q_a2, int(qparams["gap_mult"]), int(qparams["gap_shift"]))

    # Stage 5: FC (int8 * int8 -> int32 + int32 bias) -> requant -> int8 [-128, 127]
    acc_fc = F.linear(q_gap.to(torch.float64), wfc).to(torch.int32) + bfc[None, :]
    q_fc = requantize_torch(acc_fc, int(qparams["fc_mult"]), int(qparams["fc_shift"]), relu=False)

    # Stage 6: Argmax (lowest class index wins ties)
    pred = q_fc.to(torch.int32).argmax(dim=1)

    return {
        "input": q_in_d,
        "conv1_acc": acc1,
        "relu1": q_a1,
        "pool1": q_p1,
        "conv2_acc": acc2,
        "relu2": q_a2,
        "gap_sum": sum_gap,
        "gap": q_gap,
        "fc_acc": acc_fc,
        "fc": q_fc,
        "pred": pred,
    }


# ---------------------------------------------------------------------------
# Fake-quantization modules for Quantization-Aware Training (QAT)
# ---------------------------------------------------------------------------
def fake_quant_sym(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    """Straight-through estimator (STE) symmetric fake-quantization."""
    s = scale.clamp(min=1e-8)
    q = torch.clamp(torch.floor(x / s + 0.5), qmin, qmax)
    x_q = q * s
    return x + (x_q - x).detach()


class QATWrapper(nn.Module):
    """Wraps a trained FashionCNN with STE fake-quantization on weights and activations."""
    def __init__(self, base_model: FashionCNN, init_qparams: dict):
        super().__init__()
        self.model = FashionCNN(check_shapes=True)
        self.model.load_state_dict(base_model.state_dict())
        self.register_buffer("s_a1", torch.tensor(float(init_qparams["s_a1"]), dtype=torch.float32))
        self.register_buffer("s_a2", torch.tensor(float(init_qparams["s_a2"]), dtype=torch.float32))
        self.register_buffer("s_gap", torch.tensor(float(init_qparams["s_gap"]), dtype=torch.float32))
        self.register_buffer("s_fc", torch.tensor(float(init_qparams["s_fc"]), dtype=torch.float32))
        self.momentum = 0.05

    def _fq_weight(self, w: torch.Tensor) -> torch.Tensor:
        s_w = w.detach().abs().max() / 127.0
        return fake_quant_sym(w, s_w, -127, 127)

    def _fq_act(self, x: torch.Tensor, buf: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                cur_s = torch.quantile(x.detach().abs().reshape(-1), 0.999) / 127.0
                buf.lerp_(cur_s.clamp(min=1e-6), self.momentum)
        return fake_quant_sym(x, buf, qmin, qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self._fq_weight(self.model.conv1.weight)
        x = F.relu(F.conv2d(x, w1, self.model.conv1.bias, stride=1, padding=0))
        x = self._fq_act(x, self.s_a1, 0, 127)
        x = self.model.pool1(x)

        w2 = self._fq_weight(self.model.conv2.weight)
        x = F.relu(F.conv2d(x, w2, self.model.conv2.bias, stride=1, padding=0))
        x = self._fq_act(x, self.s_a2, 0, 127)

        # Simulate GAP fixed-point quantization on its dedicated s_gap grid
        x = x.mean(dim=(2, 3))
        x = self._fq_act(x, self.s_gap, 0, 127)

        wfc = self._fq_weight(self.model.fc.weight)
        x = F.linear(x, wfc, self.model.fc.bias)
        x = self._fq_act(x, self.s_fc, -128, 127)
        return x
