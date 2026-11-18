"""
Milestone 6: Independent integer-only CNN reference model (`sw/refmodel/int8_ref_model.py`).

Design requirements (Section 8 of project specification):
    1. Strictly integer-only arithmetic (INT8 / INT32 / INT64 intermediate product).
       Zero floating-point types (`float`, `np.float32`, `np.float64`) and zero
       PyTorch dependencies are used during inference.
    2. Loads all weights, biases, and dyadic requantization parameters directly
       from the exported `.mem` files in `mem/` using the exact hardware address
       equations documented in `docs/memory_map.md`.
    3. Simulates every hardware stage individually:
         - INT8 x INT8 -> INT32 convolution MAC + INT32 bias addition (`conv_engine.v`)
         - Dyadic multiply-shift-round-saturate requantization (`requantize`)
         - ReLU clamp to [0, 127] (`relu.v`)
         - 2x2 stride-2 signed MaxPool (`maxpool.v`)
         - Global Average Pooling over 13x13 = 169 positions (`gap.v`)
         - 16 -> 10 Fully Connected INT8 MAC + INT32 bias + requantization (`fully_connected.v`)
         - 10-input signed Argmax with lowest-index tie-breaking (`argmax.v`)
    4. Provides both:
         - `infer_single_hw_loop`: explicit nested-loop implementation using flat
           1D `.mem` arrays and hardware address equations (`addr = ((oc*C_in + ic)*3 + ky)*3 + kx`,
           `addr = (c*H + y)*W + x`, `addr = (oc << 4) | ic`), matching the RTL FSM
           cycle-by-cycle MAC sequence.
         - `infer_batch_int8`: pure-integer NumPy implementation that is bit-for-bit
           identical to `infer_single_hw_loop` and evaluates 10,000 images in <1 s.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from sw.config import MEM_DIR
from sw.export.formatters import (
    read_mem,
    unpack_conv_weights_oihw,
    unpack_fc_weights_oi,
)


@dataclass(frozen=True)
class HardwareMemParams:
    """
    All CNN parameters loaded directly from the exported `.mem` files in `mem/`.
    Both flat 1D hardware memory arrays (`*_flat`) and reshaped multi-dimensional
    views (`*_nd`) are stored so both hardware-address-indexed and vectorized
    integer inference can run from the exact same `.mem` source.
    """
    conv1_w_flat: np.ndarray   # (72,) int8
    conv1_b_flat: np.ndarray   # (8,) int32
    conv2_w_flat: np.ndarray   # (1152,) int8
    conv2_b_flat: np.ndarray   # (16,) int32
    fc_w_flat: np.ndarray      # (160,) int8
    fc_b_flat: np.ndarray      # (10,) int32
    qparams_flat: np.ndarray   # (8,) int32: [M1, n1, M2, n2, M_gap, n_gap, M_fc, n_fc]

    conv1_w_nd: np.ndarray     # (8, 1, 3, 3) int8
    conv2_w_nd: np.ndarray     # (16, 8, 3, 3) int8
    fc_w_nd: np.ndarray        # (10, 16) int8

    @property
    def conv1_mult(self) -> int:
        return int(self.qparams_flat[0])

    @property
    def conv1_shift(self) -> int:
        return int(self.qparams_flat[1])

    @property
    def conv2_mult(self) -> int:
        return int(self.qparams_flat[2])

    @property
    def conv2_shift(self) -> int:
        return int(self.qparams_flat[3])

    @property
    def gap_mult(self) -> int:
        return int(self.qparams_flat[4])

    @property
    def gap_shift(self) -> int:
        return int(self.qparams_flat[5])

    @property
    def fc_mult(self) -> int:
        return int(self.qparams_flat[6])

    @property
    def fc_shift(self) -> int:
        return int(self.qparams_flat[7])


def load_hw_params_from_mem(mem_dir: Path = MEM_DIR) -> HardwareMemParams:
    """Load all weights, biases, and quantization parameters from `.mem` files."""
    c1_w = read_mem(mem_dir / "conv1_weights.mem", width_bits=8)
    c1_b = read_mem(mem_dir / "conv1_bias.mem", width_bits=32)
    c2_w = read_mem(mem_dir / "conv2_weights.mem", width_bits=8)
    c2_b = read_mem(mem_dir / "conv2_bias.mem", width_bits=32)
    fc_w = read_mem(mem_dir / "fc_weights.mem", width_bits=8)
    fc_b = read_mem(mem_dir / "fc_bias.mem", width_bits=32)
    qp = read_mem(mem_dir / "quantization_params.mem", width_bits=32)

    assert c1_w.size == 72 and c1_b.size == 8
    assert c2_w.size == 1152 and c2_b.size == 16
    assert fc_w.size == 160 and fc_b.size == 10
    assert qp.size == 8

    return HardwareMemParams(
        conv1_w_flat=c1_w,
        conv1_b_flat=c1_b,
        conv2_w_flat=c2_w,
        conv2_b_flat=c2_b,
        fc_w_flat=fc_w,
        fc_b_flat=fc_b,
        qparams_flat=qp,
        conv1_w_nd=unpack_conv_weights_oihw(c1_w, (8, 1, 3, 3)),
        conv2_w_nd=unpack_conv_weights_oihw(c2_w, (16, 8, 3, 3)),
        fc_w_nd=unpack_fc_weights_oi(fc_w, (10, 16)),
    )


# ---------------------------------------------------------------------------
# Hardware-Exact Module Primitives (Pure Integer)
# ---------------------------------------------------------------------------
def requantize_signed_int8(acc_int32: np.ndarray, mult: int, shift: int) -> np.ndarray:
    """
    Dyadic fixed-point scaling with round-half-up and signed INT8 saturation [-128, 127]:
        prod   = int64(acc_int32) * int64(mult)
        rnd    = int64(1) << (shift - 1)
        scaled = (prod + rnd) >> shift
        out    = clip(scaled, -128, 127)
    """
    if acc_int32.dtype != np.int32:
        raise TypeError(f"Expected int32 accumulator, got {acc_int32.dtype}")
    prod = acc_int32.astype(np.int64) * np.int64(mult)
    rnd = np.int64(1) << np.int64(shift - 1)
    scaled = (prod + rnd) >> np.int64(shift)
    return np.clip(scaled, -128, 127).astype(np.int8)


def relu_int8(x_int8: np.ndarray) -> np.ndarray:
    """
    Hardware `relu.v`: replace negative signed INT8 values with 0, preserve [0, 127].
    Because max(0, clip(s, -128, 127)) == clip(s, 0, 127), applying `relu_int8`
    after `requantize_signed_int8` is bit-for-bit identical to clamping the
    requantized value directly to [0, 127].
    """
    if x_int8.dtype != np.int8:
        raise TypeError(f"Expected int8 input to ReLU, got {x_int8.dtype}")
    return np.maximum(x_int8, np.int8(0)).astype(np.int8)


def maxpool2x2_stride2_int8(x_int8: np.ndarray) -> np.ndarray:
    """
    Hardware `maxpool.v`: 2x2 window, stride 2, signed INT8 comparison.
    Supports shape (C, H, W) or (N, C, H, W) with even H, W.
    """
    if x_int8.dtype != np.int8:
        raise TypeError(f"Expected int8 input to MaxPool, got {x_int8.dtype}")
    h, w = x_int8.shape[-2], x_int8.shape[-1]
    assert h % 2 == 0 and w % 2 == 0, f"Expected even spatial dims, got {(h, w)}"
    leading = x_int8.shape[:-2]
    reshaped = x_int8.reshape(*leading, h // 2, 2, w // 2, 2)
    return reshaped.max(axis=(-3, -1)).astype(np.int8)


def gap_13x13_int8(x_int8: np.ndarray, mult: int, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Hardware `gap.v`:
      1. Accumulate all 13 x 13 = 169 signed INT8 values per channel into INT32 (`gap_sum`).
      2. Multiply by `mult` (`M_gap`), add `1 << (shift - 1)`, arithmetic shift right by `shift`,
         and saturate to INT8 `[0, 127]` (`gap_out`).
    Supports shape (C, 13, 13) or (N, C, 13, 13).
    """
    if x_int8.dtype != np.int8:
        raise TypeError(f"Expected int8 input to GAP, got {x_int8.dtype}")
    assert x_int8.shape[-2:] == (13, 13), f"Expected (..., 13, 13), got {x_int8.shape}"
    gap_sum = x_int8.astype(np.int32).sum(axis=(-2, -1), dtype=np.int32)
    gap_req = requantize_signed_int8(gap_sum, mult, shift)
    gap_out = relu_int8(gap_req)
    return gap_sum, gap_out


def conv2d_valid_3x3_int8_to_int32(x_int8: np.ndarray, w_int8: np.ndarray,
                                   b_int32: np.ndarray) -> np.ndarray:
    """
    Hardware `conv_engine.v` (vectorized over batch & spatial positions):
      - Signed INT8 input x Signed INT8 weight -> INT32 accumulation + INT32 bias.
      - Kernel: 3x3, stride 1, padding 0 (VALID).
    Supports `x_int8` of shape (C_in, H, W) or (N, C_in, H, W).
    """
    if x_int8.dtype != np.int8 or w_int8.dtype != np.int8 or b_int32.dtype != np.int32:
        raise TypeError("conv2d requires int8 inputs/weights and int32 bias")
    single = (x_int8.ndim == 3)
    if single:
        x_int8 = x_int8[np.newaxis, ...]

    n_batch, c_in, h_in, w_in = x_int8.shape
    c_out, c_in_w, kh, kw = w_int8.shape
    assert c_in == c_in_w and (kh, kw) == (3, 3)
    h_out = h_in - 2
    w_out = w_in - 2

    x32 = x_int8.astype(np.int32)
    w32 = w_int8.astype(np.int32)
    acc = np.broadcast_to(b_int32.reshape(1, c_out, 1, 1), (n_batch, c_out, h_out, w_out)).copy()

    # Exact 3x3 spatial accumulation in INT32
    for ky in range(3):
        for kx in range(3):
            patch = x32[:, :, ky:ky + h_out, kx:kx + w_out]  # (N, C_in, H_out, W_out)
            w_slice = w32[:, :, ky, kx]                       # (C_out, C_in)
            # Integer matrix-multiply across C_in in int32:
            acc += np.einsum("nchw,oc->nohw", patch, w_slice, optimize=True).astype(np.int32)

    return acc[0] if single else acc


def fc_int8_to_int32(x_int8: np.ndarray, w_int8: np.ndarray, b_int32: np.ndarray) -> np.ndarray:
    """
    Hardware `fully_connected.v` MAC + bias stage:
      acc[oc] = b[oc] + sum_{ic=0..15} int32(x[ic]) * int32(w[oc, ic])
    Supports `x_int8` of shape (16,) or (N, 16).
    """
    if x_int8.dtype != np.int8 or w_int8.dtype != np.int8 or b_int32.dtype != np.int32:
        raise TypeError("fc requires int8 inputs/weights and int32 bias")
    x32 = x_int8.astype(np.int32)
    w32 = w_int8.astype(np.int32)
    return (x32 @ w32.T + b_int32).astype(np.int32)


def argmax_int8_tiebreak_lowest(scores_int8: np.ndarray) -> np.ndarray:
    """
    Hardware `argmax.v`:
      Scan class index c = 0..9 using strict `>` comparison so ties break to the
      lowest class index (identical to `np.argmax`).
    """
    return np.argmax(scores_int8, axis=-1).astype(np.int64)


# ---------------------------------------------------------------------------
# 1. Explicit Hardware-Loop / Address-Mapped Single-Image Reference Model
# ---------------------------------------------------------------------------
def infer_single_hw_loop(img_flat_int8: np.ndarray,
                         params: HardwareMemParams) -> Dict[str, np.ndarray]:
    """
    Bit-exact sequential hardware simulation of a single 32x32 image stored as a
    flat 1,024-byte array (`img_flat_int8`, address = `(y << 5) | x`).

    Every memory access uses the exact 1D hardware address equations from
    `docs/memory_map.md` (`conv1_w_flat`, `conv1_b_flat`, `conv2_w_flat`,
    `conv2_b_flat`, `fc_w_flat`, `fc_b_flat`).
    """
    img_flat = np.asarray(img_flat_int8, dtype=np.int8).ravel()
    assert img_flat.size == 1024, f"Expected 1024-byte flat image, got {img_flat.size}"

    # --- Stage 1: Conv1 (1x32x32 -> 8x30x30) + Requant + ReLU1 ---
    conv1_acc_flat = np.empty(8 * 30 * 30, dtype=np.int32)
    for oc in range(8):
        b_val = np.int32(params.conv1_b_flat[oc])
        for oy in range(30):
            for ox in range(30):
                acc = b_val
                for ic in range(1):
                    for ky in range(3):
                        in_y = oy + ky
                        for kx in range(3):
                            in_x = ox + kx
                            in_addr = (in_y << 5) | in_x
                            w_addr = ((oc * 1 + ic) * 3 + ky) * 3 + kx
                            acc = np.int32(
                                acc + np.int32(img_flat[in_addr]) * np.int32(params.conv1_w_flat[w_addr])
                            )
                out_addr = (oc * 30 + oy) * 30 + ox
                conv1_acc_flat[out_addr] = acc

    conv1_req_flat = requantize_signed_int8(conv1_acc_flat, params.conv1_mult, params.conv1_shift)
    relu1_flat = relu_int8(conv1_req_flat)

    # --- Stage 2: MaxPool1 (8x30x30 -> 8x15x15, 2x2 stride 2) ---
    pool1_flat = np.empty(8 * 15 * 15, dtype=np.int8)
    for c in range(8):
        for py in range(15):
            y0 = py * 2
            for px in range(15):
                x0 = px * 2
                v00 = relu1_flat[(c * 30 + y0) * 30 + x0]
                v01 = relu1_flat[(c * 30 + y0) * 30 + (x0 + 1)]
                v10 = relu1_flat[(c * 30 + (y0 + 1)) * 30 + x0]
                v11 = relu1_flat[(c * 30 + (y0 + 1)) * 30 + (x0 + 1)]
                m0 = v00 if v00 >= v01 else v01
                m1 = v10 if v10 >= v11 else v11
                pool1_flat[(c * 15 + py) * 15 + px] = m0 if m0 >= m1 else m1

    # --- Stage 3: Conv2 (8x15x15 -> 16x13x13) + Requant + ReLU2 ---
    conv2_acc_flat = np.empty(16 * 13 * 13, dtype=np.int32)
    for oc in range(16):
        b_val = np.int32(params.conv2_b_flat[oc])
        for oy in range(13):
            for ox in range(13):
                acc = b_val
                for ic in range(8):
                    for ky in range(3):
                        in_y = oy + ky
                        for kx in range(3):
                            in_x = ox + kx
                            in_addr = (ic * 15 + in_y) * 15 + in_x
                            w_addr = ((oc * 8 + ic) * 3 + ky) * 3 + kx
                            acc = np.int32(
                                acc + np.int32(pool1_flat[in_addr]) * np.int32(params.conv2_w_flat[w_addr])
                            )
                out_addr = (oc * 13 + oy) * 13 + ox
                conv2_acc_flat[out_addr] = acc

    conv2_req_flat = requantize_signed_int8(conv2_acc_flat, params.conv2_mult, params.conv2_shift)
    relu2_flat = relu_int8(conv2_req_flat)

    # --- Stage 4: Global Average Pooling (16x13x13 -> 16) ---
    gap_sum_flat = np.empty(16, dtype=np.int32)
    for c in range(16):
        s = np.int32(0)
        base = c * 169
        for idx in range(169):
            s = np.int32(s + np.int32(relu2_flat[base + idx]))
        gap_sum_flat[c] = s
    gap_req_flat = requantize_signed_int8(gap_sum_flat, params.gap_mult, params.gap_shift)
    gap_out_flat = relu_int8(gap_req_flat)

    # --- Stage 5: Fully Connected (16 -> 10) ---
    fc_acc_flat = np.empty(10, dtype=np.int32)
    for oc in range(10):
        acc = np.int32(params.fc_b_flat[oc])
        for ic in range(16):
            w_addr = (oc << 4) | ic
            acc = np.int32(acc + np.int32(gap_out_flat[ic]) * np.int32(params.fc_w_flat[w_addr]))
        fc_acc_flat[oc] = acc
    fc_scores_flat = requantize_signed_int8(fc_acc_flat, params.fc_mult, params.fc_shift)

    # --- Stage 6: Argmax (10 -> 1 class index, strict `>` tie-break to lowest index) ---
    best_cls = 0
    best_score = fc_scores_flat[0]
    for c in range(1, 10):
        if fc_scores_flat[c] > best_score:
            best_score = fc_scores_flat[c]
            best_cls = c

    return {
        "input_int8": img_flat.reshape(1, 32, 32),
        "conv1_acc_int32": conv1_acc_flat.reshape(8, 30, 30),
        "conv1_req_int8": conv1_req_flat.reshape(8, 30, 30),
        "relu1_int8": relu1_flat.reshape(8, 30, 30),
        "pool1_int8": pool1_flat.reshape(8, 15, 15),
        "conv2_acc_int32": conv2_acc_flat.reshape(16, 13, 13),
        "conv2_req_int8": conv2_req_flat.reshape(16, 13, 13),
        "relu2_int8": relu2_flat.reshape(16, 13, 13),
        "gap_sum_int32": gap_sum_flat,
        "gap_int8": gap_out_flat,
        "fc_acc_int32": fc_acc_flat,
        "fc_scores_int8": fc_scores_flat,
        "pred_class": np.int64(best_cls),
    }


# ---------------------------------------------------------------------------
# 2. Vectorized Pure-Integer Reference Model (Single Image or Batch)
# ---------------------------------------------------------------------------
def infer_batch_int8(q_in: np.ndarray, params: HardwareMemParams) -> Dict[str, np.ndarray]:
    """
    Pure-integer batch or single-image inference using `HardwareMemParams`.
    Accepts `q_in` of shape (32, 32), (1, 32, 32), (N, 32, 32), or (N, 1, 32, 32)
    with dtype `int8`. Returns all intermediate integer tensors.
    """
    q_in = np.asarray(q_in, dtype=np.int8)
    single = False
    if q_in.ndim == 2:
        q_in = q_in[np.newaxis, np.newaxis, :, :]
        single = True
    elif q_in.ndim == 3:
        if q_in.shape == (1, 32, 32):
            q_in = q_in[np.newaxis, :, :, :]
            single = True
        else:
            q_in = q_in[:, np.newaxis, :, :]

    assert q_in.ndim == 4 and q_in.shape[1:] == (1, 32, 32), (
        f"Expected (N, 1, 32, 32) int8 input, got {q_in.shape}"
    )

    conv1_acc = conv2d_valid_3x3_int8_to_int32(q_in, params.conv1_w_nd, params.conv1_b_flat)
    conv1_req = requantize_signed_int8(conv1_acc, params.conv1_mult, params.conv1_shift)
    relu1 = relu_int8(conv1_req)
    pool1 = maxpool2x2_stride2_int8(relu1)

    conv2_acc = conv2d_valid_3x3_int8_to_int32(pool1, params.conv2_w_nd, params.conv2_b_flat)
    conv2_req = requantize_signed_int8(conv2_acc, params.conv2_mult, params.conv2_shift)
    relu2 = relu_int8(conv2_req)

    gap_sum, gap_out = gap_13x13_int8(relu2, params.gap_mult, params.gap_shift)
    fc_acc = fc_int8_to_int32(gap_out, params.fc_w_nd, params.fc_b_flat)
    fc_scores = requantize_signed_int8(fc_acc, params.fc_mult, params.fc_shift)
    preds = argmax_int8_tiebreak_lowest(fc_scores)

    out = {
        "input_int8": q_in,
        "conv1_acc_int32": conv1_acc,
        "conv1_req_int8": conv1_req,
        "relu1_int8": relu1,
        "pool1_int8": pool1,
        "conv2_acc_int32": conv2_acc,
        "conv2_req_int8": conv2_req,
        "relu2_int8": relu2,
        "gap_sum_int32": gap_sum,
        "gap_int8": gap_out,
        "fc_acc_int32": fc_acc,
        "fc_scores_int8": fc_scores,
        "pred_class": preds,
    }
    if single:
        return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in out.items()}
    return out
