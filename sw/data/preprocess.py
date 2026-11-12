"""
Pixel -> INT8 input conversion. This is the ONLY place where raw pixels are
turned into network inputs; training, quantization, export and the integer
reference model must all call these functions.

Convention (see docs/preprocessing.md for the rationale):

    q_in = p >> 1                 p in [0,255] (uint8)  ->  q_in in [0,127] (int8)
    x    = q_in * 2^-7            real value seen by the FP32 model, x in [0, 0.9921875]

    * symmetric quantization, zero point = 0
    * scale s_in = 1/128 (power of two -> exact in float32)
    * the conversion is a 1-bit logical right shift (truncation), trivially
      implementable in hardware if pixels are ever streamed in raw form
    * no mean/std normalization: a per-input affine normalization
      x' = (x - mu)/sigma is mathematically absorbed by conv1's weights and
      bias, so it adds nothing in exact arithmetic and would complicate the
      integer pipeline (non-zero zero point, negative inputs)
"""
import numpy as np

from sw.config import INPUT_FRAC_BITS, INPUT_QMAX, INPUT_QMIN, INPUT_SCALE, INPUT_SHIFT


def pixels_to_int8(pixels: np.ndarray) -> np.ndarray:
    """uint8 pixels (any shape) -> int8 hardware input codes in [0, 127]."""
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        raise TypeError(f"expected uint8 pixels, got {pixels.dtype}")
    q = (pixels >> INPUT_SHIFT).astype(np.int8)
    # Cannot fail for uint8 input, but keep the contract explicit.
    assert q.min(initial=INPUT_QMIN) >= INPUT_QMIN and q.max(initial=INPUT_QMIN) <= INPUT_QMAX
    return q


def int8_to_float(q: np.ndarray) -> np.ndarray:
    """int8 input codes -> float32 values used by the FP32 / QAT model (exact)."""
    q = np.asarray(q)
    if q.dtype != np.int8:
        raise TypeError(f"expected int8 codes, got {q.dtype}")
    return q.astype(np.float32) * np.float32(INPUT_SCALE)


def float_to_int8(x: np.ndarray) -> np.ndarray:
    """Inverse of int8_to_float for values that lie exactly on the input grid."""
    x = np.asarray(x, dtype=np.float32)
    q = x * np.float32(2 ** INPUT_FRAC_BITS)
    if not np.array_equal(q, np.round(q)):
        raise ValueError("value not on the INT8 input grid")
    if q.min() < INPUT_QMIN or q.max() > INPUT_QMAX:
        raise ValueError("value outside the input code range")
    return q.astype(np.int8)
