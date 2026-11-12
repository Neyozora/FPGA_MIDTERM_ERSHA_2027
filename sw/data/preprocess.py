"""
Pixel -> INT8 input conversion and 28x28 -> 32x32 zero-padding. This is the
ONLY place where raw pixels are turned into network inputs; training,
quantization, export and the integer reference model must all call these
functions.

Convention (see docs/preprocessing.md for the rationale):

    1. 2-pixel symmetric zero-padding around each 28x28 image -> 32x32
    2. q_in = p >> 1              p in [0,255] (uint8)  ->  q_in in [0,127] (int8)
    3. x    = q_in * 2^-7         real value seen by the FP32 model, x in [0, 0.9921875]

    * symmetric quantization, zero point = 0 (so zero-padding in uint8, int8,
      and float32 is identically 0)
    * scale s_in = 1/128 (power of two -> exact in float32)
    * the conversion is a 1-bit logical right shift (truncation), trivially
      implementable in hardware if pixels are ever streamed in raw form
    * 32x32 = 1,024 bytes = 2^10 matches Cyclone V M10K natural depth (1024x8b)
"""
import numpy as np

from sw.config import (IMG_H, IMG_W, INPUT_FRAC_BITS, INPUT_PAD, INPUT_QMAX, INPUT_QMIN,
                       INPUT_SCALE, INPUT_SHIFT, RAW_IMG_H, RAW_IMG_W)


def pad_to_32x32(images: np.ndarray, pad: int = INPUT_PAD) -> np.ndarray:
    """Zero-pad [..., 28, 28] images by `pad` pixels on all 4 sides -> [..., 32, 32]."""
    images = np.asarray(images)
    if images.ndim < 2 or images.shape[-2:] != (RAW_IMG_H, RAW_IMG_W):
        raise ValueError(f"expected [..., {RAW_IMG_H}, {RAW_IMG_W}], got {images.shape}")
    pad_width = [(0, 0)] * (images.ndim - 2) + [(pad, pad), (pad, pad)]
    out = np.pad(images, pad_width, mode="constant", constant_values=0)
    assert out.shape[-2:] == (IMG_H, IMG_W)
    return out


def pixels_to_int8(pixels: np.ndarray, pad_28_to_32: bool = False) -> np.ndarray:
    """uint8 pixels -> int8 hardware input codes in [0, 127] (optionally zero-padded to 32x32)."""
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        raise TypeError(f"expected uint8 pixels, got {pixels.dtype}")
    if pad_28_to_32:
        pixels = pad_to_32x32(pixels)
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
