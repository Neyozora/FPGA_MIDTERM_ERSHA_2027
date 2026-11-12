"""
Minimal, dependency-free reader/writer for the IDX file format used by
(Fashion-)MNIST.

IDX layout (all header integers are big-endian):
    byte 0..1 : 0x00 0x00
    byte 2    : data type code (0x08 = uint8, 0x09 = int8, 0x0B = int16,
                0x0C = int32, 0x0D = float32, 0x0E = float64)
    byte 3    : number of dimensions D
    next 4*D  : size of each dimension (uint32, big-endian)
    rest      : data, row-major (C order), last dimension varies fastest

For Fashion-MNIST:
    images: type 0x08, D = 3, dims = (N, 28, 28)   -> magic 0x00000803
    labels: type 0x08, D = 1, dims = (N,)          -> magic 0x00000801
"""
import gzip
import struct
from pathlib import Path

import numpy as np

IDX_DTYPES = {
    0x08: np.dtype("u1"),
    0x09: np.dtype("i1"),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}
_DTYPE_TO_CODE = {np.dtype("u1"): 0x08, np.dtype("i1"): 0x09}


def _open(path: Path, mode: str):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def read_idx(path) -> np.ndarray:
    """Parse an IDX (optionally gzipped) file and return a native-endian ndarray."""
    path = Path(path)
    with _open(path, "rb") as f:
        data = f.read()
    if len(data) < 4:
        raise ValueError(f"{path}: file too short for an IDX header")
    if data[0] != 0 or data[1] != 0:
        raise ValueError(f"{path}: bad IDX magic (first two bytes must be zero)")
    code, ndim = data[2], data[3]
    if code not in IDX_DTYPES:
        raise ValueError(f"{path}: unsupported IDX data type 0x{code:02X}")
    header_len = 4 + 4 * ndim
    dims = struct.unpack(">" + "I" * ndim, data[4:header_len])
    dt = IDX_DTYPES[code]
    expected = int(np.prod(dims, dtype=np.int64)) * dt.itemsize
    if len(data) - header_len != expected:
        raise ValueError(
            f"{path}: payload is {len(data) - header_len} bytes, header implies {expected}"
        )
    arr = np.frombuffer(data, dtype=dt, offset=header_len).reshape(dims)
    return arr.astype(dt.newbyteorder("="), copy=True)


def write_idx(path, arr: np.ndarray) -> None:
    """Write a uint8/int8 ndarray as IDX (used by the unit tests)."""
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _DTYPE_TO_CODE:
        raise TypeError("write_idx only supports uint8/int8")
    header = bytes([0, 0, _DTYPE_TO_CODE[arr.dtype], arr.ndim])
    header += struct.pack(">" + "I" * arr.ndim, *arr.shape)
    with _open(Path(path), "wb") as f:
        f.write(header + arr.tobytes(order="C"))
