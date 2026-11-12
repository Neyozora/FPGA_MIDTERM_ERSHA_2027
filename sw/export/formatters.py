"""
Milestone 5: Two's-complement `.mem` ($readmemh) and `.mif` (Intel Quartus)
writers, independent read-back parsers, and explicit PyTorch -> hardware
memory layout conversion functions.
"""
import re
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Two's-complement conversion helpers
# ---------------------------------------------------------------------------
def int_to_twos_hex(val: int, width_bits: int) -> str:
    """Convert a signed integer `val` to an uppercase two's-complement hex string."""
    if width_bits not in (8, 16, 32):
        raise ValueError(f"unsupported width_bits={width_bits}")
    lo = -(1 << (width_bits - 1))
    hi = (1 << (width_bits - 1)) - 1
    val = int(val)
    if val < lo or val > hi:
        raise ValueError(f"value {val} out of signed {width_bits}-bit range [{lo}, {hi}]")
    mask = (1 << width_bits) - 1
    uval = val & mask
    hex_digits = width_bits // 4
    return f"{uval:0{hex_digits}X}"


def twos_hex_to_int(hex_str: str, width_bits: int) -> int:
    """Parse a two's-complement hex token back into a signed Python int."""
    if width_bits not in (8, 16, 32):
        raise ValueError(f"unsupported width_bits={width_bits}")
    hex_str = hex_str.strip()
    expected_chars = width_bits // 4
    if len(hex_str) != expected_chars:
        raise ValueError(f"expected {expected_chars} hex chars for {width_bits}-bit word, got {hex_str!r}")
    uval = int(hex_str, 16)
    sign_bit = 1 << (width_bits - 1)
    if uval & sign_bit:
        return uval - (1 << width_bits)
    return uval


# ---------------------------------------------------------------------------
# Explicit PyTorch -> Hardware flat memory ordering conversions
# ---------------------------------------------------------------------------
def pack_conv_weights_oihw(w: np.ndarray) -> np.ndarray:
    """
    Convert a Conv2d weight tensor `w` of shape (OC, IC, KY, KX) into the 1D
    hardware ROM array ordered by (oc -> ic -> ky -> kx, kx fastest), with an
    explicit index verification against the address formula:
        addr(oc, ic, ky, kx) = ((oc * IC + ic) * KY + ky) * KX + kx
    """
    w = np.asarray(w, dtype=np.int8)
    if w.ndim != 4:
        raise ValueError(f"expected 4D conv weight tensor (OC, IC, KY, KX), got {w.shape}")
    oc_n, ic_n, ky_n, kx_n = w.shape
    flat = np.empty(oc_n * ic_n * ky_n * kx_n, dtype=np.int8)
    for oc in range(oc_n):
        for ic in range(ic_n):
            for ky in range(ky_n):
                for kx in range(kx_n):
                    addr = ((oc * ic_n + ic) * ky_n + ky) * kx_n + kx
                    flat[addr] = w[oc, ic, ky, kx]
    return flat


def unpack_conv_weights_oihw(flat: np.ndarray, shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Inverse of pack_conv_weights_oihw using the explicit hardware address formula."""
    flat = np.asarray(flat, dtype=np.int8)
    oc_n, ic_n, ky_n, kx_n = shape
    if flat.shape != (oc_n * ic_n * ky_n * kx_n,):
        raise ValueError(f"flat length {flat.shape} does not match {shape}")
    out = np.empty(shape, dtype=np.int8)
    for oc in range(oc_n):
        for ic in range(ic_n):
            for ky in range(ky_n):
                for kx in range(kx_n):
                    addr = ((oc * ic_n + ic) * ky_n + ky) * kx_n + kx
                    out[oc, ic, ky, kx] = flat[addr]
    return out


def pack_fc_weights_oi(w: np.ndarray) -> np.ndarray:
    """
    Convert an FC weight tensor `w` of shape (OUT_CLASSES, IN_CH) into the 1D
    hardware ROM array ordered by (out_class -> in_ch, in_ch fastest):
        addr(out_class, in_ch) = out_class * IN_CH + in_ch
    """
    w = np.asarray(w, dtype=np.int8)
    if w.ndim != 2:
        raise ValueError(f"expected 2D FC weight tensor (OUT, IN), got {w.shape}")
    out_n, in_n = w.shape
    flat = np.empty(out_n * in_n, dtype=np.int8)
    for o in range(out_n):
        for i in range(in_n):
            addr = o * in_n + i
            flat[addr] = w[o, i]
    return flat


def unpack_fc_weights_oi(flat: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of pack_fc_weights_oi."""
    flat = np.asarray(flat, dtype=np.int8)
    out_n, in_n = shape
    if flat.shape != (out_n * in_n,):
        raise ValueError(f"flat length {flat.shape} does not match {shape}")
    out = np.empty(shape, dtype=np.int8)
    for o in range(out_n):
        for i in range(in_n):
            out[o, i] = flat[o * in_n + i]
    return out


def pack_image_hw(img: np.ndarray) -> np.ndarray:
    """
    Convert a 2D image (H, W) or 3D image stack (N, H, W) into 1D row-major order:
        addr(row, col) = row * W + col
    """
    img = np.asarray(img, dtype=np.int8)
    if img.ndim == 2:
        h, w = img.shape
        flat = np.empty(h * w, dtype=np.int8)
        for r in range(h):
            for c in range(w):
                flat[r * w + c] = img[r, c]
        return flat
    if img.ndim == 3:
        return np.ascontiguousarray(img).reshape(-1)
    raise ValueError(f"expected 2D or 3D image array, got {img.shape}")


# ---------------------------------------------------------------------------
# .mem writer & reader (Verilog $readmemh compatible)
# ---------------------------------------------------------------------------
def write_mem(path: Path, values: np.ndarray, width_bits: int) -> None:
    """Write 1D signed integer array as a pure hex .mem file (one word per line)."""
    path = Path(path)
    arr = np.asarray(values).reshape(-1)
    lines = [int_to_twos_hex(int(v), width_bits) for v in arr]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def read_mem(path: Path, width_bits: int) -> np.ndarray:
    """Read a .mem file back into a 1D signed NumPy array (int8 or int32)."""
    path = Path(path)
    dtype = np.int8 if width_bits == 8 else np.int32
    lines = [ln.strip() for ln in path.read_text(encoding="ascii").splitlines()
             if ln.strip() and not ln.strip().startswith("//")]
    vals = [twos_hex_to_int(ln, width_bits) for ln in lines]
    return np.array(vals, dtype=dtype)


# ---------------------------------------------------------------------------
# .mif writer & reader (Intel Quartus Memory Initialization File)
# ---------------------------------------------------------------------------
def write_mif(path: Path, values: np.ndarray, width_bits: int, comment: str = "") -> None:
    """Write 1D signed integer array as an Intel Quartus .mif file."""
    path = Path(path)
    arr = np.asarray(values).reshape(-1)
    depth = int(arr.size)
    addr_hex_digits = max(1, int(np.ceil(np.log2(max(depth, 2)) / 4.0)))

    hdr = []
    if comment:
        hdr.append(f"-- {comment}")
    hdr += [
        f"DEPTH = {depth};",
        f"WIDTH = {width_bits};",
        "ADDRESS_RADIX = HEX;",
        "DATA_RADIX = HEX;",
        "CONTENT",
        "BEGIN",
    ]
    body = [
        f"    {addr:0{addr_hex_digits}X} : {int_to_twos_hex(int(v), width_bits)};"
        for addr, v in enumerate(arr)
    ]
    ftr = ["END;", ""]
    path.write_text("\n".join(hdr + body + ftr), encoding="ascii")


_MIF_DECL_RE = re.compile(r"^\s*(DEPTH|WIDTH|ADDRESS_RADIX|DATA_RADIX)\s*=\s*([^;]+);", re.IGNORECASE)
_MIF_ROW_RE = re.compile(r"^\s*([0-9A-Fa-f]+)\s*:\s*([0-9A-Fa-f]+)\s*;")


def read_mif(path: Path) -> Tuple[np.ndarray, dict]:
    """
    Parse and validate an Intel Quartus .mif file.
    Verifies the mandatory DEPTH, WIDTH, ADDRESS_RADIX, DATA_RADIX, CONTENT,
    BEGIN, and END; declarations, and returns (signed_array, metadata_dict).
    """
    path = Path(path)
    text = path.read_text(encoding="ascii")
    decls = {}
    in_content = False
    saw_begin = False
    saw_end = False
    rows = {}

    for raw_line in text.splitlines():
        line = raw_line.split("--", 1)[0].strip()
        if not line:
            continue
        if not in_content:
            m = _MIF_DECL_RE.match(line)
            if m:
                decls[m.group(1).upper()] = m.group(2).strip().upper()
                continue
            if line.upper() == "CONTENT":
                in_content = True
                continue
            raise ValueError(f"{path}: unexpected header line {raw_line!r}")
        else:
            if not saw_begin:
                if line.upper() == "BEGIN":
                    saw_begin = True
                    continue
                raise ValueError(f"{path}: expected BEGIN after CONTENT, got {raw_line!r}")
            if line.upper() == "END;":
                saw_end = True
                break
            m = _MIF_ROW_RE.match(line)
            if not m:
                raise ValueError(f"{path}: malformed CONTENT row {raw_line!r}")
            addr = int(m.group(1), 16)
            rows[addr] = m.group(2).upper()

    for req in ("DEPTH", "WIDTH", "ADDRESS_RADIX", "DATA_RADIX"):
        if req not in decls:
            raise ValueError(f"{path}: missing mandatory MIF declaration {req}")
    if not (in_content and saw_begin and saw_end):
        raise ValueError(f"{path}: incomplete CONTENT BEGIN ... END; block")
    if decls["ADDRESS_RADIX"] != "HEX" or decls["DATA_RADIX"] != "HEX":
        raise ValueError(f"{path}: expected HEX radices, got {decls}")

    depth = int(decls["DEPTH"])
    width = int(decls["WIDTH"])
    if len(rows) != depth or set(rows.keys()) != set(range(depth)):
        raise ValueError(f"{path}: expected {depth} contiguous addresses 0..{depth-1}, got {len(rows)}")

    dtype = np.int8 if width == 8 else np.int32
    arr = np.array([twos_hex_to_int(rows[a], width) for a in range(depth)], dtype=dtype)
    return arr, {"depth": depth, "width": width, **decls}
