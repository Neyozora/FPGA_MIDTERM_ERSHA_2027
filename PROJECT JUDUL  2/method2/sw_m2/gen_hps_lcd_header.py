"""
Generate `hps_m2/cifar10_lcd_data.h` with 10 verified CIFAR-10 images
(4x4 Bayer-dithered to 1-bit 32x32 bitmaps for the 128x64 DE10-Standard LCD)
and their exact Method 2 INT8 class scores.
"""
import numpy as np
from sw_m2.config import CLASS_NAMES, PROCESSED_M2_DIR, REPO_ROOT
from sw_m2.gui_app_m2 import load_m2_params
from sw_m2.train_qat import run_int8_inference

BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float32) / 16.0


def main():
    hps_dir = REPO_ROOT / "hps_m2"
    hps_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(PROCESSED_M2_DIR / "cifar10_int8.npz")
    x_hw_q = d["x_hw_q"]
    y_hw = d["y_hw"]
    params = load_m2_params()
    scores = run_int8_inference(x_hw_q, params)
    preds = scores.argmax(axis=1)

    chosen = []
    for c in range(10):
        hits = np.where((y_hw == c) & (preds == c))[0]
        chosen.append(int(hits[0]) if len(hits) > 0 else c * 10)

    lines = [
        "#ifndef CIFAR10_LCD_DATA_H",
        "#define CIFAR10_LCD_DATA_H",
        "#include <stdint.h>",
        "",
        "static const char *CIFAR10_NAMES[10] = {",
    ]
    for name in CLASS_NAMES:
        lines.append(f'    "{name.upper()}",')
    lines.append("};")
    lines.append("")
    lines.append("// 10 x 32 rows of 32-bit MSB-first 1-bit dithered image rows")
    lines.append("static const uint32_t CIFAR10_BITMAPS[10][32] = {")

    for c, idx in enumerate(chosen):
        rgb = (x_hw_q[idx].astype(np.float32) + 128.0) / 255.0  # (3, 32, 32)
        gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        lines.append(f"    {{ // Class {c}: {CLASS_NAMES[c]}")
        row_strs = []
        for r in range(32):
            word = 0
            for col in range(32):
                thresh = BAYER_4X4[r % 4, col % 4]
                if gray[r, col] > thresh:
                    word |= 1 << (31 - col)
            row_strs.append(f"0x{word:08X}U")
        for i in range(0, 32, 4):
            lines.append("        " + ", ".join(row_strs[i : i + 4]) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append("")
    lines.append("static const int8_t CIFAR10_SCORES[10][10] = {")
    for c, idx in enumerate(chosen):
        sc = ", ".join(f"{int(v):4d}" for v in scores[idx])
        lines.append(f"    {{ {sc} }}, // {CLASS_NAMES[c]}")
    lines.append("};")
    lines.append("")
    lines.append("#endif")

    out_path = hps_dir / "cifar10_lcd_data.h"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
