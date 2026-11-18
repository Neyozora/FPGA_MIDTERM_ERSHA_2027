"""Generate golden intermediate layer vectors for tb_cnn_top.v."""
from pathlib import Path
import numpy as np
import torch
from sw.export.formatters import read_mem, write_mem
from sw.quant.quantize import forward_int8_torch


def main() -> None:
    vec_dir = Path("sim/vectors")
    vec_dir.mkdir(parents=True, exist_ok=True)
    with np.load("checkpoints/int8_quantized.npz") as f:
        qz = {k: f[k] for k in f.files}

    img0 = read_mem(Path("mem/test_image.mem"), 8).reshape(1, 1, 32, 32)
    out = forward_int8_torch(torch.from_numpy(img0), qz)
    write_mem(vec_dir / "img0_relu1.mem", out["relu1"][0].numpy().ravel(), 8)
    write_mem(vec_dir / "img0_pool1.mem", out["pool1"][0].numpy().ravel(), 8)
    write_mem(vec_dir / "img0_relu2.mem", out["relu2"][0].numpy().ravel(), 8)
    write_mem(vec_dir / "img0_gap.mem", out["gap"][0].numpy().ravel(), 8)
    write_mem(vec_dir / "img0_fc.mem", out["fc"][0].numpy().ravel(), 8)
    pred0 = int(out["pred"][0])
    (vec_dir / "img0_pred.mem").write_text(f"{pred0:01X}\n", encoding="utf-8")
    print(f"Golden vectors written to sim/vectors/ (pred={pred0}, fc={out['fc'][0].tolist()})")


if __name__ == "__main__":
    main()
