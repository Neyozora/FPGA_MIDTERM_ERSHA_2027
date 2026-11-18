"""
Export Method 2 INT8 weights, biases, per-channel dyadic parameters,
and CIFAR-10 test vectors to Verilog $readmemh `.mem` files in `mem_m2/` and `sim_m2/vectors/`.
"""
import numpy as np

from sw_m2.config import (
    CHECKPOINTS_M2_DIR,
    PROCESSED_M2_DIR,
    MEM_M2_DIR,
    SIM_M2_VECTORS_DIR,
)
from sw_m2.train_qat import run_int8_inference


def write_hex_bytes(path, arr_i8: np.ndarray):
    flat = arr_i8.reshape(-1).astype(np.uint8)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v):02x}\n")


def pack_72b_kernel(k3x3: np.ndarray) -> int:
    """Pack 3x3 int8 kernel into 72-bit int: w00 in [7:0], ..., w22 in [71:64]."""
    val = 0
    flat = k3x3.reshape(-1).astype(np.uint8)
    for i, b in enumerate(flat):
        val |= int(b) << (8 * i)
    return val


def pack_53b_qparam(bias_i32: int, mult_u16: int, shift_u5: int) -> int:
    b = int(np.int32(bias_i32)) & 0xFFFFFFFF
    m = int(mult_u16) & 0xFFFF
    s = int(shift_u5) & 0x1F
    return (s << 48) | (m << 32) | b


def export_all():
    MEM_M2_DIR.mkdir(parents=True, exist_ok=True)
    SIM_M2_VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    npz = np.load(CHECKPOINTS_M2_DIR / "cifar_int8_params_m2.npz")
    params = {}
    for layer in ["conv1", "conv2", "conv3", "conv4", "fc"]:
        params[layer] = {
            "w_q": npz[f"{layer}_w_q"],
            "b_q": npz[f"{layer}_b_q"],
            "mult": npz[f"{layer}_mult"],
            "shift": npz[f"{layer}_shift"],
        }

    # 4 parallel weight banks (1676 x 72b) and 4 parallel qparam banks (44 x 53b)
    w_banks = [[] for _ in range(4)]
    q_banks = [[] for _ in range(4)]

    for layer in ["conv1", "conv2", "conv3", "conv4"]:
        w_q = params[layer]["w_q"]       # (oc, ic, 3, 3)
        b_q = params[layer]["b_q"]       # (oc,)
        mult = params[layer]["mult"]     # (oc,)
        shift = params[layer]["shift"]   # (oc,)
        oc, ic_cnt, _, _ = w_q.shape
        assert oc % 4 == 0
        for og in range(oc // 4):
            for k in range(4):
                c_out = 4 * og + k
                q_banks[k].append(pack_53b_qparam(b_q[c_out], mult[c_out], shift[c_out]))
            for ic in range(ic_cnt):
                for k in range(4):
                    c_out = 4 * og + k
                    w_banks[k].append(pack_72b_kernel(w_q[c_out, ic]))

    for k in range(4):
        assert len(w_banks[k]) == 1676, f"Expected 1676 words, got {len(w_banks[k])}"
        assert len(q_banks[k]) == 44, f"Expected 44 qparams, got {len(q_banks[k])}"
        with open(MEM_M2_DIR / f"w_bank{k}.mem", "w") as f:
            for val in w_banks[k]:
                f.write(f"{val:018x}\n")
        with open(MEM_M2_DIR / f"q_bank{k}.mem", "w") as f:
            for val in q_banks[k]:
                f.write(f"{val:014x}\n")

    # FC weights (640 x 8b) and FC qparams (10 x 53b)
    write_hex_bytes(MEM_M2_DIR / "fc_weights_m2.mem", params["fc"]["w_q"])
    with open(MEM_M2_DIR / "fc_qparams_m2.mem", "w") as f:
        for c in range(10):
            val = pack_53b_qparam(params["fc"]["b_q"][c], params["fc"]["mult"][c], params["fc"]["shift"][c])
            f.write(f"{val:014x}\n")

    # Export test vectors
    data = np.load(PROCESSED_M2_DIR / "cifar10_int8.npz")
    x_hw_q = data["x_hw_q"]  # (100, 3, 32, 32) int8
    y_hw = data["y_hw"]      # (100,)
    hw_scores = run_int8_inference(x_hw_q, params)  # (100, 10) int8

    # Single test image (class 0, idx 0)
    write_hex_bytes(MEM_M2_DIR / "test_image_m2.mem", x_hw_q[0])

    # 10-image DE10-Standard board ROM (1 correctly-classified image per class 0..9)
    hw_preds = hw_scores.argmax(axis=1)
    board_indices = []
    for c in range(10):
        c_slice = np.where((y_hw == c) & (hw_preds == c))[0]
        board_indices.append(int(c_slice[0]) if len(c_slice) > 0 else c * 10)
    board_imgs = x_hw_q[board_indices]  # (10, 3, 32, 32)
    write_hex_bytes(MEM_M2_DIR / "de10_images_10_m2.mem", board_imgs)

    # 100-image ModelSim verification vectors
    write_hex_bytes(SIM_M2_VECTORS_DIR / "hw_images_100.mem", x_hw_q)
    with open(SIM_M2_VECTORS_DIR / "hw_labels_100.mem", "w") as f:
        for lbl in y_hw:
            f.write(f"{int(lbl):01x}\n")
    write_hex_bytes(SIM_M2_VECTORS_DIR / "hw_expected_scores_100.mem", hw_scores)
    with open(SIM_M2_VECTORS_DIR / "hw_expected_preds_100.mem", "w") as f:
        for p in hw_scores.argmax(axis=1):
            f.write(f"{int(p):01x}\n")

    print(f"Exported Method 2 .mem files to {MEM_M2_DIR} and {SIM_M2_VECTORS_DIR}")
    print(f"100-image INT8 accuracy: {100.0 * (hw_scores.argmax(axis=1) == y_hw).mean():.2f}%")


if __name__ == "__main__":
    export_all()
