"""
Train Method 2 4-Layer CNN on CIFAR-10 (32x32x3 RGB):
  1. Float32 + BatchNorm2d training (80 epochs, RandomCrop(32,4) + RandomHorizontalFlip)
  2. BatchNorm folding -> Per-Channel Symmetric INT8 QAT fine-tuning (25 epochs)
  3. Exact Dyadic Integer Conversion & 10,000-image INT8 Test Evaluation
"""
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sw_m2.config import (
    PROCESSED_M2_DIR,
    CHECKPOINTS_M2_DIR,
    REPORTS_M2_DIR,
    INPUT_SCALE,
    SEED,
)
from sw_m2.model import CifarCNNFloatBN, fold_bn_into_qat


class CifarInt8Dataset(Dataset):
    def __init__(self, x_q, y, augment=False):
        self.x = torch.from_numpy(x_q).float() * INPUT_SCALE  # (N, 3, 32, 32)
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.augment:
            # RandomCrop(32, padding=4)
            padded = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)
            i = torch.randint(0, 9, (1,)).item()
            j = torch.randint(0, 9, (1,)).item()
            img = padded[:, i : i + 32, j : j + 32]
            # RandomHorizontalFlip(p=0.5)
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
        return img, self.y[idx]


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def approx_dyadic(real_scale: float, max_shift: int = 24):
    """Find (M, shift) with M in [1, 65535], shift in [1, max_shift] minimizing |real_scale - M/2^shift|."""
    best_err = float("inf")
    best_m, best_s = 1, 16
    for s in range(8, max_shift + 1):
        m = int(round(real_scale * (1 << s)))
        if 1 <= m <= 65535:
            err = abs(real_scale - m / (1 << s))
            if err < best_err:
                best_err = err
                best_m, best_s = m, s
    return best_m, best_s


def extract_integer_params(qat_model):
    """Convert QAT model into exact integer-only weights, biases, and per-channel dyadic (mult, shift)."""
    qat_model.eval()
    params = {}
    s_prev = INPUT_SCALE

    for name in ["conv1", "conv2", "conv3", "conv4"]:
        layer = getattr(qat_model, name)
        w = layer.conv.weight.detach().cpu().numpy()  # (oc, ic, 3, 3)
        b = layer.conv.bias.detach().cpu().numpy()    # (oc,)
        s_out = float(layer.act_scale.item())

        oc = w.shape[0]
        w_max = np.maximum(np.abs(w).max(axis=(1, 2, 3)), 1e-8)
        w_scale = w_max / 127.0  # (oc,)
        w_q = np.clip(np.round(w / w_scale[:, None, None, None]), -127, 127).astype(np.int8)

        acc_scale = s_prev * w_scale  # (oc,)
        b_q = np.round(b / acc_scale).astype(np.int32)

        mult = np.zeros(oc, dtype=np.uint16)
        shift = np.zeros(oc, dtype=np.uint8)
        for c in range(oc):
            m, s = approx_dyadic(float(acc_scale[c] / s_out))
            mult[c] = m
            shift[c] = s

        params[name] = {
            "w_q": w_q,
            "b_q": b_q,
            "mult": mult,
            "shift": shift,
            "s_out": s_out,
        }
        s_prev = s_out

    # FC layer
    fc = qat_model.fc
    w = fc.fc.weight.detach().cpu().numpy()  # (10, 64)
    b = fc.fc.bias.detach().cpu().numpy()    # (10,)
    s_out = float(fc.act_scale.item())

    oc = w.shape[0]
    w_max = np.maximum(np.abs(w).max(axis=1), 1e-8)
    w_scale = w_max / 127.0
    w_q = np.clip(np.round(w / w_scale[:, None]), -127, 127).astype(np.int8)
    acc_scale = s_prev * w_scale
    b_q = np.round(b / acc_scale).astype(np.int32)

    mult = np.zeros(oc, dtype=np.uint16)
    shift = np.zeros(oc, dtype=np.uint8)
    for c in range(oc):
        m, s = approx_dyadic(float(acc_scale[c] / s_out))
        mult[c] = m
        shift[c] = s

    params["fc"] = {
        "w_q": w_q,
        "b_q": b_q,
        "mult": mult,
        "shift": shift,
        "s_out": s_out,
    }
    return params


def dyadic_requant_vec(acc: np.ndarray, mult: np.ndarray, shift: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    """Vectorized hardware-exact dyadic requantization along axis 1 (channels)."""
    # acc: (N, C, ...), mult/shift: (C,)
    shape = [1] * acc.ndim
    shape[1] = -1
    m = mult.astype(np.int64).reshape(shape)
    s = shift.astype(np.int64).reshape(shape)
    prod = acc.astype(np.int64) * m
    rnd = np.left_shift(np.int64(1), s - 1)
    res = (prod + rnd) >> s
    return np.clip(res, qmin, qmax).astype(np.int8)


def int8_conv2d(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray, mult: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Exact INT8 conv2d (pad=1, 3x3) using PyTorch int64-safe im2col/conv2d on CPU/GPU or numpy."""
    # Use torch on CPU/GPU with float64 (exact up to 2^53; max acc is 64*9*127*127 ~ 9.3e6 << 2^53!)
    xt = torch.from_numpy(x_q.astype(np.float64))
    wt = torch.from_numpy(w_q.astype(np.float64))
    acc = F.conv2d(xt, wt, padding=1).numpy().astype(np.int64)
    acc += b_q.astype(np.int64)[None, :, None, None]
    return dyadic_requant_vec(acc, mult, shift, 0, 127)


def int8_maxpool2x2(x_q: np.ndarray) -> np.ndarray:
    N, C, H, W = x_q.shape
    return x_q.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def int8_gap8x8(x_q: np.ndarray) -> np.ndarray:
    # Exact hardware 8x8 GAP: (sum_of_64 + 32) >> 6, clamped to [0, 127]
    s = x_q.astype(np.int32).sum(axis=(2, 3))
    return np.clip((s + 32) >> 6, 0, 127).astype(np.int8)


def int8_fc(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray, mult: np.ndarray, shift: np.ndarray) -> np.ndarray:
    acc = x_q.astype(np.int64) @ w_q.astype(np.int64).T + b_q.astype(np.int64)[None, :]
    return dyadic_requant_vec(acc, mult, shift, -128, 127)


def run_int8_inference(x_q: np.ndarray, params: dict) -> np.ndarray:
    x = int8_conv2d(x_q, params["conv1"]["w_q"], params["conv1"]["b_q"], params["conv1"]["mult"], params["conv1"]["shift"])
    x = int8_maxpool2x2(x)
    x = int8_conv2d(x, params["conv2"]["w_q"], params["conv2"]["b_q"], params["conv2"]["mult"], params["conv2"]["shift"])
    x = int8_maxpool2x2(x)
    x = int8_conv2d(x, params["conv3"]["w_q"], params["conv3"]["b_q"], params["conv3"]["mult"], params["conv3"]["shift"])
    x = int8_conv2d(x, params["conv4"]["w_q"], params["conv4"]["b_q"], params["conv4"]["mult"], params["conv4"]["shift"])
    x = int8_gap8x8(x)
    scores = int8_fc(x, params["fc"]["w_q"], params["fc"]["b_q"], params["fc"]["mult"], params["fc"]["shift"])
    return scores


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Method 2 CIFAR-10 CNN on device: {device}")

    CHECKPOINTS_M2_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_M2_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_M2_DIR / "cifar10_int8.npz")
    train_ds = CifarInt8Dataset(data["x_train_q"], data["y_train"], augment=True)
    val_ds = CifarInt8Dataset(data["x_val_q"], data["y_val"], augment=False)
    test_ds = CifarInt8Dataset(data["x_test_q"], data["y_test"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Stage 1: Float32 + BN
    t0 = time.time()
    model = CifarCNNFloatBN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = 0.0, None
    for epoch in range(1, 81):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        if epoch % 10 == 0 or epoch >= 70:
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 10 == 0:
                print(f"[Float+BN Epoch {epoch:02d}/80] Val Acc: {val_acc:.2f}% (Best: {best_val:.2f}%)")

    model.load_state_dict(best_state)
    float_test_acc = evaluate(model, test_loader, device)
    print(f"Stage 1 Float+BN Test Acc: {float_test_acc:.2f}% ({time.time()-t0:.1f}s)")

    # Stage 2: Fold BN + Per-Channel INT8 QAT
    qat_model = fold_bn_into_qat(model.cpu()).to(device)
    qat_model.eval()
    # Calibrate act_scale
    qat_model.enable_qat(True)
    qat_model.train()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            qat_model(x.to(device))
            if i >= 15:
                break

    opt_qat = torch.optim.Adam(qat_model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched_qat = torch.optim.lr_scheduler.CosineAnnealingLR(opt_qat, T_max=25)
    best_qat_val, best_qat_state = 0.0, None
    for epoch in range(1, 26):
        qat_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_qat.zero_grad()
            loss = criterion(qat_model(x), y)
            loss.backward()
            opt_qat.step()
        sched_qat.step()
        val_acc = evaluate(qat_model, val_loader, device)
        if val_acc >= best_qat_val:
            best_qat_val = val_acc
            best_qat_state = copy.deepcopy(qat_model.state_dict())
        if epoch % 5 == 0:
            print(f"[QAT Epoch {epoch:02d}/25] Val Acc: {val_acc:.2f}% (Best: {best_qat_val:.2f}%)")

    qat_model.load_state_dict(best_qat_state)
    qat_test_acc = evaluate(qat_model, test_loader, device)
    print(f"Stage 2 QAT Fake-Quant Test Acc: {qat_test_acc:.2f}%")

    torch.save(qat_model.state_dict(), CHECKPOINTS_M2_DIR / "cifar_qat_m2.pt")

    # Stage 3: Pure Integer Dyadic Evaluation
    int_params = extract_integer_params(qat_model)
    np.savez(
        CHECKPOINTS_M2_DIR / "cifar_int8_params_m2.npz",
        **{f"{k}_{sub}": v for k, d in int_params.items() for sub, v in d.items() if isinstance(v, np.ndarray)},
    )

    # Evaluate in batches of 1000
    x_test_q = data["x_test_q"]
    y_test = data["y_test"]
    preds = []
    for i in range(0, len(y_test), 1000):
        sc = run_int8_inference(x_test_q[i : i + 1000], int_params)
        preds.append(sc.argmax(axis=1))
    preds = np.concatenate(preds)
    int8_test_acc = 100.0 * (preds == y_test).mean()

    hw_scores = run_int8_inference(data["x_hw_q"], int_params)
    hw_acc = 100.0 * (hw_scores.argmax(axis=1) == data["y_hw"]).mean()

    print(f"Stage 3 Pure Integer INT8 Test Acc: {int8_test_acc:.2f}% | 100-img HW Vector Acc: {hw_acc:.2f}%")

    metrics = {
        "float_bn_test_acc": round(float(float_test_acc), 2),
        "qat_fakequant_test_acc": round(float(qat_test_acc), 2),
        "int8_dyadic_test_acc": round(float(int8_test_acc), 2),
        "hw_100_acc": round(float(hw_acc), 2),
        "total_params": int(sum(p.numel() for p in qat_model.parameters())),
    }
    (REPORTS_M2_DIR / "training_quant_summary.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
