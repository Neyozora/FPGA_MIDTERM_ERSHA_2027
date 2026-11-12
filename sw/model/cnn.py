"""
Two-layer CNN for Fashion-MNIST (32x32 zero-padded input), kept deliberately
simple so that every operation maps 1:1 onto an RTL module.

    input  [N, 1, 32, 32]   x = q_in / 128  (2-px zero-padded 28x28 -> 32x32)
    conv1  [N, 8, 30, 30]   3x3, 1->8,  stride 1, VALID, bias
    relu1  [N, 8, 30, 30]
    pool1  [N, 8, 15, 15]   max 2x2, stride 2
    conv2  [N,16, 13, 13]   3x3, 8->16, stride 1, VALID, bias
    relu2  [N,16, 13, 13]
    gap    [N,16]           mean over the 13x13 = 169 spatial positions
    fc     [N,10]           16->10, bias      (class scores / logits)
    argmax [N]              predicted class (applied outside the module)

Every intermediate shape is checked against SHAPES at runtime. The checks
raise ShapeError (not `assert`, which `python -O` would strip).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sw.config import IMG_C, IMG_H, IMG_W, NUM_CLASSES

# ---------------------------------------------------------------------------
# Architecture constants. The RTL parameters in Milestone 7 mirror these.
# ---------------------------------------------------------------------------
KERNEL = 3
STRIDE = 1
PADDING = 0
POOL_K = 2
POOL_S = 2
CONV1_IN, CONV1_OUT = IMG_C, 8
CONV2_IN, CONV2_OUT = CONV1_OUT, 16
FC_IN, FC_OUT = CONV2_OUT, NUM_CLASSES


def out_size(n: int, k: int, s: int = 1, p: int = 0) -> int:
    """Output length of a conv/pool window: floor((n + 2p - k) / s) + 1."""
    return (n + 2 * p - k) // s + 1


C1_H, C1_W = out_size(IMG_H, KERNEL, STRIDE, PADDING), out_size(IMG_W, KERNEL, STRIDE, PADDING)
P1_H, P1_W = out_size(C1_H, POOL_K, POOL_S), out_size(C1_W, POOL_K, POOL_S)
C2_H, C2_W = out_size(P1_H, KERNEL, STRIDE, PADDING), out_size(P1_W, KERNEL, STRIDE, PADDING)
GAP_N = C2_H * C2_W  # number of values averaged per channel

# Expected per-sample shapes (batch dimension excluded), in forward order.
SHAPES = OrderedDict([
    ("input", (CONV1_IN, IMG_H, IMG_W)),     # (1, 32, 32)
    ("conv1", (CONV1_OUT, C1_H, C1_W)),      # (8, 30, 30)
    ("relu1", (CONV1_OUT, C1_H, C1_W)),      # (8, 30, 30)
    ("pool1", (CONV1_OUT, P1_H, P1_W)),      # (8, 15, 15)
    ("conv2", (CONV2_OUT, C2_H, C2_W)),      # (16, 13, 13)
    ("relu2", (CONV2_OUT, C2_H, C2_W)),      # (16, 13, 13)
    ("gap", (FC_IN,)),                       # (16,)
    ("fc", (FC_OUT,)),                       # (10,)
])

# Hard-coded expectations from the project specification: a change to any
# constant above that breaks the spec fails loudly at import time.
assert (C1_H, P1_H, C2_H, GAP_N) == (30, 15, 13, 169)
assert C1_H % POOL_S == 0, "max-pool would silently drop a row/column"


class ShapeError(RuntimeError):
    pass


class FashionCNN(nn.Module):
    def __init__(self, check_shapes: bool = True):
        super().__init__()
        self.check_shapes = check_shapes
        self.conv1 = nn.Conv2d(CONV1_IN, CONV1_OUT, KERNEL, stride=STRIDE, padding=PADDING, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_K, stride=POOL_S)
        self.conv2 = nn.Conv2d(CONV2_IN, CONV2_OUT, KERNEL, stride=STRIDE, padding=PADDING, bias=True)
        self.fc = nn.Linear(FC_IN, FC_OUT, bias=True)

    def _check(self, name: str, t: torch.Tensor) -> None:
        if not self.check_shapes:
            return
        expected = SHAPES[name]
        if t.dim() != len(expected) + 1 or tuple(t.shape[1:]) != expected:
            raise ShapeError(f"{name}: expected [N, {', '.join(map(str, expected))}], "
                             f"got {list(t.shape)}")

    def forward_intermediates(self, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        """Run the network and return every intermediate tensor (for debugging/verification)."""
        t = OrderedDict()
        self._check("input", x); t["input"] = x
        x = self.conv1(x);       self._check("conv1", x); t["conv1"] = x
        x = F.relu(x);           self._check("relu1", x); t["relu1"] = x
        x = self.pool1(x);       self._check("pool1", x); t["pool1"] = x
        x = self.conv2(x);       self._check("conv2", x); t["conv2"] = x
        x = F.relu(x);           self._check("relu2", x); t["relu2"] = x
        x = x.mean(dim=(2, 3));  self._check("gap", x);   t["gap"] = x
        x = self.fc(x);          self._check("fc", x);    t["fc"] = x
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check("input", x)
        x = F.relu(self.conv1(x))
        self._check("relu1", x)
        x = self.pool1(x)
        self._check("pool1", x)
        x = F.relu(self.conv2(x))
        self._check("relu2", x)
        x = x.mean(dim=(2, 3))
        self._check("gap", x)
        x = self.fc(x)
        self._check("fc", x)
        return x

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Argmax over the 10 class scores. torch.argmax returns the lowest index on ties."""
        return self.forward(x).argmax(dim=1)


# ---------------------------------------------------------------------------
# Analytical bookkeeping (independent of PyTorch's own counting)
# ---------------------------------------------------------------------------
def expected_param_counts() -> "OrderedDict[str, int]":
    return OrderedDict([
        ("conv1.weight", CONV1_OUT * CONV1_IN * KERNEL * KERNEL),  # 72
        ("conv1.bias", CONV1_OUT),                                 # 8
        ("conv2.weight", CONV2_OUT * CONV2_IN * KERNEL * KERNEL),  # 1152
        ("conv2.bias", CONV2_OUT),                                 # 16
        ("fc.weight", FC_OUT * FC_IN),                             # 160
        ("fc.bias", FC_OUT),                                       # 10
    ])


def expected_macs() -> "OrderedDict[str, int]":
    return OrderedDict([
        ("conv1", CONV1_OUT * C1_H * C1_W * CONV1_IN * KERNEL * KERNEL),  # 64,800
        ("conv2", CONV2_OUT * C2_H * C2_W * CONV2_IN * KERNEL * KERNEL),  # 194,688
        ("fc", FC_OUT * FC_IN),                                           # 160
    ])
