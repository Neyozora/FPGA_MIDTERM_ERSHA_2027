"""
Method 2 4-Layer CNN for CIFAR-10 (32x32x3 RGB) with BatchNorm Folding
and Per-Channel Symmetric INT8 Quantization-Aware Training (QAT).

Architecture (all feature maps & weights fit in Cyclone V M10K BRAM):
  Input: (N, 3, 32, 32) INT8 [-128, 127], scale = 1/128
  Conv1: 3x3, 3  -> 16, pad=1 + BN + ReLU -> (N, 16, 32, 32)
  Pool1: MaxPool 2x2                       -> (N, 16, 16, 16)
  Conv2: 3x3, 16 -> 32, pad=1 + BN + ReLU -> (N, 32, 16, 16)
  Pool2: MaxPool 2x2                       -> (N, 32,  8,  8)
  Conv3: 3x3, 32 -> 64, pad=1 + BN + ReLU -> (N, 64,  8,  8)
  Conv4: 3x3, 64 -> 64, pad=1 + BN + ReLU -> (N, 64,  8,  8)
  GAP:   AvgPool 8x8 (exact >> 6 in HW!)   -> (N, 64)
  FC:    Linear 64 -> 10                   -> (N, 10)
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantPerChannelWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        # w: (oc, ...), scale: (oc, 1, ...)
        w_q = torch.clamp(torch.round(w / scale), -127, 127)
        return w_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FakeQuantActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin=0, qmax=127):
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class QConv2d(nn.Module):
    """Conv2d with BatchNorm-folded weights and per-channel symmetric INT8 fake-quant."""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=True)
        self.register_buffer("act_scale", torch.tensor(1.0 / 128.0))
        self.qat_enabled = False

    def forward(self, x):
        if self.qat_enabled:
            w = self.conv.weight
            w_max = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            w_scale = w_max / 127.0
            w_q = FakeQuantPerChannelWeight.apply(w, w_scale)
            y = F.conv2d(x, w_q, self.conv.bias, padding=self.conv.padding)
            y = F.relu(y)
            if self.training:
                with torch.no_grad():
                    cur_max = y.detach().amax().clamp(min=1e-5)
                    new_scale = cur_max / 127.0
                    self.act_scale.mul_(0.95).add_(0.05 * new_scale)
            return FakeQuantActivation.apply(y, self.act_scale, 0, 127)
        else:
            return F.relu(self.conv(x))


class QLinear(nn.Module):
    """Linear layer with per-channel symmetric INT8 fake-quant."""
    def __init__(self, in_f, out_f):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=True)
        self.register_buffer("act_scale", torch.tensor(1.0 / 16.0))
        self.qat_enabled = False

    def forward(self, x):
        if self.qat_enabled:
            w = self.fc.weight
            w_max = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            w_scale = w_max / 127.0
            w_q = FakeQuantPerChannelWeight.apply(w, w_scale)
            y = F.linear(x, w_q, self.fc.bias)
            if self.training:
                with torch.no_grad():
                    cur_max = y.detach().abs().amax().clamp(min=1e-5)
                    new_scale = cur_max / 127.0
                    self.act_scale.mul_(0.95).add_(0.05 * new_scale)
            return FakeQuantActivation.apply(y, self.act_scale, -128, 127)
        else:
            return self.fc(x)


class CifarCNNFloatBN(nn.Module):
    """Stage 1 Float32 model with BatchNorm2d for clean, high-accuracy training."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean(dim=(2, 3))  # 8x8 GAP
        return self.fc(x)


class CifarCNNQAT(nn.Module):
    """Stage 2 BN-Folded QAT model matching exact hardware topology."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = QConv2d(3, 16, 3, padding=1)
        self.conv2 = QConv2d(16, 32, 3, padding=1)
        self.conv3 = QConv2d(32, 64, 3, padding=1)
        self.conv4 = QConv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = QLinear(64, num_classes)

    def enable_qat(self, enabled=True):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]:
            m.qat_enabled = enabled

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.mean(dim=(2, 3))  # 8x8 GAP
        return self.fc(x)


def fold_bn_into_qat(float_model: CifarCNNFloatBN) -> CifarCNNQAT:
    """Fold BatchNorm2d parameters into Conv2d weights and biases."""
    qat_model = CifarCNNQAT(num_classes=float_model.fc.out_features)

    pairs = [
        (float_model.conv1, float_model.bn1, qat_model.conv1.conv),
        (float_model.conv2, float_model.bn2, qat_model.conv2.conv),
        (float_model.conv3, float_model.bn3, qat_model.conv3.conv),
        (float_model.conv4, float_model.bn4, qat_model.conv4.conv),
    ]
    with torch.no_grad():
        for conv, bn, dst_conv in pairs:
            std = torch.sqrt(bn.running_var + bn.eps)
            scale = bn.weight / std
            dst_conv.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
            dst_conv.bias.copy_(bn.bias - bn.running_mean * scale)

        qat_model.fc.fc.weight.copy_(float_model.fc.weight)
        qat_model.fc.fc.bias.copy_(float_model.fc.bias)

    return qat_model
