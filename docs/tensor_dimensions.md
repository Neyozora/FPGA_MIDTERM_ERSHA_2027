# CNN Tensor Dimensions

This file is the reference for every tensor shape in the network. The PyTorch
model (Milestone 2) asserts each shape, and the RTL (Milestone 7+) is sized
from these numbers.

## Output-size formula

For a convolution or pooling window with kernel `K`, stride `S` and padding `P`:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1
$$

| Stage | Formula | Result |
|---|---|---|
| Conv1 (K=3, S=1, P=0) | (28 + 0 − 3)/1 + 1 | **26** |
| MaxPool (K=2, S=2) | ⌊(26 − 2)/2⌋ + 1 | **13** (26 is even, so no row/column is dropped) |
| Conv2 (K=3, S=1, P=0) | (13 + 0 − 3)/1 + 1 | **11** |
| GAP | average over 11 × 11 | **121** values per channel → 1 value |

## Layer-by-layer table

PyTorch uses `N × C × H × W` (NCHW) with batch size `N`. The table shows one
image (N = 1). Hardware memory layouts are defined separately in Milestone 5,
and are **not** assumed to be the same as PyTorch's.

| # | Stage | Input shape (C×H×W) | Output shape (C×H×W) | Output elements | Weights | Biases | MACs |
|---|---|---|---|---|---|---|---|
| 0 | Input (INT8, q = p >> 1) | – | 1×28×28 | 784 | – | – | – |
| 1 | Conv1 3×3, 1→8, VALID | 1×28×28 | 8×26×26 | 5,408 | 8·1·3·3 = 72 | 8 | 26·26·8·9 = 48,672 |
| 2 | ReLU | 8×26×26 | 8×26×26 | 5,408 | – | – | – |
| 3 | MaxPool 2×2 / 2 | 8×26×26 | 8×13×13 | 1,352 | – | – | (3 compares per output) |
| 4 | Conv2 3×3, 8→16, VALID | 8×13×13 | 16×11×11 | 1,936 | 16·8·3·3 = 1,152 | 16 | 11·11·16·72 = 139,392 |
| 5 | ReLU | 16×11×11 | 16×11×11 | 1,936 | – | – | – |
| 6 | Global Average Pool | 16×11×11 | 16 | 16 | – | – | (120 additions per channel) |
| 7 | Fully Connected 16→10 | 16 | 10 | 10 | 10·16 = 160 | 10 | 160 |
| 8 | Argmax | 10 | 1 (class index 0..9) | 1 | – | – | – |

**Totals**

| Quantity | Value |
|---|---|
| Parameters | 72 + 8 + 1,152 + 16 + 160 + 10 = **1,418** |
| Weights (INT8) | 1,384 → 1,384 bytes |
| Biases (INT32) | 34 → 136 bytes |
| MACs per image | 48,672 + 139,392 + 160 = **188,224** |
| Share of MACs in Conv2 | 74 % |

## Hardware-oriented observations

### On-chip storage (rough, before any optimization)

| Buffer | Elements × width | Bits | Notes |
|---|---|---|---|
| Input image | 784 × 8 | 6,272 | fits in 1 M10K (10,240 bits) |
| Conv1 output (pre-pool) | 5,408 × 8 | 43,264 | ~5 M10K; can be avoided by fusing ReLU+pool into conv1's output stage |
| Pool output / Conv2 input | 1,352 × 8 | 10,816 | ~2 M10K |
| Conv2 output | 1,936 × 8 | 15,488 | ~2 M10K; can be avoided by accumulating GAP sums on the fly |
| Conv1 + Conv2 + FC weights | 1,384 × 8 | 11,072 | ~2 M10K (or one ROM per layer) |
| Biases | 34 × 32 | 1,088 | registers or 1 M10K |

All of these are small next to the M10K blocks on the Cyclone V SoC (576 M10Ks on the DE10-Standard `5CSXFC6D6F31C6N` / 553 M10Ks on `5CSEBA6U23I7`, each 10,240 bits). The exact block counts will come from the Quartus fitter report (Milestone 10). They are estimates here.

### Cyclone V / DE10-Standard architectural optimizations

To optimize specifically for the Intel Cyclone V FPGA fabric on the DE10-Standard:

1. **Zero-hardware input quantization (`q_in = pixel >> 1`)**: Maps `[0, 255]` uint8 pixels into non-negative signed INT8 `[0, 127]` (`s_in = 2^-7`, zero-point `0`). In hardware, `>> 1` is pure wire-slicing (0 ALMs, 0 DSPs), and zero-point `0` eliminates cross-term offset corrections in Conv1 MACs.
2. **Cyclone V Variable-Precision DSP mapping**: Each Cyclone V DSP block supports either **three 9×9 signed multipliers**, **two 18×18 multipliers with accumulator**, or **one 27×27 multiplier**.
   - Signed INT8 × INT8 multiplication + INT32 accumulation maps directly into a single DSP block with internal systolic/accumulator registers, avoiding fabric carry-chain routing.
   - Requantization (`acc * M0 >>> n`) uses an **18-bit unsigned multiplier `M0`** (rather than 32-bit), so `int24_acc * uint18_M0` fits inside a single Cyclone V DSP block instead of cascading multiple DSPs.
3. **Synchronous M10K BRAM inference**: All weight ROMs and feature-map RAMs use registered read addresses and single-clock synchronous read templates (`/* synthesis ramstyle = "M10K" */`) so Quartus infers dedicated M10K blocks rather than MLAB/LUT RAM.
4. **Fused datapath options**: Because `max(ReLU(a), ReLU(b)) == ReLU(max(a, b))` and GAP only sums the `11 × 11` spatial outputs per channel, the RTL can optionally stream Conv2 outputs directly into the GAP accumulator while keeping intermediate checkpoints accessible for verification.

### Accumulator range (worst case, for the INT32 choice)

Products are INT8 × INT8. The largest magnitude is |−128 × −128| = 16,384. For
non-negative activations (inputs and post-ReLU values lie in 0..127) the bound
is 127 × 128 = 16,256.

| Layer | Products per output | Worst-case \|sum\| (excluding bias) | Bits needed (signed) |
|---|---|---|---|
| Conv1 | 9 | 9 × 16,256 = 146,304 | 19 |
| Conv2 | 72 | 72 × 16,256 = 1,170,432 | 22 |
| FC | 16 | 16 × 16,384 = 262,144 | 20 |
| GAP sum | 121 values in 0..127 | 121 × 127 = 15,367 | 15 (unsigned 14) |

INT32 accumulation therefore has at least 10 bits of headroom, even after the
bias is added. Overflow cannot happen.

### Latency estimate (analytical only, NOT a measurement)

With one sequential MAC per clock cycle and no overlap between layers:

$$
\text{cycles} \approx 188{,}224 \; (\text{MAC}) + \text{control/pool/GAP overhead}
$$

At 50 MHz (the DE10-Standard on-board oscillator) this is about **3.8 ms** plus
overhead. The real cycle count will be measured in ModelSim (Milestone 9), and
the real Fmax will come from Quartus TimeQuest (Milestone 10).

## Planned numeric format at each stage (finalized in Milestone 4)

```text
input     int8  [0,127]     scale s_in = 2^-7, zero point 0
conv1     int8 x int8 -> int32 acc  + int32 bias  -> requant -> int8 -> ReLU -> [0,127]
maxpool   int8 -> int8 (signed compare)
conv2     int8 x int8 -> int32 acc  + int32 bias  -> requant -> int8 -> ReLU -> [0,127]
gap       sum of 121 int8 -> int32 -> divide by 121 (fixed-point, documented rounding) -> int8
fc        int8 x int8 -> int32 acc  + int32 bias  -> requant -> int8 scores
argmax    index of max score (tie-break rule to be defined)
```
