# Fixed-Point and INT8 Quantization Specification (Milestone 4)

This document defines the exact arithmetic shared by the PyTorch quantization
pipeline (`sw/quant/`), the exported `.mem`/`.mif` parameter files (`sw/export/`),
the NumPy integer reference model (`sw/refmodel/`), and the Cyclone V Verilog
RTL (`rtl/`).

> [!IMPORTANT]
> Python and Verilog **never** quantize independently or use different rounding
> conventions. Once quantized, the model is stored as exact integer arrays
> (`q_w`, `q_b`, `M`, `shift`) in `checkpoints/int8_quantized.npz`. Both the
> Python evaluation and the RTL use those identical integers and the identical
> shift-add-saturate equations below.

---

## 1. Number Formats

| Quantity | Format | Range | Zero Point | Notes |
|---|---|---|---|---|
| Input activation `q_in` | Signed `INT8` (`int8_t`) | `[0, 127]` | `0` | `q_in = pixel >> 1`, `s_in = 2^-7 = 1/128` |
| Layer weights `q_w` | Signed `INT8` (`int8_t`) | `[-127, 127]` | `0` | Symmetric (`-128` avoided to prevent `-128 * -128` asymmetry) |
| Post-ReLU activations (`q_a1`, `q_a2`, `q_gap`) | Signed `INT8` (`int8_t`) | `[0, 127]` | `0` | Non-negative because of ReLU / average of non-negatives |
| FC output scores `q_fc` | Signed `INT8` (`int8_t`) | `[-128, 127]` | `0` | Signed class scores fed into `argmax` |
| Biases `q_b` | Signed `INT32` (`int32_t`) | `[-2^31, 2^31-1]` | `0` | Scale `s_b = s_in_layer * s_w_layer` |
| MAC accumulator `acc` | Signed `INT32` (`int32_t`) | `[-2^31, 2^31-1]` | `0` | Max magnitude before bias is `1,170,432` (22 bits in Conv2) |
| Requant multiplier `M` | Unsigned `UINT16` | `[1, 65535]` | – | Fits in a single Cyclone V 18×18 DSP multiplier input |
| Right-shift `n` | Unsigned `UINT8` | `[1, 31]` | – | Arithmetic right shift after rounding offset |

---

## 2. Quantization Formulas (Float → Integer Calibration)

### 2.1 Per-Layer Symmetric Weight Quantization
For a layer with floating-point weight tensor $W$:

$$
s_w = \frac{\max(|W|)}{127}, \qquad q_w = \text{clip}\!\left(\left\lfloor \frac{W}{s_w} + 0.5 \right\rfloor, -127, +127\right) \in \text{INT8}
$$

Using $\pm 127$ (instead of $-128$) keeps the weight range strictly symmetric around zero.

### 2.2 Explicit Bias Quantization
For a layer with input activation scale $s_{in}$ and weight scale $s_w$, the accumulator scale is:

$$
s_b = s_{in} \cdot s_w, \qquad q_b = \text{clip}\!\left(\left\lfloor \frac{b}{s_b} + 0.5 \right\rfloor, -2^{30}, +2^{30}-1\right) \in \text{INT32}
$$

### 2.3 Output Activation Scale & Dyadic Requantization Multiplier $(M, n)$
In exact real arithmetic, an accumulator value $\text{acc} = \sum q_x q_w + q_b$ represents the real value $\text{acc} \cdot (s_{in} s_w)$. To map this to an output integer with scale $s_{out}$:

$$
q_{out} \approx \text{acc} \cdot \underbrace{\frac{s_{in} \cdot s_w}{s_{out}}}_{m_{real}}
$$

We approximate the positive real ratio $m_{real} = \frac{s_{in} s_w}{s_{out}}$ by a dyadic rational with a 16-bit unsigned integer multiplier $M \in [2^{14}, 2^{15}-1]$ (`[16384, 32767]`) and right-shift $n$:

$$
m_{real} \approx \frac{M}{2^n}, \qquad M = \text{clip}\!\left(\left\lfloor m_{real} \cdot 2^n + 0.5 \right\rfloor, 1, 65535\right)
$$

Because $M < 2^{16}$ and $|\text{acc}| < 2^{22}$ across all layers, the product $\text{acc} \cdot M$ is bounded by $< 2^{38}$ (fits cleanly in a signed 48-bit or 64-bit intermediate wire and maps onto Cyclone V variable-precision DSP blocks).

---

## 3. Exact Hardware Integer Arithmetic (Layer by Layer)

### 3.1 Rounding and Saturation Primitives
Every requantization stage in both Python and Verilog uses **Round-Half-Up** (`+(1 << (n - 1))` before arithmetic right shift `>>> n`), followed by saturation:

$$
\text{requant}(\text{acc}, M, n) = \left(\text{ int64}(\text{acc}) \times \text{int64}(M) + 2^{n-1} \right) \gg n
$$

- **Signed INT8 + ReLU saturation (`Conv1`, `Conv2`):**
  $$
  \text{sat\_relu\_int8}(v) = \begin{cases} 0 & \text{if } v < 0 \\ 127 & \text{if } v > 127 \\ v & \text{otherwise} \end{cases}
  $$
- **Signed INT8 saturation (`FC`):**
  $$
  \text{sat\_int8}(v) = \begin{cases} -128 & \text{if } v < -128 \\ 127 & \text{if } v > 127 \\ v & \text{otherwise} \end{cases}
  $$

> [!NOTE]
> In two's-complement arithmetic, `(prod + (1 << (n - 1))) >>> n` equals $\lfloor \text{prod} / 2^n + 0.5 \rfloor$ for both positive and negative values of `prod`. This requires only a single adder and a wire-select barrel shift in Verilog RTL.

---

### 3.2 Stage-by-Stage Equations

| Stage | Exact Integer Equation | Input / Output Range |
|---|---|---|
| **0. Input** | $q_{in}[y, x] = \text{pixel}[y, x] \gg 1$ (with 2-px zero border) | `uint8 [0,255]` $\to$ `int8 [0,127]` |
| **1. Conv1 + ReLU** | $\text{acc}_1[o, y, x] = q_{b1}[o] + \sum_{k_y=0}^2 \sum_{k_x=0}^2 q_{in}[y+k_y, x+k_x] \cdot q_{w1}[o, 0, k_y, k_x]$<br>$q_{a1}[o, y, x] = \text{sat\_relu\_int8}\!\left(\text{requant}(\text{acc}_1[o,y,x], M_1, n_1)\right)$ | `acc1`: `int32`<br>`q_a1`: `int8 [0,127]` |
| **2. MaxPool 2×2** | $q_{p1}[c, y, x] = \max_{d_y \in \{0,1\}, d_x \in \{0,1\}} q_{a1}[c, 2y+d_y, 2x+d_x]$ | `int8 [0,127]` $\to$ `int8 [0,127]` |
| **3. Conv2 + ReLU** | $\text{acc}_2[o, y, x] = q_{b2}[o] + \sum_{i=0}^7 \sum_{k_y=0}^2 \sum_{k_x=0}^2 q_{p1}[i, y+k_y, x+k_x] \cdot q_{w2}[o, i, k_y, k_x]$<br>$q_{a2}[o, y, x] = \text{sat\_relu\_int8}\!\left(\text{requant}(\text{acc}_2[o,y,x], M_2, n_2)\right)$ | `acc2`: `int32`<br>`q_a2`: `int8 [0,127]` |
| **4. GAP ($\div 169$)** | $\text{sum}_{gap}[c] = \sum_{y=0}^{12} \sum_{x=0}^{12} q_{a2}[c, y, x] \in [0, 21463]$<br>$q_{gap}[c] = \text{sat\_relu\_int8}\!\left((\text{sum}_{gap}[c] \cdot M_{gap} + 2^{n_{gap}-1}) \gg n_{gap}\right)$ | `sum_gap`: `int32 [0,21463]`<br>`q_gap`: `int8 [0,127]` |
| **5. FC** | $\text{acc}_{fc}[k] = q_{b,fc}[k] + \sum_{c=0}^{15} q_{gap}[c] \cdot q_{w,fc}[k, c]$<br>$q_{fc}[k] = \text{sat\_int8}\!\left(\text{requant}(\text{acc}_{fc}[k], M_{fc}, n_{fc})\right)$ | `acc_fc`: `int32`<br>`q_fc`: `int8 [-128,127]` |
| **6. Argmax** | $\hat{y} = \min \{ k \in \{0..9\} \mid q_{fc}[k] = \max_j q_{fc}[j] \}$ (lowest index wins ties) | `uint4 [0..9]` |

### 3.3 Division by $169$ and Dynamic-Range Rescaling in Global Average Pooling (`GAP`)
In GAP, each of the 16 channels sums $13 \times 13 = 169$ integers $q_{a2} \in [0, 127]$, so $\text{sum}_{gap} \in [0, 21463]$.

1. **Unscaled exact reciprocal ($\div 169$):** For $s_{gap} = s_{a2}$, the exact integer division $\text{round}(\text{sum}_{gap} / 169)$ is achieved over the entire range $S \in [0, 21463]$ by:
   $$
   M_{gap,0} = 12409, \qquad n_{gap,0} = 21 \quad \Longrightarrow \quad \left\lfloor \frac{S \cdot 12409 + 2^{20}}{2^{21}} \right\rfloor \equiv \left\lfloor \frac{S + 84}{169} \right\rfloor \quad \forall S \in [0, 21463]
   $$
   (Verified exhaustively over all 21,464 values in `sw/tests/test_m4_quant.py`.)
2. **Combined $\div 169$ + dynamic-range rescaling ($s_{a2} \to s_{gap}$):** Because spatial averaging over $13 \times 13$ reduces the peak activation magnitude by ~6× (`relu2` max $\approx 99$ vs `gap` max $\approx 12$), keeping $s_{gap} = s_{a2}$ would leave `q_gap` in `[0, 18]` (wasting nearly 3 bits of INT8 precision before FC). Instead, we fold the $\frac{1}{169}$ division and the scale ratio $\frac{s_{a2}}{s_{gap}}$ into a single dyadic multiplier $(M_{gap}, n_{gap})$:
   $$
   \frac{M_{gap}}{2^{n_{gap}}} \approx \frac{s_{a2}}{169 \cdot s_{gap}}
   $$
   Using the exact same hardware unit `(sum_gap * M_gap + 2^(n_gap - 1)) >> n_gap` thus performs both the $\div 169$ average and full-range `[0, 127]` rescaling in one step, boosting INT8 accuracy by **+1.90 percentage points**.

---

## 4. PTQ and QAT Methodology

1. **Post-Training Quantization (PTQ):**
   - Runs a calibration pass of the trained FP32 model (`checkpoints/fp32_best.pt`) over the training split to record the 99.95th percentile activation ranges after `relu1`, `relu2`, and `fc`.
   - Computes $(s_w, s_b, s_{out}, M, n)$ for every layer and evaluates exact integer inference.
2. **Quantization-Aware Training (QAT):**
   - Fine-tunes the FP32 checkpoint for 20 epochs with straight-through-estimator (STE) fake-quantization on weights and activations using the exact same symmetric INT8 grids, then re-extracts the integer parameters $(q_w, q_b, M, n)$.
   - Whichever model (PTQ or QAT) achieves higher validation accuracy in **true integer inference** is selected as the canonical `checkpoints/int8_quantized.npz` for Milestone 5 export, while both PTQ and QAT accuracies are recorded in `reports/quantization/quantization_report.md`.
