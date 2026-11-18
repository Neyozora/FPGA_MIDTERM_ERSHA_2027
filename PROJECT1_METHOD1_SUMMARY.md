# Project 1 (Method 1): Sequential INT8 Two-Layer CNN FPGA Accelerator

**Target Board:** Terasic DE10-Standard (Intel Cyclone V SX SoC **`5CSXFC6D6F31C6`**)  
**Dataset:** Fashion-MNIST (`28 × 28 × 1` grayscale, 2-pixel zero-padded to **`32 × 32 × 1`**)  
**Toolchain:** PyTorch + ModelSim 18.1 + Intel Quartus Prime Lite 18.1

---

## 1. Architecture & Parameters (`1,418` Total Parameters)

Method 1 implements a compact, bit-exact **2-layer sequential CNN** in synthesizable Verilog-2001 (`rtl/cnn_top.v`):

| Stage | Module | Input → Output Shape | Parameters | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Input** | `img_ram` | `1 × 32 × 32` | – | `q_in = pixel >> 1 ∈ [0, 127]`, `s_in = 2^-7` (`0` gates) |
| **Conv1 + ReLU1** | `conv_engine.v` + `relu.v` | `1×32×32 → 8×30×30` | `72` weights + `8` bias | `3×3` valid conv, `M=23202, n=19` |
| **MaxPool1** | `maxpool.v` | `8×30×30 → 8×15×15` | – | `2×2` stride-2 max pooling |
| **Conv2 + ReLU2** | `conv_engine.v` + `relu.v` | `8×15×15 → 16×13×13` | `1,152` weights + `16` bias | `3×3` valid conv, `M=16868, n=20` |
| **GAP** | `gap.v` | `16×13×13 → 16` | – | `13×13` sum (`÷169` via `×16780 >> 19`) |
| **FC + Argmax** | `fully_connected.v` + `argmax.v` | `16 → 10` | `160` weights + `10` bias | `M=23793, n=21`, `INT8 [-128, 127]` scores |

---

## 2. Dataset, Quantization & Accuracy (`reports/quantization/quantization_report.md`)

- **Split:** `54,000` train / `6,000` val (`600`/class, `SEED=42`) / `10,000` test / `100` hardware test vectors (`10`/class).
- **Quantization:** Per-Layer Symmetric INT8 Quantization-Aware Training (QAT) with dyadic integer requantization (`(acc * M + 2^(n-1)) >>> n`):
  - **Train (`54,000`):** `87.69%` FP32 → **`86.84%`** QAT INT8 (`-0.85%`)
  - **Validation (`6,000`):** `87.27%` FP32 → `85.78%` PTQ → **`86.97%`** QAT INT8 (`-0.30%`)
  - **Test (`10,000`):** `86.56%` FP32 → `85.02%` PTQ → **`85.87%`** QAT INT8 (`-0.69%`, `95.49%` FP32 agreement)
  - **HW Test Vectors (`100`):** `85.0%` FP32 → **`82.0%`** QAT INT8

---

## 3. ModelSim 18.1 Verification (`reports/simulation/cnn_top_verification_report.md`)

- **Bit-Exactness:** `0 / 11,731` tensor mismatches across `img_ram` → `conv1_relu_ram` → `pool1_ram` → `conv2_relu_ram` → `gap_ram` → `fc_score_ram` → `predicted_class`, and **`0 / 100`** score mismatches on the 10-class sweep (`Max Abs Error = 0`).
- **Exact Cycle Breakdown (`302,978` cycles total = `6.060 ms` @ `50 MHz` = `165.0 FPS`):**
  - `Conv1 + ReLU1`: `86,404` cycles (`28.52%`)
  - `MaxPool1`: `10,803` cycles (`3.57%`)
  - `Conv2 + ReLU2`: `202,804` cycles (`66.94%`)
  - `GAP`: `2,755` cycles (`0.91%`)
  - `FC`: `193` cycles (`0.06%`)
  - `Argmax` + Input Load: `19` cycles (`< 0.01%`)

---

## 4. Quartus Prime 18.1 Synthesis & DE10-Standard Deployment (`reports/synthesis/synthesis_report.md`)

- **Resource Utilization (`de10_standard_top` on `5CSXFC6D6F31C6`):**
  - **ALMs:** `804 / 41,910` (`1.92%` ≈ `2%`)
  - **Combinational ALUTs / Registers:** `1,323` ALUTs / `646` registers
  - **M10K RAM Blocks:** `31 / 553` (`5.61%`, `186,192` bits) + `208` MLAB bits
  - **DSP Blocks:** `22 / 112` (`19.64%` ≈ `20%`)
  - **Timing (`CLOCK_50` = 50 MHz):** **PASS (0 violations)** — Slow `85°C` setup slack `+5.254 ns` (`Fmax = 67.82 MHz`), Slow `0°C` setup slack `+5.057 ns` (`Fmax = 66.92 MHz`).
- **Board Peripherals & GUI (`rtl/de10_standard_top.v`, `rtl/vga_cnn_hud.v`, `rtl/ps2_mouse_ctrl.v`, `sw/gui_app.py`):**
  - Live `640×480 @ 60 Hz` grayscale VGA HUD with 10-class bar chart.
  - PS/2 mouse drawing canvas + `KEY[3]` clear + `KEY[2]` LFSR randomizer + `SW[3:0]` 10-class ROM selector + `SW[9]` cycle counter (`302978`).
  - 272-bit row-burst USB-JTAG bridge (`altsource_probe`) connected to `python -m sw.gui_app` for real-time drawing & custom photo classification.
