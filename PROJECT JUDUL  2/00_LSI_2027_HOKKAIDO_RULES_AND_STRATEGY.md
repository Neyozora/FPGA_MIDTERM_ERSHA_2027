# 30th LSI Design Contest in Hokkaido (2027) — Official Overview & Strategy

Based on the official contest announcement (`lsi-contest.com`), the milestone **30th LSI Design Contest in Hokkaido (2027)** has the following structure:

---

## 1. Official Dates & Format
- **Event:** 30th LSI Design Contest in Hokkaido
- **Report Submission Deadline:** **Sunday, January 31, 2027**
- **Presentation & Conference Date:** **Friday, March 5, 2027**
- **Format:** Hybrid (On-site in Hokkaido + Online via Zoom)

---

## 2. Official Design Theme & Competition Classes
- **Main Design Theme:** **CNN (Convolutional Neural Network) Implementation — any number of layers**
- **Two Official Tracks:**
  1. **Class 1: Image Classification Class** *(Our Target Track!)*
     - Implement a CNN image classifier in HDL (`Verilog-HDL` / `VHDL`) on FPGA or ASIC.
     - *Notice the key rule update:* **"any number of layers"** is allowed!
     - **Method 1 (Midterm Baseline)** implemented a 2-layer CNN (`Conv1 → Pool1 → Conv2 → GAP → FC`).
     - **Method 2+ (Final Contest Upgrade)** can add a **3rd or 4th convolutional block** (e.g., `3×3` + `BatchNorm` + `ReLU` + `Depthwise-Separable Conv`) to boost classification accuracy above `90%` on real clothing datasets!
  2. **Class 2: ZERO7 Class**
     - AI implementation targeting the **ZERO7** Raspberry Pi HAT FPGA board (by Gigafirm).

---

## 3. Winning Strategy from Midterm (Method 1) $\rightarrow$ Final Contest (Method 2)
Because **any number of CNN layers** is permitted in Class 1 of the **2027 Hokkaido Contest**, here is the ideal progression before **January 31, 2027**:
1. **Midterm (Done — Method 1):**
   - Verified 2-layer sequential `INT8` CNN (`1,418` params, `302,978` cycles = `6.06 ms` @ 50 MHz, `85.87%` INT8 accuracy, `5%` ALMs on DE10-Standard, live VGA monitor + JTAG GUI).
2. **Contest Upgrade (Method 2 — See `01_ALTERNATIVE_CLOTHING_DATASETS.md` & `02_HARDWARE_METHODS_FOR_LSI2027.md`):**
   - **Upgrade Depth & Accuracy:** Expand to a **3-layer or 4-layer CNN** with **BatchNorm folding** + **Per-Channel `INT8` QAT** + **`2×2` Spatial Pyramid Pooling** (targeting `> 90%` accuracy).
   - **Upgrade Speed (`> 20×`):** Replace the 1-MAC sequential loop with a **`3×3` Line-Buffer + 9-MAC Parallel Array** + output-channel unrolling (`< 0.3 ms` inference).
   - **Upgrade Dataset:** Train on real clothing photos (**Apparel Images / Fashion Product Images Small**) alongside Fashion-MNIST so live camera / Pinterest clothing demos classify accurately.
   - **Optional SoC Integration (`03_HPS_OPENOCD_VS_SD_CARD.md`):** Connect `cnn_top.v` to the DE10-Standard's ARM Cortex-A9 HPS (`0xFF200000` Lightweight AXI bridge) via SD Card Linux or OpenOCD.
