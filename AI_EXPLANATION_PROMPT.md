# AI Prompt: Generate a Super Clear, Beginner-Friendly Step-by-Step Explanation of My FPGA CNN Midterm Project

Copy and paste everything below the line (`---`) into any AI (ChatGPT, Claude, Gemini) whenever you want it to generate study notes, presentation scripts, or super-simple explanations based on your **exact verified project numbers**.

---

I need you to explain my **FPGA CNN Image Classification Midterm Project (Method 1)** in a super easy, beginner-friendly, step-by-step way (like explaining to someone new to FPGAs and deep learning), using **ONLY the exact verified facts and numbers below** (do not invent or change any numbers!).

### 1. Project Goal & Hardware Setup
- **Goal:** Run a Two-Layer Convolutional Neural Network (CNN) on a real FPGA to classify `32×32` grayscale Fashion-MNIST clothing images (`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`) using **100% on-chip M10K BRAM** (zero external DDR) and **8-bit integer (`INT8`) math** (zero floating-point units in hardware).
- **Physical Board:** Terasic **DE10-Standard** (`Intel Cyclone V SX SoC 5CSXFC6D6F31C6`: `41,910` ALMs, `553` M10K BRAM blocks, `112` DSP blocks, `50 MHz` `CLOCK_50`).
- **Tools Used:** Python 3.12 + PyTorch `2.11.0+cu128`, ModelSim Intel FPGA Starter Edition 18.1 (`vsim 10.5b`), Intel Quartus Prime Lite 18.1 (`Build 625`), and Typst for PDF documentation.
- **Live Demo Features Built:**
  - `HEX5..HEX0` 7-segment displays (`0 - P A S 0` or `3 0 2 9 7 8` cycle count when `SW[9]=1`) + `LEDR[9:0]` one-hot class LEDs.
  - `640×480 @ 60 Hz` **VGA Monitor HUD** (`rtl/vga_cnn_hud.v`) showing the `32×32` image zoomed `8×` (`256×256`), 10-class score bar chart, and `WINNER` banner.
  - **272-bit Row-Burst USB-JTAG Bridge** (`sw/jtag_bridge.py` + `altsource_probe` `"CNN0"`) linking the Python Desktop GUI (`sw/gui_app.py`) to the physical FPGA so I can draw with my laptop mouse, upload Pinterest clothing photos, randomize images, and clear the screen (`KEY[3]` or `Clear Screen` button).

### 2. The 10-Milestone Route (Explain Each Step Simply)
1. **M1 (Dataset & Preprocessing):** Pad `28×28` Fashion-MNIST images with 2 zero-pixels on each side to `32×32` (`1,024` pixels = 1 M10K BRAM). Split into `54,000` train / `6,000` val / `10,000` test + `100` HW test vectors. Quantize pixels with `q_in = pixel >> 1` (`[0, 127]`, scale $s_{\text{in}} = 2^{-7}$, zero hardware gates!).
2. **M2 (PyTorch CNN Model):** `Input (1×32×32)` $\rightarrow$ `Conv1+ReLU1 (3×3, 1→8, 8×30×30)` $\rightarrow$ `MaxPool1 (2×2 s2, 8×15×15)` $\rightarrow$ `Conv2+ReLU2 (3×3, 8→16, 16×13×13)` $\rightarrow$ `GAP (13×13 avg → 16)` $\rightarrow$ `FC (16→10)` $\rightarrow$ `Argmax (0..9)`.
   - Total Parameters: **`1,418`** (`1,384` INT8 weights + `34` INT32 biases = `1,520` bytes).
   - Total MACs/image: **`259,648`** (`64,800` Conv1 + `194,688` Conv2 + `160` FC).
3. **M3 (FP32 Training):** Trained 30 epochs (AdamW + Cosine LR) $\rightarrow$ **`86.56%` test accuracy** (`87.27%` val).
4. **M4 (INT8 Quantization):** Post-Training Quantization (`PTQ`) got `85.02%`; 12-epoch Quantization-Aware Training (`QAT`) recovered to **`85.87%` INT8 test accuracy** (`-0.69%` drop, `95.49%` agreement with FP32). Derived exact integer multiplier-and-shift pairs `(acc * M + 2^(n-1)) >> n`:
   - `Conv1`: `M=23202, n=19` | `Conv2`: `M=16868, n=20` | `GAP (÷169)`: `M=16780, n=19` | `FC`: `M=23793, n=21`.
5. **M5 (`.mem` & `.mif` Export):** Exported all 16 hex memory files (`mem/*.mem`, `mem/*.mif`) with `0` bit errors.
6. **M6 (Python Integer Ref Model):** Pure integer Python loops (`sw/refmodel/int8_ref_model.py`) matching PyTorch QAT `100%` across all `10,000` test images, generating golden `.mem` vectors in `sim/vectors/`.
7. **M7 (Verilog-2001 RTL):** Wrote 7 modules in `rtl/`: `conv_engine.v`, `relu.v`, `maxpool.v`, `gap.v`, `fully_connected.v`, `argmax.v`, `cnn_top.v` (`0` errors, `0` warnings in `vlog 10.5b`).
8. **M8 (Unit Testbenches):** All 6 unit testbenches passed in ModelSim 18.1 with **`0` mismatches across `31,828` element checks**.
9. **M9 (Full Integration Sim):** Verified `cnn_top.v` in ModelSim on Image 0 (`0 / 11,731` internal BRAM mismatches, **`302,978` cycles = `6.060 ms` @ 50 MHz**) and a 10-class sweep (`10 / 10` = `100%` agreement, `0 / 100` score mismatches).
10. **M10 (Quartus Synthesis & Real DE10-Standard Board):**
    - `cnn_top` Core: `1,148` ALUTs, `582` registers, `104,272` BRAM bits (`21` M10K blocks), `22` DSP blocks (`9` conv + `6` gap + `7` fc).
    - Full `de10_standard_top` (with VGA + PS/2 + 272-bit JTAG + 10-image ROM): **`1,942 / 41,910` ALMs (`5%`)**, **`2,768` registers**, **`194,384` BRAM bits (`32 / 553` M10Ks = `6%`)**, **`22 / 112` DSPs (`20%`)**, **`98 / 499` pins (`20%`)**, Slow 85°C **$F_{\max} = 69.73\text{ MHz}$** (`+5.659 ns` setup slack).
    - Inference Speed: **`6.060 ms` (`165.0 FPS`)** at `50 MHz`, or **`4.345 ms` (`230.1 FPS`)** at `69.73 MHz`.

### What I Want You to Produce for Me
Please explain:
1. How the math works from raw `28×28` pixels $\rightarrow$ `32×32` $\rightarrow$ `Conv1` $\rightarrow$ `MaxPool1` $\rightarrow$ `Conv2` $\rightarrow$ `GAP` $\rightarrow$ `FC` (`+7, -28, -5...`) $\rightarrow$ `Argmax` (`Class 0`) $\rightarrow$ `Softmax (%)`.
2. How the hardware resource math adds up (`1,418` parameters, `259,648` MACs, `104,272` core BRAM bits, `302,978` clock cycles, `22` DSPs, `1,942` ALMs).
3. A simple 1-minute script I can read out loud while demoing the DE10-Standard board, VGA monitor, switches, and Python GUI to my lecturer!
