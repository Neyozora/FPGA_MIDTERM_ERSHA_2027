# Ready-to-Paste AI Prompt: Deep Theoretical Explanation of Method 2 (FPGA + HPS CNN Accelerator)

Copy and paste everything inside the quote block below into another AI model whenever you want a deep, intuitive, first-principles explanation of all the theories behind this project:

```text
I built a complete Hardware-Software Co-Design Image Classification system on an Intel Cyclone V SoC FPGA (Terasic DE10-Standard, 5CSXFC6D6F31C6) using CIFAR-10 (32×32×3 RGB), a 4-layer INT8 CNN accelerator in Verilog-2001, ModelSim bit-exact verification, Quartus Prime 18.1 synthesis, and an ARM Cortex-A9 HPS Linux bridge driving both a 640×480 Color VGA HUD and a 128×64 Graphic LCD.

Please explain the underlying theories behind every part of this system clearly, from first principles, with concrete numerical examples:

1. **Integer-Only Neural Network Arithmetic & Quantization Theory:**
   - Why do we quantize inputs with MSB-flip (`q_in = pixel - 128`, scale `1/128`) instead of floating-point normalization (`(x - mean) / std`), and how does that save hardware gates while keeping zero-centered activations?
   - How does **BatchNorm Folding** mathematically merge `gamma, beta, running_mean, running_var` into `Conv2d` weights and biases so the FPGA needs zero division or square-root hardware?
   - Why does **Per-Channel Symmetric INT8 QAT** (Straight-Through Estimator) outperform Per-Layer quantization on deeper 4-layer CNNs?
   - Derive **Dyadic Requantization** (`(acc * M_c + 2^(n_c - 1)) >> n_c`): why does replacing floating-point division `s_in * s_w / s_out` with a 16-bit integer multiply and bit-shift make Python and Verilog 100% bit-exact?
   - Why does an `8×8` Global Average Pooling (GAP) layer (`64 = 2^6`) require zero DSP blocks in hardware (`(sum + 32) >> 6`)?

2. **Parallel FPGA Convolution Hardware Theory:**
   - Compare Method 1 (Sequential 1-MAC/cycle) vs. Method 2 (2-Line Shift-Register Line Buffer + `4×9 = 36` Parallel MAC Array with `Po = 4`).
   - Explain how a **2-line shift-register line buffer** turns a 1D raster stream into a simultaneous `3×3` 2D sliding window with automatic `PAD=1` zero-padding.
   - Explain why Cyclone V **M10K BRAMs** require 1-cycle synchronous reads and unconditional read/write blocks (and why a conditional read accidentally infers 131,072 flip-flops!).
   - Explain **ping-pong 4-bank feature-map BRAMs** (`fmap_a0..3`, `fmap_b0..3`) and why banking by `oc % 4` eliminates write serialization bottlenecks.

3. **SoC Hardware-Software Co-Design & Display Theory:**
   - Explain how the ARM Cortex-A9 HPS communicates with the FPGA fabric over the **Lightweight HPS-to-FPGA AXI Bridge (`0xFF200000`)** via Linux `/dev/mem` and `mmap()`.
   - Explain what `202,538` clock cycles, `4.05 ms`, and `247 FPS` mean at `50 MHz`, and why Method 2 runs **1.50× faster** than Method 1 even though the CNN has **43.1× more parameters** (`61,162` vs. `1,418`).
   - Explain **4×4 Bayer ordered dithering** used to render 24-bit RGB CIFAR-10 images onto the 1-bit monochrome `128×64` HPS Graphic LCD, and how the 25 MHz VGA pixel pipeline reconstructs 24-bit color from MSB-flipped `INT8` BRAMs.
```
