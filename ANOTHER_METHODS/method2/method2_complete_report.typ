#set page(
  paper: "a4",
  margin: (x: 20mm, y: 20mm),
  numbering: "1 / 1",
)
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Hardware-Software Co-Design of a Parallel INT8 4-Layer CNN Accelerator on Cyclone V SoC FPGA
  ]
  #v(4pt)
  #text(size: 11pt, fill: luma(80))[
    Method 2 Complete Technical Report — Terasic DE10-Standard (`5CSXFC6D6F31C6`) \
    CIFAR-10 (`32×32×3` RGB) • `3×3` Line-Buffer & `4×9` MAC Array • ARM Cortex-A9 HPS + VGA + 128×64 LCD
  ]
  #v(6pt)
  #line(length: 100%, stroke: 0.6pt + luma(150))
]

#v(6pt)

= Executive Summary & Method 1 vs. Method 2 Comparison

This report documents the complete design, integer-only quantization mathematics, parallel Verilog-2001 RTL architecture, ModelSim bit-exact verification, Quartus Prime 18.1 synthesis, Cyclone V HPS (ARM Cortex-A9) Lightweight AXI integration, and live hardware/display verification of *Method 2* on the *Terasic DE10-Standard* (`5CSXFC6D6F31C6`) for the *30th LSI Design Contest in Hokkaido (2027)* (`Class 1: Image Classification Class`).

While *Method 1* established a verified 2-layer sequential single-MAC baseline on grayscale Fashion-MNIST (`1,418` parameters, `302,978` cycles = `6.06 ms`), *Method 2* upgrades every layer of the stack to handle natural *`32×32×3` RGB CIFAR-10* images using a *4-layer CNN* (`61,162` parameters — `43.1×` larger) while executing *1.50× faster* (`202,538` cycles = `4.051 ms` @ `50 MHz` = *`246.9 FPS`*) through a *2-line shift-register line buffer* and *$4 × 9 = 36$ parallel INT8 MAC array*.

#table(
  columns: (1.5fr, 1.4fr, 1.5fr),
  inset: 6pt,
  align: (left, center, center),
  stroke: 0.5pt + luma(160),
  [*Metric*], [*Method 1 (Baseline)*], [*Method 2 (Parallel + HPS)*],
  [Dataset & Input], [Fashion-MNIST (`32×32×1` Gray)], [*CIFAR-10 (`32×32×3` RGB)*],
  [Input Quantization], [`q_in = pixel >> 1` (`[0, 127]`)], [*`q_in = pixel - 128` (`[-128, 127]`, MSB flip)*],
  [CNN Topology], [2-Layer (`1→8→Pool→8→16→GAP→FC`)], [*4-Layer (`3→16→P→32→P→64→64→GAP→FC`)*],
  [Total Parameters], [`1,418`], [*`61,162` (`43.1×`)*],
  [Quantization], [Per-Layer INT8 QAT], [*BN-Folded + Per-Channel INT8 QAT*],
  [Test Accuracy (INT8)], [`85.87%` (Fashion-MNIST)], [*`83.54%` (CIFAR-10, `<0.1%` drop)*],
  [Conv Engine], [Sequential 1-MAC/cycle], [*2-Line Buffer + `4×9` Parallel MAC (`Po=4`)*],
  [Exact Clock Cycles], [`302,978` cycles], [*`202,538` cycles (`1.50×` faster)*],
  [Latency / FPS @ 50 MHz], [`6.060 ms` (`165.0 FPS`)], [*`4.051 ms` (`246.9 FPS`)*],
  [ModelSim Bit-Exact], [`100/100` preds, `1,000/1,000` scores], [*`100/100` preds, `1,000/1,000` scores*],
  [Cyclone V ALMs], [`804 / 41,910` (`2%`)], [`3,815` (`9%` FPGA-only) / `8,149` (`19%` + HPS)],
  [M10K RAM Blocks], [`31 / 553` (`6%`)], [`102` (`18%`) / `103` (`19%` + HPS)],
  [DSP Blocks], [`22 / 112` (`20%`)], [*`52 / 112` (`46%`)*],
  [50 MHz Timing], [PASS (`0` violations)], [*PASS (`+3.634 ns` setup slack, `0` violations)*],
  [Displays & Control], [VGA Grayscale + PS/2 + JTAG], [*VGA 24-bit RGB + HPS 128×64 LCD + AXI*],
)

---

= Dataset, Split & Hardware Input Quantization

- *Dataset:* Official *CIFAR-10* (`torchvision.datasets.CIFAR10`), natively *`32 × 32 × 3` RGB* (`3,072` bytes/image) across `10` real-world classes: `0: airplane`, `1: automobile`, `2: bird`, `3: cat`, `4: deer`, `5: dog`, `6: frog`, `7: horse`, `8: ship`, `9: truck`.
- *Deterministic Split (`SEED = 42`):*
  - Training: `45,000` images (`4,500` per class) with `RandomCrop(32, padding=4)` and `RandomHorizontalFlip(p=0.5)`.
  - Validation: `5,000` images (`500` per class, stratified).
  - Official Test: `10,000` images (`1,000` per class).
  - Balanced Hardware Test Vectors: `100` images (`10` per class, first `10` of each class in official test order).
- *Zero-Gate MSB-Flip Input Quantization:*
  Each `uint8` RGB channel $p in [0, 255]$ is mapped to signed `INT8` by subtracting `128`:
  $ q_"in" = p - 128 in [-128, +127], quad s_"in" = 2^(-7) = 1 / 128, quad x = q_"in" dot s_"in" in [-1.0, 127/128]. $
  In hardware, converting between `uint8` and `INT8` requires *0 logic gates*—only inverting the MSB (`{~pixel[7], pixel[6:0]}`), allowing the VGA HUD to recover the exact 24-bit RGB color by flipping the MSB back.

---

= CNN Architecture, Algorithm & Integer-Only Math

== 4-Layer Network Topology

All feature maps and weights are sized to fit inside Cyclone V on-chip M10K BRAM (`103 / 553` blocks = `19%`):
1. *`Conv1` (`3×3`, `3 → 16`, `PAD=1`) + `BN` + `ReLU`* $arrow.r$ `16 × 32 × 32`
2. *`Pool1` (`MaxPool 2×2`, stride `2`)* $arrow.r$ `16 × 16 × 16`
3. *`Conv2` (`3×3`, `16 → 32`, `PAD=1`) + `BN` + `ReLU`* $arrow.r$ `32 × 16 × 16`
4. *`Pool2` (`MaxPool 2×2`, stride `2`)* $arrow.r$ `32 × 8 × 8`
5. *`Conv3` (`3×3`, `32 → 64`, `PAD=1`) + `BN` + `ReLU`* $arrow.r$ `64 × 8 × 8`
6. *`Conv4` (`3×3`, `64 → 64`, `PAD=1`) + `BN` + `ReLU`* $arrow.r$ `64 × 8 × 8`
7. *`GAP` (`8×8` Global Average Pooling)* $arrow.r$ `64` features
8. *`FC` (`64 → 10`) + `Argmax`* $arrow.r$ `10` `INT8` class scores & `predicted_class` (`0..9`)

== Two-Stage Training: BatchNorm Folding + Per-Channel INT8 QAT

1. *Stage 1 (Float32 + BatchNorm2d, 80 epochs):* Trained with `SGD(lr=0.05, momentum=0.9, weight_decay=5e-4)` and `CosineAnnealingLR`, reaching *`83.49%`* test accuracy (`84.18%` best val).
2. *Stage 2 (BatchNorm Folding + Per-Channel QAT, 25 epochs):* Each `BatchNorm2d` $(gamma, beta, mu, sigma^2, epsilon)$ is algebraically folded into its preceding `Conv2d` weights and biases:
   $ w_"fold",c = (gamma_c / sqrt(sigma_c^2 + epsilon)) w_c, quad b_"fold",c = beta_c - (gamma_c mu_c) / sqrt(sigma_c^2 + epsilon). $
   The folded network is fine-tuned with Straight-Through Estimator (STE) fake-quantization (`[-127, 127]` per-channel weights, `[0, 127]` ReLU activations) using `Adam(lr=2e-4)`, achieving *`83.61%`* QAT test accuracy.
3. *Stage 3 (Pure Integer Dyadic Inference):* Each output channel $c$ uses an integer multiplier $M_c in [1, 65535]$ and shift $n_c in [8, 24]$ approximating $(s_"in" dot s_(w,c)) / s_"out" approx M_c / 2^(n_c)$:
   $ y_(q,c) = "clamp"_(0, 127) ( floor( ( ("acc"_c + b_(q,c)) dot M_c + 2^(n_c - 1) ) / 2^(n_c) ) ). $
   Pure integer evaluation achieves *`83.54%`* on all `10,000` CIFAR-10 test images (*`0.07%`* drop from QAT) and *`84.00%`* on the `100` hardware test vectors.

== Zero-DSP `8×8` Global Average Pooling

Because `Conv4` outputs an `8 × 8` spatial map (`64 = 2^6` pixels per channel), GAP requires *zero DSP multipliers* in hardware—just a 16-bit integer sum followed by a rounded arithmetic right-shift by `6`:
$ "gap"_c = "clamp"_(0, 127) ( (sum_(r=0)^7 sum_(k=0)^7 x_(c,r,k) + 32) >> 6 ). $

---

= Hardware Mechanism & Parallel RTL Architecture

== `conv_engine_m2.v` — 2-Line Shift-Register Buffer + `4×9` MAC Array

Instead of reading 1 pixel per cycle like Method 1, `conv_engine_m2.v` streams each `(H × W)` input channel through two `32`-deep line buffers (`lb0`, `lb1`) and a 3-column window register (`col0`, `col1`, `col2`), exposing all *9 pixels of the `3×3` window simultaneously every clock cycle* with automatic `PAD=1` zero-padding at all four borders.

Four parallel `9-MAC` units ($P_o = 4$) multiply the 9 window pixels by `4 × 9 = 36` `INT8` weights fetched in a single cycle from four `1676 × 72`-bit M10K weight ROMs (`w_bank0..3`):
- *Stage 1:* Latch `3×3` window & read prior partial sums from `psum_ram0..3` (`1024 × 32b` M10K BRAMs).
- *Stage 2:* Compute `36` signed `8 × 8` products in parallel.
- *Stage 3:* Four 9-input adder trees sum the products with `psum_old[0..3]`.
- *Stage 4:* For intermediate input channels (`ic < C_in - 1`), write partial sums back to `psum_ram0..3`. On the final input channel (`ic == C_in - 1`), add `INT32` bias, apply per-channel dyadic requantization + ReLU, and write 4 output channels simultaneously to the 4-bank ping-pong feature-map BRAMs (`fmap_a0..3` or `fmap_b0..3`).

== `maxpool_m2.v`, `gap_fc_argmax_m2.v` & Ping-Pong BRAMs

- *4-Bank Ping-Pong BRAMs:* `fmap_a0..3` (`4 × 4096 × 8b` = `16` M10Ks) and `fmap_b0..3` (`4 × 2048 × 8b` = `8` M10Ks) alternate between layers so all 4 parallel channels read/write in 1 cycle with zero serialization overhead.
- *`maxpool_m2.v`:* Performs `2×2` stride-2 MaxPooling across 4 banks simultaneously (`Pool1`: `32×32 → 16×16`; `Pool2`: `16×16 → 8×8`).
- *`gap_fc_argmax_m2.v`:* Computes `8×8` GAP (`>> 6`), runs `64 → 10` dense MAC + per-channel dyadic requantization (`[-128, 127]`), and selects `predicted_class` (`0..9`).

---

= HPS (ARM Cortex-A9) ↔ FPGA AXI Integration, VGA & LCD

The Qsys system (`hps_cnn_qsys`) enables `hps_0.h2f_lw_axi_master` (Lightweight HPS-to-FPGA AXI Bridge mapped at physical address *`0xFF200000`*), bridged to an Avalon-MM Slave inside `de10_standard_top_m2.v`:

#table(
  columns: (0.8fr, 0.7fr, 2.5fr),
  inset: 5pt,
  stroke: 0.5pt + luma(160),
  [*Offset*], [*Access*], [*Function*],
  [`0x00`], [W], [`REG_CTRL`: Write `1` to pulse `cnn_start` on the FPGA.],
  [`0x04`], [R], [`REG_STATUS`: Bit `[4]` = `cnn_done_latched`, Bits `[3:0]` = `predicted_class`.],
  [`0x08`], [W], [`REG_PIX_WR`: `{data[7:0], addr[11:0]}` writes 1 RGB byte to `img_ram` & VGA mirror.],
  [`0x0C`], [R], [`REG_CYCLES`: Exact 24-bit hardware inference clock cycle count (`202,538`).],
  [`0x10..0x34`], [R], [`REG_SCORE[0..9]`: Sign-extended `INT8` class scores for classes `0..9`.],
  [`0x38`], [R], [`REG_BOARD_IO`: `{sw_sync[9:0], active_img_class[3:0], key_sync[3:0]}` for LCD sync.],
)

- *Switch-Follow Mode (`SW[8] = 0`, Default):* HPS polls `REG_BOARD_IO` (`0x38`) and `REG_STATUS` (`0x04`). Whenever you flip `SW[3:0]` (`0..9`) or press `KEY[1]`/`KEY[2]`, *both* the 640×480 Color VGA HUD and the 128×64 HPS Graphic LCD update together.
- *100-Image Slideshow Mode (`SW[8] = 1`):* HPS streams all `100` CIFAR-10 images (`hw_images_100.bin`) over `0xFF200000` into the FPGA, triggers inference, and displays each live result on both VGA and LCD.

---

= What Every Display & Hardware Indicator Means

== HPS 128×64 Graphic LCD Indicators

- *Left Panel (`48×48` box):* `1.5×` scaled, `4×4` Bayer-dithered monochrome preview of the `32×32` CIFAR-10 image. A sweeping horizontal *scanline* animates down the image during the first frames to visualize convolution scanning.
- *Top-Right Header:*
  - `SW=X -> P=Y` *(in Switch-Follow Mode)*: `X` is the active ROM image index (`0..9`), and `Y` is the FPGA's live `predicted_class` (`0..9`).
  - `HPS->VGA #XX` *(in `SW[8]=1` Slideshow Mode)*: `XX` is the current image index (`00..99`) streamed from HPS DDR3 to the FPGA.
- *Middle-Right Label:* Predicted CIFAR-10 class name (`AIRPLANE`, `AUTOMOBILE`, `BIRD`, `CAT`, `DEER`, `DOG`, `FROG`, `HORSE`, `SHIP`, `TRUCK`).
- *Animated Confidence Bar:* Bar width scales linearly from the winning `INT8` score (`[-128, +127]`) read live from `REG_SCORE[pred]` (`20 + ((score + 128) * 50) / 255` pixels).
- *Bottom-Right Timing (`4.05ms 247FPS`):*
  - *`4.05ms`:* Exact FPGA hardware inference latency at `50.00 MHz`:
    $ T_"infer" = (202,538 "cycles") / (50 times 10^6 "Hz") = 4.05076 "ms" approx 4.05 "ms". $
  - *`247FPS`:* Maximum sustained hardware throughput (frames per second):
    $ "FPS" = (50 times 10^6 "Hz") / (202,538 "cycles") = 246.87 "FPS" approx 247 "FPS". $

== 640×480 Color VGA HUD, LEDs & 7-Segment Displays

- *VGA Left (`256×256`, `8×` zoom):* True 24-bit RGB CIFAR-10 image reconstructed from `{~q[7], q[6:0]}` across `vga_r_ram`, `vga_g_ram`, `vga_b_ram`.
- *VGA Right & Bottom:* 10-class horizontal bar chart (winner highlighted in Emerald Green) + `WINNER: <CLASS>` banner.
- *`LEDR[9:0]`:* One-hot indicator of `predicted_class` (`LEDR[k] = 1` when `predicted_class == k`).
- *`HEX5..HEX0`:* When `SW[9] = 0`, shows active image (`HEX5`) and predicted class (`HEX0`). When *`SW[9] = 1`*, displays the exact 24-bit hardware `cycle_count` (`03172A` hex = `202,538` decimal).

---

= Verification & Quartus Synthesis Results

- *ModelSim 18.1 (`tb_cnn_top_m2.v`):* `100 / 100` prediction parity, `1,000 / 1,000` `INT8` score parity, `202,538` cycles/image (`0` errors, `0` warnings).
- *Python ↔ ModelSim GUI (`sw_m2/gui_app_m2.py`):* Verified `10 / 10` bit-exact score parity on live single-image simulation (`3.57 s` wall time).
- *Quartus Prime 18.1 (`de10_standard_m2.fit.summary` & `.sta.summary`):*
  - *ALMs:* `8,149 / 41,910` (`19%`) | *Registers:* `4,527` | *M10Ks:* `103 / 553` (`19%`, `860,288` bits) | *DSPs:* `52 / 112` (`46%`)
  - *Timing (Slow 1100mV 85°C):* `CLOCK_50` Setup Slack *`+3.634 ns`*, Hold Slack *`+0.228 ns`*, `hps_sdram` Setup Slack *`+1.730 ns`* (*0 timing violations*).

---

= Step-by-Step Setup Guide (Zero to Working Hardware)

1. *Preprocess CIFAR-10 (`32×32×3` RGB):*
   ```powershell
   python -m sw_m2.preprocess
   ```
2. *Train 4-Layer CNN with BN Folding + Per-Channel INT8 QAT (GPU):*
   ```powershell
   python -u -m sw_m2.train_qat
   ```
3. *Export `.mem` Weight/QParam/Vector Files & HPS Binary Payload:*
   ```powershell
   python -m sw_m2.export_mem
   ```
4. *Verify Bit-Exact RTL in ModelSim 18.1:*
   ```powershell
   C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlib.exe work_m2
   C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlog.exe -work work_m2 rtl_m2/*.v sim_m2/tb_cnn_top_m2.v
   C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vsim.exe -c work_m2.tb_cnn_top_m2 -do "run -all; quit -f"
   ```
5. *Generate Qsys HPS Bridge, Synthesize & Program DE10-Standard:*
   ```powershell
   cd quartus_m2
   C:\intelFPGA_lite\18.1\quartus\sopc_builder\bin\qsys-script.exe --project-file=de10_standard_m2.qpf --script=build_hps_qsys.tcl
   C:\intelFPGA_lite\18.1\quartus\sopc_builder\bin\qsys-generate.exe hps_cnn_qsys.qsys --synthesis=VERILOG
   C:\intelFPGA_lite\18.1\quartus\bin64\quartus_sh.exe --flow compile de10_standard_m2
   C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe -c "DE-SoC [USB-1]" -m JTAG -o "p;output_files\de10_standard_m2.sof@2"
   ```
6. *Compile & Run HPS LCD + VGA Bridge on DE10-Standard Angstrom Linux:*
   ```bash
   cd /home/xuser/hps_cifar10_lcd
   gcc -O2 -std=gnu99 -Wall -Dsoc_cv_av \
     -Iintel-socfpga-hwlib/armv7a/hwlib/include/soc_cv_av \
     -Iintel-socfpga-hwlib/armv7a/hwlib/include -I. \
     main_cifar10_vga_bridge.c terasic_lib.c LCD_Lib.c LCD_Driver.c LCD_Hw.c lcd_graphic.c font.c \
     -o hps_vga_bridge -lrt -lm
   ./hps_vga_bridge
   ```
7. *Launch ModelSim-Integrated Python GUI (Optional Desktop Demo):*
   ```powershell
   python -m sw_m2.gui_app_m2
   ```
