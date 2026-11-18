// ============================================================================
// Deep-Dive Master Guidebook: Setup, Route, Milestones, Worked Math & Viva Q&A
// Compile with: typst compile fpga_complete_math_and_setup_guide.typ fpga_complete_math_and_setup_guide.pdf
// ============================================================================

#set document(
  title: "Deep-Dive Master Guidebook: FPGA CNN Setup, 10-Milestone Route, Worked Math, Hardware Architecture & Lecturer Q&A",
  author: "Ersha Anandita",
)

#set page(
  paper: "a4",
  margin: (x: 1.75cm, y: 2cm),
  header: align(right)[
    #text(size: 8.5pt, fill: luma(110))[
      *Deep-Dive Master Guidebook* • DE10-Standard Cyclone V FPGA CNN Accelerator (Method 1)
    ]
  ],
  footer: context [
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(2pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 8.5pt, fill: luma(100))[Terasic DE10-Standard (`5CSXFC6D6F31C6`) • Quartus Prime 18.1 + ModelSim 18.1]],
      align(right)[#text(size: 8.5pt, fill: luma(100))[Page #counter(page).display("1 of 1", both: true)]],
    )
  ],
)

#set text(
  font: ("Calibri", "Arial", "Times New Roman"),
  size: 10.2pt,
  lang: "en",
)

#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => block(above: 1.35em, below: 0.7em)[
  #text(size: 13.8pt, weight: "bold", fill: rgb("#1e3a8a"))[#it]
  #v(-4pt)
  #line(length: 100%, stroke: 1.2pt + rgb("#1e3a8a"))
]

#show heading.where(level: 2): it => block(above: 1.05em, below: 0.5em)[
  #text(size: 11.4pt, weight: "bold", fill: rgb("#0f172a"))[#it]
]

#let boxnote(title, body, color: rgb("#1d4ed8"), bg: rgb("#eff6ff")) = block(
  width: 100%,
  fill: bg,
  stroke: (left: 4pt + color, rest: 0.5pt + color.lighten(50%)),
  inset: 9.5pt,
  radius: (right: 4pt),
  above: 0.75em,
  below: 0.75em,
)[
  #text(weight: "bold", size: 10.5pt, fill: color)[#title] \
  #body
]

// ============================================================================
// TITLE BANNER
// ============================================================================
#align(center)[
  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    inset: (x: 16pt, y: 18pt),
    radius: 6pt,
  )[
    #text(size: 9.8pt, weight: "bold", fill: rgb("#38bdf8"))[
      DEEP-DIVE MASTER GUIDEBOOK • METHOD 1 (SEQUENTIAL SINGLE-MAC BASELINE)
    ]
    #v(5pt)
    #text(size: 15.5pt, weight: "bold", fill: white)[
      End-to-End FPGA CNN Image Classifier on Terasic DE10-Standard: Complete Setup, 10-Milestone Route, Worked-Out Math, RTL Timing, VGA/JTAG & Lecturer Q&A
    ]
    #v(6pt)
    #text(size: 9.5pt, fill: rgb("#cbd5e1"))[
      Device: Intel Cyclone V SX SoC (`5CSXFC6D6F31C6`) • LSI Design Contest 2027 Specification • Author: Ersha Anandita
    ]
  ]
]

// ============================================================================
// 1. COMPLETE STEP-BY-STEP ENVIRONMENT & HARDWARE SETUP
// ============================================================================
= Step-by-Step Environment & Hardware Setup (Fresh PC to Live FPGA)

Follow these exact steps if you are setting up this project on a fresh Windows 10/11 PC or demonstrating how to reproduce the environment from scratch.

== Step 1: Install Python 3.12 & Required AI/GUI Packages
1. Download and install *Python 3.12 (64-bit)* from `python.org` (check *"Add python.exe to PATH"* during installation, and keep Tcl/Tk enabled so `tkinter` works).
2. Open *PowerShell* inside the project root (`cd c:\Users\Ersha Anandita\Downloads\fpga_midterm`) and install the required Python packages:
   ```powershell
   # Optional: create an isolated virtual environment
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Install PyTorch (CUDA 12.8 for RTX 50-series GPU, or CPU wheel) + scientific & GUI libs
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install numpy matplotlib pillow pytest
   ```
3. Verify Python + GPU + Tkinter work:
   ```powershell
   python -c "import torch, numpy, matplotlib, PIL, tkinter; print('CUDA:', torch.cuda.is_available(), '| Tk OK')"
   ```

== Step 2: Install Intel Quartus Prime Lite 18.1 & ModelSim 18.1
1. Download *Intel Quartus Prime Lite Edition 18.1 (Build 625)* from Intel's FPGA Download Center.
2. During installation, make sure these 3 components are checked:
   - *Quartus Prime Lite Edition* (`C:\intelFPGA_lite\18.1\quartus\bin64\`)
   - *Cyclone V Device Support* (required for DE10-Standard `5CSXFC6D6F31C6`)
   - *ModelSim - Intel FPGA Starter Edition 18.1* (`C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\`)
3. Verify both tools from PowerShell:
   ```powershell
   C:\intelFPGA_lite\18.1\quartus\bin64\quartus_sh.exe --version
   C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlog.exe -version
   ```

== Step 3: Install Typst (for Compiling `.typ` Reports to `.pdf`)
1. Install *Typst* via Windows `winget`:
   ```powershell
   winget install --id Typst.Typst
   ```
2. Compile any `.typ` report in this repository to PDF:
   ```powershell
   typst compile fpga_complete_math_and_setup_guide.typ fpga_complete_math_and_setup_guide.pdf
   ```

== Step 4: Connect DE10-Standard & Install USB-Blaster II Driver
1. Connect the *`12V DC` power adapter* to the DE10-Standard and press the red `ON/OFF` button.
2. Plug a Micro-USB cable from your laptop into the *`USB Blaster II`* port on the board (NOT the `USB-UART` port).
3. *If Windows Device Manager shows `USB-Blaster II` with a yellow warning icon:*
   - Right-click it $arrow.r$ *Update Driver* $arrow.r$ *Browse my computer for drivers* $arrow.r$ select `C:\intelFPGA_lite\18.1\quartus\drivers\usb-blaster-ii`.
4. Verify the JTAG chain sees both chips (`SOCVHPS` + `5CSXFC6D6F31C6`):
   ```powershell
   C:\intelFPGA_lite\18.1\quartus\bin64\jtagconfig.exe
   # Expected output:
   # 1) DE-SoC [USB-1]
   #   4BA00477   SOCVHPS
   #   02D020DD   5CSEBA6(.|ES)/5CSEMA6/..
   ```
5. *(Optional)* Connect a 15-pin *VGA monitor* to the blue VGA port (`ADV7123` 24-bit Video DAC) and a *PS/2 mouse* (`PIN_AB25/AA25`) if available.

== Step 5: Verify the Entire Pipeline (`58/58` Tests) & Launch the Live Demo
1. Run the full automated verification suite covering Milestones 1 through 10:
   ```powershell
   python -m pytest -v sw/tests
   ```
2. Program the DE10-Standard board and launch the live VGA + JTAG + Pinterest GUI in one command:
   ```powershell
   .\run_demo.bat
   ```
   *(Or run `python -m sw.gui_app` and click *`⚡ Flash FPGA (.sof)`* inside the GUI window.)*

// ============================================================================
// 2. DEEP DIVE: THE 10-MILESTONE ENGINEERING ROUTE
// ============================================================================
= Deep Dive: The 10-Milestone Route (Why Every Step Exists)

Why do FPGA AI engineers split a project into 10 milestones instead of jumping straight to Verilog? Because *if a CNN outputs the wrong class on real silicon, debugging raw waveforms without intermediate golden checkpoints is nearly impossible.* Each milestone locks down one mathematical or hardware layer:

#figure(
  table(
    columns: (auto, 1.15fr, 1.85fr, 1.25fr),
    inset: 5pt,
    align: (center, left, left, left),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[MS]],
    [#text(fill: white, weight: "bold")[Milestone]],
    [#text(fill: white, weight: "bold")[Exact Engineering Task & Key Parameters]],
    [#text(fill: white, weight: "bold")[Verified Gate Result]],
    [*M1*], [Dataset & Padding], [Download Fashion-MNIST (`60k` + `10k`), stratified split (`54k/6k`, `seed=42`), pad `28×28 → 32×32` (`P=2`), quantize `pixel >> 1` (`s_in = 2^-7`)], [`12/12` pytest pass; `100` balanced HW vectors],
    [*M2*], [PyTorch CNN Model], [Build `sw/model/cnn.py`: `Conv1(1→8, 3×3) → ReLU → Pool(2×2) → Conv2(8→16, 3×3) → ReLU → GAP(13×13) → FC(16→10)`], [`14/14` pytest pass; `1,418` params; `259,648` MACs],
    [*M3*], [FP32 Training], [Train `30` epochs on GPU (`AdamW lr=1e-3, wd=1e-4, CosineAnnealingLR, batch=128`)], [`4/4` pytest pass; `86.56%` test acc (`87.27%` val)],
    [*M4*], [INT8 Quantization], [Compare PTQ (`85.02%`) vs. 12-epoch QAT (`85.87%`); derive dyadic shifts `(M, n)` & verify `GAP ÷ 169` exactness], [`7/7` pytest pass; `-0.69%` drop vs. FP32],
    [*M5*], [`.mem`/`.mif` Export], [Format INT8 weights & INT32 biases to 2's-complement hex in `(OC, IC, kH, kW)` order], [`6/6` pytest pass; `16` files bit-exact read-back],
    [*M6*], [Integer Ref Model], [Implement pure-integer Python loops reading `.mem` files & dump `15` golden `.mem` vectors (`sim/vectors/`)], [`5/5` pytest pass; `100%` parity on `10k` images],
    [*M7*], [Verilog-2001 RTL], [Write `conv_engine.v`, `relu.v`, `maxpool.v`, `gap.v`, `fully_connected.v`, `argmax.v`, `cnn_top.v`], [`4/4` pytest pass; `0` errors/warnings in `vlog`],
    [*M8*], [Unit Testbenches], [Simulate all 6 compute modules individually in ModelSim against M6 golden vectors], [`2/2` pytest pass; `0 / 31,828` mismatches],
    [*M9*], [Full Integration Sim], [Simulate `cnn_top.v` on Image 0 (`11,731` internal BRAM elements) + 10-class sweep (`Vec 0..90`)], [`2/2` pytest pass; `0` mismatches; `302,978` cycles],
    [*M10*], [Quartus + Real Board], [Wrap in `de10_standard_top.v` (VGA + PS/2 + JTAG), synthesize, program `@2`, verify timing & hardware], [`2/2` pytest pass (`58/58` total); `69.73 MHz`],
  ),
  caption: [All 10 milestones with exact hyperparameters, test counts, and verification gates.],
)

== Per-Class Accuracy Analysis (Why Method 1 Scores `85.87%`)
Across all `10,000` test images, Method 1's INT8 QAT model achieves `85.87%` accuracy (`95.49%` agreement with FP32):
- *High-Accuracy Distinct Shapes (`> 93%`):* `Bag` (`95.70%`), `Sandal` (`95.60%`), `Trouser` (`95.50%`), `Sneaker` (`94.70%`), `Ankle boot` (`93.60%`).
- *Upper-Body Garments (`64%–89%`):* `Dress` (`88.90%`), `T-shirt/top` (`80.90%`), `Pullover` (`74.90%`), `Coat` (`74.10%`), `Shirt` (`64.80%`). Because `Shirt`, `Coat`, `Pullover`, and `T-shirt/top` share nearly identical `$32 times 32$` outer silhouettes, compressing `$13 times 13$` maps down to a single GAP average loses fine collar/button location—a primary target for *Method 2*!

// ============================================================================
// 3. COMPLETE WORKED-OUT NUMERICAL EXAMPLE (IMAGE 0: T-SHIRT/TOP)
// ============================================================================
= Worked-Out Math Walkthrough (Following Image 0 Pixel-by-Pixel)

Let's trace *Test Image `0`* (`T-shirt/top`, `t10k[19]`) through every single equation from raw grayscale pixel to final FPGA decision and percentage!

== Step 1: Padding (`28×28 → 32×32`) & Input Quantization (`pixel >> 1`)
Each raw Fashion-MNIST pixel is an unsigned 8-bit integer $p in [0, 255]$.
1. *Padding:* Place the `$28 times 28$` image at rows `2..29`, columns `2..29` inside a `$32 times 32$` grid filled with `0` on the 2-pixel border:
   $ H_"in" = 28 + 2 + 2 = 32, quad W_"in" = 28 + 2 + 2 = 32, quad "Memory Address" = r times 32 + c in [0, 1023] $
2. *Quantization (`s_in = 2^-7 = 0.0078125`):* Shift every pixel right by 1 bit:
   $ q_"in" (r, c) = p(r, c) >> 1 = floor(p(r, c) / 2) in [0, 127] $
   - Example background pixel $p = 0 arrow.r q_"in" = 0$.
   - Example bright T-shirt pixel $p = 244 arrow.r q_"in" = 122$.
   - Why? Because `122` fits in signed `INT8` (`-128..+127`) with zero sign-bit overflow and *zero hardware logic gates*!

== Step 2: Spatial Dimension Formula (Why `32 → 30 → 15 → 13 → 1`)
For any 2D convolution or pooling window of kernel size $K$, padding $P$, and stride $S$:
$ H_"out" = floor((H_"in" - K + 2 P) / S) + 1 $
- *Conv1 (`K=3, P=0, S=1`, `1 → 8` channels):* $(32 - 3 + 0)/1 + 1 = 30 arrow.r bold("8 × 30 × 30 = 7,200 elements")$
- *MaxPool1 (`K=2, P=0, S=2`, `8 → 8` channels):* $(30 - 2 + 0)/2 + 1 = 15 arrow.r bold("8 × 15 × 15 = 1,800 elements")$
- *Conv2 (`K=3, P=0, S=1`, `8 → 16` channels):* $(15 - 3 + 0)/1 + 1 = 13 arrow.r bold("16 × 13 × 13 = 2,704 elements")$
- *GAP (`13×13` average, `16 → 16` channels):* $13 times 13 = 169$ values averaged per channel $arrow.r bold("16 elements")$
- *FC (`16 → 10` classes):* $16$ inputs $arrow.r bold("10 INT8 scores")$

== Step 3: Convolution Multiply-Accumulate (MAC) + Dyadic Requantization + ReLU
Inside `conv_engine.v`, for each output coordinate `(oc, oy, ox)`, the hardware initializes a `32-bit` signed accumulator `acc_reg` with the `INT32` bias $B["oc"]$, then loops over all input channels `ic` and `$3 times 3$` kernel taps `(ky, kx)`:
$ "acc"_"int32" = B["oc"] + sum_("ic"=0)^(C_"in"-1) sum_("ky"=0)^2 sum_("kx"=0)^2 q_"in" ["ic", "oy"+"ky", "ox"+"kx"] times W["oc", "ic", "ky", "kx"] $
For *Conv1* (`C_in = 1`), `9` `INT8 × INT8` multiplications are added to `B[oc]`. How do we rescale `acc_int32` back to `INT8` without floating-point division?
- True floating-point scale ratio: $M_"eff" = (s_"in" dot s_w) / s_"out"$.
- For Conv1: $s_"in" = 0.0078125$, $s_w = 0.2658627$, $s_"out" = 0.04693471 arrow.r M_"eff" = 0.0442555$.
- Multiply by $2^(19) = 524,288$: $0.0442555 times 524,288 = 23,202.6 approx bold(M = "23,202")$ with shift $bold(n = 19)$!
- Verilog `64-bit` exact rounding + saturation (`conv_engine.v` lines 113–120):
  $ "scaled" = ("acc"_"int32" times 23202 + 2^(18)) >>> 19, quad q_"relu1" = "clip"("scaled", 0, 127) $
  *Concrete Check:* Suppose `acc_int32 = +1,000`. Then $(1000 times 23202 + 262144) / 524288 = 23,464,144 / 524,288 = bold(44)$. If `acc_int32 = -500`, `scaled = -22`, and `relu.v` clamps it to *`0`*!

== Step 4: MaxPool1 (`2×2`, Stride `2`)
For each channel `c` and output `(y, x)` in `0..14`, `maxpool.v` reads 4 signed `INT8` pixels at `(2y, 2x)`, `(2y, 2x+1)`, `(2y+1, 2x)`, `(2y+1, 2x+1)` and outputs the largest:
$ q_"pool" [c, y, x] = max(a, b, c, d) $
*Example:* If a `$2 times 2$` patch of `conv1_relu_ram` contains `[12, 44, 0, 31]`, `maxpool.v` outputs *`44`*.

== Step 5: Conv2 (`8 → 16`) & GAP (`÷ 169` Without a Hardware Divider!)
`Conv2` runs the exact same `conv_engine.v` hardware, now with `8` input channels (`8 × 3 × 3 = 72` MACs per output) and dyadic parameters $M = 16868, n = 20$.
Next, *GAP* (`gap.v`) must compute the average of $13 times 13 = 169$ spatial pixels per channel (`sum / 169`) AND rescale from $s_"conv2" = 0.4377999$ to $s_"gap" = 0.08094258$:
- Ideal multiplier: $(0.4377999 / (169 times 0.08094258)) times 2^(19) = 0.0320051 times 524,288 = bold("16,780")$, shift $bold(n = 19)$!
- For *Image 0*, `gap.v` outputs these exact 16 `INT8` feature numbers (`gap_ram[0..15]`):
  $ "gap"[0..15] = [19, 20, 41, 18, 0, 52, 8, 5, 36, 17, 15, 29, 24, 61, 16, 29] $

== Step 6: Fully Connected (`16 → 10`), Argmax Winner & Softmax `%`
`fully_connected.v` computes 10 dot products (one per clothing class $k in 0..9$), scales with $M = 23793, n = 21$, and clamps to `[-128, 127]`. For *Image 0*, `fc_score_ram[0..9]` holds:
$ "fc_score_ram"[0..9] = [bold(+7), -28, -5, -2, -19, -25, -2, -36, -7, -14] $
1. *In Hardware (`argmax.v`):* Scans indices `0..9` using strict `>` comparison. Since *`+7`* at index *`0`* is the largest, the FPGA asserts `predicted_class = 4'd0` (`T-shirt/top`) and `done = 1'b1`.
2. *In Python GUI (`Softmax %`):* Multiply each score by $s_"fc" = 0.4488924$:
   - Class 0 (`T-shirt`): $+7 times 0.4488924 = +3.142 arrow.r e^(3.142) = bold(23.156)$
   - Class 3 (`Dress`) & Class 6 (`Shirt`): $-2 times 0.4488924 = -0.898 arrow.r e^(-0.898) = 0.407$ each
   - Sum of all 10 exponentials $sum_(j=0)^9 e^(q_j s_"fc") approx 24.48$
   - Winning Confidence: $23.156 / 24.48 times 100% approx bold("94.6%")$ (or `82.4%`–`94.6%` depending on exact logit temperature; all 10 bars sum to `100.0%`).

// ============================================================================
// 4. HARDWARE RESOURCE, BRAM, CYCLE & TIMING MATH
// ============================================================================
= Hardware Resource, BRAM, Cycle Latency & Timing Math

== Parameter, MAC & Accumulator Bit-Width Math
- *Total Weights:* $(8 times 1 times 3 times 3) + (16 times 8 times 3 times 3) + (10 times 16) = 72 + "1,152" + 160 = bold("1,384 INT8 weights (1,384 B)")$.
- *Total Biases:* $8 + 16 + 10 = bold("34 INT32 biases (136 B)")$.
- *Total Parameters:* $"1,384" + 34 = bold("1,418 parameters (1,520 bytes)")$.
- *Total MACs/Image:* $(7,200 times 9) + (2,704 times 72) + (10 times 16) = "64,800" + "194,688" + 160 = bold("259,648 MACs")$.
- *Why `INT32` Accumulators Never Overflow:* Worst-case Conv2 sum is $72 times 127 times 128 = plus.minus "1,170,432"$, and observed max across all 10,000 test images is `[-41,058, +14,037]`—fitting in just `17–21` bits inside a `32-bit` signed register (`±2.14e9`).

== On-Chip M10K BRAM Math (Zero External DDR!)
Each Cyclone V M10K block provides `10,240` bits (`1,024 × 10b` or `1,280` bytes).
- *CNN Core (`cnn_top.v`):*
  - `img_ram` (`1,024×8b` = `8,192` b, `1` M10K) + `conv1_relu_ram` (`7,200×7b` = `50,400` b, `5` M10Ks) + `pool1_ram` (`1,800×8b` = `14,400` b, `2` M10Ks) + `conv2_relu_ram` (`2,704×7b` = `18,928` b, `2` M10Ks) + Weight/Bias ROMs (`12,352` b, `11` M10Ks) = *`104,272` bits (`21` M10K blocks)* + `208` MLAB bits (`gap_ram` + `fc_score_ram`).
- *DE10-Standard Board Additions:* `board_img_rom` (`10,240×8b = 81,920` b, `10` M10Ks) + `vga_img_ram` (`1,024×8b = 8,192` b, `1` M10K) = *`194,384` bits total (`32 / 553` M10K blocks = `6%`)*.

== Exact Cycle Latency Math (`302,978` Cycles Explained)
Why does `cnn_top.v` take *`302,978` cycles* to do `259,648` MACs? Because `conv_engine.v` pipelines its address generation (`S_ADDR`) and synchronous M10K read (`S_READ_WAIT`) so inner-loop MACs run at `1 MAC/cycle`, plus `~3` cycles of `S_REQUANT` + BRAM drain overhead per output pixel:
- `Conv1 + ReLU1`: $"7,200" times 9 + "overhead" = bold("86,404 cycles")$ (`28.52%`)
- `MaxPool1`: $"1,800" times 6 "cycles/window" = bold("10,803 cycles")$ (`3.57%`)
- `Conv2 + ReLU2`: $"2,704" times 72 + "overhead" = bold("202,804 cycles")$ (`66.94%`)
- `GAP`: $16 times 169 + "scale/write" = bold("2,755 cycles")$ (`0.91%`)
- `FC` (`193`) + `Argmax` (`14`) + `FSM setup` (`5`) = *`212` cycles* (`0.07%`)
- *Total:* $"86,404" + "10,803" + "202,804" + "2,755" + 193 + 14 + 5 = bold("302,978 cycles")$!
- *Wall-Clock Time:* $"302,978" / "50,000,000" = bold("6.060 ms")$ (*`165.0 FPS`*); at $F_"max" = 69.73" MHz"$, $"302,978" / "69,730,000" = bold("4.345 ms")$ (*`230.1 FPS`*).

== Quartus Prime Post-Fit Resource Breakdown (`5CSXFC6D6F31C6`)
#figure(
  table(
    columns: (1.6fr, auto, auto, auto, auto),
    inset: 5.5pt,
    align: (left, right, right, right, right),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Cyclone V Resource]],
    [#text(fill: white, weight: "bold")[`cnn_top` Core]],
    [#text(fill: white, weight: "bold")[Full Board (VGA+JTAG)]],
    [#text(fill: white, weight: "bold")[Device Max]],
    [#text(fill: white, weight: "bold")[Utilization]],
    [Logic Utilization (ALMs)], [~`750`], [*`1,942`*], [`41,910`], [*`5%`*],
    [Combinational ALUTs], [*`1,148`*], [*`2,652`*], [`83,820`], [`3.2%`],
    [Dedicated Registers], [*`582`*], [*`2,768`*], [`167,640`], [`1.7%`],
    [Block Memory Bits (M10K)], [*`104,272` b*], [*`194,384` b*], [`5,662,720` b], [*`3%`*],
    [M10K RAM Blocks], [*`21`*], [*`32`*], [`553`], [*`6%`*],
    [DSP Blocks (`conv`=9, `gap`=6, `fc`=7)], [*`22`*], [*`22`*], [`112`], [*`20%`*],
    [I/O Pins (`VGA`, `PS2`, `SW`, `KEY`, `HEX`, `LEDR`)], [`0`], [*`98`*], [`499`], [`20%`],
    [Slow 85°C $F_"max"$ (`CLOCK_50` = 50 MHz)], [–], [*`69.73 MHz`*], [`50.00 MHz`], [`+5.659 ns` slack],
  ),
  caption: [Exact post-fit Quartus Prime 18.1 resource & timing results (`de10_standard.fit.rpt` & `de10_standard.sta.rpt`).],
)

*Where do the `22` DSP blocks come from?* `conv_engine` uses `9` DSPs (`8×8` MAC + `32×16` requantization + address multiply), `gap` uses `6` DSPs (`32×16` reciprocal multiply), and `fully_connected` uses `7` DSPs (`8×8` MAC + `32×16` requantization).

// ============================================================================
// 5. LIVE VGA, 272-BIT ROW-BURST JTAG & LECTURER Q&A CHEAT SHEET
// ============================================================================
= Live VGA, 272-Bit Row-Burst JTAG & Lecturer Q&A Cheat Sheet

== How VGA (`640×480 @ 60 Hz`) & 272-Bit Row-Burst JTAG Work
- *VGA Timing (`rtl/vga_cnn_hud.v`):* `CLOCK_50` (`50 MHz`) is divided by 2 to produce `25 MHz` `VGA_CLK`. A standard `640×480` frame has `800` total horizontal pixels (`640` active + `16` front porch + `96` sync + `48` back porch) and `525` vertical lines (`480 + 10 + 2 + 33`). Refresh rate = $25,000,000 / (800 times 525) = 59.52" Hz" approx 60" Hz"$.
- *272-Bit Row-Burst JTAG (`sw/jtag_bridge.py`):* Instead of sending 1 pixel at a time (`1,024` USB-JTAG transactions), `altsource_probe` (`"CNN0"`) packs an entire 32-pixel row (`256` bits) + `16` control bits into one `272-bit` JTAG write—making custom photo uploads and laptop-mouse drawing `32×` faster, with a 1-command `20.5 µs` hardware clear!

== Top 6 Questions Your Lecturer Might Ask (Ready-to-Use Answers)
1. *"Why did you pad `28×28` images to `32×32`?"*
   - *Answer:* Two reasons: (1) LSI Design Contest 2027 requires `$32 times 32$` input resolution, and (2) `32×32 = 1,024` bytes fits cleanly into *1 single M10K BRAM block* (`1024×8b`) with power-of-two `10-bit` addressing (`{row[4:0], col[4:0]}`).
2. *"Why does your network only have `1,418` parameters when normal CNNs have millions?"*
   - *Answer:* We replaced the huge `Flatten → FC` layer (`2,704 × 10 = 27,040` weights) with *Global Average Pooling (`GAP`)*, shrinking the FC layer to `16 × 10 = 160` weights so 100% of weights and activations fit in `104,272` bits (`21` M10Ks) with zero off-chip DDR!
3. *"How do you do division and scaling in Verilog without floating-point?"*
   - *Answer:* Using *dyadic fixed-point scaling* `(acc * M + 2^(n-1)) >> n`. Even `GAP ÷ 169` is replaced by `(sum * 16780 + 2^18) >> 19`, verified 100% bit-exact across all `21,464` possible sum values.
4. *"Why did accuracy drop from `86.56%` (FP32) to `85.87%` (INT8), and how would you improve it in Method 2?"*
   - *Answer:* Only `-0.69%` drop thanks to 12-epoch QAT! The remaining errors are mostly `Shirt`/`Coat`/`Pullover` overlap; in *Method 2*, we can use per-channel quantization, BatchNorm folding, and $2 times 2$ spatial pyramid pooling to keep top-vs-bottom garment structure.
5. *"Why does one inference take `302,978` cycles (`6.06 ms`), and how can you make it faster?"*
   - *Answer:* Method 1 is a resource-minimal *Sequential Single-MAC* baseline (`1` MAC/cycle, `5%` ALMs). `Conv2` + `Conv1` take `95.46%` of cycles. In *Method 2*, adding a `3×3` line-buffer with `9` parallel MACs and `4×` channel unrolling would cut latency by `> 20×` (`< 0.3 ms`).
6. *"How do you prove the real FPGA matches Python and ModelSim?"*
   - *Answer:* ModelSim verified `0 / 31,828` unit mismatches and `0 / 11,731` full-pipeline mismatches (`302,978` cycles). On the physical DE10-Standard, flipping `SW[9]=1` displays `3 0 2 9 7 8` on `HEX5..HEX0`, and our JTAG bridge reads back all 10 hardware `INT8` scores (`[+7, -28, -5, ...]`) matching Python bit-for-bit!
