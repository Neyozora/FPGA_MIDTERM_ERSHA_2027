// ============================================================================
// FPGA Midterm Project Report — Method 1 (Baseline Architecture)
// Compile with: typst compile docs/midterm_method1_report.typ docs/midterm_method1_report.pdf
// ============================================================================

#set document(
  title: "Method 1: Baseline Sequential Single-MAC On-Chip BRAM Architecture with Symmetric INT8 QAT for CNN Image Classification on Cyclone V FPGA",
  author: "Ersha Anandita",
)

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.2cm),
  header: align(right)[
    #text(size: 8.5pt, fill: luma(110))[
      *FPGA Midterm Documentation — Method 1:* Sequential Single-MAC INT8 CNN Accelerator (Cyclone V)
    ]
  ],
  footer: context [
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(2pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 8.5pt, fill: luma(100))[LSI Design Contest 2027 Target • Terasic DE10-Standard]],
      align(right)[#text(size: 8.5pt, fill: luma(100))[Page #counter(page).display("1 of 1", both: true)]],
    )
  ],
)

#set text(
  font: ("Times New Roman", "Calibri", "Arial"),
  size: 10.5pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.65em,
)

#set heading(numbering: "1.1.")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em)[
  #text(size: 14pt, weight: "bold", fill: rgb("#1e3a8a"))[#it]
  #v(-4pt)
  #line(length: 100%, stroke: 1pt + rgb("#1e3a8a"))
]
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em)[
  #text(size: 11.8pt, weight: "bold", fill: rgb("#0f172a"))[#it]
]

#let callout(title, body, color: rgb("#2563eb"), bg: rgb("#eff6ff")) = block(
  width: 100%,
  fill: bg,
  stroke: (left: 3.5pt + color, rest: 0.5pt + color.lighten(50%)),
  inset: 10pt,
  radius: (right: 4pt),
  above: 0.9em,
  below: 0.9em,
)[
  #text(weight: "bold", fill: color)[#title] \
  #body
]

// ============================================================================
// TITLE BLOCK
// ============================================================================
#align(center)[
  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    inset: (x: 18pt, y: 20pt),
    radius: 6pt,
  )[
    #text(size: 10pt, weight: "bold", fill: rgb("#38bdf8"))[
      FPGA MIDTERM PROJECT REPORT • METHOD 1 (BASELINE ARCHITECTURE)
    ]
    #v(6pt)
    #text(size: 16pt, weight: "bold", fill: white)[
      Method 1: Baseline Sequential Single-MAC On-Chip BRAM Architecture with Symmetric INT8 Quantization-Aware Training for Two-Layer CNN Image Classification on Intel Cyclone V FPGA
    ]
    #v(8pt)
    #text(size: 10pt, fill: rgb("#cbd5e1"))[
      Target Platform: Terasic DE10-Standard (`5CSXFC6D6F31C6` / `5CSEBA6U23I7`) • LSI Design Contest 2027 Specification \
      Author: *Ersha Anandita* • Date: November 2026
    ]
  ]
]

#v(8pt)

#callout(
  "Executive Summary — What is Method 1?",
  [
    *Method 1* establishes our end-to-end, bit-accurate *Baseline Hardware/Software Co-Design* for compact Convolutional Neural Network (CNN) inference on an Intel Cyclone V FPGA. It combines:
    1. *Algorithmic Pipeline:* A `1,418`-parameter Two-Layer CNN (`Conv1 → ReLU → MaxPool → Conv2 → ReLU → GAP → FC → Argmax`) trained on `32 × 32` zero-padded Fashion-MNIST images using *Symmetric Signed INT8 Quantization-Aware Training (QAT)* and dyadic power-of-two requantization (`85.87%` test accuracy vs. `86.56%` FP32 baseline, only `0.69%` degradation).
    2. *Hardware Architecture:* A resource-minimal *Sequential Single-MAC Engine* storing `100%` of weights, biases, and intermediate feature maps in on-chip *M10K Block RAM (`14,258` bytes total)* with zero external SDRAM traffic.
    3. *Cycle-Accurate Verification:* Verified in *ModelSim Intel FPGA Starter Edition 18.1* across `31,828` unit testbench checks and `11,731` full-pipeline tensor elements with *`0` numerical mismatches (`100%` bit-exact parity)* and a deterministic latency of *`302,978` clock cycles (`6.060 ms` at `50 MHz`)*.
  ],
)

// ============================================================================
// 1. INTRODUCTION & DESIGN PHILOSOPHY OF METHOD 1
// ============================================================================
= Introduction & Design Philosophy of Method 1

Deploying Convolutional Neural Networks (CNNs) on low-power edge FPGAs such as the Intel Cyclone V SoC requires overcoming three fundamental hardware constraints:
1. *Limited On-Chip Memory:* Standard CNNs with large dense classification layers require megabytes of parameter storage, forcing costly off-chip DDR3 accesses.
2. *Floating-Point Hardware Cost:* IEEE-754 `FP32` multipliers and adders consume large numbers of Adaptive Logic Modules (ALMs) and DSP blocks.
3. *Verification Gap Between Software and RTL:* Floating-point non-associativity and hardware rounding differences often cause FPGA predictions to diverge from PyTorch training results.

To solve these challenges systematically, we structure our investigation into two phases:
- *Method 1 (This Report — Sequential Single-MAC Baseline):* Focuses on *architectural simplicity, minimal DSP/ALM area, 100% on-chip M10K BRAM storage, and 100% bit-exact integer equivalence* between Python and Verilog-2001 RTL. Each layer executes sequentially via a deterministic Finite State Machine (FSM), using a single time-multiplexed Multiply-Accumulate (MAC) datapath per active layer and pure integer multiplier-and-shift scaling.
- *Method 2 (Future Enhancement — Parallel & Accuracy-Optimized Architecture):* Builds directly upon Method 1's verified memory map and quantization framework to address its two primary bottlenecks: hardware cycle latency (`302,978` cycles/image) and upper-body clothing confusion (`Shirt` vs. `T-shirt/top` vs. `Pullover`).

== End-to-End 10-Milestone Hardware/Software Co-Design Workflow
Unlike purely software-based deep learning projects, deploying a CNN on an FPGA requires a strict, verifiable bridge from floating-point training to digital logic gates. Method 1 is engineered across *10 incremental milestones*, where each milestone acts as a verification gate for the next:

#figure(
  table(
    columns: (auto, 1.3fr, 1.5fr, 1.4fr),
    inset: 5.5pt,
    align: (center, left, left, left),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[MS]],
    [#text(fill: white, weight: "bold")[Milestone Name]],
    [#text(fill: white, weight: "bold")[Core Engineering Purpose]],
    [#text(fill: white, weight: "bold")[Key Output & Verified Result]],
    [*M1*], [Dataset & Hardware Preprocessing], [Zero-pad `28×28` Fashion-MNIST to `32×32×1` & quantize via `pixel >> 1`], [`data/processed/*.npz` (`54k` train, `6k` val, `10k` test, `100` HW vectors)],
    [*M2*], [Compact PyTorch CNN Architecture], [Design `1,418`-parameter GAP-based CNN fitting inside Cyclone V M10K BRAM], [`sw/model/cnn.py` (`1,384` weights + `34` biases, `259,648` MACs)],
    [*M3*], [FP32 Baseline Training & Eval], [Train floating-point teacher/baseline model (`30` epochs, AdamW + Cosine LR)], [`checkpoints/fp32_best.pt` (`86.56%` test acc, `87.27%` val acc)],
    [*M4*], [Symmetric INT8 Quantization (QAT)], [Convert FP32 weights/activations to signed `INT8` & dyadic `(M, n)` shifts], [`checkpoints/int8_quantized.npz` (`85.87%` INT8 test acc, `-0.69%` drop)],
    [*M5*], [FPGA Memory Export (`.mem`/`.mif`)], [Pack `INT8`/`INT32` tensors into two's-complement hex for ModelSim & Quartus], [`mem/*.mem` & `*.mif` (`16` files, `100%` bit-exact read-back verified)],
    [*M6*], [Bit-Exact Python Reference Model], [Pure-integer Python hardware loop matching RTL bit-for-bit + golden vectors], [`sw/refmodel/int8_ref_model.py` (`100%` parity on `10,000` test images)],
    [*M7*], [Synthesizable Verilog-2001 RTL], [Write all 7 hardware modules (`conv_engine`, `relu`, `maxpool`, `gap`, `fc`, `argmax`, `cnn_top`)], [`rtl/*.v` (`0` errors, `0` warnings in ModelSim `vlog 10.5b`)],
    [*M8*], [Module-Level Unit Testbenches], [Verify each RTL module in isolation against M6 golden vectors in ModelSim], [`sim/tb/tb_*.v` (`6/6` passed, `0` mismatches across `31,828` checks)],
    [*M9*], [Full-System CNN RTL Integration], [Verify complete `cnn_top.v` pipeline & 10-class multi-image sweep in ModelSim], [`302,978` cycles/image (`6.06 ms` @ `50 MHz`), `0 / 11,731` mismatches],
    [*M10*], [Quartus Prime FPGA Synthesis & Board Run], [Synthesize, place-and-route, & program DE10-Standard (`5CSXFC6D6F31C6`)], [`804` ALMs (`1.92%`), `22` DSPs (`19.6%`), `31` M10Ks (`5.6%`), $F_"max" = 67.82" MHz"$],
  ),
  caption: [Overview of the 10-Milestone Hardware/Software Co-Design methodology used to build and verify Method 1.],
)

=== Detailed Explanation of Each Milestone Step
- *Milestone 1 (Dataset Preparation & Preprocessing):* Raw Fashion-MNIST images are `$28 times 28$` `uint8` (`[0, 255]`), whereas the LSI Design Contest 2027 hardware specification requires `$32 times 32 times 1$` inputs (`1,024` bytes = exactly one `1024×8b` M10K BRAM). Milestone 1 applies 2-pixel symmetric zero-padding around each image, performs a deterministic stratified split (`54,000` train / `6,000` val / `10,000` test), selects a balanced 100-image hardware test suite (`10` images per class), and applies multiplier-free input quantization (`q_in = pixel >> 1` into signed `INT8 [0, 127]`).
- *Milestone 2 (PyTorch Model Architecture):* Constructs the hardware-constrained CNN in `sw/model/cnn.py`. By using two `$3 times 3$` VALID convolution layers (`1→8` and `8→16`) followed by `$13 times 13$` Global Average Pooling (`GAP`) instead of flattening the feature map (`16×13×13 = 2,704` features), the final Fully Connected (`FC`) layer shrinks from `27,040` weights down to just `160` weights (`16→10`).
- *Milestone 3 (FP32 Baseline Training & Evaluation):* Trains the floating-point network using Cross-Entropy Loss, AdamW, and Cosine Annealing LR for `30` epochs. This establishes the upper-bound accuracy target (`86.56%` on the 10k test set) and saves the best floating-point weights (`checkpoints/fp32_best.pt`).
- *Milestone 4 (Symmetric Signed INT8 Quantization):* Converts floating-point weights, biases, and intermediate activations into fixed-point integers (`INT8` weights/activations, `INT32` biases/accumulators). Evaluates both Post-Training Quantization (`85.02%`) and 12-epoch Quantization-Aware Training (`85.87%`), and derives exact integer multipliers $M$ and right-shifts $n$ so that hardware never uses floating-point arithmetic.
- *Milestone 5 (Memory File Generation — `.mem` & `.mif`):* Translates the quantized numpy arrays into hexadecimal `.mem` files (for ModelSim `$readmemh`) and `.mif` files (for Quartus Prime M10K BRAM synthesis), using explicit `(Out_Ch, In_Ch, kH, kW)` address flattening and 2's-complement formatting.
- *Milestone 6 (Python Bit-Exact Reference Model & Golden Vectors):* Implements the exact hardware nested loops in pure integer Python (`sw/refmodel/int8_ref_model.py`) reading directly from the exported `.mem` files. Verifies `100%` prediction parity with PyTorch QAT across all `10,000` test images and exports layer-by-layer golden `.mem` vectors (`sim/vectors/vec00_*.mem`) to serve as ground truth for RTL simulation.
- *Milestone 7 (Individual Verilog-2001 RTL Modules):* Implements the 7 synthesizable hardware blocks (`conv_engine.v`, `relu.v`, `maxpool.v`, `gap.v`, `fully_connected.v`, `argmax.v`, `cnn_top.v`) using explicit Finite State Machines, synchronous BRAM read wait states, and `64-bit` intermediate shift-round-saturate logic.
- *Milestone 8 (Individual ModelSim Unit Testbenches):* Tests each of the 6 compute engines in isolation using dedicated self-checking Verilog testbenches (`sim/tb/tb_*.v`) before connecting them together. All `31,828` element checks pass with `0` errors.
- *Milestone 9 (Complete CNN Integration & ModelSim Verification):* Connects all modules and on-chip BRAM buffers inside `cnn_top.v` and simulates the complete end-to-end inference in ModelSim (`tb_cnn_top.v`), verifying `0 / 11,731` internal tensor mismatches on Image 0 (`302,978` cycles) and `10 / 10` accuracy across the 10-class sweep.
- *Milestone 10 (Quartus Prime FPGA Synthesis & Timing Analysis):* Synthesizes `cnn_top.v` for the Intel Cyclone V FPGA (`5CSXFC6D6F31C6` / `5CSEBA6U23I7`) using Quartus Prime to measure actual hardware resource utilization (ALMs, registers, DSP blocks, M10K RAM blocks) and maximum operating clock frequency ($F_"max"$).

// ============================================================================
// 2. DATASET PREPROCESSING & INPUT QUANTIZATION
// ============================================================================
= Dataset Preprocessing & Hardware Input Quantization

== Fashion-MNIST to $32 times 32$ Hardware Format
The benchmark dataset is *Fashion-MNIST*, consisting of `60,000` training images (split reproducibly into `54,000` training and `6,000` stratified validation samples using seed `42`) and `10,000` official test images across `10` clothing categories (`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`).

To satisfy the *LSI Design Contest 2027* specification requiring a `$32 times 32 times 1$` input tensor (`1,024` pixels, mapping cleanly into a single `1024 × 8-bit` Cyclone V M10K BRAM block), each raw `$28 times 28$` grayscale image ($p in [0, 255]$) is centered inside a `$32 times 32$` frame using *2-pixel symmetric zero-padding* on all four borders (top, bottom, left, right).

== Multiplier-Free Input Quantization
Instead of floating-point normalization ($p / 255.0$), Method 1 uses an exact dyadic input scale $s_"in" = 2^(-7) = 1 / 128 = 0.0078125$ with zero-point $z_"in" = 0$:
$ q_"in" (r, c) = floor(p(r, c) / 2) = p(r, c) >> 1 in [0, 127] subset.eq [-128, 127] $
$ x_"dequant" (r, c) = q_"in" (r, c) times 2^(-7) in [0.0, 0.9921875] $
In hardware, converting an 8-bit unsigned camera/memory pixel `uint8 [0, 255]` into a signed `INT8` tensor input requires *zero gates*—it is simply a 1-bit right-shift (`{1'b0, pixel[7:1]}`).

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/dataset/samples_int8_grid.png", width: 100%),
    image("reports/dataset/class_mean_images.png", width: 100%),
  ),
  caption: [Left: Preprocessed `$32 times 32$` signed `INT8` (`[0, 127]`) input samples across all 10 classes. Right: Mean `$32 times 32$` spatial silhouette per clothing category.],
)

// ============================================================================
// 3. CNN ARCHITECTURE & HOW CLASSIFICATION WORKS STEP-BY-STEP
// ============================================================================
= CNN Architecture & End-to-End Inference Principle

== Layer-by-Layer Topology
Method 1 implements a compact 7-stage feed-forward CNN topology engineered specifically to replace heavy Fully Connected layers with *Global Average Pooling (GAP)*. This reduces the total parameter footprint to just *`1,418` parameters (`1,520` bytes in quantized INT8/INT32 form)*:

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, center, center, right, right, right),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Stage]],
    [#text(fill: white, weight: "bold")[Operation]],
    [#text(fill: white, weight: "bold")[Input Shape]],
    [#text(fill: white, weight: "bold")[Output Shape]],
    [#text(fill: white, weight: "bold")[Weights]],
    [#text(fill: white, weight: "bold")[Biases]],
    [#text(fill: white, weight: "bold")[MACs]],
    [`0. Input`], [Zero-padded `INT8` image], [`1×32×32`], [`1×32×32`], [`0`], [`0`], [`0`],
    [`1. Conv1+ReLU1`], [`3×3` conv (`1→8`, VALID) + ReLU], [`1×32×32`], [`8×30×30`], [`72`], [`8`], [`64,800`],
    [`2. MaxPool1`], [`2×2` max pool, stride `2`], [`8×30×30`], [`8×15×15`], [`0`], [`0`], [`0`],
    [`3. Conv2+ReLU2`], [`3×3` conv (`8→16`, VALID) + ReLU], [`8×15×15`], [`16×13×13`], [`1,152`], [`16`], [`194,688`],
    [`4. GAP`], [`13×13` spatial average (`÷169`)], [`16×13×13`], [`16`], [`0`], [`0`], [`0`],
    [`5. FC`], [Linear layer (`16 → 10`)], [`16`], [`10`], [`160`], [`10`], [`160`],
    [`6. Argmax`], [Index of maximum score], [`10`], [`1` (`0..9`)], [`0`], [`0`], [`0`],
    [*Total*], [*End-to-End Pipeline*], [*`1,024` B*], [*Class `0..9`*], [*`1,384` B*], [*`34` (`136` B)*], [*`259,648`*],
  ),
  caption: [Complete architectural specification, parameter count (`1,418` total), and computational complexity (`259,648` MACs/image) of Method 1.],
)

== How the Hardware Concludes a Clothing Class from Raw Pixels
A key question in hardware neural network design is: *how do raw memory numbers inside an FPGA turn into a semantic classification with a confidence percentage?* Method 1 executes this transformation in four mathematical stages (illustrated on Test Image `0`, a `T-shirt/top`):

1. *From Picture to Brightness Grid (`1,024` Numbers):*
   The `$32 times 32$` input image is stored in `img_ram[0..1023]` in row-major order ($"addr" = r times 32 + c$). Background pixels are `0`; bright pixels along the T-shirt collar and sleeves hold values between `80` and `127`.
2. *Spatial Feature Extraction (`Conv1 → Pool1 → Conv2 → GAP`):*
   - *`Conv1 + ReLU1` (`8 × 30 × 30`)* applies 8 learned `$3 times 3$` filters that highlight horizontal collars, vertical torsos, and sleeve edges, zeroing out negative responses via $"ReLU"(x) = max(0, x)$.
   - *`MaxPool1` (`8 × 15 × 15`)* keeps the strongest activation in every `$2 times 2$` window, reducing spatial size by $4 times$.
   - *`Conv2 + ReLU2` (`16 × 13 × 13`)* combines the 8 edge maps into 16 higher-level part detectors (e.g., wide short-sleeved top vs. split trouser legs vs. curved shoe sole).
   - *`GAP` (`16` numbers)* averages all $13 times 13 = 169$ spatial positions of each channel into a 16-element feature vector $"gap"_c in [0, 127]$. For Test Image `0`, the hardware computes:
     $ "gap"[0..15] = [19, 20, 41, 18, 0, 52, 8, 5, 36, 17, 15, 29, 24, 61, 16, 29] $
     Notice that Channel 13 (`61`) and Channel 5 (`52`) fire strongly for T-shirt shapes, whereas Channel 4 (`0`, shoe detector) is completely inactive.
3. *Class Voting via Fully Connected Layer (`16 → 10` Scores):*
   For each clothing category $k in {0, dots, 9}$, the `fully_connected` module multiplies the 16 GAP features by 16 learned signed `INT8` weights $W_"fc"[k, c]$ (positive weights for features that support class $k$, negative weights for features that contradict class $k$), adds bias $B_"fc"[k]$, and requantizes to a signed `INT8` class score (`logit`):
   $ "fc_score_ram"[0..9] = [+7, -28, -5, -2, -19, -25, -2, -36, -7, -14] $
4. *Winning Decision (`Argmax` in Hardware) & Confidence Percentage (`Softmax` in GUI):*
   - *Inside the FPGA (`rtl/argmax.v`):* The hardware scans `"fc_score_ram"[0..9]` to find the maximum signed integer. Since *`+7`* at index *`0`* is strictly greater than all other 9 scores, the FPGA asserts `predicted_class = 4'd0` (`T-shirt/top`) and `done = 1'b1`.
   - *In the Software GUI (`sw/gui_app.py`):* To display human-interpretable confidence percentages (`%`) that sum to `100%`, the 10 hardware integer scores $q_k$ are dequantized by the output scale $s_"fc" = 0.44889$ and passed through the *Softmax* function:
     $ P("Class" = k) = frac(exp(q_k dot s_"fc"), sum_(j=0)^9 exp(q_j dot s_"fc")) times 100% $
     Because $+7 times 0.44889 = +3.14$ dominates the runner-up scores ($-2 times 0.44889 = -0.90$), Class 0 (`T-shirt/top`) achieves the winning probability of *`82.4%`*.

// ============================================================================
// 4. FP32 TRAINING & SYMMETRIC INT8 QUANTIZATION-AWARE TRAINING
// ============================================================================
= FP32 Training & Symmetric INT8 Quantization-Aware Training

== Training Methodology & PTQ vs. QAT Comparison
The baseline model was first trained in IEEE-754 `FP32` for `30` epochs using the AdamW optimizer ($"lr" = 10^(-3)$, weight decay $10^(-4)$) and cosine annealing learning rate schedule, achieving *`86.56%` test accuracy* (`87.27%` validation accuracy at best Epoch `29`).

Directly applying Post-Training Quantization (PTQ) reduced test accuracy to `85.02%` (`-1.54%` drop). To recover accuracy, Method 1 employs *Quantization-Aware Training (QAT)* for `12` fine-tuning epochs using Straight-Through Estimator (STE) fake-quantization nodes on both weights and activations. QAT recovers `+0.85%` over PTQ, reaching *`85.87%` INT8 test accuracy* (`95.49%` prediction agreement with FP32):

#figure(
  table(
    columns: (1.3fr, 1fr, 1fr, 1.2fr, 1fr, 1.1fr),
    inset: 6.5pt,
    align: (left, center, center, center, center, center),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Dataset Split]],
    [#text(fill: white, weight: "bold")[FP32 Baseline]],
    [#text(fill: white, weight: "bold")[PTQ INT8]],
    [#text(fill: white, weight: "bold")[QAT INT8 (Method 1)]],
    [#text(fill: white, weight: "bold")[Delta vs. FP32]],
    [#text(fill: white, weight: "bold")[Agreement vs. FP32]],
    [Train (`54,000`)], [`87.69%`], [–], [*`86.84%`*], [`-0.85%`], [`95.65%`],
    [Validation (`6,000`)], [`87.27%`], [`85.78%`], [*`86.97%`*], [`-0.30%`], [`96.00%`],
    [Test (`10,000`)], [`86.56%`], [`85.02%`], [*`85.87%`*], [*`-0.69%`*], [*`95.49%`*],
  ),
  caption: [Comparison of FP32 baseline, Post-Training Quantization (PTQ), and Method 1 Quantization-Aware Training (QAT).],
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 10pt,
    image("reports/training/training_curves.png", width: 100%),
    image("reports/quantization/int8_confusion_matrix.png", width: 100%),
  ),
  caption: [Left: 30-epoch FP32 training loss and accuracy convergence curves. Right: Method 1 INT8 QAT confusion matrix across all 10,000 test images (`85.87%` overall accuracy).],
)

== Hardware-Friendly Fixed-Point Requantization Arithmetic
To eliminate floating-point operations completely in Verilog RTL, every layer's effective floating-point scale ratio $M_"eff" = (s_"in" dot s_w) / s_"out"$ is approximated by an unsigned 16-bit integer multiplier $M$ and a bit-shift $n$:
$ q_"out" = "clip" lr(( floor(frac("acc"_"int32" times M + 2^(n-1), 2^n)), q_"min", q_"max" )) $
Similarly, the Global Average Pooling division by $169$ ($13 times 13$) avoids hardware divider circuits by multiplying the unsigned sum by $M_"gap" = 16780$ and shifting right by $n_"gap" = 19$ (since $16780 / 2^(19) approx 1 / 169 times (s_"conv2" / s_"gap")$, exact across all possible sums).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center, center, center, center),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Layer]],
    [#text(fill: white, weight: "bold")[Weight Scale $s_w$]],
    [#text(fill: white, weight: "bold")[Output Scale $s_"out"$]],
    [#text(fill: white, weight: "bold")[Multiplier $M$]],
    [#text(fill: white, weight: "bold")[Shift $n$]],
    [#text(fill: white, weight: "bold")[Observed Acc Range]],
    [#text(fill: white, weight: "bold")[Output Clamp]],
    [`input`], [–], [`7.8125e-03`], [`1`], [`1`], [`[0, 127]`], [`[0, 127]`],
    [`conv1`], [`2.6586e-01`], [`4.6935e-02`], [`23202`], [`19`], [`[-34468, +4097]`], [`ReLU [0, 127]`],
    [`conv2`], [`1.5005e-01`], [`4.3780e-01`], [`16868`], [`20`], [`[-41058, +14037]`], [`ReLU [0, 127]`],
    [`gap`], [–], [`8.0943e-02`], [`16780`], [`19`], [`[0, +6657]`], [`[0, 127]`],
    [`fc`], [`6.2918e-02`], [`4.4889e-01`], [`23793`], [`21`], [`[-13506, +4190]`], [`[-128, +127]`],
  ),
  caption: [Exact fixed-point quantization parameters and observed `INT32` accumulator dynamic ranges across all 10,000 test images.],
)

// ============================================================================
// 5. HARDWARE RTL ARCHITECTURE (METHOD 1 SEQUENTIAL DATAPATH)
// ============================================================================
= Hardware RTL Architecture (Method 1 Sequential Datapath)

== Top-Level Module Hierarchy & Master FSM
The hardware accelerator is implemented in strict, synthesizable *Verilog-2001* across seven modular RTL files inside `rtl/`:
1. *`cnn_top.v` (Master Controller & On-Chip BRAM Subsystem):* Instantiates all weight/bias ROMs (initialized via `$readmemh` / `.mif`), intermediate activation BRAMs, and a 16-state deterministic FSM (`S_IDLE → S_LOAD_IMG → S_CONV1_START/RUN/DRAIN → S_POOL1_START/RUN → S_CONV2_START/RUN/DRAIN → S_GAP_START/RUN → S_FC_START/RUN → S_ARGMAX_START/RUN → S_DONE`).
2. *`conv_engine.v` (Parameterized Sequential 2D Convolution Engine):* Reused for both `Conv1` (`1→8`, `$30 times 30$`) and `Conv2` (`8→16`, `$13 times 13$`). Uses a 7-state FSM (`S_IDLE, S_LOAD_BIAS, S_ADDR, S_READ_WAIT, S_MAC, S_REQUANT, S_DONE`) with a 1-cycle synchronous BRAM read wait state (`S_READ_WAIT`) to guarantee clean timing closure on Cyclone V M10K blocks.
3. *`relu.v` (Streaming Signed INT8 Activation):* Single-cycle registered pipeline stage clamping negative signed `INT8` values to `8'sd0`.
4. *`maxpool.v` (`$2 times 2$` Stride-2 Max Pooling):* Reads 4 spatial neighbors per output coordinate using a 6-state FSM and tracks the signed maximum.
5. *`gap.v` (Global Average Pooling via Reciprocal Multiply):* Accumulates `169` spatial values per channel and executes `(sum * 16780 + 262144) >> 19`.
6. *`fully_connected.v` (Dense `16 → 10` Classifier):* Computes 16-element dot products per class and requantizes via `(acc * 23793 + 1048576) >> 21`.
7. *`argmax.v` (Signed `INT8` Winner Selector):* Scans the 10 class scores in 14 clock cycles, using strict `>` comparison so ties deterministically select the lowest class index.

== On-Chip M10K BRAM Memory Map
A major advantage of Method 1 is that the *entire neural network*—both parameters and intermediate feature maps—fits inside *`14,258` bytes (`14.26 KB`)* of on-chip memory, utilizing less than *`3.5%`* of the Cyclone V's available M10K BRAM blocks:

#figure(
  table(
    columns: (1.2fr, 1.5fr, auto, auto, auto),
    inset: 6pt,
    align: (left, left, center, center, right),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Memory Buffer]],
    [#text(fill: white, weight: "bold")[Role in `cnn_top.v`]],
    [#text(fill: white, weight: "bold")[Word Width]],
    [#text(fill: white, weight: "bold")[Depth]],
    [#text(fill: white, weight: "bold")[Size (Bytes)]],
    [`img_ram`], [Input `$32 times 32$` image buffer], [`8-bit`], [`1,024`], [`1,024` B],
    [`conv1_relu_ram`], [Post-ReLU1 feature map (`8×30×30`)], [`8-bit`], [`7,200`], [`7,200` B],
    [`pool1_ram`], [Post-MaxPool1 feature map (`8×15×15`)], [`8-bit`], [`1,800`], [`1,800` B],
    [`conv2_relu_ram`], [Post-ReLU2 feature map (`16×13×13`)], [`8-bit`], [`2,704`], [`2,704` B],
    [`gap_ram` + `fc_score_ram`], [GAP vector (`16`) + FC scores (`10`)], [`8-bit`], [`26`], [`26` B],
    [`conv1/conv2/fc_w_rom`], [All signed `INT8` weights (`72+1152+160`)], [`8-bit`], [`1,384`], [`1,384` B],
    [`conv1/conv2/fc_b_rom`], [All signed `INT32` biases (`8+16+10`)], [`32-bit`], [`34`], [`120` B],
    [*Total On-Chip Memory*], [*Zero External SDRAM Required*], [–], [*`14,168` words*], [*`14,258` B*],
  ),
  caption: [Complete on-chip M10K BRAM/ROM storage allocation inside `cnn_top.v`.],
)

// ============================================================================
// 6. CYCLE-ACCURATE MODELSIM VERIFICATION & VISUAL RESULTS
// ============================================================================
= Cycle-Accurate ModelSim Verification & Visual Results

== Unit-Level & Full-System Verification Summary
All RTL modules were verified in *ModelSim Intel FPGA Starter Edition 18.1 (`vlog / vsim 10.5b`)* against golden test vectors generated by our bit-exact Python integer reference model (`sw/refmodel/int8_ref_model.py`).
- *Unit Testbenches (`Milestone 8`):* Six self-checking testbenches (`tb_conv_engine`, `tb_relu`, `tb_maxpool`, `tb_gap`, `tb_fully_connected`, `tb_argmax`) verified *`31,828` individual outputs* with *`0` mismatches*.
- *Top-Level Integration (`Milestone 9`, `tb_cnn_top.v`):* Inspected every internal BRAM buffer after each FSM stage on Test Image `0` (`11,731` elements total) and ran an end-to-end 10-class multi-image sweep (`Vec 0, 10, 20, ..., 90`), achieving *`0 / 11,731` internal mismatches* and *`10 / 10` (`100%`) classification parity*:

#figure(
  table(
    columns: (1.5fr, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, right, right, right, right),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Hardware Stage (`cnn_top.v`)]],
    [#text(fill: white, weight: "bold")[Output Shape]],
    [#text(fill: white, weight: "bold")[Elements Checked]],
    [#text(fill: white, weight: "bold")[Mismatches]],
    [#text(fill: white, weight: "bold")[Clock Cycles]],
    [#text(fill: white, weight: "bold")[Cycle Share]],
    [`S_LOAD_IMG` (FSM Setup)], [`1×32×32`], [`1,024`], [`0`], [`5`], [`< 0.01%`],
    [`Conv1 + ReLU1`], [`8×30×30`], [`7,200`], [`0`], [`86,404`], [`28.52%`],
    [`MaxPool1`], [`8×15×15`], [`1,800`], [`0`], [`10,803`], [`3.57%`],
    [`Conv2 + ReLU2`], [`16×13×13`], [`2,704`], [`0`], [`202,804`], [*`66.94%`*],
    [`GAP` (`÷169` Reciprocal)], [`16`], [`16`], [`0`], [`2,755`], [`0.91%`],
    [`Fully Connected` (`16→10`)], [`10`], [`10`], [`0`], [`193`], [`0.06%`],
    [`Argmax` (`10→1`)], [`1` (`4-bit`)], [`1`], [`0`], [`14`], [`< 0.01%`],
    [*Total End-to-End Inference*], [*Class `0..9`*], [*`11,731`*], [*`0 / 11,731`*], [*`302,978`*], [*`100.00%`* (`6.06 ms`)],
  ),
  caption: [Measured ModelSim cycle-by-cycle latency breakdown and bit-exact verification across all hardware stages (`50 MHz` clock = `6.060 ms` per image, or `165.0 FPS`).],
)

#figure(
  image("reports/simulation/modelsim_waveform.jpg", width: 100%),
  caption: [Actual ModelSim Intel FPGA Starter Edition 10.5b waveform capture (`sim/do/easy_view.do`) showing the final microseconds of inference (`6,056 us` to `6,060 us`): the 10 class scores (`+7, -28, -5, -2, -19, -25, -2, -36, -7, -14`) are written sequentially into `fc_score_ram[0..9]`, followed by `argmax` asserting `predicted_class = 0` (`T-shirt/top`) and `done = 1`.],
)

#figure(
  image("reports/simulation/fpga_cnn_pipeline_visual.png", width: 98%),
  caption: [Visualized internal FPGA memory buffers (`img_ram → conv1_relu_ram → pool1_ram → conv2_relu_ram → gap_ram → fc_score_ram`) for Test Image `0`, demonstrating how the hardware extracts edges, pools spatial features, and concludes `T-shirt/top (82.4%)`.],
)

#figure(
  image("reports/simulation/fpga_10class_detections.png", width: 98%),
  caption: [Multi-image hardware verification across all 10 Fashion-MNIST categories (`Vec 0, 10, ..., 90`): `10 / 10` (`100%`) correct classifications with `0` bit mismatches between Verilog RTL and Python reference.],
)

// ============================================================================
// 7. QUARTUS PRIME SYNTHESIS & PHYSICAL DE10-STANDARD DEPLOYMENT (M10)
// ============================================================================
= Quartus Prime Synthesis & Physical DE10-Standard Deployment

To complete *Milestone 10*, the entire RTL hierarchy (`rtl/cnn_top.v` wrapped inside `rtl/de10_standard_top.v`) was synthesized, placed, routed, timing-analyzed, and programmed onto the physical *Terasic DE10-Standard* board (*Intel Cyclone V SX SoC `5CSXFC6D6F31C6`*) using *Intel Quartus Prime Lite 18.1 (`Build 625`)*.

== Cyclone V Hardware Resource Utilization (`de10_standard.fit.rpt`)
Table 5 reports the exact post-fitting hardware resource utilization on the Cyclone V `5CSXFC6D6F31C6` FPGA, separating the standalone `cnn_top` accelerator core from the top-level `de10_standard_top` board wrapper (which adds a 10-image `81,920`-bit on-chip test suite ROM `board_img_rom`, switch/button synchronizers, a 6-digit synchronous BCD cycle counter, and 7-segment display decoders):

#figure(
  table(
    columns: (1.5fr, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, right, right, right, right),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Cyclone V Resource (`5CSXFC6D6F31C6`)]],
    [#text(fill: white, weight: "bold")[`cnn_top` Core]],
    [#text(fill: white, weight: "bold")[Full Board Design]],
    [#text(fill: white, weight: "bold")[Device Available]],
    [#text(fill: white, weight: "bold")[Utilization]],
    [Logic Utilization (Adaptive Logic Modules / ALMs)], [–], [*`804`*], [`41,910`], [*`1.92%`*],
    [Combinational ALUTs], [*`1,147`*], [*`1,323`*], [`83,820`], [*`1.58%`*],
    [Dedicated Logic Registers], [*`502`*], [*`646`*], [`167,640`], [*`0.39%`*],
    [Block Memory Bits (M10K)], [*`104,272` b*], [*`186,192` b*], [`5,662,720` b], [*`3.29%`*],
    [Inferred M10K RAM Blocks], [*`21`*], [*`31`*], [`553`], [*`5.61%`*],
    [Variable-Precision DSP Blocks], [*`22`*], [*`22`*], [`112`], [*`19.64%`*],
    [I/O Pins (`CLOCK_50`, `KEY`, `SW`, `LEDR`, `HEX0..5`)], [`0`], [*`67`*], [`499`], [*`13.43%`*],
  ),
  caption: [Post-fit hardware resource utilization on the Intel Cyclone V SoC (`5CSXFC6D6F31C6`). Within `cnn_top`, the `1,147` ALUTs and `22` DSPs break down across `conv_engine` (`468` ALUTs, `164` regs, `9` DSPs), `gap` (`246` ALUTs, `88` regs, `6` DSPs), `fully_connected` (`206` ALUTs, `78` regs, `7` DSPs), `maxpool` (`139` ALUTs, `74` regs), `argmax` (`41` ALUTs, `29` regs), and `relu` (`7` ALUTs, `21` regs).],
)

== TimeQuest Static Timing Analysis & Real-Board Verification
TimeQuest Static Timing Analysis (`de10_standard.sta.rpt`) confirmed zero setup, hold, or pulse-width violations across all temperature/voltage corners against the `50.00 MHz` (`20.000 ns`) `CLOCK_50` constraint:
- *Slow 1100mV 85°C Model:* *$F_"max" = 67.82" MHz"$* (Setup Slack = *$+5.254" ns"$*, Hold Slack = *$+0.326" ns"$*).
- *Slow 1100mV 0°C Model:* *$F_"max" = 66.92" MHz"$* (Setup Slack = *$+5.057" ns"$*, Hold Slack = *$+0.305" ns"$*).
- *Hardware Inference Speed:* At the on-board `50.00 MHz` crystal oscillator, each image executes in exactly *$302,978 / (50 times 10^6) = 6.060" ms"$ ($165.0" FPS"$)*; at $F_"max" = 67.82" MHz"$, latency drops to *$4.467" ms"$ ($223.8" FPS"$)*!

The compiled bitstream (`quartus/output_files/de10_standard.sof`, checksum `0x015EDB97`) was programmed directly onto the physical *Terasic DE10-Standard (`DE-SoC [USB-1]`, JTAG ID `0x02D020DD`)* via `quartus_pgm`. On the physical board:
- Setting `SW[3:0]` to any class `0..9` loads that class's `$32 times 32$` test image from M10K ROM and displays `# - P A S #` on `HEX5..HEX0` while lighting the corresponding one-hot LED `LEDR[0..9]`.
- Flipping `SW[9] = 1` switches `HEX5..HEX0` to the live hardware BCD cycle counter, displaying *`3 0 2 9 7 8`* on the six 7-segment displays—matching our ModelSim simulation to the single clock cycle.

== Practical Step-by-Step Guide: DE10-Standard Switches, Displays & Quartus Prime

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, 1.3fr, 1.2fr, auto, auto),
    inset: 5pt,
    align: (center, center, center, center, center, left, center, center, center),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Pic \#]],
    [#text(fill: white, weight: "bold")[`SW3`]],
    [#text(fill: white, weight: "bold")[`SW2`]],
    [#text(fill: white, weight: "bold")[`SW1`]],
    [#text(fill: white, weight: "bold")[`SW0`]],
    [#text(fill: white, weight: "bold")[Clothing Image Loaded]],
    [#text(fill: white, weight: "bold")[`HEX5..HEX0` Output]],
    [#text(fill: white, weight: "bold")[Lit LED]],
    [#text(fill: white, weight: "bold")[GUI Vec]],
    [*`0`*], [▼ 0], [▼ 0], [▼ 0], [▼ 0], [*T-shirt/top* (Class `0`)], [*`0 - P A S 0`*], [`LEDR0`], [`Vec 0`],
    [*`1`*], [▼ 0], [▼ 0], [▼ 0], [*▲ 1*], [*Trouser* (Class `1`)], [*`1 - P A S 1`*], [`LEDR1`], [`Vec 10`],
    [*`2`*], [▼ 0], [▼ 0], [*▲ 1*], [▼ 0], [*Pullover* (Class `2`)], [*`2 - P A S 2`*], [`LEDR2`], [`Vec 20`],
    [*`3`*], [▼ 0], [▼ 0], [*▲ 1*], [*▲ 1*], [*Dress* (Class `3`)], [*`3 - P A S 3`*], [`LEDR3`], [`Vec 30`],
    [*`4`*], [▼ 0], [*▲ 1*], [▼ 0], [▼ 0], [*Coat* (Class `4`)], [*`4 - P A S 4`*], [`LEDR4`], [`Vec 40`],
    [*`5`*], [▼ 0], [*▲ 1*], [▼ 0], [*▲ 1*], [*Sandal* (Class `5`)], [*`5 - P A S 5`*], [`LEDR5`], [`Vec 50`],
    [*`6`*], [▼ 0], [*▲ 1*], [*▲ 1*], [▼ 0], [*Shirt* (Class `6`)], [*`6 - P A S 6`*], [`LEDR6`], [`Vec 60`],
    [*`7`*], [▼ 0], [*▲ 1*], [*▲ 1*], [*▲ 1*], [*Sneaker* (Class `7`)], [*`7 - P A S 7`*], [`LEDR7`], [`Vec 70`],
    [*`8`*], [*▲ 1*], [▼ 0], [▼ 0], [▼ 0], [*Bag* (Class `8`)], [*`8 - P A S 8`*], [`LEDR8`], [`Vec 80`],
    [*`9`*], [*▲ 1*], [▼ 0], [▼ 0], [*▲ 1*], [*Ankle boot* (Class `9`)], [*`9 - P A S 9`*], [`LEDR9`], [`Vec 90`],
  ),
  caption: [Complete switch cheat sheet (`▼ DOWN = 0`, `▲ UP = 1`) for testing all 10 clothing classes on the physical DE10-Standard board. Flipping `SW9` (far-left switch) UP displays the live clock cycle count `302978` on `HEX5..HEX0`.],
)

- *How to Read `HEX5..HEX0` (Left to Right):* `HEX5` shows your selected picture (`0..9`), `HEX4` shows `-`, `HEX3..HEX1` shows `P A S` (`PASS`), and `HEX0` shows the FPGA's predicted class (`0..9`).
- *How to Re-Program the Board Anytime:* Run `C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe -c "DE-SoC [USB-1]" -m JTAG -o "p;quartus/output_files/de10_standard.sof@2"`, or open `quartus/de10_standard.qpf` in Quartus Prime 18.1 (`Tools → Programmer → Start`).
- *How to View Reports in Quartus GUI:* Open `quartus/de10_standard.qpf`, click `Processing → Compilation Report`, and inspect `Fitter → Summary` (`804 ALMs`, `31 M10Ks`, `22 DSPs`) and `Timing Analyzer → Slow 1100mV 85C Model → Fmax Summary` (`67.82 MHz`).

// ============================================================================
// 8. BOTTLENECK ANALYSIS OF METHOD 1 & ROADMAP FOR METHOD 2
// ============================================================================
= Bottleneck Analysis of Method 1 & Roadmap for Method 2

While *Method 1* achieves `100%` bit-exact hardware verification and fits in only `14.26 KB` of BRAM, our experimental measurements expose two clear opportunities for improvement that motivate *Method 2*:

== Hardware Latency Bottleneck (`Conv2` + `Conv1` = `95.46%` of Cycles)
As shown in Table 4, `Conv2 + ReLU2` alone consumes *`202,804` cycles (`66.94%`)* and `Conv1 + ReLU1` consumes *`86,404` cycles (`28.52%`)*. Why?
- In Method 1's `conv_engine.v`, each single MAC operation requires *`1` address setup cycle (`S_ADDR`)* + *`1` BRAM read wait cycle (`S_READ_WAIT`)* + *`1` multiply-accumulate cycle (`S_MAC`)* = *`3` clock cycles per tap* (or ~`1.17` cycles/MAC amortized in `cnn_top.v`).
- Furthermore, Method 1 computes only *`1` MAC at a time*, performing $194,688$ sequential inner-loop iterations for `Conv2`.

== Algorithmic Accuracy Bottleneck (`85.87%` INT8 Test Accuracy)
Inspecting the per-class test accuracy in our quantization report reveals that while `Trouser` (`95.50%`), `Sandal` (`95.60%`), `Sneaker` (`94.70%`), and `Bag` (`95.70%`) exceed `94%`, upper-body garments with overlapping `$32 times 32$` silhouettes experience confusion: `Shirt` (`64.80%`), `Coat` (`74.10%`), and `Pullover` (`74.90%`). This stems from three constraints in Method 1:
1. *Narrow Filter Capacity (`8` and `16` channels) & Aggressive `13×13 → 1×1` GAP Compression:* Averaging an entire `$13 times 13$` feature map into a single scalar discards spatial localization (e.g., whether buttons/collars are at the top or middle of the garment).
2. *Per-Tensor Quantization Scale:* Using a single scale factor $s_w$ across all `16` filters of `Conv2` clips low-magnitude filters to fewer quantization levels.

== Summary Comparison: Method 1 (Baseline) vs. Proposed Method 2
#figure(
  table(
    columns: (1.1fr, 1.5fr, 1.8fr),
    inset: 6.5pt,
    align: (left, left, left),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Design Dimension]],
    [#text(fill: white, weight: "bold")[Method 1 (Current Baseline)]],
    [#text(fill: white, weight: "bold")[Method 2 (Proposed Improvement)]],
    [*Convolution Datapath*],
    [Sequential Single-MAC engine (`1` tap per cycle; `302,978` cycles/image)],
    [*`$3 times 3$` Sliding-Window Line-Buffer + 9-MAC Parallel Array* (`9×` spatial parallelism + pipelined 1-cycle/window throughput)],
    [*Channel Parallelism*],
    [`1` output channel computed at a time],
    [*Output-Channel Unrolling ($P_o = 4$ or $8$)* using parallel Cyclone V DSP blocks (`< 15,000` cycles/image, $> 20 times$ speedup)],
    [*Quantization Scheme*],
    [Per-tensor symmetric `INT8` QAT],
    [*Per-Channel Dyadic INT8 Quantization + BatchNorm Folding* for higher precision per filter],
    [*Feature Representation*],
    [2 Conv layers (`8 → 16`) + `13×13` GAP (`85.87%` INT8 accuracy)],
    [*Enhanced Feature Extractor* (BatchNorm + Data Augmentation / Strided Conv or `$2 times 2$` Spatial Pyramid Pooling to preserve top-vs-bottom garment structure, targeting `> 89%` accuracy)],
  ),
  caption: [Architectural roadmap from Method 1 (Baseline Sequential Single-MAC) to Method 2 (High-Throughput Parallel & Accuracy-Enhanced Architecture).],
)
