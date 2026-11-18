#set document(
  title: "100 Lecturer Defense Questions & Answers (Easy to Hard) — Fashion-MNIST FPGA CNN Accelerator",
  author: "Ersha Anandita",
)

#set page(
  paper: "a4",
  margin: (x: 18mm, y: 18mm),
  numbering: "1 / 1",
)
#set text(font: ("Calibri", "Arial", "Times New Roman"), size: 9.5pt)
#set heading(numbering: "1.1.")
#set par(justify: true, leading: 0.55em)

#align(center)[
  #text(size: 16.5pt, weight: "bold", fill: rgb("#1e3a8a"))[
    100 Lecturer Defense Questions & Answers (Easy to Hard)
  ]
  #v(3pt)
  #text(size: 10pt, fill: luma(70))[
    Two-Layer INT8 CNN Image Classifier on Terasic DE10-Standard (`5CSXFC6D6F31C6`) — Fashion-MNIST (`32×32`)
  ]
  #v(5pt)
  #line(length: 100%, stroke: 0.7pt + rgb("#1e3a8a"))
]

#let qa(num, q, a) = block(breakable: false, below: 6.5pt)[
  *Q#num. #q* \
  #text(fill: luma(35))[#emph[Answer:] #a]
]

= Part 1: Easy (Q1 – Q25)

#qa(1, [What is the main objective of this project?],
  [To design, train, quantize, verify, and deploy a real-time, integer-only Two-Layer Convolutional Neural Network (CNN) image classifier for Fashion-MNIST on an Intel Cyclone V SoC FPGA (Terasic DE10-Standard) using 100% on-chip M10K BRAM, with live VGA monitor and Python GUI visualization.])

#qa(2, [What FPGA board and device part number are used?],
  [The Terasic DE10-Standard development board featuring the Intel Cyclone V SX SoC (`5CSXFC6D6F31C6`), which provides `41,910` ALMs, `553` M10K BRAM blocks, `112` Variable-Precision DSP blocks, and a Dual-Core ARM Cortex-A9 HPS.])

#qa(3, [What dataset is used in this project, and what is its raw image size?],
  [The Fashion-MNIST dataset, consisting of `28 × 28` grayscale clothing images (`uint8` pixels in `[0, 255]`) across `10` fashion categories.])

#qa(4, [List all 10 clothing classes in Fashion-MNIST.],
  [`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`.])

#qa(5, [How is the Fashion-MNIST dataset split in this project?],
  [`54,000` training images, `6,000` stratified validation images (using deterministic `SEED = 42`), and `10,000` official test images, plus a balanced `100`-image hardware test vector subset (`10` images per class).])

#qa(6, [What input resolution is fed into the FPGA CNN, and why?],
  [`32 × 32 × 1` (`1,024` pixels), created by adding `2` pixels of symmetric zero-padding around each `28 × 28` image. This satisfies the LSI Design Contest 2027 `32×32` requirement and fits inside a single `1024 × 8-bit` M10K BRAM block.])

#qa(7, [What clock frequency does the FPGA accelerator run at on the DE10-Standard?],
  [`50.00 MHz` (`20.000 ns` period), driven by the on-board `CLOCK_50` crystal oscillator (`PIN_AF14`).])

#qa(8, [What does `4 - P A S 4` on the board's 7-segment displays (`HEX5..HEX0`) mean?],
  [`HEX5 = 4` is the selected test image's true class (`Coat`), `HEX4 = -` is a separator, `HEX3..HEX1 = P A S` means PASS (correct prediction, shortened to 3 letters), and `HEX0 = 4` is the FPGA CNN's predicted class.])

#qa(9, [What do the 10 red LEDs (`LEDR[9:0]`) indicate?],
  [They form a one-hot indicator of `predicted_class`: if the FPGA predicts Class `0` (`T-shirt/top`), `LEDR[0]` lights up; if it predicts Class `7` (`Sneaker`), `LEDR[7]` lights up.])

#qa(10, [What do the slide switches `SW[3:0]` and `SW[9]` do on the DE10-Standard?],
  [`SW[3:0]` selects which of the `10` on-chip ROM test images (`0..9`) to load and classify in binary (`0000` to `1001`). Flipping `SW[9]` UP (`1`) switches `HEX5..HEX0` to show the live 6-digit decimal clock cycle count (`3 0 2 9 7 8`).])

#qa(11, [What do the 4 pushbuttons `KEY[3:0]` do on the physical board?],
  [`KEY[0]` is Hardware Reset (active-low), `KEY[1]` re-runs CNN inference on the current image, `KEY[2]` triggers the 16-bit hardware LFSR Randomizer, and `KEY[3]` clears the `32×32` drawing canvas to black.])

#qa(12, [How many total parameters does our Two-Layer CNN have?],
  [`1,418` total parameters: `1,384` `INT8` weights (`72` Conv1 + `1,152` Conv2 + `160` FC) and `34` `INT32` biases (`8` Conv1 + `16` Conv2 + `10` FC), totaling `1,520` bytes.])

#qa(13, [What are the layers of our CNN in order from input to output?],
  [`Input (1×32×32)` $arrow.r$ `Conv1 (3×3, 1→8) + ReLU1` $arrow.r$ `MaxPool1 (2×2, stride 2)` $arrow.r$ `Conv2 (3×3, 8→16) + ReLU2` $arrow.r$ `GAP (13×13 → 16)` $arrow.r$ `FC (16→10)` $arrow.r$ `Argmax (0..9)`.])

#qa(14, [What HDL standard and simulator were used for RTL verification?],
  [Synthesizable Verilog-2001 (`rtl/*.v`), simulated in ModelSim Intel FPGA Starter Edition 18.1 (`vlog / vsim 10.5b`).])

#qa(15, [What synthesis tool was used to generate the FPGA bitstream (`.sof`)?],
  [Intel Quartus Prime Lite Edition 18.1 (`Build 625`), using `quartus_map`, `quartus_fit`, `quartus_asm`, `quartus_sta`, and `quartus_pgm`.])

#qa(16, [Does the FPGA accelerator require external SDRAM or DDR3 memory?],
  [No. 100% of weights, biases, input pixels, and intermediate feature maps fit inside the Cyclone V's on-chip M10K BRAM and MLAB memory (`104,272` bits = `13.03 KB` for `cnn_top`).])

#qa(17, [What is the final `INT8` test accuracy of the model on all 10,000 test images?],
  [`85.87%` using Quantization-Aware Training (QAT), which is only `0.69%` lower than the `86.56%` FP32 baseline.])

#qa(18, [How long does the FPGA take to classify one `32×32` image at `50 MHz`?],
  [`302,978` clock cycles, which equals `6.060 ms` per image (`165.0` frames per second) at `50.00 MHz`.])

#qa(19, [What is the purpose of the ReLU activation layer (`relu.v`)?],
  [Rectified Linear Unit clamps all negative signed `INT8` values to `0` (`max(0, x)`), introducing non-linearity while keeping post-ReLU activations in `[0, 127]`.])

#qa(20, [What does MaxPool (`maxpool.v`) do to the feature map size?],
  [It takes the maximum value in each non-overlapping `2×2` window (stride `2`), shrinking the spatial dimensions from `8 × 30 × 30` (`7,200` elements) down to `8 × 15 × 15` (`1,800` elements), a `4×` spatial reduction.])

#qa(21, [What is Global Average Pooling (`GAP`), and why did we use it?],
  [`GAP` averages each of the `16` `13×13` feature maps (`169` pixels each) into a single number (`16` values total). This replaces a huge `Flatten(2,704) → FC(10)` layer (`27,040` weights) with `16 → 10` (`160` weights).])

#qa(22, [What does the `Argmax` module (`argmax.v`) output?],
  [It scans the `10` signed `INT8` class scores (`fc_score_ram[0..9]`) and outputs the `4-bit` index (`0..9`) of the highest score as `predicted_class`.])

#qa(23, [What resolution does the DE10-Standard's VGA output (`vga_cnn_hud.v`) render?],
  [`640 × 480 @ 60 Hz`, displaying the `32×32` input image zoomed `8×` (`256×256`), a 10-class horizontal score bar chart, and a `WINNER` banner.])

#qa(24, [How does the Python Desktop GUI (`sw/gui_app.py`) talk to the physical FPGA board?],
  [Through a 2-way USB-JTAG bridge (`sw/jtag_bridge.py` using `quartus_stp.exe` and `altsource_probe` `"CNN0"`) over the `USB-Blaster II` cable (`DE-SoC [USB-1]`).])

#qa(25, [What happens when the DE10-Standard board is turned off and back on?],
  [Because Cyclone V FPGA configuration SRAM is volatile, the board forgets the `.sof` bitstream on power-off until re-programmed via `quartus_pgm` or the `⚡ Flash FPGA (.sof)` button in the GUI.])

= Part 2: Easy to Mid (Q26 – Q50)

#qa(26, [How is a raw `uint8` pixel `[0, 255]` quantized into signed `INT8` `[0, 127]` without any logic gates?],
  [Using `q_in = pixel >> 1` (a 1-bit right shift, discarding the LSB), which corresponds to an exact dyadic input scale $s_"in" = 2^(-7) = 1/128 = 0.0078125$ and zero-point `0`. In Verilog, this is pure wiring (`{1'b0, pixel[7:1]}`).])

#qa(27, [Show the exact output dimension formula and verify `Conv1`, `MaxPool1`, and `Conv2`.],
  [Using $H_"out" = floor((H_"in" - K + 2P)/S) + 1$: Conv1 (`32, K=3, P=0, S=1`) gives $(32-3)/1+1 = 30$ (`8×30×30`); MaxPool1 (`30, K=2, P=0, S=2`) gives $(30-2)/2+1 = 15$ (`8×15×15`); Conv2 (`15, K=3, P=0, S=1`) gives $(15-3)/1+1 = 13$ (`16×13×13`).])

#qa(28, [How are the `259,648` total MACs per image calculated?],
  [Conv1 performs `7,200 × (1 × 3 × 3) = 64,800` MACs; Conv2 performs `2,704 × (8 × 3 × 3) = 194,688` MACs; FC performs `10 × 16 = 160` MACs. Total = `64,800 + 194,688 + 160 = 259,648` MACs.])

#qa(29, [What hyperparameters were used to train the FP32 baseline model (Milestone 3)?],
  [`30` epochs using `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), Cross-Entropy Loss, `batch_size = 128`, and `CosineAnnealingLR`, reaching `87.69%` train, `87.27%` val, and `86.56%` test accuracy (best epoch `29`).])

#qa(30, [Why did we run Quantization-Aware Training (QAT) after Post-Training Quantization (PTQ)?],
  [Direct PTQ dropped test accuracy from `86.56%` to `85.02%` (`-1.54%`). Fine-tuning for `12` epochs with Straight-Through Estimator (STE) fake-quantization (QAT) recovered `+0.85%`, reaching `85.87%` INT8 test accuracy (`95.49%` agreement with FP32).])

#qa(31, [What is the difference between `.mem` and `.mif` files exported in Milestone 5?],
  [`.mem` files are plain hexadecimal text files read by Verilog `$readmemh` in ModelSim and Quartus RAM initialization; `.mif` (Memory Initialization File) files include explicit `DEPTH`, `WIDTH`, and radix headers for Quartus Prime IP/BRAM tools. Both store identical 2's-complement data.])

#qa(32, [How are negative `INT8` weights and `INT32` biases formatted in `.mem` files?],
  [In two's-complement hexadecimal: `val & 0xFF` (`2` hex digits) for `INT8` weights (e.g., `-1` becomes `FF`, `-128` becomes `80`) and `val & 0xFFFFFFFF` (`8` hex digits) for `INT32` biases.])

#qa(33, [Why did we build a pure-integer Python reference model in Milestone 6?],
  [To replicate the exact nested hardware loops and bit-widths of Verilog RTL in Python, verify `100%` prediction parity with Milestone 4 across all `10,000` test images, and export layer-by-layer golden `.mem` vectors (`sim/vectors/vec00_*.mem`) for self-checking RTL testbenches.])

#qa(34, [How many total Verilog-2001 modules make up the `cnn_top` accelerator?],
  [`7` modules: `conv_engine.v` (reused for Conv1 and Conv2), `relu.v`, `maxpool.v`, `gap.v`, `fully_connected.v`, `argmax.v`, and `cnn_top.v` (master FSM + BRAM controller).])

#qa(35, [Why is `conv_engine.v` instantiated only once inside `cnn_top.v`?],
  [To minimize FPGA area and DSP usage! `cnn_top.v` multiplexes `conv_engine`'s configuration ports (`cfg_in_c`, `cfg_in_h`, `cfg_in_w`, `cfg_out_c`, `cfg_out_h`, `cfg_out_w`, `cfg_mult`, `cfg_shift`) and BRAM buses between `Conv1` (`1→8`) and `Conv2` (`8→16`).])

#qa(36, [How are `fc_score_ram` integer scores converted into confidence percentages (`%`) in the GUI?],
  [Python dequantizes the `10` signed `INT8` logits by multiplying by `s_fc = 0.4488924` and applies Softmax: $P(k) = exp(q_k dot s_"fc") / sum_j exp(q_j dot s_"fc") times 100\%$. The FPGA itself only needs `Argmax` and never computes `exp()` or division.])

#qa(37, [What were the exact ModelSim unit testbench results in Milestone 8?],
  [All `6` unit testbenches (`tb_conv_engine`, `tb_relu`, `tb_maxpool`, `tb_gap`, `tb_fully_connected`, `tb_argmax`) passed with `0` mismatches across `31,828` element checks.])

#qa(38, [What were the exact ModelSim full-integration results in Milestone 9?],
  [Phase 1 (Image 0) matched all `11,731 / 11,731` internal BRAM elements (`7,200` Conv1+ReLU1 + `1,800` Pool1 + `2,704` Conv2+ReLU2 + `16` GAP + `10` FC + `1` Argmax) with `0` mismatches in `302,978` cycles. Phase 2 (10-class sweep `Vec 0, 10..90`) achieved `10 / 10` (`100%`) agreement and `0 / 100` FC score mismatches.])

#qa(39, [Which layer dominates the execution time of Method 1?],
  [`Conv2 + ReLU2` consumes `202,804` cycles (`66.94%` of total runtime), followed by `Conv1 + ReLU1` at `86,404` cycles (`28.52%`). Together, convolutions account for `95.46%` of all clock cycles.])

#qa(40, [How does `vga_cnn_hud.v` generate the `25 MHz` VGA pixel clock from `CLOCK_50`?],
  [It toggles a `1-bit` register `pix_clk` on every rising edge of `CLOCK_50` (`50 MHz ÷ 2 = 25 MHz`) and uses `if (pix_clk)` as a synchronous clock-enable inside the `CLOCK_50` domain, avoiding PLLs and clock-domain crossing bugs.])

#qa(41, [What are the `640×480 @ 60 Hz` VGA horizontal and vertical timing totals?],
  [Horizontal: `640` active + `16` front porch + `96` sync + `48` back porch = `800` pixels/line. Vertical: `480` active + `10` front porch + `2` sync + `33` back porch = `525` lines/frame. Pixel rate: `25,000,000 / (800 × 525) = 59.52 Hz`.])

#qa(42, [How does `KEY[2]` implement a hardware randomizer without software?],
  [`de10_standard_top.v` runs a `16-bit` Linear Feedback Shift Register (`lfsr`) with taps `[15, 13, 12, 10]` on `CLOCK_50`. Pressing `KEY[2]` samples `lfsr[3:0]` (clamped to `0..9`) to load a pseudo-random ROM image.])

#qa(43, [How does `sw/gui_app.py` preprocess a real Pinterest or color clothing photo?],
  [It converts RGB to luminance grayscale, samples the outer border to auto-detect light/white studio backgrounds, subtracts the background floor, crops the garment bounding box preserving aspect ratio, pads to a square, resizes to `28×28`, normalizes contrast, zero-pads to `32×32`, and shifts `>> 1` into `INT8 [0, 127]`.])

#qa(44, [Why does `Argmax` use strict `>` instead of `>=` when comparing class scores?],
  [Using strict `>` (`if (score > max_score)`) guarantees deterministic tie-breaking toward the lowest class index (`0` before `1`, etc.), matching PyTorch's `torch.argmax` behavior.])

#qa(45, [Which clothing classes have the highest and lowest INT8 test accuracy, and why?],
  [Highest: `Bag` (`95.70%`), `Sandal` (`95.60%`), `Trouser` (`95.50%`), `Sneaker` (`94.70%`), `Ankle boot` (`93.60%`). Lowest: `Shirt` (`64.80%`), `Coat` (`74.10%`), `Pullover` (`74.90%`), because upper-body garments share nearly identical `32×32` outer silhouettes.])

#qa(46, [How is the `1024`-pixel `img_ram` addressed in row-major order?],
  [`addr = {row[4:0], col[4:0]} = row * 32 + col`, which is a simple `10-bit` bit concatenation with zero multipliers.])

#qa(47, [What is the purpose of `vga_img_ram` inside `de10_standard_top.v`?],
  [It is a dual-port `1024 × 8-bit` M10K mirror of `img_ram` so `vga_cnn_hud.v` can continuously read pixels at `25 MHz` without contending for `img_ram`'s read port during `cnn_top` convolution.])

#qa(48, [How many DSP blocks are used in the synthesized design, and by which modules?],
  [`22 / 112` (`20%`) Cyclone V Variable-Precision DSP blocks total: `9` in `conv_engine`, `6` in `gap`, and `7` in `fully_connected`.])

#qa(49, [What is the post-fit ALM, register, and M10K BRAM utilization of the full board design?],
  [`1,942 / 41,910` ALMs (`5%`), `2,768` dedicated logic registers, `194,384 / 5,662,720` block memory bits (`3%`), and `32 / 553` M10K RAM blocks (`6%`), with `98 / 499` I/O pins (`20%`).])

#qa(50, [What is the Slow `1100mV 85°C` $F_"max"$ reported by TimeQuest Static Timing Analysis?],
  [`69.73 MHz` for `CLOCK_50` (setup slack `+5.659 ns` against the `50.00 MHz` constraint, zero timing violations). At `69.73 MHz`, inference latency drops to `4.345 ms` (`230.1 FPS`).])

= Part 3: Mid to Hard (Q51 – Q75)

#qa(51, [Explain the exact mathematical formula for dyadic fixed-point requantization.],
  [Given an `INT32` accumulator `acc`, input scale $s_"in"$, weight scale $s_w$, and output scale $s_"out"$, the real ratio $M_"eff" = (s_"in" dot s_w) / s_"out"$ is approximated by $M / 2^n$ where $M$ is a `16-bit` unsigned integer. The hardware computes `q_out = clip((acc * M + 2^(n-1)) >>> n, q_min, q_max)`, implementing round-half-up using only a multiplier and bit-shift.])

#qa(52, [List the exact `(M, n)` multiplier-shift pairs and accumulator bounds for every layer.],
  [Conv1: `M=23202, n=19` (obs `[-34468, +4097]`); Conv2: `M=16868, n=20` (obs `[-41058, +14037]`); GAP: `M=16780, n=19` (obs `[0, +6657]`); FC: `M=23793, n=21` (obs `[-13506, +4190]`).])

#qa(53, [How does `gap.v` divide by `169` (`13×13`) without a hardware divider circuit?],
  [It combines `÷ 169` and the scale ratio $s_"conv2" / s_"gap"$ into a single reciprocal dyadic multiplication: `(sum * 16780 + 2^18) >> 19`. Unit tests (`test_gap_reciprocal_exact_for_all_21464_sums`) proved `0` error across all `21,464` possible `gap_sum` values (`0..169×127`).])

#qa(54, [Why must intermediate scaling in `conv_engine.v` use `64-bit` signed arithmetic (`signed [63:0]`)?],
  [Because multiplying a `32-bit` signed accumulator (`±1.17e6` worst-case) by a `16-bit` unsigned multiplier `M` (`<= 23,793`) plus `2^(n-1)` can reach `~48 bits`, which would overflow `32-bit` registers before the right-shift `>>> n` brings it back into `INT8`.])

#qa(55, [Prove mathematically that `INT32` accumulators can never overflow in this CNN.],
  [In Conv2 (`72` taps/output), each `INT8 × INT8` product is at most `127 × 128 = 16,256`. Thus the worst-case sum is `72 × 16,256 = ±1,170,432` (`< 21` bits), which is `1,800×` smaller than the `INT32` limit (`±2,147,483,647`).])

#qa(56, [Walk through the 7-state FSM inside `conv_engine.v` and explain why `S_READ_WAIT` is needed.],
  [States: `S_IDLE` $arrow.r$ `S_LOAD_BIAS` $arrow.r$ `S_ADDR` $arrow.r$ `S_READ_WAIT` $arrow.r$ `S_MAC` $arrow.r$ `S_REQUANT` $arrow.r$ `S_DONE`. Cyclone V M10K BRAMs are synchronous memories with 1 clock cycle of registered read latency; `S_READ_WAIT` gives `img_ram` and `w_rom` one cycle to present valid data before `S_MAC` multiplies them.])

#qa(57, [How does `conv_engine.v` achieve ~`1` cycle per MAC in the inner loop despite BRAM read latency?],
  [During `S_MAC`, while multiplying the current `(in_rd_data * w_rd_data)`, it simultaneously issues the next `(ic, ky, kx)` address on `in_rd_addr` and `w_rd_addr`, overlapping memory fetch with MAC execution.])

#qa(58, [Describe the bug encountered in `conv_engine.v` during Milestone 8 (`S_REQUANT`) and how it was fixed.],
  [In `S_REQUANT`, `out_valid <= 1'b1;` was followed in the same `always @(posedge clk)` block by `if (out_ready) out_valid <= 1'b0;`, overriding `out_valid` to `0` on the same clock edge. Removing the redundant `out_valid <= 1'b0` inside `S_REQUANT` (letting `S_PRE`/`S_DONE` clear it) fixed `tb_conv_engine` and `tb_cnn_top` to `0` mismatches.])

#qa(59, [Explain the weight flattening order `(OC, IC, kH, kW)` in `mem/conv2_weights.mem`.],
  [Address `= ((oc * 8 + ic) * 3 + ky) * 3 + kx`, iterating `oc` (`0..15`) outermost, then `ic` (`0..7`), `ky` (`0..2`), and `kx` (`0..2`) innermost. This matches `conv_engine.v`'s sequential `w_rd_addr` increment.])

#qa(60, [Explain the exact M10K bit math for `cnn_top` (`104,272` bits / `21` M10Ks) vs. `de10_standard_top` (`194,384` bits / `32` M10Ks).],
  [`cnn_top` buffers: `img_ram` (`8,192` b, `1` M10K) + `conv1_relu_ram` (`7,200×7b = 50,400` b, `5` M10Ks) + `pool1_ram` (`14,400` b, `2` M10Ks) + `conv2_relu_ram` (`2,704×7b = 18,928` b, `2` M10Ks) + weight/bias ROMs (`12,352` b, `11` M10Ks) = `104,272` b (`21` M10Ks). Adding `board_img_rom` (`81,920` b, `10` M10Ks) + `vga_img_ram` (`8,192` b, `1` M10K) totals `194,384` b (`32` M10Ks).])

#qa(61, [Why did Quartus synthesize `conv1_relu_ram` and `conv2_relu_ram` as `7-bit` wide (`50,400` and `18,928` bits) instead of `8-bit`?],
  [Because `relu.v` clamps negative values to `0`, all post-ReLU values lie in `[0, 127]` (`7` unsigned bits), so Quartus's synthesis optimizer automatically pruned the unused MSB sign bit (saving `12.5%` BRAM bits)!])

#qa(62, [Why were `gap_ram` (`16×8b`) and `fc_score_ram` (`10×8b`) inferred as `MLAB` (`208` bits) instead of M10K?],
  [We tagged them with `(* ramstyle = "MLAB" *)` because `16` and `10` entries are tiny; using distributed LUTRAM (MLAB) avoids wasting two entire `10,240`-bit M10K blocks for `26` bytes.])

#qa(63, [Why does `conv1` + `conv2` take `289,208` cycles when `64,800 + 194,688 = 259,488` MACs?],
  [Each of the `7,200` Conv1 outputs and `2,704` Conv2 outputs incurs `3` cycles of per-pixel overhead (`S_LOAD_BIAS`, `S_ADDR`/`S_READ_WAIT` pipeline fill, and `S_REQUANT`), adding `~9,904 × 3 = 29,712` cycles (`259,488 + 29,720 = 289,208` cycles).])

#qa(64, [How does `cnn_top.v`'s `16-state` FSM handle the 1-cycle registered latency of `relu.v`?],
  [After `conv_done` asserts, `cnn_top.v` enters `S_RELU1_DRAIN` (or `S_RELU2_DRAIN`) for `1` clock cycle before starting `MaxPool` (or `GAP`), guaranteeing the final registered `relu_out_data` is written into `conv1_relu_ram` / `conv2_relu_ram`.])

#qa(65, [How does the `272-bit` Row-Burst USB-JTAG Bridge in `de10_standard_top.v` work?],
  [`altsource_probe` (`"CNN0"`) exposes a `272-bit` source register: `source[255:0]` holds `32` `INT8` pixels (1 full row), `source[260:256]` selects `row_y` (`0..31`), bit `261` toggles to burst-write all `32` pixels in `32` clocks (`0.64 µs`), bit `262` triggers a `20.5 µs` full-RAM clear, and bit `263` triggers `cnn_top`.])

#qa(66, [Why did we switch from 1-pixel JTAG writes to `272-bit` Row-Burst JTAG writes?],
  [Sending `1,024` single-pixel JTAG commands over `quartus_stp` took several seconds and queued hundreds of updates during mouse drawing. Row-bursting sends `32` pixels per JTAG transaction (`32×` faster), and hardware clear takes `1` command instead of `1,024`.])

#qa(67, [How does `sw/gui_app.py` avoid freezing Windows ("Not Responding") during USB-JTAG transfers?],
  [All blocking `quartus_stp.exe` and `quartus_pgm.exe` subprocess calls run on a dedicated background daemon thread (`_worker_loop` in `sw/jtag_bridge.py`), while the Tkinter UI thread communicates via lock-protected delta queues and a native `32×32` Canvas (`0.01 ms` redraws).])

#qa(68, [How does `ps2_mouse_ctrl.v` initialize and decode a PS/2 mouse?],
  [After reset, it holds `PS2_CLK` low for `>120 µs` (`6,500` cycles), pulls `PS2_DAT` low, and shifts out `0xF4` ("Enable Data Reporting") with odd parity. It then synchronizes on `byte0[3] == 1` to assemble 3-byte packets (`status, dX, dY`) and updates `(mouse_x, mouse_y)`.])

#qa(69, [How does `vga_cnn_hud.v` fetch `32×32` pixels from `vga_img_ram` in time for the `25 MHz` pixel stream?],
  [It computes `img_x = (next_h - 32) >> 3` using `next_h` (look-ahead horizontal counter) so the synchronous M10K BRAM read completes on the exact clock edge when `h_cnt` enters that pixel's `8×8` zoomed block.])

#qa(70, [What bug occurred in `sw/jtag_bridge.py` with `write_source_data`, and how was it diagnosed?],
  [Running `write_source_data -instance_index 0 -value_in_hex <hex>` silently failed because `quartus_stp` expects `-value <hex> -value_in_hex` (where `-value_in_hex` is a boolean flag). Diagnosed via standalone `quartus_stp -t` Tcl testing and fixed across all JTAG methods.])

#qa(71, [How does `de10_standard.sdc` constrain timing, and why are `KEY`, `SW`, `LEDR`, `HEX`, and `VGA` marked `set_false_path`?],
  [`create_clock -name CLOCK_50 -period 20.000 [get_ports {CLOCK_50}]` constrains all synchronous internal paths to `50 MHz`. Human-speed I/O (`KEY`, `SW`, `LEDR`, `HEX`) and asynchronous DAC pins (`VGA_*`) are marked `set_false_path` because inputs pass through 2-FF synchronizers and outputs have no external setup requirement relative to `CLOCK_50`.])

#qa(72, [How are `KEY` and `SW` inputs protected against metastability?],
  [Every switch and pushbutton passes through a 2-stage flip-flop synchronizer (`sw_meta -> sw_sync`, `key_meta -> key_sync`) clocked by `CLOCK_50` before driving the FSM or edge detectors.])

#qa(73, [How does the hardware `6-digit` BCD cycle counter in `de10_standard_top.v` work without binary-to-BCD division?],
  [It increments six `4-bit` decimal digits (`bcd0..bcd5`, rolling over at `9`) synchronously on `CLOCK_50` while `b_state == B_WAIT_CNN`, directly driving `HEX5..HEX0` with zero binary-to-BCD division logic.])

#qa(74, [Why does `test_m4_quant.py` check `relu_after_signed_requant_equals_fused_relu_requant`?],
  [To prove mathematically that applying ReLU after signed `INT8` requantization (`max(0, clip(round(acc * M / 2^n), -128, 127))`) produces the exact same bits as fusing ReLU into the clamp (`clip(round(acc * M / 2^n), 0, 127)`).])

#qa(75, [Why does `Quartus` report `21` M10K blocks for `cnn_top` when `104,272` bits is only `~10.2` M10Ks worth of raw bits?],
  [Because each separate Verilog memory array (`conv1_w_rom`, `conv2_w_rom`, `fc_w_rom`, `conv1_b_rom`, `conv2_b_rom`, `fc_b_rom`, `img_ram`, etc.) maps to at least `1` dedicated physical M10K block (since M10Ks cannot be shared across independent arrays with different ports/clocks), while `conv1_relu_ram` (`50,400` b) spans `5` M10Ks.])

= Part 4: Hard (Q76 – Q100)

#qa(76, [Prove why `conv1` and `conv2` zero-points are `0` in our symmetric quantization scheme and how that saves hardware.],
  [With symmetric quantization (`zero_point = 0`) for both signed `INT8` weights and post-ReLU/input activations (`[0, 127]`), the general affine MAC term $(q_x - z_x)(q_w - z_w)$ simplifies directly to $q_x dot q_w$. This eliminates `z_x * sum(q_w)` and `z_w * sum(q_x)` correction logic, saving `50%+` adder/register overhead in `conv_engine.v` and `fully_connected.v`.])

#qa(77, [Why is bias quantized with scale $s_b = s_"in" dot s_w$ into `INT32` rather than `INT8`?],
  [In integer convolution, the dot product $sum q_x q_w$ is already in scale $s_"in" dot s_w$. Quantizing the bias with $s_b = s_"in" dot s_w$ allows adding `B_int32` directly to the `INT32` accumulator *before* dyadic scaling, preserving full `32-bit` precision without needing a separate rescaling step for the bias.])

#qa(78, [Why does rounding error in `conv_engine.v` (`round-half-up` via `+ 2^(n-1)`) stay bit-exact with PyTorch `forward_int8_torch`?],
  [PyTorch's `quantize.py` explicitly models integer arithmetic with `(acc.to(int64) * M + (1 << (n - 1))) >> n` using floor division (`>> n` on two's-complement / arithmetic right shift `>>> n` in Verilog) rather than floating-point round-to-even, ensuring `100%` bit-exact parity even on half-way `.5` boundary cases.])

#qa(79, [Could `sat_int8` in `conv_engine.v` overflow if `acc_reg` is negative and `M` is `16-bit` unsigned?],
  [No, because `mult_64 = {48'd0, cfg_mult}` zero-extends `cfg_mult` into a positive `64-bit` signed value, and `$signed(acc_reg) * mult_64` performs a signed `64-bit` multiply, keeping the sign intact before `+ rnd_64` and arithmetic right-shift `>>> cfg_shift`.])

#qa(80, [What happens in `conv_engine.v` if `cfg_shift == 0`?],
  [`rnd_64 = 64'sd1 <<< (cfg_shift - 6'd1)` would underflow `cfg_shift - 1` to `6'd63`. In our design all shifts are `19..21`, so this never occurs; for general-purpose reuse, guarding `(cfg_shift == 0) ? 64'sd0 : (64'sd1 <<< (cfg_shift - 6'd1))` is recommended.])

#qa(81, [Why does `conv_engine.v` experience `~3` cycles of overhead per output pixel (`302,978` vs `259,648` MACs), and how does a Line-Buffer eliminate it?],
  [In `conv_engine.v`, each output pixel drains `S_REQUANT`, waits for `S_LOAD_BIAS`, and fills `S_ADDR` + `S_READ_WAIT`. A `3×3` 2-line shift-register Line-Buffer streams `1` input pixel/cycle continuously, exposing all `9` window pixels in parallel so a `9-MAC` adder tree produces `1` output pixel per clock cycle (`9×` to `20×` speedup).])

#qa(82, [Calculate the theoretical minimum latency of Method 2 if `conv_engine` uses a `9-MAC` Line-Buffer with `P_o = 4` output-channel unrolling.],
  [With `9` parallel MACs and `P_o = 4` (`36` DSPs), Conv1 (`8` out ch) takes `2` passes of `32×32 = 2,048` cycles; Conv2 (`8` in ch × `16` out ch) takes `8 × 4 = 32` passes of `15×15 = 7,200` cycles (or `~900` cycles if all `8` input channels are unrolled too). Thus total latency drops from `302,978` cycles (`6.06 ms`) to `< 15,000` cycles (`< 0.30 ms`).])

#qa(83, [Why did `Quartus` report `Warning (10259): constant value overflow` in `conv_engine.v(118)` and `argmax.v(50)`?],
  [In Verilog-2001, `-8'sd128` is parsed as unary minus applied to an `8-bit` signed literal `128`, which overflows `[-128, +127]` as a positive `8-bit` literal (`128` wraps to `-128` in 8-bit 2's-complement, so the synthesized hardware value `8'h80` is still `-128` and functionally exact, though `-64'sd128` / `8'sh80` silences the warning).])

#qa(84, [Explain the clock-domain / metastability risk in `de10_standard_top.v` between `altera_reserved_tck` (`30 MHz` JTAG) and `CLOCK_50` (`50 MHz`).],
  [`altsource_probe` shifts data in on `altera_reserved_tck` (`30 MHz`) and presents `jtag_source[271:0]` to `CLOCK_50`. Our design passes the control bits `jtag_source[271:256]` through a 2-FF synchronizer (`jtag_ctrl_d1 -> jtag_ctrl_d2`) and edge-detects toggle bits (`^`) so `jtag_row_buf[255:0]` is already stable when `pc_row_wr_trig` fires.])

#qa(85, [Why did we use toggle bits (`jtag_ctrl_d1[k] ^ jtag_ctrl_d2[k]`) instead of level-high pulse signals over JTAG?],
  [Because USB-JTAG writes stay latched in `jtag_source` indefinitely. A level-high signal would re-trigger on every `50 MHz` clock cycle (`50,000,000` times/sec!), whereas toggling a bit (`0→1` or `1→0`) produces exactly one clean single-cycle pulse in `CLOCK_50` per JTAG transaction.])

#qa(86, [How could `vga_cnn_hud.v` cause tearing if `dbg_fc_scores_flat` updates mid-frame?],
  [Because `cnn_top` finishes in `6.06 ms` (`< 1` VGA frame of `16.8 ms`) and updates `dbg_fc_scores_flat` across `10` consecutive FC cycles (`0.2 µs`), at most `1` scanline could straddle the update; latching `dbg_fc_scores_flat` on `v_cnt == V_ACTIVE` (vertical blanking) guarantees tear-free updates.])

#qa(87, [Why does `cnn_top.v` achieve `69.73 MHz` $F_"max"$ while using `22` DSPs? Where is the critical path?],
  [The critical path lies in the `32×16` requantization multiplier + `64-bit` rounding adder + comparator chain inside `conv_engine.v` (`S_REQUANT`) and `gap.v`. Registering the product `prod_64 <= $signed(acc_reg) * mult_64;` in an extra pipeline stage would push $F_"max"$ above `120 MHz`.])

#qa(88, [How does `cnn_top.v` prevent BRAM read/write hazards between `Conv1`, `MaxPool1`, and `Conv2`?],
  [Method 1 uses strictly non-overlapping sequential FSM phases: `Conv1+ReLU1` only writes `conv1_relu_ram` while `MaxPool1` is idle; `MaxPool1` only reads `conv1_relu_ram` and writes `pool1_ram` after `S_RELU1_DRAIN` completes. Thus zero dual-port write conflicts can ever occur.])

#qa(89, [Why is `Shirt` (`64.80%`) frequently confused with `T-shirt/top` (`80.90%`), `Pullover` (`74.90%`), and `Coat` (`74.10%`)?],
  [Two structural reasons: (1) `GAP (13×13 → 1)` averages out vertical spatial structure (losing whether a collar/button placket is present at the top-center), and (2) per-tensor quantization shares one scale `s_w` across all `16` filters of `Conv2`, clipping small-magnitude texture filters.])

#qa(90, [How would adding `BatchNorm` folding in Method 2 improve accuracy with zero extra FPGA hardware?],
  [During training, `BatchNorm` stabilizes gradients using $y = gamma (x - mu)/sigma + beta$. Before quantization, we fold it into the conv layer via $W' = (gamma/sigma) W$ and $B' = beta + (gamma/sigma)(B - mu)$, keeping the exact same `conv_engine.v` hardware at inference time.])

#qa(91, [How would `2×2` Spatial Pyramid Pooling (SPP) improve upon `1×1` GAP in Method 2?],
  [Splitting the `13×13` feature map into `4` quadrants (`top-left`, `top-right`, `bottom-left`, `bottom-right`) yields `16 × 4 = 64` features (`FC: 64 → 10`, `640` weights), preserving top-vs-bottom spatial cues (e.g., shirt collars vs. coat hems) while still fitting inside 1 M10K BRAM.])

#qa(92, [If we connect `cnn_top.v` to the DE10-Standard's ARM Cortex-A9 HPS, which AXI bridge and base address are used?],
  [The Lightweight HPS-to-FPGA AXI Bridge (`h2f_lw_axi_master`) at physical base address `0xFF200000`, accessed either bare-metal via OpenOCD (`64 KB` On-Chip RAM `0xFFFF0000`, no SD card) or under Linux via `mmap("/dev/mem")` at `0xFF200000`.])

#qa(93, [Why is OpenOCD bare-metal HPS boot harder on Windows than SD Card Linux boot?],
  [OpenOCD requires initializing the HPS DDR3 controller via the Cyclone V Preloader (`u-boot-spl`) unless the entire program fits in `64 KB` On-Chip RAM (`0xFFFF0000`), and requires `WinUSB` (`libusb`) driver swapping vs. Intel's `USB-Blaster II` driver; SD Card Linux boots DDR3/USB/Ethernet automatically.])

#qa(94, [How does `test_m1_data.py` guarantee deterministic dataset splits and zero train/test leakage?],
  [It verifies official MD5 checksums of the raw `.idx` `.gz` files, uses a fixed `np.random.default_rng(42)` stratified split on the `60,000` training set (`54,000` train / `6,000` val) without touching the official `10,000` test set, and checks `split_manifest.json` consistency.])

#qa(95, [How does `sw/gui_app.py` verify that live FPGA scores match the Python reference model?],
  [On every `250 ms` / `150 ms` poll, `update_view()` compares the `10` signed `INT8` bytes unpacked from `jtag_probe[127:48]` (`dbg_fc_scores_flat`) via `np.array_equal` against `infer_single_hw_loop`, displaying `LIVE CYCLONE V FPGA SILICON + VGA (10/10 Scores Matched)` only on exact equality.])

#qa(96, [What happens in `sw/gui_app.py` if you drag the mouse quickly across the `32×32` drawing pad?],
  [If `_on_pad_drag` only painted `(r, c)` on OS mouse events, fast movement would leave gaps. Instead, it linearly interpolates `steps = max(|r-r0|, |c-c0|)` intermediate grid cells, updates only modified `Canvas` rectangle items (`0.01 ms`), and queues only dirty row indices (`dirty_rows`) to the `272-bit` JTAG worker thread.])

#qa(97, [Why does `ps2_mouse_ctrl.v` check `rx_shift[4] == 1'b1 && rx_shift[8:1] != 8'hFA` on byte `0`?],
  [In the PS/2 mouse protocol, Byte `0` of every 3-byte movement packet always has bit `3` (`rx_shift[4]` in the `11-bit` `{stop, parity, data[7:0], start}` shift register) set to `1`, whereas `0xFA` is the mouse's `ACK` byte. Checking both prevents packet misalignment on hot-plug or noise.])

#qa(98, [Compare Method 1's hardware efficiency (`FPS / ALM` and `FPS / DSP`) against a CPU/GPU or larger FPGA design.],
  [At `50 MHz` (`165.0 FPS`), `cnn_top` uses only `~750` ALMs (`< 2%` of Cyclone V) and `22` DSPs (`20%`), achieving `0.22 FPS/ALM` and `7.5 FPS/DSP` with `0` external memory bandwidth—leaving `95%` of the Cyclone V fabric free for Method 2's parallel MAC array and HPS/camera interfaces.])

#qa(99, [If you had to port Method 1 to an ASIC or the LSI Contest `ZERO7` board, what changes are needed?],
  [Replace FPGA `$readmemh` RAM initialization with either a ROM mask / SPI weight loader, replace `altsource_probe` (Intel JTAG IP) with a standard SPI/UART/AXI slave interface, and optionally replace `DSP` `INT8×INT8` multipliers with a `9-MAC` shared booth/array or `1-bit` XNOR-Popcount (`Method 3`) to shrink silicon area.])

#qa(100, [Summarize the entire end-to-end verification chain that proves Method 1 is 100% bug-free.],
  [`PyTorch FP32 (86.56%)` $arrow.r$ `PyTorch QAT INT8 (85.87%)` $arrow.r$ `Bit-Exact .mem/.mif Export (16 files, 0 errors)` $arrow.r$ `Pure-Integer Python Ref (100% parity on 10,000 images)` $arrow.r$ `ModelSim Unit TBs (0 / 31,828 mismatches)` $arrow.r$ `ModelSim Full Integration (0 / 11,731 internal mismatches, 10/10 sweep, 302,978 cycles)` $arrow.r$ `Quartus Cyclone V Fit (1,942 ALMs, 22 DSPs, 32 M10Ks, 69.73 MHz Fmax)` $arrow.r$ `Live DE10-Standard Silicon + VGA + JTAG (58/58 pytest passed)`.])
