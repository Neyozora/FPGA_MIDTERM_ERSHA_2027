# Development Plan and Milestone Log

Every milestone ends with (1) automated tests run and passing, (2) a README
update, and (3) a status entry in this file. The next milestone starts only
after it has been approved.

## Toolchain found on the development machine (2026-11-12)

| Tool | Found | Notes |
|---|---|---|
| Python | 3.12.0 | system interpreter |
| NumPy / Matplotlib | 2.4.6 / 3.11.1 | enough for Milestone 1 |
| pytest | installed | optional; the tests also run without it |
| PyTorch | 2.11.0+cu128 (CUDA 12.8, cuDNN 91900) | installed in system Python 3.12 |
| GPU | NVIDIA RTX 5050 Laptop (Blackwell, `sm_12.0`) | verified working with PyTorch CUDA 12.8 |
| ModelSim | `C:\intelFPGA_lite\18.1\modelsim_ase` (Intel FPGA Starter Edition) | selected simulator (no license needed) |
| Questa | `C:\altera_lite\24.1std\questa_fse` | Intel FPGA Starter Edition, needs a free license file |
| Quartus Prime Lite | 18.1 and 24.1std | both support Cyclone V (`5CSXFC6D6F31C6` & `5CSEBA6U23I7`) |
| git | 2.54 | initialized on `main` branch |

## Milestones

| # | Milestone | Main deliverables | Exit criteria (all must pass) | Status |
|---|---|---|---|---|
| 1 | Dataset download, inspection, preprocessing | `sw/data/*`, `docs/preprocessing.md`, `docs/tensor_dimensions.md`, `reports/dataset/` | MD5s verified; 12/12 tests pass; split stratified and reproducible | **Done & approved** |
| 2 | PyTorch model and shape verification | `sw/model/{cnn,summary}.py`, `sw/tests/test_m2_model.py`, `reports/model/` | shape assertions at every layer; parameter count = 1,418; 14/14 tests pass | **Done & approved** |
| 3 | FP32 training and evaluation | `sw/train/*`, `checkpoints/fp32_best.pt`, `reports/training/` | loss/acc logs, confusion matrix, per-class accuracy, test accuracy = 86.56% recorded; 4/4 tests pass | **Done & approved** |
| 4 | INT8 quantization | `docs/quantization_spec.md`, `sw/quant/*`, `checkpoints/int8_quantized.npz`, `reports/quantization/` | measured INT8 (85.87% QAT, 85.02% PTQ) vs FP32 (86.56%); scales/shifts/rounding/saturation documented per layer; 7/7 tests pass | **Done & approved** |
| 5 | `.mem` / `.mif` export | `sw/export/*`, `mem/*`, `docs/memory_map.md`, `reports/export/` | every exported value read back and compared bit-exactly; 6/6 tests pass | **Done & approved** |
| 6 | Integer reference model | `sw/refmodel/*`, `sim/vectors/*`, `reports/refmodel/` | integer model accuracy matches M4 INT8 (85.87%) bit-for-bit across all 10k images; per-layer golden `.mem` dumps; 5/5 tests pass | **Done & approved** |
| 7 | Verilog modules | `rtl/*.v`, `docs/rtl_architecture.md`, `sw/tests/test_m7_rtl.py` | all 7 modules compile clean in ModelSim 18.1 (`vlog`: 0 errors, 0 warnings); 4/4 tests pass | **Done & approved** |
| 8 | Unit testbenches | `sim/tb/tb_*.v`, `sim/do/*.do`, `reports/simulation/unit_*` | all 6 unit testbenches match golden vectors in ModelSim 18.1, 0 mismatches across 31,828 checks; 2/2 pytest checks pass | **Done & approved** |
| 9 | Full CNN integration | `rtl/cnn_top.v`, `sim/tb/tb_cnn_top.v`, `sim/do/run_tb_cnn_top.do`, `reports/simulation/cnn_top_*` | bit-exact intermediate tensors (11,731/11,731) and 100% class/score agreement on multi-image sweep; 302,978 cycles/image measured; 2/2 pytest checks pass | **Done, awaiting approval** |
| 10 | Quartus + DE10-Standard | `quartus/*`, `reports/synthesis/` | resource, Fmax and critical-path reports; on-board demo | Not started |

## Planned decisions (to be confirmed at the relevant milestone)

| Topic | Proposal | Milestone |
|---|---|---|
| Quantization granularity | per-tensor symmetric weights to start with; per-channel only if accuracy needs it | 4 |
| Requantization | `y = sat8((acc · M0 + 2^(n−1)) >>> n)`, with integer multiplier M0 and shift n per layer, round half up, saturate to [−128, 127] | 4 |
| Training method | PTQ first (simple baseline), then QAT with fake-quant that uses the *same* integer rounding rules | 4 |
| ReLU placement | apply after requantization: clamp to [0, 127] | 4/7 |
| GAP division by 169 | multiply-and-shift reciprocal (e.g. `(sum·M_gap + 2^(k−1)) >> k`), or fold into FC scale; exact formula fixed in M4 and mirrored in the reference model | 4/6 |
| Argmax input | FC INT32 accumulators *or* requantized INT8 scores; tie-break = lowest index | 4/7 |
| Activation memory layout | channel-planar CHW, address = `c·H·W + y·W + x` | 5 |
| Weight memory layout | `[oc][ic][ky][kx]` flattened, explicit conversion from PyTorch OIHW | 5 |
| Conv engine | 1 MAC/cycle sequential engine, reused for conv1 and conv2 with runtime dimension registers | 7 |
| Fusions | optional: ReLU+MaxPool fused into conv1 write-back; GAP accumulation fused into conv2 write-back (saves BRAM, keeps the arithmetic identical) | 7/9 |

## Milestone log

### Milestone 1: Dataset (2026-11-12)

- Objective: reproducible download, inspection, split and INT8 conversion of
  Fashion-MNIST.
- Files: `sw/config.py`, `sw/data/{download,idx,preprocess,dataset,prepare_dataset,inspect_dataset}.py`,
  `sw/tests/test_m1_data.py`, `docs/preprocessing.md`, `docs/tensor_dimensions.md`.
- Results:
  - All 4 files downloaded from the official GitHub repo; MD5s match.
  - Split: 54,000 train / 6,000 val (600 per class) / 10,000 test.
  - 28×28 images symmetrically zero-padded with a 2-pixel border to `32×32`
    (for LSI Design Contest 2027 compatibility and exact `1,024 × 8b` M10K BRAM
    alignment).
  - INT8 range [0, 127].
  - 100 hardware test vectors (`32×32` INT8).
- Tests: `python -m sw.tests.test_m1_data`: 12 passed, 0 failed, 0 skipped.
  `pytest`: 12 passed.
- Issue fixed during the milestone: the test skip helper raised pytest's
  `Skipped` exception even when pytest was not the runner. It now checks
  `PYTEST_CURRENT_TEST`.

### Milestone 2: PyTorch Model & Tensor Shape Verification (2026-11-12)

- Objective: implement the two-layer CNN in PyTorch (`32×32` input) with runtime
  shape checks at every stage, verify parameter counts, MAC counts, model byte
  sizes, and GPU vs CPU parity on a real Fashion-MNIST batch.
- Files: `sw/model/__init__.py`, `sw/model/cnn.py`, `sw/model/summary.py`,
  `sw/tests/_runner.py`, `sw/tests/test_m2_model.py`,
  `reports/model/{model_summary.md,model_summary.json}`.
- Results:
  - All 8 tensor stages match specification (`(1,32,32)` → `(8,30,30)` →
    `(8,30,30)` → `(8,15,15)` → `(16,13,13)` → `(16,13,13)` → `(16,)` → `(10,)`).
  - Parameter count: 1,418 (1,384 weights + 34 biases).
  - Model size: 5,672 B (FP32) / 1,520 B (INT8 weights + INT32 biases).
  - MACs per image: 259,648 (Conv1: 64,800, Conv2: 194,688, FC: 160).
  - GPU check on RTX 5050 (`sm_12.0`): max `|logits_gpu - logits_cpu| = 2.235e-08`.
- Tests: `python -m sw.tests.test_m2_model`: 14 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 26 passed.

### Milestone 3: FP32 Training and Evaluation (2026-11-12)

- Objective: train the FP32 baseline CNN on the 32×32 zero-padded hardware-grid
  dataset (`x = (pixel >> 1) / 128`), log training/validation curves, save
  checkpoints, and evaluate on train, validation, test (10,000 images), and
  the 100 hardware test vectors.
- Files: `sw/train/__init__.py`, `sw/train/metrics.py`, `sw/train/train.py`,
  `sw/train/evaluate.py`, `sw/tests/test_m3_train.py`,
  `reports/training/{training_log.csv,training_history.json,eval_metrics.json,training_report.md,training_curves.png,confusion_matrix.png}`.
- Important implementation decisions:
  - Initial run with default uniform initialization + `lr=3e-3` plateaued at
    75.11% test accuracy because `conv2` inputs are pooled over 2×2 windows and
    `gap` averages over `13×13 = 169` spatial locations, attenuating early
    gradients.
  - Switching to **Kaiming normal initialization** + **OneCycleLR (`max_lr=0.01`,
    batch size 128, 120 epochs)** and preloading tensors onto the RTX 5050 GPU
    cut epoch time to ~0.8s and lifted accuracy by **+11.45 percentage points**:
    - **Train accuracy:** 87.69% (loss 0.3458)
    - **Validation accuracy:** 87.27% (loss 0.3588, best at epoch 119)
    - **Official 10k Test accuracy:** **86.56%** (loss 0.3854, 8,656 / 10,000)
    - **100 Hardware Test Vectors accuracy:** **85.0%** (85 / 100)
- Tests: `python -m sw.tests.test_m3_train`: 4 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 30 passed.

### Milestone 4: INT8 Quantization and Evaluation (2026-11-13)

- Objective: define the exact hardware fixed-point and INT8 quantization
  specification (`docs/quantization_spec.md`), implement both Post-Training
  Quantization (PTQ) and Quantization-Aware Training (QAT), and measure exact
  integer accuracy against the FP32 baseline.
- Files: `docs/quantization_spec.md`, `sw/quant/__init__.py`,
  `sw/quant/quantize.py`, `sw/quant/evaluate_quant.py`,
  `sw/tests/test_m4_quant.py`, `checkpoints/int8_quantized.npz`,
  `reports/quantization/{quantization_report.md,quant_metrics.json,int8_confusion_matrix.png,weight_distributions.png}`.
- Important implementation decisions:
  1. **Exact GAP reciprocal + dynamic-range rescaling:** Averaging over
     `13×13 = 169` reduces the peak activation magnitude from ~99 (`relu2`) to
     ~12 (`gap`). Giving GAP its own calibrated output scale `s_gap` and folding
     `s_a2 / (169 * s_gap)` into a single 16-bit multiplier + shift
     (`M_gap = 16780, n_gap = 19`) recovered nearly 3 bits of INT8 dynamic range
     entering the FC layer and lifted INT8 test accuracy from 83.97% to
     **85.87%**.
  2. **PTQ vs QAT:** PTQ alone achieved **85.02%** test accuracy (85.78% val);
     20 epochs of QAT fine-tuning with STE fake-quantization improved INT8 test
     accuracy to **85.87%** (86.97% val), within **0.69 percentage points** of
     the FP32 baseline (86.56%), with **95.49% classification agreement**
     between INT8 and FP32 across the 10,000 test images.
- Tests: `python -m sw.tests.test_m4_quant`: 7 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 37 passed.

### Milestone 5: Weight, Bias, and Test Image Export (`.mem` and `.mif`) (2026-11-13)

- Objective: export all quantized INT8 weights, INT32 biases, dyadic
  requantization parameters, and `32×32` INT8 test images into Verilog
  `$readmemh`-compatible `.mem` files and Intel Quartus `.mif` files, document
  the exact PyTorch-to-hardware address mapping, and verify every exported file
  by reading it back and comparing bit-for-bit against the source quantized
  tensors.
- Files: `docs/memory_map.md`, `sw/export/__init__.py`,
  `sw/export/formatters.py`, `sw/export/export_mem.py`,
  `sw/tests/test_m5_export.py`, `mem/*.{mem,mif}`, `mem/export_manifest.json`,
  `reports/export/export_report.md`.
- Important implementation decisions:
  1. **Clean `$readmemh` `.mem` format:** Each line contains only the uppercase
     two's-complement hexadecimal word (`2` hex digits for 8-bit `INT8`, `8` hex
     digits for 32-bit `INT32`), with no `@addr` tokens or inline comments so
     both ModelSim 18.1 `$readmemh` and Quartus Prime BRAM synthesis infer
     memories cleanly.
  2. **Intel Quartus `.mif` compliance:** Every `.mif` file includes explicit
     `WIDTH`, `DEPTH`, `ADDRESS_RADIX = HEX`, `DATA_RADIX = HEX`, and
     `CONTENT BEGIN ... END;` blocks.
  3. **Explicit PyTorch → Hardware loop ordering:** Documented and enforced
     `addr = ((oc * C_in + ic) * 3 + ky) * 3 + kx` for Conv layers,
     `addr = oc * 16 + ic` for FC weights, and `addr = (c * H + y) * W + x` for
     feature maps (`(y << 5) | x` for `32×32` input images).
- Tests: `python -m sw.tests.test_m5_export`: 6 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 43 passed.

### Milestone 6: Python Integer Reference Model & Golden Vectors (2026-11-13)

- Objective: create an independent, pure-integer reference model that loads
  parameters directly from the exported `.mem` files in `mem/`, simulates the
  exact arithmetic of every RTL module (`conv_engine`, `relu`, `maxpool`, `gap`,
  `fully_connected`, `argmax`), verifies bit-exact parity against Milestone 4
  across all 10,000 test images, and exports golden `.mem` verification vectors
  into `sim/vectors/` for ModelSim.
- Files: `sw/refmodel/__init__.py`, `sw/refmodel/int8_ref_model.py`,
  `sw/refmodel/generate_vectors.py`, `sw/tests/test_m6_refmodel.py`,
  `sim/vectors/*.mem`, `sim/vectors/vectors_manifest.json`,
  `reports/refmodel/refmodel_report.md`.
- Important implementation decisions:
  1. **Zero floating-point / PyTorch dependency:** `int8_ref_model.py` uses only
     NumPy integer dtypes (`int8`, `int32`, `int64`) and loads weights, biases,
     and `(M, n)` parameters directly from `mem/*.mem`.
  2. **Explicit hardware address loop + vectorized batch model:** Implemented
     both `infer_single_hw_loop` (nested loops using the exact 1D hardware
     memory addresses) and `infer_batch_int8` (pure-integer vectorized NumPy),
     confirming 100% bit-exact parity between them and against Milestone 4 across
     all 10,000 test images (**85.87%** accuracy, `8,587 / 10,000`).
  3. **Pre-ReLU and post-ReLU golden vectors:** Exported both `img0_conv1_req.mem` /
     `img0_conv2_req.mem` (signed `INT8` in `[-128, 127]`) and
     `img0_relu1_out.mem` / `img0_relu2_out.mem` (`INT8` in `[0, 127]`), proving
     exhaustively that `relu_int8(requantize_signed_int8(acc))` is bit-identical
     to fused ReLU-requantization.
- Tests: `python -m sw.tests.test_m6_refmodel`: 5 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 48 passed.

### Milestone 7: Individual Verilog-2001 RTL Modules (2026-11-13)

- Objective: implement all 7 required synthesizable Verilog-2001 modules
  (`conv_engine.v`, `relu.v`, `maxpool.v`, `gap.v`, `fully_connected.v`,
  `argmax.v`, `cnn_top.v`), document the microarchitecture and M10K BRAM timing
  in `docs/rtl_architecture.md`, and verify clean compilation in ModelSim 18.1.
- Files: `docs/rtl_architecture.md`, `rtl/conv_engine.v`, `rtl/relu.v`,
  `rtl/maxpool.v`, `rtl/gap.v`, `rtl/fully_connected.v`, `rtl/argmax.v`,
  `rtl/cnn_top.v`, `sw/tests/test_m7_rtl.py`.
- Important implementation decisions:
  1. **Synchronous 1-cycle M10K BRAM pipeline (`S_PRE -> S_PIPE -> S_MAC -> S_REQUANT`):**
     Every memory-reading module registers its outgoing read address (`in_rd_addr`,
     `w_rd_addr`, `b_rd_addr`) and includes an explicit 1-cycle `S_PIPE` fill
     state so synchronous M10K BRAM read data (`in_rd_data`) aligns cycle-for-cycle
     with the accumulator without combinational memory read paths.
  2. **Multiplier-free weight address stepping in `conv_engine.v`:** Because
     `conv1_weights.mem` and `conv2_weights.mem` are packed in `(oc -> ic -> ky -> kx)`
     order, the weight address for each output channel `oc` is a contiguous
     sequential range `[w_base_oc, w_base_oc + cfg_in_c*9 - 1]`, stepped via an
     incrementing pointer `w_req_ptr`.
  3. **Reusable `conv_engine.v` and `relu.v`:** Both Conv1 (`1×32×32 → 8×30×30`)
     and Conv2 (`8×15×15 → 16×13×13`) share a single `conv_engine` + `relu`
     instance in `cnn_top.v`, multiplexed via `is_conv2`.
- Verification:
  - ModelSim Intel FPGA Edition 18.1 (`vlog 10.5b`): compiled all 7 modules with
    **0 errors, 0 warnings**.
  - Tests: `python -m sw.tests.test_m7_rtl`: 4 passed, 0 failed, 0 skipped.
    Full suite (`pytest sw/tests`): 52 passed.

### Milestone 8: Individual ModelSim Unit Testbenches (2026-11-14)

- Objective: create self-checking ModelSim unit testbenches (`sim/tb/tb_*.v`)
  and `.do` scripts (`sim/do/*.do`) for each individual processing module
  (`conv_engine`, `relu`, `maxpool`, `gap`, `fully_connected`, `argmax`), run
  them in ModelSim Intel FPGA Edition 18.1 against the golden vectors from
  Milestone 6, and verify 0 mismatches.
- Files: `sim/tb/{tb_conv_engine,tb_relu,tb_maxpool,tb_gap,tb_fully_connected,tb_argmax}.v`,
  `sim/do/{run_tb_conv_engine,run_tb_relu,run_tb_maxpool,run_tb_gap,run_tb_fully_connected,run_tb_argmax,run_all_unit_tbs}.do`,
  `reports/simulation/{unit_tests_transcript.log,unit_verification_report.md}`,
  `sw/tests/test_m8_unit_sim.py`.
- Bug caught and fixed during simulation:
  - In `rtl/conv_engine.v` (`S_REQUANT`), `out_valid <= 1'b1;` was followed in
    the same clock cycle by `if (out_ready) begin out_valid <= 1'b0; ...`, which
    suppressed the `out_valid` strobe when `out_ready == 1'b1`. Removing the
    redundant `out_valid <= 1'b0` in `S_REQUANT` resolved the issue immediately.
- Measured ModelSim 18.1 Unit Results:
  - `tb_conv_engine`: Conv1 = `7,200` outputs (`86,401` cycles), Conv2 = `2,704`
    outputs (`202,801` cycles), **0 mismatches** on both `INT32` accumulators
    and `INT8` requantized outputs.
  - `tb_relu`: `10,160` checks (`256` exhaustive `[-128..+127]` + `7,200` ReLU1
    + `2,704` ReLU2, `10,160` cycles), **0 mismatches**.
  - `tb_maxpool`: `1,804` checks (`4` directed signed negative/zero/positive +
    `1,800` Pool1 outputs, `10,801` cycles), **0 mismatches**.
  - `tb_gap`: `16` channels (`2,753` cycles), **0 mismatches** on both `INT32`
    sums and `INT8` outputs.
  - `tb_fully_connected`: `10` classes (`191` cycles), **0 mismatches** on both
    `INT32` accumulators and `INT8` scores.
  - `tb_argmax`: `4` test cases (`13` cycles/run), **0 mismatches** (golden
    scores, tie-breaking to lowest index, all-negative scores, class-9 winner).
- Tests: `python -m sw.tests.test_m8_unit_sim`: 2 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 54 passed.

### Milestone 9: Complete CNN Integration & ModelSim Verification (2026-11-14)

- Objective: verify the complete integrated `cnn_top.v` accelerator in ModelSim
  Intel FPGA Edition 18.1 across all intermediate on-chip M10K BRAM buffers and
  across a multi-image 10-class sweep, comparing every output against the
  Python integer reference model and measuring exact inference latency in clock
  cycles.
- Files: `sim/tb/tb_cnn_top.v`, `sim/do/run_tb_cnn_top.do`,
  `reports/simulation/{cnn_top_transcript.log,cnn_top_verification_report.md}`,
  `sw/tests/test_m9_cnn_top_sim.py`.
- Measured ModelSim 18.1 Integration Results:
  1. **Phase 1 (Image 0 Layer-by-Layer Buffer Verification):**
     - `conv1_relu_ram` (`7,200 × 8b`): `0` mismatches (`86,404` cycles)
     - `pool1_ram` (`1,800 × 8b`): `0` mismatches (`10,803` cycles)
     - `conv2_relu_ram` (`2,704 × 8b`): `0` mismatches (`202,804` cycles)
     - `gap_ram` (`16 × 8b`): `0` mismatches (`2,755` cycles)
     - `fc_score_ram` (`10 × 8b`): `0` mismatches (`193` cycles)
     - `predicted_class` (`1 × 4b`): `exp = 0, got = 0` (`14` cycles)
     - **Total Image 0 Intermediate Elements Verified:** `11,731 / 11,731` (`0` mismatches, `Max Abs Error = 0`)
     - **Measured End-to-End Inference Latency:** **`302,978` clock cycles** (`6.060 ms` @ 50 MHz)
  2. **Phase 2 (10-Class Multi-Image End-to-End Sweep, `Vec 0, 10, ..., 90`):**
     - **Classification Agreement vs Python Ref:** `10 / 10` (`100.0%`)
     - **FC Class Score Mismatches:** `0 / 100` (`Max Abs Error = 0`)
- Tests: `python -m sw.tests.test_m9_cnn_top_sim`: 2 passed, 0 failed, 0 skipped.
  Full suite (`pytest sw/tests`): 56 passed.
