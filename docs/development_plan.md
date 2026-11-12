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
| 5 | `.mem` / `.mif` export | `sw/export/*`, `mem/*`, `docs/memory_map.md`, `reports/export/` | every exported value read back and compared bit-exactly; 6/6 tests pass | **Done, awaiting approval** |
| 6 | Integer reference model | `sw/refmodel/*`, golden vectors | integer model accuracy matches the M4 INT8 accuracy; per-layer dumps for RTL | Not started |
| 7 | Verilog modules | `rtl/*.v` | lint/compile clean in ModelSim and Quartus Analysis & Synthesis | Not started |
| 8 | Unit testbenches | `sim/tb/tb_*.v`, `sim/do/*.do` | every module matches the golden vectors, 0 mismatches | Not started |
| 9 | Full CNN integration | `rtl/cnn_top.v`, `tb_cnn_top.v` | bit-exact intermediate tensors and class agreement on all test vectors; cycle count measured | Not started |
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
