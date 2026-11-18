# Ready-to-Paste Prompts for Your Next Session

Copy and paste **one** of these prompts into a brand-new AI session in this workspace (`c:\Users\Ersha Anandita\Downloads\fpga_midterm`) whenever you are ready to build **Method 2**, switch clothing datasets, or connect the **Cyclone V HPS**!

---

## Prompt 1: Build Method 2 on CIFAR-10 / CIFAR-100 (Parallel 9-MAC Line-Buffer + 4-Layer CNN for LSI 2027 Hokkaido)

> Copy & paste everything inside the quote block below:

```text
We are upgrading our FPGA CNN Image Classification project from Method 1 (Fashion-MNIST baseline) to **Method 2** using **CIFAR-10** (and optionally **CIFAR-100**) for the **30th LSI Design Contest in Hokkaido (2027)** (`Class 1: Image Classification Class`, report deadline Jan 31, 2027), which allows **any number of CNN layers**.

First, read these files to understand our verified Method 1 baseline and our CIFAR-10 + Method 2 plan:
- `ANOTHER_METHODS/00_LSI_2027_HOKKAIDO_RULES_AND_STRATEGY.md`
- `ANOTHER_METHODS/01_CIFAR10_CIFAR100_DATASETS.md`
- `ANOTHER_METHODS/02_HARDWARE_METHODS_FOR_LSI2027.md`
- `reports/synthesis/synthesis_report.md`

Now let's build **Method 2** (keeping Method 1 untouched for side-by-side comparison):
1. Download & preprocess **CIFAR-10 (`32×32×3` RGB)** (`45k` train / `5k` val / `10k` test, `q_in = pixel >> 1` or MSB-flipped `[-128, 127]`) with `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` augmentation.
2. Train a compact **3- or 4-layer CNN** with **BatchNorm folding**, **Per-Channel Symmetric INT8 QAT**, and **GAP** targeting **`> 86%` INT8 test accuracy on CIFAR-10** while fitting weights and feature maps inside Cyclone V M10K BRAM.
3. Upgrade the Verilog-2001 RTL (`conv_engine`) to a **`3×3` 2-Line Shift-Register Line-Buffer + 9-MAC Parallel Array** (`Po = 4` channel unrolling) for high-FPS inference.
4. Verify bit-exact parity in ModelSim 18.1, synthesize in Quartus Prime 18.1 for Terasic DE10-Standard (`5CSXFC6D6F31C6`), update the VGA HUD (`rtl/vga_cnn_hud.v`) and Python GUI (`sw/gui_app.py`) to display `32×32` **color RGB** CIFAR-10 images, and generate a Method 1 vs. Method 2 comparison table!
```

---

## Prompt 2: Switch Dataset to CIFAR-10 / CIFAR-100 (`32×32×3` RGB) Only

> Copy & paste everything inside the quote block below:

```text
Let's switch our dataset pipeline from Fashion-MNIST to **CIFAR-10** (and/or **CIFAR-100**, natively `32×32×3` RGB) for the LSI Design Contest 2027.

First, read:
- `ANOTHER_METHODS/01_CIFAR10_CIFAR100_DATASETS.md`
- `sw/data/preprocess.py`
- `sw/gui_app.py`

Tasks:
1. Download CIFAR-10 (`torchvision.datasets.CIFAR10`) and create a deterministic `45,000` train / `5,000` val / `10,000` test split + `100` balanced hardware test vectors (`10` per class: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
2. Update the model input from `1` channel to `3` RGB channels, add data augmentation + BatchNorm folding + Per-Channel INT8 QAT, and re-export `.mem`/`.mif` files.
3. Update the FPGA BRAM (`img_ram` -> `3,072` bytes), VGA HUD (`vga_img_ram` -> RGB color display!), and `sw/gui_app.py` so I can test CIFAR-10 images and real-world color photos live on my DE10-Standard board!
```

---

## Prompt 3: Add Cyclone V HPS (`ARM Cortex-A9`) via OpenOCD (No SD Card) or SD Card Linux

> Copy & paste everything inside the quote block below:

```text
I want to connect our FPGA CNN accelerator (`rtl/cnn_top.v`) to the **Cyclone V HPS Dual-Core ARM Cortex-A9** (`4BA00477 SOCVHPS`) on my Terasic DE10-Standard board so the ARM CPU feeds images into the FPGA over the Lightweight AXI Bridge (`0xFF200000`).

First, read:
- `ANOTHER_METHODS/03_HPS_OPENOCD_VS_SD_CARD.md`
- `ANOTHER_METHODS/hps_openocd_starter/openocd_de10_standard.cfg`
- `ANOTHER_METHODS/hps_openocd_starter/hps_cnn_bridge.c`
- `rtl/cnn_top.v` and `rtl/de10_standard_top.v`

Tasks:
1. Wrap `cnn_top.v` in an Avalon-MM Slave register interface and generate a Quartus Platform Designer (Qsys) system connecting `hps_0.h2f_lw_axi_master` (`0xFF200000`) to `cnn_top` + our `vga_cnn_hud`.
2. Set up the HPS C software using **OpenOCD (`0xFFFF0000` On-Chip RAM, no SD card)** OR **SD Card Linux (`/dev/mem`)**, compile it, and verify live classification on my DE10-Standard hardware!
```

---

## Prompt 4: Generate Final Typst Paper & Slides for LSI Design Contest 2027 Hokkaido

> Copy & paste everything inside the quote block below:

```text
Please generate a complete **LSI Design Contest 2027 in Hokkaido** technical report and presentation summary in **Typst (`.typ` -> `.pdf`)** comparing **Method 1 (Sequential Single-MAC Baseline)** vs. **Method 2 (Parallel Upgrade)** using ONLY verified numbers from:
- `reports/quantization/quantization_report.md`
- `reports/simulation/cnn_top_verification_report.md`
- `reports/synthesis/synthesis_report.md`
- `fpga_complete_math_and_setup_guide.typ`
```
