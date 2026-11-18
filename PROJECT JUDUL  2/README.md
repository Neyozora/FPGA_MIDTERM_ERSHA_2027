# Alternative Methods & Upgrade Roadmap (LSI Design Contest 2027)

This folder (`ANOTHER_METHODS/`) contains complete, actionable engineering blueprints for **Method 2, Method 3, and Method 4** to improve upon **Method 1 (Sequential Single-MAC Baseline)** for the **LSI Design Contest 2027 (`32×32` Image Classification)** on the **Terasic DE10-Standard (Intel Cyclone V SX SoC)**.

---

## Quick Comparison: Method 1 vs. Alternative Methods

| Method | Core Hardware Architecture | Quantization | Target Dataset (`32×32`) | Host / Control | Expected Latency (@ 50 MHz) | Expected Accuracy |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **Method 1 (Current)** | Sequential Single-MAC (`1` tap/cycle) | Per-tensor `INT8` QAT | Fashion-MNIST (`32×32×1`) | Standalone RTL + JTAG GUI | `302,978` cycles (`6.06 ms`) | `85.87%` |
| **Method 2 (Recommended)** | **`3×3` Line-Buffer + 9-MAC Array + Channel Unrolling** | **Per-Channel `INT8` + BatchNorm Fold** | **CIFAR-10 / CIFAR-100 (`32×32×3` RGB)** | Standalone RTL or HPS | **`~15,000–35,000` cycles (`< 0.7 ms`)** | **`86%–90%` (CIFAR-10)** |
| **Method 3 (Ultra-Tiny)** | **XNOR-Popcount Binary CNN (`0` DSPs)** | `1-bit` Weights / Activations (`BNN`) | CIFAR-10 (`32×32×3`) | Standalone RTL | `~10,000` cycles (`0.20 ms`) | `78%–83%` |
| **Method 4 (SoC Co-Design)** | **HPS ARM Cortex-A9 + FPGA AXI-Lite Accelerator** | `INT8` / `INT16` | CIFAR-10 / CIFAR-100 + Camera | **HPS via OpenOCD (no SD) OR SD Card Linux** | `< 1.0 ms` + Camera/USB I/O | `> 88%` |

---

## Files in This Folder
1. [`00_LSI_2027_HOKKAIDO_RULES_AND_STRATEGY.md`](00_LSI_2027_HOKKAIDO_RULES_AND_STRATEGY.md) — Official **30th LSI Design Contest in Hokkaido (2027)** dates, classes, and strategy.
2. [`01_CIFAR10_CIFAR100_DATASETS.md`](01_CIFAR10_CIFAR100_DATASETS.md) — Complete guide for switching to **CIFAR-10 & CIFAR-100 (`32×32×3` RGB)** with zero-gate quantization and 4-layer CNN architecture.
2. [`02_HARDWARE_METHODS_FOR_LSI2027.md`](02_HARDWARE_METHODS_FOR_LSI2027.md) — Detailed RTL architectures for **Method 2 (9-MAC Line-Buffer)**, **Method 3 (BNN)**, and **Method 4 (Depthwise-Separable CNN)**.
3. [`03_HPS_OPENOCD_VS_SD_CARD.md`](03_HPS_OPENOCD_VS_SD_CARD.md) — Step-by-step comparison and setup guide for using the **Cyclone V HPS (`4BA00477 SOCVHPS`)** with **OpenOCD (no SD card)** vs. **SD Card Boot (easier for full demos)**.
4. [`hps_openocd_starter/openocd_de10_standard.cfg`](hps_openocd_starter/openocd_de10_standard.cfg) & [`hps_openocd_starter/hps_cnn_bridge.c`](hps_openocd_starter/hps_cnn_bridge.c) — Ready-to-use OpenOCD config and bare-metal C AXI driver template.
