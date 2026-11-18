# Method 2: Parallel 4-Layer INT8 CNN on CIFAR-10 (`32×32×3` RGB) + Cyclone V HPS & Live VGA/LCD

**Target Board:** Terasic DE10-Standard (`5CSXFC6D6F31C6`)  
**Contest Target:** 30th LSI Design Contest in Hokkaido (2027) — *Class 1: Image Classification*

---

## 1. Key Upgrades Over Method 1

| Metric | Method 1 (Baseline) | Method 2 (Parallel + HPS) |
| :--- | :---: | :---: |
| **Dataset** | Fashion-MNIST (`32×32×1` Gray) | **CIFAR-10 (`32×32×3` Native RGB)** |
| **CNN Topology** | 2-Layer (`1→8→Pool→8→16→GAP→FC`) | **4-Layer (`3→16→Pool→32→Pool→64→64→GAP→FC`)** |
| **Parameters** | `1,418` | **`61,162` (`43.1×`)** |
| **Quantization** | Per-Layer INT8 QAT | **BatchNorm Folding + Per-Channel INT8 QAT** |
| **INT8 Test Accuracy** | `85.87%` *(Fashion-MNIST)* | **`83.54%`** *(CIFAR-10)* / `84.00%` *(100 HW vectors)* |
| **Conv Engine** | 1-MAC/cycle sequential | **2-Line Shift-Register Line Buffer + `4×9 = 36` Parallel MACs (`Po=4`)** |
| **Inference Cycles** | `302,978` cycles (`6.060 ms` / `165.0 FPS`) | **`202,538` cycles (`4.051 ms` / `246.9 FPS` @ 50 MHz)** |
| **Cyclone V Resources** | `804` ALMs (`2%`), `31` M10Ks (`6%`), `22` DSPs | `3,815` ALMs (`9%` FPGA) / `8,149` ALMs (`19%` + HPS), `103` M10Ks (`19%`), `52` DSPs (`46%`) |
| **Displays & Control** | VGA Grayscale + PS/2 + JTAG | **VGA 24-bit RGB HUD + HPS `128×64` LCD + Lightweight AXI (`0xFF200000`) + JTAG** |

---

## 2. Directory Layout

- `sw_m2/` — CIFAR-10 preprocessing, 4-layer CNN + BN folding + Per-Channel QAT, `.mem` exporter, plot generator, HPS UART uploader, and ModelSim + FPGA JTAG GUI (`gui_app_m2.py`).
- `rtl_m2/` — Synthesizable Verilog-2001 RTL (`conv_engine_m2.v`, `maxpool_m2.v`, `gap_fc_argmax_m2.v`, `cnn_top_m2.v`, `vga_cnn_hud_m2.v`, `de10_standard_top_m2.v`).
- `sim_m2/` — 100-image bit-exact testbench (`tb_cnn_top_m2.v`) & single-image GUI testbench (`tb_gui_single_m2.v`).
- `mem_m2/` — Packed 72-bit weight banks (`w_bank0..3.mem`), 53-bit qparam banks (`q_bank0..3.mem`), FC ROMs, and 10-image DE10-Standard ROM (`de10_images_10_m2.mem`).
- `quartus_m2/` — Quartus Prime 18.1 project (`de10_standard_m2.qpf/.qsf/.sdc`) & Qsys HPS AXI bridge (`build_hps_qsys.tcl`, `hps_cnn_qsys.qsys`).
- `hps_m2/` — Angstrom Linux C applications for the DE10-Standard ARM Cortex-A9 (`main_cifar10_vga_bridge.c`, `cifar10_lcd_data.h`, `hw_images_100.bin`).
- `reports_m2/` — Verified accuracy JSONs, confusion matrix, weight histograms, and comparison tables.

---

## 3. Quick Start Commands

### Python Training, Export & GUI (Windows)
```powershell
python -m sw_m2.preprocess
python -u -m sw_m2.train_qat
python -m sw_m2.export_mem
python -m sw_m2.gui_app_m2
```

### ModelSim 18.1 Bit-Exact Verification (`100 / 100` preds, `1,000 / 1,000` scores)
```powershell
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlib.exe work_m2
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vlog.exe -work work_m2 rtl_m2/*.v sim_m2/tb_cnn_top_m2.v
C:\intelFPGA_lite\18.1\modelsim_ase\win32aloem\vsim.exe -c work_m2.tb_cnn_top_m2 -do "run -all; quit -f"
```

### Program DE10-Standard FPGA & Run HPS Bridge
```powershell
C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe -c "DE-SoC [USB-1]" -m JTAG -o "p;quartus_m2\output_files\de10_standard_m2.sof@2"
```
On DE10-Standard Angstrom Linux (`COM7`, `115200` baud):
```bash
cd /home/xuser/hps_cifar10_lcd
./hps_vga_bridge
```
- **`SW[8] = 0`:** Switch-Follow Mode (`SW[3:0]` / `KEY[2]` updates both VGA RGB HUD & HPS `128×64` LCD).
- **`SW[8] = 1`:** 100-Image Slideshow Mode (HPS streams `hw_images_100.bin` over `0xFF200000` to FPGA, VGA, and LCD).
- **`SW[9] = 1`:** Displays exact hardware cycle count (`03172A` hex = `202,538`) on `HEX5..HEX0`.
