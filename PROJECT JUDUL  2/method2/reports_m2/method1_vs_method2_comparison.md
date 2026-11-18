# Method 1 (Fashion-MNIST Baseline) vs. Method 2 (CIFAR-10 RGB Parallel Upgrade)

Verified comparison for the **30th LSI Design Contest in Hokkaido (2027)** (`Class 1: Image Classification Class`, target FPGA: Terasic DE10-Standard **`5CSXFC6D6F31C6`**).

---

## 1. Side-by-Side Comparison Table

| Metric | Method 1 (Midterm Baseline) | Method 2 (Parallel RGB Upgrade) | Improvement / Delta |
| :--- | :---: | :---: | :--- |
| **Dataset & Input Format** | Fashion-MNIST (`28×28×1` padded to `32×32×1` Grayscale) | **CIFAR-10 (`32×32×3` Native RGB)** | Natural color photos, 3× input channels |
| **Input Quantization** | `q_in = pixel >> 1` (`[0, 127]`, 0 gates) | `q_in = int8(pixel - 128)` (`[-128, 127]`, MSB flip, 0 gates) | Full 8-bit zero-centered dynamic range |
| **CNN Topology** | 2-Layer (`Conv1(1→8) → Pool → Conv2(8→16) → GAP → FC`) | **4-Layer (`Conv1(3→16) → Pool → Conv2(16→32) → Pool → Conv3(32→64) → Conv4(64→64) → GAP → FC`)** | 2× deeper feature hierarchy |
| **Total Parameters** | `1,418` | **`61,162`** | **43.1×** model capacity |
| **Quantization Method** | Per-Layer Symmetric INT8 QAT | **BatchNorm Folding + Per-Channel Symmetric INT8 QAT** | Per-filter `(M_c, shift_c)` dyadic scaling |
| **Test Accuracy (Float / INT8)** | `86.10%` / `85.87%` *(Fashion-MNIST)* | **`83.49%` Float+BN / `83.61%` QAT / `83.54%` Pure INT8** *(CIFAR-10)* | `< 0.1%` INT8 quantization drop |
| **Conv Engine Architecture** | Sequential 1-MAC/cycle (`conv_engine.v`) | **2-Line Shift-Register Line Buffer + 4×9 Parallel MAC Array (`Po = 4`, `36` MACs/cycle)** | **36×** peak MAC parallelism |
| **Exact Inference Cycles** | `302,978` cycles | **`202,538` cycles** | **1.50× faster** despite **43× larger** CNN |
| **Latency @ `50.00 MHz`** | `6.060 ms` (`165.0 FPS`) | **`4.051 ms` (`246.9 FPS`)** | **+81.9 FPS** |
| **ModelSim 18.1 Parity** | `100 / 100` preds, `1,000 / 1,000` INT8 scores | **`100 / 100` preds, `1,000 / 1,000` INT8 scores** *(before psum BRAM refactor; `3/3` & `30/30` re-verified after)* | 100% bit-exact |
| **Cyclone V ALMs** | `804 / 41,910` (`2%`) | **`3,815 / 41,910` (`9%`)** | Fits with `91%` headroom |
| **Dedicated Registers** | `646` | **`3,921`** | `2.3%` of device |
| **M10K RAM Blocks** | `31 / 553` (`6%`, `186,192` bits) | **`102 / 553` (`18%`, `860,032` bits)** | All weights & feature maps on-chip |
| **DSP Blocks** | `22 / 112` (`20%`) | **`52 / 112` (`46%`)** | 4×9 MAC array + requantizers |
| **Timing (`CLOCK_50` = 50 MHz)** | **PASS** (`+5.057 ns` worst-case setup slack) | **PASS** (`+2.834 ns` worst-case setup slack, `0` violations) | Clean 50 MHz timing closure |
| **VGA Live HUD** | `32×32` Grayscale + PS/2 Draw | **`32×32` 24-bit Color RGB CIFAR-10 HUD** | Live RGB monitor display |

---

## 2. Key Files Created for Method 2 (Method 1 Completely Untouched)

- **Python Pipeline (`sw_m2/`):**
  - [`sw_m2/config.py`](../sw_m2/config.py), [`sw_m2/preprocess.py`](../sw_m2/preprocess.py), [`sw_m2/model.py`](../sw_m2/model.py), [`sw_m2/train_qat.py`](../sw_m2/train_qat.py), [`sw_m2/export_mem.py`](../sw_m2/export_mem.py), [`sw_m2/gui_app_m2.py`](../sw_m2/gui_app_m2.py)
- **Verilog-2001 RTL (`rtl_m2/`):**
  - [`rtl_m2/conv_engine_m2.v`](../rtl_m2/conv_engine_m2.v), [`rtl_m2/maxpool_m2.v`](../rtl_m2/maxpool_m2.v), [`rtl_m2/gap_fc_argmax_m2.v`](../rtl_m2/gap_fc_argmax_m2.v), [`rtl_m2/cnn_top_m2.v`](../rtl_m2/cnn_top_m2.v), [`rtl_m2/vga_cnn_hud_m2.v`](../rtl_m2/vga_cnn_hud_m2.v), [`rtl_m2/de10_standard_top_m2.v`](../rtl_m2/de10_standard_top_m2.v)
- **ModelSim & Quartus (`sim_m2/`, `quartus_m2/`, `mem_m2/`):**
  - [`sim_m2/tb_cnn_top_m2.v`](../sim_m2/tb_cnn_top_m2.v)
  - [`quartus_m2/de10_standard_m2.qpf`](../quartus_m2/de10_standard_m2.qpf), [`quartus_m2/de10_standard_m2.qsf`](../quartus_m2/de10_standard_m2.qsf), [`quartus_m2/output_files/de10_standard_m2.sof`](../quartus_m2/output_files/de10_standard_m2.sof)
