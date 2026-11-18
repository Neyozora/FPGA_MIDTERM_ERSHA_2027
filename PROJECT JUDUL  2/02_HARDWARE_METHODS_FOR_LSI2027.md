# Hardware Architecture Methods for LSI Design Contest 2027 (`32×32` CNN)

In **Method 1**, we proved 100% bit-exactness using a **Sequential Single-MAC Baseline** (`302,978` cycles = `6.06 ms` @ 50 MHz, `5%` ALMs, `22` DSPs). Because the LSI Design Contest evaluates **Throughput (FPS), Latency, Accuracy, and Area Efficiency (ALMs/DSPs)**, here are the three best hardware upgrade methods:

---

## Method 2 (Recommended): `3×3` Line-Buffer + 9-MAC Parallel Array + Per-Channel INT8 QAT

### 1. Why Method 1 Is Slow (`302,978` Cycles)
In Method 1, `Conv2` (`66.94%` of cycles) + `Conv1` (`28.52%`) consume **`95.46%` of total runtime** because `conv_engine.v` reads 1 pixel at a time from BRAM and computes 1 MAC per cycle.

### 2. Method 2 Architecture
1. **2-Line Shift-Register Line Buffer (`3×3` Sliding Window):**
   - Streams pixels in raster order (`1` pixel/cycle). Two `W`-deep shift-register FIFOs hold the previous 2 rows, exposing all **9 pixels of the `3×3` window simultaneously every clock cycle**!
2. **9-Parallel MAC Tree (`9` DSPs in Parallel):**
   - Multiplies all 9 window pixels by all 9 kernel weights in **1 clock cycle** and sums them with a 4-stage pipelined adder tree.
3. **Output-Channel Unrolling ($P_o = 4$):**
   - Replicating the 9-MAC unit `4×` (using `36` of the Cyclone V's `112` DSPs = `32%`) computes 4 output channels simultaneously.
4. **Algorithmic Upgrades:**
   - **Per-Channel Weight Quantization** (each filter gets its own $M_c, n_c$ instead of sharing 1 scale across all 16 filters).
   - **BatchNorm Folding** into `INT8` weights & `INT32` biases (`0` extra hardware!).
   - **`2×2` Spatial Pyramid Pooling (SPP)** instead of `1×1` GAP (`16 × 4 = 64` features into FC instead of `16`, preserving top-vs-bottom garment structure like collars vs. hems!).

### 3. Expected Method 2 Performance on Cyclone V (`5CSXFC6D6F31C6`)
- **Latency:** Drops from `302,978` cycles (`6.06 ms`) $\rightarrow$ **`~14,000` cycles (`0.28 ms` @ 50 MHz = `> 3,500 FPS`!)** — a **`21×` hardware speedup**.
- **Accuracy:** Jumps from `85.87%` $\rightarrow$ **`89% – 92%`** on Fashion-MNIST (and handles `32×32×3` real clothing photos cleanly).

---

## Method 3 (Ultra-Low Area / ASIC-Friendly): Binary Neural Network (`1-Bit` XNOR-Popcount)

If you want to minimize **DSP blocks (`0` DSPs!)** and **BRAM bits** for an ultra-compact ASIC or tiny FPGA (like the LSI Contest **ZERO7** board):
1. **Binary Weights & Activations (`+1` / `-1` encoded as `1` / `0`):**
   - Multiplication (`+1 * -1 = -1`) becomes a single **1-bit `XNOR` gate** (`~(a ^ w)`)!
   - Accumulation becomes **Popcount** (counting the number of `1` bits in a vector).
2. **Zero DSPs, Tiny Memory:**
   - Weights shrink by **`8×`** (`1,384` bytes $\rightarrow$ `173` bytes!).
   - Keep only the first layer (`Conv1`) in `INT8` and binarize `Conv2` + `FC` (`XNOR-Net` style) to preserve accuracy (`~81%–84%`).

---

## Method 4 (Modern Edge-AI Topology): Depthwise-Separable MobileNet-Tiny (`INT8`)

Standard `$3 times 3$` convolutions mix spatial filtering and channel mixing in one heavy step (`8 × 16 × 9 = 1,152` weights). Splitting `Conv2` into:
1. **Depthwise `3×3` Conv** (`8 × 9 = 72` weights — spatial edges per channel)
2. **Pointwise `1×1` Conv** (`8 × 24 = 192` weights — mixes channels)
lets you increase the network width to **`24` or `32` channels** while using **fewer weights and MACs** than Method 1's `16`-channel standard `Conv2`!
