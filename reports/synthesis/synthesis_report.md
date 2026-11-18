# Quartus Prime FPGA Synthesis, Timing & Physical DE10-Standard Deployment Report (Milestone 10)

- **Toolchain:** Intel Quartus Prime Lite Edition `18.1.0 Build 625 SJ Lite Edition` (`quartus_map`, `quartus_fit`, `quartus_asm`, `quartus_sta`, `quartus_pgm`)
- **Target Board:** Terasic DE10-Standard (`DE-SoC [USB-1]`, JTAG ID `0x02D020DD`)
- **Target FPGA Device:** Intel Cyclone V SX SoC **`5CSXFC6D6F31C6`** (`110K LEs`, `41,910 ALMs`, `553 M10K Blocks`, `112 DSP Blocks`)
- **Quartus Project Files:** [`quartus/de10_standard.qpf`](../../quartus/de10_standard.qpf), [`quartus/de10_standard.qsf`](../../quartus/de10_standard.qsf), [`quartus/de10_standard.sdc`](../../quartus/de10_standard.sdc), [`quartus/cnn_top.qpf`](../../quartus/cnn_top.qpf)
- **Generated FPGA Bitstream:** [`quartus/output_files/de10_standard.sof`](../../quartus/output_files/de10_standard.sof) (Checksum `0x015EDB97`)

---

## 1. Top-Level Cyclone V Resource Utilization Summary (`de10_standard.fit.rpt`)

| Resource Metric | Used (`cnn_top` Core Only) | Used (Full `de10_standard_top` Board Design) | Available on `5CSXFC6D6F31C6` | Device Utilization (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logic Utilization (in ALMs)** | – *(see ALUTs below)* | **`804`** | `41,910` | **`1.92%` (`2%`)** |
| **Combinational ALUTs** | **`1,147`** | **`1,323`** | `83,820` | **`1.58%`** |
| **Dedicated Logic Registers** | **`502`** | **`646`** | `167,640` | **`0.39%`** |
| **Block Memory Bits (M10K)** | **`104,272` bits (`13.03 KB`)** | **`186,192` bits (`23.27 KB`)** | `5,662,720` bits | **`3.29%` (`3%`)** |
| **Distributed MLAB Memory Bits** | **`208` bits** | **`208` bits** | – | `< 0.1%` |
| **Inferred M10K RAM Blocks** | **`21`** | **`31`** *(+10 for 10-image board ROM)* | `553` | **`5.61%` (`6%`)** |
| **Inferred Variable-Precision DSP Blocks** | **`22`** | **`22`** | `112` | **`19.64%` (`20%`)** |
| **I/O Pins** | `0` *(internal core)* | **`67`** (`CLOCK_50`, `KEY[3:0]`, `SW[9:0]`, `LEDR[9:0]`, `HEX0..5`) | `499` | **`13.43%` (`13%`)** |
| **PLLs / External SDRAM / HPS DDR** | `0` | `0` | `15` | **`0.00%`** |

---

## 2. Per-Module Hardware Resource Breakdown (`de10_standard.map.rpt`)

| Hierarchy Module (`Entity`) | Combinational ALUTs | Dedicated Registers | Block Memory Bits | M10K Blocks | DSP Blocks | Functional Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`|de10_standard_top`** *(Total)* | **`1,323`** | **`646`** | **`186,192`** | **`31`** | **`22`** | Complete DE10-Standard board design |
| ├── `board_img_rom` (`10×1024×8b`) | `8` | `1` | `81,920` | `10` | `0` | 10-class on-chip test image suite ROM |
| ├── Board Loader FSM, BCD Counter & 7-Seg | `168` | `88` | `0` | `0` | `0` | Switch/key sync, ROM loader, 6-digit BCD counter |
| └── **`|cnn_top:u_cnn_top`** *(CNN Core)* | **`1,147`** | **`502`** | **`104,272`** | **`21`** | **`22`** | **Standalone Two-Layer CNN Accelerator** |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `conv_engine:u_conv_engine` | `468` | `164` | `0` | `0` | `9` | Shared `Conv1` (`1→8`) & `Conv2` (`8→16`) engine |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `gap:u_gap` | `246` | `88` | `0` | `0` | `6` | `13×13` Global Average Pooling (`×16780 >> 19`) |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `fully_connected:u_fc` | `206` | `78` | `0` | `0` | `7` | `16 → 10` Dense layer + dyadic requantization |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `maxpool:u_maxpool` | `139` | `74` | `0` | `0` | `0` | `2×2` stride-2 MaxPooling engine |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `argmax:u_argmax` | `41` | `29` | `0` | `0` | `0` | 10-class signed `INT8` winner selector |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `relu:u_relu` | `7` | `21` | `0` | `0` | `0` | Registered signed `INT8` ReLU clamp |
| &nbsp;&nbsp;&nbsp;&nbsp;├── `cnn_top` Master FSM & muxes | `40` | `24` | `0` | `0` | `0` | 16-state master controller |
| &nbsp;&nbsp;&nbsp;&nbsp;└── On-Chip Weight ROMs & Feature BRAMs | `0` | `24` *(MLAB)* | `104,272` | `21` | `0` | `img_ram`, `conv1_relu_ram`, `pool1_ram`, `conv2_relu_ram`, weight/bias ROMs |

---

## 3. On-Chip Memory Inference Table (`de10_standard.map.rpt`)

| RTL Memory Array | Inferred Primitive Type | Mode | Depth × Width | Memory Bits | Initialization File (`.mif` / `.mem`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `img_ram` | **M10K** (`altsyncram`) | Simple Dual Port | `1,024 × 8b` | `8,192` | `mem/test_image.mem` |
| `conv1_w_rom` | **M10K** (`altsyncram`) | ROM | `72 × 8b` | `576` | `mem/conv1_weights.mem` |
| `conv1_b_rom` | **M10K** (`altsyncram`) | ROM | `8 × 32b` | `256` | `mem/conv1_bias.mem` |
| `conv1_relu_ram` | **M10K** (`altsyncram`) | Simple Dual Port | `7,200 × 7b` *(unsigned post-ReLU)* | `50,400` | Runtime buffer |
| `pool1_ram` | **M10K** (`altsyncram`) | Simple Dual Port | `1,800 × 8b` | `14,400` | Runtime buffer |
| `conv2_w_rom` | **M10K** (`altsyncram`) | ROM | `1,152 × 8b` | `9,216` | `mem/conv2_weights.mem` |
| `conv2_b_rom` | **M10K** (`altsyncram`) | ROM | `16 × 32b` | `512` | `mem/conv2_bias.mem` |
| `conv2_relu_ram` | **M10K** (`altsyncram`) | Simple Dual Port | `2,704 × 7b` *(unsigned post-ReLU)* | `18,928` | Runtime buffer |
| `gap_ram` | **MLAB** (`altdpram`) | Simple Dual Port | `16 × 8b` | `128` *(MLAB)* | Runtime buffer |
| `fc_w_rom` | **M10K** (`altsyncram`) | ROM | `160 × 8b` | `1,280` | `mem/fc_weights.mem` |
| `fc_b_rom` | **M10K** (`altsyncram`) | ROM | `16 × 32b` | `512` | `mem/fc_bias.mem` |
| `fc_score_ram` | **MLAB** (`altdpram`) | Simple Dual Port | `10 × 8b` | `80` *(MLAB)* | Runtime buffer |
| `board_img_rom` *(Board wrapper)* | **M10K** (`altsyncram`) | ROM | `10,240 × 8b` | `81,920` | `mem/de10_images_10.mem` |

---

## 4. TimeQuest Static Timing Analysis (`de10_standard.sta.rpt`)

| Timing Corner | Target Clock (`CLOCK_50`) | Restricted $F_{\max}$ | Setup Slack | Hold Slack | Min Pulse Width Slack | Timing Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Slow 1100mV 85°C Model** | `50.00 MHz` (`20.000 ns`) | **`67.82 MHz`** (`14.746 ns`) | **`+5.254 ns`** | **`+0.326 ns`** | `+8.863 ns` | **PASS (0 violations)** |
| **Slow 1100mV 0°C Model** | `50.00 MHz` (`20.000 ns`) | **`66.92 MHz`** (`14.943 ns`) | **`+5.057 ns`** | **`+0.305 ns`** | `+8.836 ns` | **PASS (0 violations)** |
| **Fast 1100mV 85°C Model** | `50.00 MHz` (`20.000 ns`) | **`> 125 MHz`** | **`+12.232 ns`** | **`+0.155 ns`** | `+8.428 ns` | **PASS (0 violations)** |

### Hardware Inference Latency & Throughput Summary
- **Exact Clock Cycles per Image (`cnn_top`):** **`302,978` cycles** (+ `2,050` cycles to load a new `1,024`-pixel image from `board_img_rom`)
- **Latency at On-Board `50.00 MHz` (`CLOCK_50`):**
  $$\text{Latency}_{50\text{ MHz}} = \frac{302,978}{50 \times 10^6\text{ Hz}} = \mathbf{6.060\text{ ms / image}} \quad (\mathbf{165.0\text{ FPS}})$$
- **Latency at Worst-Case Slow-Corner $F_{\max}$ (`66.92 MHz`):**
  $$\text{Latency}_{F_{\max}} = \frac{302,978}{66.92 \times 10^6\text{ Hz}} = \mathbf{4.527\text{ ms / image}} \quad (\mathbf{220.9\text{ FPS}})$$

---

## 5. Physical Terasic DE10-Standard JTAG Programming & Verification

The compiled bitstream (`quartus/output_files/de10_standard.sof`, checksum `0x015EDB97`) was programmed directly onto the physical Terasic DE10-Standard (`DE-SoC [USB-1]`, device index `@2`, JTAG ID `0x02D020DD`) via `quartus_pgm`:

```text
Info: Command: quartus_pgm -c "DE-SoC [USB-1]" -m JTAG -o p;quartus/output_files/de10_standard.sof@2
Info (213045): Using programming cable "DE-SoC [USB-1]"
Info (213011): Using programming file quartus/output_files/de10_standard.sof with checksum 0x015EDB97 for device 5CSXFC6D6F31@2
Info (209016): Configuring device index 2
Info (209017): Device 2 contains JTAG ID code 0x02D020DD
Info (209007): Configuration succeeded -- 1 device(s) configured
Info: Quartus Prime Programmer was successful. 0 errors, 0 warnings
```

### Interactive Physical Board Controls on DE10-Standard
| Control (`SW` / `KEY`) | Setting | What Appears on `HEX5 .. HEX0` & `LEDR[9:0]` |
| :--- | :---: | :--- |
| **`SW[3:0]` = `0000` (`0`)** | Class `0` (`T-shirt/top`) | `HEX5..0` = **`0 - P A S 0`**, `LEDR[0]` ON |
| **`SW[3:0]` = `0001` (`1`)** | Class `1` (`Trouser`) | `HEX5..0` = **`1 - P A S 1`**, `LEDR[1]` ON |
| **`SW[3:0]` = `0010` (`2`)** | Class `2` (`Pullover`) | `HEX5..0` = **`2 - P A S 2`**, `LEDR[2]` ON |
| **`SW[3:0]` = `0011` (`3`)** | Class `3` (`Dress`) | `HEX5..0` = **`3 - P A S 3`**, `LEDR[3]` ON |
| **`SW[3:0]` = `0100` (`4`)** | Class `4` (`Coat`) | `HEX5..0` = **`4 - P A S 4`**, `LEDR[4]` ON |
| **`SW[3:0]` = `0101` (`5`)** | Class `5` (`Sandal`) | `HEX5..0` = **`5 - P A S 5`**, `LEDR[5]` ON |
| **`SW[3:0]` = `0110` (`6`)** | Class `6` (`Shirt`) | `HEX5..0` = **`6 - P A S 6`**, `LEDR[6]` ON |
| **`SW[3:0]` = `0111` (`7`)** | Class `7` (`Sneaker`) | `HEX5..0` = **`7 - P A S 7`**, `LEDR[7]` ON |
| **`SW[3:0]` = `1000` (`8`)** | Class `8` (`Bag`) | `HEX5..0` = **`8 - P A S 8`**, `LEDR[8]` ON |
| **`SW[3:0]` = `1001` (`9`)** | Class `9` (`Ankle boot`) | `HEX5..0` = **`9 - P A S 9`**, `LEDR[9]` ON |
| **`SW[9]` = `1` (Up)** | Cycle Counter View | `HEX5..0` = **`3 0 2 9 7 8`** (exact hardware clock cycles!) |
| **`KEY[0]` (Pressed)** | Hardware Reset | Resets FSM & re-runs inference on release |
| **`KEY[1]` (Pressed)** | Re-Run Trigger | Reloads selected image from M10K ROM & re-runs CNN |
