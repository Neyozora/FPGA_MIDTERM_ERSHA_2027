# Method 2 Quartus Prime 18.1 Synthesis & Timing Report (CIFAR-10 RGB + Cyclone V HPS)

- **Toolchain:** Intel Quartus Prime Lite Edition `18.1.0 Build 625 SJ Lite Edition`
- **Target Board:** Terasic DE10-Standard (`DE-SoC [USB-1]`, JTAG ID `0x02D020DD`)
- **Target Device:** Intel Cyclone V SX SoC **`5CSXFC6D6F31C6`** (`41,910` ALMs, `553` M10K Blocks, `112` DSP Blocks)
- **Quartus Project:** `quartus_m2/de10_standard_m2.qpf` / `quartus_m2/de10_standard_m2.qsf` / `quartus_m2/de10_standard_m2.sdc`
- **Qsys HPS Bridge:** `quartus_m2/hps_cnn_qsys.qsys` (`hps_0.h2f_lw_axi_master` @ `0xFF200000`)

---

## 1. Cyclone V Resource Utilization Summary (`de10_standard_m2.fit.summary`)

| Resource Metric | Method 2 (FPGA Core Only) | Method 2 (Full FPGA + HPS Qsys + VGA) | Available on `5CSXFC6D6F31C6` |
| :--- | :---: | :---: | :---: |
| **Logic Utilization (ALMs)** | `3,815` (`9%`) | **`8,149` (`19%`)** | `41,910` |
| **Total Registers** | `3,921` | **`4,527`** | `167,640` |
| **Block Memory Bits (M10K)** | `860,032` bits (`15%`) | **`860,288` bits (`15%`)** | `5,662,720` bits |
| **M10K RAM Blocks** | `102` (`18%`) | **`103` (`19%`)** | `553` |
| **DSP Blocks** | `52` (`46%`) | **`52` (`46%`)** | `112` |
| **I/O Pins** | `96` (`19%`) | **`168` (`34%`, incl. `HPS_DDR3_*`)** | `499` |
| **DLLs** | `0` | **`1` (`25%`, for HPS DDR3)** | `4` |

---

## 2. TimeQuest Static Timing Analysis (`de10_standard_m2.sta.summary`)

| Timing Corner | Clock Domain | Setup Slack | Hold Slack | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Slow 1100mV 85°C** | `CLOCK_50` (`50.00 MHz`) | **`+3.634 ns`** | **`+0.228 ns`** | **PASS (0 violations)** |
| **Slow 1100mV 85°C** | `hps_sdram` (`afi_clk_write_clk`) | **`+1.730 ns`** | **`+0.143 ns`** | **PASS (0 violations)** |
| **Slow 1100mV 85°C** | `altera_reserved_tck` (`30.00 MHz`) | **`+11.202 ns`** | **`+0.358 ns`** | **PASS (0 violations)** |

---

## 3. Hardware Inference Latency & Throughput (`CLOCK_50` = `50.00 MHz`)

- **Exact Hardware Cycles per Image (`cnn_top_m2`):** **`202,538` cycles** (`202,539` cycles incl. `B_CNN_WAIT` done-sample cycle on HPS)
- **Latency @ `50.00 MHz`:** `202,538 / 50,000,000 = 4.051 ms / image`
- **Throughput @ `50.00 MHz`:** **`246.9 FPS`**
- **ModelSim 18.1 Verification (`tb_cnn_top_m2.v`):** **`100 / 100`** prediction parity, **`1,000 / 1,000`** INT8 score parity (`0` mismatches)
