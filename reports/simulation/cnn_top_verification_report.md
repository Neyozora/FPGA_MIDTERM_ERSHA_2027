# Complete CNN Integration & ModelSim Verification Report (Milestone 9)

Simulator: **ModelSim Intel FPGA Starter Edition 18.1 (`vlog / vsim 10.5b`)**
Top-Level DUT: [`rtl/cnn_top.v`](../../rtl/cnn_top.v)
Integration Testbench: [`sim/tb/tb_cnn_top.v`](../../sim/tb/tb_cnn_top.v)
ModelSim Script: [`sim/do/run_tb_cnn_top.do`](../../sim/do/run_tb_cnn_top.do)
Transcript Log: [`reports/simulation/cnn_top_transcript.log`](cnn_top_transcript.log)

---

## 1. End-to-End Inference Cycle Breakdown (`test_image.mem`, Image 0)

| FSM Stage inside `cnn_top.v` | Output Buffer / Register | Tensor Shape | Elements Verified | Mismatches vs Python Ref | Max Abs Error | Measured Clock Cycles | Share of Total Cycles |
|---|---|---|---|---|---|---|---|
| Input Load (`S_LOAD_IMG`) | `img_ram` | `1 × 32 × 32` | `1,024` | `0` | `0` | `5` (FSM overhead) | `< 0.01%` |
| **Conv1 + ReLU1** (`S_CONV1_RUN` + `S_RELU1_DRAIN`) | `conv1_relu_ram` | `8 × 30 × 30` | `7,200` | **`0`** | **`0`** | **`86,404`** | `28.52%` |
| **MaxPool1** (`S_POOL1_RUN`) | `pool1_ram` | `8 × 15 × 15` | `1,800` | **`0`** | **`0`** | **`10,803`** | `3.57%` |
| **Conv2 + ReLU2** (`S_CONV2_RUN` + `S_RELU2_DRAIN`) | `conv2_relu_ram` | `16 × 13 × 13` | `2,704` | **`0`** | **`0`** | **`202,804`** | `66.94%` |
| **GAP** (`S_GAP_RUN`) | `gap_ram` | `16` | `16` | **`0`** | **`0`** | **`2,755`** | `0.91%` |
| **Fully Connected** (`S_FC_RUN`) | `fc_score_ram` | `10` | `10` | **`0`** | **`0`** | **`193`** | `0.06%` |
| **Argmax** (`S_ARGMAX_RUN`) | `predicted_class` | `1` (`4-bit`) | `1` | **`0`** | **`0`** | **`14`** | `< 0.01%` |
| **Total End-to-End Inference** | `done == 1'b1` | `1×32×32 → class 0..9` | **`11,731`** | **`0 / 11,731`** | **`0`** | **`302,978` cycles** | **`100.00%`** (`6.060 ms` @ 50 MHz) |

---

## 2. Multi-Image 10-Class End-to-End Verification (`mem/test_images_100.mem`)

Ten test images spanning all 10 Fashion-MNIST categories (`Vec 0, 10, 20, 30, 40, 50, 60, 70, 80, 90`) were loaded sequentially into `cnn_top.v` via the external write port (`ext_img_wr_en / ext_img_wr_addr / ext_img_wr_data`), classified by the full RTL pipeline, and compared against the Python integer reference model (`sim/vectors/batch100_expected_scores.mem` and `batch100_expected_preds.mem`):

| Test Vec # | Ground Truth Class | Python Ref Class | RTL `predicted_class` | RTL `fc_score_ram[0..9]` (`INT8` Scores) | Score Mismatches vs Python Ref | Max Abs Error | Measured Cycles | Agreement |
|---|---|---|---|---|---|---|---|---|
| `0` | `0` (`T-shirt/top`) | `0` | **`0`** | `[  7, -28,  -5,  -2, -19, -25,  -2, -36,  -7, -14]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `10` | `1` (`Trouser`) | `1` | **`1`** | `[ -6,  36,   6,   6,   0, -19,  -4, -70, -17, -31]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `20` | `2` (`Pullover`) | `2` | **`2`** | `[  0, -17,   9,  -7,  -3, -46,   3, -63,  -9, -38]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `30` | `3` (`Dress`) | `3` | **`3`** | `[ -1,  -8, -17,   7,  -6,  -8,  -4, -28, -12, -11]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `40` | `4` (`Coat`) | `4` | **`4`** | `[ -6,  -4,   5,  -5,   7, -29,   2, -49, -10, -37]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `50` | `5` (`Sandal`) | `5` | **`5`** | `[-10, -78, -31, -13, -23,  16, -19,   7,  -2,  -1]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `60` | `6` (`Shirt`) | `6` | **`6`** | `[  1, -22,  -5,  -3,   0, -23,   4, -31, -12, -17]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `70` | `7` (`Sneaker`) | `7` | **`7`** | `[-16,-122, -36, -41, -42,  25, -35,  34,  -4,  20]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `80` | `8` (`Bag`) | `8` | **`8`** | `[ -5, -85, -16, -17, -32, -18,  -8, -21,  15,  -2]` | `0 / 10` | `0` | `302,978` | **PASS** |
| `90` | `9` (`Ankle boot`) | `9` | **`9`** | `[-13, -81, -29, -26, -27,  14, -24,  16,  -3,  22]` | `0 / 10` | `0` | `302,978` | **PASS** |

- **RTL vs. Python Integer Reference Classification Agreement:** **`10 / 10` (`100.0%`)**
- **RTL vs. Ground-Truth Label Accuracy on 10-Class Sweep:** **`10 / 10` (`100.0%`)**
- **Total FC Class Score Mismatches:** **`0 / 100` (`Max Abs Numerical Error = 0`)**
- **Measured Inference Latency:** **`302,978` clock cycles per image** (`6.060 ms` at `50 MHz`, `3.030 ms` at `100 MHz`)
