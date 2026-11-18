# ModelSim Individual Module Verification Report (Milestone 8)

Simulator: **ModelSim Intel FPGA Starter Edition 18.1 (`vlog / vsim 10.5b`)**
Master Script: [`sim/do/run_all_unit_tbs.do`](../../sim/do/run_all_unit_tbs.do)
Transcript Log: [`reports/simulation/unit_tests_transcript.log`](unit_tests_transcript.log)

---

## 1. Unit Testbench Summary

| Testbench | DUT (`rtl/*.v`) | `.do` Script | Test Vectors & Coverage | Measured Clock Cycles | Max Abs Numerical Error | Mismatches | Status |
|---|---|---|---|---|---|---|---|
| [`tb_conv_engine.v`](../../sim/tb/tb_conv_engine.v) | [`conv_engine.v`](../../rtl/conv_engine.v) | [`run_tb_conv_engine.do`](../../sim/do/run_tb_conv_engine.do) | **Conv1:** `7,200` `INT32` accumulators (`img0_conv1_acc.mem`) + `7,200` `INT8` outputs (`img0_conv1_req.mem`)<br/>**Conv2:** `2,704` `INT32` accumulators (`img0_conv2_acc.mem`) + `2,704` `INT8` outputs (`img0_conv2_req.mem`) | Conv1: **`86,401`**<br/>Conv2: **`202,801`** | `0` | `0 / 19,808` | **PASS** |
| [`tb_relu.v`](../../sim/tb/tb_relu.v) | [`relu.v`](../../rtl/relu.v) | [`run_tb_relu.do`](../../sim/do/run_tb_relu.do) | Exhaustive `[-128..+127]` (`256` combinational + `256` registered) + `7,200` ReLU1 (`img0_relu1_out.mem`) + `2,704` ReLU2 (`img0_relu2_out.mem`) | **`10,160`** (1/elem) | `0` | `0 / 10,160` | **PASS** |
| [`tb_maxpool.v`](../../sim/tb/tb_maxpool.v) | [`maxpool.v`](../../rtl/maxpool.v) | [`run_tb_maxpool.do`](../../sim/do/run_tb_maxpool.do) | Directed `1×4×4 → 1×2×2` signed negative/zero/positive test (`4` outputs) + full Pool1 `8×30×30 → 8×15×15` (`1,800` outputs vs `img0_pool1_out.mem`) | **`10,801`** (Pool1) | `0` | `0 / 1,804` | **PASS** |
| [`tb_gap.v`](../../sim/tb/tb_gap.v) | [`gap.v`](../../rtl/gap.v) | [`run_tb_gap.do`](../../sim/do/run_tb_gap.do) | Full `16×13×13 → 16` GAP on `img0_relu2_out.mem`: `16` `INT32` spatial sums (`img0_gap_sum.mem`) + `16` `INT8` rescaled outputs (`img0_gap_out.mem`) | **`2,753`** | `0` | `0 / 32` | **PASS** |
| [`tb_fully_connected.v`](../../sim/tb/tb_fully_connected.v) | [`fully_connected.v`](../../rtl/fully_connected.v) | [`run_tb_fully_connected.do`](../../sim/do/run_tb_fully_connected.do) | `16 → 10` FC on `img0_gap_out.mem`, `fc_weights.mem`, `fc_bias.mem`: `10` `INT32` accumulators (`img0_fc_acc.mem`) + `10` signed `INT8` scores (`img0_fc_scores.mem`) | **`191`** | `0` | `0 / 20` | **PASS** |
| [`tb_argmax.v`](../../sim/tb/tb_argmax.v) | [`argmax.v`](../../rtl/argmax.v) | [`run_tb_argmax.do`](../../sim/do/run_tb_argmax.do) | Image 0 golden FC scores (`class=0, max=7`) + tie-breaking (`class=1, max=25`) + all-negative scores (`class=4, max=-12`) + last-index winner (`class=9, max=127`) | **`13`** per run | `0` | `0 / 4` | **PASS** |

---

## 2. Detailed Expected vs. Actual Module Outputs (Image 0)

### A. Global Average Pooling (`tb_gap.v`)

| Channel | Expected `INT32` Sum | Actual RTL `INT32` Sum | Expected `INT8` Output | Actual RTL `INT8` Output | Abs Error |
|---|---|---|---|---|---|
| `0` | `601` | `601` | `19` | `19` | `0` |
| `1` | `615` | `615` | `20` | `20` | `0` |
| `2` | `1010` | `1010` | `32` | `32` | `0` |
| `3` | `1085` | `1085` | `35` | `35` | `0` |
| `4` | `803` | `803` | `26` | `26` | `0` |
| `5` | `290` | `290` | `9` | `9` | `0` |
| `6` | `97` | `97` | `3` | `3` | `0` |
| `7` | `1482` | `1482` | `47` | `47` | `0` |
| `8` | `1418` | `1418` | `45` | `45` | `0` |
| `9` | `567` | `567` | `18` | `18` | `0` |
| `10` | `895` | `895` | `29` | `29` | `0` |
| `11` | `2343` | `2343` | `75` | `75` | `0` |
| `12` | `644` | `644` | `21` | `21` | `0` |
| `13` | `1552` | `1552` | `50` | `50` | `0` |
| `14` | `1241` | `1241` | `40` | `40` | `0` |
| `15` | `908` | `908` | `29` | `29` | `0` |

### B. Fully Connected Layer (`tb_fully_connected.v`)

| Class Index | Fashion-MNIST Class | Expected `INT32` Acc | Actual RTL `INT32` Acc | Expected `INT8` Score | Actual RTL `INT8` Score | Abs Error |
|---|---|---|---|---|---|---|
| `0` | `T-shirt/top` | `586` | `586` | **`7`** | **`7`** | `0` |
| `1` | `Trouser` | `-2487` | `-2487` | `-28` | `-28` | `0` |
| `2` | `Pullover` | `-447` | `-447` | `-5` | `-5` | `0` |
| `3` | `Dress` | `-219` | `-219` | `-2` | `-2` | `0` |
| `4` | `Coat` | `-1690` | `-1690` | `-19` | `-19` | `0` |
| `5` | `Sandal` | `-2209` | `-2209` | `-25` | `-25` | `0` |
| `6` | `Shirt` | `-175` | `-175` | `-2` | `-2` | `0` |
| `7` | `Sneaker` | `-3145` | `-3145` | `-36` | `-36` | `0` |
| `8` | `Bag` | `-657` | `-657` | `-7` | `-7` | `0` |
| `9` | `Ankle boot` | `-1272` | `-1272` | `-14` | `-14` | `0` |

---

## 3. Bug Found & Resolved During Milestone 8 Verification

- **Non-blocking `out_valid` override in `conv_engine.v` (`S_REQUANT`):**
  On the initial run of `tb_conv_engine`, `conv_engine` ran for the exact expected cycle counts (`86,401` and `202,801`), but `c1_cnt` was `0` because inside `S_REQUANT`, `out_valid <= 1'b1;` was immediately overwritten in the same clock cycle by `if (out_ready) begin out_valid <= 1'b0; ...`. Removing the redundant `out_valid <= 1'b0` inside `S_REQUANT` (letting `S_PRE` and `S_DONE` clear `out_valid` on the following clock edge) resolved the issue, and all `7,200` Conv1 and `2,704` Conv2 outputs matched the golden vectors with `0` mismatches.
