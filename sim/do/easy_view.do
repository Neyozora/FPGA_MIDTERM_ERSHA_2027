# =============================================================================
# Easy-to-Read ModelSim Waveform Setup (`sim/do/easy_view.do`)
# Replaces cryptic hierarchical signal paths with plain-English labels, widens
# the Name and Value columns, and zooms directly into the final calculation
# where Fully Connected & Argmax output the winning class.
# =============================================================================
view wave
delete wave *

configure wave -namecolwidth 260
configure wave -valuecolwidth 150
configure wave -timelineunits us

add wave -divider "1. MAIN FPGA PINS"
add wave -label "clk (50 MHz Clock)"                  /tb_cnn_top/clk
add wave -label "start (Start Inference)"             /tb_cnn_top/start
add wave -label "done (1 = Inference Finished!)"      -color Yellow /tb_cnn_top/done
add wave -label "predicted_class (0 = T-shirt/top)"   -color Cyan -radix unsigned /tb_cnn_top/predicted_class
add wave -label "FSM State (3=Conv1,6=Pool,8=Conv2,11=GAP,13=FC,15=Argmax)" -radix unsigned /tb_cnn_top/dut/state

add wave -divider "2. LAYER STEP-BY-STEP COMPUTATION"
add wave -label "GAP Channel Index (0..15)"           -radix unsigned /tb_cnn_top/dut/gap_out_ch
add wave -label "GAP Channel Output (INT8)"           -radix decimal  /tb_cnn_top/dut/gap_out_data
add wave -label "FC Class Being Calculated (0..9)"    -radix unsigned /tb_cnn_top/dut/fc_out_idx
add wave -label "FC Score Valid Pulse"                /tb_cnn_top/dut/fc_out_valid
add wave -label "FC Calculated Score (INT8)"          -color Orange -radix decimal /tb_cnn_top/dut/fc_out_data

add wave -divider "3. FINAL 10 CLASS SCORES IN FPGA MEMORY"
add wave -label "Class 0: T-shirt/top (WINNER = +7)"  -color Green -radix decimal {/tb_cnn_top/dut/fc_score_ram[0]}
add wave -label "Class 1: Trouser (-28)"              -radix decimal {/tb_cnn_top/dut/fc_score_ram[1]}
add wave -label "Class 2: Pullover (-5)"              -radix decimal {/tb_cnn_top/dut/fc_score_ram[2]}
add wave -label "Class 3: Dress (-2)"                 -radix decimal {/tb_cnn_top/dut/fc_score_ram[3]}
add wave -label "Class 4: Coat (-19)"                 -radix decimal {/tb_cnn_top/dut/fc_score_ram[4]}
add wave -label "Class 5: Sandal (-25)"               -radix decimal {/tb_cnn_top/dut/fc_score_ram[5]}
add wave -label "Class 6: Shirt (-2)"                 -radix decimal {/tb_cnn_top/dut/fc_score_ram[6]}
add wave -label "Class 7: Sneaker (-36)"              -radix decimal {/tb_cnn_top/dut/fc_score_ram[7]}
add wave -label "Class 8: Bag (-7)"                   -radix decimal {/tb_cnn_top/dut/fc_score_ram[8]}
add wave -label "Class 9: Ankle boot (-14)"           -radix decimal {/tb_cnn_top/dut/fc_score_ram[9]}

add wave -divider "4. ARGMAX DECISION (PICK HIGHEST SCORE)"
add wave -label "Argmax Scanning Class (0..9)"        -radix unsigned /tb_cnn_top/dut/argmax_in_rd_addr
add wave -label "Argmax Best Score So Far"            -color Cyan -radix decimal  /tb_cnn_top/dut/argmax_max_score
add wave -label "Argmax Best Class So Far"            -color Cyan -radix unsigned /tb_cnn_top/dut/argmax_pred_class

# Zoom right into the final 4.5 microseconds (6055.3 us to 6059.8 us) where
# Fully Connected calculates the 10 scores and Argmax picks Class 0!
wave zoom range 6055300ns 6059800ns
