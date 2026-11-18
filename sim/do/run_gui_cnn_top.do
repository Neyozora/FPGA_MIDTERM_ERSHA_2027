# =============================================================================
# ModelSim GUI Script: sim/do/run_gui_cnn_top.do
# Launches the ModelSim Wave window with grouped signals, runs the full CNN
# simulation, and keeps the GUI open (`onfinish stop`) for interactive inspection.
# =============================================================================
transcript on
onbreak {resume}
if {![file exists work]} { vlib work }

vlog -work work \
    rtl/relu.v \
    rtl/maxpool.v \
    rtl/gap.v \
    rtl/argmax.v \
    rtl/fully_connected.v \
    rtl/conv_engine.v \
    rtl/cnn_top.v \
    sim/tb/tb_relu.v \
    sim/tb/tb_maxpool.v \
    sim/tb/tb_gap.v \
    sim/tb/tb_fully_connected.v \
    sim/tb/tb_argmax.v \
    sim/tb/tb_conv_engine.v \
    sim/tb/tb_cnn_top.v

# Load tb_cnn_top with full signal visibility (+acc)
vsim -voptargs="+acc" -t 1ps work.tb_cnn_top
onfinish stop

# Open Wave window and add organized signal groups
view wave
delete wave *

add wave -divider "=== TOP-LEVEL CNN INTERFACE ==="
add wave -noupdate /tb_cnn_top/clk
add wave -noupdate /tb_cnn_top/rst
add wave -noupdate /tb_cnn_top/start
add wave -noupdate /tb_cnn_top/done
add wave -noupdate -radix unsigned /tb_cnn_top/predicted_class
add wave -noupdate -radix unsigned /tb_cnn_top/dut/state
add wave -noupdate /tb_cnn_top/dut/is_conv2

add wave -divider "=== STAGE START / BUSY / DONE ==="
add wave -noupdate /tb_cnn_top/dut/conv_start
add wave -noupdate /tb_cnn_top/dut/conv_busy
add wave -noupdate /tb_cnn_top/dut/conv_done
add wave -noupdate /tb_cnn_top/dut/pool_start
add wave -noupdate /tb_cnn_top/dut/pool_busy
add wave -noupdate /tb_cnn_top/dut/pool_done
add wave -noupdate /tb_cnn_top/dut/gap_start
add wave -noupdate /tb_cnn_top/dut/gap_busy
add wave -noupdate /tb_cnn_top/dut/gap_done
add wave -noupdate /tb_cnn_top/dut/fc_start
add wave -noupdate /tb_cnn_top/dut/fc_busy
add wave -noupdate /tb_cnn_top/dut/fc_done
add wave -noupdate /tb_cnn_top/dut/argmax_start
add wave -noupdate /tb_cnn_top/dut/argmax_busy
add wave -noupdate /tb_cnn_top/dut/argmax_done

add wave -divider "=== CONV ENGINE + RELU ==="
add wave -noupdate -radix unsigned /tb_cnn_top/dut/conv_out_addr
add wave -noupdate /tb_cnn_top/dut/conv_out_valid
add wave -noupdate -radix decimal  /tb_cnn_top/dut/conv_out_acc_int32
add wave -noupdate -radix decimal  /tb_cnn_top/dut/conv_out_req_int8
add wave -noupdate /tb_cnn_top/dut/relu_out_valid
add wave -noupdate -radix decimal  /tb_cnn_top/dut/relu_out_data

add wave -divider "=== MAXPOOL (8x30x30 -> 8x15x15) ==="
add wave -noupdate /tb_cnn_top/dut/pool_out_valid
add wave -noupdate -radix unsigned /tb_cnn_top/dut/pool_out_addr
add wave -noupdate -radix decimal  /tb_cnn_top/dut/pool_out_data

add wave -divider "=== GAP (16x13x13 -> 16) ==="
add wave -noupdate /tb_cnn_top/dut/gap_out_valid
add wave -noupdate -radix unsigned /tb_cnn_top/dut/gap_out_ch
add wave -noupdate -radix decimal  /tb_cnn_top/dut/gap_out_sum_int32
add wave -noupdate -radix decimal  /tb_cnn_top/dut/gap_out_data

add wave -divider "=== FULLY CONNECTED (16 -> 10) & ARGMAX ==="
add wave -noupdate /tb_cnn_top/dut/fc_out_valid
add wave -noupdate -radix unsigned /tb_cnn_top/dut/fc_out_idx
add wave -noupdate -radix decimal  /tb_cnn_top/dut/fc_out_acc_int32
add wave -noupdate -radix decimal  /tb_cnn_top/dut/fc_out_data
add wave -noupdate -radix decimal  /tb_cnn_top/dut/fc_score_ram
add wave -noupdate -radix unsigned /tb_cnn_top/dut/argmax_pred_class
add wave -noupdate -radix decimal  /tb_cnn_top/dut/argmax_max_score

# Run Image 0 inference first (approx. 6.06 ms sim time = 302,978 cycles) so
# the Wave window immediately shows one complete CNN inference end-to-end, or
# type `run -all` in the Transcript to continue through all 10 batch images.
run 6060us
wave zoom full
echo "================================================================="
echo "ModelSim GUI loaded! Image 0 inference is displayed in the Wave"
echo "window. Type 'run -all' in the ModelSim Transcript to run the"
echo "remaining 10-class batch sweep, or 'do sim/do/run_tb_gap.do' etc."
echo "================================================================="
