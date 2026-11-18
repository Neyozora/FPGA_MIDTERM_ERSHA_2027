# ModelSim .do script: Compile all RTL modules and run complete CNN integration testbench (tb_cnn_top)
onbreak {resume}
onerror {quit -code 1 -f}
if {![file exists work]} { vlib work }

vlog -work work \
    rtl/relu.v \
    rtl/maxpool.v \
    rtl/gap.v \
    rtl/argmax.v \
    rtl/fully_connected.v \
    rtl/conv_engine.v \
    rtl/cnn_top.v \
    sim/tb/tb_cnn_top.v

vsim -c -t 1ps work.tb_cnn_top
onfinish stop
run -all
quit -f
